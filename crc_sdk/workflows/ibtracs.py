"""Lazy access to NOAA IBTrACS best-track observations, not hazard curves."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]

from ._remote import file_checksum, read_manifest, write_manifest

_ROOT = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv"
_SUBSETS = {
    "ALL",
    "since1980",
    "last3years",
    "active",
    "NA",
    "SA",
    "EP",
    "WP",
    "SP",
    "SI",
    "NI",
}


@dataclass(frozen=True)
class IBTrACSPlan:
    """A request for a specific NOAA CSV subset and a cached byte snapshot.

    Values retain NOAA's original column names, units and agency distinctions.
    Filters select track points, not complete storms or spatial footprints.
    """

    subset: str
    cache_dir: Path
    mode: str = "reuse"
    expected_sha256: str | None = None
    storm_ids: tuple[str, ...] | None = None
    year_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.subset not in _SUBSETS:
            raise ValueError(f"subset must be one of {sorted(_SUBSETS)}")
        if self.mode not in {"reuse", "offline", "refresh"}:
            raise ValueError("mode must be reuse, offline or refresh")
        if self.expected_sha256 is not None and (
            len(self.expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")

    @property
    def url(self) -> str:
        return f"{_ROOT}/ibtracs.{self.subset}.list.v04r01.csv"

    def for_storms(self, ids: Sequence[str]) -> IBTrACSPlan:
        values = tuple(ids)
        if not values or any(not value.strip() for value in values):
            raise ValueError("at least one non-empty storm ID is required")
        return replace(self, storm_ids=values)

    def years(self, start: int, end: int) -> IBTrACSPlan:
        if start > end or start < 0:
            raise ValueError("years must be ordered nonnegative season years")
        return replace(self, year_range=(start, end))

    def explain(self) -> dict[str, object]:
        return {
            "url": self.url,
            "version": "v04r01",
            "subset": self.subset,
            "cache_dir": str(self.cache_dir),
            "mode": self.mode,
            "expected_sha256": self.expected_sha256,
            "storm_ids": self.storm_ids,
            "season_years": self.year_range,
            "selection": "point rows; basin subsets may overlap",
        }

    def prefetch(self) -> dict[str, object]:
        """Acquire a snapshot or verify cached bytes; return source provenance."""
        directory = self.cache_dir / "ibtracs" / "v04r01" / self.subset
        path = directory / "source.csv"
        manifest = read_manifest(directory)
        if self.mode != "refresh" and path.exists() != (manifest is not None):
            raise RuntimeError("incomplete IBTrACS cache; refresh it explicitly")
        if self.mode != "refresh" and path.exists() and manifest is not None:
            digest = file_checksum(path)
            if manifest.get("url") != self.url or manifest.get("sha256") != digest:
                raise RuntimeError("IBTrACS cache checksum or source mismatch")
            if self.expected_sha256 and digest != self.expected_sha256:
                raise RuntimeError("IBTrACS snapshot does not match expected SHA-256")
            return manifest
        if self.mode == "offline":
            raise FileNotFoundError(f"IBTrACS snapshot missing from {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=directory, prefix="download-", suffix=".tmp")
        temp = Path(name)
        try:
            with (
                os.fdopen(fd, "wb") as target,
                urlopen(self.url, timeout=120) as response,
            ):
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                headers = dict(response.headers)
                content_length = response.headers.get("Content-Length")
            if content_length is not None and temp.stat().st_size != int(
                content_length
            ):
                raise RuntimeError("incomplete IBTrACS download")
            digest = file_checksum(temp)
            if self.expected_sha256 and digest != self.expected_sha256:
                raise RuntimeError("IBTrACS snapshot does not match expected SHA-256")
            with temp.open(newline="", encoding="utf-8-sig") as source:
                reader = csv.reader(source)
                columns = next(reader)
                units = next(reader)
                if len(columns) != len(units) or not {
                    "SID",
                    "SEASON",
                    "ISO_TIME",
                }.issubset(columns):
                    raise ValueError("invalid IBTrACS CSV header or units row")
            result: dict[str, object] = {
                "url": self.url,
                "version": "v04r01",
                "subset": self.subset,
                "sha256": digest,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
                "units": dict(zip(columns, units)),
                "terms_url": "https://www.ncei.noaa.gov/products/international-best-track-archive",
                "citation": (
                    "Schreck et al. (2024), IBTrACS v04r01, NOAA NCEI, "
                    "doi:10.25921/82ty-9e16"
                ),
            }
            temp.replace(path)
            write_manifest(directory, result)
            return result
        finally:
            temp.unlink(missing_ok=True)

    def batches(self, *, batch_size: int = 4096) -> Iterator[pa.RecordBatch]:
        """Yield bounded Arrow batches; empty source fields become null strings."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.prefetch()
        path = self.cache_dir / "ibtracs" / "v04r01" / self.subset / "source.csv"
        ids = set(self.storm_ids) if self.storm_ids is not None else None
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.reader(source)
            columns = next(reader)
            next(reader)
            schema = pa.schema([(column, pa.string()) for column in columns])
            sid_index, year_index = columns.index("SID"), columns.index("SEASON")
            rows: list[dict[str, str | None]] = []
            for line in reader:
                if len(line) != len(columns):
                    raise ValueError(f"malformed IBTrACS CSV row {reader.line_num}")
                if ids is not None and line[sid_index].strip() not in ids:
                    continue
                if self.year_range is not None:
                    year = int(line[year_index].strip())
                    if not self.year_range[0] <= year <= self.year_range[1]:
                        continue
                rows.append(
                    dict(zip(columns, (value.strip() or None for value in line)))
                )
                if len(rows) == batch_size:
                    yield pa.RecordBatch.from_pylist(rows, schema=schema)
                    rows.clear()
            if rows:
                yield pa.RecordBatch.from_pylist(rows, schema=schema)
