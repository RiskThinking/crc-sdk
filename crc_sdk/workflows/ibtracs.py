"""Lazy access to NOAA IBTrACS best-track observations, not hazard curves."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.csv as pacsv  # type: ignore[import-untyped]

from ._remote import file_checksum, http_request, read_manifest, write_manifest

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
        values = tuple(value.strip() for value in ids)
        if not values or any(not value for value in values):
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
                urlopen(http_request(self.url), timeout=120) as response,
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
        with path.open("rb") as source:
            header = source.readline().decode("utf-8-sig")
        columns = next(csv.reader([header]))
        schema = pa.schema([(column, pa.string()) for column in columns])
        pending: list[pa.RecordBatch] = []
        pending_rows = 0
        for batch in self._filtered_batches(path, columns, schema):
            pending.append(batch)
            pending_rows += batch.num_rows
            while pending_rows >= batch_size:
                table = pa.Table.from_batches(pending, schema=schema)
                yield from table.slice(0, batch_size).combine_chunks().to_batches()
                rest = table.slice(batch_size)
                pending = rest.combine_chunks().to_batches() if rest.num_rows else []
                pending_rows = rest.num_rows
        if pending_rows:
            table = pa.Table.from_batches(pending, schema=schema).combine_chunks()
            yield from table.to_batches()

    def _filtered_batches(
        self, path: Path, columns: list[str], schema: Any
    ) -> Iterator[pa.RecordBatch]:
        """Stripped, null-for-empty string batches after the storm/year filters."""
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(
                column_names=columns, skip_rows=2, block_size=1 << 24
            ),
            parse_options=pacsv.ParseOptions(
                newlines_in_values=True, ignore_empty_lines=False
            ),
            convert_options=pacsv.ConvertOptions(
                column_types={column: pa.string() for column in columns},
                strings_can_be_null=False,
                quoted_strings_can_be_null=False,
            ),
        )
        ids = (
            pa.array(sorted(set(self.storm_ids)), type=pa.string())
            if self.storm_ids is not None
            else None
        )
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                return
            except pa.ArrowInvalid as error:
                raise ValueError(f"malformed IBTrACS CSV: {error}") from error
            stripped = pa.RecordBatch.from_arrays(
                [pc.utf8_trim_whitespace(column) for column in batch.columns],
                schema=schema,
            )
            if ids is not None:
                stripped = stripped.filter(
                    pc.is_in(stripped.column("SID"), value_set=ids)
                )
            if self.year_range is not None:
                # Years are parsed only on rows the storm filter kept.
                year = pc.cast(stripped.column("SEASON"), pa.int64())
                stripped = stripped.filter(
                    pc.and_(
                        pc.greater_equal(year, self.year_range[0]),
                        pc.less_equal(year, self.year_range[1]),
                    )
                )
            if stripped.num_rows:
                yield pa.RecordBatch.from_arrays(
                    [
                        pc.if_else(
                            pc.equal(column, ""), pa.scalar(None, pa.string()), column
                        )
                        for column in stripped.columns
                    ],
                    schema=schema,
                )
