"""Monthly CHIRPS v3 precipitation as lazy, windowed raster observations."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.connectors.duckdb.geotiff import GeoTiffRaster

from ._remote import (
    Bounds,
    file_checksum,
    read_manifest,
    validate_bounds,
    write_manifest,
)

_ROOT = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/cogs"


@dataclass(frozen=True)
class CHIRPSMonthlyPlan:
    """Final monthly precipitation in mm per source grid cell."""

    year: int
    month: int
    bounds: Bounds | None = None
    cache_dir: Path | None = None
    mode: str = "stream"
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.year, int)
            or isinstance(self.year, bool)
            or not 1981 <= self.year <= 9999
        ):
            raise ValueError("CHIRPS v3 starts in 1981")
        if (
            not isinstance(self.month, int)
            or isinstance(self.month, bool)
            or not 1 <= self.month <= 12
        ):
            raise ValueError("month must be between 1 and 12")
        if self.mode not in {"stream", "reuse", "offline", "refresh"}:
            raise ValueError("invalid CHIRPS cache mode")
        if self.mode == "stream" and self.expected_sha256 is not None:
            raise ValueError("a checksum requires a cached source, not stream mode")
        if self.expected_sha256 is not None and (
            len(self.expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def monthly(cls, year: int, month: int) -> CHIRPSMonthlyPlan:
        return cls(year, month)

    def for_area(self, bounds: Sequence[float]) -> CHIRPSMonthlyPlan:
        result = validate_bounds(bounds)
        if result[1] < -60 or result[3] > 60:
            raise ValueError("CHIRPS v3 coverage is limited to 60S–60N")
        return replace(self, bounds=result)

    def cache(
        self, directory: str | Path, *, mode: str = "reuse", sha256: str | None = None
    ) -> CHIRPSMonthlyPlan:
        return replace(
            self, cache_dir=Path(directory), mode=mode, expected_sha256=sha256
        )

    @property
    def url(self) -> str:
        return f"{_ROOT}/chirps-v3.0.{self.year:04d}.{self.month:02d}.cog"

    def explain(self) -> dict[str, object]:
        return {
            "url": self.url,
            "product": "CHIRPS v3.0 final monthly",
            "unit": "mm",
            "bounds": self.bounds,
            "mode": self.mode,
            "checksum": "unavailable in stream mode"
            if self.mode == "stream"
            else self.expected_sha256,
            "meaning": "monthly gridded precipitation, not design rainfall",
        }

    def _directory(self) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache(directory) must be configured for this mode")
        return (
            self.cache_dir
            / "chirps-v3"
            / "monthly-final"
            / f"{self.year:04d}-{self.month:02d}"
        )

    def prefetch(self) -> dict[str, Any]:
        """Download the complete COG in bounded chunks and checksum it."""
        if self.mode == "stream":
            raise ValueError("configure cache(directory) before prefetch")
        directory = self._directory()
        path = directory / "source.cog"
        manifest = read_manifest(directory)
        if self.mode != "refresh" and path.exists() != (manifest is not None):
            raise RuntimeError("incomplete CHIRPS cache; refresh it explicitly")
        if self.mode != "refresh" and manifest is not None:
            digest = file_checksum(path)
            if manifest.get("url") != self.url or manifest.get("sha256") != digest:
                raise RuntimeError("CHIRPS cache checksum or source mismatch")
            if self.expected_sha256 and digest != self.expected_sha256:
                raise RuntimeError("CHIRPS snapshot does not match expected SHA-256")
            return manifest
        if self.mode == "offline":
            raise FileNotFoundError(f"CHIRPS source missing from {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as scratch:
            staged = Path(scratch) / "source.cog"
            with urlopen(self.url, timeout=120) as source, staged.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                headers = source.headers
                length = headers.get("Content-Length")
                etag = headers.get("ETag")
                modified = headers.get("Last-Modified")
            if length is not None and staged.stat().st_size != int(length):
                raise RuntimeError("incomplete CHIRPS download")
            digest = file_checksum(staged)
            if self.expected_sha256 and digest != self.expected_sha256:
                raise RuntimeError("CHIRPS snapshot does not match expected SHA-256")
            result: dict[str, Any] = {
                "url": self.url,
                "sha256": digest,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "etag": etag,
                "last_modified": modified,
                "product": "CHIRPS v3.0 final monthly",
                "year": self.year,
                "month": self.month,
                "unit": "mm",
                "licence": "CC-BY-4.0",
                "attribution": (
                    "Climate Hazards Center, UC Santa Barbara; "
                    "CHIRPS3 Data Repository, doi:10.15780/G2JQ0P"
                ),
            }
            staged.replace(path)
            write_manifest(directory, result)
        return result

    def batches(
        self, *, batch_size: int = 65536, strip_bytes: int = 8 * 1024 * 1024
    ) -> Iterator[pa.RecordBatch]:
        """Scan pixel centers in bounded batches; omit CHIRPS's -9999 sentinel."""
        if self.bounds is None:
            raise ValueError("for_area(bounds) must be configured")
        if batch_size < 1 or strip_bytes < 1:
            raise ValueError("batch_size and strip_bytes must be positive")
        if self.mode == "stream":
            source: str | Path = self.url
        else:
            self.prefetch()
            source = self._directory() / "source.cog"
        with GeoTiffRaster.open(source) as raster:
            if raster.crs.to_epsg() != 4326:
                raise ValueError("CHIRPS raster is not in WGS84")
            request = (
                raster.scan(self.bounds, strip_bytes=strip_bytes)
                .pipeline()
                .where("value != -9999")
                .select(
                    "longitude, latitude, value AS precipitation_mm, "
                    f"DATE '{self.year:04d}-{self.month:02d}-01' AS month"
                )
            )
            yield from request.to_arrow_reader(batch_size)
