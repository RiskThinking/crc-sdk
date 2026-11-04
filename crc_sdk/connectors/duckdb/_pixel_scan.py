"""Pixel-row and H3 scans shared by strip-readable rasters (GeoTIFF, NetCDF)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.geometry.h3 import (
    max_pixel_spacing_m,
    pixel_grid_resolution,
    reduce_h3_values,
    sample_grid_to_h3,
    subsample_offsets,
)

from .connection import DuckDBConnection
from .stream import ArrowBatchSource, DuckDBPipeline
from .zarr import Bounds

# Cap subsample points materialized per broadcast to bound worker RAM.
CHUNK_POINTS = 262_144


class StripRaster(Protocol):
    """Raster hooks the shared scans need."""

    connection: DuckDBConnection

    @property
    def pixel_size_meters(self) -> tuple[float, float]: ...

    def _validate_bounds(self, bounds: Bounds) -> None:
        """Raise `ValueError` when `bounds` select no pixel."""

    def _strips(
        self, bounds: Bounds, strip_bytes: int
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield `(row_off, col_off, band)` strips covering `bounds`."""
        ...

    def _valid_pixels(self, band: np.ndarray) -> np.ndarray:
        """Boolean mask of pixels holding data."""
        ...

    def _pixel_lonlat(
        self, rows: np.ndarray, columns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """WGS84 `(longitude, latitude)` of fractional pixel indices."""
        ...


R = TypeVar("R", bound=StripRaster)


def check_strip_bytes(strip_bytes: int) -> None:
    if strip_bytes < 1:
        raise ValueError("strip_bytes must be positive")


def h3_scan_resolution(
    raster: StripRaster,
    h3_resolution: int | None,
    *,
    max_subsample: float,
    strip_bytes: int,
) -> int:
    """Validate `scan_h3` arguments and infer the resolution when omitted."""
    check_strip_bytes(strip_bytes)
    if max_subsample <= 0:
        raise ValueError("max_subsample must be positive")
    if h3_resolution is not None and not 0 <= h3_resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    if h3_resolution is not None:
        return h3_resolution
    pixel_w, pixel_h = raster.pixel_size_meters
    return pixel_grid_resolution(max(pixel_w, pixel_h), max_subsample=max_subsample)


@dataclass(frozen=True)
class PixelScan(Generic[R]):
    """Reusable description of a one-pass raster-to-DuckDB pixel scan."""

    raster: R
    bounds: Bounds
    strip_bytes: int

    def relation(self, *, connection: DuckDBConnection | None = None) -> Any:
        """Create a fresh lazy DuckDB relation for one query execution."""
        # Validate bounds now: raised inside the batch generator, a plain
        # ValueError would surface as an opaque DuckDB InvalidInputException.
        self.raster._validate_bounds(self.bounds)
        schema = pa.schema(
            [
                ("longitude", pa.float64()),
                ("latitude", pa.float64()),
                ("value", pa.float32()),
            ]
        )
        return ArrowBatchSource(schema, self._batches).relation(
            connection=connection or self.raster.connection
        )

    def pipeline(self, *, connection: DuckDBConnection | None = None) -> DuckDBPipeline:
        """Compose this bounded scan with lazy DuckDB relational operations."""
        return DuckDBPipeline(self, connection=connection)

    def _batches(self) -> Iterator[Any]:
        raster = self.raster
        for row_off, col_off, band in raster._strips(self.bounds, self.strip_bytes):
            valid = raster._valid_pixels(band)
            if not valid.any():
                continue
            row_idx, col_idx = np.where(valid)
            values = band[row_idx, col_idx].astype(np.float32, copy=False)
            lons, lats = raster._pixel_lonlat(
                row_idx.astype(np.float64) + row_off,
                col_idx.astype(np.float64) + col_off,
            )
            yield pa.record_batch(
                {
                    "longitude": pa.array(lons),
                    "latitude": pa.array(lats),
                    "value": pa.array(values),
                }
            )


@dataclass(frozen=True)
class H3PixelScan(Generic[R]):
    """Reusable description of a one-pass raster-to-H3-cell scan."""

    raster: R
    bounds: Bounds
    h3_resolution: int
    max_subsample: float
    mask: Callable[[np.ndarray], np.ndarray] | None
    reduce: Literal["max", "min"]
    strip_bytes: int

    def relation(self, *, connection: DuckDBConnection | None = None) -> Any:
        """Create a fresh DuckDB relation, pre-reduced to one row per cell.

        Batches are already reduced within each strip; only pixels whose H3
        cell straddles a strip boundary can still repeat across batches, so
        the cross-strip merge is a small, cheap `reduce_h3_values` call --
        not routed through a DuckDB `GROUP BY`, which at this row count
        (bounded by exposed cell count, typically far below the source
        raster's pixel count) is pure per-query overhead next to a direct
        vectorized reduce.
        """
        cell_parts = []
        value_parts = []
        for batch in self._batches():
            cell_parts.append(batch.column("cell").to_numpy(zero_copy_only=False))
            value_parts.append(batch.column("value").to_numpy(zero_copy_only=False))

        if cell_parts:
            cells, values = reduce_h3_values(
                np.concatenate(cell_parts),
                np.concatenate(value_parts),
                reduce=self.reduce,
            )
        else:
            cells = np.array([], dtype=np.uint64)
            values = np.array([], dtype=np.float32)

        table = pa.table(
            {
                "cell": pa.array(cells, type=pa.uint64()),
                "value": pa.array(values.astype(np.float32), type=pa.float32()),
            }
        )
        active = (connection or self.raster.connection).connect()
        return active.from_arrow(table)

    def pipeline(self, *, connection: DuckDBConnection | None = None) -> DuckDBPipeline:
        """Compose this reduced H3 scan with lazy DuckDB relational operations."""
        return DuckDBPipeline(self, connection=connection)

    def _batches(self) -> Iterator[Any]:
        raster = self.raster
        pixel_w, pixel_h = raster.pixel_size_meters
        spacing = max_pixel_spacing_m(self.h3_resolution)
        # max_subsample caps points per pixel axis regardless of how the
        # resolution was chosen -- with an auto-picked resolution this is a
        # no-op (pixel_grid_resolution already keeps the ratio within the
        # cap), but an explicit fine h3_resolution on a coarse raster would
        # otherwise demand an arbitrarily large per-pixel sample grid.
        max_axis_points = max(1, math.ceil(self.max_subsample))
        columns = max(1, min(math.ceil(pixel_w / spacing), max_axis_points))
        rows = max(1, min(math.ceil(pixel_h / spacing), max_axis_points))
        column_offsets = subsample_offsets(columns)
        row_offsets = subsample_offsets(rows)
        n_sub = columns * rows
        pixel_chunk = max(1, CHUNK_POINTS // n_sub)

        for row_off, col_off, band in raster._strips(self.bounds, self.strip_bytes):
            valid = raster._valid_pixels(band)
            if self.mask is not None:
                valid &= self.mask(band)
            if not valid.any():
                continue

            row_idx, col_idx = np.where(valid)
            values = band[row_idx, col_idx].astype(np.float64, copy=False)
            strip_rows = row_idx + row_off
            strip_columns = col_idx + col_off

            # Each chunk is already reduced to one value per cell (via
            # _sample_chunk); yielding it straight through instead of
            # accumulating and re-reducing once per strip avoids a redundant
            # extra sort+dedupe pass -- relation() does the one merge that's
            # actually needed, across every chunk of the whole scan.
            for start in range(0, len(values), pixel_chunk):
                chunk = slice(start, start + pixel_chunk)
                cells, chunk_values = self._sample_chunk(
                    strip_rows[chunk],
                    strip_columns[chunk],
                    values[chunk],
                    row_offsets,
                    column_offsets,
                )
                yield pa.record_batch(
                    {
                        "cell": pa.array(cells, type=pa.uint64()),
                        "value": pa.array(
                            chunk_values.astype(np.float32), type=pa.float32()
                        ),
                    }
                )

    def _sample_chunk(
        self,
        pixel_rows: np.ndarray,
        pixel_columns: np.ndarray,
        values: np.ndarray,
        row_offsets: np.ndarray,
        column_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pixel_rows_f = pixel_rows.astype(np.float64)[:, None, None]
        pixel_columns_f = pixel_columns.astype(np.float64)[:, None, None]
        row_grid = pixel_rows_f + row_offsets[None, :, None]
        column_grid = pixel_columns_f + column_offsets[None, None, :]
        row_grid, column_grid = np.broadcast_arrays(row_grid, column_grid)
        lons, lats = self.raster._pixel_lonlat(row_grid.ravel(), column_grid.ravel())
        n_sub = len(row_offsets) * len(column_offsets)
        return sample_grid_to_h3(
            lons,
            lats,
            np.repeat(values, n_sub),
            resolution=self.h3_resolution,
            reduce=self.reduce,
        )
