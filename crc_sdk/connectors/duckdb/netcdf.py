"""Lazy Arrow bridge from a regular-grid NetCDF variable slice into DuckDB.

Supports simple CF-compliant `(time, lat, lon)` variables on a plain
geographic (lat/lon) grid -- the shape JRC's EDO drought indicators (and
many other Copernicus/JRC gridded products) ship in. Rotated/projected
grids and other dimension layouts are out of scope; `NetCDFRaster`
construction raises a clear error rather than silently misreading them.

Unlike GeoTIFF, there is no GDAL-VSI-equivalent single blessed way to stream
a remote NetCDF/HDF5 file. This module streams over plain HTTP range
requests via `fsspec`'s filesystem abstraction plus `h5netcdf`, which works
because NetCDF-4/HDF5's own chunk index lets h5py seek to just the bytes a
requested read needs -- confirmed against JRC's EDO server, which
advertises `Accept-Ranges: bytes`. `cache_dir` still exists for sources
that don't support ranged access, or for repeated reads of the same file,
mirroring `GeoTiffRaster.open`'s own convention exactly.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fsspec  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.geometry.h3 import (
    max_pixel_spacing_m,
    pixel_grid_resolution,
    reduce_h3_values,
    sample_grid_to_h3,
    subsample_offsets,
)

from .connection import DuckDBConnection, default_work_dir
from .geotiff import _materialize_local
from .zarr import Bounds, Point, RasterCurve, RasterMetadata

# Decompressed bytes per strip read; bounds worker RAM independent of raster
# size while keeping the HDF5 chunk-read count low. Same default as GeoTIFF.
STRIP_BYTES = 256 * 1024**2
CHUNK_POINTS = 262_144


def _require_netcdf_extra() -> None:
    """Raise a clear error before any h5netcdf import if the extra is missing."""
    try:
        import h5netcdf  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "NetCDF support requires `pip install crc-sdk[netcdf]`"
        ) from error


def _open_backend(uri: str, cache_dir: str | Path | None) -> Any:
    """Open `uri` as an `h5netcdf.File`.

    `cache_dir=None` (default) streams directly over HTTP range requests via
    `fsspec`, with no local disk write. Pass `cache_dir` to materialize a
    local copy first (via fsspec) when the same file is read more than
    once, or when the remote server doesn't support ranged access -- also
    the more robust choice for a large multi-hundred-MB file read many
    times over one session, since a many-small-range-request stream is more
    exposed to a single transient connection error (observed in practice
    against JRC's EDO server) than one bulk download would be.
    """
    _require_netcdf_extra()
    import h5netcdf

    if cache_dir is not None:
        return h5netcdf.File(str(_materialize_local(uri, cache_dir)), mode="r")
    if "://" not in uri:
        return h5netcdf.File(uri, mode="r")
    filesystem, path = fsspec.core.url_to_fs(uri)
    return h5netcdf.File(filesystem.open(path, mode="rb"), mode="r")


def _fill_value(variable: Any, override: float | None) -> float | None:
    if override is not None:
        return float(override)
    raw = variable.attrs.get("_FillValue") if hasattr(variable, "attrs") else None
    if raw is None:
        return None
    value = np.asarray(raw).reshape(-1)[0]
    return float(value)


def _index_range(
    coords: np.ndarray, low: float, high: float, step: float
) -> tuple[int, int]:
    """Half-open `[start, stop)` index range covering `[low, high]` along `coords`.

    Works regardless of whether `coords` is ascending or descending (EDO's
    own `lat` axis runs north-to-south, i.e. descending).
    """
    half_step = abs(step) / 2
    mask = (coords >= low - half_step) & (coords <= high + half_step)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("bounds do not intersect the raster")
    return int(indices.min()), int(indices.max()) + 1


def _strip_row_count(
    strip_bytes: int,
    width: int,
    itemsize: int,
    block_height: int,
    *,
    leading_axis: int = 1,
) -> int:
    """Row count per strip, bounding `leading_axis * rows * width * itemsize`
    by `strip_bytes` and aligned to `block_height`.

    `leading_axis` is 1 for a single-time-slice read (`NetCDFRaster`); a
    caller reading `leading_axis` steps per row at once (e.g.
    `EDOAnnualMinimaCurveSource` reading a whole year's dekads before
    reducing them) must pass that count, or the strip is undersized by
    roughly that factor -- the actual resident array before any reduction
    is `(leading_axis, rows, width)`, not `(rows, width)`.
    """
    rows_per_strip = max(
        block_height, strip_bytes // max(1, width * itemsize * leading_axis)
    )
    return max(block_height, (rows_per_strip // block_height) * block_height)


class NetCDFRaster:
    """A remote or local NetCDF variable time-slice with lazy DuckDB scan helpers.

    One 2D `(lat, lon)` slice per instance, at a fixed `time_index` -- the
    same one-2D-grid-per-instance shape `GeoTiffRaster` has for one band.
    Assumes `EPSG:4326` (plain geographic coordinates); nothing here
    reprojects, since the CF-compliant sources this targets already ship
    that way.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        variable: str,
        time_index: int = 0,
        lat_name: str = "lat",
        lon_name: str = "lon",
        time_name: str = "time",
        fill_value: float | None = None,
        connection: DuckDBConnection | None = None,
        work_dir: str | Path | None = None,
        _owns_dataset: bool = True,
    ) -> None:
        self._dataset = dataset
        self._owns_dataset = _owns_dataset
        self.variable_name = variable
        self.time_index = time_index

        variable_obj = dataset.variables[variable]
        dims = tuple(variable_obj.dimensions)
        expected = (time_name, lat_name, lon_name)
        if dims != expected:
            raise ValueError(
                f"{variable!r} has dimensions {dims!r}; expected exactly {expected!r}"
            )
        if not 0 <= time_index < variable_obj.shape[0]:
            raise ValueError(
                f"time_index {time_index} is outside {variable_obj.shape[0]} steps"
            )
        self._variable = variable_obj

        lat_coords = np.asarray(dataset.variables[lat_name][:], dtype=np.float64)
        lon_coords = np.asarray(dataset.variables[lon_name][:], dtype=np.float64)
        if lat_coords.size < 2 or lon_coords.size < 2:
            raise ValueError(
                f"{lat_name!r}/{lon_name!r} need at least two coordinate values"
            )
        self._lat = lat_coords
        self._lon = lon_coords
        self._lat_step = float(lat_coords[1] - lat_coords[0])
        self._lon_step = float(lon_coords[1] - lon_coords[0])
        self.nodata = _fill_value(variable_obj, fill_value)

        # An explicit connection means the caller is already in control; only
        # build (and resource-tune) one when they didn't supply their own.
        # No extensions requested -- same reasoning as GeoTiffRaster itself.
        self.connection = connection or DuckDBConnection.for_analytics(
            work_dir or default_work_dir(), extensions=()
        )

    @classmethod
    def open(
        cls,
        uri: str | Path,
        *,
        variable: str,
        time_index: int = 0,
        lat_name: str = "lat",
        lon_name: str = "lon",
        time_name: str = "time",
        fill_value: float | None = None,
        cache_dir: str | Path | None = None,
        connection: DuckDBConnection | None = None,
        work_dir: str | Path | None = None,
    ) -> NetCDFRaster:
        """Open a local path, or an `http(s)://`/`s3://`/`gs://` URI.

        See `_open_backend` for the `cache_dir` streaming-vs-materialize
        trade-off.
        """
        dataset = _open_backend(str(uri), cache_dir)
        try:
            return cls(
                dataset,
                variable=variable,
                time_index=time_index,
                lat_name=lat_name,
                lon_name=lon_name,
                time_name=time_name,
                fill_value=fill_value,
                connection=connection,
                work_dir=work_dir,
            )
        except Exception:
            dataset.close()
            raise

    def close(self) -> None:
        if self._owns_dataset:
            self._dataset.close()

    def __enter__(self) -> NetCDFRaster:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def bounds(self) -> Bounds:
        half_lat = abs(self._lat_step) / 2
        half_lon = abs(self._lon_step) / 2
        return (
            float(self._lon.min()) - half_lon,
            float(self._lat.min()) - half_lat,
            float(self._lon.max()) + half_lon,
            float(self._lat.max()) + half_lat,
        )

    @property
    def pixel_size_meters(self) -> tuple[float, float]:
        """Approximate (width, height) pixel spacing in meters.

        Evaluated at the grid's most-equator-ward row, where longitude
        degrees are widest, so the result is a conservative (largest)
        estimate for the whole raster -- same convention as
        `GeoTiffRaster.pixel_size_meters`.
        """
        equator_ward_lat = min(abs(self._lat.min()), abs(self._lat.max()))
        meters_per_degree = 111_320.0
        width_m = (
            abs(self._lon_step)
            * meters_per_degree
            * math.cos(math.radians(equator_ward_lat))
        )
        height_m = abs(self._lat_step) * meters_per_degree
        return width_m, height_m

    def scan(
        self, bounds: Bounds | None = None, *, strip_bytes: int = STRIP_BYTES
    ) -> NetCDFScan:
        """Describe a reusable, out-of-core scan of raw pixel rows."""
        if strip_bytes < 1:
            raise ValueError("strip_bytes must be positive")
        return NetCDFScan(self, bounds or self.bounds, strip_bytes)

    def scan_h3(
        self,
        bounds: Bounds | None = None,
        *,
        h3_resolution: int | None = None,
        max_subsample: float = 2.0,
        mask: Callable[[np.ndarray], np.ndarray] | None = None,
        reduce: Literal["max", "min"] = "max",
        strip_bytes: int = STRIP_BYTES,
    ) -> NetCDFH3Scan:
        """Describe a reusable scan that samples pixels straight to H3 cells.

        Same semantics as `GeoTiffRaster.scan_h3`: `h3_resolution=None`
        infers the finest resolution the grid's own pixel spacing supports;
        nodata/non-finite pixels are always excluded; `mask` is an optional
        additional predicate over a raw pixel-value window.
        """
        if strip_bytes < 1:
            raise ValueError("strip_bytes must be positive")
        if max_subsample <= 0:
            raise ValueError("max_subsample must be positive")
        if h3_resolution is not None and not 0 <= h3_resolution <= 15:
            raise ValueError("H3 resolution must be between 0 and 15")
        resolution = h3_resolution
        if resolution is None:
            pixel_w, pixel_h = self.pixel_size_meters
            resolution = pixel_grid_resolution(
                max(pixel_w, pixel_h), max_subsample=max_subsample
            )
        return NetCDFH3Scan(
            self,
            bounds or self.bounds,
            h3_resolution=resolution,
            max_subsample=max_subsample,
            mask=mask,
            reduce=reduce,
            strip_bytes=strip_bytes,
        )

    def _row_col_window(self, bounds: Bounds) -> tuple[int, int, int, int]:
        min_lon, min_lat, max_lon, max_lat = bounds
        if min_lon > max_lon or min_lat > max_lat:
            raise ValueError("bounds must be (min_lon, min_lat, max_lon, max_lat)")
        row_start, row_stop = _index_range(self._lat, min_lat, max_lat, self._lat_step)
        col_start, col_stop = _index_range(self._lon, min_lon, max_lon, self._lon_step)
        return row_start, row_stop, col_start, col_stop

    def _rows_to_lat(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self._lat[0] + rows * self._lat_step, dtype=np.float64)

    def _cols_to_lon(self, cols: np.ndarray) -> np.ndarray:
        return np.asarray(self._lon[0] + cols * self._lon_step, dtype=np.float64)

    def _row_bands(
        self,
        row_start: int,
        row_stop: int,
        col_start: int,
        col_stop: int,
        strip_bytes: int,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield (row_off, row_end, band) triples in wide, chunk-aligned strips.

        Same bounded-memory strategy as `GeoTiffRaster._strips`, aligned to
        the variable's own internal HDF5 chunk shape when the backend
        exposes one.
        """
        width = col_stop - col_start
        itemsize = np.dtype(self._variable.dtype).itemsize
        chunk_shape = getattr(self._variable, "chunks", None)
        block_height = chunk_shape[1] if chunk_shape else 1
        strip_rows = _strip_row_count(strip_bytes, width, itemsize, block_height)

        for row_off in range(row_start, row_stop, strip_rows):
            row_end = min(row_off + strip_rows, row_stop)
            band = np.asarray(
                self._variable[self.time_index, row_off:row_end, col_start:col_stop],
                dtype=np.float64,
            )
            yield row_off, row_end, band


@dataclass(frozen=True)
class NetCDFScan:
    """Reusable description of a one-pass NetCDF-to-DuckDB pixel scan."""

    raster: NetCDFRaster
    bounds: Bounds
    strip_bytes: int

    def relation(self, *, connection: DuckDBConnection | None = None) -> Any:
        """Create a fresh lazy DuckDB relation for one query execution."""
        schema = pa.schema(
            [
                ("longitude", pa.float64()),
                ("latitude", pa.float64()),
                ("value", pa.float32()),
            ]
        )
        reader = pa.RecordBatchReader.from_batches(schema, self._batches())
        active = (connection or self.raster.connection).connect()
        return active.from_arrow(reader)

    def _batches(self) -> Iterator[Any]:
        raster = self.raster
        row_start, row_stop, col_start, col_stop = raster._row_col_window(self.bounds)
        for row_off, row_end, band in raster._row_bands(
            row_start, row_stop, col_start, col_stop, self.strip_bytes
        ):
            valid = np.isfinite(band)
            if raster.nodata is not None:
                valid &= band != raster.nodata
            if not valid.any():
                continue
            local_row, local_col = np.where(valid)
            values = band[local_row, local_col].astype(np.float32, copy=False)
            lats = raster._rows_to_lat((local_row + row_off).astype(np.float64))
            lons = raster._cols_to_lon((local_col + col_start).astype(np.float64))
            yield pa.record_batch(
                {
                    "longitude": pa.array(lons),
                    "latitude": pa.array(lats),
                    "value": pa.array(values),
                }
            )


@dataclass(frozen=True)
class NetCDFH3Scan:
    """Reusable description of a one-pass NetCDF-to-H3-cell scan."""

    raster: NetCDFRaster
    bounds: Bounds
    h3_resolution: int
    max_subsample: float
    mask: Callable[[np.ndarray], np.ndarray] | None
    reduce: Literal["max", "min"]
    strip_bytes: int

    def relation(self, *, connection: DuckDBConnection | None = None) -> Any:
        """Create a fresh DuckDB relation, pre-reduced to one row per cell.

        Same cross-strip merge strategy as `GeoTiffH3Scan.relation`: batches
        are already reduced within each strip, so only pixels whose H3 cell
        straddles a strip boundary can still repeat across batches.
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

    def _batches(self) -> Iterator[Any]:
        raster = self.raster
        pixel_w, pixel_h = raster.pixel_size_meters
        spacing = max_pixel_spacing_m(self.h3_resolution)
        max_axis_points = max(1, math.ceil(self.max_subsample))
        columns = max(1, min(math.ceil(pixel_w / spacing), max_axis_points))
        rows = max(1, min(math.ceil(pixel_h / spacing), max_axis_points))
        column_offsets = subsample_offsets(columns)
        row_offsets = subsample_offsets(rows)
        n_sub = columns * rows
        pixel_chunk = max(1, CHUNK_POINTS // n_sub)

        row_start, row_stop, col_start, col_stop = raster._row_col_window(self.bounds)
        for row_off, row_end, band in raster._row_bands(
            row_start, row_stop, col_start, col_stop, self.strip_bytes
        ):
            valid = np.isfinite(band)
            if raster.nodata is not None:
                valid &= band != raster.nodata
            if self.mask is not None:
                valid &= self.mask(band)
            if not valid.any():
                continue

            local_row, local_col = np.where(valid)
            values = band[local_row, local_col].astype(np.float64, copy=False)
            strip_rows = local_row + row_off
            strip_columns = local_col + col_start

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
        rows = row_grid.ravel()
        columns = column_grid.ravel()

        lats = self.raster._rows_to_lat(rows)
        lons = self.raster._cols_to_lon(columns)
        n_sub = len(row_offsets) * len(column_offsets)
        return sample_grid_to_h3(
            lons,
            lats,
            np.repeat(values, n_sub),
            resolution=self.h3_resolution,
            reduce=self.reduce,
        )


def _pixel_boundary(
    raster: NetCDFRaster, row: int, column: int
) -> tuple[Point, Point, Point, Point]:
    """Pixel corner boundary, in the same counter-clockwise convention as
    `ZarrRaster.pixel_boundary`/`geotiff._pixel_boundary`.

    Uses `abs(step)` rather than the signed step: unlike a GeoTIFF/Zarr
    raster (always stored north-up, i.e. row index increasing always means
    latitude decreasing), a NetCDF coordinate array's direction is not
    guaranteed by the format -- EDO's own `lat` happens to run north-to-
    south (descending), but nothing else here assumes that. Corners are
    built in a fixed NW/SW/SE/NE geographic order instead of one derived
    from the coordinate array's own direction, so the ring winds the same
    way regardless of whether `lat`/`lon` are ascending or descending.
    """
    half_lat = abs(raster._lat_step) / 2
    half_lon = abs(raster._lon_step) / 2
    lat = float(raster._lat[row])
    lon = float(raster._lon[column])
    return (
        (lon - half_lon, lat + half_lat),
        (lon - half_lon, lat - half_lat),
        (lon + half_lon, lat - half_lat),
        (lon + half_lon, lat + half_lat),
    )


def _annual_minima_curve(annual_minima: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One pixel's per-year minima -> (periods, values) ready to fit.

    Treats each year's block minimum as one extreme-value sample (missing
    years, e.g. a pixel with no valid dekad that year, are dropped rather
    than zero-filled). Assigns each an empirical return period via the
    Gringorten plotting position (`a = 0.44`, a standard, mildly
    conservative choice for annual extremes) -- the estimate that this
    value's severity recurs, on average, once every that-many years. There
    is no distribution-family assumption here; the plotting position is
    purely rank-based, ready to fit like any other tabulated curve via
    `TabulatedDistribution.from_return_periods(periods, values,
    tail="lower")` (the periods decrease as the values increase, the shape
    a "lower" fit expects -- rarer years are *more* severe, i.e. *lower*
    SMI, not higher).

    Returns two empty arrays if fewer than 4 years are valid -- the same
    per-pixel curve floor `_canonical_batches` (crc_sdk.connectors.adapters)
    already enforces for any curve source, checked here too so a caller
    iterating this source directly sees the same "too few knots" signal.
    """
    valid = annual_minima[np.isfinite(annual_minima)]
    n = valid.size
    if n < 4:
        return np.array([]), np.array([])
    values = np.sort(valid)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    probabilities = (ranks - 0.44) / (n + 0.12)
    periods = 1.0 / probabilities
    return periods, values


@dataclass
class EDOAnnualMinimaCurveSource:
    """Presents N years of EDO dekadal SMI as one annual-block-minima curve source.

    Each year's SMI file is opened once (one `NetCDFRaster` per year, at
    `time_index=0` -- used here only to validate/describe the shared grid,
    not to restrict which time steps get read); for a bounded AOI, each
    year's whole dekadal time series is read for that window in one strided
    read per strip, then reduced to that year's per-pixel minimum. One
    block-minima "curve" knot per requested year is fed through the same
    `canonicalize_curve_source` (`crc_sdk.connectors.adapters`) JRC flood
    and OS-Climate both use, via `crc_sdk.connectors.jrc_edo.canonicalize_edo_drought`
    -- just with empirical (Gringorten) plotting-position return periods
    instead of literal return-period rasters or quantile samples.
    """

    rasters: dict[int, NetCDFRaster]
    metadata: RasterMetadata
    strip_bytes: int = STRIP_BYTES

    def __post_init__(self) -> None:
        if not self.rasters:
            raise ValueError("at least one year is required")
        if self.strip_bytes < 1:
            raise ValueError("strip_bytes must be positive")
        self._years = tuple(sorted(self.rasters))
        reference = self.rasters[self._years[0]]
        for year, raster in self.rasters.items():
            if not np.array_equal(raster._lat, reference._lat) or not np.array_equal(
                raster._lon, reference._lon
            ):
                raise ValueError(
                    f"year {year} raster grid does not match the others in this stack"
                )
        self._reference = reference

    def close(self) -> None:
        for raster in self.rasters.values():
            raster.close()

    def __enter__(self) -> EDOAnnualMinimaCurveSource:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def axis_name(self) -> str:
        return "return period (empirical, Gringorten plotting position)"

    @property
    def return_period_support(self) -> tuple[float, float]:
        count = len(self._years)
        probabilities = (np.array([count, 1], dtype=np.float64) - 0.44) / (count + 0.12)
        periods = 1.0 / probabilities
        return float(periods[0]), float(periods[1])

    @property
    def bounds(self) -> Bounds:
        return self._reference.bounds

    def iter_curves(self, bounds: Bounds | None = None) -> Iterator[RasterCurve]:
        reference = self._reference
        row_start, row_stop, col_start, col_stop = reference._row_col_window(
            bounds or self.bounds
        )
        width = col_stop - col_start
        years = self._years
        n_years = len(years)

        itemsize = np.dtype(np.float64).itemsize
        chunk_shape = getattr(reference._variable, "chunks", None)
        block_height = chunk_shape[1] if chunk_shape else 1
        # Budget for whichever is larger: the persistent per-strip
        # annual-minima array (n_years, strip_rows, width), or the bigger
        # transient per-year block read before it's reduced away
        # (n_dekads, strip_rows, width, one year at a time). Sizing off
        # n_years alone (as if the per-year read were already reduced)
        # undercounts by roughly n_dekads / n_years -- worst with short
        # year ranges, since a single year's raw dekadal read is what's
        # actually resident in memory at that point, not the reduced curve.
        max_dekads = max(raster._variable.shape[0] for raster in self.rasters.values())
        leading_axis = max(n_years, max_dekads)
        strip_rows = _strip_row_count(
            self.strip_bytes, width, itemsize, block_height, leading_axis=leading_axis
        )

        for row_off in range(row_start, row_stop, strip_rows):
            row_end = min(row_off + strip_rows, row_stop)
            annual_minima = np.empty(
                (n_years, row_end - row_off, width), dtype=np.float64
            )
            for index, year in enumerate(years):
                raster = self.rasters[year]
                block = np.asarray(
                    raster._variable[:, row_off:row_end, col_start:col_stop],
                    dtype=np.float64,
                )
                finite = np.isfinite(block)
                if raster.nodata is not None:
                    finite &= block != raster.nodata
                block = np.where(finite, block, np.nan)
                with warnings.catch_warnings():
                    # A pixel with zero valid dekads this year (e.g. outside
                    # the dataset's own coverage) is an expected, not
                    # exceptional, all-NaN slice.
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    annual_minima[index] = np.nanmin(block, axis=0)

            valid_any = np.isfinite(annual_minima).any(axis=0)
            if not valid_any.any():
                continue
            local_rows, local_columns = np.where(valid_any)
            for local_row, local_column in zip(
                local_rows.tolist(), local_columns.tolist()
            ):
                periods, values = _annual_minima_curve(
                    annual_minima[:, local_row, local_column]
                )
                if periods.size == 0:
                    continue
                source_row = row_off + local_row
                source_column = col_start + local_column
                yield RasterCurve(
                    row=source_row,
                    column=source_column,
                    boundary=_pixel_boundary(reference, source_row, source_column),
                    axis_values=periods,
                    values=values,
                )
