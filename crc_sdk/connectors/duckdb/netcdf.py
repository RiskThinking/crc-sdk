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
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import fsspec  # type: ignore[import-untyped]
import numpy as np

from crc_sdk.connectors.blocks import (
    STRIP_BYTES,
    BlockExtremaCurveSource,
    BlockSpec,
    BlockStatistic,
    _index_range,
    _strip_row_count,
    pixel_boundary,
    reduce_block_samples,
)

from ._pixel_scan import H3PixelScan, PixelScan, check_strip_bytes, h3_scan_resolution
from .connection import DuckDBConnection, default_work_dir
from .geotiff import _materialize_local
from .zarr import Bounds, Point, RasterMetadata


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


def _attr_values(variable: Any, name: str) -> tuple[Any, ...]:
    attrs = getattr(variable, "attrs", None)
    raw = attrs.get(name) if attrs is not None else None
    if raw is None:
        return ()
    return tuple(np.asarray(raw).reshape(-1).tolist())


def _attr_scalar(variable: Any, name: str) -> float | None:
    values = _attr_values(variable, name)
    return float(values[0]) if values else None


class _CFDecoding:
    """CF packed-data decoding h5netcdf (unlike netCDF4/xarray) never applies.

    Sentinels (``_FillValue``/``missing_value``, or an explicit override) are
    matched against the *raw* stored values first, then ``scale_factor`` and
    ``add_offset`` unpack what remains -- the CF order, so a packed sentinel
    never survives as a plausible-looking scaled number.
    """

    def __init__(self, variable: Any, override: float | None) -> None:
        sentinels = (
            (override,)
            if override is not None
            else _attr_values(variable, "_FillValue")
            + _attr_values(variable, "missing_value")
        )
        self.sentinels = tuple(dict.fromkeys(sentinels))
        self.scale = _attr_scalar(variable, "scale_factor")
        self.offset = _attr_scalar(variable, "add_offset")

    def __call__(self, raw: Any) -> np.ndarray[Any, Any]:
        """Raw stored values -> float64 physical values, NaN where missing."""
        stored = np.asarray(raw)
        invalid = np.zeros(stored.shape, dtype=bool)
        for sentinel in self.sentinels:
            invalid |= stored == sentinel
        values = stored.astype(np.float64)
        if self.scale is not None:
            values *= self.scale
        if self.offset is not None:
            values += self.offset
        if invalid.any():
            values[invalid] = np.nan
        return values


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
        # Raw-unit sentinel kept for introspection; reads mask it (plus any
        # `missing_value`) and unpack CF scale/offset via `_decode`.
        self.nodata = _fill_value(variable_obj, fill_value)
        self._decode = _CFDecoding(variable_obj, fill_value)

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
        check_strip_bytes(strip_bytes)
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
        return NetCDFH3Scan(
            self,
            bounds or self.bounds,
            h3_resolution=h3_scan_resolution(
                self,
                h3_resolution,
                max_subsample=max_subsample,
                strip_bytes=strip_bytes,
            ),
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

    def _validate_bounds(self, bounds: Bounds) -> None:
        self._row_col_window(bounds)

    def _valid_pixels(self, band: np.ndarray) -> np.ndarray:
        # `_decode` already masked fill values to NaN.
        return np.isfinite(band)

    def _pixel_lonlat(
        self, rows: np.ndarray, columns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._cols_to_lon(columns), self._rows_to_lat(rows)

    def _strips(
        self, bounds: Bounds, strip_bytes: int
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield (row_off, col_off, band) triples in wide, chunk-aligned strips.

        Same bounded-memory strategy as `GeoTiffRaster._strips`, aligned to
        the variable's own internal HDF5 chunk shape when the backend
        exposes one.
        """
        row_start, row_stop, col_start, col_stop = self._row_col_window(bounds)
        width = col_stop - col_start
        itemsize = np.dtype(self._variable.dtype).itemsize
        chunk_shape = getattr(self._variable, "chunks", None)
        block_height = chunk_shape[1] if chunk_shape else 1
        strip_rows = _strip_row_count(strip_bytes, width, itemsize, block_height)

        for row_off in range(row_start, row_stop, strip_rows):
            row_end = min(row_off + strip_rows, row_stop)
            band = self._decode(
                self._variable[self.time_index, row_off:row_end, col_start:col_stop]
            )
            yield row_off, col_start, band


class NetCDFScan(PixelScan["NetCDFRaster"]):
    """Reusable description of a one-pass NetCDF-to-DuckDB pixel scan."""


class NetCDFH3Scan(H3PixelScan["NetCDFRaster"]):
    """Reusable description of a one-pass NetCDF-to-H3-cell scan."""


def _pixel_boundary(
    raster: NetCDFRaster, row: int, column: int
) -> tuple[Point, Point, Point, Point]:
    """Pixel corner boundary, in the same counter-clockwise convention as
    `ZarrRaster.pixel_boundary`/`geotiff._pixel_boundaries`.

    See `crc_sdk.connectors.blocks.pixel_boundary` for why the corners are
    built from `abs(step)` in a fixed NW/SW/SE/NE order.
    """
    return pixel_boundary(
        raster._lat, raster._lon, raster._lat_step, raster._lon_step, row, column
    )


class NetCDFBlockReader:
    """`BlockReader` over one `NetCDFRaster` per block label (one file per year).

    Each file's whole time axis is one block: for a bounded window it is read
    in one strided read per strip and reduced by ``spec.statistic`` (EDO's
    dekadal SMI -> its annual minimum). A cached "annual extremes" file with a
    single time step reduces to itself under any statistic.
    """

    def __init__(
        self,
        rasters: dict[int, NetCDFRaster],
        *,
        spec: BlockSpec,
        strict_grid: bool = True,
    ) -> None:
        if not rasters:
            raise ValueError("at least one year is required")
        self.rasters = rasters
        self.spec = spec
        self._labels = tuple(sorted(rasters))
        self._reference = rasters[self._labels[0]]
        if strict_grid:
            for year, raster in rasters.items():
                if not np.array_equal(
                    raster._lat, self._reference._lat
                ) or not np.array_equal(raster._lon, self._reference._lon):
                    raise ValueError(
                        f"year {year} raster grid does not match the others "
                        "in this stack"
                    )

    @property
    def lat(self) -> np.ndarray[Any, Any]:
        return self._reference._lat

    @property
    def lon(self) -> np.ndarray[Any, Any]:
        return self._reference._lon

    @property
    def labels(self) -> tuple[int, ...]:
        return self._labels

    @property
    def leading_axis_samples(self) -> int:
        return max(int(raster._variable.shape[0]) for raster in self.rasters.values())

    @property
    def block_height(self) -> int:
        chunk_shape = getattr(self._reference._variable, "chunks", None)
        return int(chunk_shape[1]) if chunk_shape else 1

    def reduce_block(
        self, label: int, rows: slice, columns: slice
    ) -> np.ndarray[Any, Any]:
        raster = self.rasters[label]
        block = raster._decode(raster._variable[:, rows, columns])
        return reduce_block_samples(block, self.spec)

    def close(self) -> None:
        for raster in self.rasters.values():
            raster.close()


#: EDO's annual block is the whole file: its minimum over the dekads.
_EDO_SPEC = BlockSpec(statistic=BlockStatistic.minimum())


class EDOAnnualMinimaCurveSource(BlockExtremaCurveSource):
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

    A thin, EDO-flavoured constructor over `BlockExtremaCurveSource` with a
    `NetCDFBlockReader` -- the regression anchor for the generic source.
    """

    def __init__(
        self,
        rasters: dict[int, NetCDFRaster],
        metadata: RasterMetadata,
        strip_bytes: int = STRIP_BYTES,
    ) -> None:
        self.rasters = rasters
        super().__init__(
            reader=NetCDFBlockReader(rasters, spec=_EDO_SPEC),
            metadata=metadata,
            tail="lower",
            strip_bytes=strip_bytes,
        )

    def __enter__(self) -> EDOAnnualMinimaCurveSource:
        return self
