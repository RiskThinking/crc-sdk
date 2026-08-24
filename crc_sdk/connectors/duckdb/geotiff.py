"""Lazy Arrow bridge from windowed GeoTIFF/COG rasters into DuckDB."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping
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
from .stream import ArrowBatchSource, DuckDBPipeline
from .zarr import Bounds, Point, RasterCurve, RasterMetadata

logger = logging.getLogger(__name__)

# Decompressed bytes per strip read; bounds worker RAM independent of raster
# size while keeping the GDAL read count low.
STRIP_BYTES = 256 * 1024**2
GDAL_CACHE_MB = 128
# Cap subsample points materialized per broadcast to bound worker RAM.
CHUNK_POINTS = 262_144


class RasterBoundsError(ValueError):
    """Requested bounds do not contain a raster pixel."""


def _vsi_path(uri: str) -> str:
    """Translate a `gs://`/`s3://`/`http(s)://` URI to a GDAL VSI path."""
    if uri.startswith("gs://"):
        return "/vsigs/" + uri[len("gs://") :]
    if uri.startswith("s3://"):
        return "/vsis3/" + uri[len("s3://") :]
    if uri.startswith(("http://", "https://")):
        return "/vsicurl/" + uri
    return uri


def _default_gcs_credentials() -> str | None:
    """Path to gcloud's Application Default Credentials, if not already set.

    GDAL's `/vsigs/` driver -- unlike Python's own `google-auth` -- does not
    fall back to `gcloud auth application-default login`'s credentials on
    its own; without `GOOGLE_APPLICATION_CREDENTIALS` set, every private
    bucket read fails with an opaque `InvalidCredentials` GDAL error even
    when the ambient gcloud session is perfectly valid. Respects an
    explicitly configured `GOOGLE_APPLICATION_CREDENTIALS` (returns `None`
    so the caller's own value always wins) and only looks at gcloud's own
    well-known config location, matching the `gcloud` CLI's own convention.
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return None
    config_dir = os.environ.get("CLOUDSDK_CONFIG")
    if not config_dir:
        config_dir = (
            os.path.join(os.environ["APPDATA"], "gcloud")
            if os.name == "nt" and "APPDATA" in os.environ
            else os.path.expanduser("~/.config/gcloud")
        )
    candidate = Path(config_dir) / "application_default_credentials.json"
    return str(candidate) if candidate.is_file() else None


def _materialize_local(uri: str, cache_dir: str | Path) -> Path:
    """Fetch `uri` into `cache_dir` via fsspec if not already cached there."""
    if "://" not in uri:
        return Path(uri)

    cache_dir = Path(cache_dir)
    filesystem, path = fsspec.core.url_to_fs(uri)
    digest = hashlib.sha1(uri.encode()).hexdigest()[:16]
    local_path = cache_dir / f"{Path(path).name}.{digest}"
    if local_path.is_file() and local_path.stat().st_size > 0:
        return local_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_name(local_path.name + ".tmp")
    filesystem.get(path, str(tmp_path))
    tmp_path.replace(local_path)
    return local_path


def trim_cache_dir(
    cache_dir: str | Path, max_bytes: int, *, safe_seconds: int = 300
) -> None:
    """Evict least-recently-accessed files once `cache_dir` exceeds `max_bytes`.

    Meant to be called periodically by an orchestrator sharing one
    `cache_dir` across many :meth:`GeoTiffRaster.open` calls (each call only
    fetches its own file and never evicts on its own). Files accessed within
    `safe_seconds` are skipped, since those are likely still open by a
    concurrent reader.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return
    now = time.time()
    entries = []
    for path in cache_dir.glob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append((stat.st_atime, stat.st_size, path))

    total = sum(size for _, size, _ in entries)
    for atime, size, path in sorted(entries):
        if total <= max_bytes:
            break
        if now - atime < safe_seconds:
            continue
        try:
            path.unlink()
            total -= size
        except FileNotFoundError:
            pass


def _require_raster_extra() -> None:
    """Raise a clear error before any rasterio import if the extra is missing."""
    try:
        import rasterio  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "GeoTIFF/COG support requires `pip install crc-sdk[raster]`"
        ) from error


def _pixel_to_crs(
    transform: Any, columns: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = transform.a, transform.b, transform.c
    d, e, f = transform.d, transform.e, transform.f
    return c + a * columns + b * rows, f + d * columns + e * rows


def _wgs84() -> Any:
    _require_raster_extra()
    from rasterio.crs import CRS  # type: ignore[import-untyped]

    return CRS.from_epsg(4326)


def _to_wgs84(crs: Any, xs: Any, ys: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (latitude, longitude) arrays in WGS84 for CRS coordinates."""
    _require_raster_extra()
    import rasterio.warp  # type: ignore[import-untyped]

    warp_transform = rasterio.warp.transform
    wgs84 = _wgs84()
    if crs == wgs84:
        return np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)
    lons, lats = warp_transform(crs, wgs84, xs, ys)
    return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)


class GeoTiffRaster:
    """A remote or local GeoTIFF/COG raster with lazy DuckDB scan helpers."""

    def __init__(
        self,
        dataset: Any,
        *,
        band: int = 1,
        assumed_crs: str | None = None,
        connection: DuckDBConnection | None = None,
        work_dir: str | Path | None = None,
        _env: Any | None = None,
    ) -> None:
        self._dataset = dataset
        self._env = _env
        self.band = band
        # An explicit connection means the caller is already in control; only
        # build (and resource-tune) one when they didn't supply their own.
        # No extensions requested: scan()/scan_h3() only use core Arrow
        # ingestion plus plain relational aggregate/project, never spatial,
        # httpfs (GDAL's own VSI handles remote reads), or h3 (h3ronpy is
        # used directly) -- loading them here would be pure overhead.
        self.connection = connection or DuckDBConnection.for_analytics(
            work_dir or default_work_dir(), extensions=()
        )
        self.crs, self._crs_is_assumed = self._resolve_crs(assumed_crs)

    @classmethod
    def open(
        cls,
        uri: str | Path,
        *,
        band: int = 1,
        assumed_crs: str | None = None,
        cache_dir: str | Path | None = None,
        connection: DuckDBConnection | None = None,
        work_dir: str | Path | None = None,
    ) -> GeoTiffRaster:
        """Open a local path, or a `gs://`/`s3://`/`http(s)://` URI.

        `cache_dir=None` (default) streams directly over the network via
        GDAL's own VSI curl support, with no local disk write -- the right
        default when each file is read once. Pass `cache_dir` to materialize
        a local copy first (via fsspec) when the same file is read more than
        once, or when direct VSI network access isn't viable (e.g. private
        buckets without ambient GDAL credentials).
        """
        _require_raster_extra()
        import rasterio

        uri_str = str(uri)
        open_path = (
            str(_materialize_local(uri_str, cache_dir))
            if cache_dir is not None
            else _vsi_path(uri_str)
        )

        env_options: dict[str, Any] = {"GDAL_CACHEMAX": GDAL_CACHE_MB}
        adc_path = _default_gcs_credentials()
        if adc_path is not None:
            env_options["GOOGLE_APPLICATION_CREDENTIALS"] = adc_path
        env = rasterio.Env(**env_options)
        env.__enter__()
        try:
            dataset = rasterio.open(open_path)
        except Exception:
            env.__exit__(None, None, None)
            raise
        try:
            return cls(
                dataset,
                band=band,
                assumed_crs=assumed_crs,
                connection=connection,
                work_dir=work_dir,
                _env=env,
            )
        except Exception:
            dataset.close()
            env.__exit__(None, None, None)
            raise

    def close(self) -> None:
        self._dataset.close()
        if self._env is not None:
            self._env.__exit__(None, None, None)

    def __enter__(self) -> GeoTiffRaster:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def bounds(self) -> Bounds:
        """Raster bounds in its own CRS (as resolved via `assumed_crs`)."""
        raster_bounds = self._dataset.bounds
        return (
            raster_bounds.left,
            raster_bounds.bottom,
            raster_bounds.right,
            raster_bounds.top,
        )

    def bounds_from_wgs84(self, bounds: Bounds) -> Bounds:
        """Transform WGS84 bounds into this raster's native CRS."""
        _require_raster_extra()
        from rasterio.warp import transform_bounds

        native = transform_bounds(_wgs84(), self.crs, *bounds)
        return native[0], native[1], native[2], native[3]

    def write_crop(self, bounds: Bounds, destination: str | Path) -> Path:
        """Write a native-CRS window to a local GeoTIFF."""
        _require_raster_extra()
        import rasterio
        from rasterio.errors import WindowError
        from rasterio.windows import Window, from_bounds

        window = from_bounds(*bounds, transform=self._dataset.transform)
        try:
            window = (
                window.round_offsets()
                .round_lengths()
                .intersection(Window(0, 0, self._dataset.width, self._dataset.height))
            )
        except WindowError as error:
            raise RasterBoundsError("bounds do not intersect the raster") from error
        if window.width < 1 or window.height < 1:
            raise RasterBoundsError("bounds do not intersect the raster")
        profile = self._dataset.profile.copy()
        profile.update(
            width=int(window.width),
            height=int(window.height),
            transform=self._dataset.window_transform(window),
            compress="deflate",
            tiled=False,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        target = Path(destination)
        with rasterio.open(target, "w", **profile) as output:
            output.write(self._dataset.read(window=window))
        return target

    @property
    def nodata(self) -> float | None:
        value = self._dataset.nodata
        return None if value is None else float(value)

    @property
    def pixel_size_meters(self) -> tuple[float, float]:
        """Approximate (width, height) pixel spacing in meters.

        For geographic CRSs, evaluated at the equator-most row of the
        raster's own bounds, where longitude degrees are widest, so the
        result is a conservative (largest) estimate for the whole raster.
        """
        res_x, res_y = abs(self._dataset.res[0]), abs(self._dataset.res[1])
        if self.crs.is_projected:
            return res_x, res_y

        bounds = self._dataset.bounds
        if bounds.bottom <= 0.0 <= bounds.top:
            y0 = 0.0
        elif abs(bounds.bottom) < abs(bounds.top):
            y0 = bounds.bottom
        else:
            y0 = bounds.top
        center_x = (bounds.left + bounds.right) / 2.0

        _require_raster_extra()
        from rasterio.warp import transform as warp_transform

        lon_w, _ = warp_transform(
            self.crs, _wgs84(), [center_x, center_x + res_x], [y0, y0]
        )
        _, lat_h = warp_transform(
            self.crs, _wgs84(), [center_x, center_x], [y0, y0 + res_y]
        )
        meters_per_degree = 111_320.0
        lat_rad = math.radians(lat_h[0])
        width_m = abs(lon_w[1] - lon_w[0]) * meters_per_degree * math.cos(lat_rad)
        height_m = abs(lat_h[1] - lat_h[0]) * meters_per_degree
        return width_m, height_m

    def scan(
        self, bounds: Bounds | None = None, *, strip_bytes: int = STRIP_BYTES
    ) -> GeoTiffScan:
        """Describe a reusable, out-of-core scan of raw pixel rows."""
        if strip_bytes < 1:
            raise ValueError("strip_bytes must be positive")
        return GeoTiffScan(self, bounds or self.bounds, strip_bytes)

    def scan_h3(
        self,
        bounds: Bounds | None = None,
        *,
        h3_resolution: int | None = None,
        max_subsample: float = 2.0,
        mask: Callable[[np.ndarray], np.ndarray] | None = None,
        reduce: Literal["max", "min"] = "max",
        strip_bytes: int = STRIP_BYTES,
    ) -> GeoTiffH3Scan:
        """Describe a reusable scan that samples pixels straight to H3 cells.

        `h3_resolution=None` infers the finest resolution the raster's own
        pixel spacing supports (`crc_sdk.geometry.h3.pixel_grid_resolution`).
        Nodata and non-finite pixels are always excluded; `mask` is an
        optional additional predicate over a raw pixel-value window for
        domain-specific filtering. `reduce` picks how multiple samples
        landing in one cell are combined. `max_subsample` always caps
        subsample points per pixel axis -- including when `h3_resolution` is
        given explicitly, where it bounds the sample grid instead of
        guaranteeing every cell gets a sample (that guarantee only holds
        when the resolution was itself picked to respect this cap).
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
        return GeoTiffH3Scan(
            self,
            bounds or self.bounds,
            h3_resolution=resolution,
            max_subsample=max_subsample,
            mask=mask,
            reduce=reduce,
            strip_bytes=strip_bytes,
        )

    def _resolve_crs(self, assumed_crs: str | None) -> tuple[Any, bool]:
        """Real CRS if present; otherwise the caller-supplied fallback. A
        raster can carry a valid affine transform with no CRS tag (upstream
        metadata gap) -- a different failure mode from being genuinely
        unprojected, and one that shouldn't be treated the same."""
        _require_raster_extra()
        from rasterio.crs import CRS

        if self._dataset.crs is not None:
            return self._dataset.crs, False
        if assumed_crs is None:
            raise ValueError(
                f"{self._dataset.name}: raster has no CRS and no assumed_crs "
                "was given; pass one (e.g. assumed_crs='EPSG:4326') if the "
                "affine transform is in that datum, or fix the raster's "
                "metadata upstream"
            )
        crs = CRS.from_user_input(assumed_crs)
        if crs.is_geographic and abs(self._dataset.transform.a) > 1:
            # Geographic pixel sizes are fractions of a degree; a transform
            # step of a whole unit or more usually means the file's affine
            # is actually projected (the assumed CRS is likely wrong).
            logger.warning(
                "%s has transform step %s under assumed geographic CRS %s "
                "-- that's a huge pixel in degrees; the assumed CRS is "
                "likely wrong",
                self._dataset.name,
                self._dataset.transform.a,
                crs,
            )
        return crs, True

    def _pixel_window(self, bounds: Bounds) -> tuple[int, int, int, int]:
        _require_raster_extra()
        from rasterio.windows import from_bounds  # type: ignore[import-untyped]

        min_x, min_y, max_x, max_y = bounds
        if min_x > max_x or min_y > max_y:
            raise ValueError("bounds must be (min_x, min_y, max_x, max_y)")
        window = from_bounds(
            min_x, min_y, max_x, max_y, transform=self._dataset.transform
        )
        window = window.round_offsets().round_lengths()
        col_start = max(0, int(window.col_off))
        row_start = max(0, int(window.row_off))
        col_stop = min(self._dataset.width, int(window.col_off + window.width))
        row_stop = min(self._dataset.height, int(window.row_off + window.height))
        if col_start >= col_stop or row_start >= row_stop:
            raise ValueError("bounds do not intersect the raster")
        return col_start, row_start, col_stop, row_stop

    def _strips(
        self, bounds: Bounds, strip_bytes: int
    ) -> Iterator[tuple[Any, np.ndarray]]:
        """Yield (window, band_array) pairs in wide, block-aligned strips.

        Few wide block-aligned strips beat thousands of per-block GDAL
        reads; `strip_bytes` bounds the decompressed strip size regardless
        of how large the raster itself is.
        """
        _require_raster_extra()
        from rasterio.windows import Window

        dataset = self._dataset
        col_start, row_start, col_stop, row_stop = self._pixel_window(bounds)
        width = col_stop - col_start
        block_height = dataset.block_shapes[self.band - 1][0]
        itemsize = np.dtype(dataset.dtypes[self.band - 1]).itemsize
        rows_per_strip = max(block_height, strip_bytes // max(1, width * itemsize))
        strip_rows = max(block_height, (rows_per_strip // block_height) * block_height)

        for row_off in range(row_start, row_stop, strip_rows):
            win_h = min(strip_rows, row_stop - row_off)
            window = Window(col_start, row_off, width, win_h)
            yield window, dataset.read(self.band, window=window)


@dataclass(frozen=True)
class GeoTiffScan:
    """Reusable description of a one-pass GeoTIFF-to-DuckDB pixel scan."""

    raster: GeoTiffRaster
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
        return ArrowBatchSource(schema, self._batches).relation(
            connection=connection or self.raster.connection
        )

    def pipeline(self, *, connection: DuckDBConnection | None = None) -> DuckDBPipeline:
        """Compose this bounded scan with lazy DuckDB relational operations."""
        return DuckDBPipeline(self, connection=connection)

    def _batches(self) -> Iterator[Any]:
        raster = self.raster
        for window, band in raster._strips(self.bounds, self.strip_bytes):
            valid = np.isfinite(band)
            if raster.nodata is not None:
                valid &= band != raster.nodata
            if not valid.any():
                continue
            row_idx, col_idx = np.where(valid)
            values = band[row_idx, col_idx].astype(np.float32, copy=False)
            columns = col_idx.astype(np.float64) + window.col_off + 0.5
            rows = row_idx.astype(np.float64) + window.row_off + 0.5
            xs, ys = _pixel_to_crs(raster._dataset.transform, columns, rows)
            lats, lons = _to_wgs84(raster.crs, xs, ys)
            yield pa.record_batch(
                {
                    "longitude": pa.array(lons),
                    "latitude": pa.array(lats),
                    "value": pa.array(values),
                }
            )


@dataclass(frozen=True)
class GeoTiffH3Scan:
    """Reusable description of a one-pass GeoTIFF-to-H3-cell scan."""

    raster: GeoTiffRaster
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

        for window, band in raster._strips(self.bounds, self.strip_bytes):
            valid = np.isfinite(band)
            if raster.nodata is not None:
                valid &= band != raster.nodata
            if self.mask is not None:
                valid &= self.mask(band)
            if not valid.any():
                continue

            row_idx, col_idx = np.where(valid)
            values = band[row_idx, col_idx].astype(np.float64, copy=False)
            strip_rows = row_idx + window.row_off
            strip_columns = col_idx + window.col_off

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
        rows = row_grid.ravel() + 0.5
        columns = column_grid.ravel() + 0.5

        xs, ys = _pixel_to_crs(self.raster._dataset.transform, columns, rows)
        lats, lons = _to_wgs84(self.raster.crs, xs, ys)
        n_sub = len(row_offsets) * len(column_offsets)
        return sample_grid_to_h3(
            lons,
            lats,
            np.repeat(values, n_sub),
            resolution=self.h3_resolution,
            reduce=self.reduce,
        )


def _pixel_boundary(
    raster: GeoTiffRaster, row: int, column: int
) -> tuple[Point, Point, Point, Point]:
    """Pixel corner boundary in counter-clockwise WGS84 order.

    Mirrors `ZarrRaster.pixel_boundary`'s corner convention exactly, so a
    `RasterCurve.boundary` means the same thing regardless of which
    `CurveSource` produced it.
    """
    columns = np.array([column, column, column + 1, column + 1], dtype=np.float64)
    rows = np.array([row, row + 1, row + 1, row], dtype=np.float64)
    xs, ys = _pixel_to_crs(raster._dataset.transform, columns, rows)
    lats, lons = _to_wgs84(raster.crs, xs, ys)
    points = list(zip(lons.tolist(), lats.tolist()))
    return (points[0], points[1], points[2], points[3])


class JRCReturnPeriodRaster:
    """Presents a same-tile stack of per-return-period GeoTIFFs as one curve source.

    JRC ships one GeoTIFF per return period per tile, all sharing the same
    grid -- structurally different from `ZarrRaster`'s single array with a
    leading return-period axis, but the same *shape of information*: one
    curve of (return period, value) per pixel. This class re-presents the
    GeoTIFF stack that way, satisfying `crc_sdk.connectors.adapters.CurveSource`
    so `canonicalize_curve_source` can fit it exactly like an OS-Climate
    raster, via `crc_sdk.connectors.jrc.canonicalize_jrc_flood`. It reuses
    `GeoTiffRaster.open()` unchanged for the actual I/O -- no new low-level
    reader, just a different pixel-stack iteration order.
    """

    def __init__(
        self,
        rasters: Mapping[int, GeoTiffRaster],
        metadata: RasterMetadata,
        *,
        strip_bytes: int = STRIP_BYTES,
    ) -> None:
        if not rasters:
            raise ValueError("at least one return-period raster is required")
        if strip_bytes < 1:
            raise ValueError("strip_bytes must be positive")
        self._rasters = dict(rasters)
        self.metadata = metadata
        self.strip_bytes = strip_bytes
        self._periods = tuple(sorted(self._rasters))
        self._axis_values = np.asarray(self._periods, dtype=np.float64)
        self._reference = self._rasters[self._periods[0]]
        reference_dataset = self._reference._dataset
        for period, raster in self._rasters.items():
            dataset = raster._dataset
            if (
                dataset.transform != reference_dataset.transform
                or dataset.width != reference_dataset.width
                or dataset.height != reference_dataset.height
            ):
                raise ValueError(
                    f"return period {period} raster grid does not match "
                    "the others in this stack"
                )

    @classmethod
    def open(
        cls,
        urls: Mapping[int, str | Path],
        metadata: RasterMetadata,
        *,
        band: int = 1,
        assumed_crs: str | None = None,
        cache_dir: str | Path | None = None,
        connection: DuckDBConnection | None = None,
        work_dir: str | Path | None = None,
        strip_bytes: int = STRIP_BYTES,
    ) -> JRCReturnPeriodRaster:
        """Open one `GeoTiffRaster` per return period, keyed by return period."""
        rasters: dict[int, GeoTiffRaster] = {}
        try:
            for period, url in urls.items():
                rasters[period] = GeoTiffRaster.open(
                    url,
                    band=band,
                    assumed_crs=assumed_crs,
                    cache_dir=cache_dir,
                    connection=connection,
                    work_dir=work_dir,
                )
        except Exception:
            # Reverse of open order: each GeoTiffRaster.open() pushes its own
            # rasterio.Env onto a global, LIFO-nested stack -- closing out of
            # order raises rasterio.errors.EnvError instead of cleanly
            # unwinding it.
            for raster in reversed(list(rasters.values())):
                raster.close()
            raise
        return cls(rasters, metadata, strip_bytes=strip_bytes)

    def close(self) -> None:
        """Close every underlying `GeoTiffRaster`, most-recently-opened first.

        Reverse of open order, for the same LIFO `rasterio.Env`-stack reason
        as the cleanup path in `open()`.
        """
        for raster in reversed(list(self._rasters.values())):
            raster.close()

    def __enter__(self) -> JRCReturnPeriodRaster:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def axis_name(self) -> str:
        return "return period"

    @property
    def return_period_support(self) -> tuple[float, float]:
        return float(self._periods[0]), float(self._periods[-1])

    @property
    def bounds(self) -> Bounds:
        return self._reference.bounds

    def bounds_from_wgs84(self, bounds: Bounds) -> Bounds:
        return self._reference.bounds_from_wgs84(bounds)

    def iter_curves(self, bounds: Bounds | None = None) -> Iterator[RasterCurve]:
        """Stream source pixels with complete leading-axis (return-period) curves.

        Reads the same pixel window from every return-period file in
        lock-step, in wide row bands bounded by `strip_bytes` regardless of
        how large the tile is -- the same bounded-memory strategy
        `GeoTiffRaster._strips` already uses for a single file, generalized
        to `n_periods` files read together.
        """
        _require_raster_extra()
        from rasterio.windows import Window

        reference = self._reference
        col_start, row_start, col_stop, row_stop = reference._pixel_window(
            bounds or self.bounds
        )
        width = col_stop - col_start
        ordered = [self._rasters[period] for period in self._periods]
        n_periods = len(ordered)
        itemsize = max(
            np.dtype(raster._dataset.dtypes[raster.band - 1]).itemsize
            for raster in ordered
        )
        block_height = reference._dataset.block_shapes[reference.band - 1][0]
        rows_per_strip = max(
            block_height, self.strip_bytes // max(1, width * itemsize * n_periods)
        )
        strip_rows = max(block_height, (rows_per_strip // block_height) * block_height)

        for row_off in range(row_start, row_stop, strip_rows):
            win_h = min(strip_rows, row_stop - row_off)
            window = Window(col_start, row_off, width, win_h)
            stack = np.empty((n_periods, win_h, width), dtype=np.float64)
            for index, raster in enumerate(ordered):
                band = raster._dataset.read(raster.band, window=window).astype(
                    np.float64, copy=False
                )
                finite = np.isfinite(band)
                if raster.nodata is not None:
                    finite &= band != raster.nodata
                stack[index] = np.where(finite, band, np.nan)

            valid_any = np.isfinite(stack).any(axis=0)
            if not valid_any.any():
                continue
            local_rows, local_columns = np.where(valid_any)
            for local_row, local_column in zip(
                local_rows.tolist(), local_columns.tolist()
            ):
                source_row = row_off + local_row
                source_column = col_start + local_column
                yield RasterCurve(
                    row=source_row,
                    column=source_column,
                    boundary=_pixel_boundary(reference, source_row, source_column),
                    axis_values=self._axis_values.copy(),
                    values=stack[:, local_row, local_column].copy(),
                )
