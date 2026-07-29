from pathlib import Path

import numpy as np
import pytest
import rasterio  # type: ignore[import-untyped]
from rasterio.transform import from_origin  # type: ignore[import-untyped]

from crc_sdk.connectors.duckdb.geotiff import GeoTiffRaster, trim_cache_dir

# ~0.01 degrees per pixel near the equator, well within the "small pixel"
# regime so pixel_grid_resolution picks a coarse-enough H3 resolution that
# several pixels land in one cell (needed to exercise the max/min reduce).
_PIXEL_SIZE_DEG = 0.01
_TRANSFORM = from_origin(0.0, 0.04, _PIXEL_SIZE_DEG, _PIXEL_SIZE_DEG)


def _write_geotiff(
    path: Path,
    array: np.ndarray,
    *,
    crs: str | None = "EPSG:4326",
    nodata: float | None = None,
) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=_TRANSFORM,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


@pytest.fixture
def sample_array() -> np.ndarray:
    # 4x4 grid; row 0 holds the largest values so max-reduce is unambiguous.
    return np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [-9999.0, 0.5, 1.5, 2.5],
        ],
        dtype=np.float32,
    )


def test_open_local_reads_metadata(tmp_path: Path, sample_array: np.ndarray) -> None:
    path = _write_geotiff(tmp_path / "sample.tif", sample_array, nodata=-9999.0)
    raster = GeoTiffRaster.open(path, work_dir=tmp_path / "work")
    try:
        assert raster.crs.to_epsg() == 4326
        assert raster.nodata == -9999.0
        assert raster.bounds == pytest.approx((0.0, 0.0, 0.04, 0.04))
        width_m, height_m = raster.pixel_size_meters
        assert width_m == pytest.approx(1113.2, abs=1.0)
        assert height_m == pytest.approx(1113.2, abs=1.0)
    finally:
        raster.close()


def test_open_missing_crs_requires_assumed_crs(
    tmp_path: Path, sample_array: np.ndarray
) -> None:
    path = _write_geotiff(tmp_path / "no_crs.tif", sample_array, crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        GeoTiffRaster.open(path, work_dir=tmp_path / "work")

    raster = GeoTiffRaster.open(
        path, assumed_crs="EPSG:4326", work_dir=tmp_path / "work"
    )
    try:
        assert raster.crs.to_epsg() == 4326
    finally:
        raster.close()


def test_scan_returns_raw_pixel_rows(tmp_path: Path, sample_array: np.ndarray) -> None:
    path = _write_geotiff(tmp_path / "sample.tif", sample_array, nodata=-9999.0)
    raster = GeoTiffRaster.open(path, work_dir=tmp_path / "work")
    try:
        relation = raster.scan().relation()
        rows = relation.fetchall()
        # nodata pixel (-9999.0) is always excluded, every other pixel kept.
        assert len(rows) == sample_array.size - 1
        values = {round(value, 3) for _, _, value in rows}
        expected = {round(float(v), 3) for v in sample_array.flatten() if v != -9999.0}
        assert values == expected
    finally:
        raster.close()


def test_scan_h3_reduces_to_max_per_cell(
    tmp_path: Path, sample_array: np.ndarray
) -> None:
    path = _write_geotiff(tmp_path / "sample.tif", sample_array, nodata=-9999.0)
    raster = GeoTiffRaster.open(path, work_dir=tmp_path / "work")
    try:
        # Coarse enough that the whole 4x4 grid falls into very few cells.
        scan = raster.scan_h3(h3_resolution=2, reduce="max")
        rows = scan.relation().fetchall()
        assert rows
        assert max(value for _, value in rows) == pytest.approx(40.0, abs=0.01)
        assert all(value > -9999.0 for _, value in rows)
    finally:
        raster.close()


def test_scan_h3_min_reduce(tmp_path: Path, sample_array: np.ndarray) -> None:
    path = _write_geotiff(tmp_path / "sample.tif", sample_array, nodata=-9999.0)
    raster = GeoTiffRaster.open(path, work_dir=tmp_path / "work")
    try:
        scan = raster.scan_h3(h3_resolution=2, reduce="min")
        rows = scan.relation().fetchall()
        assert min(value for _, value in rows) == pytest.approx(0.5, abs=0.01)
    finally:
        raster.close()


def test_scan_h3_mask_filters_pixels(tmp_path: Path, sample_array: np.ndarray) -> None:
    path = _write_geotiff(tmp_path / "sample.tif", sample_array, nodata=-9999.0)
    raster = GeoTiffRaster.open(path, work_dir=tmp_path / "work")
    try:
        scan = raster.scan_h3(
            h3_resolution=2, reduce="max", mask=lambda band: band > 5.0
        )
        rows = scan.relation().fetchall()
        assert all(value > 5.0 for _, value in rows)
    finally:
        raster.close()


def test_scan_h3_cross_strip_merge_matches_single_strip(
    tmp_path: Path, sample_array: np.ndarray
) -> None:
    path = _write_geotiff(tmp_path / "sample.tif", sample_array, nodata=-9999.0)
    raster = GeoTiffRaster.open(path, work_dir=tmp_path / "work")
    try:
        single_strip = raster.scan_h3(h3_resolution=2, reduce="max").relation()
        single_result = dict(single_strip.fetchall())

        # Force one row per strip so every pixel row is read separately;
        # the final DuckDB GROUP BY must still merge them correctly.
        multi_strip = raster.scan_h3(
            h3_resolution=2, reduce="max", strip_bytes=1
        ).relation()
        multi_result = dict(multi_strip.fetchall())

        assert multi_result == single_result
    finally:
        raster.close()


def test_scan_h3_auto_resolution_covers_every_pixel(
    tmp_path: Path, sample_array: np.ndarray
) -> None:
    path = _write_geotiff(tmp_path / "sample.tif", sample_array, nodata=-9999.0)
    raster = GeoTiffRaster.open(path, work_dir=tmp_path / "work")
    try:
        scan = raster.scan_h3()
        rows = scan.relation().fetchall()
        assert len(rows) >= 1
        assert scan.h3_resolution > 0
    finally:
        raster.close()


def test_trim_cache_dir_evicts_oldest_first(tmp_path: Path) -> None:
    import os
    import time

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old_file = cache_dir / "old.tif"
    new_file = cache_dir / "new.tif"
    old_file.write_bytes(b"x" * 100)
    new_file.write_bytes(b"y" * 100)

    old_time = time.time() - 10_000
    os.utime(old_file, (old_time, old_time))

    trim_cache_dir(cache_dir, max_bytes=150, safe_seconds=0)

    assert not old_file.exists()
    assert new_file.exists()


def test_trim_cache_dir_skips_recently_accessed(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    only_file = cache_dir / "recent.tif"
    only_file.write_bytes(b"x" * 100)

    trim_cache_dir(cache_dir, max_bytes=0, safe_seconds=300)

    assert only_file.exists()
