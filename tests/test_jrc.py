from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio  # type: ignore[import-untyped]
from rasterio.transform import from_origin  # type: ignore[import-untyped]

from crc_sdk.connectors import HurdleFitPolicy, JRCIngestPolicy, canonicalize_jrc_flood
from crc_sdk.connectors.duckdb.geotiff import JRCReturnPeriodRaster
from crc_sdk.connectors.duckdb.zarr import RasterMetadata

# One row, two columns: column 0 is flat/unfittable (constant across every
# return period, like a pixel that never floods within the tile), column 1
# has a real, increasing return-level curve -- the same two-pixel shape, and
# the same exact periods/values, as `FakeMixedReturnPeriodArray`/
# `FakeReturnPeriodArray` in tests/test_os_climate.py (known to fit cleanly
# with atom_probability=0.5), just spread across one GeoTIFF file per return
# period instead of one 3D Zarr array.
_RETURN_PERIODS = (2, 5, 10, 100, 1000)
_ROW_VALUES = {
    2: [0.0, 0.0],
    5: [0.0, 0.2],
    10: [0.0, 0.5],
    100: [0.0, 1.0],
    1000: [0.0, 2.0],
}
_TRANSFORM = from_origin(0.0, 0.02, 0.01, 0.01)


def _write_tile(
    directory: Path, *, overrides: dict[int, Any] | None = None
) -> dict[int, str]:
    overrides = overrides or {}
    urls = {}
    for period in _RETURN_PERIODS:
        path = directory / f"tile_RP{period}_depth.tif"
        array = np.asarray([_ROW_VALUES[period]], dtype=np.float32)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=1,
            width=2,
            count=1,
            dtype=array.dtype,
            crs="EPSG:4326",
            transform=overrides.get(period, _TRANSFORM),
            nodata=-9999.0,
        ) as dst:
            dst.write(array, 1)
        urls[period] = str(path)
    return urls


def _metadata() -> RasterMetadata:
    return RasterMetadata(
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        scenario="historical",
        year=0,
        units="m",
        path="test/jrc/tile",
    )


def test_grid_mismatch_raises(tmp_path: Path) -> None:
    # RP1000's grid is shifted -- same shape, different origin.
    shifted = from_origin(1.0, 0.02, 0.01, 0.01)
    urls = _write_tile(tmp_path, overrides={1000: shifted})

    with pytest.raises(ValueError, match="grid does not match"):
        JRCReturnPeriodRaster.open(urls, _metadata())


class _ReadSpy:
    """Proxies a rasterio dataset, counting `.read()` calls without touching it."""

    def __init__(self, dataset: Any, counters: dict[int, int], key: int) -> None:
        self._dataset = dataset
        self._counters = counters
        self._key = key

    def read(self, *args: Any, **kwargs: Any) -> Any:
        self._counters[self._key] += 1
        return self._dataset.read(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


def test_iter_curves_is_lazy_until_consumed(tmp_path: Path) -> None:
    urls = _write_tile(tmp_path)
    source = JRCReturnPeriodRaster.open(urls, _metadata())
    try:
        read_calls = {period: 0 for period in _RETURN_PERIODS}
        for period, raster in source._rasters.items():
            real_dataset = raster._dataset
            raster._dataset = _ReadSpy(real_dataset, read_calls, period)

        policy = JRCIngestPolicy(
            h3_resolution=9,
            family="gumbel_r",
            producer="tests",
            creation_version="1",
            on_fit_failure="skip",
        )
        stream = canonicalize_jrc_flood(source, policy)
        assert all(count == 0 for count in read_calls.values())

        table = stream.read_all()
        assert table.num_rows > 0
        assert any(count > 0 for count in read_calls.values())
    finally:
        source.close()


def test_canonicalize_jrc_flood_fits_hurdle_curve(tmp_path: Path) -> None:
    urls = _write_tile(tmp_path)
    with JRCReturnPeriodRaster.open(urls, _metadata()) as source:
        policy = JRCIngestPolicy(
            h3_resolution=9,
            family="gumbel_r",
            producer="tests",
            creation_version="1",
            hurdle=HurdleFitPolicy(atom_probability=0.5, atom_location=0.0),
            on_fit_failure="skip",
        )
        table = canonicalize_jrc_flood(source, policy).read_all()

    assert table.num_rows > 0
    assert set(table["curve_kind"].to_pylist()) == {"hurdle"}
    assert set(table["hazard_name"].to_pylist()) == {"RiverineInundation"}
    assert set(table["pathway"].to_pylist()) == {"historical"}
    # The flat/all-zero pixel has zero variance and should not survive the fit.
    assert len(set(table["source_id"].to_pylist())) == 1


def test_on_fit_failure_raise_aborts_on_constant_pixel(tmp_path: Path) -> None:
    urls = _write_tile(tmp_path)
    with JRCReturnPeriodRaster.open(urls, _metadata()) as source:
        policy = JRCIngestPolicy(
            h3_resolution=9,
            family="gumbel_r",
            producer="tests",
            creation_version="1",
        )
        with pytest.raises(ValueError, match="failed to fit source pixel"):
            canonicalize_jrc_flood(source, policy).read_all()


def test_jrc_ingest_policy_is_curve_fit_ingest_policy() -> None:
    from crc_sdk.connectors import CurveFitIngestPolicy

    assert JRCIngestPolicy is CurveFitIngestPolicy
