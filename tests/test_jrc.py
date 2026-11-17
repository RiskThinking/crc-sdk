from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio  # type: ignore[import-untyped]
from rasterio.transform import from_origin  # type: ignore[import-untyped]

from crc_sdk._connectors.duckdb.geotiff import JRCReturnPeriodRaster
from crc_sdk._connectors.duckdb.zarr import RasterMetadata

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
