from __future__ import annotations

from pathlib import Path
from typing import Any

import h5netcdf  # type: ignore[import-untyped]
import numpy as np
import pytest
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from crc_sdk.connectors.duckdb.netcdf import (
    NetCDFRaster,
    _pixel_boundary,
    _strip_row_count,
)

# 4x4 grid, descending lat (like EDO's own north-to-south axis) and
# ascending lon; row 0 (northernmost) holds the largest values so
# max-reduce is unambiguous, mirroring test_geotiff.py's sample_array.
_LAT = np.array([0.04, 0.03, 0.02, 0.01], dtype=np.float64)
_LON = np.array([0.00, 0.01, 0.02, 0.03], dtype=np.float64)
_FILL = 1e20


def _write_netcdf(
    path: Path,
    array: np.ndarray,
    *,
    fill_value: float | None = _FILL,
    lat: np.ndarray = _LAT,
) -> Path:
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"time": array.shape[0], "lat": len(lat), "lon": len(_LON)}
        f.create_variable("lat", ("lat",), dtype=np.float64, data=lat)
        f.create_variable("lon", ("lon",), dtype=np.float64, data=_LON)
        variable = f.create_variable(
            "sminx", ("time", "lat", "lon"), dtype=np.float32, data=array
        )
        if fill_value is not None:
            variable.attrs["_FillValue"] = np.float32(fill_value)
    return path


@pytest.fixture
def sample_array() -> np.ndarray:
    return np.array(
        [
            [
                [10.0, 20.0, 30.0, 40.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [1e20, 0.5, 1.5, 2.5],
            ]
        ],
        dtype=np.float32,
    )


def test_open_local_reads_metadata(tmp_path: Path, sample_array: np.ndarray) -> None:
    path = _write_netcdf(tmp_path / "sample.nc", sample_array)
    raster = NetCDFRaster.open(path, variable="sminx", work_dir=tmp_path / "work")
    try:
        assert raster.bounds == pytest.approx((-0.005, 0.005, 0.035, 0.045))
        assert raster.nodata == pytest.approx(1e20)
        width_m, height_m = raster.pixel_size_meters
        assert width_m == pytest.approx(1113.2, abs=1.0)
        assert height_m == pytest.approx(1113.2, abs=1.0)
    finally:
        raster.close()


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrong_dims.nc"
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"lat": len(_LAT), "lon": len(_LON), "time": 1}
        f.create_variable("lat", ("lat",), dtype=np.float64, data=_LAT)
        f.create_variable("lon", ("lon",), dtype=np.float64, data=_LON)
        # Wrong order: (lat, lon, time) instead of (time, lat, lon).
        f.create_variable(
            "sminx",
            ("lat", "lon", "time"),
            dtype=np.float32,
            data=np.zeros((4, 4, 1), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="expected exactly"):
        NetCDFRaster.open(path, variable="sminx", work_dir=tmp_path / "work")


def test_scan_returns_raw_pixel_rows(tmp_path: Path, sample_array: np.ndarray) -> None:
    path = _write_netcdf(tmp_path / "sample.nc", sample_array)
    raster = NetCDFRaster.open(path, variable="sminx", work_dir=tmp_path / "work")
    try:
        rows = raster.scan().pipeline().select("*").relation().fetchall()
        # The fill-value pixel is always excluded, every other pixel kept.
        assert len(rows) == sample_array.size - 1
        values = {round(value, 3) for _, _, value in rows}
        expected = {round(float(v), 3) for v in sample_array.flatten() if v != 1e20}
        assert values == expected
    finally:
        raster.close()


def test_scan_h3_reduces_to_max_per_cell(
    tmp_path: Path, sample_array: np.ndarray
) -> None:
    path = _write_netcdf(tmp_path / "sample.nc", sample_array)
    raster = NetCDFRaster.open(path, variable="sminx", work_dir=tmp_path / "work")
    try:
        rows = (
            raster.scan_h3(h3_resolution=4).pipeline().select("*").relation().fetchall()
        )
        assert rows
        assert max(value for _, value in rows) == pytest.approx(40.0)
    finally:
        raster.close()


def test_scan_h3_mask_filters_pixels(tmp_path: Path, sample_array: np.ndarray) -> None:
    path = _write_netcdf(tmp_path / "sample.nc", sample_array)
    raster = NetCDFRaster.open(path, variable="sminx", work_dir=tmp_path / "work")
    try:
        rows = (
            raster.scan_h3(h3_resolution=4, mask=lambda band: band > 5.0)
            .relation()
            .fetchall()
        )
        assert rows
        assert all(value > 5.0 for _, value in rows)
    finally:
        raster.close()


class _ReadSpy:
    """Proxies an h5netcdf variable, counting reads without touching it."""

    def __init__(self, variable: Any, counter: list[int]) -> None:
        self._variable = variable
        self._counter = counter

    def __getitem__(self, key: Any) -> Any:
        self._counter.append(1)
        return self._variable[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._variable, name)


def test_scan_is_lazy_and_streams_into_duckdb(
    tmp_path: Path, sample_array: np.ndarray
) -> None:
    path = _write_netcdf(tmp_path / "sample.nc", sample_array)
    raster = NetCDFRaster.open(path, variable="sminx", work_dir=tmp_path / "work")
    try:
        reads: list[int] = []
        raster._variable = _ReadSpy(raster._variable, reads)

        relation = raster.scan().relation()
        assert reads == []

        count = relation.aggregate("count(*) AS row_count").fetchone()[0]
        assert count == sample_array.size - 1
        assert reads
    finally:
        raster.close()


@pytest.mark.parametrize("lat", [[0.04, 0.03, 0.02, 0.01], [0.01, 0.02, 0.03, 0.04]])
def test_pixel_boundary_is_always_counter_clockwise(
    tmp_path: Path, sample_array: np.ndarray, lat: list[float]
) -> None:
    """Winding must not depend on whether `lat` is ascending or descending."""
    path = _write_netcdf(
        tmp_path / "sample.nc", sample_array, lat=np.array(lat, dtype=np.float64)
    )
    raster = NetCDFRaster.open(path, variable="sminx", work_dir=tmp_path / "work")
    try:
        boundary = _pixel_boundary(raster, row=1, column=1)
        assert Polygon(boundary).exterior.is_ccw
    finally:
        raster.close()


def test_strip_row_count_accounts_for_leading_axis() -> None:
    # Many "dekads" (leading_axis) read at once per row must shrink the
    # strip just as much as an equally large year count would -- sizing
    # off year count alone (leading_axis=1 here, standing in for the old
    # bug) would pick a strip roughly `many / few` times too tall.
    few = _strip_row_count(4096, width=10, itemsize=8, block_height=1, leading_axis=1)
    many = _strip_row_count(4096, width=10, itemsize=8, block_height=1, leading_axis=40)
    assert many < few
    assert many * 10 * 8 * 40 <= 4096 + 10 * 8 * 40  # within one row's worth of budget


def test_open_requires_netcdf_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "h5netcdf":
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError, match="crc-sdk\\[netcdf\\]"):
        NetCDFRaster.open(tmp_path / "missing.nc", variable="sminx")
