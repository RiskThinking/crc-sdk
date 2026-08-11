from __future__ import annotations

from pathlib import Path

import h5netcdf  # type: ignore[import-untyped]
import numpy as np
import pytest

from crc_sdk.connectors import EDOIngestPolicy, canonicalize_edo_drought
from crc_sdk.connectors.duckdb.netcdf import EDOAnnualMinimaCurveSource, NetCDFRaster
from crc_sdk.connectors.duckdb.zarr import RasterMetadata

_LAT = np.array([1.0, 0.0], dtype=np.float64)
_LON = np.array([10.0, 11.0], dtype=np.float64)

# Two rows, two columns: row 0/column 0 is flat/unfittable (the same annual
# minimum every year, like a pixel whose soil moisture never really
# varies), row 0/column 1 has six years of annual minima with one clear
# severe (low) year -- enough years (>=4) for `_annual_minima_curve` to
# assign Gringorten plotting positions and fit. Row 1 just duplicates row 0
# so the grid has the minimum two rows `NetCDFRaster` needs to infer a
# `lat` step; the tests below don't look at it.
_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
_COL0_MINIMA = {year: 0.20 for year in _YEARS}
_COL1_MINIMA = {
    2018: 0.30,
    2019: 0.28,
    2020: 0.05,  # the severe drought year
    2021: 0.32,
    2022: 0.29,
    2023: 0.31,
}
_DEKADS = 3


def _write_year(path: Path, year: int, *, missing: bool = False) -> Path:
    # Within a year, dekad 0 holds the annual minimum for each column;
    # later dekads are higher, so the per-year minimum is unambiguous.
    data = np.empty((_DEKADS, 2, 2), dtype=np.float32)
    for dekad in range(_DEKADS):
        data[dekad, :, 0] = _COL0_MINIMA[year] + dekad * 0.1
        data[dekad, :, 1] = _COL1_MINIMA[year] + dekad * 0.1
    if missing:
        data[:, :, 1] = 1e20
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"time": _DEKADS, "lat": 2, "lon": 2}
        f.create_variable("lat", ("lat",), dtype=np.float64, data=_LAT)
        f.create_variable("lon", ("lon",), dtype=np.float64, data=_LON)
        variable = f.create_variable(
            "sminx", ("time", "lat", "lon"), dtype=np.float32, data=data
        )
        variable.attrs["_FillValue"] = np.float32(1e20)
    return path


def _open_years(
    tmp_path: Path, years: tuple[int, ...] = _YEARS
) -> dict[int, NetCDFRaster]:
    rasters = {}
    for year in years:
        path = _write_year(tmp_path / f"sminx_{year}.nc", year)
        rasters[year] = NetCDFRaster.open(
            path, variable="sminx", work_dir=tmp_path / "work"
        )
    return rasters


def _metadata(years: tuple[int, ...] = _YEARS) -> RasterMetadata:
    return RasterMetadata(
        hazard_type="Drought",
        indicator_id="soil_moisture_index",
        scenario="historical",
        year=max(years),
        units="index",
        path=f"test/jrc-edo/{min(years)}-{max(years)}",
    )


def _policy(**overrides: object) -> EDOIngestPolicy:
    defaults = dict(
        h3_resolution=9,
        family="gumbel_r",
        tail="lower",
        producer="tests",
        creation_version="1",
        on_fit_failure="skip",
    )
    defaults.update(overrides)
    return EDOIngestPolicy(**defaults)  # type: ignore[arg-type]


def test_grid_mismatch_raises(tmp_path: Path) -> None:
    rasters = _open_years(tmp_path, years=(2018, 2019))
    shifted_path = _write_year(tmp_path / "shifted.nc", 2020)
    # Reopen the shifted file's raster with a different lon grid to force a
    # mismatch -- simplest way is to hand-craft one with a shifted lon axis.
    with h5netcdf.File(shifted_path, "r+") as f:
        f.variables["lon"][:] = _LON + 1.0
    rasters[2020] = NetCDFRaster.open(
        shifted_path, variable="sminx", work_dir=tmp_path / "work"
    )

    with pytest.raises(ValueError, match="grid does not match"):
        EDOAnnualMinimaCurveSource(
            rasters=rasters, metadata=_metadata((2018, 2019, 2020))
        )


def test_annual_minima_curve_fits_lower_tail(tmp_path: Path) -> None:
    rasters = _open_years(tmp_path)
    with EDOAnnualMinimaCurveSource(rasters=rasters, metadata=_metadata()) as source:
        table = canonicalize_edo_drought(source, _policy()).read_all()

    assert table.num_rows > 0
    assert set(table["curve_kind"].to_pylist()) == {"fitted"}
    assert set(table["hazard_name"].to_pylist()) == {"Drought"}
    assert set(table["pathway"].to_pylist()) == {"historical"}
    # Column 0's flat pixels (both rows) have zero variance across years
    # and should not survive the fit; only column 1's two varying pixels
    # (row 0 and its row-1 duplicate) should remain.
    assert len(set(table["source_id"].to_pylist())) == 2


def test_lower_tail_reconstructs_more_severe_value_at_higher_return_period(
    tmp_path: Path,
) -> None:
    from crc_sdk.workflows import curve_quantiles_at, return_periods_to_probabilities

    rasters = _open_years(tmp_path)
    with EDOAnnualMinimaCurveSource(rasters=rasters, metadata=_metadata()) as source:
        table = canonicalize_edo_drought(source, _policy()).read_all()

    common, rare = return_periods_to_probabilities([2, 50], tail="lower")
    common_values = curve_quantiles_at(table, common)
    rare_values = curve_quantiles_at(table, rare)

    # A rarer drought (higher return period) means a lower (more severe) SMI.
    assert rare_values[0] < common_values[0]


def test_on_fit_failure_raise_aborts_on_constant_pixel(tmp_path: Path) -> None:
    rasters = _open_years(tmp_path)
    with EDOAnnualMinimaCurveSource(rasters=rasters, metadata=_metadata()) as source:
        with pytest.raises(ValueError, match="failed to fit source pixel"):
            canonicalize_edo_drought(source, _policy(on_fit_failure="raise")).read_all()


def test_missing_year_is_dropped_not_zero_filled(tmp_path: Path) -> None:
    years = (2018, 2019, 2020, 2021, 2022)
    rasters = {}
    for year in years:
        path = _write_year(tmp_path / f"sminx_{year}.nc", year, missing=(year == 2020))
        rasters[year] = NetCDFRaster.open(
            path, variable="sminx", work_dir=tmp_path / "work"
        )

    with EDOAnnualMinimaCurveSource(
        rasters=rasters, metadata=_metadata(years)
    ) as source:
        curves = list(source.iter_curves())

    varying_pixels = [curve for curve in curves if curve.column == 1]
    assert len(varying_pixels) == 2  # row 0 and its row-1 duplicate
    # 5 years requested, one dropped as missing -> exactly 4 valid knots.
    assert all(curve.axis_values.size == 4 for curve in varying_pixels)


def test_edo_ingest_policy_is_curve_fit_ingest_policy() -> None:
    from crc_sdk.connectors import CurveFitIngestPolicy

    assert EDOIngestPolicy is CurveFitIngestPolicy
