"""NEX annual files reuse CMIP6 reductions without discarding provenance."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crc_sdk.providers.cmip6 import RestrictiveLicenceWarning
from crc_sdk.workflows.nex import NEXYearPlan
from crc_sdk.workflows.portfolio import HazardDataset

h5netcdf = pytest.importorskip("h5netcdf")


def _plan(**kwargs: Any) -> NEXYearPlan:
    return (
        HazardDataset.nex_gddp_cmip6(
            "rx1day", model="ACCESS-CM2", scenario="historical", **kwargs
        )
        .for_area((-79.5, 43.5, -79.25, 43.75))
        .years(2014)
    )


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "year.nc"
    with h5netcdf.File(path, "w") as file:
        file.dimensions = {"time": 365, "lat": 2, "lon": 2}
        file.attrs.update(
            cmip6_source_id="ACCESS-CM2",
            scenario="historical",
            variant_label="r1i1p1f1",
            version="1.0",
            cmip6_license="CC-BY-SA 4.0",
            cmip6_institution_id="CSIRO",
        )
        time = file.create_variable(
            "time", ("time",), dtype="f8", data=np.arange(365) + 0.5
        )
        time.attrs.update(units="days since 2014-01-01", calendar="proleptic_gregorian")
        file.create_variable("lat", ("lat",), dtype="f8", data=[43.5, 43.75])
        file.create_variable("lon", ("lon",), dtype="f8", data=[280.5, 280.75])
        variable = file.create_variable("pr", ("time", "lat", "lon"), dtype="f4")
        variable[:] = np.full((365, 2, 2), 1 / 86400, dtype="f4")
        variable[3, :, :] = 2 / 86400
        variable.attrs["units"] = "kg m-2 s-1"
    monkeypatch.setattr(
        "crc_sdk.workflows.nex._open_backend", lambda *args: h5netcdf.File(path, "r")
    )
    return path


def test_streamed_values_and_provenance(source: Path) -> None:
    with pytest.warns(RestrictiveLicenceWarning, match="archive is now CC0"):
        batches = list(_plan().annual_extremes(strip_bytes=1000))
    rows = [row for batch in batches for row in batch.to_pylist()]
    assert len(rows) == 4
    assert all(row["value"] == pytest.approx(2) for row in rows)
    assert rows[0]["archive_licence"] == "CC0-1.0"
    assert rows[0]["original_model_licence"] == "CC-BY-SA 4.0"
    assert rows[0]["source_url"].endswith("gn_2014.nc")
    assert len(batches) == 2


def test_strict_licence_and_lazy_access(source: Path) -> None:
    batches = _plan(strict_licence=True).annual_extremes()
    with pytest.raises(ValueError, match="original model licence"):
        next(batches)


def test_source_validation(source: Path) -> None:
    with h5netcdf.File(source, "a") as file:
        file.attrs["version"] = "1.2"
    with pytest.raises(ValueError, match="version"):
        list(_plan().annual_extremes())
    with pytest.raises(ValueError, match="years"):
        HazardDataset.nex_gddp_cmip6("txx", model="M", scenario="ssp585").for_area(
            (0, 0, 1, 1)
        ).years(2014)


def test_grid_label_is_case_insensitive() -> None:
    plan = HazardDataset.nex_gddp_cmip6(
        "rx1day", model="ACCESS-CM2", scenario="historical", grid_label="GN"
    )
    assert plan.grid_label == "gn"
    assert plan._url(2014).endswith("_gn_2014.nc")


@pytest.fixture
def temperature_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A NEX-like file whose kelvin unit is a one-character fixed-length string.

    h5netcdf decodes fixed-length string attributes only when they are longer
    than one character, so the real archive's `units = "K"` arrives as bytes.
    """
    path = tmp_path / "temperature.nc"
    with h5netcdf.File(path, "w") as file:
        file.dimensions = {"time": 365, "lat": 2, "lon": 2}
        file.attrs.update(
            cmip6_source_id="ACCESS-CM2",
            scenario="historical",
            variant_label="r1i1p1f1",
            version="1.0",
            cmip6_license="CC0-1.0",
            cmip6_institution_id="CSIRO",
        )
        time = file.create_variable(
            "time", ("time",), dtype="f8", data=np.arange(365) + 0.5
        )
        time.attrs.update(units="days since 2014-01-01", calendar="proleptic_gregorian")
        file.create_variable("lat", ("lat",), dtype="f8", data=[43.5, 43.75])
        file.create_variable("lon", ("lon",), dtype="f8", data=[280.5, 280.75])
        for name, usual, extreme in (
            ("tasmax", 290.0, 300.0),
            ("tasmin", 280.0, 270.0),
        ):
            variable = file.create_variable(name, ("time", "lat", "lon"), dtype="f4")
            variable[:] = np.full((365, 2, 2), usual, dtype="f4")
            variable[3, :, :] = extreme
            variable.attrs["units"] = np.bytes_("K")
    monkeypatch.setattr(
        "crc_sdk.workflows.nex._open_backend", lambda *args: h5netcdf.File(path, "r")
    )
    return path


def _temperature_rows(recipe: str) -> list[dict[str, Any]]:
    plan = (
        HazardDataset.nex_gddp_cmip6(recipe, model="ACCESS-CM2", scenario="historical")
        .for_area((-79.5, 43.5, -79.25, 43.75))
        .years(2014)
    )
    return [row for batch in plan.annual_extremes() for row in batch.to_pylist()]


@pytest.mark.parametrize(("recipe", "expected"), [("txx", 26.85), ("tnn", -3.15)])
def test_kelvin_unit_stored_as_single_byte(
    temperature_source: Path, recipe: str, expected: float
) -> None:
    with h5netcdf.File(temperature_source, "r") as file:
        assert isinstance(file.variables["tasmax"].attrs["units"], bytes)
    rows = _temperature_rows(recipe)
    assert len(rows) == 4
    assert all(row["value"] == pytest.approx(expected, abs=1e-3) for row in rows)
    assert {row["unit"] for row in rows} == {"degC"}


def test_wrong_single_byte_unit_is_still_rejected(temperature_source: Path) -> None:
    with h5netcdf.File(temperature_source, "a") as file:
        file.variables["tasmax"].attrs["units"] = np.bytes_("C")
    with pytest.raises(ValueError, match="is published in 'C'"):
        _temperature_rows("txx")


def test_missing_units_are_rejected(temperature_source: Path) -> None:
    with h5netcdf.File(temperature_source, "a") as file:
        del file.variables["tasmax"].attrs["units"]
    with pytest.raises(ValueError, match="source units are missing"):
        _temperature_rows("txx")
