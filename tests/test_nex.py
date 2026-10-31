"""NEX annual files reuse CMIP6 reductions without discarding provenance."""

import numpy as np
import pytest

from crc_sdk.providers.cmip6 import RestrictiveLicenceWarning
from crc_sdk.workflows.portfolio import HazardDataset

h5netcdf = pytest.importorskip("h5netcdf")


def _plan(**kwargs):
    return (
        HazardDataset.nex_gddp_cmip6(
            "rx1day", model="ACCESS-CM2", scenario="historical", **kwargs
        )
        .for_area((-79.5, 43.5, -79.25, 43.75))
        .years(2014)
    )


@pytest.fixture
def source(tmp_path, monkeypatch):
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


def test_streamed_values_and_provenance(source):
    with pytest.warns(RestrictiveLicenceWarning, match="archive is now CC0"):
        batches = list(_plan().annual_extremes(strip_bytes=1000))
    rows = [row for batch in batches for row in batch.to_pylist()]
    assert len(rows) == 4
    assert all(row["value"] == pytest.approx(2) for row in rows)
    assert rows[0]["archive_licence"] == "CC0-1.0"
    assert rows[0]["original_model_licence"] == "CC-BY-SA 4.0"
    assert rows[0]["source_url"].endswith("gn_2014.nc")
    assert len(batches) == 2


def test_strict_licence_and_lazy_access(source):
    batches = _plan(strict_licence=True).annual_extremes()
    with pytest.raises(ValueError, match="original model licence"):
        next(batches)


def test_source_validation(source):
    with h5netcdf.File(source, "a") as file:
        file.attrs["version"] = "1.2"
    with pytest.raises(ValueError, match="version"):
        list(_plan().annual_extremes())
    with pytest.raises(ValueError, match="years"):
        HazardDataset.nex_gddp_cmip6("txx", model="M", scenario="ssp585").for_area(
            (0, 0, 1, 1)
        ).years(2014)
