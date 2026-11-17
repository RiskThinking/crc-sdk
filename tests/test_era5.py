from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("h5netcdf")

from crc_sdk._providers.era5 import (  # noqa: E402
    ERA5_STORES,
    ERA5Provider,
    ERA5Store,
    era5_recipe,
)

YEARS = tuple(range(2000, 2008))
LAT = np.array([-1.5, 0.0, 1.5])
LON = np.arange(0.0, 360.0, 45.0)  # 0..315, so a window around 0 wraps the seam
EPOCH = np.datetime64("1999-01-01", "h")


def _hours() -> np.ndarray[Any, Any]:
    start = np.datetime64("2000-01-01T00", "h")
    stop = np.datetime64("2008-01-01T00", "h")
    return np.arange(start, stop, dtype="datetime64[h]")


def _build_store(path: Path) -> ERA5Store:
    """A tiny ERA5-shaped store, dims (time, lon, lat) like the WB2 copies."""
    hours = _hours()
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    time = group.create_array(
        "time", data=(hours - EPOCH).astype(np.int64), chunks=(len(hours),)
    )
    time.attrs.update(
        {"units": "hours since 1999-01-01", "_ARRAY_DIMENSIONS": ["time"]}
    )
    for name, values in (("latitude", LAT), ("longitude", LON)):
        coordinate = group.create_array(name, data=values)
        coordinate.attrs["_ARRAY_DIMENSIONS"] = [name]

    year = hours.astype("datetime64[Y]").astype(int) + 1970
    hour_of_day = (
        hours - hours.astype("datetime64[D]").astype("datetime64[h]")
    ).astype(int)

    # Temperature: 280 K, with a hot noon each July 1 whose size grows with
    # the year, and a bigger one in the lon=0 column only.
    t2m = np.full((len(hours), len(LON), len(LAT)), 280.0, dtype=np.float32)
    july = (
        hours.astype("datetime64[D]")
        == (
            hours.astype("datetime64[Y]").astype("datetime64[D]")
            + np.timedelta64(181, "D")
        )
    ) & (hour_of_day == 12)
    t2m[july] += ((year[july] - 2000) + 5.0)[:, None, None]
    t2m[july, 0, :] += 3.0
    # A varying winter minimum also supplies genuine samples for TNn fitting.
    january = hours == hours.astype("datetime64[Y]").astype("datetime64[h]")
    t2m[january] -= ((year[january] - 2000) + 10.0)[:, None, None]

    # Precipitation: 1 mm stamped 12:00 and 10 mm stamped 00:00 the *next* day
    # (the hour ending midnight) -> Rx1day must see 11 mm on Jan 1 of 2001.
    tp = np.zeros((len(hours), len(LON), len(LAT)), dtype=np.float32)
    for sample_year in YEARS:
        jan1 = np.datetime64(f"{sample_year}-01-01T00", "h")
        tp[hours == jan1 + np.timedelta64(12, "h")] = 0.001
        tp[hours == jan1 + np.timedelta64(24, "h")] = (9 + sample_year - 2000) / 1000
    for name, data in (("2m_temperature", t2m), ("total_precipitation", tp)):
        array = group.create_array(
            name, data=data, chunks=(24 * 30, len(LON), len(LAT))
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["time", "longitude", "latitude"]
    mask = np.zeros((len(LON), len(LAT)), dtype=np.float32)
    mask[:, 1] = 1.0  # land only along lat=0
    land = group.create_array("land_sea_mask", data=mask)
    land.attrs["_ARRAY_DIMENSIONS"] = ["longitude", "latitude"]
    return ERA5Store(
        name="test",
        url=str(path),
        resolution_degrees=1.5,
        variables={"t2m": "2m_temperature", "tp": "total_precipitation"},
        storage_options={},
        first_year=2000,
        last_final_year=2007,
        description="synthetic",
    )


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ERA5Store:
    built = _build_store(tmp_path / "era5.zarr")
    monkeypatch.setitem(ERA5_STORES, "test", built)
    return built


def test_coverage_and_year_checks(store: ERA5Store) -> None:
    provider = ERA5Provider(store)
    assert provider.resolve_version("latest") == "final-through-2007-12-31"
    with pytest.raises(ValueError, match="not served"):
        provider.resolve_version("final-through-1999-12-31")
    provider.check_years([2000, 2007])
    with pytest.raises(ValueError, match="outside the final"):
        provider.check_years([1999, 2008])


def test_txx_reads_across_the_seam_with_land_mask(
    store: ERA5Store, tmp_path: Path
) -> None:
    provider = ERA5Provider(store)
    out = provider.cache_annual_extreme(
        era5_recipe("txx"),
        2003,
        (-50.0, -2.0, 50.0, 2.0),
        tmp_path / "txx-2003.nc",
        land_only=True,
    )
    import h5netcdf  # type: ignore[import-untyped]

    with h5netcdf.File(out) as dataset:
        lon = np.asarray(dataset["lon"][:])
        lat = np.asarray(dataset["lat"][:])
        values = np.asarray(dataset["extreme"][0])
    assert lon.tolist() == [-45.0, 0.0, 45.0]  # normalised, ascending
    assert lat.tolist() == [-1.5, 0.0, 1.5]
    # 280 K = 6.85 C; July noon adds (2003-2000)+5 = 8 K; lon 0 adds 3 K more.
    assert values[1, 0] == pytest.approx(6.85 + 8.0, abs=1e-4)
    assert values[1, 1] == pytest.approx(6.85 + 8.0 + 3.0, abs=1e-4)
    assert np.isnan(values[0]).all() and np.isnan(values[2]).all()  # sea masked


def _time_varying_mask(store: ERA5Store, *, finite_from: int | None) -> None:
    """Replace the mask with an ARCO-style (time, lon, lat) mask that is NaN
    before ``finite_from`` (a time index), and declare that start."""
    group = zarr.open_group(store.url, mode="a", zarr_format=2)
    hours = _hours()
    mask = np.full((len(hours), len(LON), len(LAT)), np.nan, dtype=np.float32)
    if finite_from is not None:
        mask[finite_from:, :, 1] = 1.0
        mask[finite_from:, :, 0] = 0.0
        mask[finite_from:, :, 2] = 0.0
        group.attrs["valid_time_start"] = str(hours[finite_from]) + ":00:00"
    del group["land_sea_mask"]
    land = group.create_array(
        "land_sea_mask", data=mask, chunks=(24 * 30, len(LON), len(LAT))
    )
    land.attrs["_ARRAY_DIMENSIONS"] = ["time", "longitude", "latitude"]


def test_time_varying_land_mask_is_read_at_the_record_start(
    store: ERA5Store,
) -> None:
    # ARCO's mask has a time axis that is NaN before the record starts; index 0
    # (1900 there) masked every cell. Read it at valid_time_start instead.
    _time_varying_mask(store, finite_from=48)
    reader = ERA5Provider(store).reader(
        era5_recipe("txx"), (-50.0, -2.0, 50.0, 2.0), [2003], land_only=True
    )
    assert reader._mask is not None
    assert reader._mask.tolist() == [[False] * 3, [True] * 3, [False] * 3]


def test_land_mask_without_finite_values_raises(store: ERA5Store) -> None:
    _time_varying_mask(store, finite_from=None)
    with pytest.raises(ValueError, match="no finite values"):
        ERA5Provider(store).reader(
            era5_recipe("txx"), (-50.0, -2.0, 50.0, 2.0), [2003], land_only=True
        )


def test_rx1day_assigns_interval_end_stamps_to_the_ended_day(
    store: ERA5Store, tmp_path: Path
) -> None:
    provider = ERA5Provider(store)
    out = provider.cache_annual_extreme(
        era5_recipe("rx1day"), 2001, (-50.0, -2.0, 50.0, 2.0), tmp_path / "rx.nc"
    )
    import h5netcdf

    with h5netcdf.File(out) as dataset:
        values = np.asarray(dataset["extreme"][0])
    assert np.allclose(values, 11.0)  # 1 mm + the 10 mm hour ending midnight
    rx5 = provider.cache_annual_extreme(
        era5_recipe("rx5day"), 2001, (-50.0, -2.0, 50.0, 2.0), tmp_path / "rx5.nc"
    )
    with h5netcdf.File(rx5) as dataset:
        assert np.allclose(np.asarray(dataset["extreme"][0]), 11.0)


def test_single_column_at_the_dateline_pads_locally(store: ERA5Store) -> None:
    bounds = (179.0, -1.0, 180.0, 1.0)
    reader = ERA5Provider(store).reader(era5_recipe("txx"), bounds, [2000])
    assert abs(float(reader.lon[1] - reader.lon[0])) == 45.0


def test_window_ending_at_the_dateline_keeps_the_180_meridian(
    store: ERA5Store, tmp_path: Path
) -> None:
    provider = ERA5Provider(store)
    east = provider.reader(era5_recipe("txx"), (120.0, -2.0, 180.0, 2.0), [2003])
    assert east.lon.tolist() == [135.0, 180.0]  # 180 stays at +180, not -180
    west = provider.reader(era5_recipe("txx"), (-180.0, -2.0, -120.0, 2.0), [2003])
    assert west.lon.tolist() == [-180.0, -135.0]


def test_accumulations_need_the_next_midnight_to_close_the_last_year(
    store: ERA5Store, tmp_path: Path
) -> None:
    # The record ends 2007-12-31: the hour stamped 2008-01-01 00:00 is missing,
    # so 31 December 2007 is incomplete for precipitation but not temperature.
    provider = ERA5Provider(store)
    provider.check_years([2007], era5_recipe("txx"))
    with pytest.raises(ValueError, match="outside the final"):
        provider.check_years([2007], era5_recipe("rx1day"))
    with pytest.raises(ValueError, match="outside the final"):
        provider.cache_annual_extreme(
            era5_recipe("rx5day"), 2007, (-50.0, -2.0, 50.0, 2.0), tmp_path / "x.nc"
        )
    provider.check_years([2006], era5_recipe("rx1day"))
