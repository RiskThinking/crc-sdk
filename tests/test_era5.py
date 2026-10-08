from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("h5netcdf")

from crc_sdk.connectors import CurveFitIngestPolicy, read_hazard_metadata  # noqa: E402
from crc_sdk.providers.era5 import (  # noqa: E402
    ERA5_RECIPES,
    ERA5_STORES,
    ERA5Provider,
    ERA5Store,
    era5_recipe,
    era5_store,
)
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset  # noqa: E402

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


def test_catalogue() -> None:
    assert set(ERA5_RECIPES) == {"txx", "tnn", "rx1day", "rx5day"}
    assert era5_recipe("TXx").tail == "upper"
    assert era5_recipe("tnn").tail == "lower"
    assert era5_store("arco-0p25").resolution_degrees == 0.25
    with pytest.raises(ValueError, match="unknown ERA5 recipe"):
        era5_recipe("fwi")
    with pytest.raises(ValueError, match="unknown ERA5 store"):
        HazardDataset.era5("txx", store="nope")


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


def _plan(tmp_path: Path, **policy: Any) -> Any:
    return (
        HazardDataset.era5("txx", store="test")
        .for_area((-50.0, -2.0, 50.0, 2.0), land_only=True)
        .years(2000, 2007)
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, **policy))
    )


def test_plan_materializes_with_provenance_and_reuses_cache(
    store: ERA5Store, tmp_path: Path
) -> None:
    plan = _plan(tmp_path, h3_resolution=3, diagnostics=tmp_path / "diag.parquet")
    text = plan.explain()
    assert "0 cached, 8 to fetch" in text
    assert not (tmp_path / "cache").exists()  # explain does no I/O

    prefetched = plan.prefetch()
    assert (prefetched.cache_hits, prefetched.cache_misses) == (0, 8)
    again = plan.prefetch()
    assert (again.cache_hits, again.cache_misses) == (8, 0)

    dataset = plan.materialize(tmp_path / "txx.parquet")
    result = dataset.materialization
    assert result is not None and result.source_cache_hits == 8
    metadata = read_hazard_metadata(tmp_path / "txx.parquet")
    assert metadata.source.licence == "CC-BY-4.0"
    assert metadata.source.version == "final-through-2007-12-31"
    assert metadata.temporal_window is not None
    assert (metadata.temporal_window.start_year, metadata.temporal_window.end_year) == (
        2000,
        2007,
    )
    assert (
        metadata.ensemble is not None and metadata.ensemble.pooling == "single_member"
    )
    table = pq.read_table(tmp_path / "txx.parquet")
    assert set(table.column("pathway").to_pylist()) == {"historical"}
    assert set(table.column("horizon").to_pylist()) == {2003}
    assert (tmp_path / "diag.parquet").is_file()

    extremes = plan.annual_extremes()
    assert extremes.num_rows == 8 * 3  # land cells only: 3 longitudes, lat 0
    assert set(extremes.column("year").to_pylist()) == set(YEARS)

    offline = plan.cache(tmp_path / "cache", mode="offline").prefetch()
    assert offline.cache_hits == 8


def test_annual_extremes_exclude_padding_cells(
    store: ERA5Store, tmp_path: Path
) -> None:
    plan = (
        HazardDataset.era5("txx", store="test")
        .for_area((-0.1, -0.1, 0.1, 0.1))
        .years(2000, 2007)
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, h3_resolution=3))
    )
    extremes = plan.annual_extremes()
    assert set(extremes.column("latitude").to_pylist()) == {0.0}
    assert set(extremes.column("longitude").to_pylist()) == {0.0}


def test_single_column_at_the_dateline_pads_locally(store: ERA5Store) -> None:
    bounds = (179.0, -1.0, 180.0, 1.0)
    reader = ERA5Provider(store).reader(era5_recipe("txx"), bounds, [2000])
    assert abs(float(reader.lon[1] - reader.lon[0])) == 45.0


def test_offline_without_cache_fails_clearly(store: ERA5Store, tmp_path: Path) -> None:
    plan = _plan(tmp_path).cache(tmp_path / "empty", mode="offline")
    with pytest.raises(FileNotFoundError, match="prefetch"):
        plan.prefetch()


def test_stream_mode_materializes_without_a_cache(
    store: ERA5Store, tmp_path: Path
) -> None:
    plan = (
        HazardDataset.era5("txx", store="test")
        .for_area((-50.0, -2.0, 50.0, 2.0))
        .years(2000, 2007)
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, h3_resolution=3))
    )
    with pytest.raises(ValueError, match="prefetch"):
        plan.prefetch()
    dataset = plan.materialize(tmp_path / "out.parquet")
    assert dataset.materialization is not None
    assert dataset.materialization.canonical_rows > 0


def test_minimum_years_and_years_outside_record(
    store: ERA5Store, tmp_path: Path
) -> None:
    short = (
        HazardDataset.era5("txx", store="test")
        .for_area((-50.0, -2.0, 50.0, 2.0))
        .years(2000, 2003)
        .cache(tmp_path / "c")
        .canonicalize()
    )
    with pytest.raises(ValueError, match="at least 20"):
        short.prefetch()
    outside = _plan(tmp_path).years.area.years(2000, 2010).cache(tmp_path / "c")
    with pytest.raises(ValueError, match="outside the final"):
        outside.canonicalize(policy=BlockExtremaPolicy(minimum_years=4)).prefetch()


def test_rx5day_plan_fits_from_the_cached_annual_extremes(
    store: ERA5Store, tmp_path: Path
) -> None:
    # A rolling 5-day recipe must reopen its one-step cache files for fitting.
    plan = (
        HazardDataset.era5("rx5day", store="test")
        .for_area((-50.0, -2.0, 50.0, 2.0))
        .years(2000, 2006)
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=7, h3_resolution=3))
    )
    assert plan.annual_extremes().num_rows > 0
    assert plan.materialize(tmp_path / "rx5.parquet").materialization is not None


def test_refresh_replaces_an_existing_output(store: ERA5Store, tmp_path: Path) -> None:
    plan = _plan(tmp_path, h3_resolution=3)
    first = plan.ensure_materialized()
    refreshed = plan.cache(tmp_path / "cache", mode="refresh").ensure_materialized()
    assert refreshed.provider.source == first.provider.source
    plan.materialize(tmp_path / "again.parquet")
    plan.materialize(tmp_path / "again.parquet")


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


@pytest.mark.parametrize("family", ["genextreme", "gumbel_r", "gumbel_l"])
def test_lmoments_materialization(
    store: ERA5Store, tmp_path: Path, family: str
) -> None:
    plan = _plan(tmp_path, family=family, fit_method="sample_lmoments", h3_resolution=3)
    plan.prefetch()
    dataset = plan.cache(tmp_path / "cache", mode="offline").materialize(
        tmp_path / "lmoments.parquet"
    )
    metadata = dataset.metadata()
    assert metadata.fitting is not None
    assert metadata.fitting.method == "sample_lmoments"
    assert metadata.fitting.families == (family,)
    assert metadata.fitting.initialization is None
    assert dataset.materialization is not None
    assert dataset.materialization.source_cache_misses == 0
    assert dataset.materialization.canonical_rows > 0


def test_generic_block_defaults() -> None:
    policy = BlockExtremaPolicy.curated().resolve()
    assert policy.family == "genextreme"
    assert policy.fit_method == "sample_lmoments"


@pytest.mark.parametrize("recipe", ["txx", "tnn", "rx1day", "rx5day"])
@pytest.mark.parametrize("custom_policy", [False, True])
def test_recipe_fitting_defaults(recipe: str, custom_policy: bool) -> None:
    years = HazardDataset.era5(recipe).for_area((-1, -1, 1, 1)).years(1991, 2020)
    policy = BlockExtremaPolicy.curated(h3_resolution=4) if custom_policy else "curated"
    plan = years.canonicalize(policy=policy)
    expected = ("genextreme", "sample_lmoments")
    assert (plan.policy.family, plan.policy.fit_method) == expected
    details = plan.explain(format="json")
    assert isinstance(details, dict)
    assert details["fitting"] == {
        "family": expected[0],
        "method": expected[1],
    }


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"family": "gumbel_r"}, ("gumbel_r", "sample_lmoments")),
        ({"fit_method": "sample_mle"}, ("genextreme", "sample_mle")),
        (
            {"family": "gumbel_r", "fit_method": "quantile_least_squares"},
            ("gumbel_r", "quantile_least_squares"),
        ),
    ],
)
def test_rx1day_explicit_overrides(
    overrides: dict[str, Any], expected: tuple[str, str]
) -> None:
    policy = BlockExtremaPolicy(h3_resolution=4, **overrides)
    plan = (
        HazardDataset.era5("rx1day")
        .for_area((-1, -1, 1, 1))
        .years(1991, 2020)
        .canonicalize(policy=policy)
    )
    assert (plan.policy.family, plan.policy.fit_method) == expected
    assert policy.family == overrides.get("family")  # caller's policy is immutable


def test_rx1day_explicit_ingest_policy_is_preserved() -> None:
    policy = CurveFitIngestPolicy(h3_resolution=4, family="gumbel_r", producer="test")
    plan = (
        HazardDataset.era5("rx1day")
        .for_area((-1, -1, 1, 1))
        .years(1991, 2020)
        .canonicalize(policy=policy)
    )
    assert plan.policy is policy


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"family": "genpareto"}, "supports"),
        ({"maximum_normalized_rmse": 0.1}, "quantile quality gates"),
    ],
)
def test_rx1day_invalid_resolved_policy_fails_before_fetch(
    overrides: dict[str, Any], message: str
) -> None:
    years = HazardDataset.era5("rx1day").for_area((-1, -1, 1, 1)).years(1991, 2020)
    with pytest.raises(ValueError, match=message):
        years.canonicalize(policy=BlockExtremaPolicy(**overrides))


@pytest.mark.parametrize("recipe", ["txx", "tnn", "rx1day", "rx5day"])
def test_recipe_default_materializes_lmoments_provenance(
    store: ERA5Store, tmp_path: Path, recipe: str
) -> None:
    plan = (
        HazardDataset.era5(recipe, store="test")
        .for_area((-50.0, -2.0, 50.0, 2.0), land_only=True)
        .years(2000, 2006)
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=7, h3_resolution=3))
    )
    dataset = plan.materialize(tmp_path / f"{recipe}.parquet")
    metadata = dataset.metadata()
    assert metadata.fitting is not None
    assert metadata.fitting.families == ("genextreme",)
    assert metadata.fitting.method == "sample_lmoments"
    assert metadata.fitting.input_kind == "samples"
    assert metadata.fitting.initialization is None
    assert metadata.fitting.sample_resampling is None
    assert dataset.materialization is not None
    assert dataset.materialization.canonical_rows > 0


@pytest.mark.parametrize(
    "bound", [{"minimum_return_value": 0}, {"maximum_return_value": 1000}]
)
def test_quantile_override_preserves_return_level_checks(
    bound: dict[str, float],
) -> None:
    policy = BlockExtremaPolicy(
        family="gumbel_r",
        fit_method="quantile_least_squares",
        minimum_return_value=bound.get("minimum_return_value"),
        maximum_return_value=bound.get("maximum_return_value"),
    )
    plan = (
        HazardDataset.era5("rx1day")
        .for_area((-1, -1, 1, 1))
        .years(1991, 2020)
        .canonicalize(policy=policy)
    )
    assert isinstance(plan.policy, BlockExtremaPolicy)
    ingest = plan.policy.ingest_policy(
        tail="upper", value_semantics="test", source_version="test"
    )
    assert ingest.validation_return_periods == (2, 5, 10, 20, 50, 100)
    assert not ingest.require_sample_support
    assert ingest.minimum_return_value == bound.get("minimum_return_value")
    assert ingest.maximum_return_value == bound.get("maximum_return_value")
