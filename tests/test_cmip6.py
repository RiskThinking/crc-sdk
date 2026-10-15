from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("h5netcdf")

from crc_sdk.connectors import read_hazard_metadata  # noqa: E402
from crc_sdk.connectors.blocks import BlockDefinition  # noqa: E402
from crc_sdk.providers.cmip6 import (  # noqa: E402
    CATALOGUE_COLUMNS,
    CMIP6_ARCHIVES,
    CMIP6_RECIPES,
    CFCalendar,
    CMIP6Archive,
    CMIP6Catalogue,
    CMIP6Dataset,
    CMIP6Provider,
    RestrictiveLicenceWarning,
    cmip6_archive,
    cmip6_recipe,
    decode_time_ordinals,
    normalize_licence,
)
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset  # noqa: E402

YEARS = tuple(range(2015, 2027))
LAT = np.array([-1.5, 0.0, 1.5])
LON = np.arange(0.0, 360.0, 45.0)  # 0..315, so a window around 0 wraps the seam
AREA = (-50.0, -2.0, 50.0, 2.0)
#: The day of each model year carrying that year's rainfall peak.
WET_DAY = 100
MODEL, MEMBER = "GFDL-ESM4", "r1i1p1f1"


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "name", "year_days", "leap_days", "window"),
    [
        ("standard", "proleptic_gregorian", 365, 366, "gregorian"),
        ("gregorian", "proleptic_gregorian", 365, 366, "gregorian"),
        ("noleap", "noleap", 365, 365, "noleap"),
        ("365_day", "noleap", 365, 365, "noleap"),
        ("360_day", "360_day", 360, 360, "360_day"),
        ("all_leap", "all_leap", 366, 366, "unspecified"),
    ],
)
def test_calendars_count_their_own_years(
    label: str, name: str, year_days: int, leap_days: int, window: str
) -> None:
    calendar = CFCalendar.parse(label)
    assert calendar.name == name
    assert calendar.window_calendar == window
    annual = BlockDefinition.annual()
    start, stop = calendar.block_ordinals(annual, 2015)
    leap_start, leap_stop = calendar.block_ordinals(annual, 2016)
    assert (stop - start, leap_stop - leap_start) == (year_days, leap_days)


@pytest.mark.parametrize("label", ["standard", "noleap", "360_day", "all_leap"])
def test_calendar_ordinals_round_trip(label: str) -> None:
    calendar = CFCalendar.parse(label)
    dates = [(2015, 1, 1), (2015, 2, 28), (2049, 7, 15), (2100, 12, 30)]
    ordinals = np.array([calendar.to_ordinal(*date) for date in dates])
    years, months, days = calendar.from_ordinal(ordinals)
    assert [tuple(int(v) for v in row) for row in zip(years, months, days)] == dates


def test_calendar_rejects_what_it_cannot_count() -> None:
    with pytest.raises(ValueError, match="unsupported CMIP6 calendar"):
        CFCalendar.parse("julian")


def test_decoding_floors_day_centred_stamps_in_the_model_calendar() -> None:
    # A 360-day model's day 360 is 1 January, five days before the real one.
    calendar = CFCalendar.parse("360_day")
    ordinals = decode_time_ordinals(
        np.arange(0.5, 720.0, 1.0), "days since 2015-01-01", calendar
    )
    years, months, days = calendar.from_ordinal(ordinals)
    assert (int(years[0]), int(months[0]), int(days[0])) == (2015, 1, 1)
    assert (int(years[360]), int(months[360]), int(days[360])) == (2016, 1, 1)
    assert int((months[:360] == 2).sum()) == 30  # February has thirty days


def test_decoding_accepts_sub_daily_units() -> None:
    calendar = CFCalendar.parse("noleap")
    ordinals = decode_time_ordinals(
        np.array([12.0, 36.0]), "hours since 2015-01-01 00:00:00", calendar
    )
    _, _, days = calendar.from_ordinal(ordinals)
    assert days.tolist() == [1, 2]
    with pytest.raises(ValueError, match="unsupported CMIP6 time units"):
        decode_time_ordinals(np.array([0.0]), "fortnights since 2015", calendar)


# ---------------------------------------------------------------------------
# A synthetic published dataset
# ---------------------------------------------------------------------------


def _build_zarr(path: Path, *, precip_units: str = "kg m-2 s-1") -> Path:
    """A tiny CMIP6-shaped store: noleap daily `pr` and `tasmax`."""
    steps = 365 * len(YEARS)
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.attrs.update(
        {"license": "CC-BY-4.0", "institution": "NOAA GFDL", "source_id": MODEL}
    )
    time = group.create_array("time", data=np.arange(0.5, steps, 1.0), chunks=(steps,))
    time.attrs.update(
        {
            "units": "days since 2015-01-01",
            "calendar": "noleap",
            "_ARRAY_DIMENSIONS": ["time"],
        }
    )
    for name, values in (("lat", LAT), ("lon", LON)):
        coordinate = group.create_array(name, data=values)
        coordinate.attrs["_ARRAY_DIMENSIONS"] = [name]

    # Rainfall: dry, apart from one wet day per year whose depth grows with the
    # year and the row, plus a flat five-day spell for the rolling recipe.
    precip = np.zeros((steps, LAT.size, LON.size), dtype=np.float32)
    temperature = np.full((steps, LAT.size, LON.size), 280.0, dtype=np.float32)
    for index in range(len(YEARS)):
        offset = index * 365
        depth = 20.0 + index + np.arange(LAT.size, dtype=np.float32)
        precip[offset + WET_DAY] = depth[:, None] / 86400.0
        precip[offset + 200 : offset + 205] = 3.0 / 86400.0
        temperature[offset + 180] = 280.0 + index

    for name, data, units in (
        ("pr", precip, precip_units),
        ("tasmax", temperature, "K"),
    ):
        array = group.create_array(name, data=data, chunks=(365, LAT.size, LON.size))
        array.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
        array.attrs["units"] = units
    return path


def _write_catalogue(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CATALOGUE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    zstore: Path,
    *,
    scenario: str = "ssp585",
    variable: str = "pr",
    grid_label: str = "gn",
    version: str = "20180701",
    member: str = MEMBER,
) -> dict[str, str]:
    return {
        "activity_id": "CMIP" if scenario == "historical" else "ScenarioMIP",
        "institution_id": "NOAA-GFDL",
        "source_id": MODEL,
        "experiment_id": scenario,
        "member_id": member,
        "table_id": "day",
        "variable_id": variable,
        "grid_label": grid_label,
        "zstore": str(zstore),
        "version": version,
    }


@pytest.fixture()
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CMIP6Archive:
    store = _build_zarr(tmp_path / "gfdl.zarr")
    catalogue = _write_catalogue(
        tmp_path / "pangeo-cmip6.csv",
        [
            _row(store),
            _row(store, variable="tasmax"),
            _row(store, scenario="historical", version="20190726"),
            # An older release and a regridded copy of the same dataset: the
            # resolver must prefer the native grid and the newest version.
            _row(store, version="20170101"),
            _row(store, grid_label="gr"),
            # A second member, which must not be picked up for r1i1p1f1.
            _row(store, member="r2i1p1f1"),
            # One pathway published twice under the same grid and version --
            # the catalogue really does contain such pairs.
            _row(store, scenario="ssp245"),
            _row(tmp_path / "duplicate.zarr", scenario="ssp245"),
        ],
    )
    built = CMIP6Archive(
        name="test",
        catalogue_url=str(catalogue),
        storage_options={},
        description="synthetic",
    )
    monkeypatch.setitem(CMIP6_ARCHIVES, "test", built)
    return built


# ---------------------------------------------------------------------------
# Recipes and the catalogue
# ---------------------------------------------------------------------------


def test_recipe_catalogue() -> None:
    assert set(CMIP6_RECIPES) == {"txx", "tnn", "rx1day", "rx5day"}
    assert cmip6_recipe("Rx1day").variable == "pr"
    assert cmip6_recipe("tnn").tail == "lower"
    assert cmip6_recipe("txx").tail == "upper"
    with pytest.raises(ValueError, match="unknown CMIP6 recipe"):
        cmip6_recipe("fwi")
    with pytest.raises(ValueError, match="unknown CMIP6 archive"):
        cmip6_archive("nope")
    with pytest.raises(ValueError, match="unknown CMIP6 recipe"):
        HazardDataset.cmip6("fwi", model=MODEL, scenario="ssp585")


def test_catalogue_resolves_one_dataset(archive: CMIP6Archive) -> None:
    catalogue = CMIP6Catalogue(archive)
    dataset = catalogue.resolve(
        model=MODEL, scenario="ssp585", member=MEMBER, variable="pr"
    )
    assert (dataset.grid_label, dataset.version) == ("gn", "20180701")
    assert dataset.dataset_id == (
        "CMIP6.ScenarioMIP.NOAA-GFDL.GFDL-ESM4.ssp585.r1i1p1f1.day.pr.gn.v20180701"
    )
    historical = catalogue.resolve(
        model=MODEL, scenario="historical", member=MEMBER, variable="pr"
    )
    assert historical.activity_id == "CMIP"
    assert len(catalogue.search(model=MODEL, variable="pr")) == 7


def test_catalogue_reports_what_it_cannot_resolve(archive: CMIP6Archive) -> None:
    catalogue = CMIP6Catalogue(archive)
    with pytest.raises(LookupError, match="no CMIP6 day pr published"):
        catalogue.resolve(model=MODEL, scenario="ssp126", member=MEMBER, variable="pr")
    with pytest.raises(LookupError, match="ambiguous"):
        # Grid preference and "latest" cannot separate two publications of the
        # same grid and version, so the caller is told rather than guessed at.
        catalogue.resolve(model=MODEL, scenario="ssp245", member=MEMBER, variable="pr")


def test_catalogue_caches_the_daily_subset(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    local = tmp_path / "cache" / "catalogue.parquet"
    catalogue = CMIP6Catalogue(archive)
    catalogue.ensure_local(local)
    assert local.is_file() and catalogue.source == str(local)
    # The cached subset answers the same question without the CSV.
    (tmp_path / "pangeo-cmip6.csv").unlink()
    offline = CMIP6Catalogue(archive, local=local)
    assert (
        offline.resolve(
            model=MODEL, scenario="ssp585", member=MEMBER, variable="pr"
        ).version
        == "20180701"
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _provider(archive: CMIP6Archive, **selection: str) -> CMIP6Provider:
    dataset = CMIP6Catalogue(archive).resolve(
        model=MODEL,
        scenario=selection.get("scenario", "ssp585"),
        member=MEMBER,
        variable=selection.get("variable", "pr"),
    )
    return CMIP6Provider(dataset, archive=archive)


def test_coverage_and_year_checks(archive: CMIP6Archive) -> None:
    provider = _provider(archive)
    coverage = provider.coverage()
    assert (coverage.first_year, coverage.last_year) == (YEARS[0], YEARS[-1])
    assert coverage.calendar == "noleap"
    assert provider.resolve_version("latest") == "20180701"
    with pytest.raises(ValueError, match="not the requested"):
        provider.resolve_version("19990101")
    provider.check_years([2015, 2026])
    with pytest.raises(ValueError, match="outside the complete record"):
        provider.check_years([2014, 2027])


def test_store_units_are_checked_before_any_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _build_zarr(tmp_path / "wrong.zarr", precip_units="mm/day")
    catalogue = _write_catalogue(tmp_path / "cat.csv", [_row(store)])
    built = CMIP6Archive(name="wrong", catalogue_url=str(catalogue), storage_options={})
    monkeypatch.setitem(CMIP6_ARCHIVES, "wrong", built)
    provider = _provider(built)
    with pytest.raises(ValueError, match="conversion would be wrong"):
        provider.check_variable(cmip6_recipe("rx1day"))
    provider.check_variable(cmip6_recipe("txx"))  # tasmax is still kelvin


def test_rx1day_converts_the_flux_and_reads_across_the_seam(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    provider = _provider(archive)
    out = provider.cache_annual_extreme(
        cmip6_recipe("rx1day"), 2017, AREA, tmp_path / "rx1day-2017.nc"
    )
    import h5netcdf  # type: ignore[import-untyped]

    with h5netcdf.File(out) as dataset:
        lon = np.asarray(dataset["lon"][:])
        lat = np.asarray(dataset["lat"][:])
        values = np.asarray(dataset["extreme"][0])
        assert dataset["extreme"].attrs["calendar"] == "noleap"
    assert lon.tolist() == [-45.0, 0.0, 45.0]  # normalised, ascending
    assert lat.tolist() == LAT.tolist()
    # 2017 is the third year: 20 + 2 + row millimetres on the wet day.
    assert np.allclose(values, np.array([22.0, 23.0, 24.0])[:, None], atol=1e-4)


def test_rx5day_sums_the_spell_without_crossing_the_block(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    provider = _provider(archive)
    out = provider.cache_annual_extreme(
        cmip6_recipe("rx5day"), 2017, AREA, tmp_path / "rx5day.nc"
    )
    import h5netcdf

    with h5netcdf.File(out) as dataset:
        values = np.asarray(dataset["extreme"][0])
    # Five days of 3 mm beats the single wet day only where the wet day is
    # smaller; here the spell totals 15 mm against a 22-24 mm single day, so
    # the five-day window containing the wet day wins: 22-24 mm + 0.
    assert np.allclose(values, np.array([22.0, 23.0, 24.0])[:, None], atol=1e-4)


def test_txx_reads_the_temperature_recipe(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    provider = _provider(archive, variable="tasmax")
    out = provider.cache_annual_extreme(
        cmip6_recipe("txx"), 2016, AREA, tmp_path / "txx.nc"
    )
    import h5netcdf

    with h5netcdf.File(out) as dataset:
        values = np.asarray(dataset["extreme"][0])
    assert np.allclose(values, 280.0 + 1 - 273.15, atol=1e-4)


def _build_year_with_gap(
    path: Path, missing_day: int, *, wet_day_depth: float = 20.0
) -> CMIP6Provider:
    """One noleap year whose time axis and data both skip ``missing_day``."""
    days = np.array([day for day in range(365) if day != missing_day])
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    time = group.create_array("time", data=days + 0.5, chunks=(days.size,))
    time.attrs.update(
        {
            "units": "days since 2015-01-01",
            "calendar": "noleap",
            "_ARRAY_DIMENSIONS": ["time"],
        }
    )
    for name, values in (("lat", LAT), ("lon", LON)):
        coordinate = group.create_array(name, data=values)
        coordinate.attrs["_ARRAY_DIMENSIONS"] = [name]
    precip = np.zeros((365, LAT.size, LON.size), dtype=np.float32)
    precip[WET_DAY] = wet_day_depth / 86400.0
    precip[200:205] = 3.0 / 86400.0
    array = group.create_array(
        "pr", data=precip[days], chunks=(days.size, LAT.size, LON.size)
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
    array.attrs["units"] = "kg m-2 s-1"
    return CMIP6Provider(
        CMIP6Dataset(
            activity_id="ScenarioMIP",
            institution_id="NOAA-GFDL",
            source_id=MODEL,
            experiment_id="ssp585",
            member_id=MEMBER,
            table_id="day",
            variable_id="pr",
            grid_label="gn",
            zstore=str(path),
            version="20180701",
        ),
        archive=CMIP6Archive(name="gap", catalogue_url="unused", storage_options={}),
    )


def test_a_missing_day_is_missing_rather_than_filled(tmp_path: Path) -> None:
    provider = _build_year_with_gap(tmp_path / "gap.zarr", WET_DAY)
    reader = provider.reader(cmip6_recipe("rx1day"), AREA, [2015])
    values = reader.reduce_block(2015, slice(0, 3), slice(0, 3))
    assert np.allclose(values, 3.0, atol=1e-4)  # only the flat spell remains


def test_days_are_placed_by_calendar_date_not_by_position(tmp_path: Path) -> None:
    # Day 202 sits in the middle of a five-day spell, and nothing else falls
    # in this year. Placed by date, the gap splits the spell and no five-day
    # window holds more than two wet days (6 mm); placed by position, the four
    # remaining wet days would close ranks into one window and report 12 mm.
    provider = _build_year_with_gap(tmp_path / "split.zarr", 202, wet_day_depth=0.0)
    reader = provider.reader(cmip6_recipe("rx5day"), AREA, [2015])
    values = reader.reduce_block(2015, slice(0, 3), slice(0, 3))
    assert np.allclose(values, 6.0, atol=1e-4)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def _plan(tmp_path: Path, **policy: Any) -> Any:
    return (
        HazardDataset.cmip6("rx1day", model=MODEL, scenario="ssp585", archive="test")
        .for_area(AREA)
        .years(YEARS[0], YEARS[-1])
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, **policy))
    )


def test_plan_materializes_with_provenance_and_reuses_cache(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    plan = _plan(tmp_path, h3_resolution=3, diagnostics=tmp_path / "diag.parquet")
    text = plan.explain()
    assert "0 cached, 12 to fetch" in text
    assert "not yet resolved" in text
    assert not (tmp_path / "cache").exists()  # explain does no I/O

    prefetched = plan.prefetch()
    assert (prefetched.cache_hits, prefetched.cache_misses) == (0, 12)
    assert prefetched.source_version == "20180701"
    again = plan.prefetch()
    assert (again.cache_hits, again.cache_misses) == (12, 0)
    assert "GFDL-ESM4.ssp585" in str(plan.explain())

    dataset = plan.materialize(tmp_path / "rx1day.parquet")
    result = dataset.materialization
    assert result is not None and result.source_cache_hits == 12

    metadata = read_hazard_metadata(tmp_path / "rx1day.parquet")
    assert metadata.source.licence == "CC-BY-4.0"
    assert metadata.source.version == "20180701"
    assert "NOAA GFDL" in (metadata.source.attribution or "")
    window = metadata.temporal_window
    assert window is not None
    assert (window.start_year, window.end_year) == (YEARS[0], YEARS[-1])
    assert window.calendar == "noleap"
    assert window.minimum_complete_years == 8
    ensemble = metadata.ensemble
    assert ensemble is not None
    assert ensemble.pooling == "single_member"
    assert ensemble.models == (MODEL,) and ensemble.members == (MEMBER,)
    assert ensemble.scenario == "ssp585"

    table = pq.read_table(tmp_path / "rx1day.parquet")
    assert set(table.column("pathway").to_pylist()) == {"ssp585"}
    assert set(table.column("horizon").to_pylist()) == {(YEARS[0] + YEARS[-1]) // 2}
    assert (tmp_path / "diag.parquet").is_file()

    offline = plan.cache(tmp_path / "cache", mode="offline").prefetch()
    assert offline.cache_hits == 12


def test_annual_extremes_expose_the_fitted_sample(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    plan = _plan(tmp_path, h3_resolution=3)
    extremes = plan.annual_extremes()
    assert extremes.num_rows == len(YEARS) * 3 * 3
    assert set(extremes.column("year").to_pylist()) == set(YEARS)
    by_year = dict(
        zip(
            extremes.column("year").to_pylist(),
            extremes.column("value").to_pylist(),
        )
    )
    # Row ordering is not promised, but every year's values must be its own.
    assert min(by_year) == YEARS[0]
    assert set(np.round(extremes.column("value").to_numpy(), 3)) == {
        float(20 + index + row)
        for index in range(len(YEARS))
        for row in range(LAT.size)
    }


def test_pathway_and_horizon_separate_two_scenarios(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    """Two pathways of one model produce joinable, distinguishable files."""
    outputs = {}
    for scenario in ("ssp585", "historical"):
        plan = (
            HazardDataset.cmip6(
                "rx1day", model=MODEL, scenario=scenario, archive="test"
            )
            .for_area(AREA)
            .years(YEARS[0], YEARS[-1])
            .cache(tmp_path / "cache")
            .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, h3_resolution=3))
        )
        outputs[scenario] = plan.materialize(tmp_path / f"{scenario}.parquet")
    for scenario, dataset in outputs.items():
        table = pq.read_table(dataset.provider.source)
        assert set(table.column("pathway").to_pylist()) == {scenario}
    # Different DRS versions, so the two caches never share an annual extreme.
    assert (
        read_hazard_metadata(tmp_path / "historical.parquet").source.version
        == "20190726"
    )


def test_stream_mode_materializes_without_a_cache(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    plan = (
        HazardDataset.cmip6("rx1day", model=MODEL, scenario="ssp585", archive="test")
        .for_area(AREA)
        .years(YEARS[0], YEARS[-1])
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, h3_resolution=3))
    )
    with pytest.raises(ValueError, match="prefetch"):
        plan.prefetch()
    with pytest.raises(ValueError, match="annual_extremes requires a cache"):
        plan.annual_extremes()
    dataset = plan.materialize(tmp_path / "out.parquet")
    assert dataset.materialization is not None
    assert dataset.materialization.canonical_rows > 0


def test_offline_without_cache_fails_clearly(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    plan = _plan(tmp_path).cache(tmp_path / "empty", mode="offline")
    with pytest.raises(FileNotFoundError, match="prefetch"):
        plan.prefetch()


def test_minimum_years_and_years_outside_the_record(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    short = (
        HazardDataset.cmip6("rx1day", model=MODEL, scenario="ssp585", archive="test")
        .for_area(AREA)
        .years(2015, 2018)
        .cache(tmp_path / "cache")
        .canonicalize()
    )
    with pytest.raises(ValueError, match="at least 20"):
        short.prefetch()
    outside = (
        HazardDataset.cmip6("rx1day", model=MODEL, scenario="ssp585", archive="test")
        .for_area(AREA)
        .years(2015, 2030)
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=4, h3_resolution=3))
    )
    with pytest.raises(ValueError, match="outside the complete record"):
        outside.prefetch()


def test_a_different_member_does_not_reuse_another_cache(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    _plan(tmp_path, h3_resolution=3).prefetch()
    other = (
        HazardDataset.cmip6(
            "rx1day",
            model=MODEL,
            scenario="ssp585",
            member="r2i1p1f1",
            archive="test",
        )
        .for_area(AREA)
        .years(YEARS[0], YEARS[-1])
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, h3_resolution=3))
    )
    assert other.prefetch().cache_misses == 12


def test_refresh_replaces_an_existing_output(
    archive: CMIP6Archive, tmp_path: Path
) -> None:
    plan = _plan(tmp_path, h3_resolution=3)
    first = plan.ensure_materialized()
    refreshed = plan.cache(tmp_path / "cache", mode="refresh").ensure_materialized()
    assert refreshed.provider.source == first.provider.source


# ---------------------------------------------------------------------------
# Licences (docs/licence-policy.md)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "identifier"),
    [
        # The wording NOAA-GFDL actually publishes: share-alike, not CC-BY.
        (
            "CMIP6 model data produced by NOAA-GFDL is licensed under a Creative "
            "Commons Attribution-ShareAlike 4.0 International License "
            "(https://creativecommons.org/licenses/).",
            "CC-BY-SA-4.0",
        ),
        ("Creative Commons Attribution 4.0 International", "CC-BY-4.0"),
        ("CC BY 4.0", "CC-BY-4.0"),
        ("CC-BY-SA-4.0", "CC-BY-SA-4.0"),
        ("Creative Commons Attribution-NonCommercial 4.0", "CC-BY-NC-4.0"),
        ("CC0 1.0 Universal Public Domain Dedication", "CC0-1.0"),
        ("Terms of use: ask us", "unknown"),
    ],
)
def test_licence_text_resolves_to_one_identifier(text: str, identifier: str) -> None:
    assert normalize_licence(text) == identifier


def _store_with_licence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, licence: str
) -> CMIP6Archive:
    store = _build_zarr(tmp_path / "licensed.zarr")
    zarr.open_group(str(store), mode="a").attrs["license"] = licence
    catalogue = _write_catalogue(tmp_path / "licensed.csv", [_row(store)])
    built = CMIP6Archive(
        name="licensed", catalogue_url=str(catalogue), storage_options={}
    )
    monkeypatch.setitem(CMIP6_ARCHIVES, "licensed", built)
    return built


def test_share_alike_is_reported_not_recorded_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _store_with_licence(
        tmp_path,
        monkeypatch,
        "Creative Commons Attribution-ShareAlike 4.0 International License",
    )
    provider = _provider(archive)
    with pytest.warns(RestrictiveLicenceWarning, match="beyond attribution"):
        assert provider.check_licence() == "CC-BY-SA-4.0"
    with pytest.raises(ValueError, match="beyond attribution"):
        provider.check_licence(strict=True)


def test_an_unreadable_licence_is_not_assumed_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _store_with_licence(tmp_path, monkeypatch, "see our terms of use")
    provider = _provider(archive)
    with pytest.warns(RestrictiveLicenceWarning, match="does not recognise"):
        assert provider.check_licence() == "unknown"


def test_strict_licence_stops_the_plan_before_any_data_is_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store_with_licence(
        tmp_path, monkeypatch, "Creative Commons Attribution-ShareAlike 4.0"
    )
    plan = (
        HazardDataset.cmip6(
            "rx1day",
            model=MODEL,
            scenario="ssp585",
            archive="licensed",
            strict_licence=True,
        )
        .for_area(AREA)
        .years(YEARS[0], YEARS[-1])
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, h3_resolution=3))
    )
    with pytest.raises(ValueError, match="CC-BY-SA-4.0"):
        plan.prefetch()
    assert not (tmp_path / "cache" / "annual-extremes").exists()


def test_the_recorded_licence_is_the_identifier_not_the_paragraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paragraph = (
        "CMIP6 model data produced by NOAA-GFDL is licensed under a Creative "
        "Commons Attribution-ShareAlike 4.0 International License. Consult "
        "https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use."
    )
    _store_with_licence(tmp_path, monkeypatch, paragraph)
    plan = (
        HazardDataset.cmip6(
            "rx1day", model=MODEL, scenario="ssp585", archive="licensed"
        )
        .for_area(AREA)
        .years(YEARS[0], YEARS[-1])
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy(minimum_years=8, h3_resolution=3))
    )
    with pytest.warns(RestrictiveLicenceWarning):
        dataset = plan.materialize(tmp_path / "sa.parquet")
    metadata = read_hazard_metadata(dataset.provider.source)
    assert metadata.source.licence == "CC-BY-SA-4.0"
    # The verbatim text stays in the cache manifest for audit.
    manifest = (tmp_path / "cache" / "manifest.json").read_text()
    assert "pcmdi.llnl.gov" in manifest
    assert "Licence: CC-BY-SA-4.0" in str(plan.explain())
