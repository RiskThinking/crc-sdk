from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import crc_framework as cf
import numpy as np
import pytest
import zarr

from crc_sdk import (
    Assets,
    Bounds,
    Cache,
    Calendar,
    DatasetCollection,
    FitFailure,
    Period,
    ReleaseReference,
    Runtime,
    Stage,
    ValidationError,
    Variable,
)
from crc_sdk._aggregation import (
    AnnualMaximum,
    Coverage,
    DailyTotal,
    TemporalReduction,
    TimestampPosition,
)
from crc_sdk._climate_sampling import CMIP6, ERA5, NEX
from crc_sdk._climate_sources import ECCCIDF, EDO, JRC
from crc_sdk._dataset_schema import ObservationTimeSelection, UnitConversion
from crc_sdk._errors import CapabilityError
from crc_sdk._evidence import EvidenceKind
from crc_sdk._evidence_recipes import Observations
from crc_sdk._inspection import Capability


@pytest.mark.parametrize(
    ("pages", "problem"),
    [
        (("<IsTruncated>true</IsTruncated>",), "has no continuation token"),
        (
            (
                "<IsTruncated>true</IsTruncated>"
                "<NextContinuationToken>again</NextContinuationToken>",
            )
            * 2,
            "repeats a continuation token",
        ),
        (
            (
                "<IsTruncated>false</IsTruncated>"
                "<NextContinuationToken>unexpected</NextContinuationToken>",
            ),
            "Complete NEX inventory page has a continuation token",
        ),
    ],
)
def test_nex_rejects_invalid_listing_pagination(
    monkeypatch: pytest.MonkeyPatch, pages: tuple[str, ...], problem: str
) -> None:
    from crc_sdk import _source_io

    remaining = iter(pages)

    def download(uri: str, path: Path) -> int:
        page = next(remaining, None)
        assert page is not None, f"unexpected catalogue request: {uri}"
        path.write_text(f"<ListBucketResult>{page}</ListBucketResult>")
        return 200

    monkeypatch.setattr(_source_io, "_download", download)
    with pytest.raises(ValidationError, match=problem):
        NEX().inspect()


def _array(
    group: Any, name: str, data: Any, dimensions: list[str], **attrs: Any
) -> Any:
    array = group.create_array(
        name, data=np.asarray(data), chunks=np.asarray(data).shape
    )
    array.attrs.update({"_ARRAY_DIMENSIONS": dimensions, **attrs})
    return array


def _grid(
    path: Path,
    *,
    calendar: str = "standard",
    days: int = 365,
    variable: str = "tasmax",
    hourly: bool = False,
) -> Any:
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    _array(
        group,
        "latitude" if hourly else "lat",
        [0.0, 0.25],
        ["latitude" if hourly else "lat"],
    )
    _array(
        group,
        "longitude" if hourly else "lon",
        [0.0, 0.25],
        ["longitude" if hourly else "lon"],
    )
    size = days * 24 + 1 if hourly else days
    _array(
        group,
        "time",
        np.arange(size),
        ["time"],
        units=("hours" if hourly else "days") + " since 2001-01-01 00:00:00",
        calendar=calendar,
    )
    dims = ["time", "latitude", "longitude"] if hourly else ["time", "lat", "lon"]
    values = np.full((size, 2, 2), 280.0)
    _array(group, variable, values, dims, units="K")
    if hourly:
        group.attrs.update(valid_time_start="2001-01-01", valid_time_stop="2002-01-01")
    zarr.consolidate_metadata(str(path))
    return group


AREA = Bounds(west=-0.01, south=-0.01, east=0.01, north=0.01)


def test_climate_constructors_explain_and_empty_are_pure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("pure recipe acquired source bytes")

    monkeypatch.setattr("crc_sdk._source_io._download", forbidden)
    for provider in (ERA5, CMIP6, NEX, JRC, ECCCIDF, EDO):
        recipe = provider()
        assert recipe.explain().operations
        assert recipe.select(variables=[]).read().table.num_rows == 0
    for provider in (ERA5, CMIP6, NEX):
        with pytest.raises(CapabilityError, match="indicator"):
            provider().observations()
    for raw in (
        ERA5(
            indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION
        ).observations(),
        CMIP6(
            indicator=CMIP6.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION
        ).observations(),
        NEX(indicator=NEX.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION).observations(),
    ):
        assert raw.select(variables=[]).read().table.num_rows == 0
        assert raw.explain().operations
        assert not hasattr(raw, "fit")


def test_era5_hourly_complete_days_and_owned_offline_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "era5.zarr"
    _grid(path, variable="2m_temperature", hourly=True)
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    recipe = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    ).select(area=AREA, time=Period.window(2001, 2001))
    cache = tmp_path / "cache"
    result = recipe.read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert result.table["value"].to_pylist() == pytest.approx([6.85])
    assert result.table["coverage"].to_pylist()[0]["expected"] == 365
    assert result.report.counts.usable_years == 1
    assert result.report.complete
    assert result.metadata.source is not None
    assert result.metadata.source.resources

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("offline read attempted source access")

    monkeypatch.setattr("crc_sdk._source_io._download", forbidden)
    replay = recipe.read(runtime=Runtime(cache=Cache.offline(cache)))
    assert replay.table.equals(result.table, check_metadata=False)


def test_era5_inspection_short_circuits_empty_and_unknown_selectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("selector validation acquired source bytes")

    monkeypatch.setattr("crc_sdk._climate_sampling._era5_source", forbidden)
    runtime = Runtime(cache=Cache.offline(tmp_path))
    recipe = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    ).select(area=AREA, time=Period.window(2001, 2001))
    for empty in (recipe.select(variables=[]), recipe.select(series=[])):
        inspection = empty.inspect(runtime=runtime)
        assert not inspection.series
        assert not inspection.requirements
        assert not inspection.releases
        read = empty.read(runtime=runtime)
        assert inspection.schema is not None
        assert inspection.schema.equals(read.table.schema, check_metadata=False)
        assert read.table.num_rows == 0
    invalid = recipe.select(variables="typo")
    with pytest.raises(ValueError, match="unknown requested identity 'typo'"):
        invalid.inspect(runtime=runtime)
    with pytest.raises(ValueError, match="unknown requested identity 'typo'"):
        invalid.read(runtime=runtime)


def test_era5_inspection_filters_the_discovered_series_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "era5.zarr"
    _grid(path, variable="2m_temperature", hourly=True)
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    source = ERA5().select(area=AREA, time=Period.window(2001, 2001))
    before = source.inspect()
    assert len(before.series) == 4
    filtered = source.select(
        variables=[s.variable for s in before.series[:2]]
    ).inspect()
    assert filtered.series == before.series[:2]
    assert filtered.metadata is not None
    assert filtered.metadata.spec.variables == tuple(
        series.variable for series in before.series[:2]
    )
    assert any(
        requirement.code == "indicator_required"
        for requirement in filtered.requirements
    )


def test_collection_replays_its_inspected_publication_after_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "era5.zarr"
    _grid(path, variable="2m_temperature", hourly=True)
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    fitted = (
        ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
        .select(area=AREA, time=Period.window(2001, 2001))
        .fit(
            cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT),
            on_failure=FitFailure.KEEP_NO_DATA,
        )
    )
    collection = DatasetCollection({"observed": fitted})
    (reference,) = collection.inspect().series
    assert reference.release is not None and reference.release.id.startswith(
        "inventory:"
    )
    selected = collection.select(series=reference)
    assert selected.inspect().series == (reference,)
    assert selected.validate().valid is True
    result = selected.read()
    source = result.metadata.resources[0].metadata.source
    assert source is not None and source.release is not None
    assert source.release.id == reference.release.id.removeprefix("inventory:")
    assert reference.release in {
        item.release for item in result.report.source_identities
    }
    exact = replace(reference, release=source.release)
    exact_selection = collection.select(series=exact)
    assert exact_selection.inspect().series == (reference,)
    assert exact_selection.read().table.equals(result.table, check_metadata=False)
    assert (
        exact_selection.read()
        .table.drop(["member_id", "resource_id"])
        .equals(fitted.select(series=exact).read().table, check_metadata=False)
    )

    destination = tmp_path / "collection"
    selected.write(destination)
    reopened = DatasetCollection.open(destination)
    assert reopened.validate().valid is True
    assert reopened.read().table.equals(result.table, check_metadata=False)
    assert (
        reopened.select(series=reference)
        .read()
        .table.equals(result.table, check_metadata=False)
    )

    manifest = destination / "manifest.json"
    document = json.loads(manifest.read_text())
    document["selection"]["series"][0]["release"]["sha256"] = "sha256:" + "0" * 64
    manifest.write_text(json.dumps(document))
    invalid = DatasetCollection.open(destination)
    assert invalid.validate().valid is False
    with pytest.raises(ValidationError, match="saved series selection"):
        invalid.read()


def test_era5_annual_results_retain_requested_years_outside_source_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "era5.zarr"
    _grid(path, variable="2m_temperature", hourly=True)
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    years = Period.window(2000, 2003)
    recipe = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    ).select(area=AREA, time=years)
    result = recipe.read()
    assert result.table["time"].to_pylist() == ["2000", "2001", "2002", "2003"]
    assert result.table["value"].to_pylist() == [None, pytest.approx(6.85), None, None]
    coverage = result.table["coverage"].to_pylist()
    assert coverage[0] == {"valid": 0, "expected": 366, "minimum_fraction": 1.0}
    assert coverage[-1] == {"valid": 0, "expected": 365, "minimum_fraction": 1.0}
    counts = result.report.counts
    assert counts.requested == counts.produced == 4
    assert counts.usable_years == 1
    assert counts.incomplete_years == counts.no_data == 3


def test_era5_raw_reductions_retain_empty_and_gapped_requested_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "rain.zarr"
    group = _grid(path, variable="total_precipitation", hourly=True, days=2)
    group["total_precipitation"][:] = 0.001
    group["total_precipitation"].attrs["units"] = "m"
    zarr.consolidate_metadata(str(path))
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    coverage = Coverage.minimum_fraction(0.001)
    source = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION, coverage=coverage
    )
    for years, expected, observations in (
        (Period.window(1999, 2000), [None, None], 0),
        (Period.window(2000, 2002), [None, 24.0, None], 48),
        (Period.window(2002, 2002), [None], 0),
    ):
        predefined = source.select(area=AREA, time=years).read()
        raw = source.select(area=AREA, time=years).observations()
        collected = raw.read()
        assert collected.table.num_rows == observations
        early = raw.aggregate(DailyTotal()).aggregate(AnnualMaximum(coverage=coverage))
        late = (
            source.observations()
            .aggregate(DailyTotal())
            .aggregate(AnnualMaximum(coverage=coverage))
            .select(area=AREA, time=years)
        )
        raw_path = tmp_path / f"raw-{years.start}.parquet"
        raw.write(raw_path)
        reopened = Observations.open(raw_path)
        daily = raw.aggregate(DailyTotal()).write(
            tmp_path / f"daily-{years.start}.parquet"
        )
        alternatives = (
            early,
            late,
            collected.aggregate(DailyTotal()).aggregate(
                AnnualMaximum(coverage=coverage)
            ),
            reopened.aggregate(DailyTotal()).aggregate(
                AnnualMaximum(coverage=coverage)
            ),
            daily.aggregate(AnnualMaximum(coverage=coverage)),
        )
        for explicit in alternatives:
            result = explicit.read()
            assert result.table["support_id"].equals(predefined.table["support_id"])
            assert result.table["time"].equals(predefined.table["time"])
            assert result.table["value"].to_pylist() == expected
            assert result.report.counts == predefined.report.counts
            assert result.metadata.source is not None
            assert result.metadata.source in result.report.source_identities


@pytest.mark.parametrize(
    ("time_start", "rain", "error"),
    [
        ("00:00:00", -0.001, "nonnegative source intervals"),
        ("00:30:00", 0.001, "timestamps must identify hourly intervals"),
    ],
)
def test_era5_predefined_reduction_checks_native_precipitation_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    time_start: str,
    rain: float,
    error: str,
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "rain.zarr"
    group = _grid(path, variable="total_precipitation", hourly=True)
    group["time"].attrs["units"] = f"hours since 2001-01-01 {time_start}"
    group["total_precipitation"][:] = rain
    group["total_precipitation"].attrs["units"] = "m"
    zarr.consolidate_metadata(str(path))
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    source = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION,
        coverage=Coverage.minimum_fraction(0.99),
    ).select(area=AREA, time=Period.window(2001, 2001))
    with pytest.raises(ValidationError, match=error):
        source.read()


def test_era5_inspected_period_survives_narrower_observation_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "era5.zarr"
    _grid(path, variable="2m_temperature", hourly=True)
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    source = ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
    dated = source.select(area=AREA, time=Period.window(2000, 2002)).inspect().series[0]
    window = Period.window(2001, 2001, calendar=Calendar.GREGORIAN)
    narrowed = ERA5().select(series=dated, area=AREA, time=window)
    assert narrowed.inspect().series[0].period == dated.period
    result = narrowed.read()
    assert result.table["time"].to_pylist() == ["2001"]
    assert result.metadata.spec.series[0].period == dated.period
    assert result.metadata.provenance[-1] == ObservationTimeSelection(time=window)


def test_cmip6_360_day_actual_calendar_and_reference_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module
    from crc_sdk._providers.cmip6 import CATALOGUE_COLUMNS

    path = tmp_path / "cmip.zarr"
    _grid(path, calendar="360_day", days=360)
    catalogue = tmp_path / "catalogue.csv"
    row = dict(
        zip(
            CATALOGUE_COLUMNS,
            [
                "CMIP",
                "TEST",
                "MODEL",
                "historical",
                "r1i1p1f1",
                "day",
                "tasmax",
                "gn",
                path.as_uri(),
                "20260101",
            ],
            strict=True,
        )
    )
    with catalogue.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=CATALOGUE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)
        writer.writerow(
            {
                **row,
                "source_id": "OTHER",
                "experiment_id": "ssp245",
                "member_id": "r2i1p1f1",
                "variable_id": "tasmin",
                "zstore": "file:///no-discovery-payload.zarr",
            }
        )
        writer.writerow({**row, "source_id": "UNSUPPORTED", "variable_id": "sfcWind"})
        writer.writerow({**row, "source_id": "UNSUPPORTED", "grid_label": "unknown"})
    monkeypatch.setattr(module, "CMIP6_CATALOGUE", catalogue.as_uri())
    with monkeypatch.context() as discovery:
        discovery.setattr(
            module,
            "open_zarr_snapshot",
            lambda *args: pytest.fail("catalogue discovery opened a source array"),
        )
        combinations = CMIP6().inspect().series
        assert [
            (r.model, r.scenario.id, r.member, r.variable.id)
            for r in combinations
            if r.scenario is not None
        ] == [
            (
                "MODEL",
                "historical",
                "r1i1p1f1",
                "cmip6:annual_maximum_daily_maximum_temperature",
            ),
            (
                "OTHER",
                "ssp245",
                "r2i1p1f1",
                "cmip6:annual_minimum_daily_minimum_temperature",
            ),
        ]
        assert CMIP6(model="MODEL").inspect().series == combinations[:1]
        assert combinations[0].period is None
        assert combinations[0].release is not None
        assert (
            CMIP6(release=combinations[0].release).inspect().series == combinations[:1]
        )
        with pytest.raises(ValueError, match="conflicting model identities"):
            CMIP6(model="OTHER").select(series=combinations[0], area=AREA).explain()
    recipe = CMIP6().select(series=combinations[0], area=AREA)
    inventory = recipe.inspect()
    assert inventory.series[0].release == combinations[0].release
    assert inventory.releases == (inventory.series[0].release,)
    assert inventory.series[0].period is not None
    assert inventory.series[0].period.calendar is not None
    assert inventory.series[0].period.calendar.value == "360_day"
    result = recipe.select(series=[inventory.series[0]]).read()
    assert result.table["value"].to_pylist() == pytest.approx([6.85])
    assert result.table["coverage"].to_pylist()[0]["expected"] == 360
    raw = recipe.select(series=[inventory.series[0]]).observations()
    native = raw.read(runtime=Runtime(batch_rows=30))
    assert native.metadata.kind is EvidenceKind.OBSERVATIONS
    assert native.table.num_rows == 360
    assert native.table["value"].to_pylist() == pytest.approx([280] * 360)
    assert native.table["time"].to_pylist()[59] == "2001-02-30"
    assert native.metadata.sampling is not None
    assert native.metadata.sampling.calendar.value == "360_day"
    assert native.metadata.sampling.step == cf.Duration(1, unit=cf.TimeUnit.DAY)
    assert native.metadata.spec.variables[0].id == "cmip6:tasmax"
    assert native.metadata.spec.series[0].probability is None
    assert (
        native.metadata.source is not None
        and native.metadata.source.member == "r1i1p1f1"
    )
    assert not any(
        isinstance(item, TemporalReduction) for item in native.metadata.provenance
    )
    raw_inventory = raw.inspect()
    assert Capability.AGGREGATE in raw_inventory.capabilities
    assert Capability.FIT not in raw_inventory.capabilities
    assert raw.select(series=raw_inventory.series).read().table.num_rows == 360
    assert raw.aggregate(AnnualMaximum()).read().table["value"].to_pylist() == [280.0]
    assert result.metadata.source is not None
    assert result.metadata.source.model == "MODEL"
    release = result.metadata.source.release
    assert release is not None
    pinned = recipe.select(series=replace(inventory.series[0], release=release))
    assert pinned.read().table.equals(result.table, check_metadata=False)
    saved = recipe.write(tmp_path / "annual.parquet")
    reference = saved.inspect().series[0]
    assert reference.release == release
    assert reference.model == "MODEL" and reference.member == "r1i1p1f1"
    assert (
        CMIP6()
        .select(series=reference, area=AREA)
        .read()
        .table.equals(result.table, check_metadata=False)
    )
    array = zarr.open_group(str(path), mode="r+")["tasmax"]
    assert isinstance(array, zarr.Array)
    array[0, 0, 0] = 291.0
    with pytest.raises(OSError, match="release changed"):
        pinned.read(limit=1)


def _netcdf(
    path: Path,
    variable: str,
    times: Any,
    units: str,
    values: Any,
    *,
    time_units: str = "days since 2001-01-01",
    calendar: str = "standard",
    licence_attribute: str = "license",
) -> None:
    import h5netcdf  # type: ignore[import-untyped]

    with h5netcdf.File(path, "w") as file:
        file.dimensions = {"time": len(times), "lat": 2, "lon": 2}
        for name, data, dims in (
            ("lat", [0.0, 0.25], ("lat",)),
            ("lon", [0.0, 0.25], ("lon",)),
            ("time", times, ("time",)),
            (variable, values, ("time", "lat", "lon")),
        ):
            array = file.create_variable(name, dims, dtype="f8")
            array[:] = data
        file.variables["lat"].attrs["_FillValue"] = np.nan
        file.variables["time"].attrs.update(units=time_units, calendar=calendar)
        file.variables[variable].attrs["units"] = units
        file.attrs[licence_attribute] = "CC-BY-SA-4.0"


@pytest.mark.parametrize(
    "variable,unit,source_value,annual_value,native_value",
    [("tasmax", "K", 290.0, 16.85, 290.0), ("pr", "kg m-2 s-1", 1 / 86400, 1.0, 1.0)],
)
def test_nex_real_native_netcdf_view_and_pinned_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    unit: str,
    source_value: float,
    annual_value: float,
    native_value: float,
) -> None:
    import urllib.parse

    from crc_sdk import _climate_sampling as module
    from crc_sdk._climate_io import netcdf_header_snapshot, netcdf_window_snapshot

    prefix = module.NEX_PREFIX
    directory = prefix + f"MODEL/historical/r1/{variable}/"
    paths = {}
    for year in (2001, 2002):
        name = f"{variable}_day_MODEL_historical_r1_gn_{year}.nc"
        path = tmp_path / name
        _netcdf(
            path,
            variable,
            np.arange(365),
            unit,
            np.full((365, 2, 2), source_value),
            time_units=f"days since {year}-01-01",
            licence_attribute="cmip6_license",
        )
        paths[name] = path
    keys = {
        prefix: [prefix + "MODEL/"],
        prefix + "MODEL/": [prefix + "MODEL/historical/"],
        prefix + "MODEL/historical/": [prefix + "MODEL/historical/r1/"],
        directory: [directory + name for name in paths],
    }
    other_directory = prefix + "OTHER/ssp585/r2/tasmin/"
    catalogue_pages = {
        None: keys[directory],
        "page 2": [other_directory + "tasmin_day_OTHER_ssp585_r2_gn_2003.nc"],
    }

    def acquire(uri: str, destination: Path) -> int:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(uri).query)
        requested = query["prefix"][0]
        next_page = ""
        if "delimiter" in query:
            objects = keys[requested]
        else:
            token = query.get("continuation-token", [None])[0]
            objects = [
                key for key in catalogue_pages[token] if key.startswith(requested)
            ]
            if token is None:
                next_page = "<NextContinuationToken>page 2</NextContinuationToken>"
        contents = "".join(
            (
                f"<CommonPrefixes><Prefix>{v}</Prefix></CommonPrefixes>"
                if v.endswith("/")
                else f"<Contents><Key>{v}</Key></Contents>"
            )
            for v in objects
        )
        destination.write_text(
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + contents
            + next_page
            + f"<IsTruncated>{'true' if next_page else 'false'}</IsTruncated>"
            "</ListBucketResult>"
        )
        return 200

    monkeypatch.setattr(
        module,
        "netcdf_header_snapshot",
        lambda scope, uri, variable: netcdf_header_snapshot(
            scope, paths[uri.rsplit("/", 1)[-1]].as_uri(), variable
        ),
    )
    monkeypatch.setattr(module, "NEX_BUCKET", "https://fixture.test")
    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    monkeypatch.setattr(
        module,
        "netcdf_window_snapshot",
        lambda scope, uri, **kw: netcdf_window_snapshot(
            scope, paths[uri.rsplit("/", 1)[-1]].as_uri(), **kw
        ),
    )
    indicator = (
        NEX.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION
        if variable == "pr"
        else NEX.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    )
    with monkeypatch.context() as discovery:
        discovery.setattr(
            module,
            "netcdf_header_snapshot",
            lambda *args: pytest.fail("catalogue discovery opened a NetCDF header"),
        )
        combinations = NEX().inspect().series
        assert {
            (r.model, r.scenario.id, r.member)
            for r in combinations
            if r.scenario is not None
        } == {("MODEL", "historical", "r1"), ("OTHER", "ssp585", "r2")}
        assert all(r.period is None for r in combinations)
        (choice,) = NEX(indicator=indicator).inspect().series
        assert choice.model == "MODEL" and choice.member == "r1"
        assert choice.release is not None
        assert NEX(release=choice.release, indicator=indicator).inspect().series == (
            choice,
        )
    recipe = NEX().select(series=choice, area=AREA)
    cache = tmp_path / "cache"
    result = recipe.read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert result.table["value"].to_pylist() == pytest.approx([annual_value] * 2)
    assert result.metadata.source is not None
    assert result.metadata.source.model == "MODEL"
    assert result.metadata.source.member == "r1"
    release = result.metadata.source.release
    assert release is not None
    assert any("#crc-variable=" in r.uri for r in result.metadata.source.resources)
    windows = [r for r in result.metadata.source.resources if "#crc-variable=" in r.uri]
    assert windows and all(r.license == "CC-BY-SA-4.0" for r in windows)
    raw = recipe.observations()
    native = raw.read(runtime=Runtime(cache=Cache.reuse(cache), batch_rows=31))
    assert native.table.num_rows == 730
    assert native.table["value"].to_pylist() == pytest.approx([native_value] * 730)
    assert native.metadata.sampling is not None
    assert native.metadata.sampling.step == cf.Duration(1, unit=cf.TimeUnit.DAY)
    assert native.metadata.spec.series[0].scenario is not None
    assert native.metadata.spec.series[0].scenario.id == "historical"
    assert not any(
        isinstance(item, TemporalReduction) for item in native.metadata.provenance
    )
    if variable == "pr":
        assert isinstance(
            native.metadata.spec.variables[0].measurement, cf.PrecipitationRate
        )
        assert any(
            isinstance(item, UnitConversion) and item.scale == 86400
            for item in native.metadata.provenance
        )
    monkeypatch.setattr(
        "crc_sdk._source_io._download", lambda *a: pytest.fail("offline acquisition")
    )
    offline = Runtime(cache=Cache.offline(cache))
    reference = recipe.inspect(runtime=offline).series[0]
    assert reference.release == choice.release
    assert (
        recipe.select(series=replace(reference, release=release))
        .read(runtime=offline)
        .table.equals(result.table, check_metadata=False)
    )
    assert raw.read(runtime=Runtime(cache=Cache.offline(cache))).table.equals(
        native.table, check_metadata=False
    )
    # The listing and first file used for header discovery still describe the
    # same publication. Only the second year's payload changes.
    _netcdf(
        paths[f"{variable}_day_MODEL_historical_r1_gn_2002.nc"],
        variable,
        np.arange(365),
        unit,
        np.full((365, 2, 2), source_value * 1.01),
        time_units="days since 2002-01-01",
        licence_attribute="cmip6_license",
    )
    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    pinned = recipe.select(series=replace(reference, release=release))
    with pytest.raises(OSError, match="release changed"):
        pinned.read(limit=1, runtime=Runtime(cache=Cache.refresh(cache), batch_rows=1))


def test_nex_year_without_a_published_file_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.parse

    from crc_sdk import _climate_sampling as module
    from crc_sdk._climate_io import netcdf_header_snapshot, netcdf_window_snapshot

    directory = module.NEX_PREFIX + "MODEL/historical/r1/tasmax/"
    name = "tasmax_day_MODEL_historical_r1_gn_2001.nc"
    path = tmp_path / name
    _netcdf(path, "tasmax", np.arange(365), "K", np.full((365, 2, 2), 290.0))

    def acquire(uri: str, destination: Path) -> int:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(uri).query)
        assert query["prefix"][0] == directory
        destination.write_text(
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Contents><Key>{directory + name}</Key></Contents>"
            "<IsTruncated>false</IsTruncated></ListBucketResult>"
        )
        return 200

    monkeypatch.setattr(module, "NEX_BUCKET", "https://fixture.test")
    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    monkeypatch.setattr(
        module,
        "netcdf_header_snapshot",
        lambda scope, uri, variable: netcdf_header_snapshot(
            scope, path.as_uri(), variable
        ),
    )
    monkeypatch.setattr(
        module,
        "netcdf_window_snapshot",
        lambda scope, uri, **kw: netcdf_window_snapshot(scope, path.as_uri(), **kw),
    )
    result = (
        NEX(
            indicator=NEX.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE,
            model="MODEL",
            scenario="historical",
            member="r1",
        )
        .select(area=AREA, time=Period.window(2001, 2003))
        .read()
    )
    assert result.table["time"].to_pylist() == ["2001", "2002", "2003"]
    assert result.table["value"].to_pylist() == [pytest.approx(16.85), None, None]
    assert result.report.counts.incomplete_years == 2
    (unavailable,) = (
        d for d in result.report.diagnostics if d.code == "source_year_unavailable"
    )
    assert unavailable.stage is Stage.ACQUISITION
    assert "2002-2003" in unavailable.message and "2001" in unavailable.message


def test_edo_native_dekads_are_readable_without_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk._providers import jrc_edo
    from crc_sdk._source_io import _download

    root = tmp_path / "edo"
    root.mkdir()
    name = "sminx_m_eul_20010101_20011221_t.nc"
    _netcdf(
        root / name,
        "sminx",
        [0, 10, 20],
        "1",
        np.full((3, 2, 2), 0.4),
        calendar="proleptic_gregorian",
    )
    (root / "index.html").write_text(f'<a href="{name}">{name}</a>')
    monkeypatch.setattr(jrc_edo, "SMI", replace(jrc_edo.SMI, base_url=root.as_uri()))

    def acquire(uri: str, destination: Path) -> int:
        return _download(
            (root / "index.html").as_uri() if uri == root.as_uri() + "/" else uri,
            destination,
        )

    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    recipe = EDO(product=EDO.Product.SOIL_MOISTURE_INDEX).select(area=AREA)
    cache = tmp_path / "cache"
    result = recipe.read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert result.metadata.source is not None
    release = result.metadata.source.release
    assert release is not None
    monkeypatch.setattr(
        "crc_sdk._source_io._download", lambda *a: pytest.fail("offline acquisition")
    )
    pinned = EDO(release=release).select(area=AREA)
    offline = Runtime(cache=Cache.offline(cache))
    (reference,) = pinned.inspect(runtime=offline).series
    assert reference.release is not None
    assert reference.release.id == f"inventory:{release.id}"
    selected = pinned.select(series=reference)
    assert selected.read(runtime=offline).table.equals(
        result.table, check_metadata=False
    )
    stale_inventory = replace(reference.release, sha256="sha256:" + "0" * 64)
    with pytest.raises(OSError, match="source release changed"):
        pinned.select(series=replace(reference, release=stale_inventory)).inspect(
            runtime=offline
        )
    assert result.table["time"].to_pylist() == [
        "2001-01-01",
        "2001-01-11",
        "2001-01-21",
    ]
    assert result.table["value"].to_pylist() == pytest.approx([0.4] * 3)
    assert result.metadata.sampling is None
    assert result.metadata.spec.series[0].probability is None
    _netcdf(
        root / name,
        "sminx",
        [0, 10, 20],
        "1",
        np.full((3, 2, 2), 0.6),
        calendar="proleptic_gregorian",
    )
    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    with pytest.raises(OSError, match="release changed"):
        selected.read(limit=1, runtime=Runtime(cache=Cache.refresh(cache)))


def test_edo_empty_selection_avoids_offline_metadata_lookup(tmp_path: Path) -> None:
    selected = EDO().select(series=[])
    offline = Runtime(cache=Cache.offline(tmp_path))
    inspected = selected.inspect(runtime=offline)
    assert inspected.schema is not None
    assert not (inspected.series or inspected.releases or inspected.requirements)
    assert not selected.explain().requirements
    assert selected.read(runtime=offline).table.num_rows == 0


def test_edo_rejects_station_selection_without_acquisition(tmp_path: Path) -> None:
    offline = Runtime(cache=Cache.offline(tmp_path))
    recipe = EDO().select(area=AREA)
    selected = recipe.select(stations="not-present")
    with pytest.raises(CapabilityError, match="EDO uses a grid"):
        selected.inspect(runtime=offline)
    with pytest.raises(CapabilityError, match="EDO uses a grid"):
        selected.read(runtime=offline)
    empty = recipe.select(stations=[])
    assert not empty.inspect(runtime=offline).series
    assert empty.read(runtime=offline).table.num_rows == 0


def test_jrc_preserves_published_quantile_knots_and_native_pixel_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rasterio  # type: ignore[import-untyped]
    from rasterio.transform import from_origin  # type: ignore[import-untyped]

    from crc_sdk._providers import jrc
    from crc_sdk._schema import table_to_rows

    root = tmp_path / "jrc"
    root.mkdir()
    dataset = replace(
        jrc.EFAS,
        base_url=root.as_uri(),
        filename_template="rp{return_period}.tif",
        available_return_periods=(10, 100),
    )
    (root / "README.txt").write_text("Dataset version " + dataset.version)
    for period in (10, 100):
        with rasterio.open(
            root / f"rp{period}.tif",
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float64",
            crs="EPSG:4326",
            transform=from_origin(-0.125, 0.375, 0.25, 0.25),
        ) as file:
            file.write(np.full((2, 2), period / 10), 1)
    monkeypatch.setitem(jrc.JRC_DATASETS, "efas", dataset)
    recipe = JRC(product=JRC.Product.EFAS).select(area=AREA)
    inventory = recipe.inspect()
    (reference,) = inventory.series
    assert inventory.releases == (reference.release,)
    cache = tmp_path / "cache"
    result = recipe.read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert result.metadata.source is not None
    release = result.metadata.source.release
    assert release is not None
    assert reference.release != release
    assets = Assets({"asset_id": ["site"], "longitude": [0.0], "latitude": [0.0]})
    evaluation = recipe.evaluate(assets).return_periods([10]).read()
    assert evaluation.table["value"].to_pylist() == [1.0]
    assert evaluation.metadata.source is not None
    assert evaluation.metadata.source.release is not None
    assert evaluation.metadata.source.release.sha256 == release.sha256
    assert evaluation.metadata.inputs[0].metadata.report.valid is True
    assert (
        JRC()
        .select(series=reference, area=AREA)
        .read()
        .table.equals(result.table, check_metadata=False)
    )
    for variable in (reference.variable, reference.variable.id):
        selected = JRC().select(variables=variable, area=AREA)
        assert selected.inspect().series == (reference,)
        assert selected.read().table.equals(result.table, check_metadata=False)
    glofas = "jrc:glofas:flood_depth"
    combined = JRC().select(
        series=reference, variables=[reference.variable.id, glofas], area=AREA
    )
    assert combined.inspect().series == (reference,)
    assert combined.read().table.equals(result.table, check_metadata=False)
    disjoint = JRC().select(series=reference, variables=glofas, area=AREA)
    offline = Runtime(cache=Cache.offline(tmp_path / "empty-cache"))
    assert not disjoint.explain().requirements
    assert not disjoint.inspect(runtime=offline).series
    assert disjoint.read(runtime=offline).table.num_rows == 0
    collection = DatasetCollection({"river": JRC()})
    for composed in (
        collection.select(series=reference, area=AREA),
        collection.select(variables=reference.variable, area=AREA),
        collection.select(variables=reference.variable.id, area=AREA),
    ):
        assert composed.explain().requirements == ()
        assert composed.inspect().series == (reference,)
        table = composed.read().table.drop(["member_id", "resource_id"])
        assert table.equals(result.table, check_metadata=False)
    exact = replace(reference, release=release)
    for member in (JRC(), JRC(product=JRC.Product.EFAS)):
        exact_collection = DatasetCollection({"river": member}).select(
            series=exact, area=AREA
        )
        assert exact_collection.inspect().series == (reference,)
        table = exact_collection.read().table.drop(["member_id", "resource_id"])
        assert table.equals(result.table, check_metadata=False)
    pinned = JRC(product=JRC.Product.EFAS, release=release).select(area=AREA)
    archived = recipe.write(tmp_path / "archived.parquet")
    assert pinned.read(runtime=Runtime(cache=Cache.offline(cache))).table.equals(
        result.table, check_metadata=False
    )
    assert (
        pinned.select(series=reference)
        .read(runtime=Runtime(cache=Cache.offline(cache)))
        .table.equals(result.table, check_metadata=False)
    )
    for live in (pinned, recipe):
        matching = DatasetCollection({"archived": archived, "live": live}).select(
            series=exact
        )
        assert len(matching.inspect().series) == 2
        assert matching.read().table["member_id"].to_pylist() == ["archived", "live"]

    unbounded_source = JRC(product=JRC.Product.EFAS)
    unbounded = DatasetCollection(
        {"unbounded": unbounded_source, "pinned": pinned}
    ).select(series=exact)
    unbounded_inspection = unbounded.inspect()
    assert len(unbounded_inspection.series) == 1
    assert not unbounded_inspection.requirements
    assert unbounded.read().table["member_id"].to_pylist() == ["pinned"]
    with pytest.raises(CapabilityError, match=r"requires \.select\(area="):
        DatasetCollection({"unbounded": unbounded_source}).select(series=exact).read()

    other = JRC(product=JRC.Product.EFAS).select(
        area=Bounds(west=0.2, east=0.3, south=0.2, north=0.3)
    )
    other_source = other.read().metadata.source
    assert other_source is not None and other_source.release is not None
    assert other_source.release.id == release.id
    assert other_source.release.sha256 != release.sha256
    other_exact = replace(reference, release=other_source.release)
    different_payloads = DatasetCollection({"first": recipe, "other": other}).select(
        series=[exact, other_exact],
    )
    assert different_payloads.inspect().series == (exact, other_exact)
    assert different_payloads.read().table["member_id"].to_pylist() == [
        "first",
        "other",
    ]

    with rasterio.open(root / "rp100.tif", "r+") as file:
        file.write(np.full((2, 2), 11.0), 1)
    with pytest.raises(OSError, match="release changed"):
        pinned.read(limit=1)
    mixed = DatasetCollection({"live": recipe, "archived": archived}).select(
        series=exact
    )
    assert mixed.inspect().series == (exact,)
    assert mixed.read().table["member_id"].to_pylist() == ["archived"]
    with pytest.raises(OSError, match="release changed"):
        DatasetCollection({"archived": archived, "pinned": pinned}).select(
            series=exact
        ).read()
    raster = root / "rp100.tif"
    backup = raster.rename(root / "archived-rp100.tif")
    try:
        assert mixed.inspect().series == (exact,)
        assert mixed.read().table["member_id"].to_pylist() == ["archived"]
    finally:
        backup.rename(raster)
    readme = root / "README.txt"
    backup = readme.rename(root / "archived-README.txt")
    try:
        assert len(mixed.inspect().series) == 1
        assert mixed.read().table["member_id"].to_pylist() == ["archived"]
    finally:
        backup.rename(readme)
    stale = collection.select(series=exact, area=AREA)
    with stale.batches() as batches, pytest.raises(OSError, match="release changed"):
        next(iter(batches))
    assert pinned.inspect().releases == inventory.releases
    monkeypatch.setattr(
        "crc_sdk._source_io._download", lambda *a: pytest.fail("offline acquisition")
    )
    assert recipe.read(runtime=Runtime(cache=Cache.offline(cache))).table.equals(
        result.table, check_metadata=False
    )
    assert result.table.num_rows == 1
    curve = table_to_rows(result.table, result.metadata)[0].curve
    assert curve is not None
    assert curve.quantile([0.9, 0.99]) == pytest.approx([1, 10])
    assert result.table["support_id"].to_pylist()[0].endswith(":1:0")


@pytest.mark.parametrize("product", [JRC.Product.EFAS, JRC.Product.GLOFAS])
def test_jrc_area_outside_the_product_extent_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, product: JRC.Product
) -> None:
    import rasterio
    from rasterio.transform import from_origin

    from crc_sdk._providers import jrc

    root = tmp_path / "jrc"
    root.mkdir()
    tiled = product is JRC.Product.GLOFAS
    index = root / "tile_extents.geojson"
    base = jrc.GLOFAS if tiled else jrc.EFAS
    dataset = replace(
        base,
        base_url=root.as_uri(),
        tile_index_url=index.as_uri() if tiled else None,
        filename_template=("{tile_id}_" if tiled else "") + "rp{return_period}.tif",
        available_return_periods=(10, 100),
    )
    (root / "README.txt").write_text("Dataset version " + dataset.version)
    index.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1, "name": "N0_E0"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[-0.125, -0.125], [0.375, -0.125], [0.375, 0.375]]
                                + [[-0.125, 0.375], [-0.125, -0.125]]
                            ],
                        },
                    }
                ],
            }
        )
    )
    for period in (10, 100):
        folder = root / f"RP{period}" if tiled else root
        folder.mkdir(exist_ok=True)
        with rasterio.open(
            folder / (("ID1_N0_E0_" if tiled else "") + f"rp{period}.tif"),
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float64",
            crs="EPSG:4326",
            transform=from_origin(-0.125, 0.375, 0.25, 0.25),
        ) as file:
            file.write(np.full((2, 2), period / 10), 1)
    monkeypatch.setitem(jrc.JRC_DATASETS, product.value, dataset)
    covered = JRC(product=product).select(area=AREA).read()
    assert covered.table.num_rows == 1
    assert not covered.report.diagnostics
    outside = (
        JRC(product=product)
        .select(area=Bounds(west=-122.15, south=37.42, east=-121.85, north=37.62))
        .read()
    )
    assert outside.table.num_rows == 0
    (diagnostic,) = outside.report.diagnostics
    assert diagnostic.code == "area_outside_coverage"
    assert diagnostic.stage is Stage.ACQUISITION
    assert diagnostic.severity is cf.DiagnosticSeverity.WARNING


def test_jrc_empty_selection_avoids_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "crc_sdk._source_io._download",
        lambda *a: pytest.fail("empty selection acquired"),
    )
    selected = JRC().select(series=[])
    inspection = selected.inspect()
    assert not (inspection.series or inspection.releases or inspection.requirements)
    assert selected.read().table.num_rows == 0


@pytest.mark.parametrize(
    ("provider", "variable_id"),
    [(JRC(), "jrc:efas:flood_depth"), (EDO(), "edo:soil_moisture_index")],
)
def test_empty_climate_selection_validates_known_variable_science(
    provider: JRC | EDO, variable_id: str, tmp_path: Path
) -> None:
    runtime = Runtime(cache=Cache.offline(tmp_path))
    valid = provider.select(series=[], variables=variable_id)
    assert not valid.inspect(runtime=runtime).series
    assert valid.read(runtime=runtime).table.num_rows == 0

    for variable, match in (
        ("typo", "unknown requested identity"),
        (
            Variable(variable_id, measurement=cf.TimeDuration(unit=cf.TimeUnit.DAY)),
            "conflicting scientific definition",
        ),
    ):
        invalid = provider.select(series=[], variables=variable)
        with pytest.raises(ValueError, match=match):
            invalid.inspect(runtime=runtime)
        with pytest.raises(ValueError, match=match):
            invalid.read(runtime=runtime)


def test_edo_reuses_inventory_period_and_streams_before_next_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_io
    from crc_sdk._providers import jrc_edo
    from crc_sdk._source_io import _download

    first = "sminx_m_eul_20010101_20011231_t.nc"
    second = "sminx_m_eul_20020101_20021231_t.nc"
    _netcdf(
        tmp_path / first,
        "sminx",
        [0, 10, 20],
        "1",
        np.full((3, 2, 2), 0.4),
        calendar="proleptic_gregorian",
    )
    index = tmp_path / "index.html"
    index.write_text(
        "".join(f'<a href="{name}">{name}</a>' for name in (first, second))
    )
    root = tmp_path.as_uri() + "/"
    acquired = []
    snapshot = _climate_io.netcdf_window_snapshot

    def view(scope: Any, uri: str, **kwargs: Any) -> Any:
        acquired.append(uri)
        return snapshot(scope, uri, **kwargs)

    def acquire(uri: str, destination: Path) -> int:
        acquired.append(uri)
        return _download(index.as_uri() if uri == root else uri, destination)

    monkeypatch.setattr(jrc_edo, "SMI", replace(jrc_edo.SMI, base_url=root.rstrip("/")))
    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    monkeypatch.setattr(_climate_io, "netcdf_window_snapshot", view)
    inventory = EDO().inspect()
    (reference,) = inventory.series
    assert reference.period == Period.window(
        2001, 2002, calendar=Calendar.PROLEPTIC_GREGORIAN
    )
    assert reference.release is not None and inventory.releases == (reference.release,)
    assert acquired == [root]
    selected = EDO().select(series=reference, area=AREA, time=Period.window(2001, 2001))
    assert selected.inspect().series == (reference,)
    result = selected.read()
    assert result.table["time"].to_pylist() == [
        "2001-01-01",
        "2001-01-11",
        "2001-01-21",
    ]
    assert result.table["period_id"].to_pylist() == [reference.period.id] * 3
    assert root + second not in acquired
    with EDO().select(area=AREA).batches(batch_rows=1) as stream:
        batches = iter(stream)
        assert next(batches)["time"].to_pylist() == ["2001-01-01"]
        assert root + second not in acquired
        assert [b["time"].to_pylist() for b in (next(batches), next(batches))] == [
            ["2001-01-11"],
            ["2001-01-21"],
        ]
        with pytest.raises(FileNotFoundError):
            next(batches)
    assert acquired[-1] == root + second


def test_idf_central_quantiles_and_published_confidence_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    from crc_sdk._providers import eccc_idf
    from crc_sdk._source_io import _download
    from tests.test_eccc_idf import station_text

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("_ON_6158731_STATION.txt", station_text())
    root = tmp_path / "idf"
    root.mkdir()
    archive_path = root / "idf_v3-40_2025-12-5.zip"
    with zipfile.ZipFile(archive_path, "w") as output:
        output.writestr("IDF/ON.zip", inner.getvalue())
    (root / "index.html").write_text('<a href="idf_v3-40_2025-12-5.zip">archive</a>')
    monkeypatch.setattr(eccc_idf, "ROOT", root.as_uri())

    def acquire(uri: str, destination: Path) -> int:
        return _download(
            (root / "index.html").as_uri() if uri == root.as_uri() + "/" else uri,
            destination,
        )

    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    recipe = ECCCIDF(
        province=ECCCIDF.Province.ONTARIO,
        duration=cf.Duration(1, unit=cf.TimeUnit.HOUR),
    )
    cache = tmp_path / "cache"
    result = recipe.read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert result.table.num_rows == 1
    confidence = recipe.confidence_intervals().read(
        runtime=Runtime(cache=Cache.reuse(cache))
    )
    inventory = recipe.inspect(runtime=Runtime(cache=Cache.offline(cache)))
    assert inventory.stations[0].period is not None
    assert inventory.series[0].period is not None
    assert inventory.stations[0].period.start == 1950
    assert inventory.series[0].period.end == 2017
    assert result.metadata.source is not None
    assert inventory.series[0].release == result.metadata.source.release
    assert inventory.series[0].release is not None
    offline = Runtime(cache=Cache.offline(cache))
    declaration = ECCCIDF(duration=cf.Duration(1, unit=cf.TimeUnit.HOUR)).inspect(
        runtime=offline
    )
    assert not declaration.series
    assert [r.parameter for r in declaration.requirements] == ["province"]
    assert [v.id for v in declaration.indicators] == [inventory.series[0].variable.id]
    province = ECCCIDF(province=ECCCIDF.Province.ONTARIO)
    assert len(province.inspect(runtime=offline).series) == 9
    selected = province.select(series=inventory.series[0])
    assert selected.inspect(runtime=offline).series == inventory.series
    assert selected.read(runtime=offline).table.equals(
        result.table, check_metadata=False
    )
    unknown = province.select(variables="unknown")
    with pytest.raises(ValueError, match="unknown requested identity"):
        unknown.inspect(runtime=offline)
    with pytest.raises(ValueError, match="unknown requested identity"):
        unknown.read(runtime=offline)
    outside = selected.select(area=AREA)
    assert not outside.inspect(runtime=offline).series
    assert outside.read(runtime=offline).table.num_rows == 0
    assert (
        recipe.select(series=inventory.series[0])
        .read(runtime=Runtime(cache=Cache.offline(cache)))
        .table.equals(result.table, check_metadata=False)
    )
    monkeypatch.setattr(
        "crc_sdk._source_io._download", lambda *a: pytest.fail("offline acquisition")
    )
    assert (
        recipe.confidence_intervals()
        .read(runtime=Runtime(cache=Cache.offline(cache)))
        .table.equals(confidence.table, check_metadata=False)
    )
    assert confidence.table.num_rows == 6
    first = confidence.table.to_pylist()[0]
    assert first["coordinate"] == {"kind": "return_period", "time": None, "years": 2.0}
    assert first["probability"] is None
    assert first["interval"]["confidence"] == 0.95
    assert first["interval"]["upper"] - first["interval"]["lower"] == pytest.approx(1.0)


def test_idf_station_without_a_duration_estimate_is_a_missing_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    from crc_sdk._providers import eccc_idf
    from crc_sdk._source_io import _download
    from tests.test_eccc_idf import station_text

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("_ON_6158731_STATION.txt", station_text())
        archive.writestr(
            "_ON_6068158_STATION.txt", station_text("6068158", missing=("1h",))
        )
    root = tmp_path / "idf"
    root.mkdir()
    with zipfile.ZipFile(root / "idf_v3-40_2025-12-5.zip", "w") as output:
        output.writestr("IDF/ON.zip", inner.getvalue())
    (root / "index.html").write_text('<a href="idf_v3-40_2025-12-5.zip">archive</a>')
    monkeypatch.setattr(eccc_idf, "ROOT", root.as_uri())
    monkeypatch.setattr(
        "crc_sdk._source_io._download",
        lambda uri, destination: _download(
            (root / "index.html").as_uri() if uri == root.as_uri() + "/" else uri,
            destination,
        ),
    )
    recipe = ECCCIDF(
        province=ECCCIDF.Province.ONTARIO,
        duration=cf.Duration(1, unit=cf.TimeUnit.HOUR),
    )
    runtime = Runtime(cache=Cache.reuse(tmp_path / "cache"))
    result = recipe.read(runtime=runtime)
    rows = {row["support_id"]: row for row in result.table.to_pylist()}
    assert rows["6158731"]["failure"] is None
    assert rows["6068158"]["curve"] is None
    assert rows["6068158"]["failure"]["code"] == "source_missing"
    assert result.report.counts.no_data == 1
    confidence = recipe.confidence_intervals().read(runtime=runtime)
    missing = [r for r in confidence.table.to_pylist() if r["support_id"] == "6068158"]
    assert len(missing) == 6
    assert all(r["value"] is None and r["interval"] is None for r in missing)
    assert confidence.report.counts.no_data == 6


def test_era5_precipitation_midnight_boundary_and_explicit_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module
    from crc_sdk._aggregation import Coverage

    path = tmp_path / "era5.zarr"
    group = _grid(path, variable="total_precipitation", hourly=True)
    array = group["total_precipitation"]
    array[:] = 0.001
    array.attrs["units"] = "m"
    # The final interval ending Jan 1 00:00 belongs to the previous day.
    array[-1, :, :] = 0.01
    zarr.consolidate_metadata(str(path))
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    recipe = ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION).select(
        area=AREA, time=Period.window(2001, 2001)
    )
    assert recipe.read().table["value"].to_pylist() == pytest.approx([33.0])
    array[24, :, :] = np.nan
    result = recipe.read()
    assert result.table["value"].to_pylist() == [None]
    assert result.table["coverage"].to_pylist()[0]["valid"] == 364
    relaxed = (
        ERA5(
            indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION,
            coverage=Coverage.minimum_fraction(0.99),
        )
        .select(area=AREA, time=Period.window(2001, 2001))
        .read()
    )
    assert relaxed.table["value"].to_pylist() == pytest.approx([33.0])
    assert relaxed.report.counts.usable_years == 1


def test_era5_five_day_precipitation_coverage_counts_rolling_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "five-day-rain.zarr"
    group = _grid(path, variable="total_precipitation", hourly=True)
    array = group["total_precipitation"]
    array[:] = 0.001
    array.attrs["units"] = "m"
    zarr.consolidate_metadata(str(path))
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    recipe = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_FIVE_DAY_PRECIPITATION,
        coverage=Coverage.minimum_fraction(0.5),
    ).select(area=AREA, time=Period.window(2001, 2001))
    complete = recipe.read()
    assert complete.table["value"].to_pylist() == pytest.approx([120.0])
    assert complete.table["coverage"].to_pylist() == [
        {"valid": 361, "expected": 361, "minimum_fraction": 0.5}
    ]
    values = np.asarray(array[:])
    for day in range(5, 365, 2):
        values[day * 24 + 1, :, :] = np.nan
    array[:] = values
    result = recipe.read()
    assert result.table["value"].to_pylist() == [None]
    assert result.table["coverage"].to_pylist() == [
        {"valid": 1, "expected": 361, "minimum_fraction": 0.5}
    ]
    assert result.report.counts.incomplete_years == 1
    saved = recipe.write(tmp_path / "five-day.parquet")
    assert (
        saved.read().table["coverage"].to_pylist()
        == result.table["coverage"].to_pylist()
    )


def test_strict_http_netcdf_range_snapshot_owns_only_selected_native_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import urllib.request

    from crc_sdk._climate_io import netcdf_window_snapshot
    from crc_sdk._graph import ExecutionScope

    path = tmp_path / "file.nc"
    _netcdf(path, "tasmax", np.arange(365), "K", np.full((365, 2, 2), 290.0))
    payload = path.read_bytes()
    requests = []

    class Response(io.BytesIO):
        def __init__(
            self, data: bytes, *, status: int, headers: dict[str, str]
        ) -> None:
            super().__init__(data)
            self.status, self.headers = status, headers

    def urlopen(request: urllib.request.Request, **kwargs: Any) -> Response:
        headers = {"Content-Length": str(len(payload)), "ETag": '"test-release"'}
        if request.get_method() == "HEAD":
            return Response(b"", status=200, headers=headers)
        assert request.get_header("If-match") == '"test-release"'
        header = request.get_header("Range")
        assert header is not None
        byte_range = header.removeprefix("bytes=")
        start, end = map(int, byte_range.split("-"))
        requests.append((start, end))
        headers["Content-Range"] = f"bytes {start}-{end}/{len(payload)}"
        return Response(payload[start : end + 1], status=206, headers=headers)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with ExecutionScope(Runtime()) as scope:
        directory = scope.directory
        group, identity = netcdf_window_snapshot(
            scope,
            "https://fixture.test/annual.nc",
            area=AREA,
            variable="tasmax",
            resolution=0.25,
        )
        assert group["tasmax"][:].shape == (365, 1, 1)
        assert identity.uri.endswith(
            "#crc-variable=tasmax&window=-0.01,-0.01,0.01,0.01"
        )
        assert requests
    assert not directory.exists()


def test_era5_payload_pin_rejects_changed_chunks_before_preview_or_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "era5.zarr"
    group = _grid(path, variable="2m_temperature", hourly=True)
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    indicator = ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    time = Period.window(2001, 2001)
    recipe = ERA5(indicator=indicator).select(area=AREA, time=time)
    cache = tmp_path / "cache"
    inventory = recipe.inspect(runtime=Runtime(cache=Cache.reuse(cache)))
    reference = inventory.series[0]
    assert reference.release is not None and inventory.releases == (reference.release,)
    for unknown in (
        ReleaseReference("unavailable", "sha256:" + "0" * 64),
        replace(reference.release, sha256="sha256:" + "0" * 64),
    ):
        with pytest.raises(OSError, match="source release changed"):
            ERA5(indicator=indicator, release=unknown).select(
                area=AREA, time=time
            ).inspect()
    result = (
        ERA5()
        .select(area=AREA, series=reference)
        .read(runtime=Runtime(cache=Cache.reuse(cache)))
    )
    assert result.table["value"].to_pylist() == pytest.approx([6.85])
    assert result.metadata.source is not None
    release = result.metadata.source.release
    assert release is not None
    pinned = ERA5(indicator=indicator, release=release).select(area=AREA, time=time)
    pinned_series = ERA5().select(area=AREA, series=replace(reference, release=release))
    for selected in (pinned, pinned_series):
        replay = selected.read(runtime=Runtime(cache=Cache.offline(cache)))
        assert replay.table.equals(result.table, check_metadata=False)
        assert replay.metadata.source is not None
        assert replay.metadata.source.release == release
    with pytest.raises(OSError, match="release"):
        ERA5(
            indicator=indicator,
            release=ReleaseReference(release.id, "sha256:" + "0" * 64),
        ).select(area=AREA, time=time).read()
    with pytest.raises(ValueError, match="period|series"):
        recipe.select(time=Period.window(2001, 2002), series=[reference]).read()
    metadata = (path / ".zmetadata").read_bytes()
    group["2m_temperature"][0, 0, 0] = 290.0
    assert (path / ".zmetadata").read_bytes() == metadata
    empty = pinned.read(limit=0)
    assert not empty.report.complete and empty.report.valid is None
    assert empty.metadata.source is not None
    assert empty.metadata.source.release == reference.release
    with pytest.raises(OSError, match="release changed"):
        pinned.read(limit=1)
    with pinned_series.batches(runtime=Runtime(cache=Cache.refresh(cache))) as stream:
        with pytest.raises(OSError, match="release changed"):
            next(iter(stream))
    changed = recipe.read(runtime=Runtime(cache=Cache.offline(cache)))
    assert changed.table["value"].to_pylist() == pytest.approx([16.85])


def test_era5_native_hourly_measurements_missing_data_and_bounded_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "hourly.zarr"
    group = _grid(path, variable="2m_temperature", hourly=True, days=2)
    group["2m_temperature"][3, :, :] = np.nan
    zarr.consolidate_metadata(str(path))
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    source = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    ).select(area=AREA, time=Period.window(2001, 2001))
    raw = source.observations()
    spans = []
    from crc_sdk._climate_grid import grid_values

    original = grid_values

    def bounded(*args: Any, **kwargs: Any) -> Any:
        spans.append(kwargs["stop"] - kwargs["start"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "grid_values", bounded)
    cache = tmp_path / "cache"
    preview = raw.read(limit=1, runtime=Runtime(batch_rows=1, cache=Cache.reuse(cache)))
    assert spans == [1]
    assert preview.metadata.source is not None
    assert any(
        "/2m_temperature/" in item.uri for item in preview.metadata.source.resources
    )
    result = raw.read(runtime=Runtime(batch_rows=7, cache=Cache.reuse(cache)))
    assert max(spans) <= 7
    assert result.table.num_rows == 49
    assert result.table["value"].to_pylist()[0] == 280.0
    assert result.table["value"].to_pylist()[3] is None
    assert result.report.counts.no_data == 1
    assert result.metadata.kind is EvidenceKind.OBSERVATIONS
    assert result.metadata.sampling is not None
    assert result.metadata.sampling.step == cf.Duration(1, unit=cf.TimeUnit.HOUR)
    assert result.metadata.spec.variables[0].measurement == cf.Temperature(
        unit=cf.TemperatureUnit.KELVIN
    )
    assert not any(
        isinstance(record, TemporalReduction) for record in result.metadata.provenance
    )
    monkeypatch.setattr(
        "crc_sdk._source_io._download", lambda *args: pytest.fail("offline source read")
    )
    assert raw.read(runtime=Runtime(cache=Cache.offline(cache))).table.equals(
        result.table, check_metadata=False
    )
    saved = raw.write(
        tmp_path / "raw.parquet", runtime=Runtime(cache=Cache.offline(cache))
    )
    assert type(saved) is Observations
    assert saved.read().table.num_rows == 49


def test_era5_greenwich_window_reads_only_selected_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "greenwich.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    _array(group, "latitude", [51.75, 51.5, 51.25], ["latitude"])
    _array(group, "longitude", np.arange(1440) / 4, ["longitude"])
    _array(
        group,
        "time",
        np.arange(24),
        ["time"],
        units="hours since 2001-01-01 00:00:00",
    )
    temperature = 273.15 + np.arange(3)[None, :, None] + np.arange(1440)[None, None, :]
    _array(
        group,
        "2m_temperature",
        np.broadcast_to(temperature, (24, 3, 1440)),
        ["time", "latitude", "longitude"],
        units="K",
    )
    group.attrs.update(valid_time_start="2001-01-01", valid_time_stop="2001-12-31")
    zarr.consolidate_metadata(str(path))
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )

    reads: list[tuple[slice, ...]] = []
    original = zarr.Array.__getitem__

    def read(array: zarr.Array[Any], selection: Any) -> Any:
        if array.name.endswith("/2m_temperature"):
            reads.append(selection)
        return original(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", read)
    recipe = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE,
        coverage=Coverage.minimum_fraction(1 / 365),
    ).select(
        area=Bounds(west=-0.5, south=51.25, east=0.5, north=51.75),
        time=Period.window(2001, 2001),
    )
    columns = [0, 1, 2, 1438, 1439]
    expected = [row + column for row in range(3) for column in columns]
    annual = recipe.read(runtime=Runtime(memory_limit=10_000))
    assert annual.table["value"].to_pylist() == pytest.approx(expected)
    assert reads == [
        (slice(0, 24), slice(0, 3), slice(0, 3)),
        (slice(0, 24), slice(0, 3), slice(1438, 1440)),
    ]

    reads.clear()
    with recipe.observations().batches(runtime=Runtime(memory_limit=500)) as stream:
        raw = [value for batch in stream for value in batch.column("value").to_pylist()]
    assert raw == pytest.approx([273.15 + value for value in expected] * 24)
    assert reads == [
        (slice(start, start + 4), slice(0, 3), run)
        for start in range(0, 24, 4)
        for run in (slice(0, 3), slice(1438, 1440))
    ]


def test_native_era5_precipitation_retains_interval_end_and_composes_reductions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "rain.zarr"
    group = _grid(path, variable="total_precipitation", hourly=True)
    group["total_precipitation"][:] = 0.001
    group["total_precipitation"].attrs["units"] = "m"
    zarr.consolidate_metadata(str(path))
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    raw = (
        ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION)
        .select(area=AREA, time=Period.window(2001, 2001))
        .observations()
    )
    result = raw.read()
    assert result.table.num_rows == 365 * 24
    assert result.table["time"].to_pylist()[0] == "2001-01-01T01:00:00"
    assert result.table["time"].to_pylist()[-1] == "2002-01-01T00:00:00"
    assert result.metadata.sampling is not None
    assert result.metadata.sampling.timestamp is TimestampPosition.END
    assert result.metadata.spec.variables[0].measurement == cf.PrecipitationAmount(
        unit=cf.LengthUnit.METRE,
        accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.HOUR),
    )
    annual = raw.aggregate(DailyTotal()).aggregate(AnnualMaximum()).read()
    assert annual.table["value"].to_pylist() == pytest.approx([24.0])
    assert annual.table["coverage"].to_pylist()[0]["valid"] == 365
    assert result.metadata.source is not None and result.metadata.source.release
    assert annual.metadata.source is not None
    assert annual.metadata.source.release == result.metadata.source.release
    assert [item.release for item in annual.report.source_identities] == [
        item.release for item in result.report.source_identities
    ]
    assert annual.report.source_identities[-1] == annual.metadata.source
    late = (
        ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION)
        .observations()
        .aggregate(DailyTotal())
        .aggregate(AnnualMaximum())
        .select(area=AREA, time=Period.window(2001, 2001))
        .read()
    )
    assert late.table.equals(annual.table, check_metadata=False)
    transformed = (
        ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION)
        .observations()
        .convert_units(
            variable="era5:total_precipitation", unit=cf.LengthUnit.MILLIMETRE
        )
        .rename(variables={"era5:total_precipitation": "company:rainfall"})
        .aggregate(DailyTotal())
        .aggregate(AnnualMaximum())
        .select(area=AREA, time=Period.window(2001, 2001))
        .read()
    )
    assert transformed.table["variable_id"].to_pylist() == ["company:rainfall"]
    assert transformed.table["value"].to_pylist() == pytest.approx([24.0])


def test_cmip_native_precipitation_has_typed_rate_and_explicit_source_unit_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module
    from crc_sdk._providers.cmip6 import CATALOGUE_COLUMNS

    path = tmp_path / "rain.zarr"
    group = _grid(path, variable="pr", days=365)
    group["pr"][:] = 1 / 86400
    group["pr"].attrs["units"] = "kg m-2 s-1"
    zarr.consolidate_metadata(str(path))
    catalogue = tmp_path / "catalogue.csv"
    with catalogue.open("w") as output:
        writer = csv.writer(output)
        writer.writerow(CATALOGUE_COLUMNS)
        writer.writerow(
            [
                "CMIP",
                "TEST",
                "MODEL",
                "historical",
                "r1",
                "day",
                "pr",
                "gn",
                path.as_uri(),
                "20260101",
            ]
        )
    monkeypatch.setattr(module, "CMIP6_CATALOGUE", catalogue.as_uri())
    raw = (
        CMIP6(
            indicator=CMIP6.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION,
            model="MODEL",
            scenario="historical",
            member="r1",
        )
        .select(area=AREA)
        .observations()
    )
    result = raw.read()
    assert result.table["value"].to_pylist() == pytest.approx([1.0] * 365)
    measurement = result.metadata.spec.variables[0].measurement
    assert isinstance(measurement, cf.PrecipitationRate)
    assert measurement.unit is cf.PrecipitationRateUnit.MILLIMETRES_PER_DAY
    conversion = next(
        item for item in result.metadata.provenance if isinstance(item, UnitConversion)
    )
    assert conversion.source_unit == "kg m-2 s-1"
    assert conversion.scale == 86400.0
    assert not any(
        isinstance(item, TemporalReduction) for item in result.metadata.provenance
    )
    assert raw.aggregate(DailyTotal()).aggregate(AnnualMaximum()).read().table[
        "value"
    ].to_pylist() == [1.0]


def test_native_source_validation_is_typed_but_user_and_io_errors_stay_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "bad.zarr"
    group = _grid(path, variable="2m_temperature", hourly=True, days=1)
    group["2m_temperature"].attrs["units"] = "degC"
    zarr.consolidate_metadata(str(path))
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    raw = (
        ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
        .observations()
        .select(area=AREA, time=Period.window(2001, 2001))
    )
    report = raw.validate()
    assert report.valid is False
    assert any("actual source unit" in item.message for item in report.diagnostics)
    with pytest.raises(ValidationError):
        raw.read()
    with pytest.raises(ValueError, match="calendar contradicts"):
        raw.select(time=Period.window(2001, 2001, calendar=Calendar.NO_LEAP)).validate()
    monkeypatch.setattr(
        module,
        "ERA5_STORE",
        replace(module.ERA5_STORE, url="https://fixture.test/missing.zarr"),
    )
    with pytest.raises(FileNotFoundError):
        raw.validate(runtime=Runtime(cache=Cache.offline(tmp_path / "empty-cache")))


def test_current_grid_decoder_retains_cf_packing_nodata_and_dateline_footprints(
    tmp_path: Path,
) -> None:
    from shapely import from_wkb  # type: ignore[import-untyped]

    from crc_sdk._climate_grid import GridReadPlan, grid_cells, grid_values

    path = tmp_path / "packed.zarr"
    group = _grid(path, variable="tasmax", days=2)
    group["lon"][:] = [0.0, 359.75]
    _array(group, "lon_bnds", [[-0.125, 0.125], [359.625, 359.875]], ["lon", "bounds"])
    group["lon"].attrs["bounds"] = "lon_bnds"
    group["tasmax"][:] = 20.0
    group["tasmax"][1, 0, 0] = -9999.0
    group["tasmax"][1, 0, 1] = -8888.0
    group["tasmax"].attrs.update(
        _FillValue=-9999.0,
        missing_value=[-8888.0, -7777.0],
        scale_factor=0.1,
        add_offset=273.15,
    )
    cells = grid_cells(
        group,
        lat_name="lat",
        lon_name="lon",
        area=Bounds(west=-0.4, south=-0.01, east=0.1, north=0.01),
    )
    assert len(cells) == 2
    assert {cell.longitude for cell in cells} == {0.0, -0.25}
    assert all(
        from_wkb(cell.geometry).bounds[2] - from_wkb(cell.geometry).bounds[0] < 1
        for cell in cells
    )
    values = grid_values(
        group["tasmax"],
        time_name="time",
        lat_name="lat",
        lon_name="lon",
        start=0,
        stop=2,
        plan=GridReadPlan.from_cells(cells),
    )
    assert values[0].tolist() == pytest.approx([275.15, 275.15])
    assert np.isnan(values[1]).all()


@pytest.mark.parametrize(
    ("chunks", "memory_limit"),
    [((1, 7, 1440), None), ((5, 3, 4), None), ((2, 7, 1440), 4 * 1440 * 8 * 7)],
)
def test_grid_reads_match_direct_indexing_for_any_chunk_layout(
    tmp_path: Path, chunks: tuple[int, int, int], memory_limit: int | None
) -> None:
    from crc_sdk._climate_grid import GridCell, GridReadPlan, grid_values

    rng = np.random.default_rng(7)
    source = rng.random((11, 7, 1440))
    group = zarr.open_group(tmp_path / "grid.zarr", mode="w", zarr_format=3)
    array = group.create_array("value", shape=source.shape, chunks=chunks, dtype="f8")
    array[:] = source
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
    columns = [0, 1, 2, 1438, 1439, 700, 701, 703, 2]
    cells = tuple(
        GridCell(int(rng.integers(0, 7)), column, 0.0, 0.0, b"", 0)
        for column in columns
    )
    values = grid_values(
        array,
        time_name="time",
        lat_name="lat",
        lon_name="lon",
        start=2,
        stop=11,
        plan=GridReadPlan.from_cells(cells),
        memory_limit=memory_limit,
    )
    expected = source[2:11][:, [c.row for c in cells], [c.column for c in cells]]
    assert np.array_equal(values, expected)


def test_grid_reads_decode_each_full_width_chunk_once(tmp_path: Path) -> None:
    from crc_sdk._climate_grid import GridCell, GridReadPlan, grid_values

    group = zarr.open_group(tmp_path / "global.zarr", mode="w", zarr_format=3)
    array = group.create_array(
        "value", shape=(4, 9, 1440), chunks=(1, 9, 1440), dtype="f8"
    )
    array[:] = 1.0
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
    reads: list[object] = []
    original = type(array).__getitem__

    class Counted:
        attrs = array.attrs
        chunks = array.chunks

        def __getitem__(self, key: Any) -> Any:
            reads.append(key)
            return original(array, key)

    cells = tuple(GridCell(4, column, 0.0, 0.0, b"", 0) for column in range(0, 400, 2))
    plan = GridReadPlan.from_cells(cells)
    assert len(plan.windows) == 200
    grid_values(
        Counted(),
        time_name="time",
        lat_name="lat",
        lon_name="lon",
        start=0,
        stop=4,
        plan=plan,
    )
    assert len(reads) == 1


def test_offline_native_chunk_cache_miss_does_not_become_a_source_fill_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "source.zarr"
    _grid(path, variable="2m_temperature", hourly=True, days=1)
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    raw = (
        ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
        .observations()
        .select(area=AREA, time=Period.window(2001, 2001))
    )
    cache = tmp_path / "cache"
    raw.inspect(runtime=Runtime(cache=Cache.reuse(cache)))
    monkeypatch.setattr(
        "crc_sdk._source_io._download", lambda *args: pytest.fail("offline acquisition")
    )
    with pytest.raises(FileNotFoundError, match="offline cache"):
        raw.read(runtime=Runtime(cache=Cache.offline(cache)))


def _nex_calendar_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calendars: dict[int, tuple[str, int]],
) -> None:
    import urllib.parse

    from crc_sdk import _climate_sampling as module
    from crc_sdk._climate_io import netcdf_header_snapshot, netcdf_window_snapshot

    directory = module.NEX_PREFIX + "MODEL/historical/r1/tasmax/"
    paths: dict[str, Path] = {}
    for year, (label, days) in calendars.items():
        name = f"tasmax_day_MODEL_historical_r1_gn_{year}.nc"
        values = np.full((days, 2, 2), 280.0)
        values[days - 1] = 280.0 + year - 2000
        _netcdf(
            tmp_path / name,
            "tasmax",
            np.arange(days),
            "K",
            values,
            time_units=f"days since {year}-01-01",
            calendar=label,
        )
        paths[name] = tmp_path / name

    def acquire(uri: str, destination: Path) -> int:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(uri).query)
        assert query["prefix"][0] == directory
        destination.write_text(
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + "".join(f"<Contents><Key>{directory + n}</Key></Contents>" for n in paths)
            + "<IsTruncated>false</IsTruncated></ListBucketResult>"
        )
        return 200

    monkeypatch.setattr(module, "NEX_BUCKET", "https://fixture.test")
    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    monkeypatch.setattr(
        module,
        "netcdf_header_snapshot",
        lambda scope, uri, variable: netcdf_header_snapshot(
            scope, paths[uri.rsplit("/", 1)[-1]].as_uri(), variable
        ),
    )
    monkeypatch.setattr(
        module,
        "netcdf_window_snapshot",
        lambda scope, uri, **kw: netcdf_window_snapshot(
            scope, paths[uri.rsplit("/", 1)[-1]].as_uri(), **kw
        ),
    )


@pytest.mark.parametrize(
    ("label", "days", "calendar"),
    [("365_day", 365, Calendar.NO_LEAP), ("360_day", 360, Calendar.DAYS_360)],
)
def test_nex_reads_models_on_their_native_calendar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    days: int,
    calendar: Calendar,
) -> None:
    _nex_calendar_fixture(
        tmp_path, monkeypatch, {2001: (label, days), 2002: (label, days)}
    )
    recipe = NEX(
        indicator=NEX.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE,
        model="MODEL",
        scenario="historical",
        member="r1",
    ).select(area=AREA)
    result = recipe.read()
    assert result.table["value"].to_pylist() == pytest.approx([7.85, 8.85])
    (series,) = result.metadata.spec.series
    assert series.period is not None and series.period.calendar is calendar
    assert result.metadata.sampling is not None
    assert result.metadata.sampling.calendar is calendar
    daily = recipe.observations().read()
    assert daily.table.num_rows == 2 * days
    assert daily.metadata.sampling is not None
    assert daily.metadata.sampling.calendar is calendar


def test_nex_files_of_one_series_must_share_a_calendar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _nex_calendar_fixture(
        tmp_path, monkeypatch, {2001: ("365_day", 365), 2002: ("standard", 365)}
    )
    recipe = NEX(
        indicator=NEX.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE,
        model="MODEL",
        scenario="historical",
        member="r1",
    ).select(area=AREA)
    with pytest.raises(ValidationError, match="2002 uses the gregorian calendar"):
        recipe.read()
