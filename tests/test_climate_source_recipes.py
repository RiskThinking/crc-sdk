from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import crc_framework as cf
import numpy as np
import pytest
import zarr

from crc_sdk import Bounds, Cache, Calendar, Period, Runtime, ValidationError
from crc_sdk._aggregation import (
    AnnualMaximum,
    DailyTotal,
    TemporalReduction,
    TimestampPosition,
)
from crc_sdk._climate_sampling import CMIP6, ERA5, NEX
from crc_sdk._climate_sources import ECCCIDF, EDO, JRC
from crc_sdk._dataset_schema import UnitConversion
from crc_sdk._errors import CapabilityError
from crc_sdk._evidence import EvidenceKind
from crc_sdk._evidence_recipes import Observations
from crc_sdk._inspection import Capability


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
    monkeypatch.setattr(module, "CMIP6_CATALOGUE", catalogue.as_uri())
    recipe = CMIP6(
        indicator=CMIP6.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE,
        model="MODEL",
        scenario="historical",
        member="r1i1p1f1",
    ).select(area=AREA)
    inventory = recipe.inspect()
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


def _netcdf(
    path: Path,
    variable: str,
    times: Any,
    units: str,
    values: Any,
    *,
    time_units: str = "days since 2001-01-01",
    calendar: str = "standard",
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
        file.attrs["license"] = "CC-BY-SA-4.0"


@pytest.mark.parametrize(
    "variable,unit,source_value,annual_value,native_value",
    [("tasmax", "K", 290.0, 16.85, 290.0), ("pr", "kg m-2 s-1", 1 / 86400, 1.0, 1.0)],
)
def test_nex_real_native_netcdf_view_and_offline_replay(
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

    path = tmp_path / "annual.nc"
    _netcdf(path, variable, np.arange(365), unit, np.full((365, 2, 2), source_value))
    prefix = module.NEX_PREFIX
    keys = {
        prefix: [prefix + "MODEL/"],
        prefix + "MODEL/": [prefix + "MODEL/historical/"],
        prefix + "MODEL/historical/": [prefix + "MODEL/historical/r1/"],
        prefix + f"MODEL/historical/r1/{variable}/": [
            prefix
            + f"MODEL/historical/r1/{variable}/"
            + f"{variable}_day_MODEL_historical_r1_gn_2001.nc"
        ],
    }

    def acquire(uri: str, destination: Path) -> int:
        requested = urllib.parse.parse_qs(urllib.parse.urlsplit(uri).query)["prefix"][0]
        contents = "".join(
            (
                f"<CommonPrefixes><Prefix>{v}</Prefix></CommonPrefixes>"
                if v.endswith("/")
                else f"<Contents><Key>{v}</Key></Contents>"
            )
            for v in keys[requested]
        )
        destination.write_text(
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + contents
            + "<IsTruncated>false</IsTruncated></ListBucketResult>"
        )
        return 200

    monkeypatch.setattr(
        module,
        "netcdf_header_snapshot",
        lambda scope, uri, variable: netcdf_header_snapshot(
            scope, path.as_uri(), variable
        ),
    )
    monkeypatch.setattr(module, "NEX_BUCKET", "https://fixture.test")
    monkeypatch.setattr("crc_sdk._source_io._download", acquire)
    monkeypatch.setattr(
        module,
        "netcdf_window_snapshot",
        lambda scope, uri, **kw: netcdf_window_snapshot(scope, path.as_uri(), **kw),
    )
    recipe = NEX(
        indicator=NEX.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION
        if variable == "pr"
        else NEX.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE,
        model="MODEL",
        scenario="historical",
        member="r1",
    ).select(area=AREA)
    cache = tmp_path / "cache"
    result = recipe.read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert result.table["value"].to_pylist() == pytest.approx([annual_value])
    assert result.metadata.source is not None
    assert any("#crc-variable=" in r.uri for r in result.metadata.source.resources)
    raw = recipe.observations()
    native = raw.read(runtime=Runtime(cache=Cache.reuse(cache), batch_rows=31))
    assert native.table.num_rows == 365
    assert native.table["value"].to_pylist() == pytest.approx([native_value] * 365)
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
    assert recipe.read(runtime=Runtime(cache=Cache.offline(cache))).table.equals(
        result.table, check_metadata=False
    )
    assert raw.read(runtime=Runtime(cache=Cache.offline(cache))).table.equals(
        native.table, check_metadata=False
    )


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
    monkeypatch.setattr(
        "crc_sdk._source_io._download", lambda *a: pytest.fail("offline acquisition")
    )
    assert recipe.read(runtime=Runtime(cache=Cache.offline(cache))).table.equals(
        result.table, check_metadata=False
    )
    assert result.table["time"].to_pylist() == [
        "2001-01-01",
        "2001-01-11",
        "2001-01-21",
    ]
    assert result.table["value"].to_pylist() == pytest.approx([0.4] * 3)
    assert result.metadata.sampling is None
    assert result.metadata.spec.series[0].probability is None


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
    cache = tmp_path / "cache"
    result = recipe.read(runtime=Runtime(cache=Cache.reuse(cache)))
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
    assert recipe.read(runtime=Runtime(cache=Cache.reuse(cache))).table.num_rows == 1
    confidence = recipe.confidence_intervals().read(
        runtime=Runtime(cache=Cache.reuse(cache))
    )
    inventory = recipe.inspect(runtime=Runtime(cache=Cache.offline(cache)))
    assert inventory.stations[0].period is not None
    assert inventory.series[0].period is not None
    assert inventory.stations[0].period.start == 1950
    assert inventory.series[0].period.end == 2017
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


def test_provider_release_ref_and_time_ref_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk import ReleaseReference
    from crc_sdk import _climate_sampling as module

    path = tmp_path / "era5.zarr"
    _grid(path, variable="2m_temperature", hourly=True)
    monkeypatch.setattr(
        module, "ERA5_STORE", replace(module.ERA5_STORE, url=path.as_uri())
    )
    recipe = ERA5(
        indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    ).select(area=AREA, time=Period.window(2001, 2001))
    reference = recipe.inspect().series[0]
    assert reference.release is not None
    with pytest.raises(OSError, match="release"):
        ERA5(
            indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE,
            release=ReleaseReference(reference.release.id, "sha256:" + "0" * 64),
        ).select(area=AREA, time=Period.window(2001, 2001)).read()
    with pytest.raises(ValueError, match="period|series"):
        recipe.select(time=Period.window(2001, 2002), series=[reference]).read()


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

    from crc_sdk._climate_grid import grid_cells, grid_values

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
        cells=cells,
    )
    assert values[0].tolist() == pytest.approx([275.15, 275.15])
    assert np.isnan(values[1]).all()


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
