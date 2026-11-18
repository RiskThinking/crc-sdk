"""Offline native-provider acquisition, scientific meaning, and shared execution."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast
from urllib.parse import parse_qs, urlsplit

import crc_framework as cf
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from shapely import from_wkb  # type: ignore[import-untyped]
from shapely.geometry import box  # type: ignore[import-untyped]

from crc_sdk import (
    EvidenceMetadata,
    FitFailure,
    FittingDiagnostic,
    ReadResult,
    Samples,
    UnavailableStationYear,
    ValidationError,
    _layer_sources,
    _source_io,
)
from crc_sdk._climate_io import raster_window_snapshot as acquire_raster_window
from crc_sdk._dataset_schema import ResourceIdentity
from crc_sdk._declarations import Calendar, Period
from crc_sdk._evidence import EvidenceKind, validate_evidence
from crc_sdk._graph import ExecutionScope
from crc_sdk._inspection import Bounds, Capability
from crc_sdk._layer_sources import CHIRPS, FieldsOfTheWorld
from crc_sdk._observed_sources import (
    HYDAT,
    ISD,
    USGS,
    ECCCDaily,
    GHCNh,
    IBTrACS,
    NOAASeaLevel,
    NOAAWaterLevels,
)
from crc_sdk._runtime import Cache, Runtime


def _json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False).encode()


def _feature(
    identifier: str, properties: dict[str, object], *, geometry: bool = True
) -> dict[str, object]:
    return {
        "type": "Feature",
        "id": identifier,
        "geometry": {"type": "Point", "coordinates": [-79.4, 43.7]}
        if geometry
        else None,
        "properties": properties,
    }


def _collection(features: list[dict[str, object]]) -> bytes:
    return _json(
        {
            "type": "FeatureCollection",
            "features": features,
            "numberReturned": len(features),
            "numberMatched": len(features),
            "links": [],
        }
    )


class NativeTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.absent_year: int | None = None
        self.field = self._fields()
        self.raster: bytes | None = None

    @staticmethod
    def _fields() -> bytes:
        target = io.BytesIO()
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "id": "field-1",
                        "collection": "ftw-s2-2025",
                        "geometry": box(-79.41, 43.69, -79.39, 43.71).wkb,
                        "metrics:area": 356.0,
                        "metrics:perimeter": 78.0,
                        "score": 92,
                        "determination:datetime": "2025-01-01T00:00:00Z",
                    }
                ]
            ),
            target,
        )
        return target.getvalue()

    @staticmethod
    def _raster() -> bytes:
        rasterio = pytest.importorskip("rasterio")
        import numpy as np
        from rasterio.io import MemoryFile  # type: ignore[import-untyped]

        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                height=1,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=rasterio.transform.from_origin(-79.5, 43.8, 0.1, 0.1),
                nodata=-9999,
            ) as out:
                out.write(np.array([[12.0, -9999.0]], dtype="float32"), 1)
            return cast(bytes, memory.read())

    def __call__(self, uri: str, path: Path) -> int:
        self.calls.append(uri)
        u = urlsplit(uri)
        q = parse_qs(u.query)
        data: bytes
        if u.path.endswith("ghcnh-station-list.csv"):
            data = (
                b"GHCN_ID,NAME,LATITUDE,LONGITUDE,ELEVATION\n"
                b"TEST0000001,Test,43.7,-79.4,100\n"
            )
        elif u.path.endswith("isd-history.csv"):
            data = (
                b'"USAF","WBAN","STATION NAME","LAT","LON","BEGIN","END"\n'
                b'"123456","78901","Test","43.7","-79.4","20000101","20251231"\n'
            )
        elif u.path.endswith("parquet/GHCNh_TEST0000001_2020.parquet"):
            target = io.BytesIO()
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "STATION": "TEST0000001",
                            "DATE": "2020-01-01T00:00:00",
                            "LATITUDE": 43.7,
                            "LONGITUDE": -79.4,
                            "temperature": 10.0,
                            "temperature_Quality_Code": "1",
                            "temperature_Source_Code": "220",
                        },
                        {
                            "STATION": "TEST0000001",
                            "DATE": "2020-01-01T01:00:00",
                            "LATITUDE": 43.7,
                            "LONGITUDE": -79.4,
                            "temperature": 50.0,
                            "temperature_Quality_Code": "5",
                            "temperature_Source_Code": "220",
                        },
                    ]
                ),
                target,
            )
            data = target.getvalue()
        elif u.path.endswith("/12345678901.csv"):
            year = int(u.path.split("/")[-2])
            if year == self.absent_year:
                path.write_bytes(b"NoSuchKey")
                return 404
            data = (
                "STATION,DATE,LATITUDE,LONGITUDE,TMP\n12345678901,"
                + str(year)
                + '-01-01T00:00:00,43.7,-79.4,"+0010,1"\n12345678901,'
                + str(year)
                + '-01-01T01:00:00,43.7,-79.4,"+9999,9"\n'
            ).encode()
        elif "monitoring-locations" in u.path:
            data = _collection(
                [
                    _feature(
                        "USGS-01234567",
                        {
                            "monitoring_location_id": "USGS-01234567",
                            "monitoring_location_name": "River",
                        },
                    )
                ]
            )
        elif u.path.endswith("/daily/items"):
            data = _collection(
                [
                    _feature(
                        "flow-1",
                        {
                            "monitoring_location_id": "USGS-01234567",
                            "parameter_code": "00060",
                            "statistic_id": "00003",
                            "time": "2020-01-01",
                            "unit_of_measure": "ft^3/s",
                            "value": 3.5,
                        },
                        geometry=False,
                    )
                ]
            )
        elif "hydrometric-stations" in u.path:
            data = _collection(
                [
                    _feature(
                        "gauge", {"STATION_NUMBER": "02AA001", "STATION_NAME": "Gauge"}
                    )
                ]
            )
        elif "hydrometric-daily-mean" in u.path:
            data = _collection(
                [
                    _feature(
                        "gauge-day",
                        {
                            "STATION_NUMBER": "02AA001",
                            "DATE": "2020-01-01",
                            "DISCHARGE": 3.5,
                            "LEVEL": None,
                        },
                    )
                ]
            )
        elif "climate-stations" in u.path:
            data = _collection(
                [
                    _feature(
                        "climate",
                        {"CLIMATE_IDENTIFIER": "6158355", "STATION_NAME": "Climate"},
                    )
                ]
            )
        elif "climate-daily" in u.path:
            data = _collection(
                [
                    _feature(
                        "climate-day",
                        {
                            "CLIMATE_IDENTIFIER": "6158355",
                            "LOCAL_DATE": "2020-01-01 00:00:00",
                            "LOCAL_YEAR": 2020,
                            "LOCAL_MONTH": 1,
                            "LOCAL_DAY": 1,
                            "MEAN_TEMPERATURE": 2.0,
                            "TOTAL_PRECIPITATION": 5.0,
                        },
                    )
                ]
            )
        elif u.path.endswith("stations.json"):
            data = _json(
                {
                    "stations": [
                        {
                            "id": "8518750",
                            "name": "The Battery",
                            "lng": -74.0,
                            "lat": 40.7,
                        }
                    ]
                }
            )
        elif u.path.endswith("slr_projections.json"):
            data = _json(
                {
                    "count": 1,
                    "totalPages": 1,
                    "currPage": 1,
                    "SlrProjections": [
                        {
                            "stationId": "8518750",
                            "reportYear": 2022,
                            "units": "cm",
                            "projectionYear": 2050,
                            "scenario": "Intermediate",
                            "longitude": -74.0,
                            "latitude": 40.7,
                            "projectionRsl": 30.0,
                            "projectionCiLow": 20.0,
                            "projectionCiHigh": 40.0,
                        }
                    ],
                }
            )
        elif u.path.endswith("product.json"):
            if "station" not in q:
                data = _json(
                    {
                        "ExtremeWaterLevels": [
                            {
                                "stationId": "8518750",
                                "stationName": "The Battery",
                                "longitude": -74.0,
                                "latitude": 40.7,
                                "epoch": "1983-2001",
                                "annuals": True,
                                "exceedanceLevels": True,
                            }
                        ]
                    }
                )
            elif "details" in q:
                data = _json(
                    {
                        "stationId": "8518750",
                        "units": "meters",
                        "datum": "MHHW",
                        "GEV_HI": {
                            "exceedanceLevels": [
                                {
                                    "probability": 0.1,
                                    "returnPeriod": 10,
                                    "level": 1.5,
                                    "lower90CI": 1.2,
                                    "upper90CI": 1.8,
                                },
                                {
                                    "probability": 0.01,
                                    "returnPeriod": 100,
                                    "level": 2.5,
                                    "lower90CI": 2.1,
                                    "upper90CI": 3.0,
                                },
                            ]
                        },
                    }
                )
            else:
                data = _json(
                    {
                        "stationId": "8518750",
                        "units": "meters",
                        "extremes": [{"year": 2020, "value": 0.5}],
                    }
                )
        elif "ibtracs.ALL.list" in u.path:
            data = (
                b"SID,ISO_TIME,LAT,LON,USA_WIND,USA_PRES\n"
                b" , ,degrees_north,degrees_east,kts,mb\n"
                b"2020001N,2020-01-01 00:00:00,20,181,80,980\n"
                b"2020001N,2020-01-01 06:00:00,21,182,85,975\n"
            )
        elif u.path.endswith(".cog"):
            if self.raster is None:
                self.raster = self._raster()
            data = self.raster
        elif u.path.endswith("/vector/catalog.json"):
            data = _json(
                {"links": [{"rel": "child", "href": "./2025/collection.json"}]}
            )
        elif u.path.endswith("2025/collection.json"):
            data = _json(
                {
                    "id": "ftw-s2-2025",
                    "links": [{"rel": "item", "href": "./utm17/utm17.json"}],
                }
            )
        elif u.path.endswith("utm17/utm17.json"):
            data = _json(
                {
                    "collection": "ftw-s2-2025",
                    "bbox": [-79.5, 43.5, -79.0, 44.0],
                    "properties": {"proj:code": "EPSG:4326"},
                    "assets": {
                        "data": {
                            "href": "../zone=17/utm17.parquet",
                            "file:checksum": "1220"
                            + hashlib.sha256(self.field).hexdigest(),
                        }
                    },
                }
            )
        elif u.path.endswith("utm17.parquet"):
            data = self.field
        else:
            raise AssertionError(f"Unplanned acquisition: {uri}")
        path.write_bytes(data)
        return 200


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> NativeTransport:
    result = NativeTransport()
    monkeypatch.setattr(_source_io, "_download", result)

    def local_raster_window(
        scope: ExecutionScope,
        uri: str,
        *,
        area: Bounds,
        license: str | None = None,
        attribution: str | None = None,
    ) -> tuple[Path, ResourceIdentity]:
        # Exercise the real range/window cache against native fixture bytes. GDAL
        # reads a local source; the identity truthfully names that fixture object.
        path = tmp_path / Path(urlsplit(uri).path).name
        if not path.exists():
            if result.raster is None:
                result.raster = result._raster()
            path.write_bytes(result.raster)
        return acquire_raster_window(
            scope, path.as_uri(), area=area, license=license, attribution=attribution
        )

    monkeypatch.setattr(
        _layer_sources, "raster_window_snapshot", local_raster_window, raising=False
    )
    return result


def _year() -> Period:
    return Period.window(2020, 2020, calendar=Calendar.GREGORIAN)


@pytest.mark.parametrize(
    "recipe",
    [
        GHCNh(),
        ISD(),
        USGS(),
        HYDAT(),
        ECCCDaily(),
        NOAASeaLevel(),
        NOAAWaterLevels(),
        IBTrACS(),
        CHIRPS(),
        FieldsOfTheWorld(),
    ],
)
def test_source_construction_selection_explanation_are_pure(
    recipe: GHCNh
    | ISD
    | USGS
    | HYDAT
    | ECCCDaily
    | NOAASeaLevel
    | NOAAWaterLevels
    | IBTrACS
    | CHIRPS
    | FieldsOfTheWorld,
    transport: NativeTransport,
) -> None:
    recipe.explain()
    recipe.select(variables=[]).explain()
    assert transport.calls == []


@pytest.mark.parametrize(
    "recipe",
    [
        GHCNh(),
        ISD(),
        USGS(),
        HYDAT(),
        ECCCDaily(),
        NOAASeaLevel(),
        NOAAWaterLevels(),
        IBTrACS(),
        CHIRPS(),
        FieldsOfTheWorld(),
    ],
)
def test_empty_source_selection_has_stable_valid_native_schema(
    recipe: GHCNh
    | ISD
    | USGS
    | HYDAT
    | ECCCDaily
    | NOAASeaLevel
    | NOAAWaterLevels
    | IBTrACS
    | CHIRPS
    | FieldsOfTheWorld,
    transport: NativeTransport,
) -> None:
    result = recipe.select(variables=[]).read()
    assert result.table.num_rows == 0
    assert result.report.complete and result.report.valid
    assert transport.calls == []
    assert validate_evidence(result.table, result.metadata).complete


@pytest.mark.parametrize(
    "source,station,values",
    [
        (USGS(), "USGS-01234567", [3.5]),
        (HYDAT(), "02AA001", [3.5, None]),
        (ECCCDaily(), "6158355", [2.0, 5.0]),
    ],
)
def test_gauge_native_records_are_readable_before_modelling(
    source: USGS | HYDAT | ECCCDaily,
    station: str,
    values: list[float | None],
    transport: NativeTransport,
) -> None:
    result = source.select(stations=station, time=_year()).read()
    assert result.table["value"].to_pylist() == values
    assert result.table["station_id"].to_pylist() == [station] * len(values)
    assert all(
        value.startswith(station + "@")
        for value in result.table["support_id"].to_pylist()
    )
    assert result.metadata.kind is EvidenceKind.OBSERVATIONS
    assert result.metadata.source is not None
    assert len(result.metadata.source.resources) == 2
    assert all(
        r.retrieved_at and r.license and r.attribution
        for r in result.metadata.source.resources
    )
    assert result.report.complete and result.report.valid


@pytest.mark.parametrize(
    "source,station", [(GHCNh(), "TEST0000001"), (ISD(), "12345678901")]
)
def test_hourly_qc_evidence_and_incomplete_predefined_indicator(
    source: GHCNh | ISD, station: str, transport: NativeTransport
) -> None:
    hourly = source.select(stations=station, time=_year()).observations().read()
    assert hourly.table.num_rows == 2
    assert hourly.table["value"].to_pylist()[-1] is None
    configured = (
        GHCNh(indicator=GHCNh.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
        if isinstance(source, GHCNh)
        else ISD(indicator=ISD.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
    )
    annual = configured.select(stations=station, time=_year()).read()
    assert annual.metadata.kind is EvidenceKind.SAMPLES
    assert annual.table["value"].to_pylist() == [None]
    assert annual.table["failure"].to_pylist()[0]["code"] == "incomplete_year"
    assert annual.report.counts.incomplete_years == 1


def test_native_404_is_cached_absence_with_station_year_diagnostic(
    transport: NativeTransport, tmp_path: Path
) -> None:
    transport.absent_year = 2020
    recipe = ISD(
        indicator=ISD.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    ).select(stations="12345678901", time=_year())
    first = recipe.read(runtime=Runtime(cache=Cache.reuse(tmp_path)))
    calls = len(transport.calls)
    second = recipe.read(runtime=Runtime(cache=Cache.offline(tmp_path)))
    assert len(transport.calls) == calls
    assert first.table.num_rows == second.table.num_rows == 0
    assert first.report.counts.requested == first.report.counts.incomplete_years == 1
    missing = first.report.diagnostics[0]
    assert isinstance(missing, UnavailableStationYear)
    assert missing.station_id == "12345678901" and missing.year == 2020
    assert missing.minimum_fraction == 1 and missing.support_id is None
    assert first.report.diagnostics == second.report.diagnostics
    assert first.metadata.source is not None and second.metadata.source is not None
    assert first.metadata.source.release == second.metadata.source.release


def test_station_pin_tracks_absence_and_ignores_a_volatile_404_body(
    transport: NativeTransport, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport.absent_year = 2020
    requests = 0

    def download(uri: str, path: Path) -> int:
        nonlocal requests
        status = transport(uri, path)
        if status == 404:
            requests += 1
            path.write_text(f"<Error><RequestId>{requests}</RequestId></Error>")
        return status

    monkeypatch.setattr(_source_io, "_download", download)
    indicator = ISD.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    station = "12345678901"
    initial = ISD(indicator=indicator).select(stations=station, time=_year()).read()
    assert initial.metadata.source is not None and initial.metadata.source.release
    pinned = ISD(indicator=indicator, release=initial.metadata.source.release).select(
        stations=station, time=_year()
    )
    refresh = Runtime(cache=Cache.refresh(tmp_path))
    assert pinned.read(runtime=refresh).report.counts.incomplete_years == 1
    assert requests == 2
    transport.absent_year = None
    with pytest.raises(OSError, match="source release changed"):
        pinned.read(runtime=refresh)


@pytest.mark.parametrize("input_kind", ["recipe", "collected", "reopened"])
def test_unlocated_station_year_follows_each_fit_policy(
    input_kind: str, transport: NativeTransport, tmp_path: Path
) -> None:
    transport.absent_year = 2020
    recipe = ISD(
        indicator=ISD.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    ).select(stations="12345678901", time=_year())
    evidence: Samples | ReadResult[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]]
    if input_kind == "collected":
        evidence = recipe.read()
    elif input_kind == "reopened":
        saved = recipe.write(tmp_path / "annual.parquet")
        assert saved.path is not None
        evidence = Samples.open(saved.path)
    else:
        evidence = recipe
    spec = cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT)
    calls_before_fitting = len(transport.calls)
    with evidence.fit(spec).execute() as execution:
        with pytest.raises(
            ValidationError, match="no usable annual evidence"
        ) as failure:
            execution.read()
        failed_report = failure.value.report
        assert not failed_report.valid
        assert execution.validate() == failed_report
    for policy in (FitFailure.KEEP_NO_DATA, FitFailure.SKIP):
        fitted = evidence.fit(spec, on_failure=policy).read()
        assert fitted.report.valid and fitted.report.complete
        assert fitted.table.num_rows == fitted.report.counts.no_data == 0
        assert fitted.report.counts.skipped == int(policy is FitFailure.SKIP)
        assert fitted.report.counts.incomplete_years == 1
        assert fitted.report.counts.requested == 1
        assert fitted.report.counts.usable_years == 0
        assert failed_report.counts == replace(fitted.report.counts, skipped=0)
        for report in (failed_report, fitted.report):
            missing, diagnostic = report.diagnostics
            assert isinstance(missing, UnavailableStationYear)
            assert isinstance(diagnostic, FittingDiagnostic)
            assert diagnostic.support_id is None
            assert (
                diagnostic.record.evidence_count
                == diagnostic.record.excluded_count
                == 1
            )
            assert diagnostic.record.used_count == 0 and not diagnostic.record.converged
    if input_kind != "recipe":
        assert len(transport.calls) == calls_before_fitting


def test_collected_station_year_remains_selectable_before_fitting(
    transport: NativeTransport, tmp_path: Path
) -> None:
    transport.absent_year = 2020
    annual = ISD(
        indicator=ISD.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
    ).select(stations="12345678901", time=_year())
    path = annual.write(tmp_path / "annual.parquet").path
    assert path is not None
    samples = Samples.open(path)
    spec = cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT)
    result = (
        samples.select(stations="12345678901")
        .fit(spec, on_failure=FitFailure.SKIP)
        .read()
    )
    assert result.report.counts.requested == result.report.counts.skipped == 1
    empty = samples.select(stations=[]).fit(spec).read()
    assert empty.report.complete and empty.report.valid
    assert empty.report.counts.requested == empty.report.counts.incomplete_years == 0
    assert not empty.report.diagnostics


def test_noaa_projection_bounds_are_separate_variables_without_invented_quantiles(
    transport: NativeTransport,
) -> None:
    result = NOAASeaLevel().select(stations="8518750").read()
    assert result.metadata.kind is EvidenceKind.PROJECTIONS
    assert result.table["value"].to_pylist() == [30, 20, 40]
    assert result.table["probability"].null_count == 3
    assert (
        result.table["coordinate"].to_pylist()
        == [{"kind": "time", "time": "2050", "years": None}] * 3
    )
    assert result.table["interval"].null_count == 3
    assert all(
        s.probability is not None and s.probability.kind.value == "projection"
        for s in result.metadata.spec.series
    )


def test_noaa_annual_observations_central_curves_and_confidence_evidence_stay_distinct(
    transport: NativeTransport,
) -> None:
    source = NOAAWaterLevels().select(stations="8518750")
    observed = source.read()
    assert observed.table["value"].to_pylist() == [0.5]
    assert observed.metadata.spec.variables[0].measurement is not None
    assert (
        observed.metadata.spec.variables[0].measurement.quantity
        == "water_level_deviation_from_trend"
    )
    intervals = source.confidence_intervals().read()
    assert intervals.table["probability"].null_count == 2
    assert intervals.table["interval"].to_pylist()[0] == {
        "confidence": 0.9,
        "lower": 1.2,
        "upper": 1.8,
    }
    central = source.levels().read()
    from crc_sdk._schema.dataset import table_to_rows

    curve = table_to_rows(central.table)[0].curve
    assert curve is not None
    np.testing.assert_equal(curve.quantile([0.9, 0.99]), [1.5, 2.5])


def test_ibtracs_preserves_track_points_grouping_agency_and_knots(
    transport: NativeTransport,
) -> None:
    result = IBTrACS(agency=IBTrACS.Agency.USA).select(time=_year()).read()
    assert result.table.num_rows == 4
    assert result.table["group_id"].to_pylist() == ["2020001N"] * 4
    assert len(set(result.table["support_id"].to_pylist())) == 2
    assert all(
        from_wkb(value).geom_type == "Point"
        for value in result.table["geometry"].to_pylist()
    )
    assert result.metadata.source is not None
    assert result.metadata.source.member == "USA"
    assert result.metadata.spec.variables[0].measurement is not None
    assert result.metadata.spec.variables[0].measurement.unit == "knots"
    assert Capability.FIT not in IBTrACS().inspect().capabilities


def test_chirps_reads_monthly_precipitation_actual_pixel_support_and_nodata(
    transport: NativeTransport,
) -> None:
    result = (
        CHIRPS()
        .select(
            area=Bounds(west=-79.5, south=43.7, east=-79.3, north=43.8), time=_year()
        )
        .read()
    )
    assert result.table.num_rows == 24
    assert result.table["value"].to_pylist() == [12.0, None] * 12
    assert result.table["time"].to_pylist()[:4] == [
        "2020-01-01",
        "2020-01-01",
        "2020-02-01",
        "2020-02-01",
    ]
    assert all(
        from_wkb(value).geom_type == "Polygon"
        for value in result.table["geometry"].to_pylist()
    )
    assert isinstance(
        result.metadata.spec.variables[0].measurement, cf.PrecipitationAmount
    )
    assert (
        result.metadata.spec.variables[0].measurement.accumulation_duration.unit.value
        == "month"
    )


def test_field_boundaries_metrics_and_published_checksum(
    transport: NativeTransport, tmp_path: Path
) -> None:
    recipe = FieldsOfTheWorld().select(
        area=Bounds(west=-79.5, south=43.5, east=-79.0, north=44),
        periods=Period.window(2025, 2025),
    )
    first = recipe.read(runtime=Runtime(cache=Cache.reuse(tmp_path)))
    assert first.table["value"].to_pylist() == [None, 356.0, 78.0, 92.0]
    assert first.table["label"].to_pylist() == ["field-1"] * 4
    assert first.table["failure"].null_count == 4
    assert first.metadata.kind is EvidenceKind.LAYER
    assert first.metadata.source is not None
    resource = first.metadata.source.resources[-1]
    assert resource.sha256 == "sha256:" + hashlib.sha256(transport.field).hexdigest()
    calls = len(transport.calls)
    second = (
        FieldsOfTheWorld(release=first.metadata.source.release)
        .select(
            area=Bounds(west=-79.5, south=43.5, east=-79.0, north=44),
            periods=Period.window(2025, 2025),
        )
        .read(runtime=Runtime(cache=Cache.offline(tmp_path)))
    )
    assert second.table.equals(first.table)
    assert len(transport.calls) == calls


def test_inspection_exposes_typed_inventory_and_does_no_station_modelling(
    transport: NativeTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("inspection attempted a temporal reduction")

    monkeypatch.setattr("crc_sdk._observed_sources.complete_daily_extrema", forbidden)
    inventory = GHCNh().inspect()
    assert inventory.stations[0].id == "TEST0000001"
    assert inventory.stations[0].longitude == -79.4
    assert {r.parameter for r in inventory.requirements} == {
        "indicator",
        "stations",
        "time",
        "release",
    }
    assert len(inventory.indicators) == 2
    assert all(s.probability is not None for s in inventory.series)
    assert inventory.source is not None and inventory.source.release is None
    assert inventory.releases == ()
    assert len(transport.calls) == 1
    configured = (
        GHCNh(indicator=GHCNh.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
        .select(stations=inventory.stations[0], time=_year())
        .inspect()
    )
    assert [r.code for r in configured.requirements] == ["snapshot_resolved_on_read"]
    assert configured.releases == ()
    assert len(transport.calls) == 2


@pytest.mark.parametrize("provider", [GHCNh, ISD])
def test_inspected_series_selects_station_indicator_and_native_years(
    transport: NativeTransport, provider: type[GHCNh] | type[ISD], tmp_path: Path
) -> None:
    inventory = provider().inspect()
    station = inventory.stations[0]
    recipe = provider().select(
        series=inventory.series[0],
        stations=station,
        time=Period.window(2020, 2020),
    )
    configured = recipe.inspect()
    reference = configured.series[0]
    assert reference.period == _year()
    direct = recipe.read()
    assert direct.metadata.source is not None and direct.metadata.source.release
    assert direct.table["time"].to_pylist() == ["2020"]
    saved = recipe.write(tmp_path / "annual.parquet")
    pinned = saved.inspect().series[0]
    assert pinned.release == direct.metadata.source.release
    replay = provider().select(series=pinned, stations=station).read()
    assert replay.table.equals(direct.table, check_metadata=False)
    assert replay.metadata.source is not None
    assert replay.metadata.source.release == pinned.release


@pytest.mark.parametrize("provider", [GHCNh, ISD])
def test_station_reference_period_survives_narrower_observation_window(
    transport: NativeTransport, provider: type[GHCNh] | type[ISD]
) -> None:
    inventory = provider().inspect()
    station = inventory.stations[0]
    dated = (
        provider()
        .select(
            series=inventory.series[0], stations=station, time=Period.window(2019, 2020)
        )
        .inspect()
        .series[0]
    )
    narrowed = provider().select(series=dated, stations=station, time=_year())
    assert narrowed.inspect().series[0].period == dated.period
    result = narrowed.read()
    assert result.table["time"].to_pylist() == ["2020"]
    assert result.metadata.spec.series[0].period == dated.period
    assert all("2019" not in uri for uri in transport.calls)


def test_unknown_static_variables_fail_before_any_source_acquisition(
    transport: NativeTransport,
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        USGS().select(
            variables="misspelled", stations="USGS-01234567", time=_year()
        ).read()
    assert transport.calls == []


def test_unknown_station_ids_fail_using_catalogue_without_downloading_observations(
    transport: NativeTransport,
) -> None:
    with pytest.raises(ValueError, match="Unknown isd station"):
        ISD().select(stations="absent", time=_year()).observations().read()
    assert len(transport.calls) == 1
    assert transport.calls[0].endswith("isd-history.csv")


def test_repeated_reads_inside_execution_resolve_sources_once_and_keep_owned_results(
    transport: NativeTransport,
) -> None:
    recipe = USGS().select(stations="USGS-01234567", time=_year())
    with recipe.execute() as execution:
        first = execution.read()
        second = execution.read()
        assert first.table.equals(second.table)
        assert len(transport.calls) == 2
    assert first.table["value"].to_pylist() == [3.5]
    assert first.report.complete and first.report.valid


def test_verified_native_snapshot_replays_offline_and_refresh_detects_pin_changes(
    transport: NativeTransport, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = USGS().select(stations="USGS-01234567", time=_year())
    first = recipe.read(runtime=Runtime(cache=Cache.reuse(tmp_path)))
    assert first.metadata.source is not None
    pin = first.metadata.source.release
    replay = (
        USGS(release=pin)
        .select(stations="USGS-01234567", time=_year())
        .read(runtime=Runtime(cache=Cache.offline(tmp_path)))
    )
    assert replay.table.equals(first.table)
    assert len(transport.calls) == 2
    download = transport.__call__

    def changed(uri: str, path: Path) -> int:
        status = download(uri, path)
        if urlsplit(uri).path.endswith("/daily/items"):
            payload = json.loads(path.read_text())
            payload["features"][0]["properties"]["value"] = 8.0
            path.write_bytes(_json(payload))
        return status

    monkeypatch.setattr(_source_io, "_download", changed)
    with pytest.raises(OSError, match="source release changed"):
        USGS(release=pin).select(stations="USGS-01234567", time=_year()).read(
            runtime=Runtime(cache=Cache.refresh(tmp_path))
        )
    assert first.table["value"].to_pylist() == [3.5]


def test_native_api_pagination_refuses_cross_endpoint_and_truncated_responses(
    transport: NativeTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = transport.__call__

    def hostile(uri: str, path: Path) -> int:
        status = download(uri, path)
        if urlsplit(uri).path.endswith("/daily/items"):
            payload = json.loads(path.read_text())
            payload["links"] = [{"rel": "next", "href": "https://other.example/data"}]
            path.write_bytes(_json(payload))
        return status

    monkeypatch.setattr(_source_io, "_download", hostile)
    with pytest.raises(ValueError, match="pagination changed"):
        USGS().select(stations="USGS-01234567", time=_year()).read()

    def truncated(uri: str, path: Path) -> int:
        status = download(uri, path)
        if urlsplit(uri).path.endswith("/daily/items"):
            payload = json.loads(path.read_text())
            payload["numberMatched"] = 2
            path.write_bytes(_json(payload))
        return status

    monkeypatch.setattr(_source_io, "_download", truncated)
    with pytest.raises(ValueError, match="incomplete selection"):
        USGS().select(stations="USGS-01234567", time=_year()).read()


@pytest.mark.parametrize("maximum", [True, False])
def test_relaxed_annual_coverage_requires_complete_daily_hours(
    transport: NativeTransport, monkeypatch: pytest.MonkeyPatch, maximum: bool
) -> None:
    from crc_sdk._aggregation import Coverage

    download = transport.__call__

    def hourly(uri: str, path: Path) -> int:
        if urlsplit(uri).path.endswith("/12345678901.csv"):
            content = "STATION,DATE,LATITUDE,LONGITUDE,TMP\n"
            for hour in range(24):
                content += (
                    f"12345678901,2020-01-01T{hour:02d}:00:00,"
                    f'43.7,-79.4,"+{hour * 10:04d},1"\n'
                )
            # Many valid reports in the same hour cannot create a complete day.
            for minute in range(60):
                extreme = "+0600" if maximum else "-0900"
                content += (
                    f"12345678901,2020-01-02T00:{minute:02d}:00,"
                    f'43.7,-79.4,"{extreme},1"\n'
                )
            path.write_text(content)
            return 200
        return download(uri, path)

    monkeypatch.setattr(_source_io, "_download", hourly)
    indicator = (
        ISD.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE
        if maximum
        else ISD.Indicator.ANNUAL_MINIMUM_DAILY_MINIMUM_TEMPERATURE
    )
    result = (
        ISD(indicator=indicator, coverage=Coverage.minimum_fraction(1 / 366))
        .select(stations="12345678901", time=_year())
        .read()
    )
    assert result.table["value"].to_pylist() == [23.0 if maximum else 0.0]
    assert result.table["coverage"].to_pylist() == [
        {"valid": 1, "expected": 366, "minimum_fraction": 1 / 366}
    ]
    assert result.report.counts.usable_years == 1
    assert result.report.counts.incomplete_years == 0
    assert len(result.metadata.provenance) >= 2


def test_noaa_rounded_probabilities_remain_distinct_from_confidence(
    transport: NativeTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = transport.__call__

    def rounded(uri: str, path: Path) -> int:
        status = download(uri, path)
        if "details=exceedancelevels" in uri:
            payload = json.loads(path.read_text())
            payload["GEV_HI"]["exceedanceLevels"][0].update(
                probability=0.013, returnPeriod=80.0, level=1.5
            )
            path.write_bytes(_json(payload))
        return status

    monkeypatch.setattr(_source_io, "_download", rounded)
    result = NOAAWaterLevels().select(stations="8518750").levels().read()
    from crc_sdk._schema.dataset import table_to_rows

    curve = table_to_rows(result.table)[0].curve
    assert curve is not None
    assert curve.quantile(0.987) == 1.5


def test_field_data_corruption_fails_the_published_asset_checksum(
    transport: NativeTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = transport.__call__

    def damaged(uri: str, path: Path) -> int:
        status = download(uri, path)
        if uri.endswith("utm17.parquet"):
            path.write_bytes(path.read_bytes() + b"bad")
        return status

    monkeypatch.setattr(_source_io, "_download", damaged)
    with pytest.raises(OSError, match="content identity mismatch"):
        FieldsOfTheWorld().select(
            area=Bounds(west=-79.5, south=43.5, east=-79, north=44)
        ).read()


def test_mutable_api_inspect_select_read_does_not_pin_volatile_response_timestamps(
    transport: NativeTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = transport.__call__
    tick = 0

    def changing_timestamp(uri: str, path: Path) -> int:
        nonlocal tick
        status = download(uri, path)
        if urlsplit(uri).path.endswith("/daily/items"):
            payload = json.loads(path.read_text())
            tick += 1
            payload["timeStamp"] = f"2026-01-01T00:00:{tick:02d}Z"
            path.write_bytes(_json(payload))
        return status

    monkeypatch.setattr(_source_io, "_download", changing_timestamp)
    recipe = USGS().select(stations="USGS-01234567", time=_year())
    inventory = recipe.inspect()
    assert inventory.series[0].release is None
    assert inventory.releases == ()
    assert [r.code for r in inventory.requirements] == ["snapshot_resolved_on_read"]
    result = recipe.select(series=inventory.series[0]).read()
    assert result.table["value"].to_pylist() == [3.5]
    assert (
        result.metadata.source is not None
        and result.metadata.source.release is not None
    )
    assert tick == 2


def test_bulk_evidence_inspection_stays_metadata_only(
    transport: NativeTransport,
) -> None:
    weather = (
        ISD(indicator=ISD.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
        .select(stations="12345678901", time=_year())
        .inspect()
    )
    assert weather.series and weather.releases == ()
    assert len(transport.calls) == 1
    assert transport.calls[0].endswith("isd-history.csv")
    IBTrACS().inspect()
    CHIRPS().select(
        area=Bounds(west=-79.5, south=43.5, east=-79, north=44), time=_year()
    ).inspect()
    assert len(transport.calls) == 1


def test_field_inspection_references_pin_published_assets_without_bulk_read(
    transport: NativeTransport,
) -> None:
    recipe = FieldsOfTheWorld().select(
        area=Bounds(west=-79.5, south=43.5, east=-79, north=44)
    )
    inventory = recipe.inspect()
    assert inventory.releases and inventory.series[0].release == inventory.releases[0]
    assert not any(uri.endswith(".parquet") for uri in transport.calls)
    result = recipe.select(series=inventory.series[0]).read()
    assert result.table.num_rows == 1
    assert result.metadata.source is not None
    assert result.metadata.source.release == inventory.releases[0]


@pytest.mark.parametrize(
    "source",
    [
        USGS(),
        GHCNh(),
        ISD(),
        IBTrACS(),
        NOAASeaLevel(),
        NOAAWaterLevels(),
        CHIRPS(),
        FieldsOfTheWorld(),
    ],
)
def test_empty_series_selection_avoids_source_io_and_keeps_inspection_schema(
    source: USGS
    | GHCNh
    | ISD
    | IBTrACS
    | NOAASeaLevel
    | NOAAWaterLevels
    | CHIRPS
    | FieldsOfTheWorld,
    transport: NativeTransport,
) -> None:
    recipe = source.select(series=[])
    inventory = recipe.inspect()
    result = recipe.read()
    assert result.table.num_rows == 0
    assert inventory.schema is not None
    assert inventory.requirements == ()
    assert transport.calls == []


@pytest.mark.parametrize(
    "source,station", [(GHCNh(), "TEST0000001"), (ISD(), "12345678901")]
)
def test_hourly_accessors_keep_source_selection_when_further_narrowed(
    source: GHCNh | ISD, station: str, transport: NativeTransport
) -> None:
    hourly = source.select(stations=station, time=_year()).observations()
    variable = hourly.inspect().series[0].variable
    result = hourly.select(variables=variable).read()
    assert result.table.num_rows == 2
    assert result.table["station_id"].to_pylist() == [station] * 2


def test_published_level_accessors_keep_station_selection_when_further_narrowed(
    transport: NativeTransport,
) -> None:
    levels = NOAAWaterLevels().select(stations="8518750").levels()
    variable = levels.inspect().series[0].variable
    assert levels.select(variables=variable).read().table.num_rows == 1
    intervals = NOAAWaterLevels().select(stations="8518750").confidence_intervals()
    assert (
        intervals.select(variables=intervals.inspect().series[0].variable)
        .read()
        .table.num_rows
        == 2
    )
