"""Explicit station, observation, and projection sources on the shared engine."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Literal, TypeVar, cast
from urllib.parse import urlencode, urljoin, urlsplit

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]

from ._aggregation import (
    AnnualMaximum,
    AnnualMinimum,
    Coverage,
    Sampling,
    aggregate_table,
)
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    ReleaseReference,
    ResourceIdentity,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
    UnavailableStationYear,
)
from ._declarations import (
    Calendar,
    DataSpec,
    Period,
    PeriodKind,
    Probability,
    Scenario,
    SeriesSpec,
    Variable,
)
from ._errors import CapabilityError
from ._evidence import (
    EvidenceKind,
    EvidenceMetadata,
    evidence_arrow_schema,
    evidence_table,
)
from ._evidence_recipes import Observations, Projections
from ._graph import ExecutionScope, Node, NodeStream
from ._inspection import (
    Capability,
    IndicatorReference,
    Inspection,
    Operation,
    Requirement,
    SeriesReference,
    StationReference,
)
from ._observed_decoders import (
    ECCC_FIELDS,
    array,
    eccc_rows,
    feature_collection,
    feature_coordinates,
    ghcnh_rows,
    hydat_rows,
    ibtracs_rows,
    isd_rows,
    noaa_annual_rows,
    noaa_projection_rows,
    number,
    obj,
    point_identity,
    text,
    usgs_rows,
)
from ._recipes import Dataset, Samples, _key
from ._schema.dataset import DatasetRow, rows_to_table
from ._selection import SelectionState, rename_metadata, selected_metadata
from ._source_io import (
    ReleasePins,
    SourceNode,
    check_release,
    fetch,
    fetch_json,
    fetch_optional,
    inventory_payload_pin,
    inventory_release,
    native_data,
    payload_selection,
    release_snapshot,
    selected_series_period,
    selected_source_time,
)
from ._source_io import (
    _json as decode_json,
)
from ._station_reduction import complete_daily_extrema

_NOAA_PRODUCT = "https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product.json"
_NOAA_SLR = "https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product/slr_projections.json"
_NOAA_STATIONS = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels"
_USGS_DAILY = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items"
_USGS_STATIONS = (
    "https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations/items"
)
_GEOMET = "https://api.weather.gc.ca/collections/"
_IBTRACS = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
_IBTRACS_ATTRIBUTION = "NOAA NCEI IBTrACS v04r01, doi:10.25921/82ty-9e16"
_TEMPERATURE = "noaa:hourly_air_temperature"
_ANNUAL_MAXIMUM = "annual_maximum_daily_maximum_temperature"
_ANNUAL_MINIMUM = "annual_minimum_daily_minimum_temperature"
_NOAA_TERMS = "NOAA open-use terms; originator terms may apply"
_NOAA_ATTRIBUTION = (
    "NOAA National Centers for Environmental Information and "
    "NOAA CO-OPS; original source records retained."
)
_NCEI_ATTRIBUTION = (
    "NOAA National Centers for Environmental Information; "
    "original source records retained."
)


MetadataT = TypeVar("MetadataT")


def _init(recipe: object, node: Node[MetadataT], description: str) -> None:
    for name, value in (
        ("_base", node),
        ("_selection", SelectionState()),
        ("_operations", (Operation(Stage.ACQUISITION, description),)),
        ("_path", None),
        ("_metadata", None),
    ):
        object.__setattr__(recipe, name, value)


def _release(value: ReleaseReference | None) -> None:
    if value is not None and not isinstance(value, ReleaseReference):
        raise TypeError("release must be ReleaseReference or None")


def _empty_state(state: SelectionState) -> bool:
    return any(
        value == ()
        for value in (
            state.variables,
            state.stations,
            state.series,
            state.scenarios,
            state.periods,
        )
    )


def _temperature_indicators() -> tuple[IndicatorReference, ...]:
    return (
        IndicatorReference(
            _ANNUAL_MAXIMUM,
            "Annual maximum of daily maximum air temperature",
            cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
            Probability.annual(tail=cf.Tail.UPPER),
        ),
        IndicatorReference(
            _ANNUAL_MINIMUM,
            "Annual minimum of daily minimum air temperature",
            cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
            Probability.annual(tail=cf.Tail.LOWER),
        ),
    )


def _indicator(value: Enum | IndicatorReference | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (Enum, IndicatorReference)):
        raise TypeError("indicator must be a provider Indicator or IndicatorReference")
    candidates = _temperature_indicators()
    identifier = value.value if isinstance(value, Enum) else value.id
    selected = next((item for item in candidates if item.id == identifier), None)
    if selected is None:
        raise ValueError(
            "indicator must be one of the inspected station temperature indicators"
        )
    if isinstance(value, IndicatorReference) and (
        (value.measurement is not None and value.measurement != selected.measurement)
        or (value.probability is not None and value.probability != selected.probability)
    ):
        raise ValueError(
            "indicator scientific definition contradicts the published recipe"
        )
    return cast(str, identifier)


def _variables(
    provider: str,
    *,
    indicator: str | None = None,
    agency: str | None = None,
) -> tuple[Variable, ...]:
    if provider in ("ghcnh", "isd"):
        if indicator is not None:
            return (
                Variable(
                    f"noaa:{provider}:{indicator}",
                    measurement=cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
                    label=next(
                        item.label
                        for item in _temperature_indicators()
                        if item.id == indicator
                    ),
                ),
            )
        return (
            Variable(
                _TEMPERATURE,
                measurement=cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
                label="Hourly air temperature",
            ),
        )
    if provider == "usgs":
        return (
            Variable(
                "usgs:daily_mean_discharge",
                measurement=cf.Measurement(
                    quantity="volumetric_flow_rate", unit="ft^3/s"
                ),
                label="Daily mean gauge discharge",
            ),
        )
    if provider == "hydat":
        return (
            Variable(
                "eccc:daily_mean_discharge",
                measurement=cf.Measurement(
                    quantity="volumetric_flow_rate", unit="m3/s"
                ),
                label="Daily mean gauge discharge",
            ),
            Variable(
                "eccc:daily_mean_water_level",
                measurement=cf.Measurement(
                    quantity="water_surface_elevation", unit="m"
                ),
                label="Daily mean gauge water level; datum not established",
            ),
        )
    if provider == "eccc":
        measurements: dict[str, cf.Measurement] = {
            "MEAN_TEMPERATURE": cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
            "MIN_TEMPERATURE": cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
            "MAX_TEMPERATURE": cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
            "TOTAL_PRECIPITATION": cf.PrecipitationAmount(
                unit=cf.LengthUnit.MILLIMETRE,
                accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.DAY),
            ),
            "TOTAL_RAIN": cf.PrecipitationAmount(
                unit=cf.LengthUnit.MILLIMETRE,
                accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.DAY),
            ),
            "TOTAL_SNOW": cf.Measurement(quantity="snowfall_depth", unit="cm"),
            "SNOW_ON_GROUND": cf.Measurement(quantity="snow_depth", unit="cm"),
            "SPEED_MAX_GUST": cf.WindSpeed(unit=cf.SpeedUnit.KILOMETRES_PER_HOUR),
        }
        return tuple(
            Variable(
                identifier,
                measurement=measurements[field],
                label=field.replace("_", " ").capitalize(),
            )
            for field, identifier in ECCC_FIELDS.items()
        )
    if provider == "noaa_water":
        return (
            Variable(
                "noaa:detrended_annual_high_water",
                measurement=cf.Measurement(
                    quantity="water_level_deviation_from_trend",
                    unit="m",
                ),
                label="Detrended annual high-water observations",
            ),
        )
    if provider == "noaa_slr":
        return tuple(
            Variable(
                identifier,
                measurement=cf.Measurement(
                    quantity="relative_sea_level_change",
                    unit="cm",
                    fields={
                        "baseline_year": 2005,
                        "report_year": 2022,
                        **({"reported_bound": bound} if bound else {}),
                    },
                ),
                label=label,
            )
            for identifier, bound, label in (
                (
                    "noaa:relative_sea_level_change",
                    None,
                    "Projected relative sea-level change",
                ),
                (
                    "noaa:relative_sea_level_interval_low",
                    "lower",
                    "Reported lower sea-level projection interval",
                ),
                (
                    "noaa:relative_sea_level_interval_high",
                    "upper",
                    "Reported upper sea-level projection interval",
                ),
            )
        )
    if provider == "ibtracs":
        selected = "USA" if agency is None else agency
        return (
            Variable(
                f"ibtracs:{selected}:maximum_sustained_wind",
                measurement=cf.Measurement(
                    quantity="maximum_sustained_wind_speed",
                    unit="knots",
                    fields={
                        "agency": selected,
                        **(
                            {"averaging_duration_seconds": 60}
                            if selected == "USA"
                            else {}
                        ),
                    },
                ),
                label=f"{selected} agency maximum sustained wind",
            ),
            Variable(
                f"ibtracs:{selected}:minimum_central_pressure",
                measurement=cf.Measurement(
                    quantity="air_pressure", unit="hPa", fields={"agency": selected}
                ),
                label=f"{selected} agency minimum central pressure",
            ),
        )
    raise ValueError(f"unknown observation provider {provider!r}")


@dataclass(frozen=True, slots=True)
class _Part:
    path: Path | None
    uri: str
    station: str | None = None
    year: int | None = None


@dataclass(frozen=True, slots=True)
class _Prepared:
    parts: tuple[_Part, ...]
    source: SourceIdentity | None
    stations: tuple[StationReference, ...]
    publication: SourceIdentity | None = None


def _requests(
    scope: ExecutionScope,
    uri: str,
    *,
    license: str | None,
    attribution: str,
    collect: bool = True,
) -> tuple[list[_Part], list[ResourceIdentity], list[dict[str, object]]]:
    with native_data(scope=scope, source="native_api_response"):
        return _request_pages(
            scope, uri, license=license, attribution=attribution, collect=collect
        )


def _request_pages(
    scope: ExecutionScope,
    uri: str,
    *,
    license: str | None,
    attribution: str,
    collect: bool = True,
) -> tuple[list[_Part], list[ResourceIdentity], list[dict[str, object]]]:
    start = urlsplit(uri)
    seen: set[str] = set()
    ids: set[str] = set()
    parts: list[_Part] = []
    resources: list[ResourceIdentity] = []
    records: list[dict[str, object]] = []
    following: str | None = uri
    matched: int | None = None
    record_count = 0
    while following is not None:
        current = urlsplit(following)
        if (current.scheme, current.netloc, current.path) != (
            start.scheme,
            start.netloc,
            start.path,
        ):
            raise ValueError("source pagination changed its endpoint")
        if following in seen:
            raise ValueError("source pagination repeats a URL")
        seen.add(following)
        path, identity = fetch(
            scope, following, license=license, attribution=attribution
        )
        payload = decode_json(path.read_text())
        features, next_link = feature_collection(payload)
        page = obj(payload, "source page")
        total = page.get("numberMatched")
        if type(total) is int:
            if matched is not None and total != matched:
                raise ValueError("source match count changed during pagination")
            matched = total
        for feature in features:
            identifier = text(feature.get("id"), "source feature ID")
            if identifier in ids:
                raise ValueError("source pagination repeats an observation ID")
            ids.add(identifier)
        parts.append(_Part(path, following))
        resources.append(identity)
        record_count += len(features)
        if collect:
            records.extend(features)
        following = urljoin(following, next_link) if next_link is not None else None
    if matched is not None and matched != record_count:
        raise ValueError("source pagination returned an incomplete selection")
    return parts, resources, records


@native_data(source="station_inventory")
def _station_records(provider: str, payload: object) -> tuple[StationReference, ...]:
    result = []
    if provider in ("noaa_water", "noaa_slr"):
        p = obj(payload, "NOAA station inventory")
        values = array(
            p.get("ExtremeWaterLevels")
            if provider == "noaa_water"
            else p.get("stations"),
            "NOAA station inventory",
        )
        for value in values:
            station = obj(value, "NOAA station")
            identifier = text(
                station.get("stationId")
                if provider == "noaa_water"
                else station.get("id"),
                "station ID",
            )
            lon = number(
                station.get("longitude")
                if provider == "noaa_water"
                else station.get("lng"),
                "station longitude",
                missing=True,
            )
            lat = number(
                station.get("latitude")
                if provider == "noaa_water"
                else station.get("lat"),
                "station latitude",
                missing=True,
            )
            label = text(
                station.get("stationName")
                if provider == "noaa_water"
                else station.get("name"),
                "station name",
            )
            result.append(StationReference(identifier, label, lon, lat))
    else:
        features, _ = feature_collection(payload)
        for feature in features:
            p = obj(feature.get("properties"), "station properties")
            if provider == "usgs":
                identifier = text(
                    p.get("monitoring_location_id", feature.get("id")),
                    "USGS station ID",
                )
                label = text(
                    p.get("monitoring_location_name", identifier), "station name"
                )
            else:
                field = (
                    "STATION_NUMBER" if provider == "hydat" else "CLIMATE_IDENTIFIER"
                )
                identifier = text(p.get(field), "ECCC station ID")
                label = text(p.get("STATION_NAME", identifier), "station name")
            lon, lat = (
                feature_coordinates(feature)
                if feature.get("geometry") is not None
                else (None, None)
            )
            result.append(StationReference(identifier, label, lon, lat))
    if len({item.id for item in result}) != len(result):
        raise ValueError("source station inventory contains duplicate IDs")
    return tuple(result)


@native_data(source="noaa_station_inventory")
def _catalogue_csv(provider: str, path: Path) -> tuple[StationReference, ...]:
    result = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if provider == "ghcnh":
                identifier = row["GHCN_ID"]
                label, lon, lat = row["NAME"], row["LONGITUDE"], row["LATITUDE"]
                period = None
            else:
                identifier = row["USAF"] + row["WBAN"]
                label, lon, lat = row["STATION NAME"], row["LON"], row["LAT"]
                period = (
                    None
                    if not row["BEGIN"] or not row["END"]
                    else Period.window(
                        int(row["BEGIN"][:4]),
                        int(row["END"][:4]),
                        calendar=Calendar.GREGORIAN,
                    )
                )
            longitude, latitude = (
                number(lon, "station longitude", missing=True),
                number(lat, "station latitude", missing=True),
            )
            # A half-present or out-of-range pair locates nothing; the published
            # catalogues contain both, and one such row must not reject the rest.
            if (
                longitude is None
                or latitude is None
                or not (-180 <= longitude <= 180 and -90 <= latitude <= 90)
            ):
                longitude, latitude = None, None
            result.append(
                StationReference(
                    identifier, label or identifier, longitude, latitude, period
                )
            )
    if len({item.id for item in result}) != len(result):
        raise ValueError("station catalogue repeats station IDs")
    return tuple(result)


def _catalogue(
    scope: ExecutionScope, provider: str, state: SelectionState
) -> tuple[tuple[StationReference, ...], tuple[ResourceIdentity, ...]]:
    resources: tuple[ResourceIdentity, ...]
    if provider in ("ghcnh", "isd"):
        uri = (
            "https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/doc/ghcnh-station-list.csv"
            if provider == "ghcnh"
            else "https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv"
        )
        path, identity = fetch(
            scope, uri, license=_NOAA_TERMS, attribution=_NCEI_ATTRIBUTION
        )
        stations = _catalogue_csv(provider, path)
        resources = (identity,)
    elif provider in ("noaa_water", "noaa_slr"):
        uri = (
            _NOAA_PRODUCT
            + "?"
            + urlencode({"name": "extremewaterlevels", "units": "metric"})
            if provider == "noaa_water"
            else _NOAA_STATIONS
        )
        payload, identity = fetch_json(
            scope, uri, license=_NOAA_TERMS, attribution=_NOAA_ATTRIBUTION
        )
        stations = _station_records(provider, payload)
        resources = (identity,)
    else:
        endpoint = (
            _USGS_STATIONS
            if provider == "usgs"
            else _GEOMET
            + (
                "hydrometric-stations/items"
                if provider == "hydat"
                else "climate-stations/items"
            )
        )
        params: dict[str, object] = {"f": "json", "limit": 500}
        field = (
            # The monitoring-locations collection filters on `id`; daily values keep
            # `monitoring_location_id`.
            "id"
            if provider == "usgs"
            else "STATION_NUMBER"
            if provider == "hydat"
            else "CLIMATE_IDENTIFIER"
        )
        if state.stations is not None:
            # OGC feature-property filtering is exact; one station per request avoids
            # pretending every service understands an undocumented comma syntax.
            queries = [{**params, field: identifier} for identifier in state.stations]
        else:
            if state.area is not None:
                params["bbox"] = ",".join(
                    str(v)
                    for v in (
                        state.area.west,
                        state.area.south,
                        state.area.east,
                        state.area.north,
                    )
                )
            queries = [params]
        stations_list: list[StationReference] = []
        sources = []
        for parameters in queries:
            _, identities, features = _requests(
                scope,
                endpoint + "?" + urlencode(parameters),
                license="USGS public-domain data"
                if provider == "usgs"
                else "OGL-Canada-2.0",
                attribution="USGS Water Data for the Nation"
                if provider == "usgs"
                else "Environment and Climate Change Canada, GeoMet",
            )
            stations_list.extend(
                _station_records(
                    provider, {"type": "FeatureCollection", "features": features}
                )
            )
            sources.extend(identities)
        stations, resources = tuple(stations_list), tuple(sources)
    return _select_stations(provider, stations, state), resources


def _select_stations(
    provider: str, stations: tuple[StationReference, ...], state: SelectionState
) -> tuple[StationReference, ...]:
    selected = stations
    if state.stations is not None:
        missing = set(state.stations) - {item.id for item in stations}
        if missing:
            raise ValueError(f"Unknown {provider} station IDs: {sorted(missing)}")
        selected = tuple(item for item in selected if item.id in state.stations)
    if state.area is not None:
        area = state.area
        selected = tuple(
            item
            for item in selected
            if item.longitude is not None
            and item.latitude is not None
            and area.south <= item.latitude <= area.north
            and (
                area.west <= item.longitude <= area.east
                if area.west <= area.east
                else item.longitude >= area.west or item.longitude <= area.east
            )
        )
    return selected


def _station_publication(
    scope: ExecutionScope,
    provider: str,
    state: SelectionState,
    *,
    release_id: str,
    requested: ReleasePins,
) -> tuple[tuple[StationReference, ...], tuple[ResourceIdentity, ...]]:
    # OGC discovery can start with the whole station catalogue. Retain that
    # verified inventory when a later series selection adds a station or area.
    may_use_full = (
        _scoped_api_catalogue(provider, state)
        and requested.publication_id == release_id
    )
    try:
        stations, resources = _catalogue(scope, provider, state)
    except FileNotFoundError:
        if not may_use_full:
            raise
    else:
        if (
            not may_use_full
            or requested.inventory is None
            or inventory_release(resources, id=release_id).sha256
            == requested.inventory.sha256
        ):
            return stations, resources
    stations, resources = _catalogue(scope, provider, SelectionState())
    return _select_stations(provider, stations, state), resources


def _scoped_api_catalogue(provider: str, state: SelectionState) -> bool:
    return provider in ("usgs", "hydat", "eccc") and (
        state.stations is not None or state.area is not None
    )


def _station_period(period: Period | None) -> Period | None:
    if (
        period is not None
        and period.kind is PeriodKind.WINDOW
        and period.calendar is None
    ):
        assert period.start is not None and period.end is not None
        return Period.window(period.start, period.end, calendar=Calendar.GREGORIAN)
    return period


class _ObservedNode(SourceNode[EvidenceMetadata]):
    def __init__(
        self,
        provider: str,
        *,
        release: ReleaseReference | None = None,
        indicator: str | None = None,
        coverage: Coverage | None = None,
        daily_coverage: Coverage | None = None,
        agency: str | None = None,
        state: SelectionState = SelectionState(),
        raw: bool = False,
    ) -> None:
        period = state.time
        if provider in ("ghcnh", "isd", "usgs", "hydat", "eccc"):
            time = _station_period(selected_source_time(state))
            period = _station_period(selected_series_period(state)) or time
            state = replace(
                state,
                time=time,
                series=None
                if state.series is None
                else tuple(
                    replace(ref, period=_station_period(ref.period) or time)
                    for ref in state.series
                ),
            )
        if provider in ("ghcnh", "isd") and not raw and indicator is None:
            requested = None
            if state.series is not None and len(state.series) == 1:
                requested = state.series[0].variable.id
            elif state.variables is not None and len(state.variables) == 1:
                variable = state.variables[0]
                requested = variable if isinstance(variable, str) else variable.id
            indicator = next(
                (
                    candidate.id
                    for candidate in _temperature_indicators()
                    if requested == f"noaa:{provider}:{candidate.id}"
                ),
                None,
            )
        self.raw = raw
        self.provider, self.release, self.indicator = provider, release, indicator
        if coverage is not None and not isinstance(coverage, Coverage):
            raise TypeError("coverage must be Coverage or None")
        self.coverage = Coverage.complete() if coverage is None else coverage
        if daily_coverage is not None and not isinstance(daily_coverage, Coverage):
            raise TypeError("daily_coverage must be Coverage or None")
        self.daily_coverage = (
            Coverage.complete() if daily_coverage is None else daily_coverage
        )
        self.agency, self.state = agency, state
        self.period = period
        self.key = _key(
            "native_source",
            provider,
            None if release is None else release._document(),
            indicator,
            self.coverage.fraction,
            agency,
            state.document(),
            raw,
            # Only a relaxed day rule extends the key, so complete-day keys
            # stay as they were.
            *(
                ()
                if self.daily_coverage == Coverage.complete()
                else (("daily_coverage", self.daily_coverage.fraction),)
            ),
        )

    def selected(self, state: SelectionState) -> _ObservedNode:
        return _ObservedNode(
            self.provider,
            release=self.release,
            indicator=self.indicator,
            coverage=self.coverage,
            daily_coverage=self.daily_coverage,
            agency=self.agency,
            state=state,
            raw=self.raw,
        )

    def requirements(self) -> tuple[Requirement, ...]:
        required = []
        empty = _empty_state(self.state)
        if (
            self.provider in ("ghcnh", "isd")
            and self.indicator is None
            and not self.raw
            and not empty
        ):
            required.append(
                Requirement(
                    "indicator_required",
                    (
                        "Select an inspected series or choose an inspected "
                        "IndicatorReference or the provider's Indicator enum."
                    ),
                    "indicator",
                )
            )
        if (
            self.provider not in ("ibtracs",)
            and self.state.stations is None
            and self.state.area is None
        ):
            required.append(
                Requirement(
                    "station_selection_required",
                    (
                        "Choose stations from inspect().stations, or "
                        "select(area=Bounds(...))."
                    ),
                    "stations",
                )
            )
        if (
            self.provider in ("ghcnh", "isd", "usgs", "hydat", "eccc")
            and self.state.time is None
        ):
            required.append(
                Requirement(
                    "observation_time_required",
                    "Select observation years with time=Period.window(start, end).",
                    "time",
                )
            )
        return () if empty else tuple(required)

    def _selection(self) -> tuple[SelectionState, ReleasePins]:
        state, pin = payload_selection(self.state, self.release)
        if self.state.time is not None and self.state.time.calendar not in (
            None,
            Calendar.GREGORIAN,
        ):
            raise ValueError("Native observation times use the Gregorian calendar")
        known_variables = self._declared_variables()
        known = DataSpec(
            known_variables, series=[SeriesSpec(v) for v in known_variables]
        )
        if self.state.variables is not None:
            SelectionState(variables=self.state.variables).resolve(known)
        return state, pin

    def _declared_variables(self) -> tuple[Variable, ...]:
        if (
            self.provider in ("ghcnh", "isd")
            and not self.raw
            and self.indicator is None
        ):
            return tuple(
                v
                for indicator in _temperature_indicators()
                for v in _variables(self.provider, indicator=indicator.id)
            )
        return _variables(self.provider, indicator=self.indicator, agency=self.agency)

    def _variable_ids(self) -> frozenset[str]:
        return frozenset(v.id for v in self._declared_variables())

    @property
    def _release_id(self) -> str:
        return f"{self.provider}:selected-source-snapshot"

    def _discover(
        self, scope: ExecutionScope, requested: ReleasePins
    ) -> tuple[_Prepared, ReleaseReference | None]:
        if _empty_state(self.state):
            return _Prepared((), None, ()), None
        if self.provider == "ibtracs":
            source = SourceIdentity(
                "ibtracs", "ibtracs_v04r01_tracks", member=self.agency
            )
            return _Prepared((), source, ()), requested.exact_only()
        stations, resources = _station_publication(
            scope,
            self.provider,
            self.state,
            release_id=self._release_id,
            requested=requested,
        )
        release = inventory_release(resources, id=self._release_id)
        pin = inventory_payload_pin(release, requested)
        source = SourceIdentity(
            provider=self.provider,
            product={
                "ghcnh": "ghcnh_hourly_temperature",
                "isd": "global_hourly_temperature",
                "usgs": "daily_mean_discharge",
                "hydat": "hydrometric_daily_mean",
                "eccc": "climate_daily",
                "noaa_slr": "sea_level_scenario_projections_2022",
                "noaa_water": "detrended_annual_high_water",
            }[self.provider],
            release=release,
            resources=resources,
            member=self.agency,
        )
        return _Prepared((), source, stations), pin

    def _station_parts(
        self, scope: ExecutionScope, stations: tuple[StationReference, ...]
    ) -> Iterator[tuple[_Part, ResourceIdentity]]:
        time = self.state.time
        assert time is not None and time.start is not None and time.end is not None
        for station in stations:
            for year in range(time.start, time.end + 1):
                uri = (
                    f"https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/access/by-year/{year}/parquet/GHCNh_{station.id}_{year}.parquet"
                    if self.provider == "ghcnh"
                    else f"https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station.id}.csv"
                )
                path, identity = fetch_optional(
                    scope, uri, license=_NOAA_TERMS, attribution=_NCEI_ATTRIBUTION
                )
                yield _Part(path, uri, station.id, year), identity

    def _prepare(
        self,
        scope: ExecutionScope,
        discovery: _Prepared,
        pin: ReleaseReference | None,
    ) -> _Prepared:
        if discovery.source is None:
            return discovery
        parts: list[_Part] = []
        resources = list(discovery.source.resources)
        stations = discovery.stations
        if self.provider == "ibtracs":
            path, identity = fetch(
                scope,
                _IBTRACS,
                license=_NOAA_TERMS,
                attribution=_IBTRACS_ATTRIBUTION,
            )
            parts.append(_Part(path, _IBTRACS))
            resources.append(identity)
        elif self.provider in ("ghcnh", "isd"):
            for part, resource in self._station_parts(scope, stations):
                parts.append(part)
                resources.append(resource)
        else:
            for station in stations:
                if self.provider in ("usgs", "hydat", "eccc"):
                    assert self.state.time is not None
                    start, end = self.state.time.start, self.state.time.end
                    if self.provider == "usgs":
                        endpoint, parameters = (
                            _USGS_DAILY,
                            {
                                "monitoring_location_id": station.id,
                                "parameter_code": "00060",
                                "statistic_id": "00003",
                            },
                        )
                    else:
                        endpoint = _GEOMET + (
                            "hydrometric-daily-mean/items"
                            if self.provider == "hydat"
                            else "climate-daily/items"
                        )
                        parameters = {
                            "STATION_NUMBER"
                            if self.provider == "hydat"
                            else "CLIMATE_IDENTIFIER": station.id
                        }
                    uri = (
                        endpoint
                        + "?"
                        + urlencode(
                            {
                                "f": "json",
                                "limit": 500,
                                "datetime": f"{start:04d}-01-01/{end:04d}-12-31",
                                **parameters,
                            }
                        )
                    )
                    found, identities, _ = _requests(
                        scope,
                        uri,
                        license="USGS public-domain data"
                        if self.provider == "usgs"
                        else "OGL-Canada-2.0",
                        attribution=(
                            "USGS Water Data for the Nation, doi:10.5066/F7P55KJN"
                        )
                        if self.provider == "usgs"
                        else "Environment and Climate Change Canada, GeoMet",
                        collect=False,
                    )
                    parts.extend(replace(part, station=station.id) for part in found)
                    resources.extend(identities)
                elif self.provider == "noaa_slr":
                    uri = (
                        _NOAA_SLR
                        + "?"
                        + urlencode(
                            {
                                "station": station.id,
                                "affil": "US",
                                "units": "metric",
                                "report_year": 2022,
                                "station_or_grid": "station",
                                "coastal_flag": 1,
                                "scenario": "all",
                            }
                        )
                    )
                    path, identity = fetch(
                        scope,
                        uri,
                        license=_NOAA_TERMS,
                        attribution=(
                            "NOAA 2022 Interagency Sea Level Rise Technical Report"
                        ),
                    )
                    parts.append(_Part(path, uri, station.id))
                    resources.append(identity)
                else:
                    parameters = {
                        "name": "extremewaterlevels",
                        "station": station.id,
                        "units": "metric",
                        "levelType": "high",
                        "extremeType": "annuals",
                    }
                    uri = _NOAA_PRODUCT + "?" + urlencode(parameters)
                    path, identity = fetch(
                        scope,
                        uri,
                        license=_NOAA_TERMS,
                        attribution="NOAA CO-OPS Extreme Water Levels",
                    )
                    parts.append(_Part(path, uri, station.id))
                    resources.append(identity)
        publication = discovery.source
        source = self._verified_source(publication, resources, None)
        assert source.release is not None
        if (
            pin is not None
            and source.release.sha256 != pin.sha256
            and _scoped_api_catalogue(self.provider, self.state)
        ):
            # Exact payload snapshots can include the full catalogue when
            # the original selection came from an unfiltered inventory.
            # Try that dependency too, without reacquiring observation data.
            try:
                all_stations, full_catalogue = _catalogue(
                    scope, self.provider, SelectionState()
                )
            except FileNotFoundError:
                pass
            else:
                if (
                    _select_stations(self.provider, all_stations, self.state)
                    == stations
                ):
                    candidate = replace(
                        publication,
                        resources=full_catalogue,
                        release=inventory_release(full_catalogue, id=self._release_id),
                    )
                    full_source = self._verified_source(
                        candidate,
                        [*full_catalogue, *resources[len(publication.resources) :]],
                        None,
                    )
                    assert full_source.release is not None
                    if full_source.release.sha256 == pin.sha256:
                        publication, source = candidate, full_source
        assert source.release is not None
        check_release(source.release, pin)
        return _Prepared(
            tuple(parts),
            source,
            stations,
            publication if publication.resources else None,
        )

    def _verified_source(
        self,
        publication: SourceIdentity,
        resources: list[ResourceIdentity],
        pin: ReleaseReference | None,
    ) -> SourceIdentity:
        release = release_snapshot(resources, id=self._release_id)
        check_release(release, pin)
        return replace(publication, release=release, resources=tuple(resources))

    def _raw(
        self,
        prepared: _Prepared,
        parts: tuple[_Part, ...] | None = None,
        superseded: list[int] | None = None,
    ) -> Iterator[dict[str, object]]:
        locations = {}
        if self.provider not in ("ghcnh", "isd", "ibtracs"):
            locations = {
                station.id: (
                    cast(float, station.longitude),
                    cast(float, station.latitude),
                )
                for station in prepared.stations
            }
        if self.provider == "noaa_water":
            station_map = {
                item.id: {
                    "stationId": item.id,
                    "longitude": item.longitude,
                    "latitude": item.latitude,
                }
                for item in prepared.stations
            }
        for part in prepared.parts if parts is None else parts:
            if part.path is None:
                continue
            if self.provider == "ghcnh":
                yield from ghcnh_rows(
                    part.path, cast(str, part.station), cast(int, part.year)
                )
            elif self.provider == "isd":
                yield from isd_rows(
                    part.path,
                    cast(str, part.station),
                    cast(int, part.year),
                    superseded,
                )
            elif self.provider == "ibtracs":
                yield from ibtracs_rows(part.path, self.agency)
            else:
                payload = decode_json(part.path.read_text())
                stations = tuple(locations)
                if self.provider == "usgs":
                    yield from usgs_rows(payload, stations, locations)
                elif self.provider == "hydat":
                    yield from hydat_rows(payload, stations)
                elif self.provider == "eccc":
                    yield from eccc_rows(payload, stations)
                elif self.provider == "noaa_slr":
                    yield from noaa_projection_rows(payload, stations)
                else:
                    if "extremeType=annuals" in part.uri:
                        yield from noaa_annual_rows(
                            payload, station_map[cast(str, part.station)]
                        )

    def _metadata(
        self, prepared: _Prepared, *, kind: EvidenceKind | None = None
    ) -> EvidenceMetadata:
        variables = self._declared_variables()
        probability = (
            Probability.annual(
                tail=cf.Tail.UPPER
                if self.indicator == _ANNUAL_MAXIMUM
                else cf.Tail.LOWER
            )
            if self.indicator is not None
            else Probability.projection()
            if self.provider == "noaa_slr"
            else None
        )
        # Projection scenarios/years are learned from actual native records, not a
        # guessed Cartesian product. This native decoding does no fitting.
        if self.provider == "noaa_slr":
            with native_data(source="noaa_sea_level_projections"):
                rows = list(self._raw(prepared))
            scenarios = {
                cast(str, row["scenario_id"]): Scenario(
                    cast(str, row["scenario_id"]),
                    definition={
                        "report_year": 2022,
                        "baseline_year": 2005,
                        "native_scenario": cast(str, row["scenario_id"]),
                    },
                )
                for row in rows
            }
            pairs = {
                (
                    cast(str, row["variable_id"]),
                    cast(str, row["scenario_id"]),
                    int(cast(str, row["time"])),
                )
                for row in rows
            }
            by_id = {v.id: v for v in variables}
            series = [
                SeriesSpec(
                    by_id[v],
                    scenario=scenarios[s],
                    period=Period.window(y, y, calendar=Calendar.GREGORIAN),
                    probability=Probability.projection(),
                )
                for v, s, y in sorted(pairs)
            ]
        else:
            series = [
                SeriesSpec(
                    variable,
                    period=self.period,
                    probability=(
                        Probability.annual(
                            tail=cf.Tail.UPPER
                            if variable.id.endswith(_ANNUAL_MAXIMUM)
                            else cf.Tail.LOWER
                        )
                        if self.provider in ("ghcnh", "isd") and not self.raw
                        else probability
                    ),
                )
                for variable in variables
            ]
        spec = DataSpec(variables, series=series)
        actual_kind = (
            EvidenceKind.PROJECTIONS
            if self.provider == "noaa_slr"
            else EvidenceKind.SAMPLES
            if self.provider in ("ghcnh", "isd") and not self.raw
            else EvidenceKind.OBSERVATIONS
        )
        sampling = (
            None
            if actual_kind is EvidenceKind.PROJECTIONS
            or self.provider == "ibtracs"
            or self.raw
            else Sampling(
                calendar=Calendar.GREGORIAN,
                step=cf.Duration(
                    1,
                    unit=cf.TimeUnit.YEAR
                    if self.provider == "noaa_water" or self.indicator is not None
                    else cf.TimeUnit.HOUR
                    if self.provider in ("ghcnh", "isd")
                    else cf.TimeUnit.DAY,
                ),
            )
        )
        return EvidenceMetadata(
            kind=actual_kind if kind is None else kind,
            spec=spec,
            spatial=SpatialContract(supports=(SupportKind.POINT,), index_resolution=9),
            report=ExecutionReport(
                False,
                None,
                source_identities=() if prepared.source is None else (prepared.source,),
            ),
            source=prepared.source,
            sampling=sampling,
        )

    def inspection(self, scope: ExecutionScope) -> Inspection:
        state, requested = self._selection()
        requirements = self.requirements()
        discovery, pin = self._discover(scope, requested)
        prepared = (
            self._prepare(scope, discovery, pin)
            if not requirements and (pin is not None or self.provider == "noaa_slr")
            else discovery
        )
        metadata = self._metadata(prepared)
        metadata = replace(
            metadata, spec=state.resolve(metadata.spec, source=metadata.source).spec
        )
        if prepared is discovery and prepared.source is not None:
            requirements = (
                *requirements,
                Requirement(
                    "snapshot_resolved_on_read",
                    "Native data bytes are pinned and any requested release "
                    "is verified when read.",
                    "release",
                ),
            )
        return Inspection(
            schema=evidence_arrow_schema(metadata),
            series=tuple(
                SeriesReference(
                    item.variable,
                    scenario=item.scenario,
                    period=item.period,
                    probability=item.probability,
                    release=None
                    if prepared.source is None
                    else prepared.source.release,
                    member=None if prepared.source is None else prepared.source.member,
                )
                for item in metadata.spec.series
            ),
            indicators=_temperature_indicators()
            if self.provider in ("ghcnh", "isd")
            else (),
            releases=()
            if prepared.source is None or prepared.source.release is None
            else (prepared.source.release,),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
                *(
                    (Capability.FIT,)
                    if self.indicator is not None
                    else (Capability.AGGREGATE,)
                    if metadata.kind is EvidenceKind.OBSERVATIONS
                    and metadata.sampling is not None
                    else ()
                ),
            ),
            requirements=requirements,
            source=prepared.source,
            stations=prepared.stations,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[EvidenceMetadata]:
        state, requested = self._selection()
        if requirements := self.requirements():
            raise CapabilityError(" ".join(item.message for item in requirements))
        discovery, pin = self._discover(scope, requested)
        weather = self.provider in ("ghcnh", "isd")
        prepared = discovery if weather else self._prepare(scope, discovery, pin)
        metadata = self._metadata(prepared)
        resolved = state.resolve(metadata.spec, source=metadata.source)
        actual = selected_metadata(metadata, resolved)
        publication = discovery.source if weather else prepared.publication
        resources = [] if publication is None else list(publication.resources)
        absent: list[_Part] = []

        def with_source(
            metadata: EvidenceMetadata, source: SourceIdentity | None
        ) -> EvidenceMetadata:
            identities: tuple[SourceIdentity, ...] = () if source is None else (source,)
            if publication is not None and source is not None and source != publication:
                identities = (publication, source)
            return replace(
                metadata,
                source=source,
                report=replace(metadata.report, source_identities=identities),
            )

        final = with_source(actual, actual.source)
        count = missing = 0
        superseded: list[int] = []

        def superseded_diagnostics() -> tuple[ExecutionDiagnostic, ...]:
            if not sum(superseded):
                return ()
            return (
                ExecutionDiagnostic(
                    "isd_superseded_reports",
                    cf.DiagnosticSeverity.INFO,
                    Stage.DECODING,
                    f"{sum(superseded)} ISD reports shared a timestamp with another "
                    "report and were dropped; one report per timestamp is kept "
                    "(valid temperature first, then FM-12, FM-15, FM-16, others).",
                    source=self.provider,
                ),
            )

        def parts() -> Iterator[_Part]:
            nonlocal final
            if not weather:
                yield from prepared.parts
                return
            assert publication is not None
            for part, resource in self._station_parts(scope, prepared.stations):
                resources.append(resource)
                if part.path is None:
                    absent.append(part)
                final = with_source(
                    final,
                    replace(publication, release=None, resources=tuple(resources)),
                )
                yield part

        def produce() -> Iterator[pa.RecordBatch]:
            nonlocal final, count, missing
            if not resolved.keys or self.state.stations == ():
                return
            if self.indicator is not None:
                output_variable = _variables(self.provider, indicator=self.indicator)[0]
                raw_variable = _variables(self.provider)[0]
                diagnostic_records: list[ExecutionDiagnostic] = []
                usable = incomplete = unavailable = 0
                # A station-year is an explicit blocking reduction, not a global
                # materialization of every requested station and year.
                for part in parts():
                    station, year = part.station, part.year
                    assert station is not None and year is not None
                    raw = (
                        []
                        if part.path is None
                        else list(ghcnh_rows(part.path, station, year))
                        if self.provider == "ghcnh"
                        else list(isd_rows(part.path, station, year, superseded))
                    )
                    if not raw:
                        incomplete += 1
                        unavailable += 1
                        reason = (
                            "No observation file exists"
                            if part.path is None
                            else "The file contains no observations with source support"
                        )
                        diagnostic_records.append(
                            UnavailableStationYear(
                                Stage.ACQUISITION
                                if part.path is None
                                else Stage.DECODING,
                                f"{reason} for station {station!r} in {year}. "
                                "Historical point support is unknown.",
                                source=self.provider,
                                station_id=station,
                                year=year,
                                minimum_fraction=self.coverage.fraction,
                            )
                        )
                        final = replace(
                            final,
                            report=replace(
                                final.report,
                                diagnostics=(
                                    *diagnostic_records,
                                    *superseded_diagnostics(),
                                ),
                                counts=ExecutionCounts(
                                    requested=count + unavailable,
                                    produced=count,
                                    no_data=missing,
                                    usable_years=usable,
                                    incomplete_years=incomplete,
                                ),
                            ),
                        )
                        continue
                    raw_period = Period.window(year, year, calendar=Calendar.GREGORIAN)
                    for row in raw:
                        row["period_id"] = raw_period.id
                    raw_spec = DataSpec(
                        [raw_variable],
                        series=[SeriesSpec(raw_variable, period=raw_period)],
                    )
                    raw_metadata = replace(
                        actual,
                        kind=EvidenceKind.OBSERVATIONS,
                        spec=raw_spec,
                        source=final.source,
                        provenance=(),
                        sampling=Sampling(
                            calendar=Calendar.GREGORIAN,
                            step=cf.Duration(1, unit=cf.TimeUnit.HOUR),
                        ),
                        report=ExecutionReport(
                            True,
                            True,
                            counts=ExecutionCounts(
                                requested=len(raw),
                                produced=len(raw),
                                no_data=sum(row["value"] is None for row in raw),
                            ),
                            source_identities=final.report.source_identities,
                        ),
                    )
                    reduction = (
                        AnnualMaximum(coverage=self.coverage)
                        if self.indicator == _ANNUAL_MAXIMUM
                        else AnnualMinimum(coverage=self.coverage)
                    )
                    evidence_table(raw, raw_metadata)
                    daily_table, daily_metadata = complete_daily_extrema(
                        raw,
                        raw_metadata,
                        year=year,
                        maximum=self.indicator == _ANNUAL_MAXIMUM,
                        coverage=self.daily_coverage,
                    )
                    table, reduced = aggregate_table(
                        daily_table, daily_metadata, reduction
                    )
                    changed = rename_metadata(
                        reduced, variables={raw_variable.id: output_variable.id}
                    )
                    records = table.to_pylist()
                    for row in records:
                        row["variable_id"] = output_variable.id
                        row["period_id"] = (
                            None if self.period is None else self.period.id
                        )
                    records = [row for row in records if resolved.matches_record(row)]
                    diagnostic_records.extend(changed.report.diagnostics)
                    missing_rows = sum(row["value"] is None for row in records)
                    usable += len(records) - missing_rows
                    incomplete += missing_rows
                    count += len(records)
                    missing += missing_rows
                    final = replace(
                        changed,
                        spec=actual.spec,
                        provenance=(*actual.provenance, *changed.provenance),
                        report=ExecutionReport(
                            False,
                            None,
                            diagnostics=(
                                *diagnostic_records,
                                *superseded_diagnostics(),
                            ),
                            counts=ExecutionCounts(
                                requested=count + unavailable,
                                produced=count,
                                no_data=missing,
                                usable_years=usable,
                                incomplete_years=incomplete,
                            ),
                            source_identities=final.report.source_identities,
                        ),
                    )
                    yield from evidence_table(records, final).to_batches(
                        max_chunksize=scope.runtime.batch_rows
                    )
                return
            pending = []
            groups = (
                (self._raw(prepared, (part,), superseded) for part in parts())
                if weather
                else iter((self._raw(prepared),))
            )
            for rows in groups:
                for row in rows:
                    if metadata.kind is EvidenceKind.PROJECTIONS:
                        year = int(cast(str, row["time"]))
                        row["period_id"] = Period.window(
                            year, year, calendar=Calendar.GREGORIAN
                        ).id
                        row["coordinate"] = {"kind": "time", "time": str(year)}
                        row["interval"] = None
                    else:
                        row["period_id"] = (
                            None if self.period is None else self.period.id
                        )
                    if not resolved.matches_record(row):
                        continue
                    if metadata.kind is EvidenceKind.PROJECTIONS:
                        row.pop("time")
                    pending.append(row)
                    count += 1
                    missing += int(row["value"] is None)
                    if len(pending) == scope.runtime.batch_rows:
                        yield from evidence_table(pending, actual).to_batches()
                        pending = []
                if weather and pending:
                    yield from evidence_table(pending, actual).to_batches()
                    pending = []
            if pending:
                yield from evidence_table(pending, actual).to_batches()

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal final
            yield from produce()
            if weather and publication is not None:
                final = with_source(
                    final, self._verified_source(publication, resources, pin)
                )

        def report() -> ExecutionReport:
            if self.indicator is not None:
                return final.report
            return ExecutionReport(
                False,
                None,
                diagnostics=(
                    *(
                        ExecutionDiagnostic(
                            "station_year_unavailable",
                            cf.DiagnosticSeverity.WARNING,
                            Stage.ACQUISITION,
                            f"The source has no observation file for {part.station} "
                            f"in {part.year}.",
                            source=self.provider,
                        )
                        for part in absent
                    ),
                    *superseded_diagnostics(),
                ),
                counts=ExecutionCounts(
                    requested=count + len(absent),
                    produced=count,
                    no_data=missing,
                    incomplete_years=len(absent),
                ),
                source_identities=final.report.source_identities,
            )

        return NodeStream(
            evidence_arrow_schema(actual),
            actual,
            batches(),
            report=report,
            final_metadata=lambda: final,
            verify_before_yield=weather and pin is not None,
        )


class _WaterLevelNode(SourceNode[DatasetMetadata | EvidenceMetadata]):
    _release_id = "noaa:annual_exceedance_snapshot"

    def __init__(
        self,
        release: ReleaseReference | None,
        state: SelectionState,
        *,
        confidence: bool = False,
    ) -> None:
        self.release, self.state, self.confidence = release, state, confidence
        self.key = _key(
            "noaa_water_levels",
            None if release is None else release._document(),
            state.document(),
            confidence,
        )

    def selected(self, state: SelectionState) -> _WaterLevelNode:
        return _WaterLevelNode(self.release, state, confidence=self.confidence)

    @staticmethod
    def _level_variable_id(station: str) -> str:
        return f"noaa:{station}:annual_high_water"

    def _variable_ids(self) -> frozenset[str] | None:
        if self.state.stations is None:
            return None
        return frozenset(
            self._level_variable_id(station) for station in self.state.stations
        )

    def requirements(self) -> tuple[Requirement, ...]:
        if _empty_state(self.state):
            return ()
        return (
            ()
            if self.state.stations is not None or self.state.area is not None
            else (
                Requirement(
                    "station_selection_required",
                    "Choose stations from inspect().stations or select an area.",
                    "stations",
                ),
            )
        )

    def _publication(
        self, scope: ExecutionScope, requested: ReleasePins
    ) -> tuple[tuple[StationReference, ...], SourceIdentity, ReleaseReference | None]:
        stations, resources = _catalogue(scope, "noaa_water", self.state)
        release = inventory_release(resources, id=self._release_id)
        return (
            stations,
            SourceIdentity(
                provider="noaa",
                product="annual_high_water_exceedance",
                release=release,
                resources=resources,
            ),
            inventory_payload_pin(release, requested),
        )

    def _station_variables(
        self,
        state: SelectionState,
        station_ids: tuple[str, ...],
        probability: Probability,
    ) -> frozenset[str]:
        # Station IDs determine the variables; their datums are in the payloads.
        variables = tuple(Variable(self._level_variable_id(i)) for i in station_ids)
        available = DataSpec(
            variables,
            series=[SeriesSpec(v, probability=probability) for v in variables],
        )
        selection = replace(
            state,
            variables=None
            if state.variables is None
            else tuple(v if isinstance(v, str) else v.id for v in state.variables),
            series=None
            if state.series is None
            else tuple(
                replace(ref, variable=Variable(ref.variable.id)) for ref in state.series
            ),
        ).resolve(available)
        return frozenset(v.id for v in selection.spec.variables)

    def _resolve(
        self, scope: ExecutionScope
    ) -> tuple[DatasetMetadata | EvidenceMetadata, list[dict[str, object]]]:
        if self.state.time is not None:
            raise CapabilityError(
                "Published NOAA exceedance estimates cannot be recalculated"
                " for an observation-time selection. Select raw "
                "observations and explicitly aggregate and fit them "
                "instead."
            )
        state, requested = payload_selection(self.state, self.release)
        probability = (
            Probability.confidence()
            if self.confidence
            else Probability.annual(tail=cf.Tail.UPPER)
        )
        if _empty_state(state):
            self._station_variables(state, state.stations or (), probability)
            source = publication = None
            records: list[dict[str, object]] = []
            variables: list[Variable] = []
        else:
            if self.state.stations is None and self.state.area is None:
                raise CapabilityError(
                    "Choose NOAA stations from inspect().stations or "
                    "select(area=Bounds(...))."
                )
            selected, publication, pin = self._publication(scope, requested)
            active = self._station_variables(
                state, tuple(station.id for station in selected), probability
            )
            catalogue_uri = (
                _NOAA_PRODUCT
                + "?"
                + urlencode({"name": "extremewaterlevels", "units": "metric"})
            )
            catalogue, _ = fetch_json(
                scope, catalogue_uri, license=_NOAA_TERMS, attribution=_NOAA_ATTRIBUTION
            )
            station_rows = {
                text(row.get("stationId"), "NOAA station ID"): row
                for row in (
                    obj(item, "NOAA station")
                    for item in array(
                        obj(catalogue, "NOAA catalogue").get("ExtremeWaterLevels"),
                        "NOAA stations",
                    )
                )
            }
            resources = list(publication.resources)
            records, variables = [], []
            for station in selected:
                variable_id = self._level_variable_id(station.id)
                if variable_id not in active:
                    continue
                uri = (
                    _NOAA_PRODUCT
                    + "?"
                    + urlencode(
                        {
                            "name": "extremewaterlevels",
                            "station": station.id,
                            "units": "metric",
                            "levelType": "high",
                            "details": "exceedancelevels",
                        }
                    )
                )
                with native_data(
                    scope=scope, source="noaa_published_exceedance_estimates"
                ):
                    payload, identity = fetch_json(
                        scope,
                        uri,
                        license=_NOAA_TERMS,
                        attribution=(
                            "NOAA CO-OPS annual exceedance estimates; 90% confidence "
                            "intervals"
                        ),
                    )
                    resources.append(identity)
                    document = obj(payload, "NOAA annual exceedance estimates")
                    if (
                        document.get("stationId") != station.id
                        or document.get("units") != "meters"
                    ):
                        raise ValueError(
                            "NOAA exceedance station or units differ from the request"
                        )
                    datum = text(document.get("datum"), "NOAA tidal datum")
                    epoch = text(
                        station_rows[station.id].get("epoch"), "NOAA tidal datum epoch"
                    )
                    variable = Variable(
                        variable_id,
                        measurement=cf.WaterSurfaceElevation(
                            unit=cf.LengthUnit.METRE,
                            datum=f"noaa:{station.id}:{datum}:{epoch}",
                        ),
                        label=f"Annual high-water elevation at {station.label}",
                    )
                    variables.append(variable)
                    model = document.get("GEV_HI")
                    estimates = (
                        []
                        if model is None
                        else array(
                            obj(model, "NOAA high-water estimates").get(
                                "exceedanceLevels"
                            ),
                            "NOAA exceedance levels",
                        )
                    )
                    for value in estimates:
                        row = obj(value, "NOAA exceedance level")
                        chance = number(
                            row.get("probability"), "annual exceedance probability"
                        )
                        years = number(
                            row.get("returnPeriod"), "published return period"
                        )
                        level = number(row.get("level"), "published water level")
                        assert (
                            chance is not None
                            and years is not None
                            and level is not None
                        )
                        if (
                            not 0 < chance < 1
                            or years <= 1
                            or abs(chance - 1 / years) > 0.00051
                        ):
                            raise ValueError(
                                "NOAA annual probability and return period disagree"
                            )
                        lower, upper = (
                            number(
                                row.get("lower90CI"),
                                "lower confidence limit",
                                missing=True,
                            ),
                            number(
                                row.get("upper90CI"),
                                "upper confidence limit",
                                missing=True,
                            ),
                        )
                        if lower is not None and upper is not None and lower > upper:
                            raise ValueError(
                                "NOAA confidence interval limits are reversed"
                            )
                        assert (
                            station.longitude is not None
                            and station.latitude is not None
                        )
                        records.append(
                            {
                                "variable_id": variable.id,
                                "scenario_id": None,
                                "period_id": None,
                                **point_identity(
                                    station.id, station.longitude, station.latitude
                                ),
                                "value": level,
                                "coordinate": {"kind": "return_period", "years": years},
                                "probability": None,
                                "interval": None
                                if lower is None or upper is None
                                else {
                                    "confidence": 0.9,
                                    "lower": lower,
                                    "upper": upper,
                                },
                                "failure": None,
                                "q": 1 - chance,
                            }
                        )
            actual = release_snapshot(resources, id=self._release_id)
            check_release(actual, pin)
            source = replace(publication, release=actual, resources=tuple(resources))
        spec = DataSpec(
            variables,
            series=[SeriesSpec(v, probability=probability) for v in variables],
        )
        if variables:
            acquired = {v.id for v in variables}
            # Only type-check datum declarations whose stations contributed data.
            SelectionState(
                variables=tuple(
                    v
                    for v in state.variables or ()
                    if (v if isinstance(v, str) else v.id) in acquired
                )
                or None,
                series=tuple(
                    ref for ref in state.series or () if ref.variable.id in acquired
                )
                or None,
            ).resolve(spec, source=source)
        report = ExecutionReport(
            False,
            None,
            source_identities=()
            if source is None or publication is None
            else (publication, source),
        )
        spatial = SpatialContract(supports=(SupportKind.POINT,), index_resolution=9)
        if self.confidence:
            metadata: DatasetMetadata | EvidenceMetadata = EvidenceMetadata(
                kind=EvidenceKind.PROJECTIONS,
                spec=spec,
                spatial=spatial,
                report=report,
                source=source,
            )
        else:
            metadata = DatasetMetadata(
                spec=spec, spatial=spatial, report=report, source=source
            )
        return metadata, records

    def inspection(self, scope: ExecutionScope) -> Inspection:
        if (
            not _empty_state(self.state)
            and self.state.stations is None
            and self.state.area is None
        ):
            _, requested = payload_selection(self.state, self.release)
            stations, publication, _ = self._publication(scope, requested)
            assert publication.release is not None
            return Inspection(
                source=publication,
                releases=(publication.release,),
                stations=stations,
                capabilities=(
                    Capability.INSPECT,
                    Capability.READ,
                    Capability.BATCHES,
                    Capability.WRITE,
                ),
                requirements=(
                    Requirement(
                        "station_selection_required",
                        (
                            "Choose stations from inspect().stations or "
                            "select(area=Bounds(...))."
                        ),
                        "stations",
                    ),
                ),
            )
        metadata, _ = self._resolve(scope)
        from ._schema import dataset_arrow_schema

        return Inspection(
            schema=evidence_arrow_schema(metadata)
            if isinstance(metadata, EvidenceMetadata)
            else dataset_arrow_schema(metadata),
            series=tuple(
                SeriesReference(
                    item.variable,
                    release=None
                    if metadata.source is None
                    else metadata.source.release,
                    probability=item.probability,
                )
                for item in metadata.spec.series
            ),
            releases=()
            if metadata.source is None or metadata.source.release is None
            else (metadata.source.release,),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
            ),
            source=metadata.source,
        )

    def open(
        self, scope: ExecutionScope
    ) -> NodeStream[DatasetMetadata | EvidenceMetadata]:
        metadata, records = self._resolve(scope)
        count = 0

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal count
            if self.confidence:
                output = []
                for row in records:
                    row = dict(row)
                    row.pop("q")
                    output.append(row)
                count = len(output)
                yield from evidence_table(
                    output, cast(EvidenceMetadata, metadata)
                ).to_batches(max_chunksize=scope.runtime.batch_rows)
            else:
                grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
                for row in records:
                    grouped.setdefault(
                        (
                            cast(str, row["variable_id"]),
                            cast(str, row["support_id"]),
                            cast(int, row["cell_id"]),
                        ),
                        [],
                    ).append(row)
                dataset_output: list[DatasetRow] = []
                for (variable, support, cell), values in grouped.items():
                    values.sort(key=lambda row: cast(float, row["q"]))
                    curve = cf.QuantileCurve(
                        [cast(float, row["q"]) for row in values],
                        [cast(float, row["value"]) for row in values],
                    )
                    dataset_output.append(
                        DatasetRow(
                            variable_id=variable,
                            support_id=support,
                            cell_id=cell,
                            geometry=cast(bytes, values[0]["geometry"]),
                            curve=curve,
                        )
                    )
                count = len(dataset_output)
                yield from rows_to_table(
                    dataset_output, cast(DatasetMetadata, metadata)
                ).to_batches(max_chunksize=scope.runtime.batch_rows)

        def report() -> ExecutionReport:
            return replace(
                metadata.report, counts=ExecutionCounts(requested=count, produced=count)
            )

        from ._schema import dataset_arrow_schema

        schema = (
            evidence_arrow_schema(metadata)
            if isinstance(metadata, EvidenceMetadata)
            else dataset_arrow_schema(metadata)
        )
        return NodeStream(schema, metadata, batches(), report=report)


def _provider_node(base: object, provider: str) -> _ObservedNode:
    if not isinstance(base, _ObservedNode) or base.provider != provider:
        raise CapabilityError(
            "Choose the native source product before declaring, renaming, "
            "or converting its scientific output."
        )
    return base


def _hourly_recipe(base: object, state: SelectionState, provider: str) -> Observations:
    node = _provider_node(base, provider)
    raw_variable = _variables(provider)[0]
    annual_ids = {
        v.id
        for indicator in _temperature_indicators()
        for v in _variables(provider, indicator=indicator.id)
    }

    def native(value: str | Variable) -> str | Variable:
        identifier = value.id if isinstance(value, Variable) else value
        return raw_variable if identifier in annual_ids else value

    selected = replace(
        state,
        variables=None
        if state.variables is None
        else tuple(native(v) for v in state.variables),
        series=None
        if state.series is None
        else tuple(
            replace(reference, variable=raw_variable, probability=None)
            if reference.variable.id in annual_ids
            else reference
            for reference in state.series
        ),
    )
    return Observations._create(
        cast(
            Node[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]],
            _ObservedNode(provider, release=node.release, raw=True),
        ),
        selection=selected,
        operations=(
            Operation(
                Stage.ACQUISITION,
                "Read source-quality-filtered native station temperature reports",
            ),
        ),
    )


def _published_level_selection(state: SelectionState) -> SelectionState:
    if state.variables not in (None, ()) or state.series not in (None, ()):
        raise CapabilityError(
            "Choose levels() or confidence_intervals() before selecting "
            "the published estimate variables or series."
        )
    return state


class GHCNh(Samples):
    class Indicator(str, Enum):
        ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE = _ANNUAL_MAXIMUM
        ANNUAL_MINIMUM_DAILY_MINIMUM_TEMPERATURE = _ANNUAL_MINIMUM

    def __init__(
        self,
        *,
        indicator: GHCNh.Indicator | IndicatorReference | None = None,
        release: ReleaseReference | None = None,
        coverage: Coverage | None = None,
        daily_coverage: Coverage | None = None,
    ) -> None:
        _release(release)
        node = _ObservedNode(
            "ghcnh",
            release=release,
            indicator=_indicator(indicator),
            coverage=coverage,
            daily_coverage=daily_coverage,
        )
        _init(
            self,
            node,
            (
                "Acquire GHCNh source snapshots; named indicators reduce "
                "hourly temperature before explicit fitting"
            ),
        )

    def observations(self) -> Observations:
        return _hourly_recipe(self._base, self._selection, "ghcnh")


class ISD(Samples):
    class Indicator(str, Enum):
        ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE = _ANNUAL_MAXIMUM
        ANNUAL_MINIMUM_DAILY_MINIMUM_TEMPERATURE = _ANNUAL_MINIMUM

    def __init__(
        self,
        *,
        indicator: ISD.Indicator | IndicatorReference | None = None,
        release: ReleaseReference | None = None,
        coverage: Coverage | None = None,
        daily_coverage: Coverage | None = None,
    ) -> None:
        _release(release)
        _init(
            self,
            _ObservedNode(
                "isd",
                release=release,
                indicator=_indicator(indicator),
                coverage=coverage,
                daily_coverage=daily_coverage,
            ),
            (
                "Acquire ISD source snapshots; named indicators reduce "
                "hourly temperature before explicit fitting"
            ),
        )

    def observations(self) -> Observations:
        return _hourly_recipe(self._base, self._selection, "isd")


class USGS(Observations):
    def __init__(self, *, release: ReleaseReference | None = None) -> None:
        _release(release)
        _init(
            self,
            _ObservedNode("usgs", release=release),
            "Read USGS daily mean discharge in its actual source units",
        )


class HYDAT(Observations):
    def __init__(self, *, release: ReleaseReference | None = None) -> None:
        _release(release)
        _init(
            self,
            _ObservedNode("hydat", release=release),
            "Read ECCC historical daily mean gauge discharge and water level",
        )


class ECCCDaily(Observations):
    def __init__(self, *, release: ReleaseReference | None = None) -> None:
        _release(release)
        _init(
            self,
            _ObservedNode("eccc", release=release),
            "Read ECCC native daily climate observations without fitting",
        )


class NOAASeaLevel(Projections):
    def __init__(self, *, release: ReleaseReference | None = None) -> None:
        _release(release)
        _init(
            self,
            _ObservedNode("noaa_slr", release=release),
            (
                "Read NOAA 2022 scenario projections and reported interval "
                "bounds, without annual exceedance inference"
            ),
        )


class NOAAWaterLevels(Observations):
    def __init__(self, *, release: ReleaseReference | None = None) -> None:
        _release(release)
        _init(
            self,
            _ObservedNode("noaa_water", release=release),
            (
                "Read NOAA detrended annual high-water observations; these "
                "are deviations from a trend, not flood depths"
            ),
        )

    def levels(self) -> Dataset:
        node = _provider_node(self._base, "noaa_water")
        return Dataset._create(
            cast(
                Node[DatasetMetadata], _WaterLevelNode(node.release, SelectionState())
            ),
            selection=_published_level_selection(self._selection),
            operations=(
                Operation(
                    Stage.ACQUISITION,
                    (
                        "Use NOAA published central annual high-water exceedance "
                        "estimates; confidence bounds are separate evidence"
                    ),
                ),
            ),
        )

    def confidence_intervals(self) -> Projections:
        node = _provider_node(self._base, "noaa_water")
        return Projections._create(
            cast(
                Node[EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]]],
                _WaterLevelNode(node.release, SelectionState(), confidence=True),
            ),
            selection=_published_level_selection(self._selection),
            operations=(
                Operation(
                    Stage.ACQUISITION,
                    (
                        "Read NOAA published 90% confidence bounds at actual return"
                        " periods"
                    ),
                ),
            ),
        )


class IBTrACS(Observations):
    class Agency(str, Enum):
        USA = "USA"
        TOKYO = "TOKYO"
        CMA = "CMA"
        HKO = "HKO"
        NEWDELHI = "NEWDELHI"
        REUNION = "REUNION"
        BOM = "BOM"
        NADI = "NADI"
        WELLINGTON = "WELLINGTON"
        WMO = "WMO"

    def __init__(
        self,
        *,
        agency: IBTrACS.Agency = Agency.USA,
        release: ReleaseReference | None = None,
    ) -> None:
        _release(release)
        if not isinstance(agency, IBTrACS.Agency):
            raise TypeError("agency must be IBTrACS.Agency")
        _init(
            self,
            _ObservedNode("ibtracs", release=release, agency=agency.value),
            "Read IBTrACS point observations for the declared agency",
        )
