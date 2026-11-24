"""Source-native observation decoders, independent of acquisition and execution."""

from __future__ import annotations

import csv
import hashlib
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import cast

from h3.api import basic_int as h3  # type: ignore[import-untyped]


def obj(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty exact text")
    return value


def number(value: object, label: str, *, missing: bool = False) -> float | None:
    if missing and (value is None or value == ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(cast(str | int | float, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def coordinates(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a WGS84 point")
    lon, lat = number(value[0], label), number(value[1], label)
    assert lon is not None and lat is not None
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError(f"{label} lies outside WGS84")
    return lon, lat


def feature_coordinates(feature: Mapping[str, object]) -> tuple[float, float]:
    geometry = obj(feature.get("geometry"), "feature geometry")
    if geometry.get("type") != "Point":
        raise ValueError("station observations require a source point geometry")
    return coordinates(geometry.get("coordinates"), "station coordinates")


def point_identity(station: str, lon: float, lat: float) -> dict[str, object]:
    lon, lat = coordinates((lon, lat), "source coordinates")
    geometry = struct.pack("<BIdd", 1, 1, lon, lat)
    # A station may move. A native station ID is not an immutable spatial support.
    position = hashlib.sha256(geometry).hexdigest()[:20]
    return {
        "support_id": f"{station}@{position}",
        "station_id": station,
        "cell_id": h3.latlng_to_cell(lat, lon, 9),
        "geometry": geometry,
    }


def observed_record(
    variable: str,
    station: str,
    lon: float,
    lat: float,
    when: str,
    value: float | None,
    *,
    group_id: str | None = None,
    scenario_id: str | None = None,
    period_id: str | None = None,
    missing_reason: str = "The source reports no valid observation.",
) -> dict[str, object]:
    return {
        "variable_id": variable,
        "scenario_id": scenario_id,
        "period_id": period_id,
        **point_identity(station, lon, lat),
        "time": when,
        "value": value,
        "group_id": group_id,
        "coverage": None,
        "failure": None
        if value is not None
        else {"code": "source_no_data", "message": missing_reason},
    }


def feature_collection(payload: object) -> tuple[list[dict[str, object]], str | None]:
    page = obj(payload, "OGC feature collection")
    if page.get("type") != "FeatureCollection":
        raise ValueError("source response is not an OGC FeatureCollection")
    features = [
        obj(value, "feature") for value in array(page.get("features"), "features")
    ]
    returned = page.get("numberReturned")
    if returned is not None and (
        type(returned) is not int or returned != len(features)
    ):
        raise ValueError("source feature count contradicts numberReturned")
    following = []
    for value in cast(list[object], page.get("links", [])):
        link = obj(value, "OGC link")
        if link.get("rel") == "next":
            following.append(text(link.get("href"), "next-page URL"))
    if len(following) > 1:
        raise ValueError("source response contains multiple next-page links")
    return features, following[0] if following else None


def usgs_rows(
    payload: object,
    stations: Sequence[str],
    locations: Mapping[str, tuple[float, float]],
) -> list[dict[str, object]]:
    features, _ = feature_collection(payload)
    result = []
    for feature in features:
        p = obj(feature.get("properties"), "USGS daily properties")
        station = text(p.get("monitoring_location_id"), "USGS station")
        if (
            station not in stations
            or p.get("parameter_code") != "00060"
            or p.get("statistic_id") != "00003"
        ):
            raise ValueError(
                "USGS returned an observation outside the requested series"
            )
        when = text(p.get("time"), "USGS observation time")
        date.fromisoformat(when[:10])
        unit = p.get("unit_of_measure")
        if unit != "ft^3/s":
            raise ValueError(f"USGS discharge has unexpected source unit {unit!r}")
        # Preserve USGS's actual unit; changing science is a separate recorded step.
        value = number(p.get("value"), "USGS discharge", missing=True)
        geometry = feature.get("geometry")
        lon, lat = (
            feature_coordinates(feature) if geometry is not None else locations[station]
        )
        result.append(
            observed_record("usgs:daily_mean_discharge", station, lon, lat, when, value)
        )
    return result


def hydat_rows(payload: object, stations: Sequence[str]) -> list[dict[str, object]]:
    features, _ = feature_collection(payload)
    result = []
    for feature in features:
        p = obj(feature.get("properties"), "HYDAT daily properties")
        station = text(p.get("STATION_NUMBER"), "HYDAT station")
        if station not in stations:
            raise ValueError("HYDAT returned an unrequested station")
        when = text(p.get("DATE"), "HYDAT observation date")
        date.fromisoformat(when)
        lon, lat = feature_coordinates(feature)
        for field, variable in (
            ("DISCHARGE", "eccc:daily_mean_discharge"),
            ("LEVEL", "eccc:daily_mean_water_level"),
        ):
            result.append(
                observed_record(
                    variable,
                    station,
                    lon,
                    lat,
                    when,
                    number(p.get(field), field, missing=True),
                )
            )
    return result


ECCC_FIELDS = {
    "MEAN_TEMPERATURE": "eccc:daily_mean_temperature",
    "MIN_TEMPERATURE": "eccc:daily_minimum_temperature",
    "MAX_TEMPERATURE": "eccc:daily_maximum_temperature",
    "TOTAL_PRECIPITATION": "eccc:daily_total_precipitation",
    "TOTAL_RAIN": "eccc:daily_total_rainfall",
    "TOTAL_SNOW": "eccc:daily_total_snowfall",
    "SNOW_ON_GROUND": "eccc:snow_depth",
    "SPEED_MAX_GUST": "eccc:daily_maximum_wind_gust",
}


def eccc_rows(payload: object, stations: Sequence[str]) -> list[dict[str, object]]:
    features, _ = feature_collection(payload)
    result = []
    for feature in features:
        p = obj(feature.get("properties"), "ECCC daily properties")
        station = text(p.get("CLIMATE_IDENTIFIER"), "ECCC station")
        if station not in stations:
            raise ValueError("ECCC returned an unrequested station")
        when = text(p.get("LOCAL_DATE"), "ECCC local date")[:10]
        day = date.fromisoformat(when)
        for field, expected in (
            ("LOCAL_YEAR", day.year),
            ("LOCAL_MONTH", day.month),
            ("LOCAL_DAY", day.day),
        ):
            if field in p and p[field] != expected:
                raise ValueError("ECCC date parts contradict LOCAL_DATE")
        lon, lat = feature_coordinates(feature)
        for field, variable in ECCC_FIELDS.items():
            if field in p:
                result.append(
                    observed_record(
                        variable,
                        station,
                        lon,
                        lat,
                        when,
                        number(p.get(field), field, missing=True),
                    )
                )
    return result


def noaa_annual_rows(
    payload: object, station: Mapping[str, object]
) -> list[dict[str, object]]:
    p = obj(payload, "NOAA annual extremes")
    identifier = text(station.get("stationId"), "NOAA station ID")
    if p.get("stationId") != identifier or p.get("units") != "meters":
        raise ValueError("NOAA annual station or metric units do not match")
    lon, lat = coordinates(
        [station.get("longitude"), station.get("latitude")], "NOAA station"
    )
    result = []
    for raw in array(p.get("extremes"), "NOAA annual extremes"):
        row = obj(raw, "NOAA annual extreme")
        year = row.get("year")
        if type(year) is not int or not 1 <= year <= 9999:
            raise ValueError("NOAA annual extreme has an invalid year")
        result.append(
            observed_record(
                "noaa:detrended_annual_high_water",
                identifier,
                lon,
                lat,
                str(year),
                number(row.get("value"), "NOAA annual value", missing=True),
            )
        )
    return result


def noaa_projection_rows(
    payload: object, stations: Sequence[str]
) -> list[dict[str, object]]:
    p = obj(payload, "NOAA sea-level projections")
    raw = array(p.get("SlrProjections"), "NOAA sea-level projections")
    if (
        p.get("count") != len(raw)
        or p.get("totalPages") not in ((0, 1) if not raw else (1,))
        or p.get("currPage") != 1
    ):
        raise ValueError("NOAA sea-level pagination or row count is inconsistent")
    result = []
    for value in raw:
        row = obj(value, "NOAA projection")
        station = text(row.get("stationId"), "NOAA station")
        if (
            station not in stations
            or row.get("reportYear") != 2022
            or row.get("units") != "cm"
        ):
            raise ValueError("NOAA projection station, report year, or units differ")
        year = row.get("projectionYear")
        if type(year) is not int or not 2005 <= year <= 2150:
            raise ValueError("NOAA projection has an invalid projection year")
        scenario = text(row.get("scenario"), "NOAA scenario")
        lon, lat = coordinates(
            [row.get("longitude"), row.get("latitude")], "NOAA station"
        )
        lower = number(
            row.get("projectionCiLow"), "reported lower projection bound", missing=True
        )
        upper = number(
            row.get("projectionCiHigh"), "reported upper projection bound", missing=True
        )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("NOAA reported projection bounds are reversed")
        for field, variable in (
            ("projectionRsl", "noaa:relative_sea_level_change"),
            ("projectionCiLow", "noaa:relative_sea_level_interval_low"),
            ("projectionCiHigh", "noaa:relative_sea_level_interval_high"),
        ):
            numeric = number(row.get(field), field, missing=True)
            record = observed_record(
                variable, station, lon, lat, str(year), numeric, scenario_id=scenario
            )
            record.pop("coverage")
            record.pop("group_id")
            record["probability"] = None
            result.append(record)
    return result


_ISD_ACCEPTED = {"1", "5"}
_GHCN_ACCEPTED = {"", "1", "5"}
_GHCN_ISD_SOURCES = {"313", "314", "315", "322", "335", "343", "344", "346"}
_GHCN_SOURCES = _GHCN_ISD_SOURCES | {"220", "221", "222", "223", "347", "348"}


# One report is kept per timestamp. SYNOP (FM-12) carries tenths of a degree, METAR
# (FM-15) whole degrees and specials (FM-16) are event-driven, so this order keeps
# the most precise routine report. Other types (SAO, AUTO, ...) come last.
_ISD_REPORT_PRIORITY = {"FM-12": 0, "FM-15": 1, "FM-16": 2}


def isd_rows(
    path: Path, station: str, year: int, superseded: list[int] | None = None
) -> Iterator[dict[str, object]]:
    """Decode one ISD station-year with one report per timestamp.

    Among reports sharing a timestamp, the highest-priority report with a valid
    temperature wins (a missing value only when none is valid). The number of
    reports dropped this way is appended to ``superseded`` when it is given.
    """
    chosen: dict[str, tuple[tuple[int, int], dict[str, object]]] = {}
    dropped = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"STATION", "DATE", "LATITUDE", "LONGITUDE", "TMP"}.issubset(
            reader.fieldnames or ()
        ):
            raise ValueError("ISD CSV lacks the required source columns")
        for row in reader:
            if row["STATION"] != station:
                raise ValueError("ISD returned an unrequested station")
            when = row["DATE"]
            if datetime.fromisoformat(when).year != year:
                raise ValueError("ISD returned an unrequested observation year")
            lon, lat = coordinates([row["LONGITUDE"], row["LATITUDE"]], "ISD station")
            pieces = row["TMP"].split(",")
            temperature = number(pieces[0], "ISD temperature")
            assert temperature is not None
            valid = (
                len(pieces) == 2
                and pieces[1] in _ISD_ACCEPTED
                and -932 <= temperature <= 618
            )
            rank = (
                0 if valid else 1,
                _ISD_REPORT_PRIORITY.get(
                    (row.get("REPORT_TYPE") or "").strip(), len(_ISD_REPORT_PRIORITY)
                ),
            )
            record = observed_record(
                "noaa:hourly_air_temperature",
                station,
                lon,
                lat,
                when,
                temperature / 10 if valid else None,
                missing_reason=(
                    "ISD temperature is missing or fails the source quality policy."
                ),
            )
            if when in chosen:
                dropped += 1
                if rank >= chosen[when][0]:
                    continue
            chosen[when] = (rank, record)
    if superseded is not None:
        superseded.append(dropped)
    for _, record in chosen.values():
        yield record


def ghcnh_rows(path: Path, station: str, year: int) -> Iterator[dict[str, object]]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    required = {
        "STATION",
        "DATE",
        "LATITUDE",
        "LONGITUDE",
        "temperature",
        "temperature_Quality_Code",
        "temperature_Source_Code",
    }
    with pq.ParquetFile(path) as source:
        if not required.issubset(source.schema_arrow.names):
            raise ValueError("GHCNh Parquet lacks the required source columns")
        for batch in source.iter_batches(
            columns=sorted(required), batch_size=8192, use_threads=False
        ):
            for row in batch.to_pylist():
                if row["STATION"] != station:
                    raise ValueError("GHCNh returned an unrequested station")
                when = str(row["DATE"]).replace(" ", "T")
                if datetime.fromisoformat(when).year != year:
                    raise ValueError("GHCNh returned an unrequested observation year")
                lon, lat = coordinates(
                    [row["LONGITUDE"], row["LATITUDE"]], "GHCNh station"
                )
                quality = str(row["temperature_Quality_Code"] or "").strip()
                origin = str(row["temperature_Source_Code"] or "").strip()
                temperature = number(
                    row["temperature"], "GHCNh temperature", missing=True
                )
                valid = (
                    quality in _GHCN_ACCEPTED
                    and (
                        quality == ""
                        or quality == "1"
                        and origin in _GHCN_SOURCES
                        or quality == "5"
                        and origin in _GHCN_ISD_SOURCES
                    )
                    and temperature is not None
                    and -93.2 <= temperature <= 61.8
                )
                yield observed_record(
                    "noaa:hourly_air_temperature",
                    station,
                    lon,
                    lat,
                    when,
                    temperature if valid else None,
                    missing_reason=(
                        "GHCNh temperature is missing or fails the source-aware "
                        "quality policy."
                    ),
                )


def ibtracs_rows(path: Path, agency: str | None) -> Iterator[dict[str, object]]:
    # Agency-specific values never borrow another agency's wind convention.
    selected = "USA" if agency is None else agency
    known = {
        "USA",
        "TOKYO",
        "CMA",
        "HKO",
        "NEWDELHI",
        "REUNION",
        "BOM",
        "NADI",
        "WELLINGTON",
        "WMO",
    }
    if selected not in known:
        raise ValueError(
            f"Unknown IBTrACS agency {selected!r}; choose one of {sorted(known)}"
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "SID",
            "ISO_TIME",
            "LAT",
            "LON",
            f"{selected}_WIND",
            f"{selected}_PRES",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("IBTrACS CSV lacks the requested agency columns")
        units = next(reader, None)
        if (
            units is None
            or units[f"{selected}_WIND"].strip() not in {"kts", "knots"}
            or units[f"{selected}_PRES"].strip() not in {"mb", "hPa"}
        ):
            raise ValueError("IBTrACS agency wind/pressure source units are unexpected")
        for raw in reader:
            row = {
                key: value.strip()
                for key, value in raw.items()
                if key is not None and value is not None
            }
            sid = text(row.get("SID"), "IBTrACS storm ID")
            when = text(row.get("ISO_TIME"), "IBTrACS observation time").replace(
                " ", "T"
            )
            datetime.fromisoformat(when)
            lon, lat = (
                number(row.get("LON"), "IBTrACS longitude"),
                number(row.get("LAT"), "IBTrACS latitude"),
            )
            assert lon is not None and lat is not None
            if not -180 <= lon <= 360:
                raise ValueError(
                    "IBTrACS longitude is outside its documented convention"
                )
            lon = (lon + 180) % 360 - 180
            position = f"{sid}:{when}"
            for field, variable in (
                ("WIND", f"ibtracs:{selected}:maximum_sustained_wind"),
                ("PRES", f"ibtracs:{selected}:minimum_central_pressure"),
            ):
                record = observed_record(
                    variable,
                    position,
                    lon,
                    lat,
                    when,
                    number(row.get(f"{selected}_{field}"), field, missing=True),
                    group_id=sid,
                )
                record["station_id"] = None
                yield record
