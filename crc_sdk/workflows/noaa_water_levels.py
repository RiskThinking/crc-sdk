"""NOAA CO-OPS published annual high-water exceedance estimates."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeGuard, cast
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]

from ._noaa import prepare_resource

_ENDPOINT = "https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product.json"
_RESOURCES = ("station", "levels", "annuals")
_LEVEL_SCHEMA = pa.schema(
    [
        ("station_id", pa.string()),
        ("station_name", pa.string()),
        ("longitude", pa.float64()),
        ("latitude", pa.float64()),
        ("datum", pa.string()),
        ("datum_epoch", pa.string()),
        ("annual_exceedance_probability", pa.float64()),
        ("published_return_period_years", pa.float64()),
        ("water_level_m", pa.float64()),
        ("lower_90ci_m", pa.float64()),
        ("upper_90ci_m", pa.float64()),
    ]
)
_ANNUAL_SCHEMA = pa.schema(
    [
        ("station_id", pa.string()),
        ("year", pa.int32()),
        ("detrended_extreme_m", pa.float64()),
        ("datum_label", pa.string()),
        ("datum_epoch", pa.string()),
        ("reference_notice", pa.string()),
    ]
)


@dataclass(frozen=True)
class NOAAExtremeWaterLevelPlan:
    """Station-point NOAA high-water estimates; no datum transformation is applied."""

    station_id: str
    cache_dir: Path | None = None
    mode: str = "reuse"
    checksums: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9]{7}", self.station_id):
            raise ValueError("station_id must be a seven-digit NOAA station ID")
        if self.mode not in {"reuse", "offline", "refresh"}:
            raise ValueError("mode must be reuse, offline or refresh")
        if any(
            key not in _RESOURCES or not re.fullmatch(r"[0-9a-f]{64}", value)
            for key, value in self.checksums
        ):
            raise ValueError(
                "checksums require resource names and lowercase SHA-256 digests"
            )

    @classmethod
    def for_station(cls, station_id: str) -> NOAAExtremeWaterLevelPlan:
        return cls(station_id)

    def cache(
        self,
        directory: str | Path,
        *,
        mode: str = "reuse",
        sha256: Mapping[str, str] | None = None,
    ) -> NOAAExtremeWaterLevelPlan:
        return replace(
            self,
            cache_dir=Path(directory),
            mode=mode,
            checksums=tuple(sorted((sha256 or {}).items())),
        )

    def _url(self, resource: str) -> str:
        parameters = {
            "name": "extremewaterlevels",
            "station": self.station_id,
            "units": "metric",
        }
        if resource == "levels":
            parameters.update({"details": "exceedancelevels", "levelType": "high"})
        elif resource == "annuals":
            parameters.update({"extremeType": "annuals", "levelType": "high"})
        elif resource != "station":
            raise ValueError("unknown NOAA resource")
        return _ENDPOINT + "?" + urlencode(parameters)

    def _directory(self, resource: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache(directory) must be configured")
        return self.cache_dir / "noaa-extreme-water-levels" / self.station_id / resource

    def explain(self) -> dict[str, object]:
        return {
            "station_id": self.station_id,
            "urls": {key: self._url(key) for key in _RESOURCES},
            "mode": self.mode,
            "cache_dir": str(self.cache_dir),
            "meaning": "published annual high-water exceedance levels at a station",
            "bounds": "confidence intervals are not return-probability knots",
        }

    def _fetch(self, resource: str) -> dict[str, Any]:
        return prepare_resource(
            directory=self._directory(resource),
            url=self._url(resource),
            mode=self.mode,
            expected=dict(self.checksums).get(resource),
            opener=urlopen,
            validate=lambda payload: _validate_resource(
                payload, resource, self.station_id
            ),
            metadata={"station_id": self.station_id, "resource": resource},
        )

    def _payload(self, resource: str) -> dict[str, Any]:
        self._fetch(resource)
        payload = json.loads((self._directory(resource) / "source.json").read_bytes())
        _validate_resource(payload, resource, self.station_id)
        return cast(dict[str, Any], payload)

    def prefetch(self) -> dict[str, dict[str, Any]]:
        return {resource: self._fetch(resource) for resource in _RESOURCES}

    def _context(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        station = self._payload("station")["ExtremeWaterLevels"][0]
        levels = self._payload("levels")
        annual = self._payload("annuals")
        if levels["datum"].upper() != annual["datum"].upper():
            raise ValueError("NOAA annual and exceedance datum labels differ")
        info = {
            "station": station,
            "datum": levels["datum"],
            "units": levels["units"],
            "trend": annual.get("trend"),
            "trend_units": annual.get("trendUnits"),
            "trend_start_date": annual.get("trendStartDate"),
            "trend_end_date": annual.get("trendEndDate"),
            "annual_notice": annual.get("notice"),
        }
        return info, levels, annual

    def source_info(self) -> dict[str, Any]:
        """Return station, datum and NOAA's detrending context."""
        return self._context()[0]

    def source_levels(self) -> pa.Table:
        """Return published AEPs, levels and confidence intervals separately."""
        info, levels, _ = self._context()
        station = info["station"]
        rows = [
            {
                "station_id": self.station_id,
                "station_name": station.get("stationName"),
                "longitude": station.get("longitude"),
                "latitude": station.get("latitude"),
                "datum": info["datum"],
                "datum_epoch": station.get("epoch"),
                "annual_exceedance_probability": row["probability"],
                "published_return_period_years": row["returnPeriod"],
                "water_level_m": row["level"],
                "lower_90ci_m": row.get("lower90CI"),
                "upper_90ci_m": row.get("upper90CI"),
            }
            for row in levels["GEV_HI"]["exceedanceLevels"]
        ]
        return pa.Table.from_pylist(rows, schema=_LEVEL_SCHEMA)

    def annual_extremes(self) -> pa.Table:
        """Return detrended annual maxima, retaining NOAA's distinct interpretation."""
        info, _, annual = self._context()
        rows = [
            {
                "station_id": self.station_id,
                "year": row["year"],
                "detrended_extreme_m": row["value"],
                "datum_label": annual["datum"],
                "datum_epoch": info["station"]["epoch"],
                "reference_notice": annual["notice"],
            }
            for row in annual["extremes"]
        ]
        return pa.Table.from_pylist(rows, schema=_ANNUAL_SCHEMA)


def _number(value: Any) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _validate_resource(payload: dict[str, Any], resource: str, station: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError("invalid NOAA response")
    if resource == "station":
        stations = payload.get("ExtremeWaterLevels")
        if (
            not isinstance(stations, list)
            or len(stations) != 1
            or not isinstance(stations[0], dict)
            or stations[0].get("stationId") != station
        ):
            raise ValueError("NOAA station metadata missing or mismatched")
        item = stations[0]
        lon, lat = item.get("longitude"), item.get("latitude")
        if (
            not _number(lon)
            or not _number(lat)
            or not -180 <= lon <= 180
            or not -90 <= lat <= 90
        ):
            raise ValueError("NOAA station coordinates missing or invalid")
        if not isinstance(item.get("epoch"), str) or not item["epoch"]:
            raise ValueError("NOAA station datum epoch missing")
        return
    if (
        payload.get("stationId") != station
        or payload.get("units") != "meters"
        or not isinstance(payload.get("datum"), str)
        or not payload["datum"]
    ):
        raise ValueError("NOAA station, metric units or datum missing or mismatched")
    if resource == "annuals":
        rows = payload.get("extremes")
        if not isinstance(payload.get("notice"), str) or not payload["notice"]:
            raise ValueError("NOAA annual detrending notice missing")
        if payload.get("trend") is not None and not _number(payload["trend"]):
            raise ValueError("invalid NOAA annual detrending trend")
        if not isinstance(rows, list) or any(
            not isinstance(row.get("year"), int) or not _number(row.get("value"))
            for row in rows
        ):
            raise ValueError("invalid NOAA annual extremes")
        return
    group = payload.get("GEV_HI")
    rows = group.get("exceedanceLevels") if isinstance(group, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("NOAA annual high-water levels are unavailable")
    probabilities: set[float] = set()
    for row in rows:
        p, period, level = (
            row.get("probability"),
            row.get("returnPeriod"),
            row.get("level"),
        )
        low, high = row.get("lower90CI"), row.get("upper90CI")
        if (
            not _number(p)
            or not 0 < p <= 1
            or p in probabilities
            or not _number(period)
            or period < 1
            or not _number(level)
        ):
            raise ValueError("invalid NOAA annual exceedance level")
        if (low is not None and (not _number(low) or low > level)) or (
            high is not None and (not _number(high) or high < level)
        ):
            raise ValueError("invalid NOAA confidence interval")
        probabilities.add(p)
