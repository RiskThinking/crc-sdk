"""Historical Canadian daily mean hydrometric observations via GeoMet."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]

from ._geomet import GeoMetDailyPlan, prepare_snapshot

_ENDPOINT = "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"
_SCHEMA = pa.schema(
    [
        ("station_number", pa.string()),
        ("station_name", pa.string()),
        ("province", pa.string()),
        ("date", pa.date32()),
        ("discharge_m3_s", pa.float64()),
        ("discharge_symbol_en", pa.string()),
        ("discharge_symbol_fr", pa.string()),
        ("level_m", pa.float64()),
        ("level_symbol_en", pa.string()),
        ("level_symbol_fr", pa.string()),
        ("longitude", pa.float64()),
        ("latitude", pa.float64()),
        ("feature_id", pa.string()),
        ("properties_json", pa.string()),
    ]
)


@dataclass(frozen=True)
class HYDATDailyPlan(GeoMetDailyPlan):
    """Daily mean gauge observations; water-level datum is not inferred."""

    station_number: str
    selected_years: tuple[int, ...] = ()
    cache_dir: Path | None = None
    mode: str = "reuse"
    checksums: tuple[tuple[int, str], ...] = ()

    _schema = _SCHEMA

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9]{2}[A-Z]{2}[0-9]{3}", self.station_number):
            raise ValueError("station_number must be a Water Survey of Canada ID")
        self._check_options()

    @classmethod
    def for_station(cls, station_number: str) -> HYDATDailyPlan:
        return cls(station_number)

    def _url(self, year: int) -> str:
        return (
            _ENDPOINT
            + "?"
            + urlencode(
                {
                    "f": "json",
                    "STATION_NUMBER": self.station_number,
                    "datetime": f"{year:04d}-01-01/{year:04d}-12-31",
                    "limit": 500,
                }
            )
        )

    def _directory(self, year: int) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache(directory) must be configured")
        return self.cache_dir / "hydat-geomet-daily" / self.station_number / str(year)

    def explain(self) -> dict[str, object]:
        if not self.selected_years:
            raise ValueError("years(start, end) must be configured")
        return {
            "station_number": self.station_number,
            "years": self.selected_years,
            "urls": tuple(self._url(year) for year in self.selected_years),
            "mode": self.mode,
            "cache_dir": str(self.cache_dir),
            "discharge_unit": "m3/s",
            "level_unit": "m",
            "level_datum": "unspecified",
            "meaning": "daily mean station flow and level, not asset flood depth",
        }

    def _prepare_year(self, year: int) -> dict[str, Any]:
        return prepare_snapshot(
            directory=self._directory(year),
            url=self._url(year),
            mode=self.mode,
            expected=dict(self.checksums).get(year),
            label="HYDAT daily",
            opener=urlopen,
            validate=lambda feature: _observation(feature, self.station_number, year),
            metadata={
                "station_number": self.station_number,
                "year": year,
                "discharge_unit": "m3/s",
                "level_unit": "m",
                "level_datum": "unspecified",
            },
        )

    def _parse(self, feature: dict[str, Any], year: int) -> dict[str, Any]:
        return _observation(feature, self.station_number, year)


def _observation(feature: dict[str, Any], station: str, year: int) -> dict[str, Any]:
    properties = feature["properties"]
    try:
        day = date.fromisoformat(properties["DATE"])
        if (
            feature.get("type") != "Feature"
            or properties.get("STATION_NUMBER") != station
            or day.year != year
        ):
            raise ValueError("selection mismatch")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid HYDAT daily observation or selection") from error
    values: dict[str, Any] = {}
    for name in ("DISCHARGE", "LEVEL"):
        value = properties.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid HYDAT daily value for {name}")
        values[name] = value
    geometry = feature.get("geometry")
    coordinates = (
        geometry.get("coordinates")
        if geometry and geometry.get("type") == "Point"
        else None
    )
    return {
        "station_number": station,
        "station_name": properties.get("STATION_NAME"),
        "province": properties.get("PROV_TERR_STATE_LOC"),
        "date": day,
        "discharge_m3_s": values["DISCHARGE"],
        "level_m": values["LEVEL"],
        "discharge_symbol_en": properties.get("DISCHARGE_SYMBOL_EN"),
        "discharge_symbol_fr": properties.get("DISCHARGE_SYMBOL_FR"),
        "level_symbol_en": properties.get("LEVEL_SYMBOL_EN"),
        "level_symbol_fr": properties.get("LEVEL_SYMBOL_FR"),
        "longitude": coordinates[0] if coordinates else None,
        "latitude": coordinates[1] if coordinates else None,
        "feature_id": feature.get("id"),
        "properties_json": json.dumps(
            properties, separators=(",", ":"), sort_keys=True
        ),
    }
