"""Daily climate observations from ECCC's GeoMet open-data service."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]

from ._geomet import prepare_snapshot

_ENDPOINT = "https://api.weather.gc.ca/collections/climate-daily/items"
_NUMERIC = (
    "MEAN_TEMPERATURE",
    "MIN_TEMPERATURE",
    "MAX_TEMPERATURE",
    "TOTAL_PRECIPITATION",
    "TOTAL_RAIN",
    "TOTAL_SNOW",
    "SNOW_ON_GROUND",
    "DIRECTION_MAX_GUST",
    "SPEED_MAX_GUST",
    "COOLING_DEGREE_DAYS",
    "HEATING_DEGREE_DAYS",
    "MIN_REL_HUMIDITY",
    "MAX_REL_HUMIDITY",
)
_SCHEMA = pa.schema(
    [
        ("CLIMATE_IDENTIFIER", pa.string()),
        ("STATION_NAME", pa.string()),
        ("STN_ID", pa.int64()),
        ("PROVINCE_CODE", pa.string()),
        ("LOCAL_DATE", pa.string()),
        ("date", pa.date32()),
        ("longitude", pa.float64()),
        ("latitude", pa.float64()),
        ("feature_id", pa.string()),
        *[(name, pa.float64()) for name in _NUMERIC],
        *[(name + "_FLAG", pa.string()) for name in _NUMERIC],
        ("properties_json", pa.string()),
    ]
)


@dataclass(frozen=True)
class ECCCDailyPlan:
    """A lazy request for GeoMet daily observations and their source flags."""

    climate_id: str
    selected_years: tuple[int, ...] = ()
    cache_dir: Path | None = None
    mode: str = "reuse"
    checksums: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z0-9]{7}", self.climate_id):
            raise ValueError("climate_id must be a seven-character ECCC climate ID")
        if self.mode not in {"reuse", "offline", "refresh"}:
            raise ValueError("mode must be reuse, offline or refresh")
        if any(not re.fullmatch(r"[0-9a-f]{64}", sha) for _, sha in self.checksums):
            raise ValueError("checksums must be lowercase SHA-256 digests")

    @classmethod
    def for_station(cls, climate_id: str) -> ECCCDailyPlan:
        return cls(climate_id)

    def years(self, start: int, end: int) -> ECCCDailyPlan:
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1800
            or end > 9999
            or start > end
        ):
            raise ValueError("years must be ordered calendar years from 1800 onward")
        return replace(self, selected_years=tuple(range(start, end + 1)))

    def cache(
        self,
        directory: str | Path,
        *,
        mode: str = "reuse",
        sha256: Mapping[int, str] | None = None,
    ) -> ECCCDailyPlan:
        return replace(
            self,
            cache_dir=Path(directory),
            mode=mode,
            checksums=tuple(sorted((sha256 or {}).items())),
        )

    def _url(self, year: int) -> str:
        return (
            _ENDPOINT
            + "?"
            + urlencode(
                {
                    "f": "json",
                    "CLIMATE_IDENTIFIER": self.climate_id,
                    "datetime": (
                        f"{year:04d}-01-01T00:00:00Z/{year:04d}-12-31T23:59:59Z"
                    ),
                    "limit": 500,
                }
            )
        )

    def _directory(self, year: int) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache(directory) must be configured")
        return self.cache_dir / "eccc-geomet-daily" / self.climate_id / str(year)

    def explain(self) -> dict[str, object]:
        if not self.selected_years:
            raise ValueError("years(start, end) must be configured")
        return {
            "climate_id": self.climate_id,
            "years": self.selected_years,
            "urls": tuple(self._url(year) for year in self.selected_years),
            "mode": self.mode,
            "cache_dir": str(self.cache_dir),
            "meaning": "daily observations and source flags; no QC policy applied",
        }

    def _prepare_year(self, year: int) -> dict[str, Any]:
        return prepare_snapshot(
            directory=self._directory(year),
            url=self._url(year),
            mode=self.mode,
            expected=dict(self.checksums).get(year),
            label="ECCC daily",
            opener=urlopen,
            validate=lambda feature: _observation(feature, self.climate_id, year),
            metadata={"climate_id": self.climate_id, "year": year},
        )

    def prefetch(self) -> tuple[dict[str, Any], ...]:
        if not self.selected_years:
            raise ValueError("years(start, end) must be configured")
        if set(dict(self.checksums)) - set(self.selected_years):
            raise ValueError("checksum years must belong to the selected years")
        return tuple(self._prepare_year(year) for year in self.selected_years)

    def batches(self, *, batch_size: int = 4096) -> Iterator[pa.RecordBatch]:
        """Stream typed daily observations without interpreting quality flags."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.selected_years:
            raise ValueError("years(start, end) must be configured")
        if set(dict(self.checksums)) - set(self.selected_years):
            raise ValueError("checksum years must belong to the selected years")
        for year in self.selected_years:
            self._prepare_year(year)
            path = self._directory(year) / "observations.jsonl"
            rows: list[dict[str, Any]] = []
            with path.open(encoding="utf-8") as source:
                for line in source:
                    rows.append(_observation(json.loads(line), self.climate_id, year))
                    if len(rows) == batch_size:
                        yield pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)
                        rows.clear()
                if rows:
                    yield pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)


def _observation(feature: dict[str, Any], climate_id: str, year: int) -> dict[str, Any]:
    properties = feature["properties"]
    try:
        day = date.fromisoformat(properties["LOCAL_DATE"][:10])
        if (
            feature.get("type") != "Feature"
            or properties.get("CLIMATE_IDENTIFIER") != climate_id
            or day.year != year
            or properties.get("LOCAL_YEAR") != day.year
            or properties.get("LOCAL_MONTH") != day.month
            or properties.get("LOCAL_DAY") != day.day
        ):
            raise ValueError("selection mismatch")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid ECCC daily observation or selection") from error
    geometry = feature.get("geometry")
    coordinates = (
        geometry.get("coordinates")
        if geometry and geometry.get("type") == "Point"
        else None
    )
    row: dict[str, Any] = {
        key: properties.get(key)
        for key in (
            "CLIMATE_IDENTIFIER",
            "STATION_NAME",
            "STN_ID",
            "PROVINCE_CODE",
            "LOCAL_DATE",
        )
    }
    row.update(
        {
            "date": day,
            "feature_id": feature.get("id"),
            "longitude": coordinates[0] if coordinates else None,
            "latitude": coordinates[1] if coordinates else None,
            "properties_json": json.dumps(
                properties, separators=(",", ":"), sort_keys=True
            ),
        }
    )
    for name in _NUMERIC:
        value = properties.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid ECCC daily value for {name}")
        row[name] = value
        row[name + "_FLAG"] = properties.get(name + "_FLAG")
    return row
