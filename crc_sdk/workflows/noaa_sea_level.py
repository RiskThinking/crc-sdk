"""NOAA 2022 relative sea-level scenarios at a specified U.S. station."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]

from ._noaa import prepare_resource

_ENDPOINT = "https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product/slr_projections.json"
_SCENARIOS = {"low", "intermediate-low", "intermediate", "intermediate-high", "high"}
_SCHEMA = pa.schema(
    [
        ("station_id", pa.string()),
        ("station_name", pa.string()),
        ("longitude", pa.float64()),
        ("latitude", pa.float64()),
        ("report_year", pa.int16()),
        ("baseline_year", pa.int16()),
        ("projection_year", pa.int16()),
        ("scenario", pa.string()),
        ("relative_change_cm", pa.float64()),
        ("interval_low_cm", pa.float64()),
        ("interval_high_cm", pa.float64()),
        ("properties_json", pa.string()),
    ]
)


@dataclass(frozen=True)
class NOAASeaLevelPlan:
    """2022 interagency scenarios, referenced to the API's 2005 baseline."""

    station_id: str
    selected_scenario: str = "all"
    cache_dir: Path | None = None
    mode: str = "reuse"
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9]{7}", self.station_id):
            raise ValueError("station_id must be a seven-digit U.S. NOAA station ID")
        if self.selected_scenario not in _SCENARIOS | {"all"}:
            raise ValueError(f"scenario must be all or one of {sorted(_SCENARIOS)}")
        if self.mode not in {"reuse", "offline", "refresh"}:
            raise ValueError("mode must be reuse, offline or refresh")
        if self.expected_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_sha256
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def for_station(cls, station_id: str) -> NOAASeaLevelPlan:
        return cls(station_id)

    def scenario(self, value: str) -> NOAASeaLevelPlan:
        return replace(self, selected_scenario=value.lower())

    def cache(
        self, directory: str | Path, *, mode: str = "reuse", sha256: str | None = None
    ) -> NOAASeaLevelPlan:
        return replace(
            self, cache_dir=Path(directory), mode=mode, expected_sha256=sha256
        )

    @property
    def url(self) -> str:
        return (
            _ENDPOINT
            + "?"
            + urlencode(
                {
                    "station": self.station_id,
                    "affil": "US",
                    "units": "metric",
                    "report_year": 2022,
                    "station_or_grid": "station",
                    "coastal_flag": 1,
                    "scenario": self.selected_scenario,
                }
            )
        )

    def _directory(self) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache(directory) must be configured")
        return (
            self.cache_dir
            / "noaa-sea-level-2022"
            / self.station_id
            / self.selected_scenario
        )

    def explain(self) -> dict[str, object]:
        return {
            "url": self.url,
            "report_year": 2022,
            "baseline_year": 2005,
            "scenario": self.selected_scenario,
            "source_unit": "cm",
            "cache_dir": str(self.cache_dir),
            "mode": self.mode,
            "meaning": "relative sea-level change; no annual exceedance or flood depth",
        }

    def prefetch(self) -> dict[str, Any]:
        return prepare_resource(
            directory=self._directory(),
            url=self.url,
            mode=self.mode,
            expected=self.expected_sha256,
            opener=urlopen,
            validate=lambda payload: _validate(
                payload, self.station_id, self.selected_scenario
            ),
            metadata={
                "station_id": self.station_id,
                "report_year": 2022,
                "baseline_year": 2005,
                "scenario": self.selected_scenario,
                "source_unit": "cm",
                "baseline_documentation": "https://cdn.oceanservice.noaa.gov/oceanserviceprod/hazards/sealevelrise/noaa-nos-techrpt02-global-regional-SLR-scenarios-US-application-guide.pdf",
            },
        )

    def projections(self) -> pa.Table:
        """Return scenario changes and intervals without probability inference."""
        self.prefetch()
        payload = json.loads((self._directory() / "source.json").read_bytes())
        _validate(payload, self.station_id, self.selected_scenario)
        rows = [
            {
                "station_id": self.station_id,
                "station_name": row.get("stationName"),
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "report_year": 2022,
                "baseline_year": 2005,
                "projection_year": row["projectionYear"],
                "scenario": row["scenario"],
                "relative_change_cm": row["projectionRsl"],
                "interval_low_cm": row.get("projectionCiLow"),
                "interval_high_cm": row.get("projectionCiHigh"),
                "properties_json": json.dumps(
                    row, sort_keys=True, separators=(",", ":")
                ),
            }
            for row in payload["SlrProjections"]
        ]
        return pa.Table.from_pylist(rows, schema=_SCHEMA)


def _number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _validate(payload: dict[str, Any], station: str, scenario: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError("invalid NOAA sea-level response")
    rows = payload.get("SlrProjections")
    if not isinstance(rows, list) or not rows:
        raise ValueError("NOAA sea-level projections unavailable for this selection")
    if payload.get("currPage") != 1 or payload.get("totalPages") != 1:
        raise ValueError("NOAA sea-level response requires unsupported pagination")
    if payload.get("count") != len(rows):
        raise ValueError("NOAA sea-level response count mismatch")
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid NOAA sea-level projection row")
        year, label = row.get("projectionYear"), row.get("scenario")
        if (
            row.get("stationId") != station
            or row.get("reportYear") != 2022
            or row.get("units") != "cm"
            or not isinstance(year, int)
            or isinstance(year, bool)
            or not 2005 <= year <= 2150
            or not isinstance(label, str)
            or label.lower() not in _SCENARIOS
            or (scenario != "all" and label.lower() != scenario)
        ):
            raise ValueError("NOAA sea-level selection, units or report mismatch")
        key = (year, label.lower())
        if key in seen:
            raise ValueError("duplicate NOAA sea-level scenario and year")
        seen.add(key)
        if not _number(row.get("longitude")) or not _number(row.get("latitude")):
            raise ValueError("invalid NOAA station coordinates")
        lon, lat = cast(float, row["longitude"]), cast(float, row["latitude"])
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("invalid NOAA station coordinates")
        if not _number(row.get("projectionRsl")):
            raise ValueError("invalid NOAA sea-level projection")
        low, high = row.get("projectionCiLow"), row.get("projectionCiHigh")
        if (
            (low is not None and not _number(low))
            or (high is not None and not _number(high))
            or (low is not None and high is not None and low > high)
        ):
            raise ValueError("invalid NOAA sea-level interval")
