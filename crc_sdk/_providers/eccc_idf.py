"""ECCC rainfall IDF constants and strict station-text decoding."""

from __future__ import annotations

import math
import re
from typing import Any

ROOT = "https://collaboration.cmc.ec.gc.ca/cmc/climate/Engineer_Climate/IDF"
LICENCE = "OGL-Canada-2.0"
ATTRIBUTION = (
    "Derived from Environment and Climate Change Canada rainfall IDF. "
    "Contains information licensed under the Open Government Licence – Canada."
)
PROVINCES = frozenset("AB BC MB NB NL NS NT NU ON PE QC SK YT".split())
DURATIONS = {
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "2h": 120,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
}


def parse_station(raw: bytes, station: str, province: str) -> dict[str, Any]:
    """Decode central rates and confidence half-widths; never fit annual samples."""
    text = raw.decode("cp1252")
    identity = re.search(rf"^\s*(\S[^\r\n]*?)\s+{province}\s+{station}\s*$", text, re.M)
    coordinates = re.search(
        r"Latitude:\s*(\d+)\s+(\d+(?:\.\d+)?)'([NS])\s+"
        r"Longitude:\s*(\d+)\s+(\d+(?:\.\d+)?)'([EW])",
        text,
    )
    years = re.search(r"Years/Ann[^:\r\n]*:\s*(\d{4})\s*-\s*(\d{4})", text)
    if not identity or not coordinates or not years:
        raise ValueError(f"invalid ECCC station header for {station}")
    lat_deg, lat_min, ns, lon_deg, lon_min, ew = coordinates.groups()
    latitude = (int(lat_deg) + float(lat_min) / 60) * (-1 if ns == "S" else 1)
    longitude = (int(lon_deg) + float(lon_min) / 60) * (-1 if ew == "W" else 1)
    start, end = map(int, years.groups())
    if not (
        0 <= float(lat_min) < 60
        and 0 <= float(lon_min) < 60
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
        and start <= end
    ):
        raise ValueError(f"invalid ECCC station coordinates or years for {station}")
    try:
        section = text.split("Table 2b :", 1)[1].split("Table 3", 1)[0]
    except IndexError as error:
        raise ValueError(f"missing ECCC rainfall rates for {station}") from error
    header = re.search(r"^\s*Duration/[^\r\n]+", section, re.M)
    if not header:
        raise ValueError(f"missing ECCC return-period header for {station}")
    labels = re.search(r"\s(\d+(?:\s+\d+)*)\s+#Years\s*$", header[0])
    if not labels:
        raise ValueError(f"invalid ECCC return-period labels for {station}")
    periods = [float(value) for value in labels[1].split()]
    if (
        len(periods) < 4
        or any(t <= 1 for t in periods)
        or periods != sorted(set(periods))
    ):
        raise ValueError(f"invalid ECCC return-period axis for {station}")
    curves: list[dict[str, Any]] = []
    for line in section.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s+(min|h)\s+(.+)", line)
        if match:
            tokens = match[3].split()
            if len(tokens) != len(periods) + 1:
                raise ValueError(f"mismatched ECCC rainfall rate axis for {station}")
            rates = [float(value) for value in tokens[:-1]]
            count = int(tokens[-1])
            if (
                any(not math.isfinite(v) or v < 0 for v in rates)
                or any(a > b for a, b in zip(rates, rates[1:]))
                or not 1 <= count <= end - start + 1
            ):
                raise ValueError(f"invalid ECCC rainfall rates for {station}")
            duration = int(match[1]) * (60 if match[2] == "h" else 1)
            curves.append(
                {
                    "duration_minutes": duration,
                    "intensity": rates,
                    "record_years": count,
                    "confidence_half_width": None,
                }
            )
        elif "+/-" in line and curves:
            tokens = line.replace("+/-", " ").split()
            if len(tokens) != len(periods) + 1:
                raise ValueError(f"mismatched ECCC confidence axis for {station}")
            widths = [float(value) for value in tokens[:-1]]
            if (
                any(not math.isfinite(v) or v < 0 for v in widths)
                or int(tokens[-1]) != curves[-1]["record_years"]
                or curves[-1]["confidence_half_width"] is not None
            ):
                raise ValueError(f"invalid ECCC confidence limits for {station}")
            curves[-1]["confidence_half_width"] = widths
    durations = [curve["duration_minutes"] for curve in curves]
    if sorted(durations) != sorted(DURATIONS.values()) or any(
        curve["confidence_half_width"] is None for curve in curves
    ):
        raise ValueError(f"incomplete or duplicate ECCC durations for {station}")
    return {
        "station": station,
        "name": identity[1].strip(),
        "province": province,
        "latitude": latitude,
        "longitude": longitude,
        "start_year": start,
        "end_year": end,
        "return_periods": periods,
        "curves": curves,
    }
