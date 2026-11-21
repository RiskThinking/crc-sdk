"""Published ERA5 source layout and CF timestamp decoding."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

ERA5_LICENCE = "CC-BY-4.0"
ERA5_ATTRIBUTION = "Contains modified Copernicus Climate Change Service information"


@dataclass(frozen=True, slots=True)
class ERA5Store:
    name: str
    url: str
    resolution_degrees: float
    variables: Mapping[str, str]
    lat_name: str = "latitude"
    lon_name: str = "longitude"
    time_name: str = "time"


ARCO_0P25 = ERA5Store(
    name="arco-0p25",
    url="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    resolution_degrees=0.25,
    variables={"t2m": "2m_temperature", "tp": "total_precipitation"},
)


def _decode_time(array: Any) -> np.ndarray[Any, Any]:
    """CF ``<unit> since <date>`` integer time -> ``datetime64[s]``."""
    units = str(array.attrs.get("units", ""))
    match = re.match(r"\s*(hours|days|seconds)\s+since\s+(\S+)(?:\s+(\S+))?", units)
    if match is None:
        raise ValueError(f"unsupported time units {units!r}")
    unit, date, clock = match.groups()
    origin = np.datetime64(f"{date}T{clock}" if clock else date, "s")
    seconds = {"hours": 3600, "days": 86400, "seconds": 1}[unit]
    raw = np.asarray(array[:], dtype=np.int64)
    return origin + raw * np.timedelta64(seconds, "s")
