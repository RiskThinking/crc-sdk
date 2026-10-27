"""ECCC's published short-duration rainfall rates and station archives."""

from __future__ import annotations

import hashlib
import io
import math
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast
from urllib.request import Request, urlopen

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


def archive_url(version: str, province: str) -> str:
    if province not in PROVINCES:
        raise ValueError(f"province must be one of {sorted(PROVINCES)}")
    if version == "3.40":
        return f"{ROOT}/idf_v3-40_2025-12-5.zip"
    if version == "3.20":
        if province in {"MB", "NS", "PE"}:
            raise ValueError(f"release 3.20 has no {province} archive")
        return (
            f"{ROOT}/IDF_archive/idf_v-3.20_2021_3_26/IDF_Files_Fichiers/"
            f"IDF_v-3.20_2021_03_26_{province}.zip"
        )
    raise ValueError("supported ECCC IDF releases are 3.20 and 3.40")


class _RangeReader(io.RawIOBase):
    """Seekable, validator-pinned HTTPS reads for ZIP central directories."""

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.position = 0
        with urlopen(Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
            self.last_modified = response.headers.get("Last-Modified")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence not in (0, 1, 2):
            raise ValueError("invalid ZIP seek origin")
        position = offset + (0, self.position, self.size)[whence]
        if position < 0:
            raise ValueError("invalid ZIP seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.position + size, self.size)
        if end <= self.position:
            return b""
        headers = {"Range": f"bytes={self.position}-{end - 1}"}
        if self.etag:
            headers["If-Match"] = self.etag
        elif self.last_modified:
            headers["If-Unmodified-Since"] = self.last_modified
        with urlopen(Request(self.url, headers=headers), timeout=60) as response:
            expected = f"bytes {self.position}-{end - 1}/{self.size}"
            if (
                response.status != 206
                or response.headers.get("Content-Range") != expected
            ):
                raise RuntimeError("ECCC archive requires HTTP byte-range support")
            data = response.read(end - self.position + 1)
        if len(data) != end - self.position:
            raise RuntimeError("incomplete ECCC archive range")
        self.position = end
        return cast(bytes, data)


@contextmanager
def _province_archive(
    reader: _RangeReader, version: str, province: str
) -> Iterator[tuple[zipfile.ZipFile, str]]:
    with zipfile.ZipFile(reader) as outer:
        if version == "3.20":
            yield outer, ""
        else:
            member = f"idf_v3-40_2025-12-5/IDF_Files_Fichiers/{province}.zip"
            with tempfile.TemporaryFile() as scratch:
                with outer.open(member) as source:
                    shutil.copyfileobj(source, scratch, length=1024 * 1024)
                scratch.seek(0)
                with zipfile.ZipFile(scratch) as inner:
                    yield inner, member + "!"


def fetch_stations(
    version: str, province: str, stations: Sequence[str]
) -> Iterator[tuple[str, bytes, dict[str, Any]]]:
    """Yield selected raw station files; an empty selection reads the province."""
    url = archive_url(version, province)
    with (
        _RangeReader(url) as reader,
        _province_archive(reader, version, province) as pair,
    ):
        archive, prefix = pair
        members: dict[str, str] = {}
        for name in archive.namelist():
            match = re.search(rf"_{province}_([A-Z0-9]{{7}})_.+\.txt$", name, re.I)
            if match:
                station = match[1].upper()
                if station in members:
                    raise ValueError(f"ambiguous ECCC station {station}")
                members[station] = name
        missing = set(stations) - members.keys()
        if missing:
            raise ValueError(
                f"ECCC {province} release {version} has no stations {sorted(missing)}"
            )
        for station in sorted(stations or members):
            member = members[station]
            raw = archive.read(member)
            yield (
                station,
                raw,
                {
                    "url": url,
                    "member": prefix + member,
                    "etag": reader.etag,
                    "last_modified": reader.last_modified,
                    "checksum": hashlib.sha256(raw).hexdigest(),
                },
            )


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
