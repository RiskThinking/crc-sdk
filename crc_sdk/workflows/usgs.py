"""USGS daily mean streamflow observations at explicitly selected gauges."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]

from ._remote import file_checksum, read_manifest, write_manifest

_ENDPOINT = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items"
_SCHEMA = pa.schema(
    [
        ("station", pa.string()),
        ("date", pa.date32()),
        ("flow", pa.float64()),
        ("unit", pa.string()),
        ("approval_status", pa.string()),
        ("qualifier_json", pa.string()),
        ("time_series_id", pa.string()),
        ("last_modified", pa.string()),
        ("longitude", pa.float64()),
        ("latitude", pa.float64()),
        ("observation_id", pa.string()),
    ]
)


@dataclass(frozen=True)
class USGSStreamflowPlan:
    """A lazy request for USGS daily mean discharge, in source units."""

    station: str
    start: date | None = None
    end: date | None = None
    cache_dir: Path | None = None
    mode: str = "reuse"
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"USGS-\d{8,15}", self.station):
            raise ValueError("station must be a USGS monitoring location ID")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start date must not follow end date")
        if self.mode not in {"reuse", "offline", "refresh"}:
            raise ValueError("mode must be reuse, offline or refresh")
        if self.expected_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_sha256
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def for_station(cls, station: str) -> USGSStreamflowPlan:
        return cls(station=station)

    def between(self, start: str, end: str) -> USGSStreamflowPlan:
        """Select an inclusive ISO-date interval."""
        return replace(
            self, start=date.fromisoformat(start), end=date.fromisoformat(end)
        )

    def cache(
        self, directory: str | Path, *, mode: str = "reuse", sha256: str | None = None
    ) -> USGSStreamflowPlan:
        return replace(
            self, cache_dir=Path(directory), mode=mode, expected_sha256=sha256
        )

    def _url(self) -> str:
        if self.start is None or self.end is None:
            raise ValueError("between(start, end) must be configured")
        return (
            _ENDPOINT
            + "?"
            + urlencode(
                {
                    "f": "json",
                    "monitoring_location_id": self.station,
                    "parameter_code": "00060",
                    "statistic_id": "00003",
                    "datetime": f"{self.start.isoformat()}/{self.end.isoformat()}",
                    "limit": 500,
                }
            )
        )

    def explain(self) -> dict[str, object]:
        return {
            "url": self._url(),
            "mode": self.mode,
            "cache_dir": str(self.cache_dir),
            "expected_sha256": self.expected_sha256,
            "meaning": "daily mean gauge discharge, not inundation depth",
        }

    def _directory(self) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache(directory) must be configured")
        key = hashlib.sha256(self._url().encode()).hexdigest()
        return self.cache_dir / "usgs-daily" / key

    def prefetch(self) -> dict[str, Any]:
        """Fetch a bounded page at a time and checksum the observation snapshot."""
        url = self._url()
        directory = self._directory()
        path = directory / "observations.jsonl"
        manifest = read_manifest(directory)
        if self.mode != "refresh" and path.exists() != (manifest is not None):
            raise RuntimeError("incomplete USGS cache; refresh it explicitly")
        if self.mode != "refresh" and manifest is not None:
            digest = file_checksum(path)
            if manifest.get("url") != url or manifest.get("sha256") != digest:
                raise RuntimeError("USGS cache checksum or request mismatch")
            if self.expected_sha256 and digest != self.expected_sha256:
                raise RuntimeError("USGS snapshot does not match expected SHA-256")
            return manifest
        if self.mode == "offline":
            raise FileNotFoundError(f"USGS snapshot missing from {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        pages, count = 0, 0
        seen: set[str] = set()
        page_urls: list[str] = []
        page_sha256: list[str] = []
        next_url: str | None = url
        with tempfile.TemporaryDirectory(dir=directory) as scratch:
            staged = Path(scratch) / "observations.jsonl"
            with staged.open("w", encoding="utf-8") as target:
                while next_url is not None:
                    parsed = urlsplit(next_url)
                    if (parsed.scheme, parsed.netloc, parsed.path) != (
                        "https",
                        "api.waterdata.usgs.gov",
                        "/ogcapi/v1/collections/daily/items",
                    ):
                        raise RuntimeError("unexpected USGS pagination URL")
                    if next_url in seen:
                        raise RuntimeError("repeated USGS pagination URL")
                    seen.add(next_url)
                    page_urls.append(next_url)
                    with urlopen(next_url, timeout=60) as response:
                        raw = response.read(16 * 1024 * 1024 + 1)
                    if len(raw) > 16 * 1024 * 1024:
                        raise ValueError("USGS response exceeds the page byte limit")
                    page_sha256.append(hashlib.sha256(raw).hexdigest())
                    page = json.loads(raw)
                    if page.get("type") != "FeatureCollection" or not isinstance(
                        page.get("features"), list
                    ):
                        raise ValueError("invalid USGS daily response")
                    features = page["features"]
                    if len(features) > 500:
                        raise ValueError("USGS page exceeds requested limit")
                    for feature in features:
                        _observation(feature, self.station, self.start, self.end)
                        target.write(
                            json.dumps(feature, separators=(",", ":"), sort_keys=True)
                            + "\n"
                        )
                        count += 1
                    pages += 1
                    links = [
                        link["href"]
                        for link in page.get("links", [])
                        if link.get("rel") == "next"
                    ]
                    if len(links) > 1:
                        raise ValueError("multiple USGS next links")
                    next_url = urljoin(next_url, links[0]) if links else None
            digest = file_checksum(staged)
            if self.expected_sha256 and digest != self.expected_sha256:
                raise RuntimeError("USGS snapshot does not match expected SHA-256")
            result: dict[str, Any] = {
                "url": url,
                "sha256": digest,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "pages": pages,
                "page_urls": page_urls,
                "page_sha256": page_sha256,
                "observations": count,
                "parameter_code": "00060",
                "statistic_id": "00003",
                "source": "USGS Water Data for the Nation",
                "checksum_scope": (
                    "canonicalized feature JSON lines, not raw HTTP bytes"
                ),
                "citation": "USGS Water Data for the Nation, doi:10.5066/F7P55KJN",
                "data_policy_url": "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
                "disclaimer_url": "https://water.usgs.gov/data/disclaimer.html",
            }
            staged.replace(path)
            write_manifest(directory, result)
        return result

    def batches(self, *, batch_size: int = 4096) -> Iterator[pa.RecordBatch]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.prefetch()
        path = self._directory() / "observations.jsonl"
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                rows.append(
                    _observation(json.loads(line), self.station, self.start, self.end)
                )
                if len(rows) == batch_size:
                    yield pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)
                    rows.clear()
        if rows:
            yield pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)


def _observation(
    feature: dict[str, Any], station: str, start: date | None, end: date | None
) -> dict[str, Any]:
    properties = feature["properties"]
    day = date.fromisoformat(properties["time"])
    if (
        feature.get("type") != "Feature"
        or properties.get("monitoring_location_id") != station
        or properties.get("parameter_code") != "00060"
        or properties.get("statistic_id") != "00003"
        or start is None
        or end is None
        or not start <= day <= end
    ):
        raise ValueError("USGS returned an observation outside the requested selection")
    raw_value = properties.get("value")
    value = None if raw_value is None or raw_value == "" else float(raw_value)
    if value is not None and not math.isfinite(value):
        raise ValueError("USGS observation value must be finite")
    geometry = feature.get("geometry")
    coordinates = (
        geometry.get("coordinates")
        if geometry and geometry.get("type") == "Point"
        else None
    )
    return {
        "station": station,
        "date": day,
        "flow": value,
        "unit": properties.get("unit_of_measure"),
        "approval_status": properties.get("approval_status"),
        "qualifier_json": json.dumps(properties.get("qualifier")),
        "time_series_id": properties.get("time_series_id"),
        "last_modified": properties.get("last_modified"),
        "longitude": coordinates[0] if coordinates else None,
        "latitude": coordinates[1] if coordinates else None,
        "observation_id": feature.get("id"),
    }
