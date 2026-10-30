"""Bounded, checksummed snapshots for selected GeoMet feature collections."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from ._remote import file_checksum, read_manifest, write_manifest


def prepare_snapshot(
    *,
    directory: Path,
    url: str,
    mode: str,
    expected: str | None,
    label: str,
    opener: Callable[..., Any],
    validate: Callable[[dict[str, Any]], Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Acquire a selected page sequence, or verify its retained feature bytes."""
    path = directory / "observations.jsonl"
    manifest = read_manifest(directory)
    if mode != "refresh" and path.exists() != (manifest is not None):
        raise RuntimeError(f"incomplete {label} cache; refresh it explicitly")
    if mode != "refresh" and manifest is not None:
        digest = file_checksum(path)
        if manifest.get("url") != url or manifest.get("sha256") != digest:
            raise RuntimeError(f"{label} cache checksum or request mismatch")
        if expected and digest != expected:
            raise RuntimeError(f"{label} snapshot does not match expected SHA-256")
        return manifest
    if mode == "offline":
        raise FileNotFoundError(f"{label} snapshot missing from {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    expected_path = urlsplit(url).path
    seen: set[str] = set()
    page_urls: list[str] = []
    page_sha256: list[str] = []
    count = 0
    matched: int | None = None
    next_url: str | None = url
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        staged = Path(scratch) / "observations.jsonl"
        with staged.open("w", encoding="utf-8") as target:
            while next_url is not None:
                parsed = urlsplit(next_url)
                if (parsed.scheme, parsed.netloc, parsed.path) != (
                    "https",
                    "api.weather.gc.ca",
                    expected_path,
                ):
                    raise RuntimeError(f"unexpected {label} pagination URL")
                if next_url in seen:
                    raise RuntimeError(f"repeated {label} pagination URL")
                seen.add(next_url)
                page_urls.append(next_url)
                with opener(next_url, timeout=120) as source:
                    raw = source.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError(f"{label} response exceeds page byte limit")
                page_sha256.append(hashlib.sha256(raw).hexdigest())
                page = json.loads(raw)
                if page.get("type") != "FeatureCollection" or not isinstance(
                    page.get("features"), list
                ):
                    raise ValueError(f"invalid {label} response")
                features = page["features"]
                returned = page.get("numberReturned")
                if isinstance(returned, int) and returned != len(features):
                    raise ValueError(f"{label} page count does not match features")
                total = page.get("numberMatched")
                if isinstance(total, int) and not isinstance(total, bool):
                    if matched is not None and total != matched:
                        raise RuntimeError(f"{label} match count changed during paging")
                    matched = total
                if len(features) > 500:
                    raise ValueError(f"{label} page exceeds requested limit")
                for feature in features:
                    validate(feature)
                    target.write(
                        json.dumps(feature, separators=(",", ":"), sort_keys=True)
                        + "\n"
                    )
                    count += 1
                links = [
                    link["href"]
                    for link in page.get("links", [])
                    if link.get("rel") == "next"
                ]
                if len(links) > 1:
                    raise ValueError(f"multiple {label} next links")
                next_url = urljoin(next_url, links[0]) if links else None
        if matched is not None and count != matched:
            raise RuntimeError(f"{label} pagination returned an incomplete snapshot")
        digest = file_checksum(staged)
        if expected and digest != expected:
            raise RuntimeError(f"{label} snapshot does not match expected SHA-256")
        result: dict[str, Any] = {
            "url": url,
            **metadata,
            "sha256": digest,
            "checksum_scope": "canonicalized feature JSON lines",
            "page_urls": page_urls,
            "page_sha256": page_sha256,
            "observations": count,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "licence": "OGL-Canada-2.0",
            "service_licence_url": "https://eccc-msc.github.io/open-data/licence/readme_en/",
            "attribution": "Data Source: Environment and Climate Change Canada",
        }
        staged.replace(path)
        write_manifest(directory, result)
    return result
