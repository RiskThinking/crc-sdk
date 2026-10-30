"""Checksummed raw-response caching for NOAA CO-OPS derived products."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._remote import file_checksum, read_manifest, write_manifest


def prepare_resource(
    *,
    directory: Path,
    url: str,
    mode: str,
    expected: str | None,
    opener: Callable[..., Any],
    validate: Callable[[dict[str, Any]], Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch, validate and retain one bounded, checksummed NOAA response."""
    path = directory / "source.json"
    manifest = read_manifest(directory)
    if mode != "refresh" and path.exists() != (manifest is not None):
        raise RuntimeError("incomplete NOAA cache; refresh explicitly")
    if mode != "refresh" and manifest is not None:
        digest = file_checksum(path)
        if manifest.get("url") != url or manifest.get("sha256") != digest:
            raise RuntimeError("NOAA cache checksum or request mismatch")
        if expected and digest != expected:
            raise RuntimeError("NOAA snapshot does not match expected SHA-256")
        return manifest
    if mode == "offline":
        raise FileNotFoundError(f"NOAA snapshot missing from {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        staged = Path(scratch) / "source.json"
        with opener(url, timeout=120) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            etag = response.headers.get("ETag")
            modified = response.headers.get("Last-Modified")
            content_length = response.headers.get("Content-Length")
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("NOAA response exceeds the byte limit")
        if content_length is not None and len(raw) != int(content_length):
            raise RuntimeError("incomplete NOAA download")
        payload = json.loads(raw)
        validate(payload)
        staged.write_bytes(raw)
        digest = file_checksum(staged)
        if expected and digest != expected:
            raise RuntimeError("NOAA snapshot does not match expected SHA-256")
        result: dict[str, Any] = {
            "url": url,
            **metadata,
            "sha256": digest,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "etag": etag,
            "last_modified": modified,
            "attribution": (
                "NOAA National Ocean Service, Center for Operational "
                "Oceanographic Products and Services"
            ),
            "source_documentation": "https://api.tidesandcurrents.noaa.gov/dpapi/prod/",
            "terms_url": "https://oceanservice.noaa.gov/about/faq.html",
        }
        staged.replace(path)
        write_manifest(directory, result)
    return result
