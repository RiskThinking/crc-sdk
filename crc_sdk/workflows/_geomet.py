"""Bounded, checksummed snapshots for selected GeoMet feature collections."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast
from urllib.parse import urljoin, urlsplit

import pyarrow as pa  # type: ignore[import-untyped]

from ._remote import (
    cached_snapshot,
    check_expected_sha256,
    check_new_feature,
    file_checksum,
    http_request,
    write_manifest,
)


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
    cached = cached_snapshot(path, url=url, mode=mode, expected=expected, label=label)
    if cached is not None:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    expected_path = urlsplit(url).path
    seen: set[str] = set()
    feature_ids: set[Any] = set()
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
                with opener(http_request(next_url), timeout=120) as source:
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
                    check_new_feature(feature, feature_ids, label)
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
        check_expected_sha256(digest, expected, label)
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


class GeoMetDailyPlan:
    """Mixin for frozen per-station GeoMet plans cached one year at a time."""

    _schema: ClassVar[Any]
    selected_years: tuple[int, ...]
    cache_dir: Path | None
    mode: str
    checksums: tuple[tuple[int, str], ...]

    if TYPE_CHECKING:

        def _directory(self, year: int) -> Path: ...

        def _prepare_year(self, year: int) -> dict[str, Any]: ...

        def _parse(self, feature: dict[str, Any], year: int) -> dict[str, Any]: ...

    def _check_options(self) -> None:
        if self.mode not in {"reuse", "offline", "refresh"}:
            raise ValueError("mode must be reuse, offline or refresh")
        if any(not re.fullmatch(r"[0-9a-f]{64}", sha) for _, sha in self.checksums):
            raise ValueError("checksums must be lowercase SHA-256 digests")

    def years(self, start: int, end: int) -> Self:
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
        return cast(
            Self, replace(cast(Any, self), selected_years=tuple(range(start, end + 1)))
        )

    def cache(
        self,
        directory: str | Path,
        *,
        mode: str = "reuse",
        sha256: Mapping[int, str] | None = None,
    ) -> Self:
        return cast(
            Self,
            replace(
                cast(Any, self),
                cache_dir=Path(directory),
                mode=mode,
                checksums=tuple(sorted((sha256 or {}).items())),
            ),
        )

    def _checked_years(self) -> tuple[int, ...]:
        if not self.selected_years:
            raise ValueError("years(start, end) must be configured")
        if set(dict(self.checksums)) - set(self.selected_years):
            raise ValueError("checksum years must belong to the selected years")
        return self.selected_years

    def prefetch(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._prepare_year(year) for year in self._checked_years())

    def batches(self, *, batch_size: int = 4096) -> Iterator[pa.RecordBatch]:
        """Stream typed daily observations without interpreting quality flags."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for year in self._checked_years():
            self._prepare_year(year)
            path = self._directory(year) / "observations.jsonl"
            rows: list[dict[str, Any]] = []
            with path.open(encoding="utf-8") as source:
                for line in source:
                    rows.append(self._parse(json.loads(line), year))
                    if len(rows) == batch_size:
                        yield pa.RecordBatch.from_pylist(rows, schema=self._schema)
                        rows.clear()
                if rows:
                    yield pa.RecordBatch.from_pylist(rows, schema=self._schema)
