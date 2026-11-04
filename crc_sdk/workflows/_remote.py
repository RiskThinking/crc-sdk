"""Shared primitives for lazy remote-source plans."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from urllib.request import Request
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .distributions import return_periods_to_probabilities
from .portfolio import (
    AssetPortfolio,
    ExecutionOptions,
    HazardDataset,
    HazardSelection,
    ImpactContextColumns,
    PortfolioEvaluationResult,
    _impact_spec,
    _selection,
)

Bounds = tuple[float, float, float, float]
CacheMode = Literal["reuse", "offline", "refresh", "stream"]
ProgressCallback = Callable[[str, Mapping[str, Any]], None]


class MaterializablePlan(Protocol):
    def ensure_materialized(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> HazardDataset: ...


@dataclass(frozen=True)
class MaterializationResult:
    output: Path
    source_version: str
    source_cache_hits: int
    source_cache_misses: int
    canonical_rows: int


@dataclass(frozen=True)
class PrefetchResult:
    source_version: str
    cache_hits: int
    cache_misses: int
    resources: int


def validate_bounds(bounds: Sequence[float]) -> Bounds:
    if len(bounds) != 4:
        raise ValueError("area bounds must contain min_lon, min_lat, max_lon, max_lat")
    normalized = tuple(float(value) for value in bounds)
    min_lon, min_lat, max_lon, max_lat = normalized
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("area bounds must be ordered WGS84 longitude/latitude values")
    return cast(Bounds, normalized)


def http_request(url: str) -> Request:
    """Identify SDK requests to public data services."""
    return Request(url, headers={"User-Agent": "crc-sdk"})


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_expected_sha256(digest: str, expected: str | None, label: str) -> None:
    if expected and digest != expected:
        raise RuntimeError(f"{label} snapshot does not match expected SHA-256")


def cached_snapshot(
    path: Path,
    *,
    url: str,
    mode: str,
    expected: str | None,
    label: str,
    url_role: str = "request",
    noun: str = "snapshot",
) -> dict[str, Any] | None:
    """Verified manifest of the snapshot at ``path``, or None to fetch it.

    Raises when the cache is incomplete, no longer matches ``url`` or
    ``expected``, or is missing in offline mode.
    """
    directory = path.parent
    manifest = read_manifest(directory)
    if mode != "refresh" and path.exists() != (manifest is not None):
        raise RuntimeError(f"incomplete {label} cache; refresh it explicitly")
    if mode != "refresh" and manifest is not None:
        digest = file_checksum(path)
        if manifest.get("url") != url or manifest.get("sha256") != digest:
            raise RuntimeError(f"{label} cache checksum or {url_role} mismatch")
        check_expected_sha256(digest, expected, label)
        return manifest
    if mode == "offline":
        raise FileNotFoundError(f"{label} {noun} missing from {directory}")
    return None


def check_new_feature(feature: Mapping[str, Any], seen: set[Any], label: str) -> None:
    """Reject a feature repeated across pages (a shifted offset page).

    A repeat paired with a skipped feature still matches ``numberMatched``,
    so the count check alone cannot detect it.
    """
    identifier = feature.get("id")
    if identifier is None:
        return
    key = json.dumps(identifier, sort_keys=True)
    if key in seen:
        raise RuntimeError(
            f"{label} pagination repeated feature {identifier!r}; "
            "the collection changed while paging"
        )
    seen.add(key)


def read_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return cast(dict[str, Any], json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(f"invalid cache manifest {path}: {error}") from error


def write_manifest(cache_dir: Path, manifest: Mapping[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "manifest.json"
    # A unique staging name: concurrent writers never share a partial file.
    temporary = path.with_name(f".manifest.{uuid4().hex}.json.tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def plan_manifest_key(provider: str, identity: Sequence[Any]) -> str:
    """Key of one plan's record inside a cache directory's shared manifest."""
    digest = hashlib.sha256(
        json.dumps(list(identity), separators=(",", ":"), default=str).encode()
    ).hexdigest()[:20]
    return f"{provider}:{digest}"


def read_plan_manifest(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """One plan's record from a (possibly shared) cache manifest.

    Plans sharing a cache directory keep their records under ``plans``; a
    manifest written before that layout is returned whole, for the caller
    to match against its request as before.
    """
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return None
    plans = manifest.get("plans")
    if isinstance(plans, Mapping):
        record = plans.get(key)
        if isinstance(record, dict):
            return record
    return manifest


def write_plan_manifest(cache_dir: Path, key: str, record: Mapping[str, Any]) -> None:
    """Store one plan's record without dropping other plans' records.

    The latest record is mirrored at the top level, the single-plan layout
    earlier releases read and wrote.
    """
    manifest = read_manifest(cache_dir) or {}
    plans = manifest.get("plans")
    merged = dict(plans) if isinstance(plans, Mapping) else {}
    merged[key] = dict(record)
    write_manifest(cache_dir, {**manifest, **record, "plans": merged})


def merge_manifest_entries(
    cache_dir: Path,
    entries: Mapping[str, Any],
    **fields: Any,
) -> None:
    """Add keyed entries to the shared manifest, keeping everyone else's."""
    manifest = read_manifest(cache_dir) or {}
    stored = manifest.get("entries")
    merged = dict(stored) if isinstance(stored, Mapping) else {}
    merged.update(entries)
    write_manifest(cache_dir, {**manifest, **fields, "entries": merged})


def is_complete_output(path: Path) -> bool:
    """A Parquet file whose footer reads: never a truncated earlier write."""
    if not path.is_file():
        return False
    try:
        pq.read_metadata(path)
    except (OSError, ValueError, pa.ArrowException):
        return False
    return True


def materialize_staged(
    materialize: Callable[..., HazardDataset],
    output: Path,
    *,
    progress: ProgressCallback | None = None,
) -> HazardDataset:
    """Run ``materialize`` into a staged file, then move it into place.

    An interrupted fit never leaves a partial file at a cache path that a
    later ``ensure_materialized`` would hand back as complete.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f".{output.stem}.{uuid4().hex}.partial.parquet")
    try:
        dataset = materialize(staged, progress=progress)
        os.replace(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    result = dataset.materialization
    return HazardDataset.local(
        output,
        materialization=replace(result, output=output) if result else None,
    )


@dataclass(frozen=True)
class RemotePortfolioEvaluation:
    """Portfolio request that materializes a remote plan only at write time."""

    plan: MaterializablePlan
    portfolio: AssetPortfolio
    selection: HazardSelection = HazardSelection()
    periods: tuple[float, ...] | None = None
    impact_args: tuple[Any, ...] | None = None
    impact_kwargs: Mapping[str, Any] | None = None

    def select(
        self,
        *,
        hazard_names: Sequence[str] | None = None,
        horizons: Sequence[int] | None = None,
        pathways: Sequence[str] | None = None,
    ) -> RemotePortfolioEvaluation:
        return replace(
            self,
            selection=_selection(
                self.selection,
                hazard_names=hazard_names,
                horizons=horizons,
                pathways=pathways,
            ),
        )

    def return_periods(self, values: Sequence[float]) -> RemotePortfolioEvaluation:
        periods = tuple(float(value) for value in values)
        return_periods_to_probabilities(periods)
        return replace(self, periods=periods)

    def impact(
        self,
        function: Any,
        *,
        name: str,
        value_unit: str,
        value_semantics: str,
        context: ImpactContextColumns | None = None,
    ) -> RemotePortfolioEvaluation:
        # Validate now, like a local request, not after a remote download.
        _impact_spec(
            function,
            name=name,
            value_unit=value_unit,
            value_semantics=value_semantics,
            context=context,
        )
        return replace(
            self,
            impact_args=(function,),
            impact_kwargs={
                "name": name,
                "value_unit": value_unit,
                "value_semantics": value_semantics,
                "context": context,
            },
        )

    def write_parquet(
        self,
        output: str | Path,
        *,
        execution: ExecutionOptions | None = None,
        progress: ProgressCallback | None = None,
    ) -> PortfolioEvaluationResult:
        if self.periods is None:
            raise ValueError("return_periods(...) must be configured before writing")
        request = (
            self.plan.ensure_materialized(progress=progress)
            .for_assets(self.portfolio)
            .select(
                hazard_names=self.selection.hazard_names,
                horizons=self.selection.horizons,
                pathways=self.selection.pathways,
            )
            .return_periods(self.periods)
        )
        if self.impact_args is not None:
            request = request.impact(
                *self.impact_args, **dict(self.impact_kwargs or {})
            )
        return request.write_parquet(output, execution=execution)
