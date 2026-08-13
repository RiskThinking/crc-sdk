"""Shared primitives for lazy remote-source plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .distributions import return_periods_to_probabilities
from .portfolio import (
    AssetPortfolio,
    ExecutionOptions,
    HazardDataset,
    HazardSelection,
    ImpactContextColumns,
    PortfolioEvaluationResult,
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


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
        current = self.selection
        return replace(
            self,
            selection=HazardSelection(
                hazard_names=tuple(hazard_names)
                if hazard_names is not None
                else current.hazard_names,
                horizons=tuple(horizons) if horizons is not None else current.horizons,
                pathways=tuple(pathways) if pathways is not None else current.pathways,
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
