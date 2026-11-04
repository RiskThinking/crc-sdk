"""Steps shared by the year-selected canonicalization plans (ERA5, CMIP6, EDO)."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.connectors.adapters import CurveFitIngestPolicy, canonicalize_curve_source
from crc_sdk.connectors.parquet import read_hazard_dataset, write_hazard_stream

from ._remote import (
    CacheMode,
    MaterializationResult,
    PrefetchResult,
    ProgressCallback,
    RemotePortfolioEvaluation,
    is_complete_output,
    materialize_staged,
)
from .blocks import BlockExtremaPolicy
from .portfolio import AssetPortfolio, HazardDataset


def normalize_years(
    start: int | Sequence[int], end: int | None, *, earliest: int
) -> tuple[int, ...]:
    if isinstance(start, int):
        years = (start,) if end is None else tuple(range(start, end + 1))
    else:
        if end is not None:
            raise ValueError("a year sequence does not accept an end year")
        years = tuple(start)
    if not years or any(isinstance(year, bool) or year < earliest for year in years):
        raise ValueError("years must contain valid calendar years")
    if len(set(years)) != len(years):
        raise ValueError("years must be unique")
    return tuple(sorted(years))


def cache_location(directory: str | Path | None, mode: CacheMode) -> Path | None:
    """Validate a ``cache(directory, mode=...)`` call; the directory to use."""
    if mode not in ("reuse", "offline", "refresh", "stream"):
        raise ValueError("cache mode must be reuse, offline, refresh, or stream")
    if mode == "stream":
        if directory is not None:
            raise ValueError("stream cache mode requires directory=None")
        return None
    if directory is None:
        raise ValueError(f"{mode} cache mode requires a directory")
    return Path(directory)


def block_extrema_policy(
    policy: str | BlockExtremaPolicy | CurveFitIngestPolicy,
    recipe: Any,
    requested_version: str,
) -> BlockExtremaPolicy | CurveFitIngestPolicy:
    """Resolve a ``canonicalize(policy=...)`` argument against a recipe."""
    if policy == "curated":
        normalized: BlockExtremaPolicy | CurveFitIngestPolicy = (
            BlockExtremaPolicy.curated()
        )
    elif isinstance(policy, (BlockExtremaPolicy, CurveFitIngestPolicy)):
        normalized = policy
    else:
        raise TypeError(
            "policy must be 'curated', BlockExtremaPolicy, or CurveFitIngestPolicy"
        )
    if isinstance(normalized, BlockExtremaPolicy):
        normalized = normalized.resolve(
            family=recipe.default_family, fit_method=recipe.default_fit_method
        )
        # Validate resolved combinations before any annual data is fetched.
        normalized.ingest_policy(
            tail=recipe.tail,
            value_semantics=recipe.value_semantics,
            source_version=requested_version,
        )
    return normalized


def automatic_output(cache_dir: Path | None, identity: Sequence[Any]) -> Path:
    """Canonical output path for a plan identity inside its cache directory."""
    if cache_dir is None:
        raise ValueError(
            "one-chain evaluation requires a persistent cache; call "
            ".cache(path, mode='reuse') or materialize(...) explicitly"
        )
    encoded = json.dumps(list(identity), separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:20]
    return cache_dir / "canonical" / f"hazard-{digest}.parquet"


class YearCanonicalizationPlan:
    """Mixin for frozen plans holding ``years`` (with a cache) and ``policy``."""

    #: Source name used in error messages.
    _label: ClassVar[str]
    years: Any
    policy: Any

    if TYPE_CHECKING:

        def _prepare(self, progress: ProgressCallback | None = None) -> Any: ...

        def _automatic_output(self) -> Path: ...

        def materialize(
            self, output: str | Path, *, progress: ProgressCallback | None = None
        ) -> HazardDataset: ...

    def cache(self, directory: str | Path | None, *, mode: CacheMode = "reuse") -> Self:
        return cast(
            Self, replace(cast(Any, self), years=self.years.cache(directory, mode=mode))
        )

    def _validate_years(self, years: tuple[int, ...]) -> None:
        if not isinstance(self.policy, CurveFitIngestPolicy):
            self.policy.validate_years(years)
        elif len(years) < 4:
            raise ValueError(
                f"{self._label} fitting requires at least four complete years"
            )

    def prefetch(self, *, progress: ProgressCallback | None = None) -> PrefetchResult:
        if self.years.cache_mode == "stream":
            raise ValueError("prefetch requires reuse, refresh, or offline cache mode")
        prepared = self._prepare(progress)
        return PrefetchResult(
            source_version=prepared.version,
            cache_hits=prepared.cache_hits,
            cache_misses=prepared.cache_misses,
            resources=len(prepared.resources),
        )

    def _check_reuse(self, output: Path) -> None:
        """Hook run before an already-materialized output is handed back."""

    def ensure_materialized(
        self, *, progress: ProgressCallback | None = None
    ) -> HazardDataset:
        output = self._automatic_output()
        if is_complete_output(output) and self.years.cache_mode != "refresh":
            self._check_reuse(output)
            return HazardDataset.local(output)
        return materialize_staged(self.materialize, output, progress=progress)

    def for_assets(self, assets: Any | AssetPortfolio) -> RemotePortfolioEvaluation:
        portfolio = (
            assets if isinstance(assets, AssetPortfolio) else AssetPortfolio(assets)
        )
        return RemotePortfolioEvaluation(plan=self, portfolio=portfolio)


class BlockExtremaPlan(YearCanonicalizationPlan):
    """Mixin for plans fitting curves to cached annual block extremes."""

    #: Provider name recorded on canonical output.
    _provider: ClassVar[str]

    if TYPE_CHECKING:

        @property
        def _recipe(self) -> Any: ...

        def _prepare(
            self,
            progress: ProgressCallback | None = None,
            *,
            cache_dir: Path | None = None,
        ) -> Any: ...

        def _source(self, prepared: Any) -> Any: ...

    def annual_extremes(self, *, progress: ProgressCallback | None = None) -> Any:
        """The per-year, per-cell extremes behind the curves, as an Arrow table.

        Columns ``year``, ``latitude``, ``longitude`` and ``value`` (in the
        recipe's unit), with years a cell has no valid value for left out.
        The audit trail for any fitted curve: the sample the fit saw.
        """
        if self.years.cache_mode == "stream":
            raise ValueError("annual_extremes requires a cache directory")
        prepared = self._prepare(progress)
        with self._source(prepared) as source:
            stack, lat, lon = source.block_array(self.years.area.bounds)
            labels = source.labels
        year, row, column = np.nonzero(np.isfinite(stack))
        return pa.table(
            {
                "year": pa.array(np.asarray(labels)[year], type=pa.int32()),
                "latitude": pa.array(lat[row]),
                "longitude": pa.array(lon[column]),
                "value": pa.array(stack[year, row, column]),
            }
        )

    def materialize(
        self, output: str | Path, *, progress: ProgressCallback | None = None
    ) -> HazardDataset:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"crc-{self._provider}-") as scratch:
            streaming = self.years.cache_dir is None
            prepared = self._prepare(
                progress, cache_dir=Path(scratch) if streaming else None
            )
            recipe = self._recipe
            if isinstance(self.policy, BlockExtremaPolicy):
                policy = self.policy.ingest_policy(
                    tail=recipe.tail,
                    value_semantics=recipe.value_semantics,
                    source_version=prepared.version,
                )
            else:
                policy = replace(
                    self.policy,
                    tail=recipe.tail,
                    source_version=prepared.version,
                    value_semantics=self.policy.value_semantics
                    or recipe.value_semantics,
                )
            if progress:
                progress("fit", {"years": prepared.years})
            with self._source(prepared) as source:
                stream = canonicalize_curve_source(
                    source,
                    policy,
                    provider=self._provider,
                    bounds=self.years.area.bounds,
                )
                write_hazard_stream(stream, destination, overwrite=True)
        rows = read_hazard_dataset(destination, columns=["cell_index"]).num_rows
        result = MaterializationResult(
            output=destination,
            source_version=prepared.version,
            source_cache_hits=prepared.cache_hits,
            source_cache_misses=prepared.cache_misses,
            canonical_rows=rows,
        )
        return HazardDataset.local(destination, materialization=result)
