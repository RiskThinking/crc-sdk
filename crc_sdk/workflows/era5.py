"""Fluent, lazy ERA5 historical-baseline acquisition and canonicalization.

``HazardDataset.era5("txx").for_area(bounds).years(1991, 2020)`` plans an
annual-extreme baseline: nothing is read until ``prefetch()``,
``materialize()`` or a portfolio write. The cache holds one small file per
year and recipe (the area's annual extreme), never hourly data.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.connectors import CurveFitIngestPolicy
from crc_sdk.connectors.adapters import CurveSourceInfo, canonicalize_curve_source
from crc_sdk.connectors.duckdb.zarr import RasterMetadata
from crc_sdk.connectors.parquet import read_hazard_dataset, write_hazard_stream
from crc_sdk.providers.era5 import (
    ERA5_ATTRIBUTION,
    ERA5_LICENCE,
    ERA5Provider,
    ERA5Recipe,
    ERA5Store,
    era5_recipe,
    era5_store,
)
from crc_sdk.types import EnsembleDescriptor, TemporalWindow

from ._remote import (
    Bounds,
    CacheMode,
    MaterializationResult,
    PrefetchResult,
    ProgressCallback,
    RemotePortfolioEvaluation,
    file_checksum,
    read_manifest,
    validate_bounds,
    write_manifest,
)
from .blocks import BlockExtremaPolicy
from .portfolio import AssetPortfolio, HazardDataset

#: Years fetched concurrently; the stores are latency-bound, not CPU-bound.
FETCH_WORKERS = 4


def _normalize_years(
    start: int | Sequence[int],
    end: int | None,
) -> tuple[int, ...]:
    if isinstance(start, int):
        years = (start,) if end is None else tuple(range(start, end + 1))
    else:
        if end is not None:
            raise ValueError("a year sequence does not accept an end year")
        years = tuple(start)
    if not years or any(isinstance(year, bool) or year < 1900 for year in years):
        raise ValueError("years must contain valid calendar years")
    if len(set(years)) != len(years):
        raise ValueError("years must be unique")
    return tuple(sorted(years))


@dataclass(frozen=True)
class _PreparedYears:
    version: str
    years: tuple[int, ...]
    resources: Mapping[int, str]
    cache_hits: int
    cache_misses: int
    retrieved_at: str | None
    checksum: str | None


@dataclass(frozen=True)
class ERA5SourcePlan:
    recipe: str
    store: str = "arco-0p25"
    requested_version: str = "latest"

    def version(self, value: str) -> ERA5SourcePlan:
        if not value:
            raise ValueError("ERA5 source version must not be empty")
        return replace(self, requested_version=value)

    def with_store(self, name: str) -> ERA5SourcePlan:
        """Pick the Zarr copy: ``arco-0p25`` (native) or ``wb2-1p5`` (fast)."""
        era5_store(name)
        return replace(self, store=name.lower())

    def for_area(
        self, bounds: Sequence[float], *, land_only: bool = False
    ) -> ERA5AreaPlan:
        return ERA5AreaPlan(
            source=self, bounds=validate_bounds(bounds), land_only=land_only
        )


@dataclass(frozen=True)
class ERA5AreaPlan:
    source: ERA5SourcePlan
    bounds: Bounds
    land_only: bool = False

    def years(self, start: int | Sequence[int], end: int | None = None) -> ERA5YearPlan:
        return ERA5YearPlan(area=self, selected_years=_normalize_years(start, end))


@dataclass(frozen=True)
class ERA5YearPlan:
    area: ERA5AreaPlan
    selected_years: tuple[int, ...]
    cache_dir: Path | None = None
    cache_mode: CacheMode = "stream"

    def cache(
        self,
        directory: str | Path | None,
        *,
        mode: CacheMode = "reuse",
    ) -> ERA5YearPlan:
        if mode not in ("reuse", "offline", "refresh", "stream"):
            raise ValueError("cache mode must be reuse, offline, refresh, or stream")
        if mode == "stream":
            if directory is not None:
                raise ValueError("stream cache mode requires directory=None")
            path = None
        else:
            if directory is None:
                raise ValueError(f"{mode} cache mode requires a directory")
            path = Path(directory)
        return replace(self, cache_dir=path, cache_mode=mode)

    def canonicalize(
        self,
        *,
        policy: str | BlockExtremaPolicy | CurveFitIngestPolicy = "curated",
    ) -> ERA5CanonicalizationPlan:
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
        return ERA5CanonicalizationPlan(years=self, policy=normalized)


@dataclass(frozen=True)
class ERA5CanonicalizationPlan:
    years: ERA5YearPlan
    policy: BlockExtremaPolicy | CurveFitIngestPolicy

    def cache(
        self,
        directory: str | Path | None,
        *,
        mode: CacheMode = "reuse",
    ) -> ERA5CanonicalizationPlan:
        return replace(self, years=self.years.cache(directory, mode=mode))

    # -- identity -----------------------------------------------------------

    @property
    def _recipe(self) -> ERA5Recipe:
        return era5_recipe(self.years.area.source.recipe)

    @property
    def _store(self) -> ERA5Store:
        return era5_store(self.years.area.source.store)

    def _key(self, version: str, year: int) -> str:
        area = self.years.area
        identity = json.dumps(
            [
                self._store.name,
                version,
                self._recipe.name,
                repr(self._recipe.spec),
                year,
                area.bounds,
                area.land_only,
            ],
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode()).hexdigest()[:20]

    def _cached_entries(
        self, cache_dir: Path | None
    ) -> dict[int, Mapping[str, Any]] | None:
        """Valid cached entries per requested year, newest release first."""
        manifest = read_manifest(cache_dir) if cache_dir is not None else None
        if manifest is None:
            return None
        requested = self.years.area.source.requested_version
        entries = manifest.get("entries", {})
        found: dict[int, Mapping[str, Any]] = {}
        for year in self.years.selected_years:
            candidates = [
                entry
                for key, entry in entries.items()
                if entry.get("year") == year
                and key == self._key(entry.get("resolved_version", ""), year)
                and requested in ("latest", entry.get("resolved_version"))
            ]
            candidates.sort(key=lambda entry: entry["resolved_version"], reverse=True)
            for entry in candidates:
                # Entries record paths relative to the cache directory, so a
                # cache can be moved or shipped alongside a notebook.
                assert cache_dir is not None
                path = cache_dir / entry["local"]
                if path.is_file() and file_checksum(path) == entry.get("checksum"):
                    found[year] = entry
                    break
        return found

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        store, recipe = self._store, self._recipe
        cache_dir = self.years.cache_dir
        cached = self._cached_entries(cache_dir) or {}
        area = self.years.area
        details: dict[str, Any] = {
            "recipe": recipe.name,
            "definition": recipe.value_semantics,
            "unit": recipe.unit,
            "store": store.name,
            "store_url": store.url,
            "grid_degrees": store.resolution_degrees,
            "requested_version": area.source.requested_version,
            "area": area.bounds,
            "land_only": area.land_only,
            "years": self.years.selected_years,
            "return_period_tail": recipe.tail,
            "daily_boundary": "UTC",
            "cache": {
                "mode": self.years.cache_mode,
                "directory": str(cache_dir) if cache_dir else None,
                "years_cached": sorted(cached),
                "years_to_fetch": [
                    year for year in self.years.selected_years if year not in cached
                ],
                "object": "per-year area annual extreme",
            },
            "execution": "network and fitting occur only at prefetch/materialize/write",
        }
        if format == "json":
            return details
        if format != "text":
            raise ValueError("explain format must be 'text' or 'json'")
        years = self.years.selected_years
        return (
            f"Recipe: {recipe.name} ({recipe.value_semantics}, {recipe.unit})\n"
            f"Store: {store.name} ({store.description})\n"
            f"Area: {','.join(str(value) for value in area.bounds)}"
            f"{' (land cells only)' if area.land_only else ''}\n"
            f"Years: {years[0]}-{years[-1]} ({len(years)}); "
            f"{len(cached)} cached, {len(years) - len(cached)} to fetch\n"
            f"Cache: {self.years.cache_mode}"
            f"{f' ({cache_dir})' if cache_dir else ''}\n"
            f"Tail: {recipe.tail}; daily boundary: UTC\n"
            "Network access and fitting occur only at prefetch/materialize/write."
        )

    # -- preparation --------------------------------------------------------

    def _prepare(
        self,
        progress: ProgressCallback | None = None,
        *,
        cache_dir: Path | None = None,
    ) -> _PreparedYears:
        cache_dir = cache_dir or self.years.cache_dir
        mode = self.years.cache_mode
        area = self.years.area
        years = self.years.selected_years
        self._validate_years(years)

        cached = self._cached_entries(cache_dir) if mode != "refresh" else {}
        cached = cached or {}
        if mode == "offline":
            missing = [str(year) for year in years if year not in cached]
            if missing:
                raise FileNotFoundError(
                    f"ERA5 {self._recipe.name} cannot be materialized offline; "
                    f"missing or invalid cached years: {', '.join(missing)}; "
                    "run plan.prefetch() while online"
                )
        if len(cached) == len(years) and mode in ("reuse", "offline"):
            assert cache_dir is not None
            return self._from_entries(
                [cached[year] for year in years],
                cache_dir,
                hits=len(years),
                misses=0,
            )

        assert cache_dir is not None
        provider = ERA5Provider(self._store)
        if progress:
            progress("resolve", {"store": self._store.name})
        version = provider.resolve_version(area.source.requested_version)
        provider.check_years(years, self._recipe)
        _ = provider.times  # decode once, before the worker threads start
        # Entries cached under another release do not count for this one.
        cached = {
            year: entry
            for year, entry in cached.items()
            if entry.get("resolved_version") == version
        }
        todo = [year for year in years if year not in cached]
        retrieved = datetime.now(timezone.utc).isoformat()

        def fetch(year: int) -> dict[str, Any]:
            destination = (
                cache_dir / "annual-extremes" / f"{self._recipe.name}-{year}-"
                f"{self._key(version, year)}.nc"
            )
            if progress:
                progress("fetch", {"year": year})
            provider.cache_annual_extreme(
                self._recipe,
                year,
                area.bounds,
                destination,
                land_only=area.land_only,
                attributes={"resolved_version": version},
            )
            return {
                "year": year,
                "resolved_version": version,
                "retrieved_at": retrieved,
                "local": destination.relative_to(cache_dir).as_posix(),
                "checksum": file_checksum(destination),
            }

        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            fetched = list(pool.map(fetch, todo))

        entries = [*cached.values(), *fetched]
        manifest = read_manifest(cache_dir) or {}
        stored = dict(manifest.get("entries", {}))
        for entry in entries:
            stored[self._key(version, int(entry["year"]))] = {
                **entry,
                "store": self._store.name,
                "recipe": self._recipe.name,
                "bounds": area.bounds,
                "land_only": area.land_only,
            }
        write_manifest(
            cache_dir,
            {"provider": "era5", "entries": stored},
        )
        by_year = {int(entry["year"]): entry for entry in entries}
        return self._from_entries(
            [by_year[year] for year in years],
            cache_dir,
            hits=len(cached),
            misses=len(fetched),
        )

    def _from_entries(
        self,
        entries: Sequence[Mapping[str, Any]],
        cache_dir: Path,
        *,
        hits: int,
        misses: int,
    ) -> _PreparedYears:
        versions = {entry["resolved_version"] for entry in entries}
        if len(versions) != 1:
            raise RuntimeError(
                f"cached years span several releases: {sorted(versions)}"
            )
        digest = hashlib.sha256(
            "".join(str(entry["checksum"]) for entry in entries).encode()
        ).hexdigest()
        return _PreparedYears(
            version=versions.pop(),
            years=tuple(int(entry["year"]) for entry in entries),
            resources={
                int(entry["year"]): str(cache_dir / entry["local"]) for entry in entries
            },
            cache_hits=hits,
            cache_misses=misses,
            retrieved_at=max(str(entry["retrieved_at"]) for entry in entries),
            checksum=digest,
        )

    def _validate_years(self, years: tuple[int, ...]) -> None:
        if isinstance(self.policy, BlockExtremaPolicy):
            self.policy.validate_years(years)
        elif len(years) < 4:
            raise ValueError("ERA5 fitting requires at least four complete years")

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

    def annual_extremes(self, *, progress: ProgressCallback | None = None) -> Any:
        """The per-year, per-cell extremes behind the curves, as an Arrow table.

        Columns ``year``, ``latitude``, ``longitude`` and ``value`` (in the
        recipe's unit), with years a cell has no valid value for left out.
        The audit trail for any fitted curve: the sample the fit saw.
        """
        if self.years.cache_mode == "stream":
            raise ValueError("annual_extremes requires a cache directory")
        prepared = self._prepare(progress)
        source = self._source(prepared)
        with source:
            stack, lat, lon = source.block_array()
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

    # -- canonicalization ---------------------------------------------------

    def _source(self, prepared: _PreparedYears) -> Any:
        recipe, store = self._recipe, self._store
        first, last = prepared.years[0], prepared.years[-1]
        window = TemporalWindow(
            start_year=first,
            end_year=last,
            calendar="gregorian",
            minimum_complete_years=(
                self.policy.minimum_years
                if isinstance(self.policy, BlockExtremaPolicy)
                else None
            ),
        )
        info = CurveSourceInfo(
            uri=store.url,
            licence=ERA5_LICENCE,
            attribution=(
                f"{ERA5_ATTRIBUTION} {(prepared.retrieved_at or '')[:4]}"
                f" (ERA5 via {store.name}); {recipe.value_semantics}"
            ),
            retrieved_at=prepared.retrieved_at,
            checksum=prepared.checksum,
            temporal_window=window,
            ensemble=EnsembleDescriptor(
                pooling="single_member", models=("ERA5",), scenario="historical"
            ),
            probability_semantics="annual_exceedance",
        )
        metadata = RasterMetadata(
            hazard_type=recipe.name,
            indicator_id=recipe.name,
            scenario="historical",
            year=window.horizon,
            units=recipe.unit,
            path=f"{store.name}/{recipe.name}/{first}-{last}",
        )
        provider = ERA5Provider(store)
        return provider.open_resources(
            recipe,
            dict(prepared.resources),
            metadata,
            info=info,
        )

    def materialize(
        self,
        output: str | Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> HazardDataset:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="crc-era5-") as scratch:
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
                    provider="era5",
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

    def _automatic_output(self) -> Path:
        if self.years.cache_dir is None:
            raise ValueError(
                "one-chain evaluation requires a persistent cache; call "
                ".cache(path, mode='reuse') or materialize(...) explicitly"
            )
        area = self.years.area
        identity = json.dumps(
            [
                area.source.store,
                area.source.recipe,
                area.source.requested_version,
                area.bounds,
                area.land_only,
                self.years.selected_years,
                repr(self.policy),
            ],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return self.years.cache_dir / "canonical" / f"hazard-{digest}.parquet"

    def ensure_materialized(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> HazardDataset:
        output = self._automatic_output()
        if output.is_file() and self.years.cache_mode != "refresh":
            return HazardDataset.local(output)
        return self.materialize(output, progress=progress)

    def for_assets(self, assets: Any | AssetPortfolio) -> RemotePortfolioEvaluation:
        portfolio = (
            assets if isinstance(assets, AssetPortfolio) else AssetPortfolio(assets)
        )
        return RemotePortfolioEvaluation(plan=self, portfolio=portfolio)
