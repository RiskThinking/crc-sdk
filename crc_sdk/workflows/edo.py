"""Fluent, lazy JRC/EDO drought acquisition and canonicalization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from crc_framework.distributions import DistributionFamily

from crc_sdk.connectors import CurveFitIngestPolicy
from crc_sdk.connectors.jrc_edo import canonicalize_edo_drought
from crc_sdk.connectors.parquet import write_hazard_stream
from crc_sdk.providers.jrc_edo import EDOProvider, edo_dataset

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
from .portfolio import AssetPortfolio, HazardDataset

YearSelection = tuple[int, ...] | Literal["all_complete"]


@dataclass(frozen=True)
class EDODroughtPolicy:
    """Safe defaults for annual-minimum EDO drought fitting."""

    h3_resolution: int = 6
    family: DistributionFamily = "gumbel_r"
    producer: str = "crc-sdk"
    creation_version: str = "0.2.0"
    minimum_years: int = 20
    on_fit_failure: Literal["raise", "skip"] = "skip"
    maximum_normalized_rmse: float | None = None
    maximum_absolute_residual: float | None = None

    def __post_init__(self) -> None:
        if self.minimum_years < 4:
            raise ValueError("minimum_years must be at least four")

    @classmethod
    def curated(
        cls,
        *,
        h3_resolution: int = 6,
        minimum_years: int = 20,
    ) -> EDODroughtPolicy:
        return cls(h3_resolution=h3_resolution, minimum_years=minimum_years)

    def ingest_policy(self, source_version: str) -> CurveFitIngestPolicy:
        return CurveFitIngestPolicy(
            h3_resolution=self.h3_resolution,
            family=self.family,
            producer=self.producer,
            creation_version=self.creation_version,
            tail="lower",
            value_semantics="soil moisture index annual minimum",
            source_version=source_version,
            maximum_normalized_rmse=self.maximum_normalized_rmse,
            maximum_absolute_residual=self.maximum_absolute_residual,
            on_fit_failure=self.on_fit_failure,
        )

    def validate_years(self, years: tuple[int, ...]) -> None:
        if len(years) < self.minimum_years:
            raise ValueError(
                f"EDO curated fitting requires at least {self.minimum_years} "
                f"complete years; received {len(years)}"
            )


@dataclass(frozen=True)
class _PreparedYears:
    version: str
    years: tuple[int, ...]
    resources: Mapping[int, str]
    cache_hits: int
    cache_misses: int


def _normalize_years(
    start: int | Sequence[int] | Literal["all_complete"],
    end: int | None,
) -> YearSelection:
    if start == "all_complete":
        if end is not None:
            raise ValueError("all_complete does not accept an end year")
        return "all_complete"
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


def _manifest_matches(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    requested_version: str,
    bounds: Bounds,
    years: YearSelection,
) -> bool:
    requested_years: Any = years if years == "all_complete" else list(years)
    return (
        manifest.get("dataset") == dataset
        and manifest.get("requested_version") == requested_version
        and tuple(manifest.get("bounds", ())) == bounds
        and manifest.get("requested_years") == requested_years
    )


@dataclass(frozen=True)
class EDOSourcePlan:
    dataset: str
    requested_version: str = "latest"

    def version(self, value: str) -> EDOSourcePlan:
        if not value:
            raise ValueError("EDO source version must not be empty")
        return replace(self, requested_version=value)

    def for_area(self, bounds: Sequence[float]) -> EDOAreaPlan:
        return EDOAreaPlan(source=self, bounds=validate_bounds(bounds))


@dataclass(frozen=True)
class EDOAreaPlan:
    source: EDOSourcePlan
    bounds: Bounds

    def years(
        self,
        start: int | Sequence[int] | Literal["all_complete"],
        end: int | None = None,
    ) -> EDOYearPlan:
        return EDOYearPlan(
            area=self,
            selected_years=_normalize_years(start, end),
        )


@dataclass(frozen=True)
class EDOYearPlan:
    area: EDOAreaPlan
    selected_years: YearSelection
    cache_dir: Path | None = None
    cache_mode: CacheMode = "stream"

    def cache(
        self,
        directory: str | Path | None,
        *,
        mode: CacheMode = "reuse",
    ) -> EDOYearPlan:
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
        policy: str | EDODroughtPolicy | CurveFitIngestPolicy = "curated",
    ) -> EDOCanonicalizationPlan:
        if policy == "curated":
            normalized: EDODroughtPolicy | CurveFitIngestPolicy = (
                EDODroughtPolicy.curated()
            )
        elif isinstance(policy, (EDODroughtPolicy, CurveFitIngestPolicy)):
            normalized = policy
        else:
            raise TypeError(
                "policy must be 'curated', EDODroughtPolicy, or CurveFitIngestPolicy"
            )
        return EDOCanonicalizationPlan(years=self, policy=normalized)


@dataclass(frozen=True)
class EDOCanonicalizationPlan:
    years: EDOYearPlan
    policy: EDODroughtPolicy | CurveFitIngestPolicy

    def cache(
        self,
        directory: str | Path | None,
        *,
        mode: CacheMode = "reuse",
    ) -> EDOCanonicalizationPlan:
        return replace(self, years=self.years.cache(directory, mode=mode))

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        dataset = edo_dataset(self.years.area.source.dataset)
        cache_dir = self.years.cache_dir
        manifest = read_manifest(cache_dir) if cache_dir is not None else None
        matching = manifest is not None and _manifest_matches(
            manifest,
            dataset=dataset.name,
            requested_version=self.years.area.source.requested_version,
            bounds=self.years.area.bounds,
            years=self.years.selected_years,
        )
        resolved_years = (
            tuple(manifest["resolved_years"])
            if matching and manifest is not None
            else None
        )
        details: dict[str, Any] = {
            "dataset": dataset.name,
            "requested_version": self.years.area.source.requested_version,
            "resolved_version": (
                manifest.get("resolved_version")
                if matching and manifest is not None
                else None
            ),
            "area": self.years.area.bounds,
            "requested_years": self.years.selected_years,
            "resolved_years": resolved_years,
            "cache": {
                "mode": self.years.cache_mode,
                "directory": str(cache_dir) if cache_dir else None,
                "manifest_reusable": matching,
                "object": "per-year AOI annual minimum",
            },
            "return_period_tail": "lower",
            "execution": "network and fitting occur only at prefetch/materialize/write",
        }
        if format == "json":
            return details
        if format != "text":
            raise ValueError("explain format must be 'text' or 'json'")
        years = (
            f"{resolved_years[0]}-{resolved_years[-1]}"
            if resolved_years
            else str(self.years.selected_years)
        )
        return (
            f"Dataset: {dataset.name}\n"
            f"Requested version: {self.years.area.source.requested_version}\n"
            f"Resolved version: {details['resolved_version'] or 'at execution'}\n"
            f"Area: {','.join(str(value) for value in self.years.area.bounds)}\n"
            f"Years: {years}\n"
            f"Cache: {self.years.cache_mode}"
            f"{f' ({cache_dir})' if cache_dir else ''}\n"
            "Tail: lower\n"
            "Network access and fitting occur only at prefetch/materialize/write."
        )

    def _prepare(self, progress: ProgressCallback | None = None) -> _PreparedYears:
        cache_dir = self.years.cache_dir
        source = self.years.area.source
        provider = EDOProvider(
            edo_dataset(source.dataset),
            work_dir=cache_dir,
        )
        manifest = read_manifest(cache_dir) if cache_dir is not None else None
        matches = manifest is not None and _manifest_matches(
            manifest,
            dataset=provider.dataset.name,
            requested_version=source.requested_version,
            bounds=self.years.area.bounds,
            years=self.years.selected_years,
        )

        if self.years.cache_mode == "offline":
            if not matches:
                raise FileNotFoundError(
                    f"{provider.dataset.name} cannot be materialized offline: "
                    "no matching cache manifest; run plan.prefetch() while online"
                )
            assert manifest is not None
            invalid = []
            resources = {}
            for entry in manifest["resources"]:
                path = Path(entry["local"])
                if not path.is_file() or file_checksum(path) != entry.get("checksum"):
                    invalid.append(str(entry["year"]))
                resources[int(entry["year"])] = str(path)
            if invalid:
                raise FileNotFoundError(
                    f"{provider.dataset.name} cannot be materialized offline; "
                    f"missing or invalid cached annual minima: {', '.join(invalid)}"
                )
            years = tuple(manifest["resolved_years"])
            self._validate_years(years)
            return _PreparedYears(
                manifest["resolved_version"],
                years,
                resources,
                len(resources),
                0,
            )

        if matches and self.years.cache_mode == "reuse":
            assert manifest is not None
            version = manifest["resolved_version"]
            selected = tuple(manifest["resolved_years"])
            entries = manifest["resources"]
        else:
            if progress:
                progress("resolve", {"dataset": provider.dataset.name})
            version = provider.resolve_version(source.requested_version)
            selected = (
                provider.complete_years()
                if self.years.selected_years == "all_complete"
                else self.years.selected_years
            )
            entries = [
                {"year": year, "remote": provider.year_url(year)} for year in selected
            ]
        self._validate_years(selected)

        if self.years.cache_mode == "stream":
            return _PreparedYears(
                version,
                selected,
                {int(entry["year"]): entry["remote"] for entry in entries},
                0,
                len(entries),
            )

        assert cache_dir is not None
        resources = {}
        materialized = []
        hits = 0
        misses = 0
        for entry in entries:
            year = int(entry["year"])
            identity = json.dumps(
                [provider.dataset.name, version, year, self.years.area.bounds],
                separators=(",", ":"),
            )
            digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
            destination = cache_dir / "annual-minima" / f"{year}-{digest}.nc"
            expected = entry.get("checksum")
            valid = (
                destination.is_file()
                and expected is not None
                and file_checksum(destination) == expected
            )
            if valid and self.years.cache_mode != "refresh":
                hits += 1
            else:
                misses += 1
                if progress:
                    progress("fetch", {"year": year})
                provider.cache_annual_minimum(
                    year,
                    self.years.area.bounds,
                    destination,
                )
            checksum = file_checksum(destination)
            resources[year] = str(destination)
            materialized.append(
                {
                    "year": year,
                    "remote": provider.year_url(year),
                    "local": str(destination),
                    "checksum": checksum,
                }
            )

        manifest = {
            "provider": "jrc-edo",
            "dataset": provider.dataset.name,
            "requested_version": source.requested_version,
            "resolved_version": version,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "bounds": self.years.area.bounds,
            "requested_years": (
                self.years.selected_years
                if self.years.selected_years == "all_complete"
                else list(self.years.selected_years)
            ),
            "resolved_years": selected,
            "resources": materialized,
        }
        write_manifest(cache_dir, manifest)
        return _PreparedYears(version, selected, resources, hits, misses)

    def _validate_years(self, years: tuple[int, ...]) -> None:
        if isinstance(self.policy, EDODroughtPolicy):
            self.policy.validate_years(years)
        elif len(years) < 4:
            raise ValueError("EDO fitting requires at least four complete years")

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

    def materialize(
        self,
        output: str | Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> HazardDataset:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        prepared = self._prepare(progress)
        provider = EDOProvider(
            edo_dataset(self.years.area.source.dataset),
            work_dir=self.years.cache_dir or destination.parent,
        )
        policy = (
            self.policy.ingest_policy(prepared.version)
            if isinstance(self.policy, EDODroughtPolicy)
            else replace(
                self.policy,
                tail="lower",
                source_version=prepared.version,
            )
        )
        if progress:
            progress("fit", {"years": prepared.years})
        with provider.open_resources(dict(prepared.resources)) as source:
            stream = canonicalize_edo_drought(
                source,
                policy,
                bounds=self.years.area.bounds,
            )
            write_hazard_stream(stream, destination)
        from crc_sdk.connectors.parquet import read_hazard_dataset

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
        identity = json.dumps(
            [
                self.years.area.source.dataset,
                self.years.area.source.requested_version,
                self.years.area.bounds,
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
