"""Fluent, lazy JRC acquisition and canonicalization plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework.distributions import DistributionFamily

from crc_sdk.connectors import CurveFitIngestPolicy
from crc_sdk.connectors.duckdb.geotiff import GeoTiffRaster, RasterBoundsError
from crc_sdk.connectors.jrc import canonicalize_jrc_flood
from crc_sdk.connectors.parquet import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.providers.jrc import JRCProvider, JRCRasterResource, jrc_dataset

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
from .portfolio import (
    AssetPortfolio,
    HazardDataset,
)


@dataclass(frozen=True)
class JRCFloodPolicy:
    """Discoverable defaults for fitting JRC flood return-period rasters."""

    h3_resolution: int | Literal["native"] = "native"
    family: DistributionFamily = "gumbel_r"
    producer: str = "crc-sdk"
    creation_version: str = "0.2.0"
    on_fit_failure: Literal["raise", "skip"] = "skip"
    maximum_normalized_rmse: float | None = None
    maximum_absolute_residual: float | None = None

    @classmethod
    def curated(
        cls,
        *,
        h3_resolution: int | Literal["native"] = "native",
        on_fit_failure: Literal["raise", "skip"] = "skip",
    ) -> JRCFloodPolicy:
        return cls(
            h3_resolution=h3_resolution,
            on_fit_failure=on_fit_failure,
        )

    def ingest_policy(self, source_version: str) -> CurveFitIngestPolicy:
        resolution = 10 if self.h3_resolution == "native" else self.h3_resolution
        return CurveFitIngestPolicy(
            h3_resolution=resolution,
            family=self.family,
            producer=self.producer,
            creation_version=self.creation_version,
            value_semantics="riverine flood depth",
            source_version=source_version,
            maximum_normalized_rmse=self.maximum_normalized_rmse,
            maximum_absolute_residual=self.maximum_absolute_residual,
            on_fit_failure=self.on_fit_failure,
        )


@dataclass(frozen=True)
class _PreparedSources:
    version: str
    resources: tuple[JRCRasterResource, ...]
    cache_hits: int
    cache_misses: int


def _periods(
    values: Sequence[int] | Literal["all"], available: tuple[int, ...]
) -> tuple[int, ...]:
    normalized = available if values == "all" else tuple(values)
    if len(normalized) < 4:
        raise ValueError("at least four source return periods are required for fitting")
    invalid = sorted(set(normalized) - set(available))
    if invalid:
        raise ValueError(
            f"source return periods {invalid!r} are unavailable; "
            f"choose from {available}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("source return periods must be unique")
    return normalized


def _manifest_matches(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    requested_version: str,
    bounds: Bounds,
    periods: tuple[int, ...],
) -> bool:
    return (
        manifest.get("dataset") == dataset
        and manifest.get("requested_version") == requested_version
        and tuple(manifest.get("bounds", ())) == bounds
        and tuple(manifest.get("source_periods", ())) == periods
    )


def _cached_resources(manifest: Mapping[str, Any]) -> tuple[JRCRasterResource, ...]:
    return tuple(
        JRCRasterResource(
            source_id=entry["source_id"],
            urls={int(period): path for period, path in entry["local"].items()},
        )
        for entry in manifest["resources"]
    )


def _provenance_uri(
    dataset: str,
    bounds: Bounds,
    source_ids: Sequence[str],
    periods: Sequence[int],
) -> str:
    query = urlencode(
        {
            "bounds": ",".join(format(value, ".12g") for value in bounds),
            "sources": ",".join(source_ids),
            "source_periods": ",".join(str(period) for period in periods),
        }
    )
    return f"{dataset}/aoi?{query}"


def _crop_raster(url: str, bounds: Bounds, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.tif")
    try:
        with GeoTiffRaster.open(url) as source:
            source.write_crop(source.bounds_from_wgs84(bounds), temporary)
    except RasterBoundsError:
        temporary.unlink(missing_ok=True)
        return False
    temporary.replace(destination)
    return True


@dataclass(frozen=True)
class JRCSourcePlan:
    dataset: str
    requested_version: str = "latest"

    def version(self, value: str) -> JRCSourcePlan:
        if not value:
            raise ValueError("JRC source version must not be empty")
        return replace(self, requested_version=value)

    def for_area(self, bounds: Sequence[float]) -> JRCAreaPlan:
        return JRCAreaPlan(
            source=self,
            bounds=validate_bounds(bounds),
        )


@dataclass(frozen=True)
class JRCAreaPlan:
    source: JRCSourcePlan
    bounds: Bounds
    cache_dir: Path | None = None
    cache_mode: CacheMode = "stream"
    selected_periods: tuple[int, ...] | None = None

    def cache(
        self,
        directory: str | Path | None,
        *,
        mode: CacheMode = "reuse",
    ) -> JRCAreaPlan:
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

    def source_periods(
        self,
        values: Sequence[int] | Literal["all"],
    ) -> JRCAreaPlan:
        dataset = jrc_dataset(self.source.dataset)
        return replace(
            self,
            selected_periods=_periods(values, dataset.available_return_periods),
        )

    def canonicalize(
        self,
        *,
        policy: str | JRCFloodPolicy | CurveFitIngestPolicy = "curated",
    ) -> JRCCanonicalizationPlan:
        if policy == "curated":
            normalized: JRCFloodPolicy | CurveFitIngestPolicy = JRCFloodPolicy.curated()
        elif isinstance(policy, (JRCFloodPolicy, CurveFitIngestPolicy)):
            normalized = policy
        else:
            raise TypeError(
                "policy must be 'curated', JRCFloodPolicy, or CurveFitIngestPolicy"
            )
        return JRCCanonicalizationPlan(area=self, policy=normalized)


@dataclass(frozen=True)
class JRCCanonicalizationPlan:
    area: JRCAreaPlan
    policy: JRCFloodPolicy | CurveFitIngestPolicy

    def cache(
        self,
        directory: str | Path | None,
        *,
        mode: CacheMode = "reuse",
    ) -> JRCCanonicalizationPlan:
        return replace(self, area=self.area.cache(directory, mode=mode))

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        dataset = jrc_dataset(self.area.source.dataset)
        periods = self.area.selected_periods or dataset.available_return_periods
        manifest = (
            read_manifest(self.area.cache_dir)
            if self.area.cache_dir is not None
            else None
        )
        matching = manifest is not None and _manifest_matches(
            manifest,
            dataset=dataset.name,
            requested_version=self.area.source.requested_version,
            bounds=self.area.bounds,
            periods=periods,
        )
        details: dict[str, Any] = {
            "dataset": dataset.name,
            "layout": dataset.layout,
            "requested_version": self.area.source.requested_version,
            "resolved_version": (
                manifest.get("resolved_version")
                if matching and manifest is not None
                else None
            ),
            "area": self.area.bounds,
            "source_periods": periods,
            "cache": {
                "mode": self.area.cache_mode,
                "directory": str(self.area.cache_dir) if self.area.cache_dir else None,
                "manifest_reusable": matching,
            },
            "execution": "network and fitting occur only at prefetch/materialize/write",
        }
        if format == "json":
            return details
        if format != "text":
            raise ValueError("explain format must be 'text' or 'json'")
        resolved = details["resolved_version"] or "at execution"
        return (
            f"Dataset: {details['dataset']} ({details['layout']})\n"
            f"Requested version: {details['requested_version']}\n"
            f"Resolved version: {resolved}\n"
            f"Area: {','.join(str(value) for value in self.area.bounds)}\n"
            f"Source periods: {','.join(str(value) for value in periods)}\n"
            f"Cache: {self.area.cache_mode}"
            f"{f' ({self.area.cache_dir})' if self.area.cache_dir else ''}\n"
            "Network access and fitting occur only at prefetch/materialize/write."
        )

    def _prepare(self, progress: ProgressCallback | None = None) -> _PreparedSources:
        provider = JRCProvider.for_dataset(
            self.area.source.dataset,
            work_dir=self.area.cache_dir,
        )
        periods = (
            self.area.selected_periods or provider.dataset.available_return_periods
        )
        cache_dir = self.area.cache_dir
        manifest = read_manifest(cache_dir) if cache_dir is not None else None
        matches = manifest is not None and _manifest_matches(
            manifest,
            dataset=provider.dataset.name,
            requested_version=self.area.source.requested_version,
            bounds=self.area.bounds,
            periods=periods,
        )

        if self.area.cache_mode == "offline":
            if not matches:
                raise FileNotFoundError(
                    f"{provider.dataset.name} cannot be materialized offline: "
                    "no matching cache manifest; run plan.prefetch() while online"
                )
            assert manifest is not None
            resources = _cached_resources(manifest)
            invalid = []
            for entry in manifest["resources"]:
                for period, path_text in entry["local"].items():
                    path = Path(path_text)
                    expected = entry.get("checksums", {}).get(period)
                    if (
                        not path.is_file()
                        or expected is None
                        or file_checksum(path) != expected
                    ):
                        invalid.append(f"{entry['source_id']}:RP{period}")
            if invalid:
                raise FileNotFoundError(
                    f"{provider.dataset.name} cannot be materialized offline; "
                    f"missing or invalid cached AOI resources: {', '.join(invalid)}"
                )
            return _PreparedSources(
                version=manifest["resolved_version"],
                resources=resources,
                cache_hits=sum(len(resource.urls) for resource in resources),
                cache_misses=0,
            )

        if matches and self.area.cache_mode == "reuse":
            assert manifest is not None
            version = manifest["resolved_version"]
            remote_entries = manifest["resources"]
        else:
            if progress:
                progress("resolve", {"dataset": provider.dataset.name})
            version = provider.resolve_version(self.area.source.requested_version)
            remote_resources = provider.resources_for(
                self.area.bounds,
                return_periods=periods,
            )
            remote_entries = [
                {
                    "source_id": resource.source_id,
                    "remote": {
                        str(period): url for period, url in resource.urls.items()
                    },
                }
                for resource in remote_resources
            ]

        if self.area.cache_mode == "stream":
            resources = tuple(
                JRCRasterResource(
                    source_id=entry["source_id"],
                    urls={int(period): url for period, url in entry["remote"].items()},
                )
                for entry in remote_entries
            )
            return _PreparedSources(
                version, resources, 0, len(resources) * len(periods)
            )

        assert cache_dir is not None
        cache_dir.mkdir(parents=True, exist_ok=True)
        hits = 0
        misses = 0
        materialized_entries = []
        for entry in remote_entries:
            local: dict[str, str] = {}
            checksums: dict[str, str] = {}
            remote = entry["remote"]
            intersects = True
            for period_text, url in remote.items():
                identity = json.dumps(
                    [
                        provider.dataset.name,
                        version,
                        entry["source_id"],
                        period_text,
                        self.area.bounds,
                        url,
                    ],
                    separators=(",", ":"),
                )
                digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
                destination = cache_dir / "aoi" / f"rp{period_text}-{digest}.tif"
                refresh = self.area.cache_mode == "refresh"
                expected = entry.get("checksums", {}).get(period_text)
                valid = (
                    destination.is_file()
                    and expected is not None
                    and file_checksum(destination) == expected
                )
                if valid and not refresh:
                    hits += 1
                else:
                    misses += 1
                    if progress:
                        progress(
                            "fetch",
                            {
                                "source": entry["source_id"],
                                "return_period": int(period_text),
                            },
                        )
                    intersects = _crop_raster(url, self.area.bounds, destination)
                    if not intersects:
                        break
                local[period_text] = str(destination)
                checksums[period_text] = file_checksum(destination)
            if intersects:
                materialized_entries.append(
                    {**entry, "local": local, "checksums": checksums}
                )

        manifest = {
            "provider": "jrc",
            "dataset": provider.dataset.name,
            "requested_version": self.area.source.requested_version,
            "resolved_version": version,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "bounds": self.area.bounds,
            "source_periods": periods,
            "resources": materialized_entries,
        }
        write_manifest(cache_dir, manifest)
        return _PreparedSources(
            version,
            _cached_resources(manifest),
            hits,
            misses,
        )

    def prefetch(self, *, progress: ProgressCallback | None = None) -> PrefetchResult:
        if self.area.cache_mode == "stream":
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
        provider = JRCProvider.for_dataset(
            self.area.source.dataset,
            work_dir=self.area.cache_dir or destination.parent,
        )
        periods = (
            self.area.selected_periods or provider.dataset.available_return_periods
        )
        policy = (
            self.policy.ingest_policy(prepared.version)
            if isinstance(self.policy, JRCFloodPolicy)
            else replace(self.policy, source_version=prepared.version)
        )
        tables = []
        metadata = None
        contributing_sources = []
        for resource in prepared.resources:
            if progress:
                progress("fit", {"source": resource.source_id})
            with provider.open_resource(resource) as raster:
                stream = canonicalize_jrc_flood(
                    raster,
                    policy,
                    bounds=raster.bounds_from_wgs84(self.area.bounds),
                )
                metadata = stream.metadata
                table = stream.read_all()
                if table.num_rows:
                    tables.append(table)
                    contributing_sources.append(resource.source_id)
        if metadata is None:
            raise LookupError(f"no JRC resources intersect area {self.area.bounds}")
        source_ids = contributing_sources or [
            resource.source_id for resource in prepared.resources
        ]
        metadata = metadata.model_copy(
            update={
                "source": metadata.source.model_copy(
                    update={
                        "uri": _provenance_uri(
                            provider.dataset.name,
                            self.area.bounds,
                            source_ids,
                            periods,
                        )
                    }
                )
            }
        )
        table = (
            pa.concat_tables(tables, promote_options="none")
            if tables
            else pa.Table.from_batches([], schema=hazard_arrow_schema(metadata))
        )
        write_hazard_dataset(table, destination, metadata)
        result = MaterializationResult(
            output=destination,
            source_version=prepared.version,
            source_cache_hits=prepared.cache_hits,
            source_cache_misses=prepared.cache_misses,
            canonical_rows=table.num_rows,
        )
        return HazardDataset.local(destination, materialization=result)

    def _automatic_output(self) -> Path:
        if self.area.cache_dir is None:
            raise ValueError(
                "one-chain evaluation requires a persistent cache; call "
                ".cache(path, mode='reuse') or materialize(...) explicitly"
            )
        identity = json.dumps(
            [
                self.area.source.dataset,
                self.area.source.requested_version,
                self.area.bounds,
                self.area.selected_periods,
                repr(self.policy),
            ],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return self.area.cache_dir / "canonical" / f"hazard-{digest}.parquet"

    def ensure_materialized(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> HazardDataset:
        output = self._automatic_output()
        if output.is_file() and self.area.cache_mode != "refresh":
            return HazardDataset.local(output)
        return self.materialize(output, progress=progress)

    def for_assets(self, assets: Any | AssetPortfolio) -> RemotePortfolioEvaluation:
        portfolio = (
            assets if isinstance(assets, AssetPortfolio) else AssetPortfolio(assets)
        )
        return RemotePortfolioEvaluation(plan=self, portfolio=portfolio)


JRCPortfolioEvaluation = RemotePortfolioEvaluation
