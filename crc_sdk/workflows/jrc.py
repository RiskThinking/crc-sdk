"""Fluent, lazy JRC acquisition and canonicalization plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework.distributions import DistributionFamily

from crc_sdk.connectors import CurveFitIngestPolicy
from crc_sdk.connectors.jrc import canonicalize_jrc_flood
from crc_sdk.connectors.parquet import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.providers.jrc import JRCProvider, JRCRasterResource, jrc_dataset

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


@dataclass(frozen=True)
class _PreparedSources:
    version: str
    resources: tuple[JRCRasterResource, ...]
    cache_hits: int
    cache_misses: int


def _validate_bounds(bounds: Sequence[float]) -> Bounds:
    if len(bounds) != 4:
        raise ValueError("area bounds must contain min_lon, min_lat, max_lon, max_lat")
    normalized = tuple(float(value) for value in bounds)
    min_lon, min_lat, max_lon, max_lat = normalized
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("area bounds must be ordered WGS84 longitude/latitude values")
    return normalized  # type: ignore[return-value]


def _periods(
    values: Sequence[int] | Literal["all"], available: tuple[int, ...]
) -> tuple[int, ...]:
    if values == "all":
        return available
    normalized = tuple(values)
    if not normalized:
        raise ValueError("at least one source return period is required")
    invalid = sorted(set(normalized) - set(available))
    if invalid:
        raise ValueError(
            f"source return periods {invalid!r} are unavailable; "
            f"choose from {available}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("source return periods must be unique")
    return normalized


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _read_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(cache_dir)
    if not path.is_file():
        return None
    try:
        return cast(dict[str, Any], json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(f"invalid JRC cache manifest {path}: {error}") from error


def _write_manifest(cache_dir: Path, manifest: Mapping[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(cache_dir)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crop_raster(url: str, bounds: Bounds, destination: Path) -> None:
    try:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.crs import CRS  # type: ignore[import-untyped]
        from rasterio.warp import transform_bounds  # type: ignore[import-untyped]
        from rasterio.windows import Window, from_bounds  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "JRC caching requires `pip install crc-sdk[raster]`"
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.tif")
    with rasterio.open(url) as source:
        source_bounds = transform_bounds(CRS.from_epsg(4326), source.crs, *bounds)
        window = from_bounds(*source_bounds, transform=source.transform)
        window = (
            window.round_offsets()
            .round_lengths()
            .intersection(Window(0, 0, source.width, source.height))
        )
        if window.width < 1 or window.height < 1:
            raise ValueError(f"JRC raster does not intersect area bounds: {url}")
        data = source.read(window=window)
        profile = source.profile.copy()
        profile.update(
            width=int(window.width),
            height=int(window.height),
            transform=source.window_transform(window),
            compress="deflate",
            tiled=False,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        with rasterio.open(temporary, "w", **profile) as target:
            target.write(data)
    temporary.replace(destination)


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
            bounds=_validate_bounds(bounds),
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
            _read_manifest(self.area.cache_dir)
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
        manifest = _read_manifest(cache_dir) if cache_dir is not None else None
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
                        or _file_checksum(path) != expected
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
                    and _file_checksum(destination) == expected
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
                    _crop_raster(url, self.area.bounds, destination)
                local[period_text] = str(destination)
                checksums[period_text] = _file_checksum(destination)
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
        _write_manifest(cache_dir, manifest)
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
        policy = (
            self.policy.ingest_policy(prepared.version)
            if isinstance(self.policy, JRCFloodPolicy)
            else replace(self.policy, source_version=prepared.version)
        )
        tables = []
        metadata = None
        for resource in prepared.resources:
            if progress:
                progress("fit", {"source": resource.source_id})
            with provider.open_resource(resource) as raster:
                stream = canonicalize_jrc_flood(
                    raster,
                    policy,
                    bounds=self.area.bounds,
                )
                metadata = stream.metadata
                table = stream.read_all()
                if table.num_rows:
                    tables.append(table)
        if metadata is None:
            raise LookupError(f"no JRC resources intersect area {self.area.bounds}")
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

    def for_assets(self, assets: Any | AssetPortfolio) -> JRCPortfolioEvaluation:
        portfolio = (
            assets if isinstance(assets, AssetPortfolio) else AssetPortfolio(assets)
        )
        return JRCPortfolioEvaluation(plan=self, portfolio=portfolio)


@dataclass(frozen=True)
class JRCPortfolioEvaluation:
    plan: JRCCanonicalizationPlan
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
    ) -> JRCPortfolioEvaluation:
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

    def return_periods(self, values: Sequence[float]) -> JRCPortfolioEvaluation:
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
    ) -> JRCPortfolioEvaluation:
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
