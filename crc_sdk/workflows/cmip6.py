"""Fluent, lazy CMIP6 projection acquisition and canonicalization.

``HazardDataset.cmip6("rx1day", model=..., scenario="ssp585")
.for_area(bounds).years(2021, 2050)`` plans an annual-extreme projection:
nothing is read until ``prefetch()``, ``materialize()`` or a portfolio write.
The cache holds the resolved catalogue subset and one small file per year and
recipe (the area's annual extreme), never the daily fields.

The shape is deliberately `crc_sdk.workflows.era5`'s, so a historical baseline
and a scenario run are built the same way and their canonical outputs differ
only in ``pathway`` and ``horizon``. One step has no ERA5 counterpart: the
plan must first *find* its dataset in the CMIP6 catalogue, and the dataset it
resolved -- model, member, grid, DRS version, licence -- is written into the
cache manifest so a reuse or offline run never has to look again.
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
from crc_sdk.pathways import resolve_pathway
from crc_sdk.providers.cmip6 import (
    CFCalendar,
    CMIP6Archive,
    CMIP6Catalogue,
    CMIP6Dataset,
    CMIP6Provider,
    CMIP6Recipe,
    cmip6_archive,
    cmip6_recipe,
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
#: Where a cache directory keeps its copy of the catalogue's daily subset.
CATALOGUE_CACHE = "catalogue/pangeo-cmip6-day.parquet"


def _normalize_years(start: int | Sequence[int], end: int | None) -> tuple[int, ...]:
    if isinstance(start, int):
        years = (start,) if end is None else tuple(range(start, end + 1))
    else:
        if end is not None:
            raise ValueError("a year sequence does not accept an end year")
        years = tuple(start)
    if not years or any(isinstance(year, bool) or year < 1850 for year in years):
        raise ValueError("years must contain valid calendar years")
    if len(set(years)) != len(years):
        raise ValueError("years must be unique")
    return tuple(sorted(years))


@dataclass(frozen=True)
class _PreparedYears:
    """Everything a canonicalization needs, with no further network access."""

    dataset: CMIP6Dataset
    version: str
    calendar: str
    licence: str
    attribution: str
    years: tuple[int, ...]
    resources: Mapping[int, str]
    cache_hits: int
    cache_misses: int
    retrieved_at: str | None
    checksum: str | None


@dataclass(frozen=True)
class CMIP6SourcePlan:
    """One recipe from one model, pathway and member."""

    recipe: str
    model: str
    scenario: str
    member: str = "r1i1p1f1"
    grid_label: str | None = None
    archive: str = "aws"
    requested_version: str = "latest"
    #: Refuse a source whose licence goes beyond attribution, rather than
    #: warning and recording it (docs/licence-policy.md).
    strict_licence: bool = False

    def version(self, value: str) -> CMIP6SourcePlan:
        if not value:
            raise ValueError("CMIP6 source version must not be empty")
        return replace(self, requested_version=value)

    def with_grid(self, label: str) -> CMIP6SourcePlan:
        """Pin the grid: ``gn`` (model native), ``gr``/``gr1`` (regridded)."""
        if not label:
            raise ValueError("grid label must not be empty")
        return replace(self, grid_label=label.lower())

    def with_archive(self, name: str) -> CMIP6SourcePlan:
        cmip6_archive(name)
        return replace(self, archive=name.lower())

    def for_area(self, bounds: Sequence[float]) -> CMIP6AreaPlan:
        return CMIP6AreaPlan(source=self, bounds=validate_bounds(bounds))


@dataclass(frozen=True)
class CMIP6AreaPlan:
    source: CMIP6SourcePlan
    bounds: Bounds

    def years(
        self, start: int | Sequence[int], end: int | None = None
    ) -> CMIP6YearPlan:
        return CMIP6YearPlan(area=self, selected_years=_normalize_years(start, end))


@dataclass(frozen=True)
class CMIP6YearPlan:
    area: CMIP6AreaPlan
    selected_years: tuple[int, ...]
    cache_dir: Path | None = None
    cache_mode: CacheMode = "stream"

    def cache(
        self, directory: str | Path | None, *, mode: CacheMode = "reuse"
    ) -> CMIP6YearPlan:
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
    ) -> CMIP6CanonicalizationPlan:
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
            recipe = cmip6_recipe(self.area.source.recipe)
            normalized = normalized.resolve(
                family=recipe.default_family, fit_method=recipe.default_fit_method
            )
            # Validate resolved combinations before any annual data is fetched.
            normalized.ingest_policy(
                tail=recipe.tail,
                value_semantics=recipe.value_semantics,
                source_version=self.area.source.requested_version,
            )
        return CMIP6CanonicalizationPlan(years=self, policy=normalized)


@dataclass(frozen=True)
class CMIP6CanonicalizationPlan:
    years: CMIP6YearPlan
    policy: BlockExtremaPolicy | CurveFitIngestPolicy

    def cache(
        self, directory: str | Path | None, *, mode: CacheMode = "reuse"
    ) -> CMIP6CanonicalizationPlan:
        return replace(self, years=self.years.cache(directory, mode=mode))

    # -- identity -----------------------------------------------------------

    @property
    def _recipe(self) -> CMIP6Recipe:
        return cmip6_recipe(self.years.area.source.recipe)

    @property
    def _archive(self) -> CMIP6Archive:
        return cmip6_archive(self.years.area.source.archive)

    def _catalogue(self, cache_dir: Path | None) -> CMIP6Catalogue:
        """The catalogue, reading a cached daily subset when one exists."""
        local = cache_dir / CATALOGUE_CACHE if cache_dir is not None else None
        return CMIP6Catalogue(self._archive, local=local)

    def _key(self, dataset_id: str, year: int) -> str:
        area = self.years.area
        identity = json.dumps(
            [
                self._archive.name,
                dataset_id,
                self._recipe.name,
                repr(self._recipe.spec),
                year,
                area.bounds,
            ],
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode()).hexdigest()[:20]

    def _cached_entries(
        self, cache_dir: Path | None
    ) -> dict[int, Mapping[str, Any]] | None:
        """Valid cached entries per requested year, newest DRS version first."""
        manifest = read_manifest(cache_dir) if cache_dir is not None else None
        if manifest is None:
            return None
        source = self.years.area.source
        requested = source.requested_version
        entries = manifest.get("entries", {})
        found: dict[int, Mapping[str, Any]] = {}
        for year in self.years.selected_years:
            candidates = [
                entry
                for key, entry in entries.items()
                if entry.get("year") == year
                and key == self._key(entry.get("dataset_id", ""), year)
                and requested in ("latest", entry.get("resolved_version"))
                # A cached year belongs to this plan only if it came from the
                # same model, pathway and member, not merely the same recipe.
                and entry.get("model") == source.model
                and entry.get("scenario") == source.scenario
                and entry.get("member") == source.member
                and (
                    source.grid_label is None
                    or entry.get("grid_label") == source.grid_label
                )
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
        recipe = self._recipe
        archive = self._archive
        area = self.years.area
        source = area.source
        cache_dir = self.years.cache_dir
        cached = self._cached_entries(cache_dir) or {}
        resolved = next(iter(cached.values()), None)
        details: dict[str, Any] = {
            "recipe": recipe.name,
            "definition": recipe.value_semantics,
            "unit": recipe.unit,
            "model": source.model,
            "pathway": source.scenario,
            "member": source.member,
            "variable": recipe.variable,
            "grid_label": source.grid_label,
            "archive": archive.name,
            "catalogue": self._catalogue(cache_dir).source,
            "requested_version": source.requested_version,
            "resolved_dataset": resolved.get("dataset_id") if resolved else None,
            "calendar": resolved.get("calendar") if resolved else None,
            "licence": resolved.get("licence") if resolved else None,
            "strict_licence": source.strict_licence,
            "area": area.bounds,
            "years": self.years.selected_years,
            "return_period_tail": recipe.tail,
            "ensemble": "single_member (one model, one member, one pathway)",
            "fitting": {
                "family": self.policy.family,
                "method": self.policy.fit_method,
            },
            "cache": {
                "mode": self.years.cache_mode,
                "directory": str(cache_dir) if cache_dir else None,
                "years_cached": sorted(cached),
                "years_to_fetch": [
                    year for year in self.years.selected_years if year not in cached
                ],
                "object": "per-year area annual extreme",
            },
            "execution": (
                "catalogue lookup, network and fitting occur only at "
                "prefetch/materialize/write"
            ),
        }
        if format == "json":
            return details
        if format != "text":
            raise ValueError("explain format must be 'text' or 'json'")
        years = self.years.selected_years
        grid = f" ({source.grid_label})" if source.grid_label else ""
        resolved_id = details["resolved_dataset"]
        resolution = f" -> {resolved_id}" if resolved_id else " (not yet resolved)"
        return (
            f"Recipe: {recipe.name} ({recipe.value_semantics}, {recipe.unit})\n"
            f"Dataset: {source.model} {source.scenario} {source.member} "
            f"day {recipe.variable}{grid}{resolution}\n"
            f"Archive: {archive.name} ({archive.description})\n"
            f"Area: {','.join(str(value) for value in area.bounds)}\n"
            f"Years: {years[0]}-{years[-1]} ({len(years)}); "
            f"{len(cached)} cached, {len(years) - len(cached)} to fetch\n"
            f"Cache: {self.years.cache_mode}"
            f"{f' ({cache_dir})' if cache_dir else ''}\n"
            f"Tail: {recipe.tail}; ensemble: single_member\n"
            f"Licence: {details['licence'] or 'read when the dataset is opened'}\n"
            f"Fit: {self.policy.family}, {self.policy.fit_method}\n"
            "Catalogue lookup, network access and fitting occur only at "
            "prefetch/materialize/write."
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
        source = self.years.area.source
        years = self.years.selected_years
        self._validate_years(years)

        cached = self._cached_entries(cache_dir) if mode != "refresh" else {}
        cached = cached or {}
        if mode == "offline":
            missing = [str(year) for year in years if year not in cached]
            if missing:
                raise FileNotFoundError(
                    f"CMIP6 {self._recipe.name} cannot be materialized offline; "
                    f"missing or invalid cached years: {', '.join(missing)}; "
                    "run plan.prefetch() while online"
                )
        if len(cached) == len(years) and mode in ("reuse", "offline"):
            assert cache_dir is not None
            return self._from_entries(
                [cached[year] for year in years], cache_dir, hits=len(years), misses=0
            )

        assert cache_dir is not None
        catalogue = self._catalogue(cache_dir)
        if progress:
            progress("resolve", {"catalogue": catalogue.source})
        if catalogue.local is not None and not catalogue.local.is_file():
            catalogue.ensure_local(catalogue.local)
        dataset = catalogue.resolve(
            model=source.model,
            scenario=source.scenario,
            member=source.member,
            variable=self._recipe.variable,
            grid_label=source.grid_label,
            version=source.requested_version,
        )
        provider = CMIP6Provider(dataset, archive=self._archive)
        version = provider.resolve_version(source.requested_version)
        provider.check_variable(self._recipe)
        provider.check_years(years)
        calendar = provider.calendar.name
        licence = provider.check_licence(strict=source.strict_licence)
        licence_text, attribution = provider.licence_text, provider.attribution()
        _ = provider.ordinals  # decode once, before the worker threads start
        # Entries cached from another DRS version do not count for this one.
        cached = {
            year: entry
            for year, entry in cached.items()
            if entry.get("dataset_id") == dataset.dataset_id
        }
        todo = [year for year in years if year not in cached]
        retrieved = datetime.now(timezone.utc).isoformat()

        def fetch(year: int) -> dict[str, Any]:
            destination = (
                cache_dir / "annual-extremes" / f"{self._recipe.name}-{year}-"
                f"{self._key(dataset.dataset_id, year)}.nc"
            )
            if progress:
                progress("fetch", {"year": year})
            provider.cache_annual_extreme(
                self._recipe,
                year,
                self.years.area.bounds,
                destination,
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

        identity = {
            "dataset_id": dataset.dataset_id,
            "zstore": dataset.zstore,
            "model": dataset.source_id,
            "scenario": dataset.experiment_id,
            "member": dataset.member_id,
            "grid_label": dataset.grid_label,
            "institution": dataset.institution_id,
            "activity": dataset.activity_id,
            "variable": dataset.variable_id,
            "table_id": dataset.table_id,
            "calendar": calendar,
            "licence": licence,
            "licence_text": licence_text,
            "attribution": attribution,
        }
        entries = [*cached.values(), *fetched]
        manifest = read_manifest(cache_dir) or {}
        stored = dict(manifest.get("entries", {}))
        for entry in entries:
            stored[self._key(dataset.dataset_id, int(entry["year"]))] = {
                **identity,
                **entry,
                "recipe": self._recipe.name,
                "bounds": self.years.area.bounds,
            }
        write_manifest(cache_dir, {"provider": "cmip6", "entries": stored})
        by_year = {int(entry["year"]): {**identity, **entry} for entry in entries}
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
        identities = {entry["dataset_id"] for entry in entries}
        if len(identities) != 1:
            raise RuntimeError(
                f"cached years span several datasets: {sorted(identities)}"
            )
        first = entries[0]
        digest = hashlib.sha256(
            "".join(str(entry["checksum"]) for entry in entries).encode()
        ).hexdigest()
        return _PreparedYears(
            dataset=CMIP6Dataset(
                activity_id=str(first["activity"]),
                institution_id=str(first["institution"]),
                source_id=str(first["model"]),
                experiment_id=str(first["scenario"]),
                member_id=str(first["member"]),
                table_id=str(first["table_id"]),
                variable_id=str(first["variable"]),
                grid_label=str(first["grid_label"]),
                zstore=str(first["zstore"]),
                version=str(first["resolved_version"]),
            ),
            version=str(first["resolved_version"]),
            calendar=str(first["calendar"]),
            licence=str(first["licence"]),
            attribution=str(first["attribution"]),
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
            raise ValueError("CMIP6 fitting requires at least four complete years")

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

    # -- canonicalization ---------------------------------------------------

    def _source(self, prepared: _PreparedYears) -> Any:
        recipe = self._recipe
        dataset = prepared.dataset
        first, last = prepared.years[0], prepared.years[-1]
        # The source's own pathway spelling is kept; resolving it only checks
        # that the crosswalk knows it (ADR-0002).
        pathway = resolve_pathway(dataset.experiment_id).label
        window = TemporalWindow(
            start_year=first,
            end_year=last,
            # The manifest keeps the model's own calendar name, which is the
            # more informative one for audit; `TemporalWindow` has a narrower
            # vocabulary (no `all_leap`, and `proleptic_gregorian` is spelled
            # `gregorian`), so the window records the mapped spelling.
            calendar=CFCalendar.parse(prepared.calendar).window_calendar,
            minimum_complete_years=(
                self.policy.minimum_years
                if isinstance(self.policy, BlockExtremaPolicy)
                else None
            ),
        )
        info = CurveSourceInfo(
            uri=dataset.zstore,
            licence=prepared.licence,
            attribution=(
                f"{prepared.attribution}; {dataset.dataset_id}; "
                f"{recipe.value_semantics}"
            ),
            retrieved_at=prepared.retrieved_at,
            checksum=prepared.checksum,
            temporal_window=window,
            ensemble=EnsembleDescriptor(
                pooling="single_member",
                models=(dataset.source_id,),
                members=(dataset.member_id,),
                scenario=pathway,
            ),
            probability_semantics="annual_exceedance",
        )
        metadata = RasterMetadata(
            hazard_type=recipe.name,
            indicator_id=recipe.name,
            scenario=pathway,
            year=window.horizon,
            units=recipe.unit,
            path=f"{dataset.dataset_id}/{recipe.name}/{first}-{last}",
        )
        provider = CMIP6Provider(dataset, archive=self._archive)
        return provider.open_resources(
            recipe, dict(prepared.resources), metadata, info=info
        )

    def materialize(
        self, output: str | Path, *, progress: ProgressCallback | None = None
    ) -> HazardDataset:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="crc-cmip6-") as scratch:
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
                    source, policy, provider="cmip6", bounds=self.years.area.bounds
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
        source = self.years.area.source
        identity = json.dumps(
            [
                source.archive,
                source.recipe,
                source.model,
                source.scenario,
                source.member,
                source.grid_label,
                source.requested_version,
                source.strict_licence,
                self.years.area.bounds,
                self.years.selected_years,
                repr(self.policy),
            ],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return self.years.cache_dir / "canonical" / f"hazard-{digest}.parquet"

    def ensure_materialized(
        self, *, progress: ProgressCallback | None = None
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
