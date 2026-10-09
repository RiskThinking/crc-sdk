"""Lazy ingestion of already canonical CRC curves; no fitting or resampling."""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]

from crc_sdk.connectors.parquet import read_hazard_dataset, write_hazard_dataset
from crc_sdk.geometry.h3 import cell_polygon, intersecting_cells
from crc_sdk.providers.crc_open import (
    DOCS_FIXTURES,
    FIXTURE_RELEASE,
    CRCCatalog,
    CRCOpenFixtureWarning,
    CRCOpenHazards,
    CRCPartition,
)
from crc_sdk.types import EnsembleDescriptor, HazardDatasetMetadata

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
from .distributions import HorizonExtrapolationWarning
from .portfolio import (
    AssetPortfolio,
    ExecutionOptions,
    HazardDataset,
    PortfolioEvaluationResult,
)


def _box(bounds: Bounds) -> Any:
    try:
        from shapely.geometry import box  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "CRC area selection requires `pip install crc-sdk[geometry]`"
        ) from error
    return box(*bounds)


@dataclass(frozen=True)
class CRCOpenPlan:
    """An immutable catalogue request; construction and explain do no I/O.

    Each materialized HazardDataset has one hazard's unit, tail and resolution.
    Use materialize_all for multiple hazards with different scientific metadata.
    """

    release: str = FIXTURE_RELEASE
    source: str = DOCS_FIXTURES
    fixture_fallback: bool = True
    pathway: str = "ssp585"
    bounds: Bounds | None = None
    selected_hazards: tuple[str, ...] = ()
    selected_horizons: tuple[int, ...] | None = None
    cache_dir: Path | None = None
    cache_mode: CacheMode = "stream"

    def __post_init__(self) -> None:
        CRCOpenHazards(self.source, self.release)
        if not self.pathway:
            raise ValueError("pathway must not be empty")

    def for_area(self, bounds: Sequence[float]) -> CRCOpenPlan:
        return replace(self, bounds=validate_bounds(bounds))

    def hazards(self, values: Sequence[str]) -> CRCOpenPlan:
        if not values or len(set(values)) != len(values):
            raise ValueError("hazards must be a nonempty sequence of unique names")
        return replace(self, selected_hazards=tuple(values))

    def horizons(self, values: Sequence[int]) -> CRCOpenPlan:
        if not values or any(
            isinstance(v, bool) or not isinstance(v, int) for v in values
        ):
            raise ValueError("horizons must be a nonempty sequence of integer years")
        return replace(self, selected_horizons=tuple(values))

    def cache(
        self, directory: str | Path | None, *, mode: CacheMode = "reuse"
    ) -> CRCOpenPlan:
        if mode not in {"reuse", "offline", "refresh", "stream"}:
            raise ValueError("cache mode must be reuse, offline, refresh, or stream")
        if (mode == "stream") != (directory is None):
            raise ValueError(
                "stream requires directory=None; other modes require a directory"
            )
        return replace(
            self,
            cache_dir=Path(directory) if directory is not None else None,
            cache_mode=mode,
        )

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        details = dict(
            provider="crc_open",
            release=self.release,
            source=self.source,
            fixture_fallback=self.fixture_fallback,
            pathway=self.pathway,
            area=self.bounds,
            hazards=self.selected_hazards,
            horizons=self.selected_horizons,
            cache_mode=self.cache_mode,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            execution="catalogue-driven r0 pruning; no fitting; I/O only at execution",
        )
        if format == "json":
            return details
        if format != "text":
            raise ValueError("explain format must be 'text' or 'json'")
        return json.dumps(details, indent=2)

    def _selection(self, catalog: CRCCatalog) -> list[tuple[str, CRCPartition]]:
        if self.pathway not in catalog.pathways:
            raise ValueError(
                f"pathway {self.pathway!r} is not in the open subset; "
                f"available: {catalog.pathways}"
            )
        names = self.selected_hazards or tuple(catalog.hazards)
        unknown = set(names) - set(catalog.hazards)
        if unknown:
            reasons = "; ".join(
                f"{name}: {catalog.excluded_hazards[name]}"
                for name in sorted(unknown)
                if name in catalog.excluded_hazards
            )
            if reasons:
                raise ValueError(f"hazards are not in the open subset: {reasons}")
            raise ValueError(f"hazards {sorted(unknown)} are not in the open subset")
        parents = (
            set(intersecting_cells(_box(self.bounds), 0))
            if self.bounds is not None
            else None
        )
        entries = []
        for name in names:
            hazard = catalog.hazards[name]
            if self.pathway not in hazard.pathways:
                raise ValueError(
                    f"pathway {self.pathway!r} is not in the open subset for {name}"
                )
            if self.selected_horizons is not None:
                unavailable = set(self.selected_horizons) - set(hazard.horizons)
                if unavailable:
                    warnings.warn(
                        f"horizons {sorted(unavailable)} are outside the published "
                        f"horizons for {name}: {hazard.horizons}; no interpolation",
                        HorizonExtrapolationWarning,
                        stacklevel=3,
                    )
                    raise ValueError(
                        f"requested horizons are not in the open subset for {name}"
                    )
            for entry in hazard.partitions:
                if parents is not None and int(entry.h3_r0, 16) not in parents:
                    continue
                # Skip sparse partitions whose sampled cells miss the AOI.
                if self.bounds is not None and entry.cells is not None:
                    area = _box(self.bounds)
                    if not any(
                        cell_polygon(cell).intersects(area) for cell in entry.cells
                    ):
                        continue
                entries.append((name, entry))
        if not entries:
            raise LookupError(
                "area has no coverage in the open subset; "
                "fixtures have sampled coverage"
            )
        missing = set(names) - {name for name, _ in entries}
        if missing:
            raise LookupError(f"area has no open-subset coverage for {sorted(missing)}")
        return entries

    @contextmanager
    def _prepare(
        self, progress: ProgressCallback | None = None
    ) -> Iterator[tuple[CRCCatalog, list[tuple[str, Path]], int, int, str]]:
        provider = CRCOpenHazards(self.source, self.release)
        temporary = (
            tempfile.TemporaryDirectory(prefix="crc-open-")
            if self.cache_dir is None
            else None
        )
        root = Path(temporary.name) if temporary is not None else self.cache_dir
        assert root is not None
        try:
            identity = hashlib.sha256(
                json.dumps([self.source, self.release]).encode()
            ).hexdigest()[:20]
            directory = root / identity
            manifest = read_manifest(directory)
            catalog_path = directory / "_CATALOG.json"
            cached = (
                manifest is not None
                and manifest.get("source") == self.source
                and manifest.get("release") == self.release
                and catalog_path.is_file()
                and file_checksum(catalog_path) == manifest.get("catalog_sha256")
            )
            if self.cache_mode == "offline" and not cached:
                raise FileNotFoundError(
                    "no valid pinned CRC catalogue in cache; prefetch online first"
                )
            if cached and self.cache_mode in {"reuse", "offline"}:
                payload = catalog_path.read_bytes()
            else:
                if progress:
                    progress("resolve", {"release": self.release})
                try:
                    payload = provider.catalog_bytes()
                except OSError as error:
                    raise FileNotFoundError(
                        f"CRC release {self.release!r} is unavailable "
                        f"at {self.source!r}. "
                        "Check the release ID and catalogue root. source= and "
                        "fixtures= accept HTTP(S) roots or directories "
                        "containing releases."
                    ) from error
                digest = hashlib.sha256(payload).hexdigest()
                if manifest is not None and manifest.get("catalog_sha256") != digest:
                    raise ValueError(
                        "pinned CRC release changed; use a new release ID "
                        "or cache directory"
                    )
                directory.mkdir(parents=True, exist_ok=True)
                catalog_path.write_bytes(payload)
            catalog = provider.parse_catalog(payload)
            if self.fixture_fallback or catalog.fixture:
                warnings.warn(
                    "CRC fixtures provide sampled SSP585 coverage with pooled "
                    "ensembles and no historical baseline. Check the catalogue for "
                    "available cells and hazard-specific interpretation notes. "
                    "Use source= to select another catalogue root.",
                    CRCOpenFixtureWarning,
                    stacklevel=3,
                )
            selected = self._selection(catalog)
            for name in sorted({name for name, _ in selected}):
                note = catalog.hazards[name].notes
                if note:
                    warnings.warn(
                        f"{name}: {note}", CRCOpenFixtureWarning, stacklevel=3
                    )
            paths = []
            hits = misses = 0
            for name, entry in selected:
                destination = directory / entry.path
                valid = (
                    destination.is_file()
                    and destination.stat().st_size == entry.size_bytes
                    and file_checksum(destination) == entry.sha256
                )
                if valid and self.cache_mode != "refresh":
                    hits += 1
                else:
                    if self.cache_mode == "offline":
                        raise FileNotFoundError(
                            f"missing or corrupt CRC cached partition: {entry.path}"
                        )
                    if progress:
                        progress("fetch", {"hazard": name, "partition": entry.path})
                    data = provider.partition_bytes(entry)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    pending = destination.with_suffix(".tmp")
                    pending.write_bytes(data)
                    pending.replace(destination)
                    misses += 1
                paths.append((name, destination))
            write_manifest(
                directory,
                dict(
                    provider="crc_open",
                    source=self.source,
                    release=self.release,
                    catalog_sha256=hashlib.sha256(payload).hexdigest(),
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    request=dict(
                        bounds=self.bounds,
                        hazards=self.selected_hazards,
                        pathway=self.pathway,
                        horizons=self.selected_horizons,
                    ),
                    licence=catalog.licence,
                    attribution=catalog.attribution,
                    resources=[entry.model_dump() for _, entry in selected],
                ),
            )
            yield catalog, paths, hits, misses, hashlib.sha256(payload).hexdigest()
        finally:
            if temporary is not None:
                temporary.cleanup()

    def prefetch(self, *, progress: ProgressCallback | None = None) -> PrefetchResult:
        if self.cache_mode == "stream":
            raise ValueError("prefetch requires a persistent cache")
        with self._prepare(progress) as (catalog, paths, hits, misses, checksum):
            return PrefetchResult(catalog.release, hits, misses, len(paths))

    def _write(
        self,
        output: Path,
        name: str,
        catalog: CRCCatalog,
        paths: list[tuple[str, Path]],
        hits: int,
        misses: int,
        checksum: str,
    ) -> HazardDataset:
        if any(output.resolve() == path.resolve() for _, path in paths):
            raise ValueError("output must not overwrite a cached source partition")
        tables = []
        metadata: HazardDatasetMetadata | None = None
        hazard = catalog.hazards[name]
        for source_name, path in paths:
            if source_name != name:
                continue
            table = read_hazard_dataset(path)
            entry = next(
                p for p in hazard.partitions if path.as_posix().endswith("/" + p.path)
            )
            if table.num_rows != entry.rows:
                raise ValueError("partition row count disagrees with catalogue")
            current = HazardDatasetMetadata.from_parquet_metadata(table.schema.metadata)
            if (
                current.h3_resolution != hazard.h3_resolution
                or current.value_unit != hazard.unit
                or current.return_period_tail != hazard.tail
                or current.value_semantics != hazard.value_semantics
                or current.schema_version != catalog.schema_version
                or (
                    current.schema_version == "1.3"
                    and (
                        current.probability_semantics != hazard.probability_semantics
                        or current.ensemble is None
                        or current.ensemble.pooling != hazard.pooling
                    )
                )
            ):
                raise ValueError(
                    f"partition metadata disagrees with catalogue for {name}"
                )
            if current.schema_version == "1.2":
                # Older partitions cannot encode these catalogue declarations.
                current = current.model_copy(
                    update={
                        "schema_version": "1.3",
                        "probability_semantics": hazard.probability_semantics,
                        "ensemble": EnsembleDescriptor(pooling=hazard.pooling),
                    }
                )
            if set(table["hazard_name"].to_pylist()) != {name}:
                raise ValueError(f"partition contains unexpected hazards for {name}")
            if not set(table["pathway"].to_pylist()).issubset(hazard.pathways):
                raise ValueError(
                    "partition contains a pathway outside its catalogue scope"
                )
            if not set(table["horizon"].to_pylist()).issubset(hazard.horizons):
                raise ValueError(
                    "partition contains a horizon outside its catalogue scope"
                )
            if metadata is not None:
                # Source objects may differ, scientific dataset metadata may not.
                left = metadata.model_dump(exclude={"source"})
                right = current.model_dump(exclude={"source"})
                if left != right:
                    raise ValueError(
                        f"incompatible scientific metadata across {name} partitions"
                    )
            cells_in_file = set(table["cell_index"].to_pylist())
            if entry.cells is not None and cells_in_file != set(entry.cells):
                raise ValueError("partition cells disagree with catalogue")
            for cell in cells_in_file:
                parent = (cell & ~(15 << 52)) | ((1 << 45) - 1)
                if (
                    parent != int(entry.h3_r0, 16)
                    or (cell >> 52) & 15 != hazard.h3_resolution
                ):
                    raise ValueError("partition H3 addresses disagree with catalogue")
            metadata = current
            mask = pc.equal(table["pathway"], self.pathway)
            if self.selected_horizons is not None:
                mask = pc.and_(
                    mask,
                    pc.is_in(
                        table["horizon"], value_set=pa.array(self.selected_horizons)
                    ),
                )
            if self.bounds is not None:
                area = _box(self.bounds)
                cells = [
                    cell
                    for cell in set(table["cell_index"].to_pylist())
                    if cell_polygon(cell).intersects(area)
                ]
                mask = pc.and_(
                    mask,
                    pc.is_in(
                        table["cell_index"], value_set=pa.array(cells, type=pa.uint64())
                    ),
                )
            tables.append(table.filter(mask))
        assert metadata is not None
        table = pa.concat_tables(tables)
        if not table.num_rows:
            raise LookupError(
                f"no {name} rows match the requested open-subset area/scenario/horizons"
            )
        request = json.dumps(
            dict(
                bounds=self.bounds,
                hazard=name,
                pathway=self.pathway,
                horizons=self.selected_horizons,
            ),
            sort_keys=True,
        )
        metadata = metadata.model_copy(
            update={
                "source": metadata.source.model_copy(
                    update={
                        "provider": "crc_open",
                        "version": catalog.release,
                        "uri": (
                            f"{self.source}/{self.release}/_CATALOG.json"
                            f"#request={request}"
                        ),
                        "licence": catalog.licence,
                        "attribution": catalog.attribution,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "checksum": checksum,
                    }
                )
            }
        )
        write_hazard_dataset(table, output, metadata)
        return HazardDataset.local(
            output,
            materialization=MaterializationResult(
                output, catalog.release, hits, misses, table.num_rows
            ),
        )

    def materialize(
        self, output: str | Path, *, progress: ProgressCallback | None = None
    ) -> HazardDataset:
        if len(self.selected_hazards) != 1:
            raise ValueError(
                "materialize requires hazards([name]); "
                "use materialize_all for multiple hazards"
            )
        with self._prepare(progress) as (catalog, paths, hits, misses, checksum):
            return self._write(
                Path(output),
                self.selected_hazards[0],
                catalog,
                paths,
                hits,
                misses,
                checksum,
            )

    def materialize_all(
        self, directory: str | Path, *, progress: ProgressCallback | None = None
    ) -> dict[str, HazardDataset]:
        with self._prepare(progress) as (catalog, paths, hits, misses, checksum):
            return {
                name: self._write(
                    Path(directory) / f"{name}.parquet",
                    name,
                    catalog,
                    paths,
                    hits,
                    misses,
                    checksum,
                )
                for name in (self.selected_hazards or tuple(catalog.hazards))
            }

    def ensure_materialized(
        self, *, progress: ProgressCallback | None = None
    ) -> HazardDataset:
        if self.cache_dir is None:
            raise ValueError(
                "ensure_materialized requires a persistent cache; "
                "use cache(path) or materialize(output) explicitly"
            )
        identity = json.dumps(
            [
                self.source,
                self.release,
                self.pathway,
                self.bounds,
                self.selected_hazards,
                self.selected_horizons,
            ],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        output = self.cache_dir / "canonical" / f"hazard-{digest}.parquet"
        receipt = output.with_suffix(".sha256")
        if len(self.selected_hazards) != 1:
            raise ValueError("ensure_materialized requires hazards([name])")
        with self._prepare(progress) as (catalog, paths, hits, misses, checksum):
            if (
                self.cache_mode != "refresh"
                and output.is_file()
                and receipt.is_file()
                and file_checksum(output) == receipt.read_text().strip()
            ):
                dataset = HazardDataset.local(output)
                if dataset.provenance().checksum == checksum:
                    return dataset
            dataset = self._write(
                output, self.selected_hazards[0], catalog, paths, hits, misses, checksum
            )
            receipt.write_text(file_checksum(output) + "\n")
            return dataset

    def for_assets(self, assets: Any | AssetPortfolio) -> RemotePortfolioEvaluation:
        portfolio = (
            assets if isinstance(assets, AssetPortfolio) else AssetPortfolio(assets)
        )
        return CRCOpenPortfolioEvaluation(plan=self, portfolio=portfolio)


@dataclass(frozen=True)
class CRCOpenPortfolioEvaluation(RemotePortfolioEvaluation):
    """Apply portfolio selections to acquisition before fetching partitions."""

    plan: CRCOpenPlan

    def write_parquet(
        self,
        output: str | Path,
        *,
        execution: ExecutionOptions | None = None,
        progress: ProgressCallback | None = None,
    ) -> PortfolioEvaluationResult:
        plan = self.plan
        selection = self.selection
        if selection.pathways is not None and selection.pathways != (plan.pathway,):
            raise ValueError(
                "portfolio pathway must match the CRC open plan pathway; "
                "other pathways are not in this request's open subset"
            )
        if selection.hazard_names is not None:
            if plan.selected_hazards and not set(selection.hazard_names).issubset(
                plan.selected_hazards
            ):
                raise ValueError("portfolio hazards are outside the CRC open request")
            plan = plan.hazards(selection.hazard_names)
        if selection.horizons is not None:
            if plan.selected_horizons is not None and not set(
                selection.horizons
            ).issubset(plan.selected_horizons):
                raise ValueError("portfolio horizons are outside the CRC open request")
            plan = plan.horizons(selection.horizons)
        if plan.cache_dir is None:
            # Keep the canonical file alive through evaluation, then remove it
            # and the downloaded partitions even if evaluation raises.
            with tempfile.TemporaryDirectory(prefix="crc-open-portfolio-") as scratch:
                return RemotePortfolioEvaluation.write_parquet(
                    replace(self, plan=plan.cache(scratch)),
                    output,
                    execution=execution,
                    progress=progress,
                )
        return RemotePortfolioEvaluation.write_parquet(
            replace(self, plan=plan), output, execution=execution, progress=progress
        )
