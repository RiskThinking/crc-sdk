"""Current CRC Open release indexes and exact canonical dataset inputs."""

from __future__ import annotations

import json
import os
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from h3.api import basic_int as h3  # type: ignore[import-untyped]

from ._dataset_schema import (
    DatasetMetadata,
    ExecutionReport,
    IdentityKind,
    IdentityRename,
    ReleaseReference,
    ResourceIdentity,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
    Transformation,
    _combine_reports,
    _digest,
    _identifier,
    _object,
)
from ._declarations import (
    DataSpec,
    Period,
    Scenario,
    SeriesSpec,
    Variable,
    _merge_descriptor,
    scientific_identity,
)
from ._graph import ExecutionScope, NodeStream, SourceContentChanged
from ._inspection import (
    Bounds,
    Capability,
    Inspection,
    Operation,
    SeriesReference,
)
from ._recipes import Dataset, _checked_stream, _key, _sources
from ._runtime import CacheMode, LocalPath
from ._selection import (
    ResolvedSelection,
    SelectionState,
    bounds_intersect,
    cell_bounds,
    participating_spec,
    selected_metadata,
    support_extent,
)
from ._source_io import SourceNode, _json, check_release, fetch, native_data
from ._terminals import _schema

CRC_OPEN_ROOT = "https://data.riskthinking.ai/crc-open"


def _relative(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or "?" in value
        or "#" in value
    ):
        raise ValueError("release resources must use relative paths within the release")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise ValueError("release resource escapes its release directory")
    return value


def _location(value: LocalPath | None, requested: ReleaseReference | None) -> str:
    if value is None:
        return CRC_OPEN_ROOT + (
            "/_CURRENT.json"
            if requested is None
            else "/" + urllib.parse.quote(requested.id, safe="") + "/_INDEX.json"
        )
    text = os.fspath(value)
    if urllib.parse.urlsplit(text).scheme in {"http", "https", "file"}:
        return text
    return Path(text).absolute().as_uri()


@dataclass(frozen=True, slots=True, eq=False)
class ReleaseDataset:
    """One indexed file. Its metadata is parsed when a selection first needs it."""

    id: str
    uri: str
    sha256: str
    extent: Bounds | None
    # Exact variable IDs read from the raw declaration, to skip unselected
    # entries without parsing them; None when the declaration is malformed.
    variable_ids: frozenset[str] | None
    _document: object = field(repr=False)
    _parsed: list[DatasetMetadata] = field(default_factory=list, repr=False)

    @property
    def metadata(self) -> DatasetMetadata:
        if not self._parsed:
            # A malformed entry is source data, as when the index was parsed whole.
            with native_data(source="crc-open"):
                self._parsed.append(
                    DatasetMetadata.from_json(
                        json.dumps(self._document, allow_nan=False)
                    )
                )
        return self._parsed[0]

    def declares(self, variables: Iterable[object]) -> bool:
        ids = self.variable_ids
        if ids is None:
            ids = frozenset(v.id for v in self.metadata.spec.variables)
        return not ids.isdisjoint(variables)

    @classmethod
    def parse(cls, value: object) -> ReleaseDataset:
        obj = _object(value, {"id", "uri", "sha256", "metadata", "extent"})
        return cls(
            _identifier(obj["id"], "release dataset id"),
            _relative(obj["uri"]),
            _digest(obj["sha256"]),
            _extent(obj["extent"]),
            _variable_ids(obj["metadata"]),
            obj["metadata"],
        )


def _variable_ids(document: object) -> frozenset[str] | None:
    try:
        ids = [
            variable["id"]
            for variable in document["spec"]["variables"]  # type: ignore[index]
        ]
    except (TypeError, KeyError):
        return None
    return frozenset(ids) if all(isinstance(i, str) for i in ids) else None


def _extent(value: object) -> Bounds | None:
    if value is None:
        return None
    obj = _object(value, {"west", "south", "east", "north"})
    try:
        return Bounds(**cast(dict[str, float], obj))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid release dataset extent: {error}") from None


def release_extent(path: LocalPath) -> Bounds | None:
    """The conservative extent to record for a Dataset file in a release index."""
    with pq.ParquetFile(path) as reader:
        return support_extent(
            reader.iter_batches(columns=["cell_id", "geometry"], batch_size=65_536)
        )


def release_index(value: object) -> tuple[str, tuple[ReleaseDataset, ...]]:
    obj = _object(value, {"schema_version", "id", "datasets"})
    if obj["schema_version"] != "1.1":
        raise ValueError(
            "CRC Open requires a current release index and dataset format 2.0"
        )
    if not isinstance(obj["datasets"], list):
        raise ValueError("release datasets must be an array")
    entries = tuple(ReleaseDataset.parse(item) for item in obj["datasets"])
    if len({item.id for item in entries}) != len(entries) or len(
        {item.uri for item in entries}
    ) != len(entries):
        raise ValueError("release dataset IDs and resource paths must be unique")
    return _identifier(obj["id"], "release id"), entries


# Parsed release indexes by content checksum. An index is immutable once its
# checksum is known, so repeated inspections and reads reuse the parse.
_PARSED_INDEXES: dict[str, tuple[str, tuple[ReleaseDataset, ...]]] = {}
_PARSED_INDEX_LIMIT = 4


def _parsed_index(
    document: object, sha256: str
) -> tuple[str, tuple[ReleaseDataset, ...]]:
    parsed = _PARSED_INDEXES.get(sha256)
    if parsed is None:
        parsed = release_index(document)
        while len(_PARSED_INDEXES) >= _PARSED_INDEX_LIMIT:
            _PARSED_INDEXES.pop(next(iter(_PARSED_INDEXES)), None)
        _PARSED_INDEXES[sha256] = parsed
    return parsed


def _series_key(item: SeriesSpec) -> tuple[str, str | None, str | None]:
    return (
        item.variable.id,
        None if item.scenario is None else item.scenario.id,
        None if item.period is None else item.period.id,
    )


def _selected(
    entries: tuple[ReleaseDataset, ...], state: SelectionState, source: SourceIdentity
) -> tuple[tuple[ReleaseDataset, DatasetMetadata, ResolvedSelection], ...]:
    # Only entries declaring a requested variable can participate; skipping the
    # rest before merging keeps a one-variable read from merging the whole index.
    for wanted in (
        None
        if state.variables is None
        else {getattr(item, "id", item) for item in state.variables},
        None if state.series is None else {item.variable.id for item in state.series},
    ):
        if wanted is not None:
            entries = tuple(entry for entry in entries if entry.declares(wanted))
    resolved = state.resolve(
        _merged_spec(tuple(entry.metadata for entry in entries)), source=source
    )
    # Requested identities resolve against every entry declaring them; only
    # then are files whose extent misses the area left out.
    if state.area is not None:
        area = state.area
        entries = tuple(
            entry
            for entry in entries
            if entry.extent is None or bounds_intersect(entry.extent, area)
        )
    result = []
    for entry in entries:
        spec = entry.metadata.spec
        series = tuple(
            item for item in spec.series if _series_key(item) in resolved.keys
        )
        if series:
            local = replace(
                resolved,
                spec=participating_spec(spec, series),
                keys=frozenset(_series_key(item) for item in series),
            )
            result.append((entry, selected_metadata(entry.metadata, local), local))
    return tuple(result)


def _merged_spec(parts: tuple[DatasetMetadata, ...]) -> DataSpec:
    variables: dict[str, Variable] = {}
    scenarios: dict[str, Scenario] = {}
    periods: dict[str, Period] = {}
    series: dict[tuple[str, str | None, str | None], SeriesSpec] = {}
    for metadata in parts:
        for value in metadata.spec.variables:
            _merge_descriptor(value, variables)
        for scenario in metadata.spec.scenarios:
            _merge_descriptor(scenario, scenarios)
        for period in metadata.spec.periods:
            _merge_descriptor(period, periods)
        for item in metadata.spec.series:
            key = _series_key(item)
            if (
                key in series
                and series[key] != item
                and scientific_identity(series[key]) != scientific_identity(item)
            ):
                raise ValueError(
                    "participating release series have conflicting "
                    "scientific definitions"
                )
            series[key] = item
    return DataSpec(
        tuple(variables.values()),
        series=tuple(series.values()),
        scenarios=tuple(scenarios.values()),
        periods=tuple(periods.values()),
    )


def _project_provenance(
    metadata: DatasetMetadata,
    variables: tuple[Variable, ...],
    scenarios: tuple[Scenario, ...],
) -> tuple[Transformation, ...]:
    spec = DataSpec(variables, scenarios=scenarios, series=())
    return selected_metadata(
        metadata, ResolvedSelection(spec=spec, keys=frozenset())
    ).provenance


def _merged_provenance(
    parts: tuple[DatasetMetadata, ...],
    spec: DataSpec,
) -> tuple[Transformation, ...]:
    if not any(part.provenance for part in parts):
        return ()
    if len(parts) == 1:
        return parts[0].provenance

    history: dict[tuple[IdentityKind, str], tuple[Transformation, ...]] = {}
    additions: list[tuple[Transformation, ...]] = []
    for part in parts:
        variables = []
        scenarios = []
        for variable in part.spec.variables:
            key = IdentityKind.VARIABLE, variable.id
            steps = _project_provenance(part, (variable,), ())
            if key not in history:
                variables.append(variable)
            elif history[key] != steps:
                raise ValueError(
                    f"Conflicting release provenance for variable {variable.id!r}"
                )
            history[key] = steps
        for scenario in part.spec.scenarios:
            key = IdentityKind.SCENARIO, scenario.id
            steps = tuple(
                step
                for step in _project_provenance(part, (), (scenario,))
                if isinstance(step, IdentityRename)
            )
            if key not in history:
                scenarios.append(scenario)
            elif history[key] != steps:
                raise ValueError(
                    f"Conflicting release provenance for scenario {scenario.id!r}"
                )
            history[key] = steps
        if variables or scenarios:
            additions.append(
                _project_provenance(part, tuple(variables), tuple(scenarios))
            )
    return _join_provenance(additions, spec)


def _rewind_rename(ids: set[str], mapping: Mapping[str, str]) -> set[str] | None:
    targets = set(mapping.values())
    previous = (ids - targets) | mapping.keys()
    return previous if targets <= ids and len(previous) == len(ids) else None


def _join_provenance(
    additions: list[tuple[Transformation, ...]], spec: DataSpec
) -> tuple[Transformation, ...]:
    # Keep each input's step order. When independent shards project different
    # sides of an atomic rename, rewind the sides together to avoid collisions.
    pending = [list(steps) for steps in reversed(additions) if steps]
    identities = {
        IdentityKind.VARIABLE: {variable.id for variable in spec.variables},
        IdentityKind.SCENARIO: {scenario.id for scenario in spec.scenarios},
    }
    merged: list[Transformation] = []
    while pending:
        blocked: dict[
            IdentityKind, list[tuple[list[Transformation], IdentityRename]]
        ] = {}
        for steps in pending:
            step = steps[-1]
            if isinstance(step, IdentityRename):
                previous = _rewind_rename(identities[step.dimension], step.mapping)
                if previous is None:
                    blocked.setdefault(step.dimension, []).append((steps, step))
                    continue
                identities[step.dimension] = previous
            merged.append(steps.pop())
            break
        else:
            for dimension, group in blocked.items():
                mapping = {
                    old: new for _, step in group for old, new in step.mapping.items()
                }
                if len(mapping) != sum(len(step.mapping) for _, step in group):
                    continue
                previous = _rewind_rename(identities[dimension], mapping)
                if previous is not None:
                    identities[dimension] = previous
                    merged.append(IdentityRename(dimension=dimension, mapping=mapping))
                    for steps, _ in group:
                        steps.pop()
                    break
            else:
                raise ValueError(
                    "Release provenance contains conflicting historical identities"
                )
        pending = [steps for steps in pending if steps]
    return tuple(reversed(merged))


def _merge(
    parts: tuple[DatasetMetadata, ...], source: SourceIdentity
) -> DatasetMetadata:
    spec = _merged_spec(parts)
    supports = [support for part in parts for support in part.spatial.supports]
    resolutions = {part.spatial.index_resolution for part in parts}
    return DatasetMetadata(
        spec=spec,
        spatial=SpatialContract(
            supports=tuple(dict.fromkeys(supports)) or (SupportKind.CELL,),
            index_resolution=next(iter(resolutions)) if len(resolutions) == 1 else None,
        ),
        source=source,
        report=ExecutionReport(False, None, source_identities=(source,)),
    )


def _superseded(
    scope: ExecutionScope, source: SourceIdentity, error: SourceContentChanged
) -> OSError:
    """Explain a release file whose published bytes differ from the release index."""
    release = "" if source.release is None else f" {source.release.id}"
    if scope.runtime.cache.mode is CacheMode.REUSE:
        return OSError(
            f"CRC Open release{release} in the reuse cache is no longer published "
            f"({error}). Cache.reuse keeps the release it first recorded, and this "
            "file was never cached. Read once with Cache.refresh(path) to move the "
            "cache to the current release."
        )
    return OSError(
        f"CRC Open file differs from its release index{release} ({error}). The "
        "release may have been replaced while it was read; read it again."
    )


@dataclass(frozen=True)
class _CRCNode(SourceNode[DatasetMetadata]):
    catalog: str
    release: ReleaseReference | None
    state: SelectionState = SelectionState()
    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _key(
                "crc-open",
                self.catalog,
                None
                if self.release is None
                else (self.release.id, self.release.sha256),
                self.state.document(),
            ),
        )

    def selected(self, state: SelectionState) -> _CRCNode:
        return replace(self, state=state)

    def _resolve(
        self, scope: ExecutionScope
    ) -> tuple[str, tuple[ReleaseDataset, ...], SourceIdentity]:
        with native_data(scope=scope, source="crc-open"):
            pointer_id: str | None = None
            path, resource = fetch(
                scope, self.catalog, license="CC-BY-NC-4.0", attribution="RiskThinking"
            )
            resources = [resource]
            uri = self.catalog
            # A parsed index is reused by checksum before its JSON is read again.
            parsed = _PARSED_INDEXES.get(resource.sha256)
            if parsed is None:
                document = _json(path.read_text(encoding="utf-8-sig"))
                if isinstance(document, Mapping) and set(document) == {
                    "schema_version",
                    "release",
                }:
                    if document["schema_version"] != "1.0":
                        raise ValueError("unsupported CRC Open release pointer")
                    pointer = _object(document["release"], {"id", "uri", "sha256"})
                    pointer_id = _identifier(pointer["id"], "release pointer id")
                    uri = urllib.parse.urljoin(self.catalog, _relative(pointer["uri"]))
                    path, resource = fetch(
                        scope,
                        uri,
                        expected_sha256=_digest(pointer["sha256"]),
                        license="CC-BY-NC-4.0",
                        attribution="RiskThinking",
                    )
                    resources.append(resource)
                    parsed = _PARSED_INDEXES.get(resource.sha256)
                    if parsed is None:
                        document = _json(path.read_text(encoding="utf-8-sig"))
                if parsed is None:
                    parsed = _parsed_index(document, resource.sha256)
            id, entries = parsed
            if pointer_id is not None and pointer_id != id:
                raise ValueError("release pointer identity differs from its index")
            actual = ReleaseReference(
                id, resource.sha256, retrieved_at=resource.retrieved_at
            )
            check_release(actual, self.release)
            source = SourceIdentity(
                provider="crc",
                product="crc-open",
                release=actual,
                resources=tuple(resources),
            )
            return uri, entries, source

    def _empty(self) -> bool:
        return any(
            value == ()
            for value in (
                self.state.variables,
                self.state.scenarios,
                self.state.periods,
                self.state.series,
            )
        )

    def inspection(self, scope: ExecutionScope) -> Inspection:
        if self._empty():
            source = SourceIdentity(provider="crc", product="crc-open")
            selected: tuple[
                tuple[ReleaseDataset, DatasetMetadata, ResolvedSelection], ...
            ] = ()
        else:
            _, entries, source = self._resolve(scope)
            selected = _selected(entries, self.state, source)
        metadata = _merge(tuple(part for _, part, _ in selected), source)
        return Inspection(
            schema=_schema(metadata),
            series=tuple(
                SeriesReference(
                    s.variable,
                    scenario=s.scenario,
                    period=s.period,
                    probability=s.probability,
                    release=source.release,
                )
                for s in metadata.spec.series
            ),
            releases=() if source.release is None else (source.release,),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
                Capability.FIT,
            ),
            source=source,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        if self._empty():
            return _checked_stream(
                scope,
                _merge((), SourceIdentity(provider="crc", product="crc-open")),
                iter(()),
            )
        catalog, entries, source = self._resolve(scope)
        selected = _selected(entries, self.state, source)
        parts = tuple(part for _, part, _ in selected)
        metadata = _merge(parts, source)
        with native_data(scope=scope, source=source.product):
            metadata = replace(
                metadata, provenance=_merged_provenance(parts, metadata.spec)
            )
        identities = list(source.resources)
        verified_reports: list[ExecutionReport] = []

        # Cells restrict which files are read, not what the release declares:
        # a finest-resolution candidate cell contains its asset's point, so a
        # file whose extent misses every such cell cannot cover any asset.
        boxes = (
            None
            if self.state.cells is None
            else tuple(
                cell_bounds(cell)
                for cell in self.state.cells
                if h3.get_resolution(cell) == 15
            )
        )

        read = tuple(
            (entry, resolved)
            for entry, _, resolved in selected
            if boxes is None
            or entry.extent is None
            or any(bounds_intersect(entry.extent, box) for box in boxes)
        )

        def acquire(
            item: tuple[ReleaseDataset, ResolvedSelection],
        ) -> tuple[Path, ResourceIdentity]:
            try:
                return fetch(
                    scope,
                    urllib.parse.urljoin(catalog, item[0].uri),
                    expected_sha256=item[0].sha256,
                    license="CC-BY-NC-4.0",
                    attribution="RiskThinking",
                )
            except SourceContentChanged as error:
                raise _superseded(scope, source, error) from error

        def batches() -> Iterator[pa.RecordBatch]:
            # Files are acquired ahead on Runtime(workers=...) threads.
            with closing(scope.prefetched(read, acquire)) as acquired:
                yield from verified(acquired)

        def verified(
            acquired: Iterator[tuple[Path, ResourceIdentity]],
        ) -> Iterator[pa.RecordBatch]:
            for (entry, resolved), (path, resource) in zip(read, acquired, strict=True):
                identities.append(resource)
                with pq.ParquetFile(path) as reader:
                    schema = reader.schema_arrow
                    stored = (schema.metadata or {}).get(b"crc.dataset")
                    if (
                        stored is None
                        or DatasetMetadata.from_json(stored.decode()).to_json()
                        != entry.metadata.to_json()
                    ):
                        raise ValueError(
                            "release dataset header differs from "
                            "its indexed declaration"
                        )
                    if not schema.equals(_schema(entry.metadata), check_metadata=False):
                        raise ValueError(
                            "release dataset Arrow schema contradicts its "
                            "indexed declaration"
                        )
                    report = entry.metadata.report
                    if not report.complete or report.valid is not True:
                        raise ValueError(
                            "release dataset requires a complete, valid report"
                        )
                    if reader.metadata.num_rows != report.counts.produced:
                        raise ValueError(
                            "release dataset produced count disagrees with "
                            "the Parquet footer"
                        )
                    # The checksum pins every byte to an index entry whose rows
                    # were fully validated when written, so the whole file is
                    # checked by its counts here and the delivered rows keep
                    # identity and support checks without decoding each curve.
                    produced = no_data = 0
                    supports: dict[object, bool] = {}
                    for batch in reader.iter_batches(
                        batch_size=scope.runtime.batch_rows
                    ):
                        produced += batch.num_rows
                        failures = batch.column("failure")
                        no_data += len(failures) - failures.null_count
                        selected_batch = batch.filter(
                            resolved.record_mask(batch, supports)
                        )
                        if selected_batch.num_rows:
                            yield selected_batch.replace_schema_metadata(
                                schema.metadata
                            )
                    if (
                        produced != report.counts.produced
                        or no_data != report.counts.no_data
                    ):
                        raise ValueError(
                            "release dataset report counts disagree with "
                            "the consumed rows"
                        )
                    verified_reports.append(
                        replace(report, source_identities=_sources(entry.metadata))
                    )

        stream = _checked_stream(scope, metadata, batches(), payload_verified=True)
        stream.verify_before_yield = True
        original = stream.report

        def report() -> ExecutionReport:
            assert original is not None
            final = replace(source, resources=tuple(identities))
            selected_counts = original().counts
            combined = _combine_reports(verified_reports)
            return replace(
                combined,
                counts=replace(
                    combined.counts,
                    requested=max(combined.counts.requested, selected_counts.requested),
                    produced=selected_counts.produced,
                    no_data=selected_counts.no_data,
                ),
                source_identities=tuple(
                    dict.fromkeys((final, *combined.source_identities))
                ),
            )

        stream.report = report
        stream.final_metadata = lambda: replace(
            metadata, source=replace(source, resources=tuple(identities))
        )
        return stream


class CRCOpen(Dataset):
    class Variable(str, Enum):
        """Common published CRC Open variables.

        Select any other published variable by its exact ID, such as
        ``"crc:hot_days"``; ``inspect()`` lists them all.
        """

        ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE = "crc:tx_max"
        ANNUAL_MINIMUM_DAILY_MAXIMUM_TEMPERATURE = "crc:tx_min"
        ANNUAL_MAXIMUM_DAILY_MINIMUM_TEMPERATURE = "crc:tn_max"
        ANNUAL_MINIMUM_DAILY_MINIMUM_TEMPERATURE = "crc:tn_min"
        ANNUAL_MAXIMUM_DAILY_PRECIPITATION = "crc:rx1day"
        ANNUAL_MAXIMUM_FIVE_DAY_PRECIPITATION = "crc:rx5day"
        ANNUAL_MEAN_WET_DAY_PRECIPITATION = "crc:sdii"
        ANNUAL_MAXIMUM_CONSECUTIVE_DRY_DAYS = "crc:max_consecutive_dry_days"
        ANNUAL_MAXIMUM_CONSECUTIVE_WET_DAYS = "crc:maximum_consecutive_wet_days"
        ANNUAL_FROST_DAY_COUNT = "crc:frost_days"
        ANNUAL_SUMMER_DAY_COUNT = "crc:summer_days"
        ANNUAL_TROPICAL_NIGHT_COUNT = "crc:tropical_nights"
        ANNUAL_MEAN_DAILY_TEMPERATURE_RANGE = "crc:dtr"
        ANNUAL_TOTAL_WET_DAY_PRECIPITATION = "crc:prcptot"

    class Scenario(str, Enum):
        SSP585 = "ssp585"

    def __init__(
        self,
        *,
        release: ReleaseReference | None = None,
        catalog: LocalPath | None = None,
    ) -> None:
        if release is not None and not isinstance(release, ReleaseReference):
            raise TypeError("release must be an inspected ReleaseReference")
        node = _CRCNode(_location(catalog, release), release)
        for name, value in (
            ("_base", node),
            ("_selection", SelectionState()),
            (
                "_operations",
                (Operation(Stage.DECODING, "Read current CRC Open release datasets"),),
            ),
            ("_path", None),
            ("_metadata", None),
        ):
            object.__setattr__(self, name, value)
