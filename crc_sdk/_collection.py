"""Compose independent scientific members in one reusable execution."""

from __future__ import annotations

import base64
import hashlib
import os
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ._collection_bundle import (
    MANIFEST,
    bundle_destination,
    publish_bundle,
    read_manifest,
    stage_bundle,
)
from ._collection_schema import (
    CollectionMember,
    CollectionMetadata,
    CollectionResource,
    CollectionSelection,
    _reference_matches,
    _release_key,
    _requested_ids,
    _typed_match,
    collection_arrow_schema,
)
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    SourceIdentity,
    SpatialContract,
    Stage,
    _dump,
    _identifier,
)
from ._declarations import Scenario, Variable
from ._errors import ValidationError
from ._graph import DataValidationError, ExecutionScope, Node, NodeStream
from ._inspection import (
    Bounds,
    Capability,
    Explanation,
    Inspection,
    Operation,
    Requirement,
    SeriesReference,
)
from ._recipes import (
    CanonicalReader,
    Dataset,
    _bounded_node,
    _Checked,
    _checked_stream,
    _invalid,
    _key,
    _report_view,
    _select_node,
    _sources,
    _TransformNode,
)
from ._runtime import LocalPath, Runtime, _local_path
from ._schema import dataset_arrow_schema
from ._selection import (
    IdentifierSelector,
    PeriodSelector,
    ResolvedSelection,
    ScenarioSelector,
    SelectionState,
    SeriesSelector,
    VariableSelector,
    _series_key,
    participating_spec,
    rename_metadata,
    rename_spec,
    selected_metadata,
)
from ._source_io import native_data
from ._terminals import BatchStream, Execution, ReadResult

if TYPE_CHECKING:
    from ._assets import Assets
    from ._evaluation import Evaluation
    from ._evaluation_spatial import SpatialMatch


@dataclass(frozen=True, slots=True)
class _Entry:
    resource: CollectionResource
    node: Node[DatasetMetadata]


@dataclass(frozen=True, slots=True)
class _Plan:
    metadata: CollectionMetadata
    entries: Mapping[str, _Entry]
    requirements: tuple[Requirement, ...] = ()


def _known_ids(
    requested: set[str] | None,
    available: set[str],
    *,
    dimension: str,
) -> None:
    if requested is not None and (missing := requested - available):
        raise ValueError(f"unknown requested {dimension} {sorted(missing)[0]!r}")


def _project_entries(
    entries: Mapping[str, _Entry],
    selection: CollectionSelection,
    *,
    discovery_incomplete: bool,
) -> dict[str, _Entry]:
    variables = _requested_ids(selection.variables)
    scenarios = _requested_ids(selection.scenarios)
    periods = _requested_ids(selection.periods)
    specs = [entry.resource.metadata.spec for entry in entries.values()]
    if not discovery_incomplete:
        _known_ids(
            variables,
            {v.id for spec in specs for v in spec.variables},
            dimension="identity",
        )
        _known_ids(
            scenarios,
            {s.id for spec in specs for s in spec.scenarios},
            dimension="identity",
        )
        _known_ids(
            periods,
            {p.id for spec in specs for p in spec.periods},
            dimension="identity",
        )
    references = selection.series
    matching_references: set[int] = set()
    series_keys = {_series_key(item) for spec in specs for item in spec.series}
    projected = {}
    for resource_id, entry in entries.items():
        original = entry.resource.metadata
        release = None if original.source is None else original.source.release
        selected = []
        for item in original.spec.series:
            if references is not None:
                matches = {
                    index
                    for index, reference in enumerate(references)
                    if _reference_matches(item, reference, release)
                }
                matching_references.update(matches)
                if not matches:
                    continue
            if variables is not None and item.variable.id not in variables:
                continue
            if scenarios is not None and (
                item.scenario is None or item.scenario.id not in scenarios
            ):
                continue
            if periods is not None and (
                item.period is None or item.period.id not in periods
            ):
                continue
            _typed_match(selection.variables, item.variable)
            _typed_match(selection.scenarios, item.scenario)
            _typed_match(selection.periods, item.period)
            selected.append(item)
        if not selected:
            if not original.spec.series and all(
                value is None for value in (variables, scenarios, periods, references)
            ):
                projected[resource_id] = entry
            continue
        spec = participating_spec(original.spec, selected)
        if spec == original.spec and selection.area is None:
            projected[resource_id] = entry
            continue
        resolved = ResolvedSelection(
            spec=spec,
            keys=frozenset(_series_key(item) for item in selected),
            area=selection.area,
        )
        descriptors = tuple(
            SeriesReference(
                item.variable, item.scenario, item.period, probability=item.probability
            )
            for item in selected
        )
        state = SelectionState(series=descriptors, area=selection.area)
        metadata = selected_metadata(original, resolved)
        # Selecting over a bounded source must preserve its upstream extent.
        node = _select_node(entry.node, state)
        projected[resource_id] = _Entry(
            replace(entry.resource, metadata=metadata, path=None, sha256=None), node
        )
    if references is not None and not discovery_incomplete:
        for index, reference in enumerate(references):
            if index in matching_references:
                continue
            if _series_key(reference) not in series_keys:
                raise ValueError(f"unknown series identity {_series_key(reference)!r}")
            raise ValueError("selected series release differs from the source snapshot")
    return projected


def _renamed_selection(
    selected: CollectionSelection,
    variables: Mapping[str, str],
    scenarios: Mapping[str, str],
) -> CollectionSelection:
    def variable(value: Variable | str) -> Variable | str:
        if isinstance(value, str):
            return variables.get(value, value)
        return Variable(
            variables.get(value.id, value.id),
            measurement=value.measurement,
            label=value.label,
        )

    def scenario(value: Scenario | str) -> Scenario | str:
        if isinstance(value, str):
            return scenarios.get(value, value)
        return Scenario(
            scenarios.get(value.id, value.id),
            definition=value.definition,
            schema=value.schema,
            label=value.label,
        )

    def reference(value: SeriesReference) -> SeriesReference:
        renamed_variable = variable(value.variable)
        renamed_scenario = None if value.scenario is None else scenario(value.scenario)
        assert isinstance(renamed_variable, Variable)
        assert renamed_scenario is None or isinstance(renamed_scenario, Scenario)
        return replace(value, variable=renamed_variable, scenario=renamed_scenario)

    return replace(
        selected,
        variables=None
        if selected.variables is None
        else tuple(variable(item) for item in selected.variables),
        scenarios=None
        if selected.scenarios is None
        else tuple(scenario(item) for item in selected.scenarios),
        series=None
        if selected.series is None
        else tuple(reference(item) for item in selected.series),
    )


class _ReaderNode(Node[DatasetMetadata]):
    def __init__(self, reader: CanonicalReader) -> None:
        self.reader = reader
        self.key = _key("canonical-reader", id(reader))

    def _view_report(
        self, scope: ExecutionScope, report: ExecutionReport
    ) -> ExecutionReport:
        metadata = scope.computation_metadata(self)
        declared = replace(metadata.report, source_identities=_sources(metadata))
        if declared.complete:
            return _report_view(report, baseline=declared)
        return _report_view(
            report,
            parent=replace(declared, complete=True, valid=declared.valid is not False),
        )

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        metadata = self.reader.inspect()
        if not isinstance(metadata, DatasetMetadata):
            raise TypeError("CanonicalReader.inspect() must return DatasetMetadata")

        def batches() -> Iterator[pa.RecordBatch]:
            with self.reader.batches(batch_rows=scope.runtime.batch_rows) as stream:
                yield from stream

        return _checked_stream(scope, metadata, batches())


class _BundleResourceNode(Node[DatasetMetadata]):
    def __init__(
        self, directory: Path, resource: CollectionResource, area: Bounds | None
    ) -> None:
        assert resource.path is not None and resource.sha256 is not None
        self.directory = directory
        self.resource = resource
        self.area = area
        self.key = _key(
            "collection-file",
            str(directory),
            resource.id,
            resource.sha256,
            SelectionState(area=area).document()["area"],
        )

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        metadata = self.resource.metadata
        assert self.resource.path is not None
        path = self.directory / self.resource.path

        def batches() -> Iterator[pa.RecordBatch]:
            if (
                self.directory.is_symlink()
                or path.parent.is_symlink()
                or path.is_symlink()
            ):
                raise _invalid(
                    ValueError("collection resources must be owned local files")
                )
            pinned, identity = scope.pin_local(path)
            if identity.sha256 != self.resource.sha256:
                raise OSError(
                    f"collection resource checksum mismatch for {self.resource.id!r}"
                )
            with (
                native_data(scope=scope, source=self.resource.id),
                pq.ParquetFile(pinned) as reader,
            ):
                expected = dataset_arrow_schema(metadata)
                if (
                    not reader.schema_arrow.equals(expected, check_metadata=False)
                    or reader.schema_arrow.metadata != expected.metadata
                    or reader.metadata.num_rows != metadata.report.counts.produced
                ):
                    raise _invalid(
                        ValueError(
                            "collection resource metadata differs from manifest "
                            f"for {self.resource.id!r}"
                        )
                    )
                if reader.num_row_groups:
                    selection = ResolvedSelection(
                        spec=metadata.spec,
                        keys=frozenset(_series_key(s) for s in metadata.spec.series),
                        area=self.area,
                    )
                    for batch in reader.iter_batches(
                        batch_size=scope.runtime.batch_rows
                    ):
                        if self.area is not None and any(
                            not selection.matches_record(row)
                            for row in batch.to_pylist()
                        ):
                            raise _invalid(
                                ValueError(
                                    "resource rows contradict the saved area selection"
                                )
                            )
                        yield batch

        return _checked_stream(scope, metadata, batches())

    def _view_report(
        self, scope: ExecutionScope, report: ExecutionReport
    ) -> ExecutionReport:
        return _report_view(report, baseline=self.resource.metadata.report)


class _CollectionNode(Node[CollectionMetadata]):
    def __init__(self, recipe: DatasetCollection) -> None:
        self.recipe = recipe
        self.key = recipe._key

    def open(self, scope: ExecutionScope) -> NodeStream[CollectionMetadata]:
        plan = self.recipe._plan(scope)
        metadata = plan.metadata
        started: list[_Entry] = []
        produced = 0
        no_data = 0

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal produced, no_data
            for member in metadata.members:
                member_check = None
                for resource_id in member.resource_ids:
                    entry = plan.entries[resource_id]
                    started.append(entry)
                    if member_check is None and len(member.resource_ids) > 1:
                        contracts = tuple(
                            plan.entries[key].resource.metadata.spatial
                            for key in member.resource_ids
                        )
                        supports = tuple(
                            dict.fromkeys(
                                support
                                for contract in contracts
                                for support in contract.supports
                            )
                        )
                        resolutions = {
                            contract.index_resolution for contract in contracts
                        }
                        spatial = SpatialContract(
                            supports=supports,
                            index_resolution=(
                                next(iter(resolutions))
                                if len(resolutions) == 1
                                else None
                            ),
                        )
                        combined = replace(
                            entry.resource.metadata,
                            spec=metadata.spec,
                            spatial=spatial,
                            provenance=(),
                        )
                        member_check = _Checked(scope, combined)
                    for batch in scope.iter_batches(entry.node):
                        if member_check is not None:
                            member_check.check(batch)
                        result = pa.RecordBatch.from_arrays(
                            [
                                pa.repeat(member.id, batch.num_rows),
                                pa.repeat(resource_id, batch.num_rows),
                                *batch.columns,
                            ],
                            schema=collection_arrow_schema(),
                        )
                        produced += batch.num_rows
                        no_data += batch.num_rows - batch.column("failure").null_count
                        yield result
                    child = scope.report(entry.node)
                    if child.valid is False:
                        if child.diagnostics:
                            raise DataValidationError(child.diagnostics)
                        raise _invalid(
                            ValueError(
                                f"collection resource {resource_id!r} is invalid"
                            )
                        )

        def report() -> ExecutionReport:
            reports = [scope.report(entry.node) for entry in started]
            diagnostics: dict[str, ExecutionDiagnostic] = {}
            sources: dict[str, SourceIdentity] = {}
            for child in reports:
                for diagnostic in child.diagnostics:
                    diagnostics.setdefault(_dump(diagnostic._document()), diagnostic)
                for source in child.source_identities:
                    sources.setdefault(_dump(source._document()), source)
            counts = {
                name: sum(getattr(result.counts, name) for result in reports)
                for name in ExecutionCounts.__dataclass_fields__
            }
            counts.update(produced=produced, no_data=no_data)
            return ExecutionReport(
                False,
                None,
                tuple(diagnostics.values()),
                ExecutionCounts(**counts),
                tuple(sources.values()),
            )

        def final_metadata() -> CollectionMetadata:
            resources = []
            reported = {entry.resource.id for entry in started}
            for resource in metadata.resources:
                if resource.id not in reported:
                    resources.append(resource)
                    continue
                node = plan.entries[resource.id].node
                updated = scope.metadata(node)
                resources.append(
                    replace(
                        resource,
                        metadata=replace(
                            updated,
                            report=scope.report(node),
                        ),
                    )
                )
            return replace(metadata, resources=tuple(resources))

        return NodeStream(
            collection_arrow_schema(metadata),
            metadata,
            batches(),
            report=report,
            final_metadata=final_metadata,
        )


class DatasetCollection:
    """Independent named datasets read as one normalized Arrow table."""

    def __init__(self, members: Mapping[str, Dataset | CanonicalReader]) -> None:
        if not isinstance(members, Mapping):
            raise TypeError(
                "members must map exact identifiers to Dataset or CanonicalReader"
            )
        inputs = []
        for key, member in members.items():
            identifier = _identifier(key, "member id")
            if not isinstance(member, (Dataset, CanonicalReader)):
                raise TypeError(
                    f"member {identifier!r} must be Dataset or CanonicalReader"
                )
            inputs.append((identifier, member))
        self._initialize(inputs=tuple(inputs))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("collection recipes are immutable")

    _inputs: tuple[tuple[str, Dataset | CanonicalReader], ...]
    _path: Path | None
    _snapshot: CollectionMetadata | None
    _collected_entries: Mapping[str, _Entry] | None
    _parent: DatasetCollection | None
    _variables: Mapping[str, str]
    _scenarios: Mapping[str, str]
    _selection: CollectionSelection
    _overrides: frozenset[str]
    _key: str

    def _initialize(
        self,
        *,
        inputs: tuple[tuple[str, Dataset | CanonicalReader], ...] = (),
        path: Path | None = None,
        snapshot: CollectionMetadata | None = None,
        collected_entries: Mapping[str, _Entry] | None = None,
        parent: DatasetCollection | None = None,
        variables: Mapping[str, str] | None = None,
        scenarios: Mapping[str, str] | None = None,
        selection: CollectionSelection | None = None,
        overrides: frozenset[str] = frozenset(),
    ) -> None:
        selection = selection or CollectionSelection()
        variables = MappingProxyType(dict(variables or {}))
        scenarios = MappingProxyType(dict(scenarios or {}))
        for name, value in (
            ("_inputs", inputs),
            ("_path", path),
            ("_snapshot", snapshot),
            ("_collected_entries", collected_entries),
            ("_parent", parent),
            ("_variables", variables),
            ("_scenarios", scenarios),
            ("_selection", selection),
            ("_overrides", overrides),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "_key",
            _key(
                "collection",
                [
                    [
                        name,
                        dataset._node.key
                        if isinstance(dataset, Dataset)
                        else id(dataset),
                        id(dataset),
                    ]
                    for name, dataset in inputs
                ],
                None if path is None else str(path),
                None if snapshot is None else snapshot.to_json(),
                None
                if collected_entries is None
                else {
                    identifier: entry.node.key
                    for identifier, entry in collected_entries.items()
                },
                None if parent is None else parent._key,
                dict(variables),
                dict(scenarios),
                selection._document(),
                sorted(overrides),
            ),
        )

    @classmethod
    def open(cls, directory: LocalPath) -> DatasetCollection:
        value = object.__new__(cls)
        value._initialize(path=_local_path(directory))
        return value

    @classmethod
    def _collected(
        cls, table: pa.Table, metadata: CollectionMetadata
    ) -> DatasetCollection:
        if not table.schema.equals(
            collection_arrow_schema(metadata), check_metadata=True
        ):
            raise ValueError("collected Arrow table must match its collection metadata")
        if not metadata.report.complete or metadata.report.valid is not True:
            raise ValueError("collection reuse requires a complete valid result")
        if table.num_rows != metadata.report.counts.produced:
            raise ValueError("collected row count differs from its collection report")
        entries = {}
        for resource in metadata.resources:
            member = next(m for m in metadata.members if resource.id in m.resource_ids)
            selected = table.filter(
                pc.and_(
                    pc.equal(table["member_id"], member.id),
                    pc.equal(table["resource_id"], resource.id),
                )
            )
            values = selected.drop(
                ["member_id", "resource_id"]
            ).replace_schema_metadata(dataset_arrow_schema(resource.metadata).metadata)
            entries[resource.id] = _Entry(
                resource, Dataset._collected(values, resource.metadata)._node
            )
        value = object.__new__(cls)
        value._initialize(
            snapshot=metadata,
            collected_entries=MappingProxyType(entries),
        )
        return value

    @property
    def _node(self) -> Node[CollectionMetadata]:
        return _CollectionNode(self)

    @property
    def path(self) -> Path | None:
        return None if self._overrides else self._path

    def _read_manifest(self) -> CollectionMetadata:
        if self._snapshot is not None:
            return self._snapshot
        assert self._path is not None
        return read_manifest(self._path)

    @property
    def metadata(self) -> CollectionMetadata | None:
        if self._overrides or self._parent is not None:
            return None
        if self._snapshot is None and self._path is None:
            return None
        try:
            return self._read_manifest()
        except (ValueError, TypeError) as error:
            problem = _invalid(error)
            raise ValidationError(
                report=ExecutionReport(False, False, problem.diagnostics)
            ) from error

    @property
    def report(self) -> ExecutionReport | None:
        metadata = self.metadata
        return None if metadata is None else metadata.report

    def select(
        self,
        *,
        members: IdentifierSelector | types.EllipsisType = ...,
        variables: VariableSelector | types.EllipsisType = ...,
        scenarios: ScenarioSelector | types.EllipsisType = ...,
        periods: PeriodSelector | types.EllipsisType = ...,
        series: SeriesSelector | types.EllipsisType = ...,
        area: Bounds | None | types.EllipsisType = ...,
    ) -> Self:
        arguments = dict(
            members=members,
            variables=variables,
            scenarios=scenarios,
            periods=periods,
            series=series,
            area=area,
        )
        overrides = self._overrides | {
            name for name, item in arguments.items() if item is not ...
        }
        value = object.__new__(type(self))
        value._initialize(
            inputs=self._inputs,
            path=self._path,
            snapshot=self._snapshot,
            collected_entries=self._collected_entries,
            parent=self._parent,
            variables=self._variables,
            scenarios=self._scenarios,
            selection=self._selection._updated(
                members=members,
                variables=variables,
                scenarios=scenarios,
                periods=periods,
                series=series,
                area=area,
            ),
            overrides=overrides,
        )
        return value

    def _project_variables(self, ids: tuple[str, ...]) -> DatasetCollection:
        selected = self._selection.variables
        if selected is None:
            return self.select(variables=ids)
        names = set(ids)
        return self.select(
            variables=tuple(
                value
                for value in selected
                if (value if isinstance(value, str) else value.id) in names
            )
        )

    def rename(
        self,
        *,
        variables: Mapping[str, str] | None = None,
        scenarios: Mapping[str, str] | None = None,
    ) -> Self:
        if variables is None and scenarios is None:
            return self
        for mapping in (variables, scenarios):
            if mapping is not None:
                if not isinstance(mapping, Mapping):
                    raise TypeError("renaming requires mappings of exact identifiers")
                for old, new in mapping.items():
                    _identifier(old, "old id")
                    _identifier(new, "new id")
        value = object.__new__(type(self))
        value._initialize(parent=self, variables=variables, scenarios=scenarios)
        return value

    def _manifest_inputs(self) -> _Plan:
        try:
            manifest = self._read_manifest()
        except (ValueError, TypeError) as error:
            raise _invalid(error) from error
        if self._collected_entries is not None:
            return _Plan(manifest, self._collected_entries)
        assert self._path is not None
        entries = {
            resource.id: _Entry(
                resource,
                _BundleResourceNode(self._path, resource, manifest.selection.area),
            )
            for resource in manifest.resources
        }
        return _Plan(manifest, entries)

    def _memory_inputs(self, scope: ExecutionScope, inspect: bool) -> _Plan:
        ids = tuple(name for name, _ in self._inputs)
        _known_ids(
            _requested_ids(self._selection.members), set(ids), dimension="member"
        )
        names = (
            set(ids)
            if self._selection.members is None
            else set(self._selection.members)
        )
        members = []
        entries = {}
        requirements: list[Requirement] = []
        discovered: dict[int, Inspection] = {}
        for name, dataset in self._inputs:
            if name not in names:
                continue
            if self._selection._empty():
                members.append(CollectionMember(id=name, resource_ids=()))
                continue
            if isinstance(dataset, Dataset):
                node = dataset._node
                if self._selection.area is not None:
                    node = _bounded_node(
                        node, SelectionState(area=self._selection.area)
                    )
                if inspect:
                    identity = id(dataset)
                    if identity not in discovered:
                        temporary = Dataset._create(node, operations=())
                        discovered[identity] = temporary._inspect(scope)
                    details = discovered[identity]
                    requirements.extend(details.requirements)
                    metadata = details.metadata
                    if not isinstance(metadata, DatasetMetadata):
                        members.append(CollectionMember(id=name, resource_ids=()))
                        continue
                else:
                    metadata = scope.metadata(node)
            else:
                node = _ReaderNode(dataset)
                metadata = scope.metadata(node)
            resource_id = _key(
                "resource",
                node.key,
                metadata.spec._document(),
                None
                if metadata.source is None
                else (
                    metadata.source.provider,
                    metadata.source.product,
                    metadata.source.member,
                    _release_key(metadata.source.release),
                    [resource.sha256 for resource in metadata.source.resources],
                ),
                [p._document() for p in metadata.provenance],
            )
            members.append(CollectionMember(id=name, resource_ids=(resource_id,)))
            entries[resource_id] = _Entry(
                CollectionResource(id=resource_id, metadata=metadata), node
            )
        # The current selection may remove conflicting definitions from the
        # input inventory, so it is applied before reconciled metadata exists.
        return self._participating(
            tuple(members),
            entries,
            CollectionSelection(),
            tuple(requirements),
        )

    def _participating(
        self,
        members: tuple[CollectionMember, ...],
        entries: Mapping[str, _Entry],
        previous: CollectionSelection,
        requirements: tuple[Requirement, ...],
        source: SourceIdentity | None = None,
    ) -> _Plan:
        allowed = _requested_ids(self._selection.members)
        _known_ids(allowed, {member.id for member in members}, dimension="member")
        members = tuple(
            member for member in members if allowed is None or member.id in allowed
        )
        used = {key for member in members for key in member.resource_ids}
        selected = (
            {}
            if self._selection._empty()
            else _project_entries(
                {key: value for key, value in entries.items() if key in used},
                self._selection,
                discovery_incomplete=bool(requirements),
            )
        )
        names = set(selected)
        members = tuple(
            replace(
                member,
                resource_ids=tuple(key for key in member.resource_ids if key in names),
            )
            for member in members
        )
        state = replace(
            previous,
            **{name: getattr(self._selection, name) for name in self._overrides},
        )
        try:
            metadata = CollectionMetadata(
                members=members,
                resources=tuple(entry.resource for entry in selected.values()),
                selection=state,
                report=ExecutionReport(False, None),
                source=source,
            )
        except (ValueError, TypeError) as error:
            raise _invalid(error, stage=Stage.SELECTION) from error
        return _Plan(metadata, selected, tuple(dict.fromkeys(requirements)))

    def _plan(self, scope: ExecutionScope, *, inspect: bool = False) -> _Plan:
        if self._path is not None or self._collected_entries is not None:
            base = self._manifest_inputs()
        elif self._parent is not None:
            base = self._parent._plan(scope, inspect=inspect)
            # Reconcile the effective mapping once across participating members,
            # including unchanged identities in different physical resources.
            rename_spec(
                base.metadata.spec, variables=self._variables, scenarios=self._scenarios
            )
            entries = {}
            for resource_id, entry in base.entries.items():
                source = entry.resource.metadata
                variables = {
                    old: new
                    for old, new in self._variables.items()
                    if old in {v.id for v in source.spec.variables}
                }
                scenarios = {
                    old: new
                    for old, new in self._scenarios.items()
                    if old in {s.id for s in source.spec.scenarios}
                }
                metadata = rename_metadata(
                    source, variables=variables, scenarios=scenarios
                )
                node = _TransformNode(
                    entry.node,
                    "rename",
                    {"variables": variables, "scenarios": scenarios},
                )
                entries[resource_id] = _Entry(
                    replace(entry.resource, metadata=metadata, path=None, sha256=None),
                    node,
                )
            return self._participating(
                base.metadata.members,
                entries,
                _renamed_selection(
                    base.metadata.selection, self._variables, self._scenarios
                ),
                base.requirements,
                base.metadata.source,
            )
        else:
            return self._memory_inputs(scope, inspect)
        return self._participating(
            base.metadata.members,
            base.entries,
            base.metadata.selection,
            base.requirements,
            base.metadata.source,
        )

    def _inspect(self, scope: ExecutionScope) -> Inspection:
        plan = self._plan(scope, inspect=True)
        metadata = plan.metadata
        references = tuple(
            SeriesReference(
                series.variable,
                series.scenario,
                series.period,
                release=None
                if resource.metadata.source is None
                else resource.metadata.source.release,
                probability=series.probability,
            )
            for resource in metadata.resources
            for series in resource.metadata.spec.series
        )
        return Inspection(
            schema=collection_arrow_schema(metadata),
            series=references,
            releases=tuple(
                dict.fromkeys(
                    ref.release for ref in references if ref.release is not None
                )
            ),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
                Capability.EVALUATE,
            ),
            requirements=plan.requirements,
            source=metadata.source,
        )

    def inspect(self, *, runtime: Runtime | None = None) -> Inspection:
        with ExecutionScope(runtime) as scope:
            try:
                return self._inspect(scope)
            except DataValidationError as error:
                raise ValidationError(
                    report=ExecutionReport(False, False, error.diagnostics)
                ) from error

    def explain(self, *, runtime: Runtime | None = None) -> Explanation:
        config = Runtime() if runtime is None else runtime
        if not isinstance(config, Runtime):
            raise TypeError("runtime must be Runtime")
        operations: list[Operation] = []
        requirements: list[Requirement] = []
        names = self._selection.members
        children: list[Dataset | DatasetCollection] = []
        for name, dataset in self._inputs:
            if isinstance(dataset, Dataset) and (names is None or name in names):
                if self._selection.area is not None:
                    dataset = Dataset._create(
                        _bounded_node(
                            dataset._node, SelectionState(area=self._selection.area)
                        ),
                        operations=dataset.explain(runtime=config).operations,
                    )
                children.append(dataset)
        if self._parent is not None:
            children.append(self._parent)
        if not self._selection._empty():
            for child in children:
                details = child.explain(runtime=config)
                operations.extend(details.operations)
                requirements.extend(details.requirements)
        operations.append(
            Operation(Stage.SELECTION, "Select independent collection members")
        )
        return Explanation(
            operations=tuple(dict.fromkeys(operations)),
            requirements=tuple(dict.fromkeys(requirements)),
            cache=config.cache,
            workers=config.workers,
            batch_rows=config.batch_rows,
            memory_limit=config.memory_limit,
            source=None if self._snapshot is None else self._snapshot.source,
        )

    def evaluate(
        self, assets: Assets, *, spatial: SpatialMatch | None = None
    ) -> Evaluation:
        from ._evaluation import Evaluation

        return Evaluation._create(self, assets, spatial=spatial)

    def read(
        self, *, limit: int | None = None, runtime: Runtime | None = None
    ) -> ReadResult[CollectionMetadata]:
        with self.execute(runtime=runtime) as execution:
            return execution.read(limit=limit)

    def batches(
        self,
        *,
        batch_rows: int | None = None,
        runtime: Runtime | None = None,
    ) -> BatchStream[CollectionMetadata]:
        return BatchStream(ExecutionScope(runtime), self._node, batch_rows=batch_rows)

    def validate(self, *, runtime: Runtime | None = None) -> ExecutionReport:
        with self.execute(runtime=runtime) as execution:
            return execution.validate()

    def execute(
        self, *, runtime: Runtime | None = None
    ) -> Execution[CollectionMetadata]:
        return _CollectionExecution(self, runtime=runtime)

    def write(
        self,
        path: LocalPath,
        *,
        overwrite: bool = False,
        runtime: Runtime | None = None,
    ) -> DatasetCollection:
        with _CollectionExecution(self, runtime=runtime) as execution:
            return execution.write(path, overwrite=overwrite)

    def _saved(
        self, path: Path, metadata: CollectionMetadata, digest: str
    ) -> DatasetCollection:
        raise RuntimeError("a collection must be written as a directory bundle")


class _CollectionExecution(Execution[CollectionMetadata]):
    def write(self, path: LocalPath, *, overwrite: bool = False) -> DatasetCollection:
        self._active()
        destination, existing = bundle_destination(path, overwrite)
        scope = self._scope
        node = self._recipe._node
        destination.parent.mkdir(parents=True, exist_ok=True)
        with stage_bundle(destination) as staging:
            try:
                metadata = _write_resources(scope, node, staging)
            except DataValidationError as error:
                raise ValidationError(report=self.report) from error
            with (staging / MANIFEST).open("w", encoding="utf-8") as stream:
                stream.write(metadata.to_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if read_manifest(staging) != metadata:
                raise ValueError("staged collection manifest failed its round trip")
            publish_bundle(staging, destination, existing=existing)
            result = object.__new__(DatasetCollection)
            result._initialize(path=destination, snapshot=metadata)
            return result


def _finish_resource(
    scope: ExecutionScope,
    node: Node[CollectionMetadata],
    writer: pq.ParquetWriter,
    resource_id: str,
    staging: Path,
) -> CollectionResource:
    resource = next(
        item for item in scope.metadata(node).resources if item.id == resource_id
    )
    if (
        not resource.metadata.report.complete
        or resource.metadata.report.valid is not True
    ):
        raise ValidationError(report=scope.report(node))
    schema = dataset_arrow_schema(resource.metadata)
    footer = dict(schema.metadata or {})
    footer[b"ARROW:schema"] = base64.b64encode(schema.serialize().to_pybytes())
    try:
        writer.add_key_value_metadata(footer)
    finally:
        writer.close()
    relative = _resource_path(resource_id)
    with (staging / relative).open("rb") as handle:
        os.fsync(handle.fileno())
        digest = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
    return replace(resource, path=relative, sha256=digest)


def _resource_path(resource_id: str) -> str:
    return "resources/" + hashlib.sha256(resource_id.encode()).hexdigest() + ".parquet"


def _write_resources(
    scope: ExecutionScope, node: Node[CollectionMetadata], staging: Path
) -> CollectionMetadata:
    metadata = scope.metadata(node)
    owners: dict[str, str] = {}
    for member in metadata.members:
        for resource_id in member.resource_ids:
            owners.setdefault(resource_id, member.id)
    if owners:
        (staging / "resources").mkdir()
    written = 0
    persisted = {}
    writer = None
    current = None
    try:
        for batch in scope.iter_batches(node):
            if not batch.num_rows:
                continue
            member_id = batch.column("member_id")[0].as_py()
            resource_id = batch.column("resource_id")[0].as_py()
            if owners[resource_id] != member_id:
                continue
            if resource_id != current:
                if writer is not None:
                    assert current is not None
                    persisted[current] = _finish_resource(
                        scope, node, writer, current, staging
                    )
                current = resource_id
                writer = pq.ParquetWriter(
                    staging / _resource_path(resource_id), dataset_arrow_schema()
                )
            values = pa.RecordBatch.from_arrays(
                batch.columns[2:], schema=dataset_arrow_schema()
            )
            assert writer is not None
            writer.write_batch(values)
            written += batch.num_rows
            scope.emit(Stage.WRITING, written)
        if writer is not None:
            assert current is not None
            persisted[current] = _finish_resource(scope, node, writer, current, staging)
            writer = None
        report = scope.report(node)
        if report.valid is not True or not report.complete:
            raise ValidationError(report=report)
        for resource_id in owners.keys() - persisted.keys():
            writer = pq.ParquetWriter(
                staging / _resource_path(resource_id), dataset_arrow_schema()
            )
            persisted[resource_id] = _finish_resource(
                scope, node, writer, resource_id, staging
            )
            writer = None
        scope.emit(Stage.WRITING, written, written)
    finally:
        if writer is not None:
            writer.close()
    return replace(
        metadata,
        resources=tuple(persisted[identifier] for identifier in owners),
        report=report,
    )
