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
from typing import Self, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

import crc_sdk

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
    _collection_report,
    _reference_matches,
    _release_key,
    _requested_ids,
    _typed_match,
    collection_arrow_schema,
)
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionReport,
    ReleaseReference,
    SourceIdentity,
    Stage,
    _dump,
    _identifier,
)
from ._declarations import Period, Scenario, SeriesSpec, Variable, scientific_identity
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
    _declared_variable_ids,
    _invalid,
    _key,
    _pending_payload_release,
    _report_view,
    _select_node,
    _sources,
    _TransformNode,
)
from ._runtime import LocalPath, Runtime, _local_path
from ._schema import dataset_arrow_schema
from ._schema.dataset import _ROW_KEY
from ._selection import (
    IdentifierSelector,
    PeriodSelector,
    ResolvedSelection,
    ScenarioSelector,
    SelectionState,
    SeriesSelector,
    VariableSelector,
    _rename_mapping,
    _series_key,
    participating_spec,
    rename_metadata,
    selected_metadata,
)
from ._source_io import check_release, native_data
from ._terminals import BatchStream, Execution, ReadResult


@dataclass(frozen=True, slots=True)
class _Entry:
    resource: CollectionResource
    node: Node[DatasetMetadata]
    payload_pin: ReleaseReference | None = None

    def matches(
        self, item: SeriesSpec, reference: SeriesReference, *, check_science: bool
    ) -> bool:
        source = self.resource.metadata.source
        if (
            reference.release is not None
            and _release_key(reference.release) == _release_key(self.payload_pin)
            and source is not None
            and source.release is not None
            and source.release.id == f"inventory:{reference.release.id}"
        ):
            reference = replace(reference, release=None)
        return _reference_matches(
            item, reference, self.resource.metadata, check_science=check_science
        )

    def verify_payload(self, scope: ExecutionScope) -> None:
        if self.payload_pin is None:
            return
        source = scope.metadata(self.node).source
        if source is None or source.release is None:
            raise OSError("source did not report its required exact payload release")
        check_release(source.release, self.payload_pin)


@dataclass(frozen=True, slots=True)
class _Plan:
    members: tuple[CollectionMember, ...]
    entries: Mapping[str, _Entry]
    selection: CollectionSelection
    requirements: tuple[Requirement, ...] = ()
    source: SourceIdentity | None = None
    bundle_resources: tuple[_BundleResourceNode, ...] = ()

    def metadata(self) -> CollectionMetadata:
        try:
            return CollectionMetadata(
                members=self.members,
                resources=tuple(entry.resource for entry in self.entries.values()),
                selection=self.selection,
                report=ExecutionReport(False, None),
                source=self.source,
            )
        except (ValueError, TypeError) as error:
            raise _invalid(error, stage=Stage.SELECTION) from error


@dataclass(frozen=True, slots=True)
class _PlanningScope:
    selections: tuple[CollectionSelection, ...] = ()
    variables: frozenset[str] | None = None
    rename_variables: frozenset[str] = frozenset()
    rename_scenarios: frozenset[str] = frozenset()

    @property
    def members(self) -> frozenset[str] | None:
        names = None
        for selection in self.selections:
            if selection.members is not None:
                selected = frozenset(selection.members)
                names = selected if names is None else names & selected
        return names

    @property
    def active_variables(self) -> frozenset[str] | None:
        active = self.variables
        for selection in self.selections:
            if selection._empty():
                return frozenset()
            for ids in (
                _requested_ids(selection.variables),
                None
                if selection.series is None
                else {ref.variable.id for ref in selection.series},
            ):
                if ids is not None:
                    active = frozenset(ids) if active is None else active & ids
        return active

    @property
    def source_state(self) -> SelectionState:
        active = self.active_variables
        references: list[SeriesReference] = []
        for selection in self.selections:
            for reference in selection.series or ():
                if (
                    active is None or reference.variable.id in active
                ) and reference not in references:
                    references.append(reference)
        return SelectionState(
            area=next((s.area for s in self.selections if s.area is not None), None),
            variables=None if active is None else tuple(sorted(active)),
            series=tuple(references) or None,
        )

    def before_rename(
        self, variables: Mapping[str, str], scenarios: Mapping[str, str]
    ) -> _PlanningScope:
        old_variables = {new: old for old, new in variables.items()}
        old_scenarios = {new: old for old, new in scenarios.items()}
        return _PlanningScope(
            selections=tuple(
                _renamed_selection(s, old_variables, old_scenarios)
                for s in self.selections
            ),
            variables=None
            if self.variables is None
            else frozenset(old_variables.get(x, x) for x in self.variables),
            rename_variables=(
                frozenset(old_variables.get(x, x) for x in self.rename_variables)
                | frozenset(variables)
                | frozenset(old_variables)
            ),
            rename_scenarios=(
                frozenset(old_scenarios.get(x, x) for x in self.rename_scenarios)
                | frozenset(scenarios)
                | frozenset(old_scenarios)
            ),
        )


def _known_ids(
    requested: set[str] | None,
    available: set[str],
    *,
    dimension: str,
) -> None:
    if requested is not None and (missing := requested - available):
        raise ValueError(f"unknown requested {dimension} {sorted(missing)[0]!r}")


def _matches_descriptor(
    requested: str | Variable | Scenario | Period,
    actual: Variable | Scenario | Period,
) -> bool:
    if isinstance(requested, str):
        return requested == actual.id
    return requested.id == actual.id and scientific_identity(
        requested
    ) == scientific_identity(actual)


def _typed_choices(
    requested: tuple[str | Variable | Scenario | Period, ...] | None,
    available: list[Variable | Scenario | Period],
) -> None:
    for value in requested or ():
        if isinstance(value, str):
            continue
        candidates = [item for item in available if item.id == value.id]
        if candidates and not any(_matches_descriptor(value, v) for v in candidates):
            _typed_match((value,), candidates[0])


def _needed_inventory(
    planning: _PlanningScope, known: Mapping[str, _Entry]
) -> tuple[set[str], bool]:
    """Find selectors still waiting for a declaring member."""
    variables: set[str] = set()
    any_variable = False
    for selection in planning.selections:
        if not selection._has_scientific_filters():
            continue
        entries = [
            entry
            for name, entry in known.items()
            if not selection.members or name in selection.members
        ]
        for key in ("variables", "scenarios", "periods"):
            available = [
                descriptor
                for entry in entries
                for descriptor in getattr(entry.resource.metadata.spec, key)
            ]
            for value in getattr(selection, key) or ():
                if not any(_matches_descriptor(value, actual) for actual in available):
                    if key == "variables":
                        variables.add(value if isinstance(value, str) else value.id)
                    else:
                        any_variable = True
        for ref in selection.series or ():
            if not any(
                entry.matches(item, ref, check_science=False)
                for entry in entries
                for item in entry.resource.metadata.spec.series
            ):
                variables.add(ref.variable.id)
    return variables, any_variable


def _project_entries(
    entries: Mapping[str, _Entry],
    selection: CollectionSelection,
    *,
    discovery_incomplete: bool,
    projected_variables: frozenset[str] | None,
) -> dict[str, _Entry]:
    variables = _requested_ids(selection.variables)
    scenarios = _requested_ids(selection.scenarios)
    periods = _requested_ids(selection.periods)
    specs = [entry.resource.metadata.spec for entry in entries.values()]
    _typed_choices(selection.variables, [v for spec in specs for v in spec.variables])
    _typed_choices(selection.scenarios, [s for spec in specs for s in spec.scenarios])
    _typed_choices(selection.periods, [p for spec in specs for p in spec.periods])
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
    conflicting_references: dict[int, ValueError] = {}
    series_keys = {_series_key(item) for spec in specs for item in spec.series}
    projected = {}
    for resource_id, entry in entries.items():
        original = entry.resource.metadata
        selected = []
        for item in original.spec.series:
            matches: set[int] = set()
            if references is not None:
                matches = {
                    index
                    for index, reference in enumerate(references)
                    if entry.matches(item, reference, check_science=False)
                }
                for index in matches:
                    try:
                        entry.matches(item, references[index], check_science=True)
                    except ValueError as error:
                        conflicting_references[index] = error
                    else:
                        matching_references.add(index)
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
            if (
                projected_variables is not None
                and item.variable.id not in projected_variables
            ):
                continue
            if references is not None:
                for index in matches:
                    entry.matches(item, references[index], check_science=True)
            _typed_match(selection.variables, item.variable)
            _typed_match(selection.scenarios, item.scenario)
            _typed_match(selection.periods, item.period)
            selected.append(item)
        if not selected:
            if not original.spec.series and all(
                value is None
                for value in (
                    variables,
                    scenarios,
                    periods,
                    references,
                    projected_variables,
                )
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
        projected[resource_id] = replace(
            entry,
            resource=replace(entry.resource, metadata=metadata, path=None, sha256=None),
            node=node,
        )
    if references is not None:
        for index, reference in enumerate(references):
            if index in matching_references:
                continue
            if index in conflicting_references:
                raise conflicting_references[index]
            if discovery_incomplete:
                continue
            if _series_key(reference) not in series_keys:
                raise ValueError(f"unknown series identity {_series_key(reference)!r}")
            if reference.model is not None or reference.member is not None:
                raise ValueError(
                    "selected series release, model or member differs "
                    "from the source snapshot"
                )
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

    def verified_path(self, scope: ExecutionScope) -> Path:
        assert self.resource.path is not None
        path = self.directory / self.resource.path
        if self.directory.is_symlink() or path.parent.is_symlink() or path.is_symlink():
            raise _invalid(ValueError("collection resources must be owned local files"))
        pinned, identity = scope.pin_local(path)
        if identity.sha256 != self.resource.sha256:
            raise OSError(
                f"collection resource checksum mismatch for {self.resource.id!r}"
            )
        return pinned

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        metadata = self.resource.metadata

        def batches() -> Iterator[pa.RecordBatch]:
            pinned = self.verified_path(scope)
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
    def __init__(
        self,
        recipe: DatasetCollection,
        *,
        projected_variables: frozenset[str] | None = None,
    ) -> None:
        self.recipe = recipe
        self.projected_variables = projected_variables
        self.key = (
            recipe._key
            if projected_variables is None
            else _key("collection-projection", recipe._key, sorted(projected_variables))
        )

    def open(self, scope: ExecutionScope) -> NodeStream[CollectionMetadata]:
        plan = self.recipe._plan(scope, projected_variables=self.projected_variables)
        metadata = plan.metadata()
        started: list[_Entry] = []
        produced = 0
        no_data = 0
        completed: CollectionMetadata | None = None

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal produced, no_data, completed
            for member in metadata.members:
                member_check = (
                    _Checked(
                        scope, plan.entries[member.resource_ids[0]].resource.metadata
                    )
                    if len(member.resource_ids) > 1
                    else None
                )
                for resource_id in member.resource_ids:
                    entry = plan.entries[resource_id]
                    started.append(entry)
                    verified = entry.payload_pin is None
                    for batch in scope.iter_batches(entry.node):
                        if not verified:
                            entry.verify_payload(scope)
                            verified = True
                        if member_check is not None:
                            # The resource streams already validate their own models.
                            try:
                                for row in batch.select(
                                    (*_ROW_KEY, "geometry")
                                ).to_pylist():
                                    member_check._remember(
                                        _dump(tuple(row[key] for key in _ROW_KEY)),
                                        row["support_id"],
                                        row["geometry"],
                                        row["cell_id"],
                                    )
                            except (ValueError, TypeError) as error:
                                raise _invalid(error) from error
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
                    if not verified:
                        entry.verify_payload(scope)
                    child = scope.report(entry.node)
                    if child.valid is False:
                        if child.diagnostics:
                            raise DataValidationError(child.diagnostics)
                        raise _invalid(
                            ValueError(
                                f"collection resource {resource_id!r} is invalid"
                            )
                        )
            # A full read verifies the physical bundle independently of the
            # series it decodes. Participating resources reuse their pinned bytes.
            for resource in plan.bundle_resources:
                resource.verified_path(scope)
            with native_data(scope=scope, source="collection"):
                completed = replace(
                    final_metadata(),
                    report=replace(report(), complete=True, valid=True),
                )

        def report() -> ExecutionReport:
            combined = _collection_report(scope.report(entry.node) for entry in started)
            return replace(
                combined,
                counts=replace(combined.counts, produced=produced, no_data=no_data),
            )

        def final_metadata() -> CollectionMetadata:
            if completed is not None:
                return completed
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
            return _Plan(
                manifest.members,
                self._collected_entries,
                manifest.selection,
                source=manifest.source,
            )
        assert self._path is not None
        resources = tuple(
            _BundleResourceNode(self._path, resource, manifest.selection.area)
            for resource in manifest.resources
        )
        entries = {node.resource.id: _Entry(node.resource, node) for node in resources}
        return _Plan(
            manifest.members,
            entries,
            manifest.selection,
            source=manifest.source,
            bundle_resources=resources,
        )

    def _memory_inputs(
        self,
        scope: ExecutionScope,
        inspect: bool,
        projected_variables: frozenset[str] | None,
        planning: _PlanningScope,
    ) -> _Plan:
        ids = tuple(name for name, _ in self._inputs)
        requested_members = _requested_ids(self._selection.members)
        _known_ids(requested_members, set(ids), dimension="member")
        scientific_filters = self._selection._has_scientific_filters()
        names = (
            set(ids)
            if requested_members is None
            or (not requested_members and scientific_filters)
            else requested_members
        )
        known: dict[str, _Entry] = {}
        requirements: list[Requirement] = []
        discovered: dict[tuple[int, str], Inspection] = {}
        active_members = planning.members
        active_variables = planning.active_variables
        source_state = planning.source_state

        def plan_member(
            name: str, dataset: Dataset | CanonicalReader, *, metadata_only: bool
        ) -> None:
            if isinstance(dataset, Dataset):
                node = _bounded_node(dataset._node, source_state, discover=True)
                if inspect or metadata_only:
                    identity = (id(dataset), node.key)
                    if identity not in discovered:
                        temporary = Dataset._create(node, operations=())
                        discovered[identity] = temporary._inspect(scope)
                    details = discovered[identity]
                    if not metadata_only:
                        requirements.extend(details.requirements)
                    metadata = details.metadata
                    if not isinstance(metadata, DatasetMetadata):
                        return
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
                    metadata.source.model,
                    _release_key(metadata.source.release),
                    [resource.sha256 for resource in metadata.source.resources],
                ),
                [p._document() for p in metadata.provenance],
            )
            known[name] = _Entry(
                CollectionResource(id=resource_id, metadata=metadata),
                node,
                _pending_payload_release(cast(Node[object], node)),
            )

        deferred = []
        for name, dataset in self._inputs:
            if name not in names:
                continue
            declared = (
                _declared_variable_ids(cast(Node[object], dataset._node))
                if isinstance(dataset, Dataset)
                else None
            )
            if (
                active_members is not None
                and name not in active_members
                or active_variables is not None
                and (
                    not active_variables
                    or declared is not None
                    and declared.isdisjoint(active_variables)
                )
            ):
                deferred.append((name, dataset, declared))
            else:
                plan_member(name, dataset, metadata_only=False)
        # An excluded member may declare a selector used independently of the
        # active series. Discover it only if the active metadata cannot do so.
        for name, dataset, declared in deferred:
            missing, any_variable = _needed_inventory(planning, known)
            needed = missing | planning.rename_variables
            if (
                any_variable
                or planning.rename_scenarios
                or needed
                and (declared is None or not declared.isdisjoint(needed))
            ):
                plan_member(name, dataset, metadata_only=True)

        members = tuple(
            CollectionMember(
                id=name,
                resource_ids=() if name not in known else (known[name].resource.id,),
            )
            for name in ids
            if name in names
        )
        entries = {
            known[name].resource.id: known[name] for name in ids if name in known
        }
        return self._participating(
            members,
            entries,
            CollectionSelection(),
            tuple(requirements),
            projected_variables=projected_variables,
        )

    def _participating(
        self,
        members: tuple[CollectionMember, ...],
        entries: Mapping[str, _Entry],
        previous: CollectionSelection,
        requirements: tuple[Requirement, ...],
        source: SourceIdentity | None = None,
        bundle_resources: tuple[_BundleResourceNode, ...] = (),
        projected_variables: frozenset[str] | None = None,
    ) -> _Plan:
        allowed = _requested_ids(self._selection.members)
        _known_ids(allowed, {member.id for member in members}, dimension="member")
        inventory = members
        members = tuple(
            member for member in members if allowed is None or member.id in allowed
        )
        if self._selection.members != ():
            inventory = members
        used = {key for member in inventory for key in member.resource_ids}
        empty = self._selection._empty()
        selected = (
            {}
            if empty and not self._selection._has_scientific_filters()
            else _project_entries(
                {key: value for key, value in entries.items() if key in used},
                self._selection,
                discovery_incomplete=bool(requirements),
                projected_variables=frozenset() if empty else projected_variables,
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
        return _Plan(
            members,
            selected,
            state,
            tuple(dict.fromkeys(requirements)),
            source,
            bundle_resources,
        )

    def _plan(
        self,
        scope: ExecutionScope,
        *,
        inspect: bool = False,
        projected_variables: frozenset[str] | None = None,
        downstream: _PlanningScope = _PlanningScope(),
    ) -> _Plan:
        planning = replace(
            downstream,
            selections=(self._selection, *downstream.selections),
            variables=downstream.variables
            if projected_variables is None
            else projected_variables,
        )
        if self._path is not None or self._collected_entries is not None:
            base = self._manifest_inputs()
        elif self._parent is not None:
            base = self._parent._plan(
                scope,
                inspect=inspect,
                downstream=planning.before_rename(self._variables, self._scenarios),
            )
            # Validate injectivity over all participating identities. Unused scientific
            # descriptors need not reconcile until after downstream selection.
            specs = tuple(
                entry.resource.metadata.spec for entry in base.entries.values()
            )
            _rename_mapping(
                self._variables,
                sorted({v.id for spec in specs for v in spec.variables}),
                "variable",
            )
            _rename_mapping(
                self._scenarios,
                sorted({s.id for spec in specs for s in spec.scenarios}),
                "scenario",
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
                entries[resource_id] = replace(
                    entry,
                    resource=replace(
                        entry.resource, metadata=metadata, path=None, sha256=None
                    ),
                    node=node,
                )
            return self._participating(
                base.members,
                entries,
                _renamed_selection(base.selection, self._variables, self._scenarios),
                base.requirements,
                base.source,
                base.bundle_resources,
                projected_variables,
            )
        else:
            return self._memory_inputs(scope, inspect, projected_variables, planning)
        return self._participating(
            base.members,
            base.entries,
            base.selection,
            base.requirements,
            base.source,
            base.bundle_resources,
            projected_variables,
        )

    def _inspect(
        self,
        scope: ExecutionScope,
        *,
        projected_variables: frozenset[str] | None = None,
    ) -> Inspection:
        plan = self._plan(scope, inspect=True, projected_variables=projected_variables)
        metadata = plan.metadata()
        references = tuple(
            SeriesReference(
                series.variable,
                series.scenario,
                series.period,
                release=None
                if resource.metadata.source is None
                else resource.metadata.source.release,
                probability=series.probability,
                model=None
                if resource.metadata.source is None
                else resource.metadata.source.model,
                member=None
                if resource.metadata.source is None
                else resource.metadata.source.member,
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
        return self._explain(config, _PlanningScope())

    def _explain(self, config: Runtime, downstream: _PlanningScope) -> Explanation:
        operations: list[Operation] = []
        requirements: list[Requirement] = []
        planning = replace(
            downstream, selections=(self._selection, *downstream.selections)
        )
        names = planning.members
        active_variables = planning.active_variables
        source_state = planning.source_state
        children: list[Explanation] = []
        for name, dataset in self._inputs:
            if isinstance(dataset, Dataset) and (names is None or name in names):
                if active_variables is not None:
                    if not active_variables:
                        continue
                    declared = _declared_variable_ids(cast(Node[object], dataset._node))
                    if declared is not None and declared.isdisjoint(active_variables):
                        continue
                candidate = Dataset._create(
                    _bounded_node(dataset._node, source_state, discover=True),
                    operations=dataset._operations,
                )
                children.append(candidate.explain(runtime=config))
        if self._parent is not None:
            children.append(
                self._parent._explain(
                    config, planning.before_rename(self._variables, self._scenarios)
                )
            )
        for details in children:
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
        self, assets: crc_sdk.Assets, *, spatial: crc_sdk.SpatialMatch | None = None
    ) -> crc_sdk.Evaluation:
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
    ) -> Execution[CollectionMetadata, DatasetCollection]:
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


class _CollectionExecution(Execution[CollectionMetadata, DatasetCollection]):
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
