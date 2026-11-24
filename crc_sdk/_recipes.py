"""Immutable dataset and sample recipes backed by one execution graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import types
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Generic, Literal, Self, TypeAlias, TypeVar, cast

import crc_framework
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from shapely import from_wkb  # type: ignore[import-untyped]

import crc_sdk

from ._aggregation import Sampling, _annual_sampling
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    FittingDiagnostic,
    ObservationTimeSelection,
    ReleaseReference,
    SourceIdentity,
    Stage,
    UnavailableStationYear,
)
from ._declaration_updates import refine_spec
from ._declarations import (
    DataSpec,
    Period,
    Probability,
    Scenario,
    Variable,
    scientific_identity,
)
from ._errors import CapabilityError, ValidationError
from ._evidence import (
    EvidenceKind,
    EvidenceMetadata,
    FitFailure,
    _SampleFitError,
    evidence_row_key,
    fit_sample_table,
    metadata_from_evidence_table,
    validate_evidence,
)
from ._graph import (
    DataValidationError,
    ExecutionScope,
    Node,
    NodeStream,
)
from ._inspection import (
    Bounds,
    Capability,
    Explanation,
    Inspection,
    Operation,
    Requirement,
    SeriesReference,
)
from ._runtime import LocalPath, Runtime, _local_path, _memory_bytes
from ._schema import (
    DatasetRow,
    dataset_arrow_schema,
    metadata_from_table,
    rows_to_table,
    table_to_rows,
)
from ._selection import (
    PeriodSelector,
    ResolvedSelection,
    ScenarioSelector,
    SelectionState,
    SeriesSelector,
    StationSelector,
    Unit,
    VariableSelector,
    conversion_metadata,
    conversion_row,
    rename_metadata,
    rename_row,
    selected_metadata,
)
from ._source_io import SourceNode
from ._table_input import (
    QuantileColumns,
    TableInput,
    TimeSeriesColumns,
    normalize_quantiles,
    normalize_samples,
)
from ._terminals import (
    BatchStream,
    Execution,
    QuantileFitSpec,
    ReadResult,
    SampleFitSpec,
    _publication_path,
    _schema,
    _with_report,
)

Metadata: TypeAlias = DatasetMetadata | EvidenceMetadata
MetadataT = TypeVar("MetadataT", bound=Metadata)
CanonicalT = TypeVar("CanonicalT")


def _key(operation: str, *values: object) -> str:
    return (
        operation
        + ":sha256:"
        + hashlib.sha256(
            json.dumps(
                values,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )


def _spec_document(spec: DataSpec) -> list[dict[str, object]]:
    return sorted(
        [
            {
                "identity": [
                    item.variable.id,
                    None if item.scenario is None else item.scenario.id,
                    None if item.period is None else item.period.id,
                ],
                "science": scientific_identity(item),
            }
            for item in spec.series
        ],
        key=lambda item: json.dumps(item, sort_keys=True),
    )


def _table_key(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    values = table.replace_schema_metadata(None).combine_chunks()
    with pa.ipc.new_stream(sink, values.schema) as writer:
        writer.write_table(values)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def _invalid(
    error: ValueError | TypeError, *, stage: Stage = Stage.VALIDATION
) -> DataValidationError:
    return DataValidationError(
        (
            ExecutionDiagnostic(
                "invalid_scientific_data",
                crc_framework.DiagnosticSeverity.ERROR,
                stage,
                str(error),
            ),
        )
    )


def _fit_invalid(
    error: crc_framework.FitError,
    *,
    source: str | None = None,
    support_id: str | None = None,
    variable_id: str | None = None,
) -> DataValidationError:
    return DataValidationError(
        (
            FittingDiagnostic(
                error.code,
                crc_framework.DiagnosticSeverity.ERROR,
                Stage.FITTING,
                error.message
                if variable_id is None
                else f"variable {variable_id!r}: {error.message}",
                source=source,
                support_id=support_id,
                record=error.record,
            ),
        )
    )


class CanonicalReader(ABC):
    """A stable canonical source, explicitly distinct from a lazy recipe.

    Implementations own the bytes they inspect and iterate. Recipe classes do
    not inherit this contract and cannot qualify through method names alone.
    """

    @abstractmethod
    def inspect(self) -> DatasetMetadata:
        raise NotImplementedError

    @abstractmethod
    def batches(
        self, *, batch_rows: int = 65536
    ) -> AbstractContextManager[Iterator[pa.RecordBatch]]:
        raise NotImplementedError


class _Checked:
    """Incremental global invariants, with identity indexes owned on disk.

    `payload_verified` streams carry rows whose bytes a checksum pins to an
    artifact validated when it was written; their numerical objects are not
    decoded again, while identities, supports and counts are still checked.
    """

    def __init__(
        self,
        scope: ExecutionScope,
        metadata: DatasetMetadata | EvidenceMetadata,
        *,
        payload_verified: bool = False,
    ) -> None:
        self.scope = scope
        self.metadata = metadata
        self.payload_verified = payload_verified
        self.produced = 0
        self.no_data = 0
        descriptor, filename = tempfile.mkstemp(
            prefix="identity-", suffix=".sqlite", dir=scope.directory
        )
        import os

        os.close(descriptor)
        self.db = sqlite3.connect(filename)
        # A scratch index for one execution: durability is not needed.
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA cache_size=-65536")
        self.db.execute("CREATE TABLE row_keys (identity TEXT PRIMARY KEY)")
        self.db.execute(
            "CREATE TABLE supports (id TEXT PRIMARY KEY, geometry BLOB, cell TEXT)"
        )
        scope.callback(self.db.close)

    def _remember(
        self,
        identities: list[str],
        supports: list[tuple[str, bytes | None, object]],
    ) -> None:
        try:
            with self.db:
                self.db.executemany(
                    "INSERT INTO row_keys VALUES (?)", ((i,) for i in identities)
                )
        except sqlite3.IntegrityError:
            seen: set[str] = set()
            for identity in identities:
                if (
                    identity in seen
                    or self.db.execute(
                        "SELECT 1 FROM row_keys WHERE identity=?", (identity,)
                    ).fetchone()
                ):
                    raise ValueError(
                        f"duplicate scientific row key: {identity}"
                    ) from None
                seen.add(identity)
            raise
        batch: dict[str, tuple[bytes | None, str | None]] = {}
        for support_id, geometry, cell_id in supports:
            value = (geometry, str(cell_id) if geometry is None else None)
            if batch.setdefault(support_id, value) != value:
                self._conflict(support_id, batch[support_id], value)
        for support_id, value in batch.items():
            found = self.db.execute(
                "SELECT geometry,cell FROM supports WHERE id=?", (support_id,)
            ).fetchone()
            if found is None:
                self.db.execute(
                    "INSERT INTO supports VALUES (?,?,?)", (support_id, *value)
                )
            elif found != value:
                self._conflict(support_id, found, value)

    @staticmethod
    def _conflict(
        support_id: str,
        old: tuple[bytes | None, str | None],
        new: tuple[bytes | None, str | None],
    ) -> None:
        (old_geometry, old_cell), (geometry, cell) = old, new
        if (
            old_cell != cell
            or old_geometry is None
            or geometry is None
            or not from_wkb(old_geometry).equals(from_wkb(geometry))
        ):
            raise ValueError(
                f"support_id {support_id!r} identifies conflicting source geometry"
            )

    def check(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        planned = replace(
            self.metadata,
            report=replace(
                self.metadata.report,
                complete=False,
                valid=False if self.metadata.report.valid is False else None,
            ),
        )
        table = pa.Table.from_batches([batch]).replace_schema_metadata(
            _schema(planned).metadata
        )
        try:
            if isinstance(planned, DatasetMetadata):
                if not self.payload_verified:
                    table_to_rows(table)
                columns = {
                    name: table.column(name).to_pylist()
                    for name in (
                        "variable_id",
                        "scenario_id",
                        "period_id",
                        "support_id",
                        "cell_id",
                        "geometry",
                    )
                }
                identities = [
                    json.dumps(list(key), ensure_ascii=False)
                    for key in zip(
                        columns["variable_id"],
                        columns["scenario_id"],
                        columns["period_id"],
                        columns["support_id"],
                        columns["cell_id"],
                    )
                ]
                supports = list(
                    zip(columns["support_id"], columns["geometry"], columns["cell_id"])
                )
            else:
                validate_evidence(table, planned, check_counts=False)
                records = table.to_pylist()
                identities = [
                    json.dumps(evidence_row_key(record, planned), ensure_ascii=False)
                    for record in records
                ]
                supports = [
                    (record["support_id"], record["geometry"], record["cell_id"])
                    for record in records
                ]
            self._remember(identities, supports)
        except (ValueError, TypeError) as error:
            raise _invalid(error) from error
        if "failure" in table.column_names:
            failures = table.column("failure")
            self.no_data += len(failures) - failures.null_count
        self.produced += batch.num_rows
        self.scope.emit(Stage.VALIDATION, self.produced)
        return batch.replace_schema_metadata(_schema(planned).metadata)

    def report(self) -> ExecutionReport:
        return ExecutionReport(
            False,
            None,
            counts=ExecutionCounts(
                requested=self.produced, produced=self.produced, no_data=self.no_data
            ),
        )


def _checked_stream(
    scope: ExecutionScope,
    metadata: MetadataT,
    batches: Iterator[pa.RecordBatch],
    *,
    payload_verified: bool = False,
) -> NodeStream[MetadataT]:
    state = _Checked(scope, metadata, payload_verified=payload_verified)

    def checked() -> Iterator[pa.RecordBatch]:
        for batch in batches:
            yield state.check(batch)

    return NodeStream(
        _schema(metadata), metadata, checked(), report=state.report, checked=True
    )


def _sources(metadata: Metadata) -> tuple[SourceIdentity, ...]:
    values = metadata.report.source_identities
    if metadata.source is not None and metadata.source not in values:
        values = (*values, metadata.source)
    return values


def _report_view(
    shared: ExecutionReport,
    *,
    parent: ExecutionReport | None = None,
    baseline: ExecutionReport | None = None,
    source_known: bool = False,
    source: SourceIdentity | None = None,
) -> ExecutionReport:
    inherited = ExecutionReport(True, True) if parent is None else parent
    base = ExecutionReport(True, True) if baseline is None else baseline
    diagnostics = list((*inherited.diagnostics, *base.diagnostics))
    for diagnostic in shared.diagnostics:
        if source_known and diagnostic.source is not None:
            diagnostic = replace(
                diagnostic, source=None if source is None else source.product
            )
        if diagnostic not in diagnostics:
            diagnostics.append(diagnostic)
    if (
        baseline is not None
        and shared.complete
        and baseline.complete
        and (
            baseline.counts.produced != shared.counts.produced
            or baseline.counts.no_data != shared.counts.no_data
        )
    ):
        diagnostics.append(
            ExecutionDiagnostic(
                "stored_count_mismatch",
                crc_framework.DiagnosticSeverity.ERROR,
                Stage.VALIDATION,
                "Stored complete report counts disagree with the consumed rows",
            )
        )
    if base.valid is False and not any(
        d.severity is crc_framework.DiagnosticSeverity.ERROR for d in diagnostics
    ):
        diagnostics.append(
            ExecutionDiagnostic(
                "source_declared_invalid",
                crc_framework.DiagnosticSeverity.ERROR,
                Stage.VALIDATION,
                "The source report declares this scientific data invalid",
            )
        )
    complete = (
        shared.complete
        and inherited.complete
        and (baseline is None or baseline.complete)
    )
    invalid = (
        shared.valid is False
        or inherited.valid is False
        or base.valid is False
        or any(
            d.severity is crc_framework.DiagnosticSeverity.ERROR for d in diagnostics
        )
    )
    return ExecutionReport(
        complete,
        False if invalid else True if complete else None,
        tuple(
            diagnostic
            for index, diagnostic in enumerate(diagnostics)
            if diagnostic not in diagnostics[:index]
        ),
        shared.counts
        if baseline is None
        else replace(
            baseline.counts,
            produced=shared.counts.produced,
            no_data=shared.counts.no_data,
        ),
        inherited.source_identities if baseline is None else baseline.source_identities,
    )


def _static_source(
    scope: ExecutionScope, node: Node[MetadataT]
) -> tuple[bool, SourceIdentity | None]:
    if isinstance(node, _TableNode):
        return True, node.metadata_value.source
    if isinstance(node, _ParquetNode):
        metadata = node.metadata_value or scope.computation_metadata(node)
        return (False, None) if metadata is None else (True, metadata.source)
    if isinstance(node, (_SelectionNode, _TransformNode, _FitNode)):
        return _static_source(scope, node.parent)
    return False, None


class _ScientificNode(Node[MetadataT], Generic[MetadataT]):
    def _view_batch(
        self, scope: ExecutionScope, batch: pa.RecordBatch
    ) -> pa.RecordBatch:
        metadata = scope.metadata(self)
        report = scope.report(self)
        metadata = replace(
            metadata,
            report=replace(
                report, complete=False, valid=False if report.valid is False else None
            ),
        )
        return batch.replace_schema_metadata(_schema(metadata).metadata)


class _TableNode(_ScientificNode[MetadataT]):
    def __init__(self, table: pa.Table, metadata: MetadataT) -> None:
        self.table = table
        self.metadata_value: MetadataT = metadata
        self.key = _key(
            "table",
            _table_key(table),
            _spec_document(metadata.spec),
            None if isinstance(metadata, DatasetMetadata) else metadata.kind.value,
            None
            if isinstance(metadata, DatasetMetadata) or metadata.sampling is None
            else metadata.sampling._document(),
            metadata.spatial._document(),
            [
                step.time._document()
                for step in metadata.provenance
                if isinstance(step, ObservationTimeSelection)
            ],
            [
                (d.station_id, d.year)
                for d in metadata.report.diagnostics
                if isinstance(d, UnavailableStationYear)
            ],
            []
            if isinstance(metadata, DatasetMetadata)
            else [support._document() for support in metadata._empty_supports],
        )

    def _view_metadata(
        self, scope: ExecutionScope, replay_metadata: MetadataT
    ) -> MetadataT:
        return replace(self.metadata_value, report=scope.report(self))

    def _view_report(
        self, scope: ExecutionScope, shared: ExecutionReport
    ) -> ExecutionReport:
        baseline = replace(
            self.metadata_value.report, source_identities=_sources(self.metadata_value)
        )
        report = _report_view(shared, baseline=baseline)
        return replace(
            report,
            counts=replace(
                baseline.counts,
                requested=max(shared.counts.produced, baseline.counts.requested),
                produced=shared.counts.produced,
                no_data=shared.counts.no_data,
            ),
        )

    def open(self, scope: ExecutionScope) -> NodeStream[MetadataT]:
        return _checked_stream(
            scope,
            self.metadata_value,
            iter(self.table.to_batches(max_chunksize=scope.runtime.batch_rows)),
        )


class _ParquetNode(_ScientificNode[MetadataT]):
    def __init__(
        self,
        path: Path,
        metadata_type: type[MetadataT],
        *,
        digest: str | None = None,
        metadata: MetadataT | None = None,
        expected_kind: EvidenceKind = EvidenceKind.SAMPLES,
    ) -> None:
        self.expected_kind = expected_kind
        self.metadata_value: MetadataT | None = metadata
        self.path = path
        self.metadata_type: type[MetadataT] = metadata_type
        self.digest = digest
        self.key = _key(
            "parquet",
            str(path.absolute()),
            metadata_type.__name__,
            digest,
            expected_kind.value,
        )

    def _view_metadata(
        self, scope: ExecutionScope, replay_metadata: MetadataT
    ) -> MetadataT:
        return replace(
            self.metadata_value or replay_metadata, report=scope.report(self)
        )

    def _view_report(
        self, scope: ExecutionScope, shared: ExecutionReport
    ) -> ExecutionReport:
        baseline = self.metadata_value or scope.computation_metadata(self)
        if baseline is None:
            return shared
        report = replace(baseline.report, source_identities=_sources(baseline))
        result = _report_view(shared, baseline=report)
        # The original FD pin is an execution resource, separate from any
        # upstream source identity already present in artifact metadata.
        return replace(
            result,
            source_identities=tuple(
                dict.fromkeys((*result.source_identities, *shared.source_identities))
            ),
        )

    def open(self, scope: ExecutionScope) -> NodeStream[MetadataT]:
        handle = None
        discovery = None
        if self.metadata_value is None:
            handle = scope.enter_context(self.path.open("rb"))
            try:
                discovery = scope.enter_context(pq.ParquetFile(handle))
                schema = discovery.schema_arrow
                empty = pa.Table.from_batches([], schema=schema)
                metadata = cast(
                    MetadataT,
                    metadata_from_table(empty)
                    if self.metadata_type is DatasetMetadata
                    else metadata_from_evidence_table(empty),
                )
                if not schema.equals(_schema(metadata), check_metadata=False):
                    raise ValueError(
                        "Persisted Arrow schema contradicts its declared artifact kind"
                    )
                if (
                    isinstance(metadata, EvidenceMetadata)
                    and metadata.kind is not self.expected_kind
                ):
                    raise ValueError(
                        f"This reader requires {self.expected_kind.value} evidence"
                    )
                if (
                    metadata.report.complete
                    and discovery.metadata.num_rows != metadata.report.counts.produced
                ):
                    raise ValueError(
                        "Stored complete report count disagrees with the Parquet footer"
                    )
            except (ValueError, TypeError) as error:
                raise _invalid(error) from error
        else:
            metadata = self.metadata_value
            schema = _schema(metadata)
        state = _Checked(scope, metadata)
        resource_source: SourceIdentity | None = None

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal resource_source
            if handle is None:
                pinned, resource = scope.pin_local(
                    self.path, expected_sha256=self.digest
                )
            else:
                pinned, resource = scope.pin_handle(
                    handle, uri=str(self.path), expected_sha256=self.digest
                )
            reader = scope.enter_context(pq.ParquetFile(pinned))
            if (
                not reader.schema_arrow.equals(schema, check_metadata=False)
                or reader.schema_arrow.metadata != schema.metadata
                or (
                    discovery is not None
                    and reader.metadata.num_rows != discovery.metadata.num_rows
                )
            ):
                raise OSError(
                    "The artifact changed after metadata discovery; no data was "
                    "delivered under the earlier inspection"
                )
            resource_source = SourceIdentity(
                provider="local",
                product=self.metadata_type.__name__,
                resources=(resource,),
            )
            if reader.num_row_groups == 0:
                return
            for batch in reader.iter_batches(batch_size=scope.runtime.batch_rows):
                yield state.check(batch)

        def report() -> ExecutionReport:
            return replace(
                state.report(),
                source_identities=() if resource_source is None else (resource_source,),
            )

        return NodeStream(schema, metadata, batches(), report=report)


def _select_unavailable_years(
    report: ExecutionReport, selected: ResolvedSelection
) -> ExecutionReport:
    def retain(diagnostic: ExecutionDiagnostic) -> bool:
        if not isinstance(diagnostic, UnavailableStationYear):
            return True
        if selected.empty or (
            selected.stations is not None
            and diagnostic.station_id not in selected.stations
        ):
            return False
        if selected.time is not None:
            start, stop = selected.time.start, selected.time.end
            assert start is not None and stop is not None
            return start <= diagnostic.year <= stop
        return True

    diagnostics = tuple(d for d in report.diagnostics if retain(d))
    removed = len(report.diagnostics) - len(diagnostics)
    if not removed:
        return report
    return replace(
        report,
        diagnostics=diagnostics,
        counts=replace(
            report.counts,
            requested=max(report.counts.produced, report.counts.requested - removed),
            incomplete_years=max(0, report.counts.incomplete_years - removed),
        ),
    )


class _SelectionNode(_ScientificNode[MetadataT]):
    def __init__(self, parent: Node[MetadataT], state: SelectionState) -> None:
        self.parent: Node[MetadataT] = parent
        self.state = state
        self.key = _key("select", parent.key, state.document())

    def _view_metadata(
        self, scope: ExecutionScope, replay_metadata: MetadataT
    ) -> MetadataT:
        original = scope.metadata(self.parent)
        metadata = selected_metadata(
            original, self.state.resolve(original.spec, source=original.source)
        )
        return replace(metadata, report=scope.report(self))

    def _view_report(
        self, scope: ExecutionScope, shared: ExecutionReport
    ) -> ExecutionReport:
        parent = scope.report(self.parent)
        metadata = scope.metadata(self.parent)
        selected = self.state.resolve(metadata.spec, source=metadata.source)
        parent = _select_unavailable_years(parent, selected)
        if _annual_samples(metadata) and shared.counts.no_data != parent.counts.no_data:
            diagnostics = tuple(
                d
                for d in parent.diagnostics
                if d.stage is not Stage.REDUCTION
                or d.code not in {"incomplete_year", "incomplete_years"}
            )
            if shared.counts.no_data:
                diagnostics = (
                    *diagnostics,
                    ExecutionDiagnostic(
                        "incomplete_years",
                        crc_framework.DiagnosticSeverity.WARNING,
                        Stage.REDUCTION,
                        f"{shared.counts.no_data} annual observations did not meet "
                        "the explicit completeness rule",
                        source=None
                        if metadata.source is None
                        else metadata.source.product,
                    ),
                )
            parent = replace(parent, diagnostics=diagnostics)
        if selected.empty:
            parent = replace(
                parent, complete=True, valid=False if parent.valid is False else True
            )
        return _report_view(shared, parent=parent)

    def open(self, scope: ExecutionScope) -> NodeStream[MetadataT]:
        original = scope.metadata(self.parent)
        resolved = self.state.resolve(original.spec, source=original.source)
        known_supports = _support_inventory(self.parent)
        if resolved.stations is not None and known_supports is not None:
            missing = set(resolved.stations) - known_supports
            if missing:
                raise ValueError(
                    f"unknown requested station identities {sorted(missing)!r}"
                )
        metadata = selected_metadata(original, resolved)
        metadata = _with_report(
            metadata,
            replace(
                metadata.report, complete=False, valid=None, counts=ExecutionCounts()
            ),
        )

        def batches() -> Iterator[pa.RecordBatch]:
            if not resolved.keys or resolved.stations == ():
                return
            seen_stations: set[str] = set()
            for batch in scope.iter_batches(self.parent):
                if resolved.stations is not None:
                    seen_stations.update(
                        str(record.get("station_id") or record["support_id"])
                        for record in batch.to_pylist()
                    )
                if isinstance(metadata, DatasetMetadata):
                    table = pa.Table.from_batches([batch]).replace_schema_metadata(
                        _schema(
                            _with_report(
                                original,
                                replace(
                                    original.report,
                                    complete=False,
                                    valid=False
                                    if original.report.valid is False
                                    else None,
                                ),
                            )
                        ).metadata
                    )
                    rows = [
                        row for row in table_to_rows(table) if resolved.matches(row)
                    ]
                    if rows:
                        yield from rows_to_table(rows, metadata).to_batches()
                else:
                    records = [
                        row
                        for row in batch.to_pylist()
                        if resolved.matches_record(
                            row,
                            sampling=original.sampling
                            if isinstance(original, EvidenceMetadata)
                            else None,
                        )
                    ]
                    if records:
                        yield pa.RecordBatch.from_pylist(
                            records, schema=_schema(metadata)
                        )
            if resolved.stations is not None:
                consumed = scope.metadata(self.parent)
                if isinstance(consumed, EvidenceMetadata):
                    seen_stations.update(s.id for s in consumed._empty_supports)
                seen_stations.update(
                    d.station_id
                    for d in scope.report(self.parent).diagnostics
                    if isinstance(d, UnavailableStationYear)
                )
                missing = set(resolved.stations) - seen_stations
                if missing:
                    raise ValueError(
                        f"unknown requested station identities {sorted(missing)!r}"
                    )

        stream = _checked_stream(scope, metadata, batches())
        own_report = stream.report
        annual = _annual_samples(metadata)

        def report() -> ExecutionReport:
            assert own_report is not None
            own = own_report()
            parent = _select_unavailable_years(scope.report(self.parent), resolved)
            unavailable = sum(
                isinstance(d, UnavailableStationYear) for d in parent.diagnostics
            )
            return replace(
                own,
                counts=replace(
                    own.counts,
                    requested=own.counts.requested + unavailable,
                    skipped=parent.counts.skipped,
                    usable_years=own.counts.produced - own.counts.no_data
                    if annual
                    else parent.counts.usable_years,
                    incomplete_years=own.counts.no_data + unavailable
                    if annual
                    else parent.counts.incomplete_years,
                ),
                source_identities=parent.source_identities,
            )

        stream.report = report
        return stream


def _annual_samples(metadata: Metadata) -> bool:
    return (
        isinstance(metadata, EvidenceMetadata)
        and metadata.kind is EvidenceKind.SAMPLES
        and _annual_sampling(metadata.sampling)
    )


def _support_inventory(node: Node[MetadataT]) -> set[str] | None:
    if isinstance(node, _TableNode):
        inventory = {
            str(record.get("station_id") or record["support_id"])
            for record in node.table.to_pylist()
        } | {
            d.station_id
            for d in node.metadata_value.report.diagnostics
            if isinstance(d, UnavailableStationYear)
        }
        if isinstance(node.metadata_value, EvidenceMetadata):
            inventory.update(s.id for s in node.metadata_value._empty_supports)
        return inventory
    if isinstance(node, _TransformNode):
        return _support_inventory(node.parent)
    return None


def _fit_kind(spec: crc_framework.FitSpec, *, samples: bool) -> None:
    expected = (
        (crc_framework.SampleFit,)
        if samples
        else (
            crc_framework.QuantileFit,
            crc_framework.CensoredQuantileFit,
            crc_framework.HurdleQuantileFit,
        )
    )
    candidates = (
        spec.candidates if isinstance(spec, crc_framework.ModelSelection) else (spec,)
    )
    if not all(isinstance(candidate, expected) for candidate in candidates):
        kind = "sample" if samples else "quantile"
        raise TypeError(f"{kind} evidence requires {kind} fitting specifications")


class _FitNode(_ScientificNode[DatasetMetadata], Generic[MetadataT]):
    def __init__(
        self,
        parent: Node[MetadataT],
        specification: crc_framework.FitSpec,
        failure: FitFailure,
    ) -> None:
        if not isinstance(specification, crc_framework.FitSpec):
            raise TypeError("spec must be a framework FitSpec")
        if not isinstance(failure, FitFailure):
            raise TypeError("on_failure must be FitFailure")
        self.parent: Node[MetadataT] = parent
        self.specification = specification
        self.failure = failure
        self.key = _key("fit", parent.key, specification.to_json(), failure.value)

    def selected(self, state: SelectionState) -> _FitNode[MetadataT]:
        return _FitNode(
            _select_node(self.parent, state), self.specification, self.failure
        )

    def _view_metadata(
        self, scope: ExecutionScope, replay_metadata: DatasetMetadata
    ) -> DatasetMetadata:
        parent = scope.metadata(self.parent)
        return DatasetMetadata(
            spec=parent.spec,
            spatial=parent.spatial,
            report=scope.report(self),
            source=parent.source,
            provenance=parent.provenance,
        )

    def _view_report(
        self, scope: ExecutionScope, shared: ExecutionReport
    ) -> ExecutionReport:
        known, source = _static_source(scope, self.parent)
        parent = scope.report(self.parent)
        own = tuple(d for d in shared.diagnostics if d not in parent.diagnostics)
        return _report_view(
            replace(shared, diagnostics=own),
            parent=parent,
            source_known=known,
            source=source,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        original = scope.metadata(self.parent)
        metadata = DatasetMetadata(
            spec=original.spec,
            spatial=original.spatial,
            report=ExecutionReport(False, None),
            source=original.source,
            provenance=original.provenance,
        )
        result_metadata = metadata

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal result_metadata
            parts: list[pa.RecordBatch] = []
            size = 0
            budget = _memory_bytes(scope.runtime.memory_limit)
            for batch in scope.iter_batches(self.parent):
                size += batch.nbytes
                if budget is not None and size > budget:
                    raise MemoryError(
                        "Selected fitting evidence exceeds Runtime(memory_limit=...). "
                        "Narrow select() or increase the limit."
                    )
                parts.append(batch)
            parent_report = scope.report(self.parent)
            actual = _with_report(scope.metadata(self.parent), parent_report)
            table = pa.Table.from_batches(
                parts, schema=_schema(actual)
            ).replace_schema_metadata(_schema(actual).metadata)
            unlocated_annual = (
                isinstance(actual, EvidenceMetadata)
                and actual.spec.series
                and any(
                    isinstance(d, UnavailableStationYear)
                    for d in parent_report.diagnostics
                )
            )
            if table.num_rows == 0 and not unlocated_annual:
                result_metadata = replace(
                    metadata,
                    report=ExecutionReport(
                        True,
                        True,
                        parent_report.diagnostics,
                        ExecutionCounts(
                            usable_years=parent_report.counts.usable_years,
                            incomplete_years=parent_report.counts.incomplete_years,
                        ),
                        parent_report.source_identities,
                    ),
                )
                return
            scope.emit(Stage.FITTING, 0)
            if isinstance(actual, EvidenceMetadata):
                try:
                    rows, result_metadata = fit_sample_table(
                        table, actual, self.specification, on_failure=self.failure
                    )
                except crc_framework.FitError as error:
                    result_metadata = replace(
                        metadata,
                        report=ExecutionReport(
                            False,
                            None,
                            error.diagnostics
                            if isinstance(error, _SampleFitError)
                            else parent_report.diagnostics,
                            ExecutionCounts(
                                requested=error.requested
                                if isinstance(error, _SampleFitError)
                                else 0,
                                usable_years=parent_report.counts.usable_years,
                                incomplete_years=parent_report.counts.incomplete_years,
                            ),
                            parent_report.source_identities,
                        ),
                    )
                    raise _fit_invalid(
                        error,
                        source=error.source
                        if isinstance(error, _SampleFitError)
                        else None
                        if actual.source is None
                        else actual.source.product,
                        support_id=error.support_id
                        if isinstance(error, _SampleFitError)
                        else None,
                        variable_id=error.variable_id
                        if isinstance(error, _SampleFitError)
                        else None,
                    ) from error
            else:
                rows, result_metadata = _fit_quantiles(
                    table, actual, self.specification, self.failure
                )
            scope.emit(Stage.FITTING, len(rows), len(rows))
            yield from rows_to_table(rows, result_metadata).to_batches(
                max_chunksize=scope.runtime.batch_rows
            )

        return NodeStream(
            dataset_arrow_schema(metadata),
            metadata,
            batches(),
            report=lambda: result_metadata.report,
            final_metadata=lambda: result_metadata,
        )


def _fit_quantiles(
    table: pa.Table,
    metadata: DatasetMetadata,
    spec: crc_framework.FitSpec,
    failure: FitFailure,
) -> tuple[tuple[DatasetRow, ...], DatasetMetadata]:
    from ._dataset_schema import RetainedFailure

    if not isinstance(
        spec,
        (
            crc_framework.QuantileFit,
            crc_framework.CensoredQuantileFit,
            crc_framework.HurdleQuantileFit,
            crc_framework.ModelSelection,
        ),
    ):
        raise TypeError("quantile evidence requires a quantile fitting specification")
    if isinstance(spec, crc_framework.ModelSelection) and any(
        isinstance(candidate, crc_framework.SampleFit) for candidate in spec.candidates
    ):
        raise TypeError("quantile model selection requires quantile fitting candidates")
    rows = table_to_rows(table)
    result: list[DatasetRow] = []
    diagnostics = list(metadata.report.diagnostics)
    skipped = no_data = 0
    for row in rows:
        if not isinstance(row.curve, crc_framework.QuantileCurve):
            raise CapabilityError(
                "fit() requires supplied quantile evidence. "
                "Existing distributions are already models."
            )
        try:
            fitted = crc_framework.fit_quantiles(row.curve, spec)
            result.append(replace(row, curve=fitted.distribution, fit=fitted.record))
        except crc_framework.FitError as error:
            if failure is FitFailure.RAISE:
                raise _fit_invalid(
                    error,
                    source=None if metadata.source is None else metadata.source.product,
                    support_id=row.support_id,
                    variable_id=row.variable_id,
                ) from error
            diagnostics.append(
                FittingDiagnostic(
                    error.code,
                    crc_framework.DiagnosticSeverity.WARNING,
                    Stage.FITTING,
                    f"variable {row.variable_id!r}: {error.message}",
                    source=None if metadata.source is None else metadata.source.product,
                    support_id=row.support_id,
                    record=error.record,
                )
            )
            if failure is FitFailure.SKIP:
                skipped += 1
            else:
                no_data += 1
                result.append(
                    replace(
                        row,
                        curve=None,
                        fit=None,
                        failure=RetainedFailure.from_error(error),
                    )
                )
    report = ExecutionReport(
        True,
        True,
        tuple(diagnostics),
        ExecutionCounts(
            requested=len(rows), produced=len(result), skipped=skipped, no_data=no_data
        ),
        metadata.report.source_identities,
    )
    return tuple(result), replace(metadata, report=report)


class _TransformNode(_ScientificNode[MetadataT]):
    def __init__(
        self, parent: Node[MetadataT], operation: str, arguments: Mapping[str, object]
    ) -> None:
        self.parent: Node[MetadataT] = parent
        self.operation = operation
        copied: dict[str, object] = {}
        key_arguments: dict[str, object] = {}
        for name, value in arguments.items():
            if isinstance(value, Mapping):
                value = dict(value)
                key_arguments[name] = value
                copied[name] = types.MappingProxyType(value)
            else:
                copied[name] = value
                if isinstance(value, Variable):
                    key_arguments[name] = {
                        "id": value.id,
                        "science": scientific_identity(value),
                    }
                elif isinstance(value, DataSpec):
                    key_arguments[name] = _spec_document(value)
                elif isinstance(value, (Scenario, Period, Probability)):
                    key_arguments[name] = {
                        "id": getattr(value, "id", None),
                        "science": scientific_identity(value),
                    }
                elif isinstance(value, Enum):
                    key_arguments[name] = value.value
                else:
                    key_arguments[name] = value
        self.arguments = types.MappingProxyType(copied)
        self.key = _key(operation, parent.key, key_arguments)

    def _metadata_view(self, original: MetadataT) -> MetadataT:
        if self.operation == "rename":
            return rename_metadata(
                original,
                variables=cast(Mapping[str, str] | None, self.arguments["variables"]),
                scenarios=cast(Mapping[str, str] | None, self.arguments["scenarios"]),
            )
        if self.operation == "declare":
            spec, _ = refine_spec(
                original.spec,
                variable=cast(Variable | None, self.arguments["variable"]),
                scenario=cast(Scenario | None, self.arguments["scenario"]),
                period=cast(Period | None, self.arguments["period"]),
                probability=cast(Probability | None, self.arguments["probability"]),
                spec=cast(DataSpec | None, self.arguments["spec"]),
            )
            return replace(original, spec=spec)
        return conversion_metadata(
            original,
            variable=cast(Variable | str, self.arguments["variable"]),
            unit=cast(Unit, self.arguments["unit"]),
        )[0]

    def _view_metadata(
        self, scope: ExecutionScope, replay_metadata: MetadataT
    ) -> MetadataT:
        return replace(
            self._metadata_view(scope.metadata(self.parent)), report=scope.report(self)
        )

    def _view_report(
        self, scope: ExecutionScope, shared: ExecutionReport
    ) -> ExecutionReport:
        return _report_view(shared, parent=scope.report(self.parent))

    def open(self, scope: ExecutionScope) -> NodeStream[MetadataT]:
        original = scope.metadata(self.parent)
        if self.operation == "declare":
            spec, identities = refine_spec(
                original.spec,
                variable=cast(Variable | None, self.arguments["variable"]),
                scenario=cast(Scenario | None, self.arguments["scenario"]),
                period=cast(Period | None, self.arguments["period"]),
                probability=cast(Probability | None, self.arguments["probability"]),
                spec=cast(DataSpec | None, self.arguments["spec"]),
            )
            metadata = replace(original, spec=spec)

            def transform(row: DatasetRow) -> DatasetRow:
                variable, scenario, period = identities[
                    (row.variable_id, row.scenario_id, row.period_id)
                ]
                return replace(
                    row, variable_id=variable, scenario_id=scenario, period_id=period
                )

            def record(value: dict[str, object]) -> dict[str, object]:
                variable, scenario, period = identities[
                    (
                        cast(str, value["variable_id"]),
                        cast(str | None, value["scenario_id"]),
                        cast(str | None, value["period_id"]),
                    )
                ]
                return {
                    **value,
                    "variable_id": variable,
                    "scenario_id": scenario,
                    "period_id": period,
                }
        elif self.operation == "rename":
            variables = cast(Mapping[str, str] | None, self.arguments["variables"])
            scenarios = cast(Mapping[str, str] | None, self.arguments["scenarios"])
            metadata = rename_metadata(
                original, variables=variables, scenarios=scenarios
            )

            def transform(row: DatasetRow) -> DatasetRow:
                return rename_row(row, variables=variables, scenarios=scenarios)

            def record(value: dict[str, object]) -> dict[str, object]:
                return {
                    **value,
                    "variable_id": (variables or {}).get(
                        cast(str, value["variable_id"]), value["variable_id"]
                    ),
                    "scenario_id": (scenarios or {}).get(
                        cast(str, value["scenario_id"]), value["scenario_id"]
                    ),
                }
        else:
            requested_variable = cast(Variable | str, self.arguments["variable"])
            variable = (
                requested_variable.id
                if isinstance(requested_variable, Variable)
                else requested_variable
            )
            unit = cast(Unit, self.arguments["unit"])
            metadata, conversion = conversion_metadata(
                original, variable=requested_variable, unit=unit
            )
            before = next(
                v.measurement for v in original.spec.variables if v.id == variable
            )
            after = next(
                v.measurement for v in metadata.spec.variables if v.id == variable
            )

            def transform(row: DatasetRow) -> DatasetRow:
                if conversion is None:
                    return row
                assert before is not None and after is not None
                return conversion_row(row, conversion, input=before, output=after)

            def record(value: dict[str, object]) -> dict[str, object]:
                if conversion is None or value["variable_id"] != variable:
                    return value
                result = dict(value)
                if value["value"] is not None:
                    result["value"] = (
                        float(cast(float, value["value"])) * conversion.scale
                        + conversion.offset
                    )
                interval = value.get("interval")
                if isinstance(interval, Mapping):
                    result["interval"] = {
                        **interval,
                        **{
                            name: float(cast(float, interval[name])) * conversion.scale
                            + conversion.offset
                            for name in ("lower", "upper")
                        },
                    }
                return result

        metadata = _with_report(
            metadata, replace(metadata.report, complete=False, valid=None)
        )

        def batches() -> Iterator[pa.RecordBatch]:
            for batch in scope.iter_batches(self.parent):
                if isinstance(original, DatasetMetadata):
                    table = pa.Table.from_batches([batch]).replace_schema_metadata(
                        _schema(
                            _with_report(
                                original,
                                replace(
                                    original.report,
                                    complete=False,
                                    valid=False
                                    if original.report.valid is False
                                    else None,
                                ),
                            )
                        ).metadata
                    )
                    yield from rows_to_table(
                        [transform(row) for row in table_to_rows(table)],
                        cast(DatasetMetadata, metadata),
                    ).to_batches()
                else:
                    yield pa.RecordBatch.from_pylist(
                        [record(v) for v in batch.to_pylist()], schema=_schema(metadata)
                    )

        stream = _checked_stream(scope, metadata, batches())
        own = stream.report

        def report() -> ExecutionReport:
            assert own is not None
            local = own()
            parent = scope.computation_report(self.parent)
            return replace(
                local,
                diagnostics=parent.diagnostics,
                source_identities=parent.source_identities,
                counts=replace(
                    parent.counts,
                    produced=local.counts.produced,
                    no_data=local.counts.no_data,
                ),
            )

        stream.report = report
        return stream


def _bounded_node(
    parent: Node[MetadataT], state: SelectionState, *, discover: bool = False
) -> Node[MetadataT]:
    """Supply source requirements without replacing earlier scientific selections."""
    from ._evidence_recipes import _AggregationNode

    if isinstance(parent, SourceNode):
        selected = replace(
            parent.state,
            area=parent.state.area if parent.state.area is not None else state.area,
            time=parent.state.time if parent.state.time is not None else state.time,
            stations=parent.state.stations
            if parent.state.stations is not None
            else state.stations,
        )
        bounded = parent.selected(selected)
        if discover and (state.series or bounded.requirements()):
            available = bounded._variable_ids()
            if available is not None:
                required = set(bounded.requirements())
                references = tuple(
                    replace(ref, release=None)
                    for ref in state.series or ()
                    if ref.variable.id in available
                )
                variables = tuple(
                    var
                    for var in state.variables or ()
                    if (var if isinstance(var, str) else var.id) in available
                )
                if len(references) == 1 and selected.series is None:
                    candidate = bounded.selected(replace(selected, series=references))
                    if set(candidate.requirements()) < required:
                        return candidate
                if len(variables) == 1 and selected.variables is None:
                    candidate = bounded.selected(replace(selected, variables=variables))
                    if set(candidate.requirements()) < required:
                        return candidate
        return bounded
    if isinstance(parent, _TransformNode):
        return _TransformNode(
            _bounded_node(parent.parent, state, discover=discover),
            parent.operation,
            parent.arguments,
        )
    if isinstance(parent, _SelectionNode):
        return _SelectionNode(
            _bounded_node(parent.parent, state, discover=discover), parent.state
        )
    if isinstance(parent, _AggregationNode):
        return cast(
            Node[MetadataT],
            _AggregationNode(
                _bounded_node(parent.parent, state, discover=discover),
                parent.specification,
                parent.selection,
            ),
        )
    if isinstance(parent, _FitNode):
        return cast(
            Node[MetadataT],
            _FitNode(
                _bounded_node(parent.parent, state, discover=discover),
                parent.specification,
                parent.failure,
            ),
        )
    return parent


def _payload_source(node: Node[object]) -> SourceNode[object] | None:
    while not isinstance(node, SourceNode):
        parent = getattr(node, "parent", None)
        if not isinstance(parent, Node):
            return None
        node = parent
    return node if node._publishes_payload_release else None


def _configured_payload_pin(node: Node[object]) -> ReleaseReference | None:
    source = _payload_source(node)
    return None if source is None else source._configured_payload_pin()


def _local_variable_declarations(node: Node[object]) -> tuple[Variable, ...] | None:
    if isinstance(node, SourceNode):
        return node._variable_declarations()
    if isinstance(node, _FitNode):
        return _local_variable_declarations(cast(Node[object], node.parent))
    return None


def _select_node(parent: Node[MetadataT], state: SelectionState) -> Node[MetadataT]:
    from ._evidence_recipes import _AggregationNode

    if isinstance(parent, _AggregationNode):
        if parent.selection != SelectionState():
            return _SelectionNode(
                _bounded_node(cast(Node[MetadataT], parent), state), state
            )
        return cast(Node[MetadataT], parent.selected(state))
    if isinstance(parent, SourceNode):
        if parent.state != SelectionState():
            return _SelectionNode(_bounded_node(parent, state), state)
        return parent.selected(state)
    if isinstance(parent, _FitNode):
        return cast(Node[MetadataT], parent.selected(state))
    return _SelectionNode(_bounded_node(parent, state), state)


def _inspection_view(
    node: Node[object], source: Node[object], metadata: Metadata
) -> Metadata | None:
    """Project declared source science through the graph without opening its data."""
    from ._aggregation import aggregation_metadata
    from ._evidence_recipes import _AggregationNode

    if node is source:
        return metadata
    parent = getattr(node, "parent", None)
    if not isinstance(parent, Node):
        return None
    previous = _inspection_view(parent, source, metadata)
    if previous is None:
        return None
    if isinstance(node, _TransformNode):
        return cast(Metadata, node._metadata_view(previous))
    if isinstance(node, _SelectionNode):
        resolved = node.state.resolve(previous.spec, source=previous.source)
        return selected_metadata(previous, resolved)
    if isinstance(node, _FitNode):
        return DatasetMetadata(
            spec=previous.spec,
            spatial=previous.spatial,
            source=previous.source,
            provenance=previous.provenance,
            report=ExecutionReport(False, None),
        )
    if isinstance(node, _AggregationNode) and isinstance(previous, EvidenceMetadata):
        reduced = aggregation_metadata(previous, node.specification)
        return selected_metadata(
            reduced, node.selection.resolve(reduced.spec, source=reduced.source)
        )
    return None


def _declared_ids(
    node: Node[object], dimension: Literal["variables", "scenarios"]
) -> frozenset[str] | None:
    """Find a safe upper bound without acquiring resources or running a model."""
    from ._evidence_recipes import _AggregationNode

    if isinstance(node, SourceNode):
        return (
            node._variable_ids() if dimension == "variables" else node._scenario_ids()
        )
    if isinstance(node, (_TableNode, _ParquetNode)):
        metadata = node.metadata_value
        if metadata is not None:
            declarations = (
                metadata.spec.variables
                if dimension == "variables"
                else metadata.spec.scenarios
            )
            return frozenset(item.id for item in declarations)
        return None
    if not isinstance(
        node, (_AggregationNode, _FitNode, _SelectionNode, _TransformNode)
    ):
        return None
    ids = _declared_ids(cast(Node[object], node.parent), dimension)
    if (
        dimension == "scenarios"
        and isinstance(node, _TransformNode)
        and node.operation == "declare"
    ):
        scenario = node.arguments["scenario"]
        spec = node.arguments["spec"]
        if isinstance(scenario, Scenario):
            return frozenset((scenario.id,))
        if isinstance(spec, DataSpec):
            declared = frozenset(item.id for item in spec.scenarios)
            if all(item.scenario is not None for item in spec.series):
                return declared
            if ids is not None:
                return ids | declared
    if isinstance(node, _TransformNode) and node.operation == "rename":
        mapping = cast(Mapping[str, str] | None, node.arguments[dimension])
        if ids is not None and mapping is not None:
            return frozenset(mapping.get(name, name) for name in ids)
    state = (
        node.state
        if isinstance(node, _SelectionNode)
        else node.selection
        if isinstance(node, _AggregationNode)
        else None
    )
    if state is not None:
        choices = state.variables if dimension == "variables" else state.scenarios
        if choices is not None:
            selected = frozenset(
                item if isinstance(item, str) else item.id for item in choices
            )
            ids = selected if ids is None else ids & selected
        if state.series is not None:
            selected = (
                frozenset(item.variable.id for item in state.series)
                if dimension == "variables"
                else frozenset(
                    item.scenario.id
                    for item in state.series
                    if item.scenario is not None
                )
            )
            ids = selected if ids is None else ids & selected
    return ids


class _RecipeBase(Generic[MetadataT, CanonicalT]):
    __slots__ = ("_base", "_selection", "_operations", "_path", "_metadata")
    _base: Node[MetadataT]
    _selection: SelectionState
    _operations: tuple[Operation, ...]
    _path: Path | None
    _metadata: MetadataT | None

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            "Recipes are immutable; use a builder to create a new recipe"
        )

    def __init__(self) -> None:
        raise TypeError("Use an input constructor or open()")

    @classmethod
    def _create(
        cls,
        node: Node[MetadataT],
        *,
        operations: tuple[Operation, ...],
        selection: SelectionState | None = None,
        path: Path | None = None,
        metadata: MetadataT | None = None,
    ) -> Self:
        value = object.__new__(cls)
        object.__setattr__(value, "_base", node)
        object.__setattr__(
            value, "_selection", SelectionState() if selection is None else selection
        )
        object.__setattr__(value, "_operations", operations)
        object.__setattr__(value, "_path", path)
        object.__setattr__(value, "_metadata", metadata)
        return value

    @property
    def _node(self) -> Node[MetadataT]:
        if self._selection == SelectionState():
            return self._base
        return _select_node(self._base, self._selection)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def metadata(self) -> MetadataT | None:
        if self._metadata is not None or self._path is None:
            return self._metadata
        with ExecutionScope() as scope:
            try:
                metadata = scope.metadata(self._base)
                if isinstance(self._base, _ParquetNode):
                    reader = cast(_ParquetNode[MetadataT], self._base)
                    return reader.metadata_value or scope.computation_metadata(reader)
                return metadata
            except DataValidationError as error:
                raise ValidationError(report=scope.report(self._base)) from error

    @property
    def report(self) -> ExecutionReport | None:
        metadata = self.metadata
        return None if metadata is None else metadata.report

    def explain(self, *, runtime: Runtime | None = None) -> Explanation:
        config = Runtime() if runtime is None else runtime
        if not isinstance(config, Runtime):
            raise TypeError("runtime must be Runtime")
        node: Node[object] | None = cast(Node[object], self._node)
        requirements: tuple[Requirement, ...] = ()
        while node is not None:
            if isinstance(node, SourceNode):
                requirements = node.requirements()
                break
            parent = getattr(node, "parent", None)
            node = parent if isinstance(parent, Node) else None
        return Explanation(
            operations=self._operations,
            requirements=requirements,
            cache=config.cache,
            source=None if self._metadata is None else self._metadata.source,
            workers=config.workers,
            batch_rows=config.batch_rows,
            memory_limit=config.memory_limit,
        )

    def inspect(self, *, runtime: Runtime | None = None) -> Inspection:
        with ExecutionScope(runtime) as scope:
            return self._inspect(scope)

    def _inspect(self, scope: ExecutionScope) -> Inspection:
        node = self._node
        source: Node[object] | None = cast(Node[object], node)
        details: Inspection | None = None
        while source is not None:
            if isinstance(source, SourceNode):
                try:
                    details = source.inspection(scope)
                except DataValidationError as error:
                    raise ValidationError(
                        report=ExecutionReport(
                            False,
                            False,
                            diagnostics=error.diagnostics,
                        )
                    ) from error
                break
            parent = getattr(source, "parent", None)
            source = parent if isinstance(parent, Node) else None
        if details is not None and (
            details.schema is None
            or node is source
            or isinstance(source, SourceNode)
            and source.requirements()
        ):
            if (
                isinstance(self, Dataset)
                and Capability.READ in details.capabilities
                and Capability.EVALUATE not in details.capabilities
            ):
                return replace(
                    details, capabilities=(*details.capabilities, Capability.EVALUATE)
                )
            return details
        try:
            declared = None if details is None else details.metadata
            projected = (
                None
                if not isinstance(declared, (DatasetMetadata, EvidenceMetadata))
                or source is None
                else _inspection_view(cast(Node[object], node), source, declared)
            )
            metadata: Metadata = (
                scope.metadata(node) if projected is None else projected
            )
        except DataValidationError as error:
            raise ValidationError(report=scope.report(node)) from error
        capabilities: tuple[Capability, ...] = (
            Capability.INSPECT,
            Capability.READ,
            Capability.BATCHES,
            Capability.WRITE,
        )
        if isinstance(metadata, DatasetMetadata):
            capabilities += (Capability.FIT, Capability.EVALUATE)
        elif metadata.kind is EvidenceKind.SAMPLES:
            capabilities += (Capability.FIT,)
        elif (
            metadata.kind is EvidenceKind.OBSERVATIONS and metadata.sampling is not None
        ):
            capabilities += (Capability.AGGREGATE,)
        return Inspection(
            schema=_schema(metadata),
            series=tuple(
                SeriesReference(
                    item.variable,
                    scenario=item.scenario,
                    period=item.period,
                    probability=item.probability,
                    release=None
                    if metadata.source is None
                    else metadata.source.release,
                    model=None if metadata.source is None else metadata.source.model,
                    member=None if metadata.source is None else metadata.source.member,
                )
                for item in metadata.spec.series
            ),
            capabilities=capabilities,
            requirements=() if details is None else details.requirements,
            source=metadata.source,
            indicators=() if details is None else details.indicators,
            releases=() if details is None else details.releases,
            stations=() if details is None else details.stations,
        )

    def read(
        self, *, limit: int | None = None, runtime: Runtime | None = None
    ) -> ReadResult[MetadataT]:
        from ._terminals import _limit

        _limit(limit)
        with self.execute(runtime=runtime) as execution:
            return execution.read(limit=limit)

    def batches(
        self, *, batch_rows: int | None = None, runtime: Runtime | None = None
    ) -> BatchStream[MetadataT]:
        return BatchStream(ExecutionScope(runtime), self._node, batch_rows=batch_rows)

    def validate(self, *, runtime: Runtime | None = None) -> ExecutionReport:
        with self.execute(runtime=runtime) as execution:
            return execution.validate()

    def execute(
        self, *, runtime: Runtime | None = None
    ) -> Execution[MetadataT, CanonicalT]:
        return Execution(self, runtime=runtime)

    def write(
        self,
        path: LocalPath,
        *,
        overwrite: bool = False,
        runtime: Runtime | None = None,
    ) -> CanonicalT:
        _publication_path(path, overwrite)
        with self.execute(runtime=runtime) as execution:
            return execution.write(path, overwrite=overwrite)

    def _saved(self, path: Path, metadata: MetadataT, digest: str) -> CanonicalT:
        saved = self._canonical_type()._create(
            _ParquetNode[MetadataT](
                path,
                type(metadata),
                digest=digest,
                metadata=metadata,
                expected_kind=EvidenceKind.SAMPLES
                if isinstance(metadata, DatasetMetadata)
                else metadata.kind,
            ),
            operations=(Operation(Stage.DECODING, "Read the pinned saved artifact"),),
            path=path,
            metadata=metadata,
        )
        return cast(CanonicalT, saved)

    def _canonical_type(self) -> type[_RecipeBase[MetadataT, CanonicalT]]:
        raise NotImplementedError

    def _derived(
        self, node: Node[MetadataT], *, operations: tuple[Operation, ...]
    ) -> CanonicalT:
        return cast(
            CanonicalT, self._canonical_type()._create(node, operations=operations)
        )

    def declare(
        self,
        *,
        variable: Variable | None = None,
        scenario: Scenario | None = None,
        period: Period | None = None,
        probability: Probability | None = None,
        spec: DataSpec | None = None,
    ) -> CanonicalT:
        for name, value, expected in (
            ("variable", variable, Variable),
            ("scenario", scenario, Scenario),
            ("period", period, Period),
            ("probability", probability, Probability),
            ("spec", spec, DataSpec),
        ):
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__} or None")
        if spec is not None and any(
            value is not None for value in (variable, scenario, period, probability)
        ):
            raise ValueError("declare() accepts either spec or direct declarations")
        if all(
            value is None for value in (variable, scenario, period, probability, spec)
        ):
            return self._derived(self._node, operations=self._operations)
        node = _TransformNode(
            self._node,
            "declare",
            {
                "variable": variable,
                "scenario": scenario,
                "period": period,
                "probability": probability,
                "spec": spec,
            },
        )
        return self._derived(
            node,
            operations=(
                *self._operations,
                Operation(
                    Stage.SELECTION, "Declare previously unknown scientific meaning"
                ),
            ),
        )

    def rename(
        self,
        *,
        variables: Mapping[str, str] | None = None,
        scenarios: Mapping[str, str] | None = None,
    ) -> CanonicalT:
        mappings = {
            "variables": None if variables is None else dict(variables),
            "scenarios": None if scenarios is None else dict(scenarios),
        }
        return self._derived(
            _TransformNode(self._node, "rename", mappings),
            operations=(
                *self._operations,
                Operation(Stage.SELECTION, "Rename exact scientific identities"),
            ),
        )

    def convert_units(self, *, variable: Variable | str, unit: Unit) -> CanonicalT:
        units = (
            crc_framework.LengthUnit,
            crc_framework.TimeUnit,
            crc_framework.RatioUnit,
            crc_framework.TemperatureUnit,
            crc_framework.SpeedUnit,
            crc_framework.PrecipitationRateUnit,
            crc_framework.DegreeDayUnit,
        )
        if not isinstance(unit, units):
            raise TypeError("unit must be a framework unit enum")
        if not isinstance(variable, (Variable, str)):
            raise TypeError("variable must be Variable or str")
        node = _TransformNode(
            self._node, "convert_units", {"variable": variable, "unit": unit}
        )
        return self._derived(
            node,
            operations=(
                *self._operations,
                Operation(
                    Stage.SELECTION, "Convert declared units and numerical values"
                ),
            ),
        )


class Dataset(_RecipeBase[DatasetMetadata, "Dataset"]):
    __slots__ = ()

    def _canonical_type(self) -> type[Dataset]:
        return Dataset

    @classmethod
    def from_raster(
        cls,
        rasters: Mapping[float, LocalPath],
        *,
        variable: Variable,
        probability: Probability | None = None,
        scenario: Scenario | None = None,
        period: Period | None = None,
    ) -> Dataset:
        from ._raster_input import raster_node

        return Dataset._create(
            raster_node(
                rasters,
                variable=variable,
                probability=probability,
                scenario=scenario,
                period=period,
            ),
            operations=(
                Operation(Stage.DECODING, "Read supplied quantile raster pixels"),
            ),
        )

    @classmethod
    def open(cls, path: LocalPath) -> Dataset:
        destination = _local_path(path)
        return Dataset._create(
            _ParquetNode(destination, DatasetMetadata),
            operations=(
                Operation(Stage.DECODING, "Read a canonical dataset snapshot"),
            ),
            path=destination,
        )

    @classmethod
    def _collected(cls, table: pa.Table, metadata: DatasetMetadata) -> Dataset:
        return Dataset._create(
            _TableNode(table, metadata),
            operations=(
                Operation(Stage.DECODING, "Read owned collected dataset values"),
            ),
            metadata=metadata,
        )

    @classmethod
    def from_quantiles(
        cls,
        table: TableInput,
        *,
        variable: Variable | None = None,
        probability: Probability | None = None,
        scenario: Scenario | None = None,
        period: Period | None = None,
        spec: DataSpec | None = None,
        columns: QuantileColumns | None = None,
    ) -> Dataset:
        rows, metadata = normalize_quantiles(
            table,
            variable=variable,
            probability=probability,
            scenario=scenario,
            period=period,
            spec=spec,
            columns=columns,
        )
        return Dataset._create(
            _TableNode(rows_to_table(rows, metadata), metadata),
            operations=(
                Operation(Stage.DECODING, "Use supplied quantile evidence directly"),
            ),
        )

    def evaluate(
        self, assets: crc_sdk.Assets, *, spatial: crc_sdk.SpatialMatch | None = None
    ) -> crc_sdk.Evaluation:
        from ._evaluation import Evaluation

        return Evaluation._create(self, assets, spatial)

    def select(
        self,
        *,
        variables: VariableSelector | types.EllipsisType = ...,
        scenarios: ScenarioSelector | types.EllipsisType = ...,
        periods: PeriodSelector | types.EllipsisType = ...,
        series: SeriesSelector | types.EllipsisType = ...,
        area: Bounds | None | types.EllipsisType = ...,
    ) -> Self:
        state = self._selection.updated(
            variables=variables,
            scenarios=scenarios,
            periods=periods,
            series=series,
            area=area,
        )
        return type(self)._create(
            self._base,
            selection=state,
            operations=(
                *self._operations,
                Operation(
                    Stage.SELECTION,
                    "Select exact series and actual source support: "
                    + json.dumps(state.document(), ensure_ascii=False, sort_keys=True),
                ),
            ),
        )

    def fit(
        self, spec: QuantileFitSpec, *, on_failure: FitFailure = FitFailure.RAISE
    ) -> Dataset:
        _fit_kind(spec, samples=False)
        return Dataset._create(
            _FitNode(self._base, spec, on_failure),
            selection=self._selection,
            operations=(
                *self._operations,
                Operation(
                    Stage.FITTING,
                    "Fit explicitly selected quantile evidence (blocking)",
                ),
            ),
        )


class Samples(_RecipeBase[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]], "Samples"]):
    __slots__ = ()

    def _canonical_type(self) -> type[Samples]:
        return Samples

    @classmethod
    def open(cls, path: LocalPath) -> Samples:
        destination = _local_path(path)
        return Samples._create(
            _ParquetNode(destination, EvidenceMetadata),
            operations=(Operation(Stage.DECODING, "Read a sample evidence snapshot"),),
            path=destination,
        )

    @classmethod
    def _collected(
        cls, table: pa.Table, metadata: EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]
    ) -> Samples:
        return Samples._create(
            _TableNode(table, metadata),
            operations=(
                Operation(Stage.DECODING, "Read owned collected sample values"),
            ),
            metadata=metadata,
        )

    @classmethod
    def from_table(
        cls,
        table: TableInput,
        *,
        variable: Variable | None = None,
        probability: Probability | None = None,
        scenario: Scenario | None = None,
        period: Period | None = None,
        spec: DataSpec | None = None,
        columns: TimeSeriesColumns | None = None,
        sampling: Sampling | None = None,
    ) -> Samples:
        normalized, metadata = normalize_samples(
            table,
            variable=variable,
            probability=probability,
            scenario=scenario,
            period=period,
            spec=spec,
            columns=columns,
            sampling=sampling,
        )
        return Samples._create(
            _TableNode(normalized, metadata),
            operations=(
                Operation(Stage.DECODING, "Use owned long-form sample evidence"),
            ),
        )

    def select(
        self,
        *,
        variables: VariableSelector | types.EllipsisType = ...,
        scenarios: ScenarioSelector | types.EllipsisType = ...,
        periods: PeriodSelector | types.EllipsisType = ...,
        series: SeriesSelector | types.EllipsisType = ...,
        area: Bounds | None | types.EllipsisType = ...,
        stations: StationSelector | types.EllipsisType = ...,
        time: Period | None | types.EllipsisType = ...,
    ) -> Self:
        state = self._selection.updated(
            variables=variables,
            scenarios=scenarios,
            periods=periods,
            series=series,
            area=area,
            stations=stations,
            time=time,
        )
        return type(self)._create(
            self._base,
            selection=state,
            operations=(
                *self._operations,
                Operation(
                    Stage.SELECTION,
                    "Select sample identities, source support and observation time: "
                    + json.dumps(state.document(), ensure_ascii=False, sort_keys=True),
                ),
            ),
        )

    def fit(
        self, spec: SampleFitSpec, *, on_failure: FitFailure = FitFailure.RAISE
    ) -> Dataset:
        _fit_kind(spec, samples=True)
        return Dataset._create(
            _FitNode(self._base, spec, on_failure),
            selection=self._selection,
            operations=(
                *self._operations,
                Operation(
                    Stage.FITTING, "Fit explicitly selected genuine samples (blocking)"
                ),
            ),
        )
