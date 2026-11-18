"""Capability-specific recipes for native observations, projections, and layers."""

from __future__ import annotations

import json
import types
from collections.abc import Iterator
from dataclasses import replace
from typing import Literal, Self, cast, overload

import pyarrow as pa  # type: ignore[import-untyped]

from ._aggregation import (
    Aggregation,
    AnnualMaximum,
    AnnualMinimum,
    DailyTotal,
    Sampling,
    aggregate_table,
    aggregation_metadata,
)
from ._dataset_schema import ExecutionReport, Stage
from ._declarations import DataSpec, Period, Probability, Scenario, Variable
from ._errors import CapabilityError
from ._evidence import EvidenceKind, EvidenceMetadata, evidence_arrow_schema
from ._graph import ExecutionScope, Node, NodeStream
from ._inspection import Bounds, Operation, SeriesReference
from ._recipes import (
    Samples,
    _invalid,
    _key,
    _ParquetNode,
    _RecipeBase,
    _report_view,
    _ScientificNode,
    _select_node,
    _SelectionNode,
    _static_source,
    _TableNode,
)
from ._runtime import LocalPath, _local_path, _memory_bytes
from ._selection import (
    PeriodSelector,
    ScenarioSelector,
    SelectionState,
    SeriesSelector,
    StationSelector,
    VariableSelector,
    selected_metadata,
)
from ._table_input import (
    LayerColumns,
    ProjectionColumns,
    TableInput,
    TimeSeriesColumns,
    normalize_layer,
    normalize_observations,
    normalize_projections,
)


class _AggregationNode(_ScientificNode[EvidenceMetadata]):
    def __init__(
        self,
        parent: Node[EvidenceMetadata],
        spec: Aggregation,
        selection: SelectionState | None = None,
    ) -> None:
        if not isinstance(spec, (DailyTotal, AnnualMaximum, AnnualMinimum)):
            raise TypeError("spec must be DailyTotal, AnnualMaximum, or AnnualMinimum")
        self.parent = parent
        self.specification = spec
        self.selection = SelectionState() if selection is None else selection
        self.key = _key(
            "aggregate",
            parent.key,
            type(spec).__name__,
            spec.coverage.fraction,
            self.selection.document(),
        )

    def selected(self, state: SelectionState) -> _AggregationNode:
        return _AggregationNode(self.parent, self.specification, state)

    def _source(self) -> Node[EvidenceMetadata]:
        # Bounds determine which native evidence can be acquired. Scientific
        # selectors describe the reduced output and are resolved only afterward.
        state = SelectionState(
            area=self.selection.area,
            stations=self.selection.stations,
            time=self.selection.time,
        )

        def bounded(node: Node[EvidenceMetadata]) -> Node[EvidenceMetadata]:
            if isinstance(node, _AggregationNode):
                return _AggregationNode(
                    bounded(node.parent), node.specification, node.selection
                )
            return _select_node(node, state)

        return self.parent if state == SelectionState() else bounded(self.parent)

    def _planned(self, scope: ExecutionScope) -> EvidenceMetadata:
        try:
            metadata = aggregation_metadata(
                scope.metadata(self._source()), self.specification
            )
        except ValueError as error:
            raise CapabilityError(str(error)) from error
        resolved = self.selection.resolve(metadata.spec, source=metadata.source)
        return selected_metadata(metadata, resolved)

    def _input(self, scope: ExecutionScope) -> Node[EvidenceMetadata]:
        source = self._source()
        original = scope.metadata(source)
        planned = self._planned(scope)
        keys = {
            (
                s.variable.id,
                None if s.scenario is None else s.scenario.id,
                None if s.period is None else s.period.id,
            )
            for s in planned.spec.series
        }
        references = tuple(
            SeriesReference(
                s.variable,
                scenario=s.scenario,
                period=s.period,
                probability=s.probability,
            )
            for s in original.spec.series
            if (
                s.variable.id,
                None if s.scenario is None else s.scenario.id,
                None if s.period is None else s.period.id,
            )
            in keys
        )
        state = SelectionState().updated(series=references)
        return (
            _SelectionNode(source, state)
            if isinstance(source, _AggregationNode)
            else _select_node(source, state)
        )

    def _view_metadata(
        self, scope: ExecutionScope, metadata: EvidenceMetadata
    ) -> EvidenceMetadata:
        known, source = _static_source(scope, self._source())
        return replace(
            self._planned(scope),
            source=source if known else metadata.source,
            report=scope.report(self),
        )

    def _view_report(
        self, scope: ExecutionScope, report: ExecutionReport
    ) -> ExecutionReport:
        known, source = _static_source(scope, self._source())
        parent = scope.report(self._input(scope))
        return _report_view(
            report,
            parent=parent,
            source_known=True,
            source=source if known else scope.computation_metadata(self).source,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[EvidenceMetadata]:
        planned = self._planned(scope)
        parent = self._input(scope)
        result = planned

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal result
            parts: list[pa.RecordBatch] = []
            size = 0
            budget = _memory_bytes(scope.runtime.memory_limit)
            for batch in scope.iter_batches(parent):
                size += batch.nbytes
                if budget is not None and size > budget:
                    raise MemoryError(
                        "Temporal grouping exceeds Runtime(memory_limit=...). "
                        "Narrow select() or increase the limit."
                    )
                parts.append(batch)
            metadata = replace(scope.metadata(parent), report=scope.report(parent))
            table = pa.Table.from_batches(
                parts, schema=evidence_arrow_schema(metadata)
            ).replace_schema_metadata(evidence_arrow_schema(metadata).metadata)
            scope.emit(Stage.REDUCTION, 0)
            try:
                values, result = aggregate_table(table, metadata, self.specification)
            except ValueError as error:
                raise _invalid(error, stage=Stage.REDUCTION) from error
            scope.emit(Stage.REDUCTION, values.num_rows, values.num_rows)
            yield from values.to_batches(max_chunksize=scope.runtime.batch_rows)

        return NodeStream(
            evidence_arrow_schema(planned),
            planned,
            batches(),
            report=lambda: result.report,
            final_metadata=lambda: result,
        )


class Observations(
    _RecipeBase[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]], "Observations"]
):
    __slots__ = ()

    def _canonical_type(self) -> type[Observations]:
        return Observations

    @classmethod
    def open(cls, path: LocalPath) -> Observations:
        destination = _local_path(path)
        node = _ParquetNode(
            destination, EvidenceMetadata, expected_kind=EvidenceKind.OBSERVATIONS
        )
        return Observations._create(
            cast(Node[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]], node),
            operations=(Operation(Stage.DECODING, "Read native observation evidence"),),
            path=destination,
        )

    @classmethod
    def _collected(
        cls,
        table: pa.Table,
        metadata: EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]],
    ) -> Observations:
        if metadata.kind is not EvidenceKind.OBSERVATIONS:
            raise TypeError("observations require observation evidence metadata")
        return Observations._create(
            _TableNode(table, metadata),
            operations=(
                Operation(Stage.DECODING, "Read owned collected observations"),
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
    ) -> Observations:
        values, metadata = normalize_observations(
            table,
            variable=variable,
            probability=probability,
            scenario=scenario,
            period=period,
            spec=spec,
            columns=columns,
            sampling=sampling,
        )
        return Observations._create(
            _TableNode(values, metadata),
            operations=(Operation(Stage.DECODING, "Use native observation evidence"),),
        )

    def select(
        self,
        *,
        variables: VariableSelector | types.EllipsisType = ...,
        series: SeriesSelector | types.EllipsisType = ...,
        area: Bounds | None | types.EllipsisType = ...,
        stations: StationSelector | types.EllipsisType = ...,
        time: Period | None | types.EllipsisType = ...,
    ) -> Self:
        state = self._selection.updated(
            variables=variables, series=series, area=area, stations=stations, time=time
        )
        return type(self)._create(
            self._base,
            selection=state,
            operations=(
                *self._operations,
                Operation(
                    Stage.SELECTION,
                    "Select native observations: "
                    + json.dumps(state.document(), ensure_ascii=False, sort_keys=True),
                ),
            ),
        )

    @overload
    def aggregate(self, spec: DailyTotal) -> Observations: ...
    @overload
    def aggregate(self, spec: AnnualMaximum | AnnualMinimum) -> Samples: ...
    def aggregate(self, spec: Aggregation) -> Samples | Observations:
        node = _AggregationNode(cast(Node[EvidenceMetadata], self._node), spec)
        operations = (
            *self._operations,
            Operation(
                Stage.REDUCTION,
                f"{type(spec).__name__} with coverage {spec.coverage.fraction:g} "
                "(blocking temporal grouping)",
            ),
        )
        if isinstance(spec, DailyTotal):
            return Observations._create(
                cast(Node[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]], node),
                operations=operations,
            )
        return Samples._create(
            cast(Node[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]], node),
            operations=operations,
        )


class Projections(
    _RecipeBase[EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]], "Projections"]
):
    __slots__ = ()

    def _canonical_type(self) -> type[Projections]:
        return Projections

    @classmethod
    def open(cls, path: LocalPath) -> Projections:
        destination = _local_path(path)
        node = _ParquetNode(
            destination, EvidenceMetadata, expected_kind=EvidenceKind.PROJECTIONS
        )
        return Projections._create(
            cast(Node[EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]]], node),
            operations=(Operation(Stage.DECODING, "Read projection evidence"),),
            path=destination,
        )

    @classmethod
    def _collected(
        cls,
        table: pa.Table,
        metadata: EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]],
    ) -> Projections:
        if metadata.kind is not EvidenceKind.PROJECTIONS:
            raise TypeError("projections require projection evidence metadata")
        return Projections._create(
            _TableNode(table, metadata),
            operations=(Operation(Stage.DECODING, "Read owned collected projections"),),
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
        columns: ProjectionColumns | None = None,
    ) -> Projections:
        values, metadata = normalize_projections(
            table,
            variable=variable,
            probability=probability,
            scenario=scenario,
            period=period,
            spec=spec,
            columns=columns,
        )
        return Projections._create(
            _TableNode(values, metadata),
            operations=(
                Operation(
                    Stage.DECODING, "Use supplied projection coordinates and intervals"
                ),
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
                    "Select projection coordinates: "
                    + json.dumps(state.document(), ensure_ascii=False, sort_keys=True),
                ),
            ),
        )


class Layer(_RecipeBase[EvidenceMetadata[Literal[EvidenceKind.LAYER]], "Layer"]):
    __slots__ = ()

    def _canonical_type(self) -> type[Layer]:
        return Layer

    @classmethod
    def open(cls, path: LocalPath) -> Layer:
        destination = _local_path(path)
        node = _ParquetNode(
            destination, EvidenceMetadata, expected_kind=EvidenceKind.LAYER
        )
        return Layer._create(
            cast(Node[EvidenceMetadata[Literal[EvidenceKind.LAYER]]], node),
            operations=(Operation(Stage.DECODING, "Read static spatial evidence"),),
            path=destination,
        )

    @classmethod
    def _collected(
        cls, table: pa.Table, metadata: EvidenceMetadata[Literal[EvidenceKind.LAYER]]
    ) -> Layer:
        if metadata.kind is not EvidenceKind.LAYER:
            raise TypeError("layers require layer evidence metadata")
        return Layer._create(
            _TableNode(table, metadata),
            operations=(
                Operation(Stage.DECODING, "Read owned collected layer features"),
            ),
            metadata=metadata,
        )

    @classmethod
    def from_table(
        cls,
        table: TableInput,
        *,
        variable: Variable | None = None,
        scenario: Scenario | None = None,
        period: Period | None = None,
        spec: DataSpec | None = None,
        columns: LayerColumns | None = None,
    ) -> Layer:
        values, metadata = normalize_layer(
            table,
            variable=variable,
            scenario=scenario,
            period=period,
            spec=spec,
            columns=columns,
        )
        return Layer._create(
            _TableNode(values, metadata),
            operations=(Operation(Stage.DECODING, "Use supplied spatial features"),),
        )

    def select(
        self,
        *,
        variables: VariableSelector | types.EllipsisType = ...,
        periods: PeriodSelector | types.EllipsisType = ...,
        series: SeriesSelector | types.EllipsisType = ...,
        area: Bounds | None | types.EllipsisType = ...,
    ) -> Self:
        state = self._selection.updated(
            variables=variables, periods=periods, series=series, area=area
        )
        return type(self)._create(
            self._base,
            selection=state,
            operations=(
                *self._operations,
                Operation(
                    Stage.SELECTION,
                    "Select static layer support: "
                    + json.dumps(state.document(), ensure_ascii=False, sort_keys=True),
                ),
            ),
        )
