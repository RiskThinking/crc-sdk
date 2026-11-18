"""Asset evaluation recipes and portable evaluation artifacts."""

from __future__ import annotations

import math
import types
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from dataclasses import replace as _replace
from pathlib import Path
from typing import Self, cast

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ._assets import Assets
from ._collection import DatasetCollection, _CollectionNode
from ._collection_schema import CollectionMetadata
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionDiagnostic,
    ExecutionReport,
    Stage,
    _identifier,
)
from ._declarations import SeriesSpec, Variable, scientific_identity
from ._errors import CapabilityError, ValidationError
from ._evaluation_responses import Response
from ._evaluation_schema import (
    AssetId,
    EvaluationFailure,
    EvaluationInput,
    EvaluationMetadata,
    EvaluationQuery,
    EvaluationResolution,
    EvaluationResponse,
    EvaluationRow,
    EvaluationValidator,
    QueryKind,
    _apply_response,
    evaluation_arrow_schema,
    evaluation_metadata_from_schema,
    evaluation_table,
)
from ._evaluation_spatial import SpatialIndex, SpatialMatch, SpatialMatchError
from ._graph import DataValidationError, ExecutionScope, Node, NodeStream
from ._inspection import (
    Bounds,
    Capability,
    Explanation,
    Inspection,
    Operation,
    Requirement,
)
from ._recipes import Dataset, _key
from ._runtime import LocalPath, Runtime, _local_path, _memory_bytes
from ._schema import DatasetRow, dataset_arrow_schema, table_to_rows
from ._selection import (
    IdentifierSelector,
    PeriodSelector,
    ScenarioSelector,
    SelectionState,
    SeriesKey,
    SeriesSelector,
    VariableSelector,
    _sequence,
    _series_key,
    selected_metadata,
)
from ._source_io import native_data
from ._terminals import BatchStream, Execution, ReadResult

_CAPABILITIES = (Capability.INSPECT, Capability.READ, Capability.BATCHES)
_QUERY_REQUIRED = Requirement(
    "evaluation_query_required",
    "Choose probabilities(...) or return_periods(...) before evaluation.",
)
_ASSET_SCHEMA_UNRESOLVED = Requirement(
    "asset_schema_unresolved",
    "File and SQL asset schemas become available when the evaluation runs.",
)


def _ids(values: IdentifierSelector) -> tuple[str, ...] | None:
    if values is None:
        return None
    entries = (values,) if isinstance(values, str) else _sequence(values)
    return tuple(dict.fromkeys(_identifier(v, "selection id") for v in entries))


@dataclass(frozen=True, slots=True)
class _Selectors:
    members: IdentifierSelector | types.EllipsisType = ...
    variables: VariableSelector | types.EllipsisType = ...
    scenarios: ScenarioSelector | types.EllipsisType = ...
    periods: PeriodSelector | types.EllipsisType = ...
    series: SeriesSelector | types.EllipsisType = ...
    responses: tuple[str, ...] | None = None
    area: Bounds | None | types.EllipsisType = ...


@dataclass(frozen=True, slots=True)
class _Plan:
    source: Dataset | DatasetCollection
    assets: Assets
    spatial: SpatialMatch
    query: EvaluationQuery | None = None
    responses: tuple[Response, ...] = ()
    selection: _Selectors = _Selectors()

    def _empty(self, source: Dataset | DatasetCollection) -> bool:
        state = source._selection
        selectors = (
            state.variables,
            state.scenarios,
            state.periods,
            state.series,
            source._selection.members
            if isinstance(source, DatasetCollection)
            else None,
        )
        # Other explicit IDs still need validation before an empty result.
        return (self.selection.responses == () or () in selectors) and not any(
            selectors
        )

    def _selected_source(self) -> Dataset | DatasetCollection:
        state = self.selection
        source = self.source
        if isinstance(source, Dataset):
            if state.members is not ... and state.members is not None:
                raise ValueError("Member IDs select a DatasetCollection")
            if state.members is ... and all(
                value is ...
                for value in (
                    state.variables,
                    state.scenarios,
                    state.periods,
                    state.series,
                    state.area,
                )
            ):
                return source
            return source.select(
                variables=state.variables,
                scenarios=state.scenarios,
                periods=state.periods,
                series=state.series,
                area=state.area,
            )
        if all(
            value is ...
            for value in (
                state.members,
                state.variables,
                state.scenarios,
                state.periods,
                state.series,
                state.area,
            )
        ):
            return source
        return source.select(
            members=state.members,
            variables=state.variables,
            scenarios=state.scenarios,
            periods=state.periods,
            series=state.series,
            area=state.area,
        )

    def _selected_responses(self) -> tuple[Response, ...]:
        names = self.selection.responses
        if names is None:
            return self.responses
        available = {item.name: item for item in self.responses}
        missing = set(names) - available.keys()
        if missing:
            raise ValueError(f"unknown response identities {sorted(missing)!r}")
        return tuple(available[name] for name in names)

    def _response_source(self) -> Dataset | DatasetCollection:
        source = self._selected_source()
        if (
            self.selection.responses
            and source._selection.variables is None
            and not self._empty(source)
        ):
            targets = tuple(
                dict.fromkeys(r.variable_id for r in self._selected_responses())
            )
            return source.select(variables=targets)
        return source

    def _collection_projection(
        self, source: DatasetCollection
    ) -> frozenset[str] | None:
        if self.selection.responses == ():
            return frozenset()
        if not self.selection.responses or source._selection.variables is None:
            return None
        selected = {
            item if isinstance(item, str) else item.id
            for item in source._selection.variables
        }
        participating = frozenset(
            item.variable_id
            for item in self._selected_responses()
            if item.variable_id in selected
        )
        return participating or None

    def _require_query(self) -> EvaluationQuery:
        if self.query is None:
            raise CapabilityError(_QUERY_REQUIRED.message)
        return self.query


def _inputs(
    metadata: DatasetMetadata | CollectionMetadata,
) -> tuple[EvaluationInput, ...]:
    if isinstance(metadata, DatasetMetadata):
        return (EvaluationInput(None, None, metadata),)
    resources = {item.id: item.metadata for item in metadata.resources}
    return tuple(
        EvaluationInput(member.id, resource, resources[resource])
        for member in metadata.members
        for resource in member.resource_ids
    )


def _diagnostic(
    code: str,
    message: str,
    *,
    stage: Stage,
    support_id: str | None = None,
    query: float | None = None,
) -> DataValidationError:
    return DataValidationError(
        (
            ExecutionDiagnostic(
                code,
                cf.DiagnosticSeverity.ERROR,
                stage,
                message,
                support_id=support_id,
                query_value=query,
            ),
        )
    )


_EvaluationKey = tuple[str | None, SeriesKey]


def _series(inputs: Sequence[EvaluationInput]) -> dict[_EvaluationKey, SeriesSpec]:
    result: dict[_EvaluationKey, SeriesSpec] = {}
    for item in inputs:
        for series in item.metadata.spec.series:
            key = (item.member_id, _series_key(series))
            existing = result.setdefault(key, series)
            if scientific_identity(existing) != scientific_identity(series):
                raise ValueError(
                    f"conflicting science across resources for series {key!r}"
                )
    return result


def _definitions(
    series: Sequence[SeriesSpec],
    responses: Sequence[Response],
    query: EvaluationQuery,
) -> tuple[EvaluationResponse, ...]:
    by_variable: dict[str, list[SeriesSpec]] = {}
    for item in series:
        by_variable.setdefault(item.variable.id, []).append(item)
        try:
            query._validate_series(item)
        except CapabilityError as error:
            raise _diagnostic(
                "invalid_query_science",
                str(error),
                stage=Stage.EVALUATION,
            ) from error
    definitions = []
    for response in responses:
        target = response.variable
        variable_id = target.id if isinstance(target, Variable) else target
        applicable = by_variable.get(variable_id, ())
        if not applicable:
            if not series:
                continue
            raise _diagnostic(
                "response_variable_missing",
                f"Response {response.name!r} requires selected "
                f"variable {variable_id!r}.",
                stage=Stage.EVALUATION,
            )
        for item in applicable:
            if isinstance(target, Variable) and (
                scientific_identity(target) != scientific_identity(item.variable)
            ):
                raise _diagnostic(
                    "response_variable_mismatch",
                    f"Response {response.name!r} has conflicting "
                    f"science for {variable_id!r}.",
                    stage=Stage.EVALUATION,
                )
            try:
                response.bind(item.variable.measurement)
            except cf.ImpactError as error:
                raise _diagnostic(
                    error.code,
                    f"Response {response.name!r} for {variable_id!r}: {error.message}",
                    stage=Stage.EVALUATION,
                ) from error
        definitions.append(
            EvaluationResponse(
                response.name,
                variable_id,
                response.model,
                response.context,
            )
        )
    return tuple(definitions)


def _response_series(
    series: dict[_EvaluationKey, SeriesSpec],
    responses: Sequence[Response],
    *,
    exposure: bool,
) -> dict[_EvaluationKey, SeriesSpec]:
    if exposure:
        return series
    variables = {response.variable_id for response in responses}
    # Keep the original series if every target is missing so binding can report it.
    selected = {
        key: item for key, item in series.items() if item.variable.id in variables
    }
    return selected or series


def _response_inputs(
    inputs: tuple[EvaluationInput, ...],
    groups: dict[_EvaluationKey, SeriesSpec],
) -> tuple[EvaluationInput, ...]:
    result = []
    for item in inputs:
        matching = tuple(
            s.variable.id
            for s in item.metadata.spec.series
            if (item.member_id, _series_key(s)) in groups
        )
        if not matching:
            continue
        resolved = SelectionState(variables=matching).resolve(item.metadata.spec)
        result.append(
            _replace(item, metadata=selected_metadata(item.metadata, resolved))
        )
    return tuple(result)


def _restrict_variables(
    source: Dataset | DatasetCollection, ids: tuple[str, ...]
) -> Dataset | DatasetCollection:
    if not isinstance(source, Dataset):
        return source._project_variables(ids)
    requested = source._selection.variables
    if requested is None:
        return source.select(variables=ids)
    return source.select(
        variables=tuple(
            item
            for item in requested
            if (item if isinstance(item, str) else item.id) in ids
        )
    )


def _asset_fields(assets: Assets, schema: pa.Schema | None = None) -> pa.Schema | None:
    if schema is None:
        schema = assets._retained_schema()
    if schema is None:
        return None
    if assets.id_column not in schema.names:
        raise ValueError(f"missing asset id column {assets.id_column!r}")
    attributes = tuple(field for field in schema if field.name != assets.id_column)
    if any(field.name == "asset_id" for field in attributes):
        raise ValueError(
            "Retained column 'asset_id' conflicts with the evaluation identifier. "
            "Choose it as id_column or omit it from columns."
        )
    id_type = schema.field(assets.id_column).type
    if pa.types.is_dictionary(id_type):
        id_type = id_type.value_type
    if pa.types.is_integer(id_type):
        id_type = pa.uint64() if pa.types.is_unsigned_integer(id_type) else pa.int64()
    else:
        id_type = pa.string()
    return pa.schema([pa.field("asset_id", id_type, nullable=False), *attributes])


def _empty_metadata(query: EvaluationQuery, plan: _Plan) -> EvaluationMetadata:
    fields = _asset_fields(plan.assets)
    return EvaluationMetadata(
        query=query,
        spatial=plan.spatial,
        asset_ids=(),
        include_exposure=False,
        asset_fields=(
            pa.schema([pa.field("asset_id", pa.string(), nullable=False)])
            if fields is None
            else fields
        ),
    )


class _EvaluationNode(Node[EvaluationMetadata]):
    def __init__(self, plan: _Plan) -> None:
        self.plan = plan
        self.source = plan._response_source()
        source_node = (
            _CollectionNode(
                self.source,
                projected_variables=plan._collection_projection(self.source),
            )
            if isinstance(self.source, DatasetCollection)
            else self.source._node
        )
        self.parent = cast(Node[DatasetMetadata | CollectionMetadata], source_node)
        self.responses = plan._selected_responses()
        self.key = _key(
            "evaluation",
            self.parent.key,
            plan.assets._key,
            plan.spatial.kind.value,
            None
            if plan.spatial.max_distance is None
            else plan.spatial.max_distance.value_metres,
            None if plan.query is None else (plan.query.kind.value, plan.query.values),
            plan.selection.responses,
            [
                (
                    r.name,
                    r.model.sha256,
                    r.variable
                    if isinstance(r.variable, str)
                    else (r.variable.id, scientific_identity(r.variable)),
                    None if r.context is None else r.context.to_json(),
                )
                for r in self.responses
            ],
        )

    def open(self, scope: ExecutionScope) -> NodeStream[EvaluationMetadata]:
        plan = self.plan
        query = plan._require_query()
        if plan._empty(self.source):
            return self._empty(query)
        parent = self.parent
        declaration = scope.metadata(parent)
        if plan.selection.responses == ():
            return self._empty(query)
        inputs = _inputs(declaration)
        series = _series(inputs)
        exposure = plan.selection.responses is None
        groups = _response_series(series, self.responses, exposure=exposure)
        definitions = _definitions(tuple(groups.values()), self.responses, query)
        if len(groups) < len(series):
            targets = tuple(dict.fromkeys(item.variable.id for item in groups.values()))
            projected = _restrict_variables(self.source, targets)
            parent = cast(Node[DatasetMetadata | CollectionMetadata], projected._node)
            declaration = scope.metadata(parent)
            inputs = _inputs(declaration)
        responders = {
            variable: tuple(r for r in self.responses if r.variable_id == variable)
            for variable in {s.variable.id for s in groups.values()}
        }
        if not groups:
            return self._empty(query)
        required_columns = tuple(
            dict.fromkeys(col for r in self.responses for col in r.columns)
        )
        with native_data(scope=scope, source="assets"):
            assets = plan.assets._materialize(scope, required_columns=required_columns)
        ids = tuple(assets.table[assets.id_column].to_pylist())
        asset_schema = _asset_fields(plan.assets, assets.table.schema)
        assert asset_schema is not None
        asset_columns = tuple(asset_schema.names[1:])
        planned = EvaluationMetadata(
            query=query,
            inputs=_response_inputs(inputs, groups),
            responses=definitions,
            spatial=plan.spatial,
            asset_fields=asset_schema,
            asset_ids=ids,
            include_exposure=exposure,
            source=declaration.source,
        )
        diagnostics: dict[str, ExecutionDiagnostic] = {}
        resolutions: dict[tuple[str, str], tuple[cf.Impact, dict[AssetId, None]]] = {}
        resolved_models: dict[tuple[str, str], cf.Impact] = {}
        produced = no_data = matched = unmatched = 0
        requested = (
            len(ids)
            * len(query.values)
            * sum(
                int(exposure) + len(responders[s.variable.id]) for s in groups.values()
            )
        )
        current_inputs = planned.inputs
        validator = EvaluationValidator(planned, scope_directory=scope.directory)
        scope.callback(validator.close)

        def report() -> ExecutionReport:
            upstream = scope.report(parent)
            sources = list(upstream.source_identities)
            if declaration.source is not None and declaration.source not in sources:
                sources.append(declaration.source)
            if assets.source is not None and assets.source not in sources:
                sources.append(assets.source)
            return ExecutionReport(
                False,
                None,
                tuple((*upstream.diagnostics, *diagnostics.values())),
                _replace(
                    upstream.counts,
                    requested=requested,
                    produced=produced,
                    no_data=no_data,
                    matched_assets=matched,
                    unmatched_assets=unmatched,
                ),
                tuple(sources),
            )

        def metadata(*, complete: bool = False) -> EvaluationMetadata:
            outcome = report()
            if complete:
                outcome = _replace(outcome, complete=True, valid=True)
            return _replace(
                planned,
                inputs=current_inputs,
                resolutions=tuple(
                    EvaluationResolution(name, impact, asset_ids=tuple(assigned_assets))
                    for (name, _), (impact, assigned_assets) in resolutions.items()
                ),
                report=outcome,
            )

        def note(
            failure: EvaluationFailure, *, support: str | None, value: float
        ) -> None:
            diagnostics.setdefault(
                failure.code,
                ExecutionDiagnostic(
                    failure.code,
                    cf.DiagnosticSeverity.WARNING,
                    Stage.EVALUATION,
                    failure.message,
                    support_id=support,
                    query_value=value,
                ),
            )

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal current_inputs, produced, no_data, matched, unmatched
            if not ids:
                validator.finish(metadata(complete=True))
                return
            scope.emit(Stage.JOINING, 0)
            grouped, resources = _index_inputs(scope, parent, inputs, groups)
            current_inputs = _response_inputs(_inputs(scope.metadata(parent)), groups)
            upstream = scope.report(parent)
            if upstream.valid is False:
                raise DataValidationError(upstream.diagnostics)
            try:
                indices = {
                    key: SpatialIndex(grouped.get(key, ()), plan.spatial)
                    for key in groups
                }
            except SpatialMatchError as error:
                raise _diagnostic(
                    error.code, str(error), stage=Stage.JOINING
                ) from error
            scope.emit(Stage.JOINING, sum(map(len, grouped.values())))
            pending: list[EvaluationRow] = []
            asset_values = assets.table.select(asset_columns) if asset_columns else None
            contexts = assets.context
            for i, asset_id in enumerate(ids):
                asset = (
                    None
                    if asset_values is None
                    else {name: asset_values[name][i].as_py() for name in asset_columns}
                )
                context = {name: contexts[name][i].as_py() for name in required_columns}
                matches = {}
                for key, index in indices.items():
                    try:
                        matches[key] = index.match(assets.geometries[i])
                    except SpatialMatchError as error:
                        raise _diagnostic(
                            error.code,
                            f"Asset {asset_id!r}: {error}",
                            stage=Stage.JOINING,
                        ) from error
                if any(match.row is not None for match in matches.values()):
                    matched += 1
                else:
                    unmatched += 1
                for key, item in groups.items():
                    member, (variable, scenario, period) = key
                    match = matches[key]
                    support = match.row
                    resource = (
                        None
                        if support is None
                        else resources[(key, support.support_id, support.cell_id)]
                    )
                    exposures = tuple(
                        _value(query, value, item, support, match.failure_code)
                        for value in query.values
                    )
                    finite_exposure = any(
                        level is not None and math.isfinite(level)
                        for level, _ in exposures
                    )
                    active = responders[variable]
                    outputs: tuple[Response | None, ...] = (
                        (None,) if exposure else ()
                    ) + active
                    for response in outputs:
                        resolved = None
                        resolution_id = None
                        if (
                            response is not None
                            and support is not None
                            and finite_exposure
                        ):
                            try:
                                resolved = response.resolve(context, resolved_models)
                            except cf.ImpactError as error:
                                raise _diagnostic(
                                    error.code,
                                    f"Asset {asset_id!r}, response {response.name!r}: "
                                    f"{error.message}",
                                    stage=Stage.EVALUATION,
                                    support_id=support.support_id,
                                ) from error
                            if isinstance(response.model, cf.ImpactReference):
                                resolution_id = resolved.sha256
                                resolution_key = (response.name, resolution_id)
                                if resolution_key not in resolutions:
                                    record = EvaluationResolution(
                                        response.name, resolved, asset_ids=(asset_id,)
                                    )
                                    validator.register_resolution(record)
                                    resolutions[resolution_key] = (resolved, {})
                                _, assigned_assets = resolutions[resolution_key]
                                assigned_assets.setdefault(asset_id, None)
                        for value, (result, failure) in zip(
                            query.values,
                            exposures,
                            strict=True,
                        ):
                            if response is not None and failure is None:
                                assert result is not None
                                result, failure = _apply_response(result, resolved)
                            if failure is not None:
                                no_data += 1
                                note(
                                    failure,
                                    support=None
                                    if support is None
                                    else support.support_id,
                                    value=value,
                                )
                            produced += 1
                            pending.append(
                                EvaluationRow(
                                    asset_id=asset_id,
                                    member_id=member,
                                    resource_id=resource,
                                    variable_id=variable,
                                    scenario_id=scenario,
                                    period_id=period,
                                    support_id=None
                                    if support is None
                                    else support.support_id,
                                    distance_metres=match.distance_metres,
                                    response_id=None
                                    if response is None
                                    else response.name,
                                    resolution_id=resolution_id,
                                    query_value=value,
                                    value=result,
                                    failure=failure,
                                    asset=asset,
                                )
                            )
                            if len(pending) >= scope.runtime.batch_rows:
                                yield from _evaluation_batches(
                                    scope, pending, planned, validator
                                )
                                pending.clear()
            if pending:
                yield from _evaluation_batches(scope, pending, planned, validator)
            with native_data(scope=scope, source="evaluation"):
                validator.finish(metadata(complete=True))

        return NodeStream(
            evaluation_arrow_schema(planned, include_metadata=False),
            planned,
            batches(),
            report=report,
            final_metadata=metadata,
        )

    def _empty(self, query: EvaluationQuery) -> NodeStream[EvaluationMetadata]:
        metadata = _empty_metadata(query, self.plan)
        return NodeStream(evaluation_arrow_schema(metadata), metadata, iter(()))


def _index_inputs(
    scope: ExecutionScope,
    parent: Node[DatasetMetadata | CollectionMetadata],
    inputs: tuple[EvaluationInput, ...],
    groups: dict[_EvaluationKey, SeriesSpec],
) -> tuple[
    dict[_EvaluationKey, list[DatasetRow]],
    dict[tuple[_EvaluationKey, str, int], str | None],
]:
    declarations = {(i.member_id, i.resource_id): i.metadata for i in inputs}
    grouped: dict[_EvaluationKey, list[DatasetRow]] = {}
    resources: dict[tuple[_EvaluationKey, str, int], str | None] = {}
    budget = _memory_bytes(scope.runtime.memory_limit)
    size = 0
    for batch in scope.iter_batches(parent):
        size += batch.nbytes
        if budget is not None and size > budget:
            raise MemoryError(
                "Spatial indexing exceeds Runtime(memory_limit=...). "
                "Select a smaller area or fewer series, or increase the limit."
            )
        table = pa.Table.from_batches([batch])
        parts: Iterator[tuple[tuple[str | None, str | None], pa.Table]]
        if "member_id" in table.column_names:
            # A normalized batch can contain rows from more than one shard.
            pair_rows: dict[tuple[str | None, str | None], list[int]] = {}
            for i, pair in enumerate(
                zip(
                    table["member_id"].to_pylist(),
                    table["resource_id"].to_pylist(),
                    strict=True,
                )
            ):
                pair_rows.setdefault(pair, []).append(i)
            parts = (
                (pair, table.take(indices).drop(["member_id", "resource_id"]))
                for pair, indices in pair_rows.items()
            )
        else:
            parts = iter((((None, None), table),))
        with native_data(scope=scope, source="dataset"):
            for (member, resource), rows in parts:
                metadata = declarations[(member, resource)]
                schema = dataset_arrow_schema(
                    _replace(metadata, report=ExecutionReport(False, None))
                )
                for row in table_to_rows(rows.replace_schema_metadata(schema.metadata)):
                    key = (member, (row.variable_id, row.scenario_id, row.period_id))
                    if key in groups:
                        grouped.setdefault(key, []).append(row)
                        resources[(key, row.support_id, row.cell_id)] = resource
    return grouped, resources


def _value(
    query: EvaluationQuery,
    value: float,
    series: SeriesSpec,
    row: DatasetRow | None,
    match_failure: str | None,
) -> tuple[float | None, EvaluationFailure | None]:
    if row is None:
        return None, EvaluationFailure(
            match_failure or "no_covering_support",
            "No source support matched this asset and series.",
        )
    if row.failure is not None:
        return None, EvaluationFailure(
            row.failure.code, row.failure.message, row.failure.record
        )
    assert row.curve is not None
    q = query.probability(value, series)
    if isinstance(row.curve, cf.QuantileCurve):
        lo, hi = row.curve.probability_interval
        if q < lo or q > hi:
            return None, EvaluationFailure(
                "query_outside_support",
                f"Probability {q:g} lies outside supplied interval [{lo:g}, {hi:g}].",
            )
    return cast(float, row.curve.quantile(q)), None


def _evaluation_batches(
    scope: ExecutionScope,
    rows: list[EvaluationRow],
    metadata: EvaluationMetadata,
    validator: EvaluationValidator,
) -> Iterator[pa.RecordBatch]:
    with native_data(scope=scope, source="evaluation"):
        for batch in evaluation_table(
            rows, metadata, include_metadata=False
        ).to_batches():
            validator.check(batch)
            scope.emit(Stage.EVALUATION, validator._produced)
            yield batch


class _Results:
    """Common result terminals with the shared execute and stream lifetimes."""

    __slots__ = ()

    @property
    def _node(self) -> Node[EvaluationMetadata]:
        raise NotImplementedError

    def _saved(
        self, path: Path, metadata: EvaluationMetadata, digest: str
    ) -> EvaluationArtifact:
        return EvaluationArtifact._create(path, metadata=metadata, digest=digest)

    def read(
        self, *, limit: int | None = None, runtime: Runtime | None = None
    ) -> ReadResult[EvaluationMetadata]:
        with Execution(self, runtime=runtime) as execution:
            return execution.read(limit=limit)

    def batches(
        self, *, batch_rows: int | None = None, runtime: Runtime | None = None
    ) -> BatchStream[EvaluationMetadata]:
        return BatchStream(ExecutionScope(runtime), self._node, batch_rows=batch_rows)

    def validate(self, *, runtime: Runtime | None = None) -> ExecutionReport:
        with Execution(self, runtime=runtime) as execution:
            return execution.validate()


class Evaluation(_Results):
    """An immutable recipe for assets, a query, and optional named responses."""

    __slots__ = ("_plan",)
    _plan: _Plan

    def __init__(self) -> None:
        raise TypeError("Create an evaluation with dataset.evaluate(assets)")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Evaluation recipes are immutable")

    @classmethod
    def _create(
        cls,
        source: Dataset | DatasetCollection,
        assets: Assets,
        spatial: SpatialMatch | None,
    ) -> Evaluation:
        if not isinstance(assets, Assets):
            raise TypeError("assets must be Assets")
        if spatial is not None and not isinstance(spatial, SpatialMatch):
            raise TypeError("spatial must be SpatialMatch")
        value = object.__new__(cls)
        object.__setattr__(
            value, "_plan", _Plan(source, assets, spatial or SpatialMatch.covering())
        )
        return value

    def _updated(self, plan: _Plan) -> Self:
        value = object.__new__(type(self))
        object.__setattr__(value, "_plan", plan)
        return value

    @property
    def _node(self) -> Node[EvaluationMetadata]:
        return _EvaluationNode(self._plan)

    def select(
        self,
        *,
        members: IdentifierSelector | types.EllipsisType = ...,
        variables: VariableSelector | types.EllipsisType = ...,
        scenarios: ScenarioSelector | types.EllipsisType = ...,
        periods: PeriodSelector | types.EllipsisType = ...,
        series: SeriesSelector | types.EllipsisType = ...,
        responses: IdentifierSelector | types.EllipsisType = ...,
        area: Bounds | None | types.EllipsisType = ...,
    ) -> Self:
        old = self._plan.selection
        state = SelectionState().updated(
            variables=variables,
            scenarios=scenarios,
            periods=periods,
            series=series,
            area=area,
        )
        return self._updated(
            _replace(
                self._plan,
                selection=_Selectors(
                    old.members if members is ... else _ids(members),
                    old.variables if variables is ... else state.variables,
                    old.scenarios if scenarios is ... else state.scenarios,
                    old.periods if periods is ... else state.periods,
                    old.series if series is ... else state.series,
                    old.responses if responses is ... else _ids(responses),
                    old.area if area is ... else state.area,
                ),
            )
        )

    def probabilities(self, values: Sequence[float]) -> Evaluation:
        return self._updated(
            _replace(
                self._plan, query=EvaluationQuery(QueryKind.PROBABILITY, tuple(values))
            )
        )

    def return_periods(self, values: Sequence[float]) -> Evaluation:
        return self._updated(
            _replace(
                self._plan,
                query=EvaluationQuery(QueryKind.RETURN_PERIOD, tuple(values)),
            )
        )

    def impact(
        self,
        name: str,
        model: cf.Impact | cf.ImpactReference,
        *,
        variable: Variable | str,
        context: cf.ImpactContext | None = None,
        replace: bool = False,
    ) -> Evaluation:
        if type(replace) is not bool:
            raise TypeError("replace must be bool")
        response = Response(name=name, variable=variable, model=model, context=context)
        items = {item.name: item for item in self._plan.responses}
        if name in items and not replace:
            raise ValueError(
                f"Response {name!r} already exists. Pass replace=True to replace it."
            )
        items[name] = response
        return self._updated(_replace(self._plan, responses=tuple(items.values())))

    def explain(self, *, runtime: Runtime | None = None) -> Explanation:
        source = self._plan._response_source().explain(runtime=runtime)
        operations = (
            *source.operations,
            Operation(Stage.JOINING, "Index selected source supports and match assets"),
        )
        query = self._plan.query
        if query is None:
            return _replace(
                source,
                operations=operations,
                requirements=(*source.requirements, _QUERY_REQUIRED),
            )
        return _replace(
            source,
            operations=(
                *operations,
                Operation(
                    Stage.EVALUATION,
                    f"Evaluate {len(query.values)} {query.kind.value} "
                    "coordinates per asset and series",
                ),
            ),
        )

    def inspect(self, *, runtime: Runtime | None = None) -> Inspection:
        plan = self._plan
        recipe = plan._response_source()
        responses = plan._selected_responses()
        with ExecutionScope(runtime) as scope:
            if plan._empty(recipe) and plan.query is not None:
                return Inspection(
                    schema=evaluation_arrow_schema(_empty_metadata(plan.query, plan)),
                    capabilities=(*_CAPABILITIES, Capability.WRITE),
                )
            try:
                source = (
                    recipe._inspect(
                        scope, projected_variables=plan._collection_projection(recipe)
                    )
                    if isinstance(recipe, DatasetCollection)
                    else recipe._inspect(scope)
                )
            except DataValidationError as error:
                raise ValidationError(
                    report=ExecutionReport(False, False, error.diagnostics)
                ) from error
            if plan.query is None:
                return _replace(
                    source,
                    schema=None,
                    capabilities=(Capability.INSPECT,),
                    requirements=(*source.requirements, _QUERY_REQUIRED),
                )
            declaration = source.metadata
            if Capability.READ not in source.capabilities or not isinstance(
                declaration,
                (DatasetMetadata, CollectionMetadata),
            ):
                return _replace(source, schema=None, capabilities=(Capability.INSPECT,))
            inputs = _inputs(declaration)
            try:
                groups = (
                    {}
                    if plan.selection.responses == ()
                    else _response_series(
                        _series(inputs),
                        responses,
                        exposure=plan.selection.responses is None,
                    )
                )
                definitions = _definitions(
                    tuple(groups.values()), responses, plan.query
                )
            except DataValidationError as error:
                raise ValidationError(
                    report=ExecutionReport(False, False, error.diagnostics)
                ) from error
            if not groups:
                metadata = _empty_metadata(plan.query, plan)
            else:
                asset_fields = _asset_fields(plan.assets)
                if asset_fields is None:
                    return _replace(
                        source,
                        schema=None,
                        capabilities=(*_CAPABILITIES, Capability.WRITE),
                        requirements=(*source.requirements, _ASSET_SCHEMA_UNRESOLVED),
                    )
                metadata = EvaluationMetadata(
                    query=plan.query,
                    inputs=_response_inputs(inputs, groups),
                    responses=definitions,
                    spatial=plan.spatial,
                    source=declaration.source,
                    include_exposure=plan.selection.responses is None,
                    asset_fields=asset_fields,
                )
            return _replace(
                source,
                schema=evaluation_arrow_schema(metadata),
                capabilities=(*_CAPABILITIES, Capability.WRITE),
            )

    def execute(
        self, *, runtime: Runtime | None = None
    ) -> Execution[EvaluationMetadata]:
        return Execution(self, runtime=runtime)

    def write(
        self,
        path: LocalPath,
        *,
        overwrite: bool = False,
        runtime: Runtime | None = None,
    ) -> EvaluationArtifact:
        with self.execute(runtime=runtime) as execution:
            return cast(EvaluationArtifact, execution.write(path, overwrite=overwrite))


class _EvaluationFile(Node[EvaluationMetadata]):
    def __init__(
        self, path: Path, expected: EvaluationMetadata | None, digest: str | None
    ) -> None:
        self.path, self.expected, self.digest = path, expected, digest
        self.key = _key("evaluation-parquet", str(path), digest)

    def open(self, scope: ExecutionScope) -> NodeStream[EvaluationMetadata]:
        handle = scope.enter_context(self.path.open("rb"))
        with native_data(scope=scope, source=str(self.path)):
            discovery = scope.enter_context(pq.ParquetFile(handle))
            schema = discovery.schema_arrow
            metadata = evaluation_metadata_from_schema(schema)
            if not metadata.report.complete or metadata.report.valid is not True:
                raise ValueError(
                    "The evaluation artifact requires a complete valid report"
                )
            if discovery.metadata.num_rows != metadata.report.counts.produced:
                raise ValueError(
                    "Stored evaluation counts differ from the Parquet row count"
                )
        if self.expected is not None and metadata.to_json() != self.expected.to_json():
            raise OSError("Evaluation artifact metadata changed after publication")
        validator = EvaluationValidator(metadata, scope_directory=scope.directory)
        scope.callback(validator.close)
        produced = no_data = 0

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal produced, no_data
            handle.seek(0)
            path, _ = scope.pin_handle(
                handle, uri=str(self.path), expected_sha256=self.digest
            )
            with native_data(scope=scope, source=str(self.path)):
                with pq.ParquetFile(path) as reader:
                    if (
                        not reader.schema_arrow.equals(schema, check_metadata=True)
                        or reader.metadata.num_rows != discovery.metadata.num_rows
                    ):
                        raise OSError(
                            "Evaluation artifact changed after metadata discovery"
                        )
                    if reader.num_row_groups:
                        for batch in reader.iter_batches(
                            batch_size=scope.runtime.batch_rows
                        ):
                            validator.check(batch, metadata)
                            produced += batch.num_rows
                            no_data += batch.column(
                                batch.schema.get_field_index("value")
                            ).null_count
                            yield batch
                    validator.finish(metadata)

        def report() -> ExecutionReport:
            stored = metadata.report
            return _replace(
                stored,
                complete=False,
                valid=None,
                counts=_replace(
                    stored.counts,
                    produced=produced,
                    no_data=no_data,
                ),
            )

        return NodeStream(schema, metadata, batches(), report=report)


class EvaluationArtifact(_Results):
    """A saved evaluation with its selected science and resolved model identities."""

    __slots__ = ("_file",)
    _file: _EvaluationFile

    def __init__(self) -> None:
        raise TypeError("Use EvaluationArtifact.open(path) or Evaluation.write(path)")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Evaluation artifacts are immutable")

    @classmethod
    def _create(
        cls,
        path: Path,
        *,
        metadata: EvaluationMetadata | None = None,
        digest: str | None = None,
    ) -> EvaluationArtifact:
        value = object.__new__(cls)
        object.__setattr__(value, "_file", _EvaluationFile(path, metadata, digest))
        return value

    @classmethod
    def open(cls, path: LocalPath) -> EvaluationArtifact:
        return cls._create(_local_path(path))

    @property
    def _node(self) -> Node[EvaluationMetadata]:
        return self._file

    @property
    def path(self) -> Path:
        return self._file.path

    @property
    def metadata(self) -> EvaluationMetadata:
        with ExecutionScope() as scope:
            return self._metadata(scope)

    def _metadata(self, scope: ExecutionScope) -> EvaluationMetadata:
        if self._file.expected is not None:
            return self._file.expected
        try:
            return scope.metadata(self._file)
        except DataValidationError as error:
            raise ValidationError(report=scope.report(self._file)) from error

    @property
    def report(self) -> ExecutionReport:
        return self.metadata.report

    def inspect(self, *, runtime: Runtime | None = None) -> Inspection:
        from ._inspection import SeriesReference

        with ExecutionScope(runtime) as scope:
            metadata = self._metadata(scope)
            return Inspection(
                schema=evaluation_arrow_schema(metadata),
                series=tuple(
                    SeriesReference(
                        s.variable,
                        scenario=s.scenario,
                        period=s.period,
                        probability=s.probability,
                        release=None
                        if item.metadata.source is None
                        else item.metadata.source.release,
                    )
                    for item in metadata.inputs
                    for s in item.metadata.spec.series
                ),
                capabilities=_CAPABILITIES,
                source=metadata.source,
            )
