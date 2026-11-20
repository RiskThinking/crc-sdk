"""Evaluation 2.0 metadata and validation for normalized Arrow results."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TypeAlias, cast

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]

from ._assets import _sql_materialized_array, _sql_materialized_type
from ._contextual_impact import ContextualImpact, ResponseContext
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionReport,
    SourceIdentity,
    Stage,
    _array,
    _boolean,
    _dump,
    _identifier,
    _load,
    _number,
    _object,
    _report,
    _source,
    _text,
)
from ._declarations import (
    DataSpec,
    Period,
    ProbabilityKind,
    Scenario,
    SeriesSpec,
    Variable,
    _merge_descriptor,
    scientific_identity,
)
from ._errors import CapabilityError
from ._evaluation_responses import (
    Model,
    _wire_context_values,
    bind_response,
    response_context,
)
from ._evaluation_spatial import (
    SpatialMatch,
    SpatialMatchKind,
    spatial_match_from_document,
)
from ._selection import SeriesKey, _series_key

_METADATA_KEY = b"crc.evaluation"
AssetId: TypeAlias = str | int
_SourceKey: TypeAlias = tuple[str | None, str | None]
_EvaluationSeries: TypeAlias = tuple[str | None, SeriesKey]

_UNMATCHED = frozenset({"no_covering_support", "outside_max_distance"})
_EXPOSURE_FAILURES_WITHOUT_FIT = _UNMATCHED | {"query_outside_support"}
_RELATIVE_RESULT_TOLERANCE = 1e-12


class QueryKind(Enum):
    """The coordinate supplied to an evaluation."""

    PROBABILITY = "probability"
    RETURN_PERIOD = "return_period"


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """An ordered collection of distinct query coordinates."""

    kind: QueryKind
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, QueryKind):
            raise TypeError("query kind must be QueryKind")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("query requires at least one coordinate")
        values = tuple(_number(value, "query value") for value in self.values)
        if len(set(values)) != len(values):
            raise ValueError("query values must be distinct")
        if self.kind is QueryKind.PROBABILITY and any(
            not 0 <= value <= 1 for value in values
        ):
            raise ValueError("probabilities must be in [0, 1]")
        if self.kind is QueryKind.RETURN_PERIOD and any(value < 1 for value in values):
            raise ValueError("return periods must be at least one year")
        object.__setattr__(self, "values", values)

    def _validate_series(self, series: SeriesSpec) -> None:
        probability = series.probability
        if self.kind is QueryKind.RETURN_PERIOD and (
            probability is None or probability.kind is not ProbabilityKind.ANNUAL
        ):
            raise CapabilityError(
                f"Return-period evaluation of {series.variable.id!r} requires "
                "declared annual probabilities with an explicit tail. "
                "Use declare(probability=Probability.annual(tail=...)) when "
                "the source science establishes that meaning."
            )
        if probability is not None and probability.kind in (
            ProbabilityKind.CONFIDENCE,
            ProbabilityKind.PROJECTION,
        ):
            raise CapabilityError(
                f"{series.variable.id!r} has {probability.kind.value} coordinates "
                "and cannot be evaluated as non-exceedance probabilities."
            )

    def probability(self, value: float, series: SeriesSpec) -> float:
        """Derive non-exceedance probability from this query and the series."""
        coordinate, tail = self._coordinate(value, series)
        return 1.0 - coordinate if tail is cf.Tail.UPPER else coordinate

    def _coordinate(self, value: float, series: SeriesSpec) -> tuple[float, cf.Tail]:
        """Keep the small tail directly representable through numerical inversion."""
        coordinate = _number(value, "query value")
        self._validate_series(series)
        if self.kind is QueryKind.PROBABILITY:
            if not 0 <= coordinate <= 1:
                raise ValueError("probabilities must be in [0, 1]")
            return coordinate, cf.Tail.LOWER
        if coordinate < 1:
            raise ValueError("return periods must be at least one year")
        assert series.probability is not None and series.probability.tail is not None
        return 1.0 / coordinate, series.probability.tail

    def _document(self) -> dict[str, object]:
        return {"kind": self.kind.value, "values": self.values}


@dataclass(frozen=True, slots=True)
class EvaluationInput:
    """One selected dataset or collection resource and its scientific metadata."""

    member_id: str | None
    resource_id: str | None
    metadata: DatasetMetadata

    def __post_init__(self) -> None:
        if (self.member_id is None) != (self.resource_id is None):
            raise ValueError(
                "collection evaluation inputs require member and resource IDs"
            )
        if self.member_id is not None:
            _identifier(self.member_id, "member_id")
            _identifier(self.resource_id, "resource_id")
        if not isinstance(self.metadata, DatasetMetadata):
            raise TypeError("evaluation inputs require DatasetMetadata")

    def _document(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "resource_id": self.resource_id,
            "metadata": self.metadata._document(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationResponse:
    """A user's named response and its calibrated model and context."""

    name: str
    variable_id: str
    model: cf.Impact | cf.ImpactReference | ContextualImpact
    context: cf.ImpactContext | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, "response name")
        _identifier(self.variable_id, "response variable_id")
        object.__setattr__(self, "context", response_context(self.model, self.context))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EvaluationResponse) and (
            self._document() == other._document()
        )

    @property
    def output(self) -> cf.Measurement:
        return (
            self.model.output
            if isinstance(self.model, cf.Impact)
            else self.model.requirements.output
        )

    def _document(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "variable_id": self.variable_id,
            "model": _load(self.model.to_json()),
        }
        if self.context is not None:
            result["context"] = _load(self.context.to_json())
        return result


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationResolution:
    """A calibrated response and the assets to which that resolution applied."""

    response_id: str
    impact: cf.Impact
    asset_ids: tuple[str | int, ...] = field(kw_only=True)
    context: cf.ImpactContext | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _identifier(self.response_id, "response_id")
        if not isinstance(self.impact, cf.Impact):
            raise TypeError("resolution requires a framework Impact")
        if not isinstance(self.asset_ids, tuple):
            raise TypeError("resolution asset_ids must be an immutable tuple")
        ids = tuple(_asset_id(item) for item in self.asset_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("resolution requires nonempty, unique asset IDs")
        configured = self.context
        if configured is not None and not isinstance(configured, cf.ImpactContext):
            raise TypeError("resolution context requires an ImpactContext")
        if isinstance(configured, ResponseContext):
            actual: cf.ImpactContext | None = ResponseContext.from_json(
                configured.to_json()
            )
        else:
            provenance = _load(self.impact.to_json()).get("provenance")
            actual = (
                None
                if provenance is None
                else cf.ImpactContext.from_json(
                    _dump(_object(provenance, {"model", "context"})["context"])
                )
            )
            if configured is not None and (
                actual is None or configured.to_json() != actual.to_json()
            ):
                raise ValueError("resolution context contradicts the framework Impact")
        object.__setattr__(self, "context", actual)

    @property
    def id(self) -> str:
        return _resolution_id(
            self.impact,
            self.context if isinstance(self.context, ResponseContext) else None,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EvaluationResolution) and (
            self.response_id == other.response_id
            and self.id == other.id
            and self.asset_ids == other.asset_ids
        )

    def _document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "response_id": self.response_id,
            "id": self.id,
            "impact": _load(self.impact.to_json()),
            "asset_ids": self.asset_ids,
        }
        if isinstance(self.context, ResponseContext):
            document["context"] = _load(self.context.to_json())
        return document


def _resolution_id(impact: cf.Impact, context: ResponseContext | None) -> str:
    if context is None:
        return impact.sha256
    evidence = _dump({"impact": impact.sha256, "context": _load(context.to_json())})
    return "sha256:" + hashlib.sha256(evidence.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    code: str
    message: str
    record: cf.FitRecord | None = None

    def __post_init__(self) -> None:
        _identifier(self.code, "failure code")
        _text(self.message, "failure message")
        if self.record is not None:
            if not isinstance(self.record, cf.FitRecord) or self.record.converged:
                raise ValueError(
                    "evaluation failure records must describe a failed fit"
                )
            object.__setattr__(
                self, "record", cf.FitRecord.from_json(self.record.to_json())
            )

    def _arrow(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "record": None if self.record is None else self.record.to_json(),
        }


def _apply_response(
    exposure: float, impact: cf.Impact | None
) -> tuple[float | None, EvaluationFailure | None]:
    if not math.isfinite(exposure):
        return None, EvaluationFailure(
            "nonfinite_exposure", "The impact requires a finite exposure value."
        )
    assert impact is not None
    try:
        return cast(float, impact.evaluate(exposure)), None
    except cf.ImpactError as error:
        return None, EvaluationFailure(error.code, error.message)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRow:
    asset_id: AssetId
    variable_id: str
    query_value: float
    member_id: str | None = None
    resource_id: str | None = None
    scenario_id: str | None = None
    period_id: str | None = None
    support_id: str | None = None
    distance_metres: float | None = None
    response_id: str | None = None
    resolution_id: str | None = None
    value: float | None = None
    failure: EvaluationFailure | None = None
    exposure: tuple[float | None, EvaluationFailure | None] | None = None
    asset: Mapping[str, object] | None = None

    def _arrow(self, *, asset_names: tuple[str, ...]) -> dict[str, object]:
        result = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"asset", "exposure", "failure"}
        }
        result["failure"] = None if self.failure is None else self.failure._arrow()
        if self.exposure is not None:
            value, failure = self.exposure
            result["exposure"] = {
                "value": value,
                "failure": None if failure is None else failure._arrow(),
            }
        if asset_names:
            if self.asset is None or set(self.asset) != set(asset_names):
                raise ValueError(
                    "row assets disagree with declared retained asset fields"
                )
            result["asset"] = self.asset
        elif self.asset is not None:
            raise ValueError("asset fields must be declared in evaluation metadata")
        return result


def _covering() -> SpatialMatch:
    return SpatialMatch.covering()


def _asset_id(value: object) -> AssetId:
    if isinstance(value, str):
        return _identifier(value, "asset_id")
    if type(value) is int and -(2**63) <= value < 2**64:
        return value
    raise ValueError("asset_id must be a nonempty string or a 64-bit integer")


def _asset_schema(schema: pa.Schema) -> pa.Schema:
    if not isinstance(schema, pa.Schema) or not len(schema):
        raise TypeError("asset_fields must be an Arrow schema beginning with asset_id")
    if (
        schema.names[0] != "asset_id"
        or schema.field(0).type not in (pa.string(), pa.int64(), pa.uint64())
        or schema.field(0).nullable
        or len(set(schema.names)) != len(schema.names)
    ):
        raise ValueError(
            "asset_fields requires a non-nullable string/int64/uint64 asset_id"
        )
    for name in schema.names:
        _identifier(name, "asset column")
    return schema.remove_metadata()


def _valid_asset_id(value: AssetId, type_: pa.DataType) -> bool:
    if pa.types.is_string(type_):
        return isinstance(value, str)
    if not isinstance(value, int):
        return False
    if pa.types.is_uint64(type_):
        return 0 <= value < 2**64
    return -(2**63) <= value < 2**63


def _optional_id(value: object, name: str) -> str | None:
    return None if value is None else _identifier(value, name)


def _typed_tuple(value: object, cls: type[object], name: str) -> None:
    if not isinstance(value, tuple) or not all(isinstance(v, cls) for v in value):
        raise TypeError(f"{name} must be an immutable tuple of {cls.__name__}")


def _resolution_context(
    resolution: EvaluationResolution, response: EvaluationResponse
) -> dict[str, object]:
    model = response.model
    if isinstance(model, cf.Impact):
        raise ValueError("resolved Impact responses do not need resolution records")
    requirements = model.requirements
    if (
        resolution.impact.input != requirements.input
        or resolution.impact.output != requirements.output
    ):
        raise ValueError(f"calibration contradicts response {response.name!r}")
    if isinstance(model, ContextualImpact):
        if not isinstance(resolution.context, ResponseContext):
            raise ValueError(
                f"resolution requires actual context for {response.name!r}"
            )
        if model.resolve(resolution.context).sha256 != resolution.impact.sha256:
            raise ValueError(
                f"resolution contradicts the calibrated case for {response.name!r}"
            )
        resolved = _wire_context_values(resolution.context)
    else:
        if isinstance(resolution.context, ResponseContext):
            raise ValueError(
                "built-in model resolution context belongs to its framework Impact"
            )
        document = _load(resolution.impact.to_json())
        provenance = _object(document.get("provenance"), {"model", "context"})
        if _dump(provenance["model"]) != model.to_json():
            raise ValueError(
                f"resolution does not belong to response {response.name!r}"
            )
        resolved = _load(_dump(provenance["context"]))
    context = response.context if response.context is not None else cf.EmptyContext()
    configured = _wire_context_values(context)
    if configured.keys() != resolved.keys() or any(
        resolved[name] != value
        for name, value in configured.items()
        if not isinstance(value, dict)
    ):
        raise ValueError(f"resolved context contradicts response {response.name!r}")
    return resolved


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationMetadata:
    """Typed query, selected science, calibration, source and execution facts."""

    query: EvaluationQuery
    inputs: tuple[EvaluationInput, ...] = ()
    responses: tuple[EvaluationResponse, ...] = ()
    resolutions: tuple[EvaluationResolution, ...] = ()
    report: ExecutionReport = field(
        default_factory=lambda: ExecutionReport(False, None)
    )
    source: SourceIdentity | None = None
    spatial: SpatialMatch = field(default_factory=_covering)
    asset_fields: pa.Schema | None = None
    asset_ids: tuple[str | int, ...] | None = None
    include_exposure: bool = True
    _spec: DataSpec = field(init=False, compare=False, repr=False)
    _series: Mapping[_EvaluationSeries, SeriesSpec] = field(
        init=False, compare=False, repr=False
    )
    _resource_series: Mapping[_SourceKey, frozenset[SeriesKey]] = field(
        init=False, compare=False, repr=False
    )
    _response_by_id: Mapping[str, EvaluationResponse] = field(
        init=False, compare=False, repr=False
    )
    _asset_resolutions: Mapping[tuple[AssetId, str], str] = field(
        init=False, compare=False, repr=False
    )
    _resolved_contexts: Mapping[tuple[str, str], dict[str, object]] = field(
        init=False, compare=False, repr=False
    )

    @property
    def schema_version(self) -> str:
        return "2.0"

    @property
    def spec(self) -> DataSpec:
        """The participating series and their scientific meaning."""
        return self._spec

    def _check_upstream_report(self) -> None:
        def revision(source: SourceIdentity) -> tuple[object, ...]:
            release = source.release
            return (
                source.provider,
                source.product,
                source.model,
                source.member,
                None if release is None else (release.id, release.sha256),
            )

        def resources(source: SourceIdentity) -> set[tuple[str, str]]:
            return {(resource.uri, resource.sha256) for resource in source.resources}

        def contains(
            expected: SourceIdentity, recorded: tuple[SourceIdentity, ...]
        ) -> bool:
            return any(
                revision(expected) == revision(source)
                and resources(expected) <= resources(source)
                for source in recorded
            )

        if self.source is not None and not contains(
            self.source, self.report.source_identities
        ):
            raise ValueError(
                "evaluation source version disagrees with recorded provenance"
            )
        diagnostics = {_dump(item._document()) for item in self.report.diagnostics}
        count_fields = ("skipped", "usable_years", "incomplete_years")
        inherited_counts = dict.fromkeys(count_fields, 0)
        for item in self.inputs:
            upstream = item.metadata.report
            if not upstream.complete or upstream.valid is not True:
                raise ValueError(
                    "complete evaluation requires complete validated input provenance"
                )
            if any(
                not contains(identity, self.report.source_identities)
                for identity in upstream.source_identities
            ):
                raise ValueError("evaluation report omits upstream source provenance")
            if any(
                _dump(d._document()) not in diagnostics for d in upstream.diagnostics
            ):
                raise ValueError("evaluation report omits upstream diagnostics")
            for name in count_fields:
                inherited_counts[name] += getattr(upstream.counts, name)
            source = item.metadata.source
            if source is None:
                continue
            local = upstream.source_identities
            if not contains(source, self.report.source_identities) or (
                local and not contains(source, local)
            ):
                raise ValueError(
                    "evaluation input source version disagrees with recorded "
                    f"provenance for {source.product!r}"
                )
        if any(
            getattr(self.report.counts, field) != count
            for field, count in inherited_counts.items()
        ):
            raise ValueError("evaluation report counts contradict its upstream reports")

    def __post_init__(self) -> None:
        if not isinstance(self.query, EvaluationQuery):
            raise TypeError("query must be EvaluationQuery")
        if not isinstance(self.report, ExecutionReport):
            raise TypeError("report must be ExecutionReport")
        if self.source is not None and not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity or None")
        if not isinstance(self.spatial, SpatialMatch):
            raise TypeError("spatial must be SpatialMatch")
        _boolean(self.include_exposure, "include_exposure")
        for name, cls in (
            ("inputs", EvaluationInput),
            ("responses", EvaluationResponse),
            ("resolutions", EvaluationResolution),
        ):
            _typed_tuple(getattr(self, name), cls, name)
        if self.asset_fields is not None:
            object.__setattr__(self, "asset_fields", _asset_schema(self.asset_fields))
        if self.asset_ids is not None:
            if not isinstance(self.asset_ids, tuple):
                raise TypeError("asset_ids must be an immutable tuple")
            asset_ids = tuple(_asset_id(item) for item in self.asset_ids)
            if len(set(asset_ids)) != len(asset_ids):
                raise ValueError("asset_ids must be unique")
            if asset_ids and (
                self.asset_fields is None
                or any(
                    not _valid_asset_id(item, self.asset_fields.field(0).type)
                    for item in asset_ids
                )
            ):
                raise ValueError(
                    "asset IDs disagree with the declared asset field type"
                )
        if self.report.complete and self.asset_ids is None:
            raise ValueError("a complete evaluation requires its requested asset IDs")
        self._validate_science()

    def _validate_science(self) -> None:
        variables: dict[str, Variable] = {}
        scenarios: dict[str, Scenario] = {}
        periods: dict[str, Period] = {}
        combined: dict[SeriesKey, SeriesSpec] = {}
        selected: dict[_EvaluationSeries, SeriesSpec] = {}
        resource_series: dict[_SourceKey, frozenset[SeriesKey]] = {}
        direct = any(source.member_id is None for source in self.inputs)
        if direct and len(self.inputs) != 1:
            raise ValueError("direct evaluation has exactly one input dataset")
        response_variables = {response.variable_id for response in self.responses}
        for source in self.inputs:
            key = source.member_id, source.resource_id
            if key in resource_series:
                raise ValueError(f"duplicate evaluation input {key!r}")
            for series in source.metadata.spec.series:
                if (
                    not self.include_exposure
                    and series.variable.id not in response_variables
                ):
                    continue
                identifier = _series_key(series)
                self.query._validate_series(series)
                _merge_descriptor(series.variable, variables)
                _merge_descriptor(series.scenario, scenarios)
                _merge_descriptor(series.period, periods)
                if identifier in combined and scientific_identity(
                    combined[identifier]
                ) != scientific_identity(series):
                    raise ValueError(
                        f"conflicting evaluation science for series {identifier!r}"
                    )
                combined.setdefault(identifier, series)
                selected.setdefault((source.member_id, identifier), series)
            resource_series[key] = frozenset(
                _series_key(series) for series in source.metadata.spec.series
            )
        responses: dict[str, EvaluationResponse] = {}
        for response in self.responses:
            if response.name in responses:
                raise ValueError(f"duplicate evaluation response {response.name!r}")
            variable = variables.get(response.variable_id)
            if variable is None or variable.measurement is None:
                raise ValueError(
                    f"response {response.name!r} requires measured series "
                    f"{response.variable_id!r}"
                )
            bind_response(response.model, variable.measurement)
            responses[response.name] = response
        contexts: dict[tuple[str, str], dict[str, object]] = {}
        asset_resolutions: dict[tuple[AssetId, str], str] = {}
        assets = set(self.asset_ids or ())
        for item in self.resolutions:
            resolution_key = item.response_id, item.id
            if resolution_key in contexts:
                raise ValueError(f"duplicate evaluation resolution {resolution_key!r}")
            if item.response_id not in responses:
                raise ValueError(
                    f"resolution references unknown response {item.response_id!r}"
                )
            contexts[resolution_key] = _resolution_context(
                item, responses[item.response_id]
            )
            for asset_id in item.asset_ids:
                if asset_id not in assets:
                    raise ValueError(
                        f"resolution references undeclared asset {asset_id!r}"
                    )
                assignment = (asset_id, item.response_id)
                if assignment in asset_resolutions:
                    raise ValueError(
                        "one asset response cannot resolve multiple actual contexts"
                    )
                asset_resolutions[assignment] = item.id
        object.__setattr__(
            self,
            "_spec",
            DataSpec(
                tuple(variables.values()),
                series=tuple(combined.values()),
                scenarios=tuple(scenarios.values()),
                periods=tuple(periods.values()),
            ),
        )
        object.__setattr__(self, "_series", selected)
        object.__setattr__(self, "_resource_series", resource_series)
        object.__setattr__(self, "_response_by_id", responses)
        object.__setattr__(self, "_asset_resolutions", asset_resolutions)
        object.__setattr__(self, "_resolved_contexts", contexts)

    def _document(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "query": self.query._document(),
            "inputs": [value._document() for value in self.inputs],
            "responses": [value._document() for value in self.responses],
            "resolutions": [value._document() for value in self.resolutions],
            "spatial": self.spatial._document(),
            "include_exposure": self.include_exposure,
            "report": self.report._document(),
        }
        if self.source is not None:
            result["source"] = self.source._document()
        if self.asset_fields is not None:
            result["asset_fields"] = base64.b64encode(
                self.asset_fields.serialize().to_pybytes()
            ).decode("ascii")
        if self.asset_ids is not None:
            result["asset_ids"] = self.asset_ids
        return result

    def to_json(self) -> str:
        return _dump(self._document())

    @classmethod
    def from_json(cls, text: str) -> EvaluationMetadata:
        doc = _object(
            _load(text),
            {
                "schema_version",
                "query",
                "inputs",
                "responses",
                "resolutions",
                "spatial",
                "include_exposure",
                "report",
            },
            {"source", "asset_fields", "asset_ids"},
        )
        if doc["schema_version"] != "2.0":
            raise ValueError("expected evaluation schema 2.0")
        query = _object(doc["query"], {"kind", "values"})
        asset_fields = None
        if "asset_fields" in doc:
            try:
                encoded = _identifier(doc["asset_fields"], "asset_fields")
                raw = base64.b64decode(encoded, validate=True)
                reader = pa.BufferReader(raw)
                asset_fields = pa.ipc.read_schema(reader)
                if reader.tell() != len(raw) or asset_fields.metadata:
                    raise ValueError(
                        "asset_fields must contain exactly one Arrow schema"
                    )
            except (binascii.Error, pa.ArrowException) as error:
                raise ValueError(
                    "asset_fields must be a base64 Arrow IPC schema"
                ) from error
        try:
            return cls(
                query=EvaluationQuery(
                    QueryKind(query["kind"]),
                    tuple(
                        _number(item, "query value")
                        for item in _array(query["values"], "values")
                    ),
                ),
                inputs=tuple(_input(item) for item in _array(doc["inputs"], "inputs")),
                responses=tuple(
                    _response(item) for item in _array(doc["responses"], "responses")
                ),
                resolutions=tuple(
                    _resolution(item)
                    for item in _array(doc["resolutions"], "resolutions")
                ),
                spatial=spatial_match_from_document(doc["spatial"]),
                include_exposure=_boolean(doc["include_exposure"], "include_exposure"),
                report=_report(doc["report"]),
                source=None if "source" not in doc else _source(doc["source"]),
                asset_fields=asset_fields,
                asset_ids=None
                if "asset_ids" not in doc
                else tuple(
                    _asset_id(item) for item in _array(doc["asset_ids"], "asset_ids")
                ),
            )
        except cf.ImpactError as error:
            raise ValueError(f"invalid evaluation model: {error}") from error


def _input(value: object) -> EvaluationInput:
    doc = _object(value, {"member_id", "resource_id", "metadata"})
    return EvaluationInput(
        _optional_id(doc["member_id"], "member_id"),
        _optional_id(doc["resource_id"], "resource_id"),
        DatasetMetadata.from_json(_dump(doc["metadata"])),
    )


def _response(value: object) -> EvaluationResponse:
    doc = _object(value, {"name", "variable_id", "model"}, {"context"})
    model = _load(_dump(doc["model"]))
    parsed: Model
    if model.get("kind") == "contextual":
        parsed = ContextualImpact.from_json(_dump(model))
    elif "id" in model:
        parsed = cf.ImpactReference.from_json(_dump(model))
    else:
        parsed = cf.Impact.from_json(_dump(model))
    context = None
    if "context" in doc:
        context_json = _dump(doc["context"])
        context = (
            ResponseContext.from_json(context_json)
            if isinstance(parsed, ContextualImpact)
            else cf.ImpactContext.from_json(context_json)
        )
    return EvaluationResponse(
        _identifier(doc["name"], "response name"),
        _identifier(doc["variable_id"], "response variable_id"),
        parsed,
        context,
    )


def _resolution(value: object) -> EvaluationResolution:
    doc = _object(value, {"id", "response_id", "impact", "asset_ids"}, {"context"})
    result = EvaluationResolution(
        _identifier(doc["response_id"], "response_id"),
        cf.Impact.from_json(_dump(doc["impact"])),
        asset_ids=tuple(
            _asset_id(item) for item in _array(doc["asset_ids"], "resolution asset_ids")
        ),
        context=None
        if "context" not in doc
        else ResponseContext.from_json(_dump(doc["context"])),
    )
    if doc["id"] != result.id:
        raise ValueError("evaluation resolution identity disagrees with the impact")
    return result


def evaluation_arrow_schema(
    metadata: EvaluationMetadata, *, include_metadata: bool = True
) -> pa.Schema:
    query_unit = "year" if metadata.query.kind is QueryKind.RETURN_PERIOD else "1"
    failure_type = pa.struct(
        [
            pa.field("code", pa.string(), nullable=False),
            pa.field("message", pa.string(), nullable=False),
            pa.field("record", pa.string()),
        ]
    )
    fields = [
        pa.field(
            "asset_id",
            pa.string()
            if metadata.asset_fields is None
            else metadata.asset_fields.field(0).type,
            nullable=False,
        ),
        pa.field("member_id", pa.string()),
        pa.field("resource_id", pa.string()),
        pa.field("variable_id", pa.string(), nullable=False),
        pa.field("scenario_id", pa.string()),
        pa.field("period_id", pa.string()),
        pa.field("support_id", pa.string()),
        pa.field("distance_metres", pa.float64(), metadata={b"crc.unit": b"m"}),
        pa.field("response_id", pa.string()),
        pa.field("resolution_id", pa.string()),
        pa.field(
            "query_value",
            pa.float64(),
            nullable=False,
            metadata={
                b"crc.query_kind": metadata.query.kind.value.encode(),
                b"crc.unit": query_unit.encode(),
            },
        ),
        pa.field(
            "value",
            pa.float64(),
            metadata={
                b"crc.measurement_by": (
                    b"member_id,variable_id,scenario_id,period_id,response_id"
                )
            },
        ),
        pa.field("failure", failure_type),
    ]
    if metadata.responses and not metadata.include_exposure:
        fields.append(
            pa.field(
                "exposure",
                pa.struct(
                    [pa.field("value", pa.float64()), pa.field("failure", failure_type)]
                ),
                nullable=False,
            )
        )
    if metadata.asset_fields is not None and len(metadata.asset_fields) > 1:
        fields.append(
            pa.field(
                "asset", pa.struct(list(metadata.asset_fields)[1:]), nullable=False
            )
        )
    headers = {_METADATA_KEY: metadata.to_json().encode()} if include_metadata else None
    return pa.schema(fields, metadata=headers)


def _same_fields(schema: pa.Schema, expected: pa.Schema) -> bool:
    """Check exact Arrow science tags and physical types after Parquet decoding."""
    if not schema.equals(expected, check_metadata=False):
        return False

    def metadata_equal(left: pa.Field, right: pa.Field) -> bool:
        return (left.metadata or {}) == (right.metadata or {}) and all(
            metadata_equal(left.type.field(i), right.type.field(i))
            for i in range(left.type.num_fields)
        )

    # Parquet gives list children the standard name "element". Their label
    # doesn't change the list type; retain and check every actual field tag.
    return all(
        metadata_equal(left, right)
        for left, right in zip(schema, expected, strict=True)
    )


def _parquet_asset_type(kind: pa.DataType) -> pa.DataType:
    """Account for the logical types the Parquet decoder normalizes."""
    if pa.types.is_timestamp(kind) and kind.unit == "s":
        return pa.timestamp("ms", kind.tz)
    if pa.types.is_time32(kind) and kind.unit == "s":
        return pa.time32("ms")
    if pa.types.is_date64(kind):
        return pa.date32()
    if pa.types.is_dictionary(kind):
        value = kind.value_type
        if pa.types.is_large_string(value):
            value = pa.string()
        elif pa.types.is_large_binary(value):
            value = pa.binary()
        if pa.types.is_string(value) or pa.types.is_binary(value):
            return pa.dictionary(kind.index_type, value, ordered=kind.ordered)
        return _parquet_asset_type(value)

    def convert(field: pa.Field) -> pa.Field:
        return field.with_type(_parquet_asset_type(field.type))

    if pa.types.is_struct(kind):
        return pa.struct([convert(field) for field in kind])
    if pa.types.is_list(kind):
        return pa.list_(convert(kind.value_field))
    if pa.types.is_large_list(kind):
        return pa.large_list(convert(kind.value_field))
    if pa.types.is_fixed_size_list(kind):
        return pa.list_(convert(kind.value_field), kind.list_size)
    if pa.types.is_list_view(kind):
        return pa.list_view(convert(kind.value_field))
    if pa.types.is_large_list_view(kind):
        return pa.large_list_view(convert(kind.value_field))
    if pa.types.is_map(kind):
        return pa.map_(
            convert(kind.key_field),
            convert(kind.item_field),
            keys_sorted=kind.keys_sorted,
        )
    return kind


def _check_evaluation_fields(
    schema: pa.Schema, expected: pa.Schema, *, parquet: bool = False
) -> None:
    if _same_fields(schema, expected):
        return
    if parquet and any(
        _same_fields(schema, _map_asset_field(candidate, _parquet_asset_type))
        for candidate in (expected, _evaluation_parquet_schema(expected))
    ):
        return
    raise ValueError("Arrow fields do not match canonical evaluation schema 2.0")


def _map_asset_field(
    schema: pa.Schema, convert: Callable[[pa.DataType], pa.DataType]
) -> pa.Schema:
    index = schema.get_field_index("asset")
    if index == -1:
        return schema
    field = schema.field(index)
    return schema.set(index, field.with_type(convert(field.type)))


def _evaluation_parquet_schema(schema: pa.Schema) -> pa.Schema:
    # Physical arrays make view types writable on every supported Arrow version.
    return _map_asset_field(schema, _sql_materialized_type)


def _restore_asset_type(array: pa.Array, target: pa.DataType) -> pa.Array:
    if array.type == target:
        return array
    if pa.types.is_dictionary(target):
        encoded = (
            array if pa.types.is_dictionary(array.type) else array.dictionary_encode()
        )
        return pa.DictionaryArray.from_arrays(
            encoded.indices.cast(target.index_type),
            encoded.dictionary.cast(target.value_type),
            ordered=target.ordered,
        )
    if pa.types.is_struct(target):
        return pa.StructArray.from_arrays(
            [
                _restore_asset_type(array.field(i), field.type)
                for i, field in enumerate(target)
            ],
            fields=list(target),
            mask=array.is_null(),
        )
    if (pa.types.is_list_view(target) or pa.types.is_large_list_view(target)) and (
        isinstance(array, (pa.ListArray, pa.LargeListArray))
    ):
        offsets = array.offsets
        view = (
            pa.ListViewArray if pa.types.is_list_view(target) else pa.LargeListViewArray
        )
        return view.from_arrays(
            offsets.slice(0, len(array)),
            pc.subtract(offsets.slice(1), offsets.slice(0, len(array))),
            _restore_asset_type(array.values, target.value_type),
            type=target,
            mask=array.is_null(),
        )
    if isinstance(
        array,
        (
            pa.ListArray,
            pa.LargeListArray,
            pa.FixedSizeListArray,
            pa.ListViewArray,
            pa.LargeListViewArray,
            pa.MapArray,
        ),
    ):
        return pa.Array.from_buffers(
            target,
            len(array),
            array.buffers()[: array.type.num_buffers],
            offset=array.offset,
            children=[_restore_asset_type(array.values, target.field(0).type)],
        )
    return array.cast(target)


def _convert_parquet_assets(
    batch: pa.RecordBatch, schema: pa.Schema, *, writing: bool = False
) -> pa.RecordBatch:
    index = schema.get_field_index("asset")
    if index == -1:
        return batch
    field = schema.field(index)
    assets = batch.column(index)
    if assets.type == field.type:
        return batch
    convert = _sql_materialized_array if writing else _restore_asset_type
    return batch.set_column(index, field, convert(assets, field.type))


def evaluation_metadata_from_schema(
    schema: pa.Schema, *, parquet: bool = False
) -> EvaluationMetadata:
    value = (schema.metadata or {}).get(_METADATA_KEY)
    if value is None:
        raise ValueError("canonical evaluation metadata is missing")
    metadata = EvaluationMetadata.from_json(value.decode("utf-8"))
    expected = evaluation_arrow_schema(metadata, include_metadata=False)
    _check_evaluation_fields(schema, expected, parquet=parquet)
    return metadata


def evaluation_table(
    rows: Iterable[EvaluationRow],
    metadata: EvaluationMetadata,
    *,
    include_metadata: bool = True,
    retained: pa.StructArray | None = None,
) -> pa.Table:
    schema = evaluation_arrow_schema(metadata, include_metadata=include_metadata)
    if retained is not None:
        field = schema.field("asset")
        base = pa.Table.from_pylist(
            [row._arrow(asset_names=()) for row in rows],
            schema=schema.remove(schema.get_field_index("asset")),
        )
        return base.append_column(field, retained)
    asset_names = (
        () if metadata.asset_fields is None else tuple(metadata.asset_fields.names[1:])
    )
    return pa.Table.from_pylist(
        [row._arrow(asset_names=asset_names) for row in rows], schema=schema
    )


def _read_failure(value: object) -> EvaluationFailure | None:
    if value is None:
        return None
    data = _object(value, {"code", "message", "record"})
    record = data["record"]
    if record is not None and not isinstance(record, str):
        raise ValueError("evaluation failure fit record must be canonical JSON")
    return EvaluationFailure(
        _identifier(data["code"], "failure code"),
        _text(data["message"], "failure message"),
        None if record is None else cf.FitRecord.from_json(record),
    )


def _validate_match(
    row: Mapping[str, object],
    metadata: EvaluationMetadata,
    member: str | None,
    series: SeriesKey,
    failure: EvaluationFailure | None,
) -> tuple[str | None, str | None, float | None]:
    support = _optional_id(row["support_id"], "support_id")
    resource = _optional_id(row["resource_id"], "resource_id")
    distance = row["distance_metres"]
    if support is None:
        if resource is not None or distance is not None:
            raise ValueError("an unmatched asset cannot have a resource or distance")
        expected = (
            "no_covering_support"
            if metadata.spatial.kind is SpatialMatchKind.COVERING
            else "outside_max_distance"
        )
        if failure is None or failure.code != expected or failure.record is not None:
            raise ValueError("an unmatched asset requires its spatial failure reason")
        return None, None, None
    if series not in metadata._resource_series.get((member, resource), ()):
        raise ValueError("matched resource does not declare the evaluation series")
    if failure is not None and failure.code in _UNMATCHED:
        raise ValueError("matched support cannot carry an unmatched failure reason")
    if metadata.spatial.kind is SpatialMatchKind.COVERING:
        if distance is not None:
            raise ValueError("covering matches do not carry a distance")
        return support, resource, None
    value = _number(distance, "distance_metres")
    assert metadata.spatial.max_distance is not None
    if not 0 <= value <= metadata.spatial.max_distance.value_metres:
        raise ValueError("nearest match distance exceeds the selected bound")
    return support, resource, value


def _validate_exposure_failure(failure: EvaluationFailure | None) -> None:
    if failure is None:
        return
    requires_fit = failure.code not in _EXPOSURE_FAILURES_WITHOUT_FIT
    if (failure.record is not None) != requires_fit:
        raise ValueError(
            "exposure failure must describe a spatial match, query, or failed fit"
        )


def _validate_response(
    row: Mapping[str, object],
    metadata: EvaluationMetadata,
    variable: str,
    failure: EvaluationFailure | None,
    contexts: Mapping[tuple[str, str], Mapping[str, object]],
    asset: pa.StructScalar | None,
) -> tuple[str | None, str | None]:
    response_id = _optional_id(row["response_id"], "response_id")
    resolution_id = _optional_id(row["resolution_id"], "resolution_id")
    if response_id is None:
        if not metadata.include_exposure:
            raise ValueError("exposure was excluded from the evaluation")
        if resolution_id is not None:
            raise ValueError("an exposure row cannot reference a model resolution")
        _validate_exposure_failure(failure)
        return None, None
    response = metadata._response_by_id.get(response_id)
    if response is None or response.variable_id != variable:
        raise ValueError("row does not reference an applicable evaluation response")
    if isinstance(response.model, cf.Impact):
        if resolution_id is not None:
            raise ValueError("a resolved response does not require a resolution ID")
        return response_id, None
    if resolution_id is None:
        if failure is None:
            raise ValueError(
                "successful referenced response rows require a resolution ID"
            )
        return response_id, None
    key = response_id, resolution_id
    actual = contexts.get(key)
    if actual is None:
        raise ValueError("row references an unknown response resolution")
    if row["support_id"] is None or (
        failure is not None and failure.record is not None
    ):
        raise ValueError("a response cannot resolve without evaluated source evidence")
    if response.context is not None and asset is not None:
        for field, value in _wire_context_values(response.context).items():
            if not isinstance(value, dict):
                continue
            binding = value.get("column")
            if not isinstance(binding, str) or binding not in asset:
                continue
            supplied = asset[binding].as_py()
            if isinstance(supplied, Decimal):
                supplied = float(supplied)
            if supplied != actual[field] or isinstance(supplied, bool):
                raise ValueError("row assets contradict the recorded response context")
    return response_id, resolution_id


class EvaluationValidator:
    """Verify row references, the ordered grid, and saved response calculations."""

    def __init__(
        self, metadata: EvaluationMetadata, *, scope_directory: Path | None = None
    ) -> None:
        import tempfile

        self._metadata = metadata
        self._fields = evaluation_arrow_schema(metadata, include_metadata=False)
        self._coordinates = {
            value: index for index, value in enumerate(metadata.query.values)
        }
        self._ranks = {
            value: rank for rank, value in enumerate(sorted(metadata.query.values))
        }
        self._assets = frozenset(metadata.asset_ids or ())
        self._contexts = dict(metadata._resolved_contexts)
        # Live rows already execute the response kernel. Verify it on replay
        # against either visible exposure rows or a retained exposure outcome.
        self._response_variables = (
            frozenset(response.variable_id for response in metadata.responses)
            if metadata.report.complete
            else frozenset()
        )
        self._produced = 0
        self._no_data = 0
        self._failure_codes: set[str] = set()
        self._stored_failure_details = {
            (d.code, d.message, d.support_id, d.query_value)
            for d in metadata.report.diagnostics
            if metadata.report.complete
            and d.stage is Stage.EVALUATION
            and d.severity is cf.DiagnosticSeverity.WARNING
        }
        self._verified_failure_codes: set[str] = set()
        self._path: Path | None = None
        self._closed = False
        if scope_directory is not None:
            with tempfile.NamedTemporaryFile(
                suffix=".sqlite",
                prefix="evaluation-",
                dir=scope_directory,
                delete=False,
            ) as file:
                self._path = Path(file.name)
        self._db = sqlite3.connect(":memory:" if self._path is None else self._path)
        self._db.executescript(
            """
            CREATE TABLE queries (identity TEXT PRIMARY KEY, next INTEGER NOT NULL);
            CREATE TABLE matches (identity TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE matched_assets (identity TEXT PRIMARY KEY);
            CREATE TABLE retained_assets (
                identity TEXT PRIMARY KEY, digest BLOB NOT NULL
            );
            CREATE TABLE contexts (identity TEXT PRIMARY KEY, resolution TEXT NOT NULL);
            CREATE TABLE source_supports (
                identity TEXT PRIMARY KEY, member TEXT, resource TEXT, fit_failure TEXT
            );
            CREATE TABLE source_exposures (
                support TEXT NOT NULL, query_rank INTEGER NOT NULL,
                value REAL, failure TEXT,
                PRIMARY KEY (support, query_rank)
            );
            CREATE TABLE outcomes (
                event TEXT NOT NULL,
                response TEXT NOT NULL,
                resolution TEXT,
                value REAL,
                failure TEXT,
                PRIMARY KEY (event, response)
            );
            """
        )

    def close(self) -> None:
        if not self._closed:
            self._db.close()
            if self._path is not None:
                self._path.unlink(missing_ok=True)
            self._closed = True

    def _same_request(self, metadata: EvaluationMetadata) -> None:
        original = self._metadata
        if (
            metadata.query != original.query
            or metadata.spatial != original.spatial
            or metadata.asset_ids != original.asset_ids
            or metadata.asset_fields != original.asset_fields
            or metadata.include_exposure != original.include_exposure
            or metadata._resource_series != original._resource_series
            or metadata.responses != original.responses
        ):
            raise ValueError("evaluation request changed during execution")
        if metadata is not original and any(
            scientific_identity(metadata._series[key]) != scientific_identity(series)
            for key, series in original._series.items()
        ):
            raise ValueError("evaluation input science changed during execution")

    def register_resolution(self, item: EvaluationResolution) -> None:
        key = item.response_id, item.id
        if key not in self._contexts:
            response = self._metadata._response_by_id.get(item.response_id)
            if response is None:
                raise ValueError("resolution references an unknown evaluation response")
            self._contexts[key] = _resolution_context(item, response)

    def check(
        self, batch: pa.RecordBatch, metadata: EvaluationMetadata | None = None
    ) -> pa.RecordBatch:
        current = self._metadata if metadata is None else metadata
        if metadata is not None and metadata is not self._metadata:
            self._same_request(metadata)
            for item in metadata.resolutions:
                self.register_resolution(item)
        if not isinstance(batch, pa.RecordBatch) or not _same_fields(
            batch.schema, self._fields
        ):
            raise ValueError(
                "Arrow fields do not match canonical evaluation schema 2.0"
            )
        index = batch.schema.get_field_index("asset")
        if index == -1:
            for row in batch.to_pylist():
                self._check_row(row, current, None)
        else:
            assets = batch.column(index)
            values = batch.select(
                [name for name in batch.schema.names if name != "asset"]
            )
            for i, row in enumerate(values.to_pylist()):
                self._check_row(row, current, assets[i])
        return batch

    def _check_retained_asset(self, identity: AssetId, asset: pa.StructScalar) -> None:
        # Rebox one Arrow scalar to normalize physical encodings such as
        # a shared dictionary while preserving exact nested and temporal values.
        field = self._fields.field("asset")
        schema = pa.schema([field])
        row = pa.RecordBatch.from_arrays(
            [pa.array([asset], type=field.type)], schema=schema
        )
        with pa.BufferOutputStream() as sink:
            with pa.ipc.new_stream(sink, schema) as writer:
                writer.write_batch(row)
            digest = hashlib.sha256(sink.getvalue()).digest()
        key = _dump(identity)
        self._db.execute(
            "INSERT OR IGNORE INTO retained_assets VALUES (?, ?)", (key, digest)
        )
        saved = self._db.execute(
            "SELECT digest FROM retained_assets WHERE identity = ?", (key,)
        ).fetchone()
        if saved != (digest,):
            raise ValueError(
                f"retained attributes disagree across rows for asset {identity!r}"
            )

    def _check_source_exposure(
        self,
        key: tuple[str | None, SeriesKey, str],
        resource: str | None,
        rank: int,
        value: object,
        failure: EvaluationFailure | None,
    ) -> None:
        identity = _dump(key)
        outcome = None if failure is None else _dump(failure._arrow())
        fit_failure = (
            outcome if failure is not None and failure.record is not None else None
        )
        support = (resource, fit_failure)
        self._db.execute(
            "INSERT OR IGNORE INTO source_supports VALUES (?, ?, ?, ?)",
            (identity, key[0], *support),
        )
        saved = self._db.execute(
            "SELECT resource, fit_failure FROM source_supports WHERE identity = ?",
            (identity,),
        ).fetchone()
        if saved != support:
            raise ValueError("source support has inconsistent resource or fit failure")
        self._db.execute(
            "INSERT OR IGNORE INTO source_exposures VALUES (?, ?, ?, ?)",
            (identity, rank, value, outcome),
        )
        saved = self._db.execute(
            "SELECT value, failure FROM source_exposures "
            "WHERE support = ? AND query_rank = ?",
            (identity, rank),
        ).fetchone()
        if saved != (value, outcome):
            raise ValueError("exposures disagree for the same source support and query")

    def _check_row(
        self,
        row: Mapping[str, object],
        metadata: EvaluationMetadata,
        asset: pa.StructScalar | None,
    ) -> None:
        asset_id = _asset_id(row["asset_id"])
        if asset_id not in self._assets:
            raise ValueError(f"row references undeclared asset {asset_id!r}")
        member = _optional_id(row["member_id"], "member_id")
        variable = _identifier(row["variable_id"], "variable_id")
        series = (
            variable,
            _optional_id(row["scenario_id"], "scenario_id"),
            _optional_id(row["period_id"], "period_id"),
        )
        if (member, series) not in metadata._series:
            raise ValueError(
                f"row references undeclared evaluation series {(member, series)!r}"
            )
        query = _number(row["query_value"], "query_value")
        query_position = self._coordinates.get(query)
        if query_position is None:
            raise ValueError(f"row references unrequested query {query}")
        failure = _read_failure(row["failure"])
        value = row["value"]
        if failure is None:
            if not isinstance(value, float) or math.isnan(value):
                raise ValueError(
                    "a successful result requires a numerical value other than NaN"
                )
        elif value is not None:
            raise ValueError("an evaluation failure cannot contain a numerical value")
        if metadata.asset_fields is not None and len(metadata.asset_fields) > 1:
            if (
                asset is None
                or not asset.is_valid
                or any(
                    not field.nullable and not asset[field.name].is_valid
                    for field in list(metadata.asset_fields)[1:]
                )
            ):
                raise ValueError(
                    "evaluation row is missing declared retained asset fields"
                )
            if metadata.report.complete:
                self._check_retained_asset(asset_id, asset)
        match = _validate_match(row, metadata, member, series, failure)
        response, resolution = _validate_response(
            row, metadata, variable, failure, self._contexts, asset
        )
        if (
            response is not None
            and isinstance(value, float)
            and not math.isfinite(value)
        ):
            raise ValueError("a successful named response requires a finite value")
        exposure_value = None
        exposure_failure = None
        if not metadata.include_exposure:
            exposure = _object(row["exposure"], {"value", "failure"})
            exposure_value = exposure["value"]
            exposure_failure = _read_failure(exposure["failure"])
            if (
                exposure_failure is None
                and (
                    not isinstance(exposure_value, float) or math.isnan(exposure_value)
                )
                or exposure_failure is not None
                and exposure_value is not None
            ):
                raise ValueError("a retained exposure requires one value or failure")
            _validate_match(row, metadata, member, series, exposure_failure)
            _validate_exposure_failure(exposure_failure)
        if (
            metadata.report.complete
            and (support := match[0]) is not None
            and (response is None or not metadata.include_exposure)
        ):
            rank = self._ranks[query]
            probability = metadata._series[member, series].probability
            if (
                metadata.query.kind is QueryKind.RETURN_PERIOD
                and probability is not None
                and probability.tail is cf.Tail.LOWER
            ):
                # A longer lower-tail return period has a smaller probability.
                rank = -rank
            self._check_source_exposure(
                (member, series, support),
                match[1],
                rank,
                value if response is None else exposure_value,
                failure if response is None else exposure_failure,
            )
        match_key = _dump((asset_id, member, *series))
        match_record = _dump(match)
        self._db.execute(
            "INSERT OR IGNORE INTO matches VALUES (?, ?)", (match_key, match_record)
        )
        previous_match = self._db.execute(
            "SELECT record FROM matches WHERE identity = ?", (match_key,)
        ).fetchone()
        if previous_match is None or previous_match[0] != match_record:
            raise ValueError("an asset's matched support changed for the same series")
        query_key = _dump((asset_id, member, *series, response))
        if query_position == 0:
            try:
                self._db.execute("INSERT INTO queries VALUES (?, 1)", (query_key,))
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate evaluation row identity") from error
        else:
            updated = self._db.execute(
                "UPDATE queries SET next = next + 1 WHERE identity = ? AND next = ?",
                (query_key, query_position),
            )
            if updated.rowcount != 1:
                raise ValueError(
                    "evaluation queries are duplicated, missing, or out of order"
                )
        if match[0] is not None:
            self._db.execute(
                "INSERT OR IGNORE INTO matched_assets VALUES (?)", (_dump(asset_id),)
            )
        if response is not None and resolution is not None:
            # Live execution verifies its final assignments at finish, while
            # replay can check against its published resolution inventory now.
            has_record = (response, resolution) in metadata._resolved_contexts
            assigned = metadata._asset_resolutions.get((asset_id, response))
            if has_record and assigned != resolution:
                raise ValueError(
                    f"asset {asset_id!r} did not use the recorded response resolution"
                )
            key = _dump((asset_id, response))
            previous = self._db.execute(
                "SELECT resolution FROM contexts WHERE identity = ?", (key,)
            ).fetchone()
            if previous is None:
                self._db.execute(
                    "INSERT INTO contexts VALUES (?, ?)", (key, resolution)
                )
            elif previous[0] != resolution:
                raise ValueError(
                    "one asset response cannot resolve multiple actual contexts"
                )
        if variable in self._response_variables:
            event = _dump((asset_id, member, *series, query_position))
            if not metadata.include_exposure:
                proof = (
                    exposure_value,
                    None
                    if exposure_failure is None
                    else _dump(exposure_failure._arrow()),
                )
                self._db.execute(
                    "INSERT OR IGNORE INTO outcomes VALUES (?, '', NULL, ?, ?)",
                    (event, *proof),
                )
                saved = self._db.execute(
                    "SELECT value, failure FROM outcomes "
                    "WHERE event = ? AND response = ''",
                    (event,),
                ).fetchone()
                if saved != proof:
                    raise ValueError(
                        f"retained exposures disagree across responses for {event}"
                    )
            self._db.execute(
                "INSERT INTO outcomes VALUES (?, ?, ?, ?, ?)",
                (
                    event,
                    response or "",
                    resolution,
                    value,
                    None if failure is None else _dump(failure._arrow()),
                ),
            )
        self._produced += 1
        self._no_data += int(failure is not None)
        if failure is not None:
            self._failure_codes.add(failure.code)
            if (
                failure.code,
                failure.message,
                match[0],
                query,
            ) in self._stored_failure_details:
                self._verified_failure_codes.add(failure.code)

    def _check_exposure_grid(self, metadata: EvaluationMetadata) -> None:
        declared_counts = {
            (item.member_id, item.resource_id): item.metadata.report.counts
            for item in metadata.inputs
        }
        for member, resource, total, failed in self._db.execute(
            "SELECT member, resource, count(*), count(fit_failure) "
            "FROM source_supports GROUP BY member, resource"
        ):
            counts = declared_counts[member, resource]
            if total > counts.produced:
                raise ValueError(
                    "evaluation matched supports exceed the input row count"
                )
            if failed > counts.no_data:
                raise ValueError(
                    "evaluation fit failures exceed the input no-data count"
                )
            if total - failed > counts.produced - counts.no_data:
                raise ValueError(
                    "evaluation supports with curves exceed the available input rows"
                )

        current = None
        largest: float | None = None
        ended = False
        for support, value in self._db.execute(
            "SELECT exposure.support, exposure.value FROM source_exposures AS exposure "
            "JOIN source_supports ON exposure.support = source_supports.identity "
            "WHERE source_supports.fit_failure IS NULL "
            "ORDER BY exposure.support, exposure.query_rank"
        ):
            if support != current:
                current, largest, ended = support, None, False
            if value is None:
                ended = largest is not None
                continue
            if ended:
                raise ValueError(
                    f"query support has a gap for source support {support}"
                )
            if (
                largest is not None
                and value < largest
                and not math.isclose(value, largest, rel_tol=_RELATIVE_RESULT_TOLERANCE)
            ):
                raise ValueError(
                    f"exposure quantiles decrease for source support {support}"
                )
            largest = value if largest is None else max(largest, value)

    def _check_response_values(self, metadata: EvaluationMetadata) -> None:
        impacts = {
            (item.response_id, item.id): item.impact for item in metadata.resolutions
        }
        pairs = self._db.execute(
            """
            SELECT result.event, result.response, result.resolution, result.value,
                result.failure, exposure.value, exposure.failure
            FROM outcomes AS result
            JOIN outcomes AS exposure ON exposure.event = result.event
            WHERE result.response != '' AND exposure.response = ''
            """
        )
        for event, name, resolution, value, failure, exposure, source_failure in pairs:
            if source_failure is not None:
                if failure != source_failure:
                    raise ValueError(
                        f"response {name!r} contradicts its exposure failure "
                        f"for {event}"
                    )
                continue
            model = metadata._response_by_id[name].model
            impact = (
                model
                if isinstance(model, cf.Impact)
                else impacts.get((name, resolution))
            )
            if impact is None and math.isfinite(exposure):
                raise ValueError(
                    f"response {name!r} lacks the resolution for its finite exposure "
                    f"for {event}"
                )
            expected, problem = _apply_response(exposure, impact)
            if problem is not None:
                actual = {} if failure is None else _load(failure)
                matches = (
                    actual.get("code") == problem.code and actual.get("record") is None
                )
            else:
                assert expected is not None
                matches = value is not None and math.isclose(
                    value, expected, rel_tol=_RELATIVE_RESULT_TOLERANCE
                )
            if not matches:
                raise ValueError(
                    f"response {name!r} contradicts its model and exposure for {event}"
                )

    def finish(self, metadata: EvaluationMetadata | None = None) -> None:
        current = self._metadata if metadata is None else metadata
        self._same_request(current)
        if current.asset_ids is None:
            raise ValueError("evaluation cannot complete without requested asset IDs")
        result_series = sum(
            int(current.include_exposure)
            + sum(response.variable_id == key[1][0] for response in current.responses)
            for key in current._series
        )
        expected = len(self._assets) * result_series * len(current.query.values)
        if expected != self._produced:
            raise ValueError(
                f"incomplete evaluation query grid: expected {expected} rows, "
                f"found {self._produced}"
            )
        used = self._db.execute("SELECT count(*) FROM contexts").fetchone()[0]
        if used != len(current._asset_resolutions):
            raise ValueError("evaluation resolution assignments disagree with its rows")
        for key, resolution in current._asset_resolutions.items():
            actual = self._db.execute(
                "SELECT resolution FROM contexts WHERE identity = ?", (_dump(key),)
            ).fetchone()
            if actual is None or actual[0] != resolution:
                raise ValueError(
                    "evaluation resolution assignments disagree with its rows"
                )
        if self._metadata.report.complete:
            self._check_exposure_grid(current)
        if self._response_variables:
            self._check_response_values(current)
        if current.report.complete:
            matched = self._db.execute(
                "SELECT count(*) FROM matched_assets"
            ).fetchone()[0]
            unmatched = len(self._assets) - matched if result_series else 0
            counts = current.report.counts
            if (
                counts.requested != expected
                or counts.produced != self._produced
                or counts.no_data != self._no_data
                or counts.matched_assets != matched
                or counts.unmatched_assets != unmatched
            ):
                raise ValueError("evaluation report counts contradict its result rows")
            diagnostics = (
                self._verified_failure_codes
                if self._metadata.report.complete
                else {
                    d.code
                    for d in current.report.diagnostics
                    if d.stage is Stage.EVALUATION
                    and d.severity is cf.DiagnosticSeverity.WARNING
                }
            )
            if missing := self._failure_codes - diagnostics:
                raise ValueError(
                    f"evaluation report omits diagnostics for {sorted(missing)!r}"
                )
            current._check_upstream_report()


def validate_evaluation(
    table: pa.Table, metadata: EvaluationMetadata | None = None
) -> EvaluationMetadata:
    """Validate an owned table, including the requested grid when complete."""
    if not isinstance(table, pa.Table):
        raise TypeError("table must be an Arrow Table")
    stored = evaluation_metadata_from_schema(table.schema)
    if metadata is not None and metadata.to_json() != stored.to_json():
        raise ValueError("supplied and stored evaluation metadata disagree")
    validator = EvaluationValidator(stored)
    try:
        for batch in table.to_batches():
            validator.check(batch)
        if stored.report.complete:
            validator.finish()
    finally:
        validator.close()
    return stored
