"""Bind measurements and resolve typed response contexts using the framework."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeAlias, cast

import crc_framework as cf

from ._contextual_impact import ContextualImpact, ResponseContext
from ._dataset_schema import _dump, _identifier, _load
from ._declarations import Variable

Model: TypeAlias = cf.Impact | cf.ImpactReference | ContextualImpact
ResolutionCache: TypeAlias = MutableMapping[tuple[str, str], cf.Impact]


def response_context(
    model: Model, context: cf.ImpactContext | None
) -> cf.ImpactContext | None:
    """Check the declared context and own its column-binding names."""
    if isinstance(model, ContextualImpact) and type(model) is ContextualImpact:
        return model._context(context)
    if isinstance(model, cf.Impact):
        if type(model) is not cf.Impact:
            raise TypeError("use an exact framework Impact for a resolved response")
        if context is not None:
            raise cf.ImpactError(
                "unexpected_context",
                "/context",
                "A resolved Impact already has its calibration. "
                "Supply context with an ImpactReference.",
            )
        return None
    if (
        not isinstance(model, cf.ImpactReference)
        or type(model) is not cf.ImpactReference
    ):
        raise TypeError(
            "model must be a calibrated Impact, ImpactReference, or ContextualImpact"
        )
    requirements = model.requirements
    constructor = requirements.context_type
    if context is None and constructor is cf.EmptyContext:
        return None
    if type(context) is not constructor:
        fields = ", ".join(field.name + "=..." for field in requirements.context_fields)
        raise cf.ImpactError(
            "missing_context" if context is None else "invalid_context",
            "/context",
            f"Use {constructor.__name__}({fields}). "
            "Supply typed constants or assets.column(...) for its fields.",
        )
    assert context is not None
    return cf.ImpactContext.from_json(context.to_json())


def bind_response(model: Model, measurement: cf.Measurement | None) -> None:
    """Validate complete source science against the calibrated input."""
    if measurement is None:
        expected = (
            model.input if isinstance(model, cf.Impact) else model.requirements.input
        )
        raise cf.ImpactError(
            "missing_measurement",
            "/measurement",
            f"This response requires {expected!r}. "
            "Declare the input measurement with Variable(..., measurement=...).",
        )
    if isinstance(model, ContextualImpact):
        model.requirements.bind(measurement)
        for impact in model._cases.values():
            bind_response(impact, measurement)
        return
    if isinstance(model, cf.ImpactReference):
        model.requirements.bind(measurement)
        return
    # A resolved published response retains its model proof. Binding the
    # numerical snapshot still entails that model's scientific requirements.
    proof = _load(model.to_json()).get("provenance")
    if isinstance(proof, dict):
        reference = cf.ImpactReference.from_json(_dump(proof["model"]))
        reference.requirements.bind(measurement)
    elif measurement != model.input:
        same_quantity = measurement.quantity == model.input.quantity
        incompatible_unit = same_quantity and measurement.unit != model.input.unit
        advice = (
            " Convert numerical values with convert_units(variable=..., unit=...)."
            if incompatible_unit
            else ""
        )
        raise cf.ImpactError(
            "unit_mismatch" if incompatible_unit else "measurement_mismatch",
            "/measurement/unit" if incompatible_unit else "/measurement",
            f"Requires {model.input!r}, received {measurement!r}.{advice}",
        )


def _context_bindings(
    document: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field, cast(str, value["column"]))
        for field, value in document.items()
        if isinstance(value, dict)
    )


def _wire_context_values(context: cf.ImpactContext) -> dict[str, object]:
    document = _load(context.to_json())
    return (
        _load(_dump(document["values"]))
        if isinstance(context, ResponseContext)
        else document
    )


def context_columns(context: cf.ImpactContext | None) -> tuple[str, ...]:
    if context is None:
        return ()
    bindings = _context_bindings(_wire_context_values(context))
    return tuple(dict.fromkeys(column for _, column in bindings))


def _concrete_context(
    context: cf.ImpactContext,
    asset: Mapping[str, object],
) -> cf.ImpactContext:
    document = _wire_context_values(context)
    bindings = _context_bindings(document)
    if not bindings:
        return context
    for field, column in bindings:
        if column not in asset:
            raise cf.ImpactError(
                "missing_context_column",
                "/context/" + field,
                f"Asset column {column!r} was bound with assets.column({column!r}). "
                "Add it to the asset input or supply a typed context constant.",
            )
        value = asset[column]
        if value is None:
            raise cf.ImpactError(
                "missing_context_value",
                "/context/" + field,
                f"Asset column {column!r} has no value. "
                "Supply a value in the input or use a typed context constant.",
            )
        # DuckDB decimal columns become Decimal objects in Arrow. Framework
        # numeric context fields use finite JSON numbers and consume doubles.
        document[field] = float(value) if isinstance(value, Decimal) else value
    try:
        if isinstance(context, ResponseContext):
            return ResponseContext.from_json(
                _dump({"kind": "response_context", "values": document})
            )
        return cf.ImpactContext.from_json(_dump(document))
    except (TypeError, ValueError) as error:
        columns = ", ".join(f"{field} from {column!r}" for field, column in bindings)
        raise cf.ImpactError(
            "invalid_context_value",
            "/context",
            f"Check bound asset values ({columns}). {error}",
        ) from error


def resolve_response(
    model: Model,
    context: cf.ImpactContext | None,
    asset: Mapping[str, object],
    cache: ResolutionCache,
) -> tuple[cf.Impact, ResponseContext | None]:
    if isinstance(model, cf.Impact):
        return model, None
    concrete = (
        cf.EmptyContext() if context is None else _concrete_context(context, asset)
    )
    key = (model.sha256, concrete.to_json())
    if isinstance(model, ContextualImpact):
        assert isinstance(concrete, ResponseContext)
        if key not in cache:
            cache[key] = model.resolve(concrete)
        return cache[key], concrete
    if key not in cache:
        cache[key] = model.resolve(concrete)
    return cache[key], None


@dataclass(frozen=True, slots=True)
class Response:
    """An unevaluated named response and its exact input declaration."""

    name: str
    variable: Variable | str
    model: Model
    context: cf.ImpactContext | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, "response name")
        if not isinstance(self.variable, Variable):
            _identifier(self.variable, "response variable")
        object.__setattr__(self, "context", response_context(self.model, self.context))

    @property
    def variable_id(self) -> str:
        return (
            self.variable.id if isinstance(self.variable, Variable) else self.variable
        )

    @property
    def columns(self) -> tuple[str, ...]:
        return context_columns(self.context)

    def bind(self, measurement: cf.Measurement | None) -> None:
        bind_response(self.model, measurement)

    def resolve(
        self, asset: Mapping[str, object], cache: ResolutionCache
    ) -> tuple[cf.Impact, ResponseContext | None]:
        return resolve_response(self.model, self.context, asset, cache)
