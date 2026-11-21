"""Portable selection of calibrated framework impacts by declared context."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, make_dataclass
from enum import Enum
from keyword import iskeyword
from threading import RLock
from types import MappingProxyType
from typing import TypeAlias, cast
from weakref import WeakValueDictionary

import crc_framework as cf

from ._assets import ColumnBinding
from ._dataset_schema import _array, _dump, _identifier, _load, _object
from ._declarations import ScientificValue

_Value: TypeAlias = str | ColumnBinding
_Key: TypeAlias = tuple[tuple[str, str], ...]
_EXACT_NAME = r"^\S(?:[\s\S]*\S)?$"


def _column_schema() -> dict[str, ScientificValue]:
    return {
        "type": "object",
        "properties": {"column": {"type": "string", "pattern": _EXACT_NAME}},
        "required": ["column"],
        "additionalProperties": False,
    }


def _name(value: object) -> str:
    result = _identifier(value, "response context field")
    if (
        not result.isidentifier()
        or iskeyword(result)
        or result.startswith("_")
        or result in {"schema", "to_json", "from_json"}
    ):
        raise ValueError(f"invalid response context field {result!r}")
    return result


def _value(value: object, name: str) -> _Value:
    if isinstance(value, ColumnBinding):
        return value
    if isinstance(value, Enum):
        value = value.value
    return _identifier(value, f"context field {name!r}")


def _context_values(context: ResponseContext) -> dict[str, _Value]:
    if (
        not is_dataclass(context)
        or isinstance(context, type)
        or not getattr(type(context), "__dataclass_params__").frozen
    ):
        raise TypeError("Declare the ResponseContext as a frozen dataclass.")
    return {
        _name(field.name): _value(getattr(context, field.name), field.name)
        for field in fields(context)
    }


def _context_document(values: Mapping[str, _Value]) -> dict[str, object]:
    return {
        "kind": "response_context",
        "values": {
            name: {"column": value.name} if isinstance(value, ColumnBinding) else value
            for name, value in values.items()
        },
    }


def _recorded_context(document: object) -> ResponseContext:
    record = _object(document, {"kind", "values"})
    if record["kind"] != "response_context":
        raise ValueError("expected a ResponseContext")
    values = _load(_dump(record["values"]))
    if not values:
        raise ValueError("ResponseContext requires at least one field")
    return _own_context(
        {
            _name(name): ColumnBinding(
                _identifier(_object(value, {"column"})["column"], "asset column")
            )
            if isinstance(value, dict)
            else _value(value, name)
            for name, value in values.items()
        }
    )


class ResponseContext(cf.ImpactContext):
    """Base for a model author's frozen dataclass of categorical case fields."""

    @property
    def schema(self) -> Mapping[str, ScientificValue]:
        names = list(_context_values(self))
        return {
            "type": "object",
            "properties": {
                "kind": {"const": "response_context"},
                "values": {
                    "type": "object",
                    "properties": {
                        name: {
                            "anyOf": [
                                {"type": "string", "pattern": _EXACT_NAME},
                                _column_schema(),
                            ]
                        }
                        for name in names
                    },
                    "required": names,
                    "additionalProperties": False,
                },
            },
            "required": ["kind", "values"],
            "additionalProperties": False,
        }

    def to_json(self) -> str:
        return _dump(_context_document(_context_values(self)))

    @classmethod
    def from_json(cls, text: str) -> ResponseContext:
        """Read a canonical record without requiring the author's Python class."""
        return _recorded_context(_load(text))


_declared_types: WeakValueDictionary[tuple[str, ...], type[ResponseContext]] = (
    WeakValueDictionary()
)
_declared_types_lock = RLock()


def _declared_context(names: tuple[str, ...]) -> type[ResponseContext]:
    with _declared_types_lock:
        context = _declared_types.get(names)
        if context is None:
            context = cast(
                type[ResponseContext],
                make_dataclass(
                    "DeclaredResponseContext",
                    [(name, str | ColumnBinding) for name in names],
                    bases=(ResponseContext,),
                    frozen=True,
                    slots=True,
                    kw_only=True,
                ),
            )
            _declared_types[names] = context
        return context


def _own_context(values: Mapping[str, _Value]) -> ResponseContext:
    constructor = cast(
        Callable[..., ResponseContext], _declared_context(tuple(sorted(values)))
    )
    return constructor(**values)


def _case_key(values: Mapping[str, _Value]) -> _Key:
    return tuple(
        (_name(name), _identifier(value, f"case field {name!r}"))
        for name, value in sorted(values.items())
    )


@dataclass(frozen=True, slots=True, init=False, repr=False, eq=False)
class ContextualImpact:
    """Select an exact framework Impact from calibrated categorical cases."""

    _cases: Mapping[_Key, cf.Impact]
    _choices: Mapping[str, tuple[str, ...]]
    _requirements: cf.ImpactRequirements
    _sha256: str

    def __init__(self, cases: Sequence[tuple[ResponseContext, cf.Impact]]) -> None:
        if not cases:
            raise ValueError("ContextualImpact requires at least one case")
        context, first = cases[0]
        if not isinstance(context, ResponseContext):
            raise TypeError("cases require a ResponseContext dataclass")
        if type(first) is not cf.Impact:
            raise TypeError("cases require calibrated framework Impact objects")
        names = _context_values(context).keys()
        if not names:
            raise ValueError("ResponseContext requires at least one field")
        context_type = type(context)
        models: dict[_Key, cf.Impact] = {}
        choices: dict[str, set[str]] = {name: set() for name in sorted(names)}
        input, output = first.input, first.output
        for record, impact in cases:
            if type(record) is not type(context):
                raise TypeError("all cases require the same ResponseContext dataclass")
            values = _context_values(record)
            if values.keys() != names:
                raise ValueError("case context fields differ")
            key = _case_key(values)
            if key in models:
                raise ValueError(f"duplicate response context {dict(key)!r}")
            if type(impact) is not cf.Impact:
                raise TypeError("cases require calibrated framework Impact objects")
            if impact.input != input or impact.output != output:
                raise ValueError(
                    "all cases require identical input and output measurements"
                )
            models[key] = impact
            for name, value in key:
                choices[name].add(value)
        field_choices = {
            name: tuple(sorted(values)) for name, values in choices.items()
        }
        unit = input.unit
        assert unit is not None
        requirements = cf.ImpactRequirements(
            input=input,
            measurement_type=type(input),
            measurement_schema=input.schema or MappingProxyType({}),
            accepted_units=(unit,),
            context_type=context_type,
            context_fields=tuple(
                cf.ContextField(name, "str | ColumnBinding", True, allowed)
                for name, allowed in field_choices.items()
            ),
            unestablished_facts=(),
            output=output,
        )
        object.__setattr__(
            self, "_cases", MappingProxyType(dict(sorted(models.items())))
        )
        object.__setattr__(self, "_choices", MappingProxyType(field_choices))
        object.__setattr__(self, "_requirements", requirements)
        object.__setattr__(
            self,
            "_sha256",
            "sha256:" + hashlib.sha256(self.to_json().encode()).hexdigest(),
        )

    def __repr__(self) -> str:
        context = self.requirements.context_type.__qualname__
        return f"ContextualImpact({context}, cases={len(self._cases)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContextualImpact) and self.sha256 == other.sha256

    def __hash__(self) -> int:
        return hash(self.sha256)

    @property
    def requirements(self) -> cf.ImpactRequirements:
        return self._requirements

    @property
    def sha256(self) -> str:
        return self._sha256

    def _context(self, context: cf.ImpactContext | None) -> ResponseContext:
        if not isinstance(context, ResponseContext) or type(context) not in (
            self.requirements.context_type,
            _declared_context(tuple(self._choices)),
        ):
            raise cf.ImpactError(
                "missing_context" if context is None else "invalid_context",
                "/context",
                f"Use {self.requirements.context_type.__name__}("
                + ", ".join(f"{name}=..." for name in self._choices)
                + "). Supply categorical constants or assets.column(...).",
            )
        values = _context_values(context)
        if values.keys() != self._choices.keys():
            raise cf.ImpactError(
                "invalid_context",
                "/context",
                "Context fields differ from the model declaration.",
            )
        for name, value in values.items():
            if isinstance(value, str) and value not in self._choices[name]:
                raise cf.ImpactError(
                    "invalid_context_value",
                    f"/context/{name}",
                    f"Expected one of {self._choices[name]!r}; received {value!r}.",
                )
        if (
            all(isinstance(value, str) for value in values.values())
            and _case_key(values) not in self._cases
        ):
            raise cf.ImpactError(
                "unknown_context_case",
                "/context",
                f"No calibrated case matches {values!r}.",
            )
        return _own_context(values)

    def resolve(self, context: ResponseContext) -> cf.Impact:
        """Resolve a concrete context after its asset bindings have been supplied."""
        values = _context_values(self._context(context))
        if any(isinstance(value, ColumnBinding) for value in values.values()):
            raise cf.ImpactError(
                "unresolved_context",
                "/context",
                "Supply context values before resolving the model.",
            )
        return self._cases[_case_key(values)]

    def to_json(self) -> str:
        return _dump(
            {
                "kind": "contextual",
                "cases": [
                    {
                        "context": _context_document(dict(key)),
                        "impact": _load(impact.to_json()),
                    }
                    for key, impact in self._cases.items()
                ],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ContextualImpact:
        document = _object(_load(text), {"kind", "cases"})
        if document["kind"] != "contextual":
            raise ValueError("expected a ContextualImpact")
        cases = []
        for value in _array(document["cases"], "impact cases"):
            case = _object(value, {"context", "impact"})
            cases.append(
                (
                    _recorded_context(case["context"]),
                    cf.Impact.from_json(_dump(case["impact"])),
                )
            )
        return cls(cases)
