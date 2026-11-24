"""Immutable scientific declarations and exact scientific identity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias, TypeVar, cast
from urllib.parse import unquote

import crc_framework
import jsonschema
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

ScientificValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | Sequence["ScientificValue"]
    | Mapping[str, "ScientificValue"]
)


class Calendar(Enum):
    GREGORIAN = "gregorian"
    NO_LEAP = "no_leap"
    DAYS_360 = "360_day"
    ALL_LEAP = "all_leap"
    JULIAN = "julian"
    PROLEPTIC_GREGORIAN = "proleptic_gregorian"


class PeriodKind(Enum):
    WINDOW = "window"
    TIMELESS = "timeless"
    UNKNOWN = "unknown"


class ProbabilityKind(Enum):
    ANNUAL = "annual"
    DISTRIBUTION = "distribution"
    CONFIDENCE = "confidence"
    PROJECTION = "projection"


def _identifier(value: str, name: str = "id") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty without boundary whitespace")
    return value


def _freeze(value: ScientificValue) -> ScientificValue:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("scientific object keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("scientific values must be finite JSON values")


def _mapping(
    value: Mapping[str, ScientificValue] | None, name: str
) -> Mapping[str, ScientificValue] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None")
    return cast(Mapping[str, ScientificValue], _freeze(value))


def _plain(value: ScientificValue) -> ScientificValue:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_plain(v) for v in value]
    return value


def _json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def _load(text: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate field {key!r}")
            result[key] = value
        return result

    def invalid(token: str) -> None:
        raise ValueError(f"non-finite JSON number {token}")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return cast(dict[str, object], value)


def _fields(
    value: object, required: set[str], optional: set[str] | frozenset[str] = frozenset()
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    if set(value) - required - optional or not required <= set(value):
        raise ValueError(
            f"invalid fields: expected {sorted(required)}; optional {sorted(optional)}"
        )
    return cast(dict[str, object], value)


def _local_schema_references(value: object) -> None:
    """Inspect schema positions without interpreting const/enum/example data."""
    pending = [value]
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if not isinstance(node, dict) or id(node) in visited:
            continue
        visited.add(id(node))
        if (
            "$schema" in node
            and node["$schema"] != "https://json-schema.org/draft/2020-12/schema"
        ):
            raise ValueError("scenario schemas use JSON Schema draft2020-12")
        for key in ("$ref", "$dynamicRef"):
            if key not in node:
                continue
            reference = node[key]
            if not isinstance(reference, str) or not reference.startswith("#"):
                raise ValueError(
                    "scenario schemas use local references without remote acquisition"
                )
            pointer = unquote(reference[1:])
            target = value
            if pointer.startswith("/"):
                try:
                    for part in pointer[1:].split("/"):
                        part = part.replace("~1", "/").replace("~0", "~")
                        target = (
                            target[int(part)]
                            if isinstance(target, list)
                            else cast(dict[str, object], target)[part]
                        )
                except (KeyError, IndexError, ValueError, TypeError):
                    continue
                pending.append(target)
        pending.extend(DRAFT202012.subresources_of(node))


def _label(label: str | None) -> None:
    if label is not None and not isinstance(label, str):
        raise TypeError("label must be a string or None")


@dataclass(frozen=True, init=False)
class Variable:
    id: str
    measurement: crc_framework.Measurement | None
    label: str | None

    def __init__(
        self,
        id: str,
        *,
        measurement: crc_framework.Measurement | None = None,
        label: str | None = None,
    ) -> None:
        _identifier(id)
        if measurement is not None and not isinstance(
            measurement, crc_framework.Measurement
        ):
            raise TypeError("measurement must be a framework Measurement or None")
        _label(label)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "label", label)

    def _document(self) -> dict[str, ScientificValue]:
        result: dict[str, ScientificValue] = {"id": self.id}
        if self.measurement is not None:
            result["measurement"] = cast(
                ScientificValue, json.loads(self.measurement.to_json())
            )
        if self.label is not None:
            result["label"] = self.label
        return result


@dataclass(frozen=True, init=False)
class Scenario:
    id: str
    definition: Mapping[str, ScientificValue] | None
    schema: Mapping[str, ScientificValue] | None
    label: str | None

    def __init__(
        self,
        id: str,
        *,
        definition: Mapping[str, ScientificValue] | None = None,
        schema: Mapping[str, ScientificValue] | None = None,
        label: str | None = None,
    ) -> None:
        _identifier(id)
        _label(label)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "definition", _mapping(definition, "definition"))
        object.__setattr__(self, "schema", _mapping(schema, "schema"))
        object.__setattr__(self, "label", label)
        if self.schema is not None:
            if self.definition is None:
                raise ValueError("a scenario schema requires its scientific definition")
            document = cast(dict[str, object], _plain(self.schema))
            if (
                document.get("$schema", "https://json-schema.org/draft/2020-12/schema")
                != "https://json-schema.org/draft/2020-12/schema"
            ):
                raise ValueError("scenario schemas use JSON Schema draft 2020-12")
            try:
                jsonschema.Draft202012Validator.check_schema(document)
                _local_schema_references(document)
                jsonschema.Draft202012Validator(document, registry=Registry()).validate(
                    _plain(self.definition)
                )
            except (
                jsonschema.exceptions.SchemaError,
                jsonschema.exceptions.ValidationError,
                Unresolvable,
            ) as error:
                raise ValueError(
                    f"scenario definition/schema is invalid: {error}"
                ) from error

    def _document(self) -> dict[str, ScientificValue]:
        result: dict[str, ScientificValue] = {"id": self.id}
        if self.definition is not None:
            result["definition"] = _plain(self.definition)
        if self.schema is not None:
            result["schema"] = _plain(self.schema)
        if self.label is not None:
            result["label"] = self.label
        return result


@dataclass(frozen=True, init=False)
class Period:
    id: str
    start: int | None
    end: int | None
    calendar: Calendar | None
    kind: PeriodKind

    def __init__(self) -> None:
        raise TypeError("construct a Period with window, timeless, or unknown")

    @classmethod
    def _create(
        cls,
        kind: PeriodKind,
        start: int | None = None,
        end: int | None = None,
        calendar: Calendar | None = None,
    ) -> Period:
        result = object.__new__(cls)
        definition: dict[str, ScientificValue] = {"kind": kind.value}
        if kind is PeriodKind.WINDOW:
            definition.update(start=start, end=end)
            if calendar is not None:
                definition["calendar"] = calendar.value
        identifier = "period:" + hashlib.sha256(_json(definition).encode()).hexdigest()
        for name, value in (
            ("id", identifier),
            ("kind", kind),
            ("start", start),
            ("end", end),
            ("calendar", calendar),
        ):
            object.__setattr__(result, name, value)
        return result

    @classmethod
    def window(
        cls, start: int, end: int, *, calendar: Calendar | None = None
    ) -> Period:
        if type(start) is not int or type(end) is not int:
            raise TypeError("window bounds must be integer years")
        if start > end:
            raise ValueError("inclusive window start must not exceed end")
        if calendar is not None and not isinstance(calendar, Calendar):
            raise TypeError("calendar must be Calendar or None")
        return cls._create(PeriodKind.WINDOW, start, end, calendar)

    @classmethod
    def timeless(cls) -> Period:
        return cls._create(PeriodKind.TIMELESS)

    @classmethod
    def unknown(cls) -> Period:
        return cls._create(PeriodKind.UNKNOWN)

    def _document(self) -> dict[str, ScientificValue]:
        result: dict[str, ScientificValue] = {"id": self.id, "kind": self.kind.value}
        if self.kind is PeriodKind.WINDOW:
            result.update(start=self.start, end=self.end)
            if self.calendar is not None:
                result["calendar"] = self.calendar.value
        return result


@dataclass(frozen=True, init=False)
class Probability:
    kind: ProbabilityKind
    tail: crc_framework.Tail | None

    def __init__(self) -> None:
        raise TypeError(
            "construct Probability with annual, distribution, confidence, or projection"
        )

    @classmethod
    def _create(
        cls, kind: ProbabilityKind, tail: crc_framework.Tail | None = None
    ) -> Probability:
        result = object.__new__(cls)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "tail", tail)
        return result

    @classmethod
    def annual(cls, *, tail: crc_framework.Tail) -> Probability:
        if not isinstance(tail, crc_framework.Tail):
            raise TypeError("tail must be framework Tail")
        return cls._create(ProbabilityKind.ANNUAL, tail)

    @classmethod
    def distribution(cls) -> Probability:
        return cls._create(ProbabilityKind.DISTRIBUTION)

    @classmethod
    def confidence(cls) -> Probability:
        return cls._create(ProbabilityKind.CONFIDENCE)

    @classmethod
    def projection(cls) -> Probability:
        return cls._create(ProbabilityKind.PROJECTION)

    def _document(self) -> dict[str, ScientificValue]:
        result: dict[str, ScientificValue] = {"kind": self.kind.value}
        if self.tail is not None:
            result["tail"] = self.tail.value
        return result


@dataclass(frozen=True, init=False)
class SeriesSpec:
    variable: Variable
    scenario: Scenario | None
    period: Period | None
    probability: Probability | None

    def __init__(
        self,
        variable: Variable,
        *,
        scenario: Scenario | None = None,
        period: Period | None = None,
        probability: Probability | None = None,
    ) -> None:
        if not isinstance(variable, Variable):
            raise TypeError("variable must be Variable")
        for name, value, expected in (
            ("scenario", scenario, Scenario),
            ("period", period, Period),
            ("probability", probability, Probability),
        ):
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__} or None")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "variable", variable)

    def _document(self) -> dict[str, ScientificValue]:
        result: dict[str, ScientificValue] = {
            "variable_id": self.variable.id,
            "scenario_id": None if self.scenario is None else self.scenario.id,
            "period_id": None if self.period is None else self.period.id,
        }
        if self.probability is not None:
            result["probability"] = self.probability._document()
        return result


def _meaning(
    value: Variable
    | Scenario
    | Period
    | Probability
    | SeriesSpec
    | crc_framework.Measurement,
) -> ScientificValue:
    if isinstance(value, Variable):
        return {
            "kind": "variable",
            "measurement": None
            if value.measurement is None
            else cast(ScientificValue, json.loads(value.measurement.to_json())),
        }
    if isinstance(value, Scenario):
        return {
            "kind": "scenario",
            "definition": _plain(value.definition),
            "schema": _plain(value.schema),
        }
    if isinstance(value, Period):
        return {k: v for k, v in value._document().items() if k != "id"}
    if isinstance(value, Probability):
        return value._document()
    if isinstance(value, SeriesSpec):
        return {
            "variable": _meaning(value.variable),
            "scenario": None if value.scenario is None else _meaning(value.scenario),
            "period": None if value.period is None else _meaning(value.period),
            "probability": None
            if value.probability is None
            else _meaning(value.probability),
        }
    if isinstance(value, crc_framework.Measurement):
        return cast(ScientificValue, json.loads(value.to_json()))
    raise TypeError(
        "scientific identity requires a scientific declaration or framework Measurement"
    )


def _canonical_numbers(value: ScientificValue) -> ScientificValue:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {key: _canonical_numbers(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_canonical_numbers(item) for item in value]
    return value


def scientific_identity(
    value: Variable
    | Scenario
    | Period
    | Probability
    | SeriesSpec
    | crc_framework.Measurement,
) -> str:
    """Hash scientific meaning, excluding descriptor IDs, labels, and provenance."""
    return (
        "sha256:"
        + hashlib.sha256(
            _json(_canonical_numbers(_meaning(value))).encode()
        ).hexdigest()
    )


_Descriptor = TypeVar("_Descriptor", Variable, Scenario, Period)


def _descriptors(
    values: Sequence[_Descriptor], expected: type[_Descriptor]
) -> dict[str, _Descriptor]:
    result: dict[str, _Descriptor] = {}
    for value in values:
        if not isinstance(value, expected):
            raise TypeError(f"descriptors must contain {expected.__name__} objects")
        if value.id in result:
            raise ValueError(f"duplicate {expected.__name__} id {value.id!r}")
        result[value.id] = value
    return result


def _merge_descriptor(
    value: _Descriptor | None, descriptors: dict[str, _Descriptor]
) -> None:
    if value is not None:
        existing = descriptors.get(value.id)
        if (
            existing is not None
            and existing != value
            and scientific_identity(existing) != scientific_identity(value)
        ):
            raise ValueError(f"conflicting scientific definition for {value.id!r}")
        descriptors.setdefault(value.id, value)


@dataclass(frozen=True, init=False)
class DataSpec:
    variables: tuple[Variable, ...]
    scenarios: tuple[Scenario, ...]
    periods: tuple[Period, ...]
    series: tuple[SeriesSpec, ...]

    def __init__(
        self,
        variables: Sequence[Variable],
        *,
        series: Sequence[SeriesSpec],
        scenarios: Sequence[Scenario] = (),
        periods: Sequence[Period] = (),
    ) -> None:
        variable_map = _descriptors(variables, Variable)
        scenario_map = _descriptors(scenarios, Scenario)
        period_map = _descriptors(periods, Period)
        resolved = tuple(series)
        identities: set[tuple[str, str | None, str | None]] = set()
        for item in resolved:
            if not isinstance(item, SeriesSpec):
                raise TypeError("series must contain SeriesSpec objects")
            variable = variable_map.get(item.variable.id)
            if variable is None:
                raise ValueError(
                    f"series references undeclared variable {item.variable.id!r}"
                )
            if variable != item.variable and scientific_identity(
                variable
            ) != scientific_identity(item.variable):
                raise ValueError(
                    f"conflicting measurement for variable {item.variable.id!r}"
                )
            _merge_descriptor(item.scenario, scenario_map)
            _merge_descriptor(item.period, period_map)
            key = (
                item.variable.id,
                None if item.scenario is None else item.scenario.id,
                None if item.period is None else item.period.id,
            )
            if key in identities:
                raise ValueError(f"duplicate series identity {key!r}")
            identities.add(key)
        object.__setattr__(self, "variables", tuple(variable_map.values()))
        object.__setattr__(self, "scenarios", tuple(scenario_map.values()))
        object.__setattr__(self, "periods", tuple(period_map.values()))
        object.__setattr__(self, "series", resolved)

    def _document(self) -> dict[str, ScientificValue]:
        return {
            "variables": [v._document() for v in self.variables],
            "scenarios": [s._document() for s in self.scenarios],
            "periods": [p._document() for p in self.periods],
            "series": [s._document() for s in self.series],
        }


def _variable(document: object) -> Variable:
    value = _fields(document, {"id"}, {"measurement", "label"})
    if "label" in value and not isinstance(value["label"], str):
        raise TypeError("serialized labels must be strings; omit unknown labels")
    measurement = (
        None
        if "measurement" not in value
        else crc_framework.Measurement.from_json(_json(value["measurement"]))
    )
    return Variable(
        cast(str, value["id"]),
        measurement=measurement,
        label=cast(str | None, value.get("label")),
    )


def _scenario(document: object) -> Scenario:
    value = _fields(document, {"id"}, {"definition", "schema", "label"})
    for name in ("definition", "schema"):
        if name in value and not isinstance(value[name], dict):
            raise TypeError(f"serialized {name} must be an object; omit unknown facts")
    if "label" in value and not isinstance(value["label"], str):
        raise TypeError("serialized labels must be strings; omit unknown labels")
    return Scenario(
        cast(str, value["id"]),
        definition=cast(Mapping[str, ScientificValue] | None, value.get("definition")),
        schema=cast(Mapping[str, ScientificValue] | None, value.get("schema")),
        label=cast(str | None, value.get("label")),
    )


def _period(document: object) -> Period:
    value = _fields(document, {"id", "kind"}, {"start", "end", "calendar"})
    kind = PeriodKind(value["kind"])
    if kind is PeriodKind.WINDOW:
        _fields(value, {"id", "kind", "start", "end"}, {"calendar"})
        calendar = None if "calendar" not in value else Calendar(value["calendar"])

        def year(item: object) -> int:
            if type(item) is int:
                return item
            if isinstance(item, float) and math.isfinite(item) and item.is_integer():
                return int(item)
            raise TypeError("period year must be an integer JSON number")

        result = Period.window(
            year(value["start"]), year(value["end"]), calendar=calendar
        )
    else:
        _fields(value, {"id", "kind"})
        result = Period.timeless() if kind is PeriodKind.TIMELESS else Period.unknown()
    if value["id"] != result.id:
        raise ValueError("period id does not match its scientific definition")
    return result


def _probability(document: object) -> Probability:
    value = _fields(document, {"kind"}, {"tail"})
    kind = ProbabilityKind(value["kind"])
    if kind is ProbabilityKind.ANNUAL:
        _fields(value, {"kind", "tail"})
        return Probability.annual(tail=crc_framework.Tail(value["tail"]))
    _fields(value, {"kind"})
    return Probability._create(kind)


def _series(
    document: object,
    variables: Mapping[str, Variable],
    scenarios: Mapping[str, Scenario],
    periods: Mapping[str, Period],
) -> SeriesSpec:
    value = _fields(
        document, {"variable_id", "scenario_id", "period_id"}, {"probability"}
    )
    try:
        variable = variables[cast(str, value["variable_id"])]
        scenario = (
            None
            if value["scenario_id"] is None
            else scenarios[cast(str, value["scenario_id"])]
        )
        period = (
            None
            if value["period_id"] is None
            else periods[cast(str, value["period_id"])]
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"dangling series descriptor reference: {error}") from error
    return SeriesSpec(
        variable,
        scenario=scenario,
        period=period,
        probability=None
        if "probability" not in value
        else _probability(value["probability"]),
    )


def _data_spec(document: object) -> DataSpec:
    value = _fields(document, {"variables", "scenarios", "periods", "series"})
    if not all(isinstance(v, list) for v in value.values()):
        raise TypeError("declaration collections must be arrays")
    variables = tuple(_variable(v) for v in cast(list[object], value["variables"]))
    scenarios = tuple(_scenario(s) for s in cast(list[object], value["scenarios"]))
    periods = tuple(_period(p) for p in cast(list[object], value["periods"]))
    variable_map = _descriptors(variables, Variable)
    scenario_map = _descriptors(scenarios, Scenario)
    period_map = _descriptors(periods, Period)
    series = tuple(
        _series(s, variable_map, scenario_map, period_map)
        for s in cast(list[object], value["series"])
    )
    return DataSpec(variables, series=series, scenarios=scenarios, periods=periods)


def _schema_definitions() -> dict[str, object]:
    """Declaration schema fragments shared by dataset and catalog schemas."""
    identifier = {"type": "string", "minLength": 1, "pattern": r"^\S(?:[\s\S]*\S)?$"}
    measurement = {
        "$ref": "https://climateriskcommons.org/schemas/measurement-1.0.schema.json"
    }

    def obj(
        required: Sequence[str], properties: dict[str, object]
    ) -> dict[str, object]:
        return {
            "type": "object",
            "required": list(required),
            "properties": properties,
            "additionalProperties": False,
        }

    period_id = {"type": "string", "pattern": "^period:[0-9a-f]{64}$"}
    scientific_value: dict[str, object] = {
        "anyOf": [
            {"type": ["null", "boolean", "number", "string"]},
            {"type": "array", "items": {"$ref": "#/$defs/scientific_value"}},
            {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/scientific_value"},
            },
        ]
    }
    scientific_mapping = {
        "type": "object",
        "additionalProperties": {"$ref": "#/$defs/scientific_value"},
    }
    return {
        "scientific_value": scientific_value,
        "variable": obj(
            ("id",),
            {"id": identifier, "measurement": measurement, "label": {"type": "string"}},
        ),
        "scenario": {
            **obj(
                ("id",),
                {
                    "id": identifier,
                    "definition": scientific_mapping,
                    "schema": scientific_mapping,
                    "label": {"type": "string"},
                },
            ),
            "dependentRequired": {"schema": ["definition"]},
        },
        "period": {
            "oneOf": [
                obj(
                    ("id", "kind", "start", "end"),
                    {
                        "id": period_id,
                        "kind": {"const": "window"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "calendar": {"enum": [c.value for c in Calendar]},
                    },
                ),
                obj(
                    ("id", "kind"),
                    {"id": period_id, "kind": {"enum": ["timeless", "unknown"]}},
                ),
            ]
        },
        "probability": {
            "oneOf": [
                obj(
                    ("kind", "tail"),
                    {
                        "kind": {"const": "annual"},
                        "tail": {"enum": [t.value for t in crc_framework.Tail]},
                    },
                ),
                obj(
                    ("kind",),
                    {"kind": {"enum": ["distribution", "confidence", "projection"]}},
                ),
            ]
        },
        "series": obj(
            ("variable_id", "scenario_id", "period_id"),
            {
                "variable_id": identifier,
                "scenario_id": {"anyOf": [identifier, {"type": "null"}]},
                "period_id": {"anyOf": [period_id, {"type": "null"}]},
                "probability": {"$ref": "#/$defs/probability"},
            },
        ),
        "data_spec": obj(
            ("variables", "scenarios", "periods", "series"),
            {
                name: {
                    "type": "array",
                    "items": {"$ref": "#/$defs/" + definition},
                    "uniqueItems": True,
                }
                for name, definition in (
                    ("variables", "variable"),
                    ("scenarios", "scenario"),
                    ("periods", "period"),
                    ("series", "series"),
                )
            },
        ),
    }
