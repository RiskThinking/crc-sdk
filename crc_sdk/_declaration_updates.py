"""Refine absent scientific declarations without changing established meaning."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from typing import cast

import crc_framework

from ._declarations import (
    DataSpec,
    Period,
    PeriodKind,
    Probability,
    Scenario,
    ScientificValue,
    SeriesSpec,
    Variable,
    scientific_identity,
)
from ._selection import SeriesKey, _series_key, participating_spec


def _same(first: ScientificValue, second: ScientificValue) -> bool:
    return scientific_identity(
        crc_framework.Measurement(fields={"value": first})
    ) == scientific_identity(crc_framework.Measurement(fields={"value": second}))


def _merge_fields(
    first: Mapping[str, ScientificValue],
    second: Mapping[str, ScientificValue],
    path: str,
) -> dict[str, ScientificValue]:
    merged = dict(first)
    for name, value in second.items():
        if name not in merged:
            merged[name] = value
        elif isinstance(merged[name], Mapping) and isinstance(value, Mapping):
            merged[name] = _merge_fields(
                cast(Mapping[str, ScientificValue], merged[name]),
                value,
                f"{path}.{name}",
            )
        elif not _same(merged[name], value):
            raise ValueError(f"declare() contradicts established {path}.{name}")
    return merged


def _known(
    first: ScientificValue, second: ScientificValue, path: str
) -> ScientificValue:
    if first is None:
        return second
    if second is not None and not _same(first, second):
        raise ValueError(f"declare() contradicts established {path}")
    return first


def _unit(measurement: crc_framework.Measurement) -> str | None:
    unit = measurement.unit
    return unit.value if isinstance(unit, Enum) else unit


def _measurement(
    first: crc_framework.Measurement | None,
    second: crc_framework.Measurement | None,
) -> crc_framework.Measurement | None:
    if first is None:
        return second
    if second is None:
        return first
    quantity = cast(str | None, _known(first.quantity, second.quantity, "quantity"))
    unit = cast(
        str | None, _known(_unit(first), _unit(second), "unit; use convert_units()")
    )
    old = cast(dict[str, ScientificValue], json.loads(first.to_json()))
    new = cast(dict[str, ScientificValue], json.loads(second.to_json()))
    if old["kind"] == new["kind"] == "measurement":
        schema = first.schema
        if schema is None:
            schema = second.schema
        elif second.schema is not None and not _same(schema, second.schema):
            raise ValueError("declare() contradicts the established measurement schema")
        return crc_framework.Measurement(
            quantity=quantity,
            unit=unit,
            fields=_merge_fields(first.fields, second.fields, "measurement"),
            schema=schema,
        )
    if old["kind"] == "measurement":
        if first.schema is not None:
            raise ValueError(
                "declare() cannot drop an established custom measurement schema"
            )
        for name, value in first.fields.items():
            if name not in second.fields or not _same(value, second.fields[name]):
                raise ValueError(
                    f"declare() cannot drop established measurement.{name}"
                )
        return second
    if new["kind"] == "measurement":
        if second.schema is not None:
            raise ValueError(
                "a built-in measurement cannot carry a custom extension schema"
            )
        document = _merge_fields(old, second.fields, "measurement")
        return crc_framework.Measurement.from_json(json.dumps(document))
    if old["kind"] != new["kind"]:
        raise ValueError("declare() cannot replace an established measurement type")
    return crc_framework.Measurement.from_json(
        json.dumps(_merge_fields(old, new, "measurement"))
    )


def _variable(first: Variable, second: Variable) -> Variable:
    if first.id != second.id:
        raise ValueError("declare() cannot change a variable ID; use rename()")
    return Variable(
        first.id,
        measurement=_measurement(first.measurement, second.measurement),
        label=first.label if second.label is None else second.label,
    )


def _scenario(first: Scenario | None, second: Scenario | None) -> Scenario | None:
    if first is None:
        return second
    if second is None:
        return first
    if first.id != second.id:
        raise ValueError("declare() cannot change a scenario ID; use rename()")
    definition = first.definition
    if definition is None:
        definition = second.definition
    elif second.definition is not None:
        definition = _merge_fields(definition, second.definition, "scenario.definition")
    schema = first.schema
    if schema is None:
        schema = second.schema
    elif second.schema is not None and not _same(schema, second.schema):
        raise ValueError("declare() contradicts the established scenario schema")
    return Scenario(
        first.id,
        definition=definition,
        schema=schema,
        label=first.label if second.label is None else second.label,
    )


def _period(first: Period | None, second: Period | None) -> Period | None:
    if first is None:
        return second
    if second is None or second.kind is PeriodKind.UNKNOWN:
        return first
    if first.kind is PeriodKind.UNKNOWN:
        return second
    if first.kind is not second.kind:
        raise ValueError("declare() contradicts the established temporal meaning")
    if first.kind is PeriodKind.TIMELESS:
        return first
    if first.start != second.start or first.end != second.end:
        raise ValueError("declare() cannot change established period boundaries")
    if (
        first.calendar is not None
        and second.calendar is not None
        and first.calendar is not second.calendar
    ):
        raise ValueError("declare() contradicts the established source calendar")
    assert first.start is not None and first.end is not None
    return Period.window(
        first.start, first.end, calendar=first.calendar or second.calendar
    )


def _probability(
    first: Probability | None, second: Probability | None
) -> Probability | None:
    if first is None:
        return second
    if second is not None and scientific_identity(first) != scientific_identity(second):
        raise ValueError("declare() contradicts the established probability meaning")
    return first


def _series(first: SeriesSpec, second: SeriesSpec) -> SeriesSpec:
    return SeriesSpec(
        _variable(first.variable, second.variable),
        scenario=_scenario(first.scenario, second.scenario),
        period=_period(first.period, second.period),
        probability=_probability(first.probability, second.probability),
    )


def refine_spec(
    current: DataSpec,
    *,
    variable: Variable | None = None,
    scenario: Scenario | None = None,
    period: Period | None = None,
    probability: Probability | None = None,
    spec: DataSpec | None = None,
) -> tuple[DataSpec, dict[SeriesKey, SeriesKey]]:
    if spec is not None and any(
        value is not None for value in (variable, scenario, period, probability)
    ):
        raise ValueError("declare() accepts either spec or direct declarations")
    participating = participating_spec(current, current.series)
    if variable is not None and variable.id not in {
        v.id for v in participating.variables
    }:
        raise ValueError("declare() cannot change a variable ID; use rename()")
    if (
        scenario is not None
        and participating.scenarios
        and scenario.id not in {s.id for s in participating.scenarios}
    ):
        raise ValueError("declare() cannot change a scenario ID; use rename()")
    incoming = None if spec is None else list(spec.series)
    if spec is not None and {item.variable.id for item in spec.series} != {
        item.variable.id for item in participating.series
    }:
        raise ValueError("declare() cannot change variable IDs; use rename()")
    if incoming is not None and len(incoming) != len(participating.series):
        raise ValueError(
            "declare(spec=...) must describe the same participating series"
        )
    refined: list[SeriesSpec] = []
    identities: dict[SeriesKey, SeriesKey] = {}
    for item in participating.series:
        if incoming is None:
            candidate = SeriesSpec(
                variable
                if variable is not None and variable.id == item.variable.id
                else item.variable,
                scenario=scenario,
                period=period,
                probability=probability,
            )
            merged = _series(item, candidate)
        else:
            candidates: list[tuple[int, SeriesSpec]] = []
            errors: list[ValueError] = []
            for index, candidate in enumerate(incoming):
                if candidate.variable.id != item.variable.id:
                    continue
                try:
                    candidates.append((index, _series(item, candidate)))
                except ValueError as error:
                    errors.append(error)
            if len(candidates) != 1:
                if not candidates and errors:
                    raise errors[0]
                raise ValueError(
                    "declare(spec=...) requires one unambiguous declaration "
                    "for each existing series"
                )
            index, merged = candidates[0]
            incoming.pop(index)
        refined.append(merged)
        identities[_series_key(item)] = _series_key(merged)
    variables = {item.variable.id: item.variable for item in refined}
    return DataSpec(tuple(variables.values()), series=refined), identities
