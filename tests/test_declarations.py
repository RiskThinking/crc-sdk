"""Scientific declarations preserve exact identities and genuinely absent facts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import crc_framework
import pytest

from crc_sdk import (
    Calendar,
    DataSpec,
    Period,
    PeriodKind,
    Probability,
    ProbabilityKind,
    Scenario,
    SeriesSpec,
    Variable,
    scientific_identity,
)
from crc_sdk._declarations import (
    ScientificValue,
    _data_spec,
    _period,
    _probability,
    _scenario,
)


def test_single_series_needs_no_taxonomy_scenario_period_or_measurement() -> None:
    variable = Variable("customer:temperature change")
    declaration = DataSpec([variable], series=[SeriesSpec(variable)])
    assert declaration.scenarios == ()
    assert declaration.periods == ()
    assert variable.measurement is None
    serialized = declaration._document()
    assert serialized["series"] == [
        {"variable_id": variable.id, "scenario_id": None, "period_id": None}
    ]
    assert _data_spec(serialized) == declaration


def test_window_bounds_are_inclusive_and_calendar_is_not_invented() -> None:
    period = Period.window(2030, 2030)
    assert period.start == period.end == 2030
    assert period.calendar is None
    assert "calendar" not in period._document()
    assert period.id == Period.window(2030, 2030).id
    native = Period.window(2030, 2030, calendar=Calendar.DAYS_360)
    assert native.id != period.id
    assert _period(native._document()) == native
    with pytest.raises(ValueError, match="start"):
        Period.window(2040, 2039)
    with pytest.raises(TypeError, match="integer"):
        Period.window(True, 2030)
    with pytest.raises(TypeError, match="Calendar"):
        Period.window(2030, 2040, calendar="gregorian")  # type: ignore[arg-type]


def test_unknown_is_distinct_from_timeless_and_absence() -> None:
    assert Period.unknown().kind is PeriodKind.UNKNOWN
    assert Period.timeless().kind is PeriodKind.TIMELESS
    assert Period.unknown().id != Period.timeless().id
    for period in (Period.unknown(), Period.timeless()):
        assert period.start is None and period.end is None and period.calendar is None
        assert set(period._document()) == {"id", "kind"}


def test_period_ids_cannot_be_forged_or_window_facts_added_to_unknown() -> None:
    payload = Period.window(2030, 2040)._document()
    payload["end"] = 2041
    with pytest.raises(ValueError, match="id"):
        _period(payload)
    payload = Period.unknown()._document()
    payload["start"] = 2000
    with pytest.raises(ValueError, match="invalid fields"):
        _period(payload)


def test_probability_requires_an_explicit_annual_tail() -> None:
    annual = Probability.annual(tail=crc_framework.Tail.UPPER)
    assert annual.kind is ProbabilityKind.ANNUAL
    assert annual.tail is crc_framework.Tail.UPPER
    assert _probability(annual._document()) == annual
    for probability in (
        Probability.distribution(),
        Probability.confidence(),
        Probability.projection(),
    ):
        assert probability.tail is None
        assert "tail" not in probability._document()
    with pytest.raises(TypeError, match="Tail"):
        Probability.annual(tail="upper")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid fields"):
        _probability({"kind": "projection", "tail": "upper"})
    with pytest.raises(ValueError, match="invalid fields"):
        _probability({"kind": "annual"})


def test_series_references_gather_scenario_and_period_without_duplicate_setup() -> None:
    variable = Variable("depth")
    scenario = Scenario("planning", definition={"temperature_change": 1.5})
    period = Period.window(2030, 2040, calendar=Calendar.NO_LEAP)
    series = SeriesSpec(variable, scenario=scenario, period=period)
    spec = DataSpec([variable], series=[series])
    assert spec.scenarios == (scenario,)
    assert spec.periods == (period,)
    assert _data_spec(spec._document()) == spec


def test_duplicate_series_missing_dimensions_are_not_distinct_identities() -> None:
    variable = Variable("depth")
    with pytest.raises(ValueError, match="duplicate series"):
        DataSpec([variable], series=[SeriesSpec(variable), SeriesSpec(variable)])
    with pytest.raises(ValueError, match="undeclared variable"):
        DataSpec([variable], series=[SeriesSpec(Variable("different"))])


def test_every_descriptor_reference_is_checked_and_unknown_fields_fail() -> None:
    variable = Variable("depth")
    spec = DataSpec([variable], series=[SeriesSpec(variable)])
    payload = spec._document()
    series = payload["series"]
    assert isinstance(series, list) and isinstance(series[0], dict)
    series[0]["scenario_id"] = "unknown"
    with pytest.raises(ValueError, match="dangling"):
        _data_spec(payload)
    payload = spec._document()
    payload["pathway"] = "historic"
    with pytest.raises(ValueError, match="invalid fields"):
        _data_spec(payload)
    payload = spec._document()
    variables = payload["variables"]
    assert isinstance(variables, list)
    variables.append(variables[0])
    with pytest.raises(ValueError, match="duplicate"):
        _data_spec(payload)


def test_identifiers_preserve_unicode_internal_whitespace_and_case() -> None:
    assert Variable("Δ तापमान").id == "Δ तापमान"
    assert Variable("Risk").id != Variable("risk").id
    for id in ("", " leading", "trailing ", "\u00a0name"):
        with pytest.raises(ValueError, match="boundary whitespace|nonempty"):
            Variable(id)
    with pytest.raises(TypeError, match="string"):
        Variable(1)  # type: ignore[arg-type]


def test_scientific_identity_ignores_display_and_external_identity_only() -> None:
    measurement = crc_framework.Measurement(quantity="user:custom", unit="user:unit")
    first = Variable("id-one", measurement=measurement, label="First display")
    second = Variable("id-two", measurement=measurement, label="Second display")
    assert scientific_identity(first) == scientific_identity(second)
    assert scientific_identity(Scenario("one", label="A")) == scientific_identity(
        Scenario("two", label="B")
    )
    different = Variable(
        "id-one",
        measurement=crc_framework.Measurement(quantity="other", unit="user:unit"),
    )
    assert scientific_identity(first) != scientific_identity(different)


def test_all_custom_scientific_fields_and_schema_fields_participate() -> None:
    first = Scenario("s", definition={"label": "scientific label", "delta": 1.5})
    second = Scenario(
        "s", definition={"label": "changed scientific label", "delta": 1.5}
    )
    assert scientific_identity(first) != scientific_identity(second)
    schema: dict[str, ScientificValue] = {
        "type": "object",
        "properties": {"delta": {"type": "number"}},
        "required": ["delta"],
    }
    scenario = Scenario("s", definition={"delta": 1.5}, schema=schema)
    changed_schema: dict[str, ScientificValue] = {
        **schema,
        "$comment": "Scientific schema revision",
    }
    assert scientific_identity(scenario) != scientific_identity(
        Scenario("s", definition={"delta": 1.5}, schema=changed_schema)
    )
    assert _scenario(scenario._document()) == scenario


def test_optional_custom_schema_is_enforced_without_network_io() -> None:
    schema: dict[str, ScientificValue] = {
        "type": "object",
        "properties": {"delta": {"type": "number"}},
        "required": ["delta"],
        "additionalProperties": False,
    }
    with pytest.raises(ValueError, match="invalid"):
        Scenario("s", definition={"delta": "not numeric"}, schema=schema)
    with pytest.raises(ValueError, match="requires"):
        Scenario("s", schema=schema)
    with pytest.raises(ValueError, match="invalid"):
        Scenario("s", definition={}, schema={"type": "nonexistent"})
    with pytest.raises(ValueError, match="local references"):
        Scenario(
            "s", definition={}, schema={"$ref": "https://unavailable.invalid/science"}
        )
    with pytest.raises(TypeError, match="mapping"):
        Scenario("s", definition=[1, 2])  # type: ignore[arg-type]


def test_declarations_and_nested_scientific_values_are_immutable() -> None:
    source = {"nested": {"value": [1, 2]}}
    scenario = Scenario("s", definition=source)
    source["nested"]["value"].append(3)
    assert json.loads(json.dumps(scenario._document()))["definition"]["nested"][
        "value"
    ] == [1, 2]
    with pytest.raises(FrozenInstanceError):
        scenario.id = "changed"  # type: ignore[misc]
    assert scenario.definition is not None
    with pytest.raises(TypeError):
        scenario.definition["other"] = 1  # type: ignore[index]


@pytest.mark.parametrize("field", ["label", "definition", "schema"])
def test_serialized_unknown_facts_are_omitted_instead_of_null(field: str) -> None:
    with pytest.raises(TypeError, match="serialized"):
        _scenario({"id": "s", field: None})


def test_custom_object_and_schema_keys_are_literal_not_descriptor_ids() -> None:
    schema: dict[str, ScientificValue] = {
        "type": "object",
        "patternProperties": {"": {"type": "number"}},
    }
    scenario = Scenario("s", definition={"": 1, " field ": 2}, schema=schema)
    assert scenario.definition is not None
    assert scenario.definition[""] == 1
    assert scenario.definition[" field "] == 2
    assert _scenario(scenario._document()) == scenario
    assert scientific_identity(scenario) != scientific_identity(
        Scenario("s", definition={"": 1, "field": 2}, schema=schema)
    )


def test_scientific_numeric_spelling_does_not_create_new_meaning() -> None:
    first = Scenario("s", definition={"value": 3, "zero": 0})
    second = Scenario("s", definition={"value": 3.0, "zero": -0.0})
    assert scientific_identity(first) == scientific_identity(second)
    period = Period.window(2030, 2040, calendar=Calendar.PROLEPTIC_GREGORIAN)
    document = period._document()
    document["start"] = 2030.0
    document["end"] = 2040.0
    assert _period(document) == period


def test_schema_keyword_spelling_inside_literals_remains_user_data() -> None:
    literal: dict[str, ScientificValue] = {
        "$ref": "https://literal.invalid/data",
        "$schema": "customer-owned value",
    }
    scenario = Scenario("s", definition=literal, schema={"const": literal})
    assert _scenario(scenario._document()) == scenario
    with pytest.raises(ValueError, match="local references"):
        Scenario(
            "s",
            definition={},
            schema={"anyOf": [True, {"$ref": "https://remote.invalid/schema"}]},
        )
