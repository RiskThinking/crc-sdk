from __future__ import annotations

from pathlib import Path

import crc_framework as cf
import pytest

from crc_sdk._declaration_updates import refine_spec
from crc_sdk._declarations import (
    Calendar,
    DataSpec,
    Period,
    Probability,
    Scenario,
    SeriesSpec,
    Variable,
)
from crc_sdk._recipes import Dataset, Samples
from crc_sdk._schema import table_to_rows


def quantiles(variable: Variable | None = None) -> Dataset:
    return Dataset.from_quantiles(
        {
            "support_id": ["station"] * 3,
            "longitude": [0.0] * 3,
            "latitude": [0.0] * 3,
            "probability": [0.1, 0.5, 0.9],
            "value": [1.0, 2.0, 3.0],
        },
        variable=variable or Variable("company:depth"),
    )


def samples() -> Samples:
    return Samples.from_table(
        {
            "support_id": ["station"] * 4,
            "longitude": [0.0] * 4,
            "latitude": [0.0] * 4,
            "time": [2000, 2001, 2002, 2003],
            "value": [1.0, 2.0, 3.0, 5.0],
        },
        variable=Variable("company:depth"),
    )


def test_declare_fills_science_without_changing_supplied_values() -> None:
    original = quantiles()
    measurement = cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
    period = Period.window(2000, 2003, calendar=Calendar.GREGORIAN)
    scenario = Scenario("company:scenario", definition={"rainfall": 1.2})
    result = original.declare(
        variable=Variable("company:depth", measurement=measurement),
        scenario=scenario,
        period=period,
        probability=Probability.annual(tail=cf.Tail.UPPER),
    ).read()
    assert result.metadata.spec.variables[0].measurement == measurement
    assert result.metadata.spec.scenarios == (scenario,)
    assert result.metadata.spec.periods == (period,)
    row = table_to_rows(result.table)[0]
    assert row.scenario_id == scenario.id and row.period_id == period.id
    assert row.fit is None and row.curve is not None
    assert row.curve.quantile(0.5) == 2
    unchanged = original.read()
    assert unchanged.metadata.spec.variables[0].measurement is None
    assert unchanged.metadata.spec.periods == ()


def test_declaration_and_rename_builders_are_immutable_snapshots() -> None:
    original = quantiles()
    mapping = {"company:depth": "new:depth"}
    changed = original.rename(variables=mapping)
    mapping["company:depth"] = "wrong:depth"
    assert changed.read().metadata.spec.variables[0].id == "new:depth"
    with pytest.raises(AttributeError, match="immutable"):
        setattr(original, "extra", "mutate")
    with pytest.raises(AttributeError, match="immutable"):
        original._selection = original._selection


def test_declaration_does_not_relabel_known_units_or_ids() -> None:
    original = quantiles(
        Variable(
            "company:depth", measurement=cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
        )
    )
    with pytest.raises(ValueError, match="convert_units"):
        original.declare(
            variable=Variable(
                "company:depth",
                measurement=cf.DepthAboveGround(unit=cf.LengthUnit.CENTIMETRE),
            )
        ).read()
    with pytest.raises(ValueError, match="rename"):
        original.declare(variable=Variable("new:depth")).read()
    with pytest.raises(ValueError, match="rename"):
        original.declare(
            spec=DataSpec(
                [Variable("new:depth")], series=[SeriesSpec(Variable("new:depth"))]
            )
        ).read()


def test_custom_measurement_refinement_retains_known_fields_and_literal_null() -> None:
    original = Variable(
        "company:depth",
        measurement=cf.Measurement(
            quantity="company:water", fields={"reference": None}
        ),
    )
    base = DataSpec([original], series=[SeriesSpec(original)])
    refined, _ = refine_spec(
        base,
        variable=Variable(
            "company:depth",
            measurement=cf.Measurement(unit="m", fields={"method": "sensor"}),
        ),
    )
    measurement = refined.variables[0].measurement
    assert measurement is not None
    assert measurement.quantity == "company:water" and measurement.unit == "m"
    assert measurement.fields == {"reference": None, "method": "sensor"}
    with pytest.raises(ValueError, match="reference"):
        refine_spec(
            base,
            variable=Variable(
                "company:depth", measurement=cf.Measurement(fields={"reference": "new"})
            ),
        )


def test_unknown_generic_measurement_can_become_a_typed_measurement() -> None:
    original = Variable("company:depth", measurement=cf.Measurement())
    typed = Variable(
        "company:depth", measurement=cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
    )
    refined, _ = refine_spec(
        DataSpec([original], series=[SeriesSpec(original)]), variable=typed
    )
    assert refined.variables[0].measurement == typed.measurement


def test_period_calendar_refinement_updates_references_without_changing_bounds() -> (
    None
):
    variable = Variable("company:depth")
    old = Period.window(2000, 2003)
    updated = Period.window(2000, 2003, calendar=Calendar.DAYS_360)
    spec = DataSpec([variable], series=[SeriesSpec(variable, period=old)])
    refined, identities = refine_spec(spec, period=updated)
    assert refined.periods == (updated,)
    assert identities[(variable.id, None, old.id)] == (variable.id, None, updated.id)
    with pytest.raises(ValueError, match="boundaries"):
        refine_spec(spec, period=Period.window(2001, 2003))
    with pytest.raises(ValueError, match="calendar"):
        refine_spec(
            refined, period=Period.window(2000, 2003, calendar=Calendar.GREGORIAN)
        )


def test_probability_direction_cannot_be_changed_through_declaration() -> None:
    variable = Variable("company:depth")
    spec = DataSpec(
        [variable],
        series=[
            SeriesSpec(variable, probability=Probability.annual(tail=cf.Tail.UPPER))
        ],
    )
    with pytest.raises(ValueError, match="probability meaning"):
        refine_spec(spec, probability=Probability.annual(tail=cf.Tail.LOWER))


def test_declaration_spec_form_and_direct_form_are_exclusive() -> None:
    variable = Variable("company:depth")
    spec = DataSpec([variable], series=[SeriesSpec(variable)])
    with pytest.raises(ValueError, match="either spec"):
        quantiles().declare(spec=spec, variable=variable)
    assert quantiles().declare().read().table.num_rows == 1


def test_inspection_of_declared_fit_never_runs_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prohibited(*args: object, **kwargs: object) -> None:
        raise AssertionError("Inspection cannot fit evidence")

    monkeypatch.setattr("crc_sdk._recipes.fit_sample_table", prohibited)
    recipe = (
        samples()
        .declare(
            variable=Variable(
                "company:depth", measurement=cf.Ratio(unit=cf.RatioUnit.FRACTION)
            )
        )
        .fit(cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT))
    )
    inspected = recipe.inspect()
    assert inspected.series[0].measurement == cf.Ratio(unit=cf.RatioUnit.FRACTION)
    assert "company:depth" in repr(recipe.select(variables="company:depth").explain())


def test_fit_kind_validation_is_pure_even_for_empty_selection() -> None:
    sample = samples().select(variables=[])
    dataset = quantiles().select(variables=[])
    with pytest.raises(TypeError, match="sample"):
        sample.fit(cf.QuantileFit(family=cf.DistributionFamily.GUMBEL_RIGHT))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="quantile"):
        dataset.fit(cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sample"):
        sample.fit(
            cf.ModelSelection(
                candidates=(
                    cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT),
                    cf.QuantileFit(family=cf.DistributionFamily.GUMBEL_RIGHT),
                )
            )
        )


def test_reopened_handle_exposes_stored_metadata_without_bulk_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = samples().write(tmp_path / "samples.parquet")
    assert saved.path is not None
    reopened = Samples.open(saved.path)

    def prohibited(*args: object, **kwargs: object) -> None:
        raise AssertionError("Header discovery must not copy or acquire the data")

    monkeypatch.setattr("crc_sdk._graph.ExecutionScope.pin_handle", prohibited)
    assert reopened.metadata is not None
    assert reopened.metadata.spec.variables[0].id == "company:depth"
    assert reopened.report is not None and reopened.report.complete
    assert saved.metadata is not None and saved.metadata.spec == reopened.metadata.spec
