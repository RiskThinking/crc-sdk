"""The loss model: crc-sdk impacts on the hazard, times the asset value."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact, TransformContext, impacts

from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import LossModel, SampleError
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.workflows import AssetPortfolio, CellColumn, ImpactContextColumns

UNTYPED_LOSS: Any = LossModel  # for calls with deliberately invalid arguments
RAMP = PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0])
HAZARD = np.array([20.0, 40.0, 70.0, 100.0, 150.0])
RAMP_RATIO = np.array([0.0, 0.0, 0.5, 1.0, 1.0])

ASSETS = pa.table(
    {
        "asset_id": ["warehouse-a", "warehouse-b"],
        "longitude": [6.96, 7.10],
        "latitude": [50.94, 50.73],
        "value": [2.0e6, 5.0e5],
        "country": ["DEU", None],
        "building": ["commercial", "residential"],
    }
)


class ContextImpact:
    """An ImpactFunction that reads its context: hazard × historic mean."""

    def __init__(self) -> None:
        self.context = TransformContext(historic_mean=1.0, country="FRA")
        self.seen: list[TransformContext | None] = []

    def evaluate(self, values: Any, *, context: TransformContext | None = None) -> Any:
        self.seen.append(context)
        active = context or self.context
        assert active.historic_mean is not None
        return np.asarray(values) * 0.01 * active.historic_mean


class FirstOnly:
    """An ImpactFunction that drops all but the first value."""

    def evaluate(self, values: Any, *, context: TransformContext | None = None) -> Any:
        del context
        return np.asarray(values)[:1]


def _expression() -> ImpactExpression:
    return ImpactExpression.linear(
        0.01,
        calibration=ImpactCalibration(
            exposure_variable="rx1day",
            exposure_unit="mm/day",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
        clamp={"minimum": 0.0, "maximum": 1.0},
    )


def test_damage_ratio_without_a_value() -> None:
    model = LossModel(RAMP)
    np.testing.assert_allclose(model.impact_values(HAZARD), RAMP_RATIO)
    np.testing.assert_allclose(model.evaluate(HAZARD), RAMP_RATIO)
    assert model.value_unit == "1"
    document = model.to_document()
    assert document["impact"] == {
        "kind": "function",
        "type": "PiecewiseLinearImpact",
        "name": "PiecewiseLinearImpact",
        "definition_hash": None,
        "exposure_unit": None,
    }
    assert document["value"] == {
        "source": "none",
        "column": None,
        "unit": "1",
        "asset_value_hash": None,
    }


def test_a_stated_value_scales_the_loss_and_is_only_hashed() -> None:
    model = LossModel(RAMP, value=1e6, value_unit="EUR")
    np.testing.assert_allclose(model.evaluate(HAZARD), RAMP_RATIO * 1e6)
    document = model._document("warehouse-a")
    assert document["value"]["source"] == "constant"
    assert document["value"]["asset_value_hash"] == content_hash(
        {"asset_id": "warehouse-a", "value": 1e6}
    )
    assert "1000000" not in str(document)


def test_an_expression_records_its_definition_hash() -> None:
    expression = _expression()
    model = LossModel(expression)
    np.testing.assert_allclose(
        model.impact_values(HAZARD), np.clip(HAZARD * 0.01, 0, 1)
    )
    impact = model.to_document()["impact"]
    assert impact["kind"] == "expression"
    assert impact["definition_hash"] == expression.definition_hash
    assert impact["name"] == "linear"
    assert impact["exposure_unit"] == "mm/day"


def test_a_registry_impact_and_a_callable() -> None:
    cyclone = impacts.for_factor(
        "cyclone",
        context=TransformContext(
            country="DEU", continent="Europe", building_type="residential"
        ),
    )
    wind = np.array([0.5, 50.0, 300.0])
    np.testing.assert_allclose(
        LossModel(cyclone).impact_values(wind), cyclone.evaluate(wind)
    )
    assert LossModel(cyclone).name == "cyclone"

    def half(values: Any) -> Any:
        return np.asarray(values) * 0.5

    model = LossModel(half, name="half")
    np.testing.assert_allclose(model.impact_values([1.0, 2.0]), [0.5, 1.0])
    assert model.to_document()["impact"]["name"] == "half"
    assert LossModel(half).name.endswith("half")


def test_the_value_column_needs_an_asset() -> None:
    model = LossModel(RAMP, value_column="value", value_unit="EUR")
    with pytest.raises(SampleError, match="for_asset"):
        model.evaluate(HAZARD)
    bound = model.for_asset(AssetPortfolio(ASSETS), "warehouse-b")
    assert bound.asset_id == "warehouse-b"
    np.testing.assert_allclose(bound.evaluate(HAZARD), RAMP_RATIO * 5e5)
    document = bound.to_document()
    assert document["value"]["source"] == "column"
    assert document["value"]["column"] == "value"
    assert document["asset_id"] == "warehouse-b"
    assert model.asset_id is None  # binding returns a copy


def test_a_plain_table_uses_the_asset_id_column() -> None:
    bound = LossModel(RAMP, value_column="value", value_unit="EUR").for_asset(
        ASSETS, "warehouse-a"
    )
    np.testing.assert_allclose(bound.evaluate([100.0]), [2.0e6])


def test_unmatched_and_repeated_assets_raise() -> None:
    model = LossModel(RAMP, value_column="value", value_unit="EUR")
    with pytest.raises(SampleError, match="not in the asset table"):
        model.for_asset(ASSETS, "warehouse-z")
    twice = pa.concat_tables([ASSETS, ASSETS])
    with pytest.raises(SampleError, match="appears 2 times"):
        model.for_asset(twice, "warehouse-a")


@pytest.mark.parametrize("value", [None, -1.0, float("nan")])
def test_invalid_asset_values_raise(value: float | None) -> None:
    assets = pa.table({"asset_id": ["a"], "value": pa.array([value], pa.float64())})
    with pytest.raises(SampleError, match="value"):
        LossModel(RAMP, value_column="value", value_unit="EUR").for_asset(assets, "a")


def test_a_missing_value_column_raises() -> None:
    with pytest.raises(SampleError, match="no column 'worth'"):
        LossModel(RAMP, value_column="worth", value_unit="EUR").for_asset(
            ASSETS, "warehouse-a"
        )


def test_asset_files_are_not_read(tmp_path: Path) -> None:
    path = tmp_path / "assets.parquet"
    pq.write_table(ASSETS, path)
    with pytest.raises(TypeError, match="no I/O"):
        LossModel(RAMP).for_asset(AssetPortfolio(path), "warehouse-a")


def test_context_columns_fill_the_framework_context() -> None:
    impact = ContextImpact()
    assets = pa.table(
        {
            "asset_id": ["a", "b"],
            "cell_index": [0x8728308281FFFFF, 0x8728308283FFFFF],
            "mean": [2.0, None],
            "country": ["DEU", "BEL"],
        }
    )
    portfolio = AssetPortfolio(assets, location=CellColumn())
    columns = ImpactContextColumns(historic_mean="mean", country="country")
    model = LossModel(impact, context=columns)
    with pytest.raises(SampleError, match="for_asset"):
        model.impact_values([10.0])
    np.testing.assert_allclose(
        model.for_asset(portfolio, "a").impact_values([10.0]), [0.2]
    )
    context = impact.seen[-1]
    assert context == TransformContext(
        cell=0x8728308281FFFFF, country="DEU", historic_mean=2.0
    )
    # A null column keeps the impact's own context value.
    np.testing.assert_allclose(
        model.for_asset(portfolio, "b").impact_values([10.0]), [0.1]
    )
    assert impact.seen[-1] is not None and impact.seen[-1].historic_mean == 1.0
    resolved = model.for_asset(portfolio, "a").to_document()["context"]
    assert resolved["columns"] == {"country": "country", "historic_mean": "mean"}
    assert resolved["resolved"]["cell"] == "8728308281fffff"


def test_a_fixed_context_is_passed() -> None:
    impact = ContextImpact()
    context = TransformContext(historic_mean=3.0)
    np.testing.assert_allclose(
        LossModel(impact, context=context).impact_values([10.0]), [0.3]
    )
    assert impact.seen[-1] == context


def test_expressions_and_callables_take_no_context() -> None:
    with pytest.raises(SampleError, match="carry their own context"):
        LossModel(_expression(), context=ImpactContextColumns(country="country"))
    with pytest.raises(SampleError, match="takes no context"):
        LossModel(lambda values: values, context=TransformContext(country="DEU"))


def test_invalid_definitions_raise() -> None:
    with pytest.raises(TypeError, match="ImpactExpression"):
        UNTYPED_LOSS(42)
    with pytest.raises(SampleError, match="not both"):
        LossModel(RAMP, value_column="value", value=1.0)
    with pytest.raises(SampleError, match="damage ratio"):
        LossModel(RAMP, value_unit="EUR")
    with pytest.raises(SampleError, match="value"):
        LossModel(RAMP, value=-5.0, value_unit="EUR")
    with pytest.raises(TypeError, match="context"):
        UNTYPED_LOSS(RAMP, context={"country": "DEU"})


def test_negative_or_non_finite_impacts_raise() -> None:
    with pytest.raises(SampleError, match="negative or non-finite"):
        LossModel(lambda values: np.asarray(values) - 50.0).impact_values(HAZARD)
    with pytest.raises(SampleError, match="hazard values must be finite"):
        LossModel(RAMP).impact_values([np.nan])
    with pytest.raises(SampleError, match="shape"):
        LossModel(FirstOnly()).impact_values(HAZARD)
    with pytest.raises(SampleError, match="finite and non-negative"):
        LossModel(RAMP, value=1e308, value_unit="EUR").loss_values([10.0])


# -- the hazard unit ---------------------------------------------------------


def test_the_hazard_unit_must_be_the_expressions_calibration_unit() -> None:
    model = LossModel(_expression())
    assert model.exposure_unit == "mm/day"
    model.check_hazard_unit("mm/day")
    np.testing.assert_allclose(
        model.evaluate(HAZARD, hazard_unit="mm/day"), model.evaluate(HAZARD)
    )
    for unit in ("mm", "MM/day", "m"):
        with pytest.raises(SampleError, match="calibrated for hazard values in"):
            model.check_hazard_unit(unit)
    with pytest.raises(SampleError, match="'mm'"):
        model.impact_values(HAZARD, hazard_unit="mm")
    with pytest.raises(SampleError, match="calibrated"):
        model.evaluate(HAZARD, hazard_unit="mm")
    with pytest.raises(SampleError, match="hazard_unit"):
        model.check_hazard_unit(" mm/day")
    # The framework's expression declares the same unit.
    framework = LossModel(_expression().framework_expression)
    assert framework.exposure_unit == "mm/day"
    with pytest.raises(SampleError, match="calibrated"):
        framework.check_hazard_unit("mm")
    # Binding to an asset keeps the check.
    bound = LossModel(_expression(), value_column="value", value_unit="EUR")
    with pytest.raises(SampleError, match="calibrated"):
        bound.for_asset(ASSETS, "warehouse-a").check_hazard_unit("mm")


def test_units_compare_through_the_framework_aliases() -> None:
    depth = ImpactExpression.linear(
        0.5,
        calibration=ImpactCalibration(
            exposure_variable="flood_depth",
            exposure_unit="m",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
    )
    model = LossModel(depth)
    model.check_hazard_unit("metres")
    model.check_hazard_unit("m")
    with pytest.raises(SampleError, match="calibrated"):
        model.check_hazard_unit("cm")
    registry = ImpactExpression.for_factor(
        "inundation",
        context_defaults={
            "continent": "Europe",
            "building_type": "Commercial buildings",
        },
    )
    assert LossModel(registry).exposure_unit == "m"
    LossModel(registry).check_hazard_unit("metres")


def test_impacts_that_declare_no_unit_are_not_checked() -> None:
    def half(values: Any) -> Any:
        return np.asarray(values) * 0.5

    for model in (LossModel(RAMP), LossModel(half), LossModel(ContextImpact())):
        assert model.exposure_unit is None
        model.check_hazard_unit("any unit")
        assert model.to_document()["impact"]["exposure_unit"] is None
    np.testing.assert_allclose(
        LossModel(RAMP).evaluate(HAZARD, hazard_unit="mm/day"), RAMP_RATIO
    )
