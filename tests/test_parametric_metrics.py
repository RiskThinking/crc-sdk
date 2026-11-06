"""Metric options, column names and the metric kernels in closed form."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import Cover, IndexDefinition, LinearPayout, MetricOptions
from crc_sdk.parametric.metrics import metric_columns, metric_values

UNTYPED_OPTIONS: Any = MetricOptions

# A linear payout on a uniform index: years 0..99 with index x = d and loss
# L = x, attachment 50, exhaustion 90, limit 40, so payout = clip(x - 50, 0, 40).
N = 100
INDEX = np.arange(N, dtype=np.float64)
LOSS = INDEX.copy()
PROBABILITY = np.full(N, 1.0 / N)
COVER = Cover(
    IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum"),
    LinearPayout(50.0, 90.0, 40.0),
    payout_unit="EUR",
)
PAYOUT = COVER.payout_values(INDEX)


def _values(premium: float, **options: Any) -> dict[str, float | None]:
    return metric_values(
        LOSS, PAYOUT, PROBABILITY, premium=premium, options=MetricOptions(**options)
    )


def test_default_options_give_the_adr_columns_in_order() -> None:
    names = [column.name for column in metric_columns(MetricOptions())]
    assert len(names) == 62
    assert names[:6] == [
        "expected_loss",
        "expected_payout",
        "premium",
        "expected_uncovered",
        "coverage_ratio",
        "payout_to_loss_ratio",
    ]
    assert names[6:14] == [
        "tail_coverage_b0_9",
        "tail_coverage_b0_95",
        "tail_coverage_b0_99",
        "cvar_uncovered_to_loss_b0_9",
        "cvar_uncovered_to_loss_b0_95",
        "cvar_uncovered_to_loss_b0_99",
        "false_negative_rate_l0",
        "false_positive_rate_l0",
    ]
    assert names[14:16] == ["rmse", "correlation"]
    assert names[16:19] == [
        "var_uncovered_b0_9",
        "var_uncovered_b0_95",
        "var_uncovered_b0_99",
    ]
    assert names[-1] == "net_loss_rp250"
    assert (
        "loss_rp2" in names
        and "payout_rp100" in names
        and "cvar_net_loss_b0_99" in names
    )
    assert len(set(names)) == len(names)


def test_labels_follow_the_return_period_column_rule() -> None:
    options = MetricOptions(
        betas=(0.995, 0.5),
        return_periods=(2.5, 1000),
        loss_thresholds=(0, 1e-05, 2.5e6),
    )
    names = {column.name for column in metric_columns(options)}
    assert {
        "var_loss_b0_995",
        "cvar_net_loss_b0_5",
        "loss_rp2_5",
        "net_loss_rp1000",
    } <= names
    assert {
        "false_negative_rate_l0",
        "false_negative_rate_l0_00001",
        "false_positive_rate_l2500000",
    } <= names


def test_columns_document_metric_parameter_unit_and_definition() -> None:
    columns = {column.name: column for column in metric_columns(MetricOptions())}
    tail = columns["tail_coverage_b0_99"].to_document("EUR")
    assert tail["metric"] == "tail_coverage"
    assert tail["parameter"] == {"beta": 0.99}
    assert tail["unit"] == "1"
    assert columns["loss_rp100"].to_document("EUR") == {
        "definition": columns["loss_rp100"].definition,
        "metric": "loss_rp",
        "parameter": {"return_period": 100.0},
        "unit": "EUR",
    }
    assert columns["false_negative_rate_l0"].parameter == {"loss_threshold": 0.0}
    assert columns["rmse"].to_document("EUR")["parameter"] is None
    assert "crc-framework" in columns["cvar_uncovered_b0_9"].definition


def test_a_linear_payout_on_a_uniform_index_in_closed_form() -> None:
    premium = 11.8  # the pure premium, E[payout]
    values = _values(premium)
    exact = {
        "expected_loss": 49.5,
        "expected_payout": (780 + 400) / 100,  # sum 1..39, plus 10 years at 40
        "premium": premium,
        "expected_uncovered": 49.5 - 11.8,
        "coverage_ratio": 11.8 / 49.5,  # the payout never exceeds the loss
        "payout_to_loss_ratio": 11.8 / 49.5,
        "tail_coverage_b0_9": 400 / 945,  # years 90..99
        "tail_coverage_b0_99": 40 / 99,
        "cvar_uncovered_to_loss_b0_9": 54.5 / 94.5,
        "false_negative_rate_l0": 50 / 99,  # years 1..50 lose and get nothing
        "false_positive_rate_l0": 0.0,
        "rmse": math.sqrt((42925 + 39 * 2500 + 29785) / 100),
        "var_uncovered_b0_9": 50.0,
        "cvar_uncovered_b0_9": 54.5,
        "var_net_loss_b0_9": 50.0 + premium,
        "cvar_net_loss_b0_9": 54.5 + premium,
        "var_loss_b0_9": 89.0,  # 0.9 lies on a step: the lower value
        "cvar_loss_b0_9": 94.5,
        "var_loss_b0_99": 98.0,
        "loss_rp2": 49.0,
        "loss_rp10": 89.0,
        "loss_rp100": 98.0,
        "loss_rp250": 99.0,
        "payout_rp2": 0.0,
        "payout_rp10": 39.0,  # on a step again: 39, not 40
        "uncovered_rp10": 50.0,
        "net_loss_rp10": 50.0 + premium,
    }
    for name, expected in exact.items():
        assert values[name] == pytest.approx(expected, rel=1e-12, abs=1e-12), name
    assert values["correlation"] == pytest.approx(np.corrcoef(LOSS, PAYOUT)[0, 1])
    assert list(values) == [column.name for column in metric_columns(MetricOptions())]


def test_payouts_above_the_loss_count_only_in_the_payout_ratio() -> None:
    loss = np.array([0.0, 10.0])
    payout = np.array([5.0, 20.0])
    values = metric_values(
        loss, payout, np.full(2, 0.5), premium=0.0, options=MetricOptions()
    )
    assert values["coverage_ratio"] == pytest.approx(1.0)
    assert values["payout_to_loss_ratio"] == pytest.approx(2.5)
    assert values["false_positive_rate_l0"] == pytest.approx(1.0)
    assert values["expected_uncovered"] == 0.0


def test_thresholds_are_stated() -> None:
    values = _values(0.0, loss_thresholds=(0.0, 60.0), payout_threshold=5.0)
    # Payouts of at most 5 count as none: years 0..55.
    assert values["false_negative_rate_l60"] == 0.0  # years 61.. pay over 5
    assert values["false_positive_rate_l60"] == pytest.approx(5 / 61)  # 56..60
    assert values["false_negative_rate_l0"] == pytest.approx(55 / 99)


def test_undefined_metrics_are_null() -> None:
    zero = metric_values(
        np.zeros(4), np.zeros(4), np.full(4, 0.25), premium=0.0, options=MetricOptions()
    )
    for name in (
        "coverage_ratio",
        "payout_to_loss_ratio",
        "tail_coverage_b0_9",
        "cvar_uncovered_to_loss_b0_9",
        "false_negative_rate_l0",
        "correlation",
    ):
        assert zero[name] is None, name
    assert zero["false_positive_rate_l0"] == 0.0
    assert zero["rmse"] == 0.0
    flat_payout = metric_values(
        np.array([0.0, 1.0]),
        np.zeros(2),
        np.full(2, 0.5),
        premium=0.0,
        options=MetricOptions(loss_thresholds=(5.0,)),
    )
    assert flat_payout["correlation"] is None
    assert flat_payout["false_negative_rate_l5"] is None  # no loss above 5
    assert flat_payout["false_positive_rate_l5"] == 0.0


def test_options_document() -> None:
    assert MetricOptions().to_document() == {
        "betas": [0.9, 0.95, 0.99],
        "loss_thresholds": [0.0],
        "payout_threshold": 0.0,
        "return_periods": [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0],
    }
    assert MetricOptions(betas=()).to_document()["betas"] == []


@pytest.mark.parametrize(
    "options",
    [
        {"betas": (0.0,)},
        {"betas": (1.0,)},
        {"betas": (0.9, 0.9)},
        {"betas": (math.nan,)},
        {"betas": ("0.9",)},
        {"return_periods": (1.0,)},
        {"return_periods": (10, 10.0)},
        {"loss_thresholds": (-1.0,)},
        {"loss_thresholds": (0.0, 0.0)},
        {"payout_threshold": -0.5},
        {"payout_threshold": True},
        {"alpha": 0.9},
    ],
)
def test_invalid_options_raise(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        UNTYPED_OPTIONS(**options)


@pytest.mark.parametrize(
    ("loss", "payout", "probability", "premium"),
    [
        (np.ones(2), np.ones(3), np.full(2, 0.5), 0.0),
        (np.ones(2), np.ones(2), np.full(2, 0.4), 0.0),
        (np.array([-1.0, 1.0]), np.ones(2), np.full(2, 0.5), 0.0),
        (np.ones(2), np.array([1.0, -1.0]), np.full(2, 0.5), 0.0),
        (np.ones(2), np.ones(2), np.full(2, 0.5), math.inf),
        (np.ones(2), np.ones(2), np.full(2, 0.5), True),
    ],
)
def test_invalid_inputs_raise(
    loss: Any, payout: Any, probability: Any, premium: Any
) -> None:
    with pytest.raises(ValueError):
        metric_values(
            loss, payout, probability, premium=premium, options=MetricOptions()
        )
