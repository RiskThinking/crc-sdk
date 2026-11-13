"""Risk measures over years: the VaR tolerance rule, CVaR and tail weights."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric._risk import (
    PROBABILITY_TOLERANCE,
    Draws,
    cvar,
    expectation,
    lower_quantile,
    tail_weights,
    var,
)

TENTHS = np.full(10, 0.1)
VALUES = np.arange(1.0, 11.0)


def test_cumulative_tenths_fall_short_of_the_step() -> None:
    # The premise of the tolerance rule: summed tenths miss 0.8 from below.
    assert np.cumsum(TENTHS)[7] < 0.8


@pytest.mark.parametrize(
    ("beta", "expected"),
    [(0.1, 1.0), (0.3, 3.0), (0.5, 5.0), (0.8, 8.0), (0.9, 9.0), (0.95, 10.0)],
)
def test_var_on_a_step_gives_the_lower_value(beta: float, expected: float) -> None:
    # A strict comparison of the summed probabilities would give 9 at 0.8.
    assert var(VALUES, TENTHS, beta) == expected


def test_var_between_steps_and_with_unsorted_ties() -> None:
    values = np.array([5.0, 1.0, 5.0, 3.0])
    probability = np.array([0.1, 0.4, 0.2, 0.3])
    assert var(values, probability, 0.39) == 1.0
    assert var(values, probability, 0.4) == 1.0
    assert var(values, probability, 0.41) == 3.0
    assert var(values, probability, 0.7 + 0.5 * PROBABILITY_TOLERANCE) == 3.0
    assert var(values, probability, 0.71) == 5.0


def test_lower_quantile_takes_level_zero_and_returns_positive_zero() -> None:
    assert lower_quantile(VALUES, TENTHS, 0.0) == 1.0
    result = lower_quantile(np.array([-0.0, 1.0]), np.array([0.5, 0.5]), 0.25)
    assert math.copysign(1.0, result) == 1.0


def test_cvar_is_the_mean_of_the_worst_tail_whichever_atom_the_var_takes() -> None:
    assert cvar(VALUES, TENTHS, 0.8) == pytest.approx(9.5, abs=1e-12)
    assert cvar(VALUES, TENTHS, 0.9) == pytest.approx(10.0, abs=1e-12)
    # 0.85 splits the atom at 9: half of its mass is in the tail.
    assert cvar(VALUES, TENTHS, 0.85) == pytest.approx((0.05 * 9 + 0.1 * 10) / 0.15)


def test_tail_weights_split_the_atom_at_the_var() -> None:
    values = np.array([0.0, 0.0, 0.0, 10.0])
    probability = np.full(4, 0.25)
    weights = tail_weights(values, probability, 0.5)
    np.testing.assert_allclose(weights, [0.25 / 3, 0.25 / 3, 0.25 / 3, 0.25])
    assert math.fsum(weights.tolist()) == pytest.approx(0.5)
    assert cvar(values, probability, 0.5) == pytest.approx(5.0)
    assert math.fsum((weights * values).tolist()) / 0.5 == pytest.approx(5.0)


def test_tail_weights_sum_to_the_tail_mass() -> None:
    rng = np.random.default_rng(7)
    values = rng.integers(0, 20, 500).astype(np.float64)
    probability = rng.random(500)
    probability /= probability.sum()
    for beta in (0.5, 0.9, 0.99):
        weights = tail_weights(values, probability, beta)
        assert weights.sum() == pytest.approx(1.0 - beta, abs=1e-12)
        assert (weights >= 0).all() and (weights <= probability + 1e-15).all()
        assert (weights * values).sum() / (1 - beta) == pytest.approx(
            cvar(values, probability, beta)
        )


def test_draws_share_one_sort_and_expectation_is_the_weighted_mean() -> None:
    draws = Draws(VALUES, TENTHS)
    assert draws.mean() == pytest.approx(5.5)
    assert expectation(VALUES, TENTHS) == draws.mean()
    assert [draws.quantile(level) for level in (0.5, 0.8)] == [5.0, 8.0]


@pytest.mark.parametrize(
    ("values", "probability"),
    [
        (np.array([]), np.array([])),
        (np.ones((2, 2)), np.full((2, 2), 0.25)),
        (np.ones(3), np.full(2, 0.5)),
        (np.array([1.0, np.nan]), np.full(2, 0.5)),
        (np.ones(2), np.array([1.0, 0.0])),
        (np.ones(2), np.array([np.inf, 0.5])),
    ],
)
def test_invalid_arrays_raise(values: Any, probability: Any) -> None:
    with pytest.raises(ValueError):
        Draws(values, probability)


@pytest.mark.parametrize("beta", [0.0, 1.0, -0.1, math.nan, True, "0.9"])
def test_invalid_levels_raise(beta: Any) -> None:
    with pytest.raises(ValueError):
        var(VALUES, TENTHS, beta)
    with pytest.raises(ValueError):
        cvar(VALUES, TENTHS, beta)
    with pytest.raises(ValueError):
        tail_weights(VALUES, TENTHS, beta)


def test_quantile_level_one_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        lower_quantile(VALUES, TENTHS, 1.0)
