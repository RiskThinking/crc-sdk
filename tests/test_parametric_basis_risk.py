"""Basis-risk models: the protocol, no basis risk and lognormal impact noise."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import BasisRisk, ImpactNoise, NoBasisRisk, SampleError

UNTYPED_NOISE: Any = ImpactNoise  # for calls with deliberately invalid arguments


def test_built_in_models_meet_the_protocol() -> None:
    assert isinstance(NoBasisRisk(), BasisRisk)
    assert isinstance(ImpactNoise(0.2), BasisRisk)


def test_no_basis_risk_changes_nothing_and_draws_nothing() -> None:
    index = np.array([1.0, 2.0, 3.0])
    impact = np.array([0.0, 0.1, 0.5])
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    new_index, new_impact = NoBasisRisk().apply(index, index, impact, rng)
    assert new_index is index and new_impact is impact
    assert rng.bit_generator.state == state
    assert NoBasisRisk().to_document() == {"kind": "no_basis_risk"}
    assert NoBasisRisk().name == "no_basis_risk"


def test_impact_noise_is_the_mean_one_lognormal_factor() -> None:
    sigma = 0.4
    impact = np.full(200_000, 0.25)
    index = np.arange(impact.size, dtype=np.float64)
    new_index, noisy = ImpactNoise(sigma).apply(
        index, index, impact, np.random.default_rng(11)
    )
    assert new_index is index
    factor = noisy / impact
    # One rng.normal(-sigma**2/2, sigma) call, as the thesis noise layer.
    expected = np.exp(
        np.random.default_rng(11).normal(-0.5 * sigma**2, sigma, impact.size)
    )
    np.testing.assert_array_equal(factor, expected)
    assert abs(factor.mean() - 1.0) < 0.01
    assert abs(np.log(factor).std() - sigma) < 0.01


def test_impact_noise_keeps_zero_impact_and_caps() -> None:
    impact = np.array([0.0, 0.9, 0.95, 0.99] * 1000)
    _, noisy = ImpactNoise(0.5, upper=1.0).apply(
        impact, impact, impact, np.random.default_rng(0)
    )
    assert (noisy[impact == 0.0] == 0.0).all()
    assert noisy.max() == 1.0
    assert (noisy == 1.0).sum() > 0
    assert ImpactNoise(0.5, upper=1.0).to_document() == {
        "kind": "impact_noise",
        "sigma": 0.5,
        "upper": 1.0,
    }


def test_impact_noise_needs_a_generator() -> None:
    impact = np.array([0.1, 0.2])
    with pytest.raises(SampleError, match="rng"):
        ImpactNoise(0.1).apply(impact, impact, impact, None)


@pytest.mark.parametrize("sigma", [0.0, -0.1, math.inf, math.nan, True, "0.2"])
def test_impact_noise_needs_a_positive_sigma(sigma: Any) -> None:
    with pytest.raises(SampleError, match="sigma"):
        UNTYPED_NOISE(sigma)


@pytest.mark.parametrize("upper", [0.0, -1.0, math.inf])
def test_impact_noise_needs_a_positive_cap(upper: float) -> None:
    with pytest.raises(SampleError, match="upper"):
        ImpactNoise(0.2, upper=upper)


def test_impact_noise_is_frozen() -> None:
    model = ImpactNoise(0.2)
    with pytest.raises(AttributeError):
        model.sigma = 0.3  # type: ignore[misc]
