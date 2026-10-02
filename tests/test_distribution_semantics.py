from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from crc_sdk.types import TemporalWindow
from crc_sdk.workflows import (
    HorizonExtrapolationWarning,
    ProbabilitySemanticsWarning,
    check_return_period_semantics,
    warn_if_outside_window,
)


@pytest.mark.parametrize(
    "semantics", ["within_period_percentile", "projection_uncertainty"]
)
def test_return_period_warns_then_refuses_under_strict(semantics: str) -> None:
    with pytest.warns(ProbabilitySemanticsWarning, match=semantics):
        assert check_return_period_semantics(semantics) is False
    with pytest.raises(ValueError, match=semantics):
        check_return_period_semantics(semantics, strict=True)


@pytest.mark.parametrize(
    "semantics",
    [
        None,
        "annual_exceedance",
        "annual_value_distribution",
        "estimate_confidence",
    ],
)
def test_return_period_silent_for_supported_or_unspecified(
    semantics: str | None,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_return_period_semantics(semantics, strict=True) is True


def test_return_period_accepts_metadata_objects_and_mappings() -> None:
    meta = SimpleNamespace(probability_semantics="projection_uncertainty")
    with pytest.raises(ValueError):
        check_return_period_semantics(meta, strict=True)
    with pytest.raises(ValueError):
        check_return_period_semantics(
            {"probability_semantics": "within_period_percentile"}, strict=True
        )
    # Old metadata without the field: unchanged behaviour.
    assert check_return_period_semantics(SimpleNamespace(), strict=True)
    assert check_return_period_semantics({}, strict=True)


def test_horizon_inside_window_is_silent_and_bounds_are_inclusive() -> None:
    window = TemporalWindow(kind="window", start_year=2041, end_year=2060)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for horizon in (2041, 2050, 2060):
            assert warn_if_outside_window(horizon, window, strict=True)
        assert warn_if_outside_window(1900, None, strict=True)


@pytest.mark.parametrize("horizon", [2040, 2061])
def test_horizon_outside_window_warns_or_raises(horizon: int) -> None:
    window = TemporalWindow(kind="window", start_year=2041, end_year=2060)
    with pytest.warns(HorizonExtrapolationWarning, match="2041-2060"):
        assert warn_if_outside_window(horizon, window) is False
    with pytest.raises(ValueError, match=str(horizon)):
        warn_if_outside_window(horizon, window, strict=True)


def test_time_invariant_window_only_matches_reference_year() -> None:
    window = TemporalWindow(kind="time_invariant", reference_year=2025)
    assert warn_if_outside_window(2025, window, strict=True)
    with pytest.warns(HorizonExtrapolationWarning, match="reference year 2025"):
        warn_if_outside_window(2050, window)


def test_horizon_must_be_an_integer() -> None:
    window = TemporalWindow(kind="window", start_year=2041, end_year=2060)
    with pytest.raises(TypeError):
        warn_if_outside_window(2050.0, window)  # type: ignore[arg-type]
