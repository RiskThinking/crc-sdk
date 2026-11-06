"""Design objectives: their values, documents and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import (
    MSE,
    CoverDefinitionError,
    ExpectedUncovered,
    UserObjective,
    WeightedCVaR,
)
from crc_sdk.parametric.objectives import objective_from_document

LOSS = np.array([0.0, 10.0, 30.0, 40.0])
PAYOUT = np.array([5.0, 10.0, 20.0, 20.0])
PROBABILITY = np.full(4, 0.25)


def test_mse() -> None:
    # Residuals -5, 0, 10, 20.
    assert MSE().value(LOSS, PAYOUT, PROBABILITY) == pytest.approx(131.25)
    assert MSE().homogeneity == 2


def test_expected_uncovered() -> None:
    # Uncovered 0, 0, 10, 20; paying more than the loss earns nothing.
    assert ExpectedUncovered().value(LOSS, PAYOUT, PROBABILITY) == pytest.approx(7.5)
    assert ExpectedUncovered().homogeneity == 1


def test_weighted_cvar() -> None:
    # The worst half of the uncovered loss is 10 and 20: CVaR_0.5 = 15.
    objective = WeightedCVaR(0.9, 0.5)
    assert objective.value(LOSS, PAYOUT, PROBABILITY) == pytest.approx(
        0.9 * 15.0 + 0.1 * 7.5
    )
    assert WeightedCVaR(0.0, 0.5).value(LOSS, PAYOUT, PROBABILITY) == pytest.approx(7.5)
    assert WeightedCVaR(1.0, 0.75).value(LOSS, PAYOUT, PROBABILITY) == pytest.approx(
        20.0
    )
    # A level inside an atom splits it: the worst 0.4 is 20 (0.25) and 10 (0.15).
    assert WeightedCVaR(1.0, 0.6).value(LOSS, PAYOUT, PROBABILITY) == pytest.approx(
        (0.25 * 20.0 + 0.15 * 10.0) / 0.4
    )


@pytest.mark.parametrize(
    ("alpha", "beta"), [(-0.1, 0.5), (1.1, 0.5), (0.5, 0.0), (0.5, 1.0)]
)
def test_weighted_cvar_validation(alpha: float, beta: float) -> None:
    with pytest.raises(CoverDefinitionError):
        WeightedCVaR(alpha, beta)


def test_values_check_their_arrays() -> None:
    for objective in (MSE(), ExpectedUncovered(), WeightedCVaR(0.5, 0.5)):
        with pytest.raises(ValueError, match="one shape"):
            objective.value(LOSS, PAYOUT[:2], PROBABILITY)
        with pytest.raises(ValueError, match="finite"):
            objective.value(LOSS, np.array([np.nan, 0, 0, 0]), PROBABILITY)
        with pytest.raises(ValueError, match="positive"):
            objective.value(LOSS, PAYOUT, np.array([0.5, 0.5, 0.0, 0.0]))


def test_user_objective() -> None:
    calls: list[tuple[Any, ...]] = []

    def mean_absolute(loss: Any, payout: Any, probability: Any) -> float:
        calls.append((loss, payout, probability))
        return float(np.sum(probability * np.abs(loss - payout)))

    objective = UserObjective(mean_absolute, name="mae", parameters={"order": 1.0})
    assert objective.value(LOSS, PAYOUT, PROBABILITY) == pytest.approx(8.75)
    loss, _, _ = calls[0]
    assert not loss.flags.writeable
    assert objective.to_document() == {
        "kind": "user",
        "name": "mae",
        "parameters": {"order": 1.0},
    }
    assert UserObjective(mean_absolute, name="mae").parameters == {}


@pytest.mark.parametrize("result", [float("nan"), "1.0", None, True])
def test_user_objective_results_must_be_finite_numbers(result: Any) -> None:
    objective = UserObjective(lambda loss, payout, p: result, name="bad")
    with pytest.raises(TypeError):
        objective.value(LOSS, PAYOUT, PROBABILITY)


def test_user_objective_validation() -> None:
    with pytest.raises(CoverDefinitionError, match="callable"):
        UserObjective(1.0, name="x")  # type: ignore[arg-type]
    for name in ("", " x", 3):
        with pytest.raises(CoverDefinitionError, match="name"):
            UserObjective(lambda *a: 0.0, name=name)  # type: ignore[arg-type]
    with pytest.raises(CoverDefinitionError, match="JSON"):
        UserObjective(lambda *a: 0.0, name="x", parameters={"f": object()})
    with pytest.raises(CoverDefinitionError, match="mapping"):
        UserObjective(lambda *a: 0.0, name="x", parameters=[1])  # type: ignore[arg-type]


def test_documents_round_trip() -> None:
    for objective in (MSE(), ExpectedUncovered(), WeightedCVaR(0.9, 0.99)):
        assert objective_from_document(objective.to_document()) == objective
    assert WeightedCVaR(0.9, 0.99).to_document() == {
        "alpha": 0.9,
        "beta": 0.99,
        "kind": "weighted_cvar",
    }
    with pytest.raises(CoverDefinitionError, match="user objective"):
        objective_from_document({"kind": "user", "name": "x", "parameters": {}})
    documents: list[Any] = [{"kind": "mae"}, {"kind": "mse", "extra": 1}, [], None]
    for document in documents:
        with pytest.raises(CoverDefinitionError):
            objective_from_document(document)
