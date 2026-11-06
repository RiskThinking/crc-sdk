"""Design constraints: the budget and the limit."""

from __future__ import annotations

from typing import Any

import pytest

from crc_sdk.parametric import Budget, CoverDefinitionError, Limit
from crc_sdk.parametric.constraints import budget_from_document, limit_from_document


def test_budget_bounds() -> None:
    assert Budget(ratio=0.8).bound(250.0) == pytest.approx(200.0)
    assert Budget(amount=120.0).bound(250.0) == 120.0
    assert Budget(ratio=0.0).bound(10.0) == 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"ratio": 0.5, "amount": 1.0},
        {"ratio": -0.1},
        {"amount": -1.0},
        {"ratio": float("inf")},
    ],
)
def test_budget_validation(arguments: dict[str, Any]) -> None:
    with pytest.raises(CoverDefinitionError):
        Budget(**arguments)


def test_limit_validation() -> None:
    assert Limit(10.0).amount == 10.0
    for amount in (-1.0, float("nan")):
        with pytest.raises(CoverDefinitionError):
            Limit(amount)


def test_documents_round_trip() -> None:
    for budget in (Budget(ratio=0.8), Budget(amount=1e6)):
        assert budget_from_document(budget.to_document()) == budget
    assert Budget(ratio=0.8).to_document() == {"ratio": 0.8}
    assert Budget(amount=5.0).to_document() == {"amount": 5.0}
    assert budget_from_document(None) is None
    assert limit_from_document(None) is None
    assert limit_from_document({"amount": 3.0}) == Limit(3.0)
    assert Limit(3.0).to_document() == {"amount": 3.0}
    budgets: list[Any] = [{"loading": 0.8}, {"ratio": 0.8, "amount": 1.0}, [0.8]]
    for document in budgets:
        with pytest.raises(CoverDefinitionError):
            budget_from_document(document)
    limits: list[Any] = [{"value": 3.0}, 3.0]
    for document in limits:
        with pytest.raises(CoverDefinitionError):
            limit_from_document(document)
