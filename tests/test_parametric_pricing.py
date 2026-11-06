"""Pricing: the pure premium and the cost-of-capital load."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import CostOfCapitalLoad, CoverDefinitionError, PurePremium
from crc_sdk.parametric.pricing import pricing_from_document

UNTYPED_LOAD: Any = CostOfCapitalLoad
PAYOUT = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
PROBABILITY = np.full(5, 0.2)


def test_pure_premium_is_the_expected_payout() -> None:
    assert PurePremium().premium(PAYOUT, PROBABILITY) == pytest.approx(2.0)
    assert PurePremium().to_document() == {"kind": "pure"}
    assert PurePremium() == PurePremium()


def test_cost_of_capital_adds_a_load_on_the_payout_tail() -> None:
    pricing = CostOfCapitalLoad(0.5, 0.8)
    # CVaR_0.8 of the payout is the worst 20 %: the year paying 10.
    assert pricing.premium(PAYOUT, PROBABILITY) == pytest.approx(0.5 * 2 + 0.5 * 10)
    assert CostOfCapitalLoad(0.0, 0.9).premium(PAYOUT, PROBABILITY) == pytest.approx(
        2.0
    )
    assert CostOfCapitalLoad(rate=1.0, q=0.5).premium(PAYOUT, PROBABILITY) == (
        pytest.approx(4.0)
    )
    assert pricing.to_document() == {"kind": "cost_of_capital", "q": 0.8, "rate": 0.5}


@pytest.mark.parametrize(
    "document", [{"kind": "pure"}, {"kind": "cost_of_capital", "q": 0.99, "rate": 0.06}]
)
def test_documents_round_trip(document: dict[str, Any]) -> None:
    assert pricing_from_document(document).to_document() == document


@pytest.mark.parametrize(
    ("rate", "q"),
    [
        (-0.1, 0.9),
        (1.1, 0.9),
        (0.1, 0.0),
        (0.1, 1.0),
        (math.nan, 0.9),
        (True, 0.9),
        ("0.1", 0.9),
    ],
)
def test_invalid_terms_raise(rate: Any, q: Any) -> None:
    with pytest.raises(CoverDefinitionError):
        UNTYPED_LOAD(rate, q)


@pytest.mark.parametrize(
    "document",
    [
        None,
        {"kind": "loaded"},
        {"kind": "pure", "rate": 0.1},
        {"kind": "cost_of_capital", "rate": 0.1},
        {"kind": "cost_of_capital", "q": 2.0, "rate": 0.1},
    ],
)
def test_invalid_documents_raise(document: Any) -> None:
    with pytest.raises(CoverDefinitionError):
        pricing_from_document(document)


def test_invalid_payouts_raise() -> None:
    with pytest.raises(ValueError):
        PurePremium().premium(np.array([1.0, np.nan]), np.full(2, 0.5))
    with pytest.raises(ValueError):
        CostOfCapitalLoad(0.1, 0.9).premium(np.ones(3), np.full(2, 0.5))
