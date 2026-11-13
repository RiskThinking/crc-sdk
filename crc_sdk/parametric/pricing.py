"""Pricing: the premium of a cover's payouts on an outcome sample.

- `PurePremium()`: `P = E[payout]`.
- `CostOfCapitalLoad(rate, q)`: `P = (1 - rate) * E[payout] + rate *
  CVaR_q(payout)`, with `rate` in [0, 1] and `q` in (0, 1). The load is the
  cost of holding capital against the payout's tail. It is convex in the
  payout, so a design under it stays a linear or quadratic programme.

The CVaR is over the years of the sample, as everywhere in this module (see
`crc_sdk.parametric.metrics`); it is not crc-framework's CVaR over binary
outcomes. Both pricings are frozen value objects with a JSON document
(`to_document()`), which evaluation metadata records next to every premium.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self, TypeAlias

from numpy.typing import ArrayLike
from pydantic import model_validator

from crc_sdk.impacts._types import FiniteFloat
from crc_sdk.parametric._risk import Draws
from crc_sdk.parametric.errors import CoverDefinitionError
from crc_sdk.parametric.parameters import _DefinitionModel

__all__ = ["CostOfCapitalLoad", "Pricing", "PurePremium", "pricing_from_document"]


class PurePremium(_DefinitionModel):
    """The pure premium: the expected payout, `P = E[payout]`."""

    kind: Literal["pure"] = "pure"

    def __init__(self) -> None:
        super().__init__()

    def premium(self, payout: ArrayLike, probability: ArrayLike) -> float:
        """Return `sum(probability * payout)`.

        Raises `ValueError` for arrays of different shapes, non-finite
        payouts or non-positive probabilities.
        """
        return Draws(payout, probability).mean()

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "pure"}`."""
        return {"kind": "pure"}


class CostOfCapitalLoad(_DefinitionModel):
    """The pure premium plus a cost-of-capital load on the payout's tail.

    `P = (1 - rate) * E[payout] + rate * CVaR_q(payout)`, with `rate` in
    [0, 1] and `q` in (0, 1). Raises `CoverDefinitionError` otherwise.
    """

    kind: Literal["cost_of_capital"] = "cost_of_capital"
    rate: FiniteFloat
    q: FiniteFloat

    def __init__(self, rate: float, q: float) -> None:
        super().__init__(rate=rate, q=q)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not 0.0 <= self.rate <= 1.0:
            raise CoverDefinitionError(f"rate must lie in [0, 1], got {self.rate!r}")
        if not 0.0 < self.q < 1.0:
            raise CoverDefinitionError(f"q must lie in (0, 1), got {self.q!r}")
        return self

    def premium(self, payout: ArrayLike, probability: ArrayLike) -> float:
        """Return `(1 - rate) * E[payout] + rate * CVaR_q(payout)`.

        Raises `ValueError` for arrays of different shapes, non-finite
        payouts or non-positive probabilities.
        """
        draws = Draws(payout, probability)
        return (1.0 - self.rate) * draws.mean() + self.rate * draws.cvar(self.q) + 0.0

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "cost_of_capital", "q": q, "rate": rate}`."""
        return {"kind": "cost_of_capital", "q": self.q, "rate": self.rate}


Pricing: TypeAlias = PurePremium | CostOfCapitalLoad
"""The pricings a cover's premium can be computed with."""


def pricing_from_document(document: Any) -> Pricing:
    """Read a pricing document, as `to_document()` writes it.

    Raises `CoverDefinitionError` for an unknown kind or invalid terms.
    """
    if not isinstance(document, Mapping):
        raise CoverDefinitionError("a pricing document must be an object")
    kind = document.get("kind")
    if kind == "pure":
        if set(document) != {"kind"}:
            raise CoverDefinitionError('a pure premium document is {"kind": "pure"}')
        return PurePremium()
    if kind == "cost_of_capital":
        if set(document) != {"kind", "q", "rate"}:
            raise CoverDefinitionError(
                "a cost-of-capital document has the keys kind, q and rate"
            )
        return CostOfCapitalLoad(document["rate"], document["q"])
    raise CoverDefinitionError(f"unknown pricing kind {kind!r}")
