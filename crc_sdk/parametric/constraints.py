"""Design constraints: the premium budget and the payout limit.

- `Budget(ratio=k)`: the premium is at most `k * E[L]`, the expected loss on
  the design sample. `Budget(amount=B)`: the premium is at most `B`, in
  payout units. Exactly one of the two. The premium is the design's pricing
  (`PurePremium` or `CostOfCapitalLoad`) on the design sample.
- `Limit(amount)`: the payout never exceeds `amount`, in payout units. For
  a payout linear in its amounts (`LinearInAmounts`) it bounds the amounts
  through the family's `amount_constraints`; for the built-in families that
  is "every amount at most `amount`", since their payout is at most their
  largest amount.

The payout family adds its own constraints: amounts are non-negative, levels
are non-decreasing under `monotone=True`, and every `Free` parameter keeps
its bounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import model_validator

from crc_sdk.impacts._types import FiniteFloat
from crc_sdk.parametric.errors import CoverDefinitionError
from crc_sdk.parametric.parameters import _DefinitionModel

__all__ = ["Budget", "Limit", "budget_from_document", "limit_from_document"]


class Budget(_DefinitionModel):
    """The premium budget of a design: `ratio` of the expected loss, or `amount`.

    `Budget(ratio=k)` means premium <= `k * E[L]` and `Budget(amount=B)`
    means premium <= `B`, both on the design sample, with `k >= 0` and
    `B >= 0` in payout units. Exactly one of `ratio` and `amount` is given.

    Raises `CoverDefinitionError` otherwise.
    """

    ratio: FiniteFloat | None = None
    amount: FiniteFloat | None = None

    def __init__(
        self, *, ratio: float | None = None, amount: float | None = None
    ) -> None:
        super().__init__(ratio=ratio, amount=amount)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.ratio is None) == (self.amount is None):
            raise CoverDefinitionError("give exactly one of ratio= and amount=")
        value = self.ratio if self.ratio is not None else self.amount
        if value is not None and value < 0:
            raise CoverDefinitionError(f"a budget must be >= 0, got {value!r}")
        return self

    def bound(self, expected_loss: float) -> float:
        """Return the largest premium allowed, in payout units.

        `ratio * expected_loss` for a ratio, else the amount.
        """
        if self.ratio is not None:
            return self.ratio * float(expected_loss) + 0.0
        assert self.amount is not None
        return self.amount

    def to_document(self) -> dict[str, Any]:
        """Return `{"ratio": k}` or `{"amount": B}`."""
        if self.ratio is not None:
            return {"ratio": self.ratio}
        return {"amount": self.amount}


class Limit(_DefinitionModel):
    """The payout limit of a design: no year pays more than `amount`.

    `amount >= 0`, in payout units. It is a contract term the user states.
    Raises `CoverDefinitionError` for a negative or non-finite amount.
    """

    amount: FiniteFloat

    def __init__(self, amount: float) -> None:
        super().__init__(amount=amount)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.amount < 0:
            raise CoverDefinitionError(f"a limit must be >= 0, got {self.amount!r}")
        return self

    def to_document(self) -> dict[str, Any]:
        """Return `{"amount": amount}`."""
        return {"amount": self.amount}


def budget_from_document(document: Any) -> Budget | None:
    """Read a budget document (`{"ratio": k}` or `{"amount": B}`), or `None`.

    Raises `CoverDefinitionError` for any other document.
    """
    if document is None:
        return None
    if not isinstance(document, Mapping) or set(document) not in (
        {"ratio"},
        {"amount"},
    ):
        raise CoverDefinitionError(
            f'a budget document is {{"ratio": k}} or {{"amount": B}}, got {document!r}'
        )
    return Budget(**{str(key): value for key, value in document.items()})


def limit_from_document(document: Any) -> Limit | None:
    """Read a limit document (`{"amount": a}`), or `None`.

    Raises `CoverDefinitionError` for any other document.
    """
    if document is None:
        return None
    if not isinstance(document, Mapping) or set(document) != {"amount"}:
        raise CoverDefinitionError(
            f'a limit document is {{"amount": a}}, got {document!r}'
        )
    return Limit(document["amount"])
