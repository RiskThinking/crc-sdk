"""Design objectives: what a design minimises on its design sample.

On a sample with draw probabilities `p`, loss `L`, payout `pi` and the
uncovered loss `U = max(L - pi, 0)`:

- `MSE()`: `sum(p * (L - pi)^2)`, the mean squared difference between loss
  and payout; paying more than the loss counts as much as paying less.
- `ExpectedUncovered()`: `E[U] = sum(p * U)`; paying more than the loss
  earns nothing.
- `WeightedCVaR(alpha, beta)`: `alpha * CVaR_beta(U) + (1 - alpha) * E[U]`,
  with `alpha` in [0, 1] the weight of the tail term and `beta` in (0, 1).
  `CVaR_beta(U)` is the mean of the worst `1 - beta` of the probability of
  the uncovered loss. A design solves it in the Rockafellar-Uryasev form,
  `CVaR_beta(U) = min_z z + sum(p * max(U - z, 0)) / (1 - beta)`, which keeps
  the problem linear.
- `UserObjective(function, *, name, parameters=None)`: any
  `function(loss, payout, probability) -> float`. It is evaluated, not
  compiled, so a design with it searches a grid.

None of these is a better goal than the others in general: they weigh the
years differently, and which one suits a cover depends on what its owner
needs. The CVaR here is over the years of a sample (see
`crc_sdk.parametric.metrics`); it is not crc-framework's CVaR over binary
outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import model_validator

from crc_sdk.impacts._types import FiniteFloat
from crc_sdk.parametric._canonical import canonical_json
from crc_sdk.parametric._risk import Draws
from crc_sdk.parametric.errors import CoverDefinitionError
from crc_sdk.parametric.parameters import _DefinitionModel

__all__ = [
    "MSE",
    "ExpectedUncovered",
    "Objective",
    "UserObjective",
    "WeightedCVaR",
    "objective_from_document",
]

FloatArray = NDArray[np.float64]


def _arrays(
    loss: ArrayLike, payout: ArrayLike, probability: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    items = (loss, payout, probability)
    values = [np.asarray(item, dtype=np.float64) for item in items]
    shape = values[0].shape
    if len(shape) != 1 or shape[0] == 0 or any(item.shape != shape for item in values):
        raise ValueError(
            "loss, payout and probability must be non-empty 1-D arrays of one "
            f"shape, got {[item.shape for item in values]}"
        )
    if not all(np.isfinite(item).all() for item in values):
        raise ValueError("loss, payout and probability must be finite")
    if (values[2] <= 0).any():
        raise ValueError("probability must be positive")
    return values[0], values[1], values[2]


def _uncovered(loss: FloatArray, payout: FloatArray) -> FloatArray:
    uncovered: FloatArray = np.maximum(loss - payout, 0.0)
    return uncovered


class MSE(_DefinitionModel):
    """The mean squared difference between loss and payout.

    `sum(p * (L - pi)^2)`, in squared payout units. A design minimises it as
    a quadratic programme over the amounts.
    """

    kind: Literal["mse"] = "mse"
    homogeneity: ClassVar[int] = 2

    def __init__(self) -> None:
        super().__init__()

    def value(
        self, loss: ArrayLike, payout: ArrayLike, probability: ArrayLike
    ) -> float:
        """Return `sum(probability * (loss - payout)^2)`.

        Raises `ValueError` for arrays of different shapes, non-finite values
        or non-positive probabilities.
        """
        x, y, p = _arrays(loss, payout, probability)
        return math.fsum((p * (x - y) ** 2).tolist()) + 0.0

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "mse"}`."""
        return {"kind": "mse"}


class ExpectedUncovered(_DefinitionModel):
    """The expected uncovered loss, `E[U] = sum(p * max(L - pi, 0))`.

    In payout units. A design minimises it as a linear programme.
    """

    kind: Literal["expected_uncovered"] = "expected_uncovered"
    homogeneity: ClassVar[int] = 1

    def __init__(self) -> None:
        super().__init__()

    def value(
        self, loss: ArrayLike, payout: ArrayLike, probability: ArrayLike
    ) -> float:
        """Return `sum(probability * max(loss - payout, 0))`.

        Raises `ValueError` as `MSE.value` does.
        """
        x, y, p = _arrays(loss, payout, probability)
        return Draws(_uncovered(x, y), p).mean()

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "expected_uncovered"}`."""
        return {"kind": "expected_uncovered"}


class WeightedCVaR(_DefinitionModel):
    """A weighted sum of the tail and the mean of the uncovered loss.

    `alpha * CVaR_beta(U) + (1 - alpha) * E[U]`, in payout units, with
    `alpha` in [0, 1] and `beta` in (0, 1). `alpha = 0` is `E[U]`; `alpha =
    1` is the CVaR alone. A design minimises it as a linear programme in the
    Rockafellar-Uryasev form. When `beta` is at most the probability of a
    zero loss, or the tail holds at most one draw, the objective degenerates
    (to a multiple of `E[U]`, or to the single worst year); the design
    records which (`DesignResult.degenerate_tail`).

    Raises `CoverDefinitionError` for `alpha` or `beta` out of range.
    """

    kind: Literal["weighted_cvar"] = "weighted_cvar"
    homogeneity: ClassVar[int] = 1
    alpha: FiniteFloat
    beta: FiniteFloat

    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__(alpha=alpha, beta=beta)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not 0.0 <= self.alpha <= 1.0:
            raise CoverDefinitionError(f"alpha must lie in [0, 1], got {self.alpha!r}")
        if not 0.0 < self.beta < 1.0:
            raise CoverDefinitionError(f"beta must lie in (0, 1), got {self.beta!r}")
        return self

    def value(
        self, loss: ArrayLike, payout: ArrayLike, probability: ArrayLike
    ) -> float:
        """Return `alpha * CVaR_beta(U) + (1 - alpha) * E[U]`.

        Raises `ValueError` as `MSE.value` does.
        """
        x, y, p = _arrays(loss, payout, probability)
        draws = Draws(_uncovered(x, y), p)
        tail = draws.cvar(self.beta) if self.alpha > 0.0 else 0.0
        return self.alpha * tail + (1.0 - self.alpha) * draws.mean() + 0.0

    def to_document(self) -> dict[str, Any]:
        """Return `{"alpha": alpha, "beta": beta, "kind": "weighted_cvar"}`."""
        return {"alpha": self.alpha, "beta": self.beta, "kind": "weighted_cvar"}


UserFunction: TypeAlias = Callable[[FloatArray, FloatArray, FloatArray], float]
"""`function(loss, payout, probability) -> float`, over the draws of a sample."""


@dataclass(frozen=True, eq=False, init=False)
class UserObjective:
    """A user's objective: a function of the loss, the payout and the probability.

    `function(loss, payout, probability)` receives three read-only 1-D float
    arrays, one entry per draw, in payout units, and returns the value to
    minimise as a finite float. `name` and `parameters` (plain JSON values)
    identify the objective in the design record and the design id; the
    function itself is not recorded, so give each function its own name.

    A user objective is evaluated, not compiled: a design with it searches a
    grid over every free parameter of the template, so each needs a grid.

    Raises `CoverDefinitionError` for a function that is not callable, an
    empty name or parameters that are not plain JSON.
    """

    function: UserFunction
    name: str
    parameters: Mapping[str, Any]
    kind: ClassVar[str] = "user"

    def __init__(
        self,
        function: UserFunction,
        *,
        name: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(function):
            raise CoverDefinitionError(
                f"function must be callable, got {type(function).__name__}"
            )
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise CoverDefinitionError(
                f"name must be a non-empty string without surrounding whitespace, "
                f"got {name!r}"
            )
        chosen = {} if parameters is None else parameters
        if not isinstance(chosen, Mapping):
            raise CoverDefinitionError("parameters must be a mapping")
        try:
            canonical_json(dict(chosen))
        except (TypeError, ValueError) as error:
            raise CoverDefinitionError(
                f"parameters must hold plain, finite JSON values: {error}"
            ) from error
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parameters", dict(chosen))

    def value(
        self, loss: ArrayLike, payout: ArrayLike, probability: ArrayLike
    ) -> float:
        """Return `function(loss, payout, probability)` as a float.

        Raises `ValueError` for invalid arrays, and `TypeError` if the
        function does not return a finite real number.
        """
        x, y, p = _arrays(loss, payout, probability)
        for array in (x, y, p):
            array.setflags(write=False)
        result = self.function(x, y, p)
        if isinstance(result, bool) or not isinstance(result, (int, float, np.number)):
            raise TypeError(
                f"the user objective {self.name!r} must return a real number, "
                f"got {type(result).__name__}"
            )
        number = float(result)
        if not math.isfinite(number):
            raise TypeError(
                f"the user objective {self.name!r} returned {number!r}; it must "
                "return a finite number"
            )
        return number + 0.0

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "user", "name": name, "parameters": parameters}`."""
        return {"kind": "user", "name": self.name, "parameters": dict(self.parameters)}


Objective: TypeAlias = MSE | ExpectedUncovered | WeightedCVaR | UserObjective
"""The objectives a design can minimise."""


def objective_from_document(document: Any) -> MSE | ExpectedUncovered | WeightedCVaR:
    """Read a built-in objective's document, as `to_document()` writes it.

    A user objective cannot be read back: its function is not in the
    document. Raises `CoverDefinitionError` for an unknown kind, a user
    objective or invalid terms.
    """
    if not isinstance(document, Mapping):
        raise CoverDefinitionError("an objective document must be an object")
    kind = document.get("kind")
    if kind == "mse" and set(document) == {"kind"}:
        return MSE()
    if kind == "expected_uncovered" and set(document) == {"kind"}:
        return ExpectedUncovered()
    if kind == "weighted_cvar" and set(document) == {"alpha", "beta", "kind"}:
        return WeightedCVaR(document["alpha"], document["beta"])
    if kind == "user":
        raise CoverDefinitionError(
            "a user objective cannot be read from its document: pass the "
            "UserObjective with its function"
        )
    raise CoverDefinitionError(f"not a built-in objective document: {document!r}")
