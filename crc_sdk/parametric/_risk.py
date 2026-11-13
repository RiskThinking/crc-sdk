"""Risk measures over the draws of an outcome sample, in numpy.

Every function here takes one quantity per draw (a loss, a payout, an
uncovered or a net loss) and the draws' probabilities. One draw is one year,
so these are measures over years of a continuous quantity. They are not
crc-framework's `compute_risk` VaR and CVaR, which are over the 2^n branches
of independent binary factor outcomes: the random variable, the outcomes and
the dependence assumptions all differ, and the two are never compared.

- `lower_quantile(x, p, level)`: the smallest value v with
  `sum(p[x <= v]) >= level`. The comparison allows `PROBABILITY_TOLERANCE`
  (1e-10) for the rounding of the summed probabilities, so a level that
  falls exactly on a step of the distribution gives the lower of the two
  neighbouring values on every platform.
- `var(x, p, beta)`: `lower_quantile(x, p, beta)`.
- `cvar(x, p, beta)`: `VaR + sum(p * max(x - VaR, 0)) / (1 - beta)`, the mean
  of the worst `1 - beta` of the probability, with the probability at the
  VaR split as needed. It does not depend on which neighbouring value the
  VaR takes.
- `tail_weights(x, p, beta)`: the weights behind `cvar`: `p` where `x` is
  above the VaR, a pro-rata share of the remaining mass where `x` equals it,
  0 elsewhere. They sum to `1 - beta`.

`Draws(x, p)` sorts the values once and answers each of these at several
levels. Expectations use `math.fsum`, so they do not depend on the
platform's summation order.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PROBABILITY_TOLERANCE",
    "Draws",
    "cvar",
    "expectation",
    "lower_quantile",
    "tail_weights",
    "var",
]

FloatArray = NDArray[np.float64]

PROBABILITY_TOLERANCE = 1e-10
"""The rounding allowance of the cumulative probability in `lower_quantile`."""


def _arrays(values: ArrayLike, probability: ArrayLike) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(values, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] == 0 or p.shape != x.shape:
        raise ValueError(
            "values and probability must be non-empty 1-D arrays of one shape, "
            f"got {x.shape} and {p.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("values must be finite")
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("probability must be finite and positive")
    return x, p


def _level(name: str, value: float, *, closed_low: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    level = float(value)
    low_ok = level >= 0.0 if closed_low else level > 0.0
    if not (math.isfinite(level) and low_ok and level < 1.0):
        interval = "[0, 1)" if closed_low else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}, got {value!r}")
    return level


def _fsum(values: FloatArray) -> float:
    return math.fsum(values.tolist())


class Draws:
    """One quantity per draw with the draws' probabilities, sorted once.

    The measures below share one sort of the values, so several levels cost
    one sort. Raises `ValueError` for arrays of different shapes, non-finite
    values or non-positive probabilities.
    """

    __slots__ = ("_at_distinct", "_distinct", "probability", "values")

    def __init__(self, values: ArrayLike, probability: ArrayLike) -> None:
        x, p = _arrays(values, probability)
        self.values = x
        self.probability = p
        order = np.argsort(x, kind="stable")
        ordered = x[order]
        cumulative = np.cumsum(p[order])
        # The cumulative probability at each distinct value is the one at its
        # last occurrence in the sorted order.
        last = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
        self._distinct = ordered[last]
        self._at_distinct = cumulative[last]

    def mean(self) -> float:
        """`sum(p * x)`."""
        return _fsum(self.probability * self.values) + 0.0

    def quantile(self, level: float) -> float:
        """The smallest v with `sum(p[x <= v]) >= level - 1e-10`; level in [0, 1)."""
        checked = _level("level", level, closed_low=True)
        reached = np.flatnonzero(self._at_distinct >= checked - PROBABILITY_TOLERANCE)
        position = int(reached[0]) if reached.size else self._distinct.shape[0] - 1
        return float(self._distinct[position]) + 0.0

    def var(self, beta: float) -> float:
        """VaR at `beta` in (0, 1): the lower `beta`-quantile."""
        return self.quantile(_level("beta", beta))

    def cvar(self, beta: float) -> float:
        """CVaR at `beta` in (0, 1): `VaR + sum(p (x - VaR)^+) / (1 - beta)`."""
        level = _level("beta", beta)
        value_at_risk = self.quantile(level)
        excess = _fsum(self.probability * np.maximum(self.values - value_at_risk, 0.0))
        return value_at_risk + excess / (1.0 - level) + 0.0

    def tail_weights(self, beta: float) -> FloatArray:
        """The CVaR weights at `beta` in (0, 1); see `tail_weights`."""
        level = _level("beta", beta)
        value_at_risk = self.quantile(level)
        x, p = self.values, self.probability
        weights = np.where(x > value_at_risk, p, 0.0)
        remaining = max(0.0, (1.0 - level) - _fsum(weights))
        at = x == value_at_risk
        mass_at = _fsum(p[at])
        if mass_at > 0.0 and remaining > 0.0:
            weights[at] = p[at] * (min(remaining, mass_at) / mass_at)
        return weights


def expectation(values: ArrayLike, probability: ArrayLike) -> float:
    """Return `sum(p * x)`, the probability-weighted mean of `values`."""
    return Draws(values, probability).mean()


def lower_quantile(values: ArrayLike, probability: ArrayLike, level: float) -> float:
    """Return the smallest v with `sum(p[x <= v]) >= level`, within 1e-10.

    `level` lies in [0, 1). A level exactly on a step of the distribution
    gives the lower of the two neighbouring values. Raises `ValueError` for
    invalid inputs.
    """
    return Draws(values, probability).quantile(level)


def var(values: ArrayLike, probability: ArrayLike, beta: float) -> float:
    """Return VaR at level `beta` in (0, 1): `lower_quantile(x, p, beta)`.

    This is a quantile over the years of an outcome sample, not
    crc-framework's VaR over binary outcomes.
    """
    return Draws(values, probability).var(beta)


def cvar(values: ArrayLike, probability: ArrayLike, beta: float) -> float:
    """Return CVaR at level `beta` in (0, 1), over the years of a sample.

    `CVaR = VaR + sum(p * max(x - VaR, 0)) / (1 - beta)`: the mean of the
    worst `1 - beta` of the probability of a continuous quantity over years.
    It is not crc-framework's CVaR, which is over the branches of independent
    binary outcomes. Raises `ValueError` for invalid inputs.
    """
    return Draws(values, probability).cvar(beta)


def tail_weights(values: ArrayLike, probability: ArrayLike, beta: float) -> FloatArray:
    """Return the CVaR weights of `values` at level `beta` in (0, 1).

    A draw above the VaR gets its probability; draws equal to the VaR share
    the remaining mass `1 - beta - P(x > VaR)` pro rata to their
    probabilities; every other draw gets 0. The weights sum to `1 - beta`
    (up to the rounding the VaR rule allows), so `sum(w * x) / (1 - beta)`
    is the CVaR. Raises `ValueError` for invalid inputs.
    """
    return Draws(values, probability).tail_weights(beta)
