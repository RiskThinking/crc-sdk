"""Basis-risk models: how an outcome sample's index and loss differ.

A cover pays on its index; the owner bears the loss. Basis risk is the gap
between the two. Every sample states the model it was built with, and none is
assumed: the samplers take `basis_risk` as a required argument.

- `NoBasisRisk()`: the index is the hazard the loss model reads, so the index
  and the loss move together (comonotone) and a cover can track the loss
  almost exactly. The samplers warn with `ComonotoneIndexWarning` when it is
  used without a separate index series.
- `ImpactNoise(sigma, *, upper=None)`: multiplies each year's impact by a
  mean-one lognormal factor `exp(sigma * Z - sigma**2 / 2)`, one draw per
  year, then caps it at `upper` if given (1 for a damage ratio). The index is
  unchanged. The cap lowers the mean impact a little; the samplers record the
  realised ratio.
- A user model implements the `BasisRisk` protocol.

A per-year index read from another cell, station or hazard than the loss is
not a model here: pass it to `sample_annual` as `index=`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from crc_sdk.parametric.errors import SampleError

__all__ = ["BasisRisk", "ImpactNoise", "NoBasisRisk"]

FloatArray = NDArray[np.float64]


@runtime_checkable
class BasisRisk(Protocol):
    """What every basis-risk model provides.

    `name` identifies the model in the sample's provenance, and
    `to_document()` returns its parameters as plain JSON types, including a
    `kind` key.

    `apply(index, hazard, impact, rng)` takes one value per draw: the index
    the cover would pay on, the hazard the loss model read, and the impact
    (usually a damage ratio) the loss model computed from it. It returns the
    new `(index, impact)`, each of the input's shape, finite, and the impact
    non-negative. The loss is the returned impact times the asset value.
    `rng` is the sample's generator, or `None` when the caller gave none; a
    model that draws must raise without one. The samplers call `apply` after
    the hazard draws, so the draw order is fixed: hazard first, then the
    basis-risk model.
    """

    @property
    def name(self) -> str: ...

    def to_document(self) -> dict[str, Any]: ...

    def apply(
        self,
        index: FloatArray,
        hazard: FloatArray,
        impact: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]: ...


@dataclass(frozen=True)
class NoBasisRisk:
    """No basis risk: the index is the hazard at the asset.

    The loss is then a function of the index, so a cover can track it almost
    exactly. Write it out to state that choice; the samplers warn with
    `ComonotoneIndexWarning` when the index is not read from another series.
    """

    @property
    def name(self) -> str:
        """`"no_basis_risk"`."""
        return "no_basis_risk"

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "no_basis_risk"}`."""
        return {"kind": self.name}

    def apply(
        self,
        index: FloatArray,
        hazard: FloatArray,
        impact: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the index and the impact unchanged; draws nothing."""
        del hazard, rng
        return index, impact


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SampleError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SampleError(f"{name} must be finite and positive, got {value!r}")
    return number


@dataclass(frozen=True)
class ImpactNoise:
    """Mean-one lognormal noise on each year's impact.

    Each draw's impact is multiplied by `exp(sigma * Z - sigma**2 / 2)` with
    `Z` standard normal, one draw per year, so the factor has mean 1 and
    `sigma` is the standard deviation of its logarithm. With `upper`, the
    noisy impact is then capped at `upper` (1 for a damage ratio), which
    lowers its mean a little. The index is unchanged, and a year with zero
    impact keeps zero impact.

    The factors are drawn with one call `rng.normal(-sigma**2 / 2, sigma,
    size=n)`, the noise layer of the thesis code (homoscedastic) with one
    draw per year.

    Raises `SampleError` unless `sigma` and `upper` are finite and positive.
    """

    sigma: float
    upper: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive("sigma", self.sigma))
        if self.upper is not None:
            object.__setattr__(self, "upper", _positive("upper", self.upper))

    @property
    def name(self) -> str:
        """`"impact_noise"`."""
        return "impact_noise"

    def to_document(self) -> dict[str, Any]:
        """Return the kind, `sigma` and `upper`."""
        return {"kind": self.name, "sigma": self.sigma, "upper": self.upper}

    def apply(
        self,
        index: FloatArray,
        hazard: FloatArray,
        impact: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the index unchanged and the noisy, capped impact.

        Raises `SampleError` without a `numpy.random.Generator`.
        """
        del hazard
        if not isinstance(rng, np.random.Generator):
            raise SampleError(
                "ImpactNoise draws one factor per year; pass rng= a "
                "numpy.random.Generator"
            )
        sigma = self.sigma
        log_factor = rng.normal(
            loc=-0.5 * sigma * sigma, scale=sigma, size=impact.shape
        )
        noisy = np.asarray(impact, dtype=np.float64) * np.exp(log_factor)
        if self.upper is not None:
            noisy = np.minimum(noisy, self.upper)
        return index, noisy
