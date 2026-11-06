"""Metrics of a cover on an outcome sample: coverage, tail coverage, basis risk.

On a sample of draws d (one draw is one year) with probabilities p_d, a
cover's payout pi_d and the loss L_d, write U_d = max(L_d - pi_d, 0) for the
uncovered loss, R_d = L_d - pi_d for the residual, P for the premium and
N_d = L_d - pi_d + P for the net loss, the owner's cost of the year with the
cover. `metric_values` returns, in this order:

- `expected_loss` E[L], `expected_payout` E[pi], `premium` P,
  `expected_uncovered` E[U];
- `coverage_ratio`: E[min(pi, L)] / E[L], the share of the expected loss
  that payouts cover (payouts above the loss do not count); null when
  E[L] = 0;
- `payout_to_loss_ratio`: E[pi] / E[L]; null when E[L] = 0;
- `tail_coverage_b<beta>`: sum w min(pi, L) / sum w L, with w the CVaR_beta
  weights of L: the share of the loss covered in the worst 1 - beta of loss
  years; null when sum w L = 0;
- `cvar_uncovered_to_loss_b<beta>`: CVaR_beta(U) / CVaR_beta(L); null when
  CVaR_beta(L) = 0;
- `false_negative_rate_l<l>`: P(pi <= eps, L > l) / P(L > l), loss years
  without payout; null when P(L > l) = 0;
- `false_positive_rate_l<l>`: P(pi > eps, L <= l) / P(L <= l), payouts in
  years without a loss above l; null when P(L <= l) = 0;
- `rmse`: sqrt(E[R^2]); `correlation`: the p-weighted Pearson correlation of
  L and pi, null when either is constant;
- `var_uncovered_b<beta>`, `cvar_uncovered_b<beta>`, `var_net_loss_b<beta>`,
  `cvar_net_loss_b<beta>`, `var_loss_b<beta>`, `cvar_loss_b<beta>`: VaR and
  CVaR of U, N and L;
- `loss_rp<T>`, `payout_rp<T>`, `uncovered_rp<T>`, `net_loss_rp<T>`: the
  value exceeded with probability 1/T, the lower (1 - 1/T)-quantile; T is in
  years, because a draw is a year.

`MetricOptions` states beta, T, the loss thresholds l and the payout
threshold eps; nothing is hidden. Labels write the number with the point as
an underscore, as crc-sdk's return-period columns do: beta = 0.99 gives
`b0_99`, T = 100 gives `rp100`, l = 0 gives `l0`. Invalid options and inputs
raise `EvaluationError`.

**VaR and CVaR here are over years.** `VaR_beta(X)` is the smallest x with
`sum(p[X <= x]) >= beta`, the comparison allowing 1e-10 for rounding, so a
beta exactly on a step of the distribution gives the lower value on every
platform. `CVaR_beta(X)` is the mean of the worst 1 - beta of the
probability. Both are measures over the years of an outcome sample, of one
continuous quantity of one cover. crc-framework's `compute_risk` VaR and CVaR
are a different quantity, over the 2^n branches of independent binary factor
outcomes; the two never share a column and are never compared.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import model_validator

from crc_sdk.impacts._types import FiniteFloat
from crc_sdk.parametric._risk import Draws
from crc_sdk.parametric.errors import EvaluationError
from crc_sdk.parametric.metadata import _EvaluationModel

__all__ = [
    "MetricColumn",
    "MetricOptions",
    "metric_columns",
    "metric_values",
]

FloatArray = NDArray[np.float64]

_PROBABILITY_SUM_TOLERANCE = 1e-12


def _label(value: float) -> str:
    """The decimal text of `value` with the point as an underscore."""
    text = format(Decimal(repr(float(value))), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", "_")


def _distinct(name: str, values: tuple[float, ...]) -> None:
    labels = [_label(value) for value in values]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} repeat a value, so their column names collide")


class MetricOptions(_EvaluationModel):
    """The levels and thresholds of the metrics. Every one is stated.

    - `betas`: the levels beta in (0, 1) of the tail metrics: tail coverage,
      the CVaR ratio, and VaR and CVaR of the uncovered loss, the net loss
      and the loss.
    - `return_periods`: the return periods T > 1, in years (a draw is a
      year), of the return-period columns.
    - `loss_thresholds`: the losses l >= 0 of the basis-risk rates: a year
      with a loss above l is a loss year.
    - `payout_threshold`: eps >= 0, the payout at or below which a year
      counts as a year without payout.

    Raises `EvaluationError` (a `ValueError`) for values outside these
    ranges, non-finite values, unknown options and repeated values (their
    column names would collide).
    """

    betas: tuple[FiniteFloat, ...] = (0.9, 0.95, 0.99)
    return_periods: tuple[FiniteFloat, ...] = (
        2.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
    )
    loss_thresholds: tuple[FiniteFloat, ...] = (0.0,)
    payout_threshold: FiniteFloat = 0.0

    @model_validator(mode="after")
    def _check(self) -> Self:
        if any(not 0.0 < beta < 1.0 for beta in self.betas):
            raise ValueError(f"betas must lie in (0, 1), got {self.betas}")
        if any(not period > 1.0 for period in self.return_periods):
            raise ValueError(
                f"return_periods must be greater than 1 year, got {self.return_periods}"
            )
        if any(threshold < 0.0 for threshold in self.loss_thresholds):
            raise ValueError(
                f"loss_thresholds must be non-negative, got {self.loss_thresholds}"
            )
        if self.payout_threshold < 0.0:
            raise ValueError(
                f"payout_threshold must be non-negative, got {self.payout_threshold}"
            )
        _distinct("betas", self.betas)
        _distinct("return_periods", self.return_periods)
        _distinct("loss_thresholds", self.loss_thresholds)
        return self

    def to_document(self) -> dict[str, Any]:
        """Return the options as plain JSON types."""
        return {
            "betas": list(self.betas),
            "loss_thresholds": list(self.loss_thresholds),
            "payout_threshold": self.payout_threshold,
            "return_periods": list(self.return_periods),
        }


Unit = Literal["loss", "ratio"]


@dataclass(frozen=True)
class MetricColumn:
    """One metric column: its name, metric, parameter, unit and definition.

    `parameter` maps the parameter's name (`beta`, `return_period`,
    `loss_threshold`) to its value, or is empty. `unit` is `"loss"` for
    columns in the loss unit (the cover's payout unit) and `"ratio"` for
    unitless ones.
    """

    name: str
    metric: str
    parameter: Mapping[str, float]
    unit: Unit
    definition: str

    def to_document(self, loss_unit: str) -> dict[str, Any]:
        """Return the column's metadata entry, with the loss unit written out."""
        return {
            "definition": self.definition,
            "metric": self.metric,
            "parameter": dict(self.parameter) or None,
            "unit": loss_unit if self.unit == "loss" else "1",
        }


def _per_beta(
    metric: str, unit: Unit, definition: str, options: MetricOptions
) -> list[MetricColumn]:
    return [
        MetricColumn(
            f"{metric}_b{_label(beta)}",
            metric,
            {"beta": beta},
            unit,
            definition.format(beta=beta),
        )
        for beta in options.betas
    ]


def _per_period(metric: str, what: str, options: MetricOptions) -> list[MetricColumn]:
    return [
        MetricColumn(
            f"{metric}{_label(period)}",
            metric,
            {"return_period": period},
            "loss",
            f"the {what} exceeded with probability 1/{period:g} per year: its "
            f"lower {1.0 - 1.0 / period:g}-quantile over years",
        )
        for period in options.return_periods
    ]


def metric_columns(options: MetricOptions) -> tuple[MetricColumn, ...]:
    """Return the metric columns for `options`, in table order."""
    eps = options.payout_threshold
    columns = [
        MetricColumn("expected_loss", "expected_loss", {}, "loss", "E[L]"),
        MetricColumn("expected_payout", "expected_payout", {}, "loss", "E[payout]"),
        MetricColumn(
            "premium", "premium", {}, "loss", "the premium used (see premium_basis)"
        ),
        MetricColumn(
            "expected_uncovered",
            "expected_uncovered",
            {},
            "loss",
            "E[U], U = max(L - payout, 0)",
        ),
        MetricColumn(
            "coverage_ratio",
            "coverage_ratio",
            {},
            "ratio",
            "E[min(payout, L)] / E[L]: the share of the expected loss that "
            "payouts cover; payouts above the loss do not count",
        ),
        MetricColumn(
            "payout_to_loss_ratio",
            "payout_to_loss_ratio",
            {},
            "ratio",
            "E[payout] / E[L]: payouts above the loss count too",
        ),
    ]
    columns += _per_beta(
        "tail_coverage",
        "ratio",
        "the share of the loss covered in the worst {beta:g}-tail years of the "
        "loss: sum w min(payout, L) / sum w L, w the CVaR weights of L",
        options,
    )
    columns += _per_beta(
        "cvar_uncovered_to_loss",
        "ratio",
        "CVaR_{beta:g}(U) / CVaR_{beta:g}(L): the tail of the uncovered loss "
        "against the tail of the uninsured loss",
        options,
    )
    columns += [
        MetricColumn(
            f"false_negative_rate_l{_label(threshold)}",
            "false_negative_rate",
            {"loss_threshold": threshold},
            "ratio",
            f"P(payout <= {eps:g}, L > {threshold:g}) / P(L > {threshold:g}): "
            "loss years without payout",
        )
        for threshold in options.loss_thresholds
    ]
    columns += [
        MetricColumn(
            f"false_positive_rate_l{_label(threshold)}",
            "false_positive_rate",
            {"loss_threshold": threshold},
            "ratio",
            f"P(payout > {eps:g}, L <= {threshold:g}) / P(L <= {threshold:g}): "
            "payouts in years without a loss above the threshold",
        )
        for threshold in options.loss_thresholds
    ]
    columns += [
        MetricColumn("rmse", "rmse", {}, "loss", "sqrt(E[(L - payout)^2])"),
        MetricColumn(
            "correlation",
            "correlation",
            {},
            "ratio",
            "the probability-weighted Pearson correlation of L and payout",
        ),
    ]
    over_years = "over years of the sample, not crc-framework's binary-outcome {}"
    for quantity, what in (
        ("uncovered", "U"),
        ("net_loss", "the net loss N = L - payout + premium"),
        ("loss", "L"),
    ):
        columns += _per_beta(
            f"var_{quantity}",
            "loss",
            f"VaR_{{beta:g}} of {what}: the lower {{beta:g}}-quantile "
            + over_years.format("VaR"),
            options,
        )
        columns += _per_beta(
            f"cvar_{quantity}",
            "loss",
            f"CVaR_{{beta:g}} of {what}: the mean of its worst 1 - {{beta:g}} "
            "of the probability " + over_years.format("CVaR"),
            options,
        )
    columns += _per_period("loss_rp", "loss", options)
    columns += _per_period("payout_rp", "payout", options)
    columns += _per_period("uncovered_rp", "uncovered loss", options)
    columns += _per_period("net_loss_rp", "net loss", options)
    return tuple(columns)


def _fsum(values: FloatArray) -> float:
    return math.fsum(values.tolist())


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator + 0.0


def _correlation(
    loss: FloatArray, payout: FloatArray, probability: FloatArray
) -> float | None:
    if np.ptp(loss) == 0.0 or np.ptp(payout) == 0.0:
        return None
    loss_centred = loss - _fsum(probability * loss)
    payout_centred = payout - _fsum(probability * payout)
    covariance = _fsum(probability * loss_centred * payout_centred)
    variance = _fsum(probability * loss_centred**2) * _fsum(
        probability * payout_centred**2
    )
    if variance == 0.0:
        return None
    return min(1.0, max(-1.0, covariance / math.sqrt(variance))) + 0.0


def metric_values(
    loss: ArrayLike,
    payout: ArrayLike,
    probability: ArrayLike,
    *,
    premium: float,
    options: MetricOptions,
) -> dict[str, float | None]:
    """Return every metric of `metric_columns(options)`, in that order.

    `loss`, `payout` and `probability` hold one value per draw: the loss
    (>= 0), the cover's payout (>= 0) and the draw's probability (> 0,
    summing to 1 within 1e-12). `premium` is the premium used, in the loss
    unit. A value is `None` where the metric is undefined (see the module
    docstring). Raises `EvaluationError` (a `ValueError`) for arrays of
    different shapes, non-finite or negative values, probabilities that do
    not sum to 1, or a premium that is not a finite number.
    """
    try:
        loss_draws = Draws(loss, probability)
        payout_draws = Draws(payout, probability)
    except ValueError as error:
        raise EvaluationError(str(error)) from error
    p = loss_draws.probability
    if payout_draws.values.shape != p.shape:
        raise EvaluationError("loss and payout must have one value per draw")
    if abs(_fsum(p) - 1.0) > _PROBABILITY_SUM_TOLERANCE:
        raise EvaluationError("probability must sum to 1 within 1e-12")
    if (
        isinstance(premium, bool)
        or not isinstance(premium, (int, float, np.number))
        or not math.isfinite(float(premium))
    ):
        raise EvaluationError(f"premium must be a finite number, got {premium!r}")
    L = loss_draws.values
    pi = payout_draws.values
    if (L < 0.0).any() or (pi < 0.0).any():
        raise EvaluationError("loss and payout must be non-negative")
    P = float(premium) + 0.0
    covered = np.minimum(pi, L)
    residual = L - pi
    uncovered_draws = Draws(np.maximum(residual, 0.0), p)
    net_draws = Draws(residual + P, p)
    expected_loss = loss_draws.mean()
    eps = options.payout_threshold

    values: dict[str, float | None] = {
        "expected_loss": expected_loss,
        "expected_payout": payout_draws.mean(),
        "premium": P,
        "expected_uncovered": uncovered_draws.mean(),
        "coverage_ratio": _ratio(_fsum(p * covered), expected_loss),
        "payout_to_loss_ratio": _ratio(payout_draws.mean(), expected_loss),
    }
    for beta in options.betas:
        weights = loss_draws.tail_weights(beta)
        values[f"tail_coverage_b{_label(beta)}"] = _ratio(
            _fsum(weights * covered), _fsum(weights * L)
        )
    for beta in options.betas:
        values[f"cvar_uncovered_to_loss_b{_label(beta)}"] = _ratio(
            uncovered_draws.cvar(beta), loss_draws.cvar(beta)
        )
    no_payout = pi <= eps
    for threshold in options.loss_thresholds:
        loss_year = L > threshold
        values[f"false_negative_rate_l{_label(threshold)}"] = _ratio(
            _fsum(p[no_payout & loss_year]), _fsum(p[loss_year])
        )
    for threshold in options.loss_thresholds:
        quiet_year = L <= threshold
        values[f"false_positive_rate_l{_label(threshold)}"] = _ratio(
            _fsum(p[~no_payout & quiet_year]), _fsum(p[quiet_year])
        )
    values["rmse"] = math.sqrt(_fsum(p * residual**2)) + 0.0
    values["correlation"] = _correlation(L, pi, p)
    for quantity, draws in (
        ("uncovered", uncovered_draws),
        ("net_loss", net_draws),
        ("loss", loss_draws),
    ):
        for beta in options.betas:
            values[f"var_{quantity}_b{_label(beta)}"] = draws.var(beta)
        for beta in options.betas:
            values[f"cvar_{quantity}_b{_label(beta)}"] = draws.cvar(beta)
    for quantity, draws in (
        ("loss", loss_draws),
        ("payout", payout_draws),
        ("uncovered", uncovered_draws),
        ("net_loss", net_draws),
    ):
        for period in options.return_periods:
            values[f"{quantity}_rp{_label(period)}"] = draws.quantile(
                1.0 - 1.0 / period
            )
    return values
