"""Arrow-batched fitting of tabulated CDF quantiles into canonical curves."""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, NamedTuple

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import fit_hurdle_quantiles, fit_quantiles
from crc_framework.distributions import DistributionFamily, HurdleDistribution

from crc_sdk._version import framework_version, platform_tag, sdk_version
from crc_sdk.connectors.adapters import CanonicalHazardBatch, CanonicalHazardStream
from crc_sdk.connectors.duckdb import detected_cpu_count
from crc_sdk.connectors.parquet import hazard_arrow_schema
from crc_sdk.fitting.diagnostics import DiagnosticsWriter
from crc_sdk.registry import HazardRegistry
from crc_sdk.types import (
    CurveFitProvenance,
    EnsembleDescriptor,
    HazardDatasetMetadata,
    ProbabilitySemantics,
    SourceProvenance,
    TemporalWindow,
)

# A parametric fit needs this many interior probability knots; shorter axes
# follow ``CDFCurveFitPolicy.short_axis_action`` inside the fitter.
MINIMUM_FIT_KNOTS = 4


class NoDataRateWarning(UserWarning):
    """A hazard's ``no_data`` rate exceeded the policy threshold."""


class NoDataRateError(ValueError):
    """Raised instead of warning when ``CDFCurveFitPolicy.strict`` is set."""


class _RowRejected(ValueError):
    """A source row cannot be fitted; carries a stable machine-readable reason."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class CDFColumnSchema:
    """Column mapping for one row-per-distribution Arrow source."""

    cell: str = "hex_id"
    hazard: str = "index_name"
    horizon: str = "year"
    pathway: str = "pathway"
    quantiles: str = "cdf_quantiles"
    source_id: str | None = None
    # Optional per-row probability axis. At most one of ``probabilities`` and
    # ``return_periods`` may be set; both are Arrow lists parallel to
    # ``quantiles``. Without either, the shared axis passed to
    # ``fit_cdf_quantile_batches`` applies to every row.
    probabilities: str | None = None
    return_periods: str | None = None
    # Raw samples (an unordered Arrow list per row) instead of quantiles. They
    # are sorted and resampled to ``CDFCurveFitPolicy.sample_quantile_count``
    # evenly spaced probabilities, replacing ``quantiles``.
    samples: str | None = None

    def __post_init__(self) -> None:
        if self.probabilities is not None and self.return_periods is not None:
            raise ValueError("set at most one of probabilities and return_periods")
        if self.samples is not None and (
            self.probabilities is not None or self.return_periods is not None
        ):
            raise ValueError("samples are resampled to a shared axis; drop the axis")


@dataclass(frozen=True)
class CDFCurveFitPolicy:
    """Model and quality policy for source CDF quantile rows."""

    h3_resolution: int
    family: DistributionFamily
    value_unit: str
    value_semantics: str
    producer: str
    source: SourceProvenance
    source_id: str
    creation_version: str = field(default_factory=sdk_version)
    fallback_families: tuple[DistributionFamily, ...] = ()
    atom_policy: Literal["none", "infer_min_plateau"] = "infer_min_plateau"
    minimum_informative_value: float | None = None
    minimum_informative_knots: int = 0
    minimum_distinct_informative_values: int = 0
    parametric_failure_action: Literal["raise", "skip", "tabulated"] = "raise"
    maximum_normalized_rmse: float | None = None
    maximum_absolute_residual: float | None = None
    on_fit_failure: Literal["raise", "skip"] = "raise"
    max_workers: int | None = None
    prefetch: bool = True
    # Physical lower bound (for example 0 for depths). Values are censored to
    # it before fitting, and a fitted base distribution with mass below it
    # becomes a hurdle atom at the bound.
    lower_bound: float | None = None
    # Rows with fewer than ``MINIMUM_FIT_KNOTS`` interior knots: keep them as
    # tabulated curves, or reject them (``on_fit_failure`` then decides).
    short_axis_action: Literal["tabulated", "reject"] = "tabulated"
    sample_quantile_count: int = 1001
    return_period_convention: Literal["one_minus_inverse", "poisson"] = (
        "one_minus_inverse"
    )
    return_period_tail: Literal["upper", "lower"] = "upper"
    source_probability_support: tuple[float, float] | None = None
    # Eligibility: explicit override > registry > the policy values above.
    registry: HazardRegistry | None = None
    minimum_informative_overrides: Mapping[str, float | None] = field(
        default_factory=dict
    )
    # Warn (raise when ``strict``) once a hazard's no_data share of at least
    # ``no_data_min_rows`` rows exceeds the threshold.
    no_data_rate_threshold: float | None = 0.5
    no_data_min_rows: int = 1000
    strict: bool = False
    diagnostics: str | Path | None = None
    diagnostics_rows: Literal["all", "exceptions"] = "all"
    # "1.2" emits files that SDKs older than 0.8 can read; it rejects every
    # option that only schema 1.3 can record.
    schema_version: Literal["1.2", "1.3"] = "1.3"
    probability_semantics: ProbabilitySemantics | None = None
    temporal_window: TemporalWindow | None = None
    ensemble: EnsembleDescriptor | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.h3_resolution <= 15:
            raise ValueError("H3 resolution must be between 0 and 15")
        if self.short_axis_action not in ("tabulated", "reject"):
            raise ValueError("short_axis_action must be 'tabulated' or 'reject'")
        if self.diagnostics_rows not in ("all", "exceptions"):
            raise ValueError("diagnostics_rows must be 'all' or 'exceptions'")
        if self.sample_quantile_count < 2 + MINIMUM_FIT_KNOTS:
            raise ValueError("sample_quantile_count is too small to fit")
        if self.schema_version not in ("1.2", "1.3"):
            raise ValueError("schema_version must be '1.2' or '1.3'")
        if self.schema_version == "1.2":
            only_13 = [
                name
                for name in (
                    "probability_semantics",
                    "temporal_window",
                    "ensemble",
                    "lower_bound",
                )
                if getattr(self, name) is not None
            ]
            if (
                self.source.licence
                or self.source.attribution
                or self.source.retrieved_at
                or self.source.checksum
            ):
                only_13.append("source licence/attribution/retrieved_at/checksum")
            if only_13:
                raise ValueError(f"{', '.join(only_13)} require schema_version='1.3'")
        if self.lower_bound is not None and not np.isfinite(self.lower_bound):
            raise ValueError("lower_bound must be finite")
        if self.no_data_rate_threshold is not None and not (
            0.0 <= self.no_data_rate_threshold <= 1.0
        ):
            raise ValueError("no_data_rate_threshold must be within [0, 1]")
        if self.no_data_min_rows < 1:
            raise ValueError("no_data_min_rows must be positive")
        if self.atom_policy not in ("none", "infer_min_plateau"):
            raise ValueError("unknown atom policy")
        if self.on_fit_failure not in ("raise", "skip"):
            raise ValueError("on_fit_failure must be 'raise' or 'skip'")
        if self.parametric_failure_action not in ("raise", "skip", "tabulated"):
            raise ValueError("unknown parametric failure action")
        if self.family in self.fallback_families:
            raise ValueError("fallback families must not repeat the primary family")
        if len(set(self.fallback_families)) != len(self.fallback_families):
            raise ValueError("fallback families must be unique")
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        for name, value in (
            ("minimum_informative_value", self.minimum_informative_value),
            ("maximum_normalized_rmse", self.maximum_normalized_rmse),
            ("maximum_absolute_residual", self.maximum_absolute_residual),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if (
                name != "minimum_informative_value"
                and value is not None
                and value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_informative_knots < 0:
            raise ValueError("minimum_informative_knots must be non-negative")
        if self.minimum_distinct_informative_values < 0:
            raise ValueError("minimum_distinct_informative_values must be non-negative")

    @property
    def families(self) -> tuple[DistributionFamily, ...]:
        """Ordered parametric families attempted for each eligible row."""
        return (self.family, *self.fallback_families)

    def eligibility_for(self, hazard: str) -> _Eligibility:
        """Effective informative-support screens for one hazard name."""
        value = self.minimum_informative_value
        knots = self.minimum_informative_knots
        distinct = self.minimum_distinct_informative_values
        if self.registry is not None:
            defaults = self.registry.eligibility(hazard)
            if defaults is not None:
                value = defaults.minimum_informative_value
                if defaults.minimum_informative_knots is not None:
                    knots = defaults.minimum_informative_knots
                if defaults.minimum_distinct_informative_values is not None:
                    distinct = defaults.minimum_distinct_informative_values
        # An override may be keyed by the row's own name, or by the registry's
        # canonical name or any alias of it; the row's own name wins.
        labels = [hazard]
        spec = self.registry.get(hazard) if self.registry is not None else None
        if spec is not None:
            labels += [spec.name, *spec.aliases]
        for label in labels:
            if label in self.minimum_informative_overrides:
                value = self.minimum_informative_overrides[label]
                break
        return _Eligibility(value, knots, distinct)


class _Eligibility(NamedTuple):
    minimum_informative_value: float | None
    minimum_informative_knots: int
    minimum_distinct_informative_values: int


@dataclass
class CDFFitSummary:
    """Counters populated as a one-shot fitted stream is consumed."""

    source_rows: int = 0
    canonical_rows: int = 0
    parametric_rows: int = 0
    hurdle_rows: int = 0
    point_mass_rows: int = 0
    tabulated_rows: int = 0
    no_data_rows: int = 0
    skipped_rows: int = 0
    rejected_rows: int = 0
    source_batches: int = 0
    input_wait_seconds: float = 0.0
    fit_and_canonicalize_seconds: float = 0.0
    arrow_build_seconds: float = 0.0
    family_attempts: Counter[str] = field(default_factory=Counter)
    family_successes: Counter[str] = field(default_factory=Counter)
    family_failure_reasons: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    no_data_reasons: Counter[str] = field(default_factory=Counter)
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    treatment_counts: Counter[str] = field(default_factory=Counter)
    rows_by_hazard: Counter[str] = field(default_factory=Counter)
    no_data_by_hazard: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    diagnostics_path: str | None = None
    diagnostics_rows: int = 0


@dataclass(frozen=True)
class CDFFitResult:
    stream: CanonicalHazardStream
    summary: CDFFitSummary


class _ParametricFitSkipped(ValueError):
    """Internal signal carrying failed-family diagnostics for a skipped row."""

    def __init__(self, errors: Sequence[tuple[str, str]]) -> None:
        self.family_errors = tuple(errors)
        joined = "; ".join(f"{family}: {error}" for family, error in errors)
        super().__init__(f"all parametric families failed ({joined})")


class _RowTask(NamedTuple):
    values: np.ndarray[Any, Any]
    axis: np.ndarray[Any, Any] | None
    eligibility: _Eligibility


def _record_batches(values: Any) -> Iterator[pa.RecordBatch]:
    if isinstance(values, pa.RecordBatchReader):
        yield from values
    elif isinstance(values, pa.Table):
        yield from values.to_batches()
    elif isinstance(values, pa.RecordBatch):
        yield values
    else:
        yield from values


def _prefetch_one(values: Iterable[Any]) -> Iterator[Any]:
    """Pull one item ahead on a single thread with a strict one-item bound."""
    iterator = iter(values)
    sentinel = object()

    def next_or_sentinel() -> Any:
        return next(iterator, sentinel)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(next_or_sentinel)
        while True:
            item = future.result()
            if item is sentinel:
                break
            future = executor.submit(next_or_sentinel)
            yield item


def _list_views(
    array: Any, *, label: str = "CDF quantiles", allow_integer: bool = False
) -> Iterator[np.ndarray[Any, Any]]:
    """Yield views over an Arrow list's contiguous child buffer."""
    if isinstance(array, pa.ChunkedArray):
        for chunk in array.chunks:
            yield from _list_views(chunk, label=label, allow_integer=allow_integer)
        return
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise TypeError(f"{label} column must be an Arrow list of numeric values")
    child = array.values.type
    if not (
        pa.types.is_floating(child) or (allow_integer and pa.types.is_integer(child))
    ):
        raise TypeError(f"{label} values must be floating point")
    offsets = array.offsets.to_numpy(zero_copy_only=False)
    child_values = array.values
    if pa.types.is_integer(child):
        # Integers have no NaN: cast first so a null becomes NaN and its row is
        # rejected with a reason instead of failing the whole batch.
        child_values = child_values.cast(pa.float64())
    values = child_values.to_numpy(zero_copy_only=False)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        yield values[int(start) : int(stop)]


def _quality_error(diagnostics: Any, policy: CDFCurveFitPolicy) -> str | None:
    if not diagnostics.converged:
        return f"optimizer did not converge after {diagnostics.iterations} iterations"
    if (
        policy.maximum_normalized_rmse is not None
        and diagnostics.normalized_rmse > policy.maximum_normalized_rmse
    ):
        return (
            f"normalized RMSE {diagnostics.normalized_rmse} exceeds policy "
            f"{policy.maximum_normalized_rmse}"
        )
    if (
        policy.maximum_absolute_residual is not None
        and diagnostics.maximum_absolute_residual > policy.maximum_absolute_residual
    ):
        return (
            f"maximum absolute residual {diagnostics.maximum_absolute_residual} "
            f"exceeds policy {policy.maximum_absolute_residual}"
        )
    return None


def _empty_curve(curve_kind: str, curve_type: str) -> dict[str, Any]:
    return {
        "curve_kind": curve_kind,
        "curve_type": curve_type,
        "curve_shape": None,
        "curve_location": None,
        "curve_scale": None,
        "curve_atom_probability": None,
        "curve_atom_location": None,
        "curve_probabilities": None,
        "curve_values": None,
    }


def _no_data(reason: str) -> dict[str, Any]:
    curve = _empty_curve("no_data", reason)
    curve["_treatment"] = f"no_data:{reason}"
    return curve


def _compact_tabulated(
    probabilities: np.ndarray[Any, Any],
    values: np.ndarray[Any, Any],
    treatment: str = "tabulated_fallback",
) -> dict[str, Any]:
    """Remove only redundant plateau interiors from a quantile function."""
    starts = np.concatenate(([True], values[1:] != values[:-1]))
    ends = np.concatenate((values[:-1] != values[1:], [True]))
    keep = starts | ends
    curve = _empty_curve("tabulated", "linear_probability")
    curve["curve_probabilities"] = probabilities[keep].tolist()
    curve["curve_values"] = values[keep].tolist()
    curve["_treatment"] = treatment
    return curve


def _censor_at_bound(curve: dict[str, Any], base: Any, bound: float) -> None:
    """Turn mass below ``bound`` into an atom at it (a truncated-tail hurdle).

    With mass ``F(bound)`` the hurdle preserves every quantile above the bound
    and maps the base distribution's support below it to the bound: the
    framework's ``HurdleDistribution`` evaluates ``p > atom`` on the *base*
    quantile function (a truncated-tail hurdle), so no rescaling or
    truncated refit is needed. ``test_bound_censoring_preserves_base_quantiles``
    pins this.
    """
    mass = float(base.cdf(bound))
    if mass <= 0.0:
        return
    curve["_censored"] = True
    if mass >= 1.0:
        curve.update(
            curve_kind="point_mass",
            curve_type="point_mass",
            curve_shape=None,
            curve_location=bound,
            curve_scale=0.0,
            curve_atom_probability=1.0,
            curve_atom_location=bound,
        )
    else:
        curve.update(
            curve_kind="hurdle",
            curve_atom_probability=mass,
            curve_atom_location=bound,
        )


def _fit_family(
    probabilities: np.ndarray[Any, Any],
    values: np.ndarray[Any, Any],
    plateau_count: int,
    family: DistributionFamily,
    policy: CDFCurveFitPolicy,
) -> dict[str, Any]:
    distribution: Any
    diagnostics: Any
    if policy.atom_policy == "infer_min_plateau" and plateau_count >= 2:
        hurdle_result = fit_hurdle_quantiles(
            probabilities.tolist(),
            values.tolist(),
            family=family,
            atom_probability=float(probabilities[plateau_count - 1]),
            atom_location=float(values[0]),
        )
        distribution = hurdle_result.distribution
        diagnostics = hurdle_result.diagnostics.tail
    else:
        quantile_result = fit_quantiles(
            probabilities.tolist(),
            values.tolist(),
            family=family,
        )
        distribution = quantile_result.distribution
        diagnostics = quantile_result.diagnostics
    quality_error = _quality_error(diagnostics, policy)
    if quality_error is not None:
        raise ValueError(quality_error)
    base = (
        distribution.base
        if isinstance(distribution, HurdleDistribution)
        else distribution
    )
    curve = _empty_curve(
        "hurdle" if isinstance(distribution, HurdleDistribution) else "fitted",
        base.family,
    )
    curve.update(
        curve_shape=base.shape,
        curve_location=base.location,
        curve_scale=base.scale,
        curve_atom_probability=(
            distribution.atom_probability
            if isinstance(distribution, HurdleDistribution)
            else None
        ),
        curve_atom_location=(
            distribution.atom_location
            if isinstance(distribution, HurdleDistribution)
            else None
        ),
        _treatment=f"parametric:{family}",
        _normalized_rmse=float(diagnostics.normalized_rmse),
        _maximum_absolute_residual=float(diagnostics.maximum_absolute_residual),
    )
    if policy.lower_bound is not None and curve["curve_kind"] == "fitted":
        _censor_at_bound(curve, base, policy.lower_bound)
    return curve


def _return_period_probabilities(
    labels: np.ndarray[Any, Any],
    convention: str,
    tail: str,
) -> np.ndarray[Any, Any]:
    """Vectorized ADR-0003 conversion to canonical non-exceedance probabilities."""
    periods = np.asarray(labels, dtype=np.float64)
    if not np.all(np.isfinite(periods)) or np.any(periods <= 1.0):
        raise _RowRejected(
            "invalid_return_periods", "return periods must be finite and above one"
        )
    if convention == "poisson":
        upper = np.exp(-1.0 / periods)
        return upper if tail == "upper" else 1.0 - upper
    return 1.0 - 1.0 / periods if tail == "upper" else 1.0 / periods


@lru_cache(maxsize=32)
def _sample_positions(count: int) -> np.ndarray[Any, Any]:
    return np.linspace(0.0, 1.0, count)


def _resample_samples(
    samples: np.ndarray[Any, Any], count: int, lower_bound: float | None
) -> np.ndarray[Any, Any]:
    """Linear order statistics: sorted samples on ``count`` even probabilities."""
    array = np.asarray(samples, dtype=np.float64)
    if len(array) == 0:
        raise _RowRejected("missing_values", "row has no samples")
    if not np.all(np.isfinite(array)):
        raise _RowRejected("non_finite_values", "samples must be finite")
    array = np.sort(array)
    if lower_bound is not None:
        array = np.maximum(array, lower_bound)
    if len(array) == 1:
        return np.full(count, array[0])
    resampled: np.ndarray[Any, Any] = np.interp(
        _sample_positions(count), _sample_positions(len(array)), array
    )
    return resampled


def _prepare_row(
    task: _RowTask,
    shared: np.ndarray[Any, Any] | None,
    axis_kind: Literal["shared", "probabilities", "return_periods"],
    sample_mode: bool,
    policy: CDFCurveFitPolicy,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Resolve one row to a strictly increasing probability axis and its values."""
    if sample_mode:
        assert shared is not None
        return shared, _resample_samples(
            task.values, policy.sample_quantile_count, policy.lower_bound
        )
    values = np.asarray(task.values, dtype=np.float64)
    if axis_kind == "shared":
        assert shared is not None
        if len(values) != len(shared):
            raise _RowRejected(
                "axis_length_mismatch",
                f"expected {len(shared)} CDF quantiles, received {len(values)}",
            )
        return shared, values
    assert task.axis is not None
    if len(task.axis) == 0:
        raise _RowRejected("missing_axis_labels", "row has no probability axis labels")
    if len(task.axis) != len(values):
        raise _RowRejected(
            "axis_length_mismatch",
            f"axis has {len(task.axis)} labels for {len(values)} values",
        )
    if axis_kind == "return_periods":
        probabilities = _return_period_probabilities(
            task.axis, policy.return_period_convention, policy.return_period_tail
        )
    else:
        probabilities = np.asarray(task.axis, dtype=np.float64)
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise _RowRejected(
                "invalid_probabilities", "probabilities must be finite within [0, 1]"
            )
    order = np.argsort(probabilities, kind="stable")
    probabilities = probabilities[order]
    if np.any(np.diff(probabilities) <= 0.0):
        raise _RowRejected(
            "duplicate_axis_labels", "probability axis labels must be unique"
        )
    return probabilities, values[order]


def _fit_parameters(
    probabilities: np.ndarray[Any, Any],
    raw_values: np.ndarray[Any, Any],
    policy: CDFCurveFitPolicy,
    eligibility: _Eligibility,
) -> dict[str, Any]:
    values = np.asarray(raw_values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise _RowRejected("non_finite_values", "CDF quantiles must be finite")
    if policy.lower_bound is not None:
        values = np.maximum(values, policy.lower_bound)
    if np.any(np.diff(values) < 0.0):
        raise ValueError("CDF quantiles must be non-decreasing")

    # Point-mass identity is mathematical, not a fit-quality decision. Check
    # the complete source support (including probabilities zero and one)
    # before applying scientific eligibility screens to the interior knots.
    if np.all(values == values[0]):
        location = float(values[0])
        curve = _empty_curve("point_mass", "point_mass")
        curve.update(
            curve_location=location,
            curve_scale=0.0,
            curve_atom_probability=1.0,
            curve_atom_location=location,
            _treatment="point_mass",
        )
        return curve

    active = (probabilities > 0.0) & (probabilities < 1.0)
    knot_probabilities = probabilities[active]
    knot_values = values[active]
    if len(knot_values) < 2:
        raise _RowRejected(
            "insufficient_axis", "at least two interior probability knots are required"
        )

    floor = eligibility.minimum_informative_value
    if floor is not None:
        informative = knot_values[knot_values >= floor]
        if len(informative) == 0:
            return _no_data("below_effective_resolution")
    else:
        informative = knot_values
    if len(informative) < eligibility.minimum_informative_knots:
        return _no_data("insufficient_informative_support")
    if len(np.unique(informative)) < eligibility.minimum_distinct_informative_values:
        return _no_data("degenerate_effective_range")

    if len(knot_values) < MINIMUM_FIT_KNOTS:
        if policy.short_axis_action == "reject":
            raise _RowRejected(
                "short_axis",
                f"at least {MINIMUM_FIT_KNOTS} interior probability knots "
                f"are required, received {len(knot_values)}",
            )
        return _compact_tabulated(
            knot_probabilities, knot_values, treatment="tabulated_short_axis"
        )

    plateau_count = int(np.searchsorted(knot_values, knot_values[0], side="right"))
    errors: list[tuple[str, str]] = []
    for family in policy.families:
        try:
            curve = _fit_family(
                knot_probabilities,
                knot_values,
                plateau_count,
                family,
                policy,
            )
            curve["_attempts"] = [name for name, _ in errors] + [family]
            curve["_family_errors"] = errors
            return curve
        except ValueError as error:
            errors.append((family, str(error)))
    if policy.parametric_failure_action == "tabulated":
        curve = _compact_tabulated(knot_probabilities, knot_values)
        curve["_attempts"] = [name for name, _ in errors]
        curve["_family_errors"] = errors
        return curve
    if policy.parametric_failure_action == "skip":
        raise _ParametricFitSkipped(errors)
    joined = "; ".join(f"{family}: {error}" for family, error in errors)
    raise ValueError(f"all parametric families failed ({joined})")


def _fit_or_error(
    task: _RowTask,
    *,
    shared: np.ndarray[Any, Any] | None,
    axis_kind: Literal["shared", "probabilities", "return_periods"],
    sample_mode: bool,
    policy: CDFCurveFitPolicy,
) -> dict[str, Any] | ValueError:
    try:
        probabilities, values = _prepare_row(
            task, shared, axis_kind, sample_mode, policy
        )
        curve = _fit_parameters(probabilities, values, policy, task.eligibility)
        curve["_minimum_informative_value"] = task.eligibility.minimum_informative_value
        return curve
    except ValueError as error:
        return error


def _family_failure_reason(message: str) -> str:
    if "did not converge" in message:
        return "optimizer_nonconvergence"
    if "normalized RMSE" in message:
        return "normalized_rmse_gate"
    if "maximum absolute residual" in message:
        return "absolute_residual_gate"
    if "at least four" in message:
        return "insufficient_fit_points"
    if "non-zero range" in message:
        return "zero_fit_range"
    return "fit_error"


def _validated_shared_axis(probabilities: Sequence[float]) -> np.ndarray[Any, Any]:
    axis = np.asarray(probabilities, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 2:
        raise ValueError("probabilities must be a one-dimensional sequence")
    if not np.all(np.isfinite(axis)):
        raise ValueError("probabilities must be finite")
    if np.any((axis < 0.0) | (axis > 1.0)):
        raise ValueError("probabilities must be within [0, 1]")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("probabilities must be strictly increasing")
    return axis


def _provenance(
    policy: CDFCurveFitPolicy, input_kind: Literal["probability_labelled", "samples"]
) -> CurveFitProvenance:
    legacy = policy.schema_version == "1.2"
    return CurveFitProvenance(
        input_kind=None if legacy else input_kind,
        sample_resampling=(
            policy.sample_quantile_count
            if input_kind == "samples" and not legacy
            else None
        ),
        lower_bound=policy.lower_bound,
        platform=None if legacy else platform_tag(),
        crc_framework_version=None if legacy else framework_version(),
        families=policy.families,
        selection_metric=(
            "fixed_family" if len(policy.families) == 1 else "first_acceptable"
        ),
        atom_policy=policy.atom_policy,
        constant_policy="point_mass",
        minimum_informative_value=policy.minimum_informative_value,
        minimum_informative_knots=policy.minimum_informative_knots,
        minimum_distinct_informative_values=(
            policy.minimum_distinct_informative_values
        ),
        parametric_failure_action=policy.parametric_failure_action,
        maximum_normalized_rmse=policy.maximum_normalized_rmse,
        maximum_absolute_residual=policy.maximum_absolute_residual,
        on_fit_failure=policy.on_fit_failure,
    )


def _diagnostic_record(
    identity: Mapping[str, Any],
    outcome: str,
    *,
    curve_type: str | None = None,
    normalized_rmse: float | None = None,
    maximum_absolute_residual: float | None = None,
    attempted: Sequence[str] = (),
    failed: Sequence[str] = (),
    reason: str | None = None,
    message: str | None = None,
    treatment: str = "",
    minimum_informative_value: float | None = None,
) -> dict[str, Any]:
    return {
        **identity,
        "outcome": outcome,
        "curve_type": curve_type,
        "normalized_rmse": normalized_rmse,
        "maximum_absolute_residual": maximum_absolute_residual,
        "attempted_families": list(attempted),
        "failed_families": list(failed),
        "fallback": len(attempted) > 1,
        "reason": reason,
        "message": message,
        "treatment": treatment,
        "minimum_informative_value": minimum_informative_value,
    }


def fit_cdf_quantile_batches(
    batches: Iterable[pa.RecordBatch] | pa.RecordBatchReader | pa.Table,
    probabilities: Sequence[float] | None,
    policy: CDFCurveFitPolicy,
    *,
    columns: CDFColumnSchema = CDFColumnSchema(),
) -> CDFFitResult:
    """Lazily canonicalize Arrow distribution rows into schema-1.3 batches.

    Each row carries either quantiles on the shared ``probabilities`` axis, on
    a per-row axis (``columns.probabilities`` or ``columns.return_periods``),
    or raw samples (``columns.samples``, where ``probabilities`` must be
    ``None``). Rows the fitter cannot use are rejected with a reason rather
    than dropped silently: they raise under ``on_fit_failure="raise"`` and are
    counted (and, with ``policy.diagnostics``, recorded) when skipped.
    """
    sample_mode = columns.samples is not None
    axis_kind: Literal["shared", "probabilities", "return_periods"] = (
        "probabilities"
        if columns.probabilities is not None
        else "return_periods"
        if columns.return_periods is not None
        else "shared"
    )
    shared: np.ndarray[Any, Any] | None
    if sample_mode:
        if probabilities is not None:
            raise ValueError("samples are resampled; pass probabilities=None")
        shared = _sample_positions(policy.sample_quantile_count)
    elif axis_kind == "shared":
        if probabilities is None:
            raise ValueError(
                "probabilities are required without a per-row axis or samples"
            )
        shared = _validated_shared_axis(probabilities)
    else:
        if probabilities is not None:
            raise ValueError("pass probabilities=None with a per-row axis column")
        shared = None
    support = policy.source_probability_support
    if shared is not None:
        interior = shared[(shared > 0.0) & (shared < 1.0)]
        if len(interior) < 2:
            raise ValueError("at least two interior probabilities are required")
        support = (float(interior[0]), float(interior[-1]))

    metadata = HazardDatasetMetadata(
        schema_version=policy.schema_version,
        h3_resolution=policy.h3_resolution,
        return_period_tail=policy.return_period_tail,
        source_probability_support=support,
        value_unit=policy.value_unit,
        value_semantics=policy.value_semantics,
        producer=policy.producer,
        source=policy.source,
        fitting=_provenance(
            policy, "samples" if sample_mode else "probability_labelled"
        ),
        probability_semantics=policy.probability_semantics,
        source_return_period_convention=(
            policy.return_period_convention
            if axis_kind == "return_periods" and policy.schema_version == "1.3"
            else None
        ),
        temporal_window=policy.temporal_window,
        ensemble=policy.ensemble,
        creation_version=policy.creation_version,
    )
    summary = CDFFitSummary()
    if policy.diagnostics is not None:
        summary.diagnostics_path = str(policy.diagnostics)
    value_column = columns.samples if columns.samples is not None else columns.quantiles
    axis_column = columns.probabilities or columns.return_periods

    def serial_fitted_batches() -> Iterator[CanonicalHazardBatch]:
        canonical_schema = hazard_arrow_schema(metadata)
        workers = policy.max_workers or detected_cpu_count()
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        sidecar = (
            DiagnosticsWriter(policy.diagnostics)
            if policy.diagnostics is not None
            else None
        )
        guarded: set[str] = set()
        completed = False

        def check_no_data_rates() -> None:
            threshold = policy.no_data_rate_threshold
            if threshold is None:
                return
            for hazard, rows in summary.rows_by_hazard.items():
                if hazard in guarded or rows < policy.no_data_min_rows:
                    continue
                rate = summary.no_data_by_hazard[hazard] / rows
                if rate <= threshold:
                    continue
                guarded.add(hazard)
                text = (
                    f"{rate:.1%} of {rows} {hazard!r} rows are no_data "
                    f"(threshold {threshold:.1%}); the eligibility floor may "
                    "not suit this hazard's value range"
                )
                if policy.strict:
                    raise NoDataRateError(text)
                warnings.warn(text, NoDataRateWarning, stacklevel=2)

        try:
            source_batches = _record_batches(batches)
            batch_iterator = iter(
                _prefetch_one(source_batches) if policy.prefetch else source_batches
            )
            while True:
                waited = perf_counter()
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    break
                summary.input_wait_seconds += perf_counter() - waited
                summary.source_batches += 1
                identity_columns = {
                    columns.cell,
                    columns.hazard,
                    columns.horizon,
                    columns.pathway,
                }
                required = identity_columns | {value_column}
                if columns.source_id is not None:
                    required.add(columns.source_id)
                if axis_column is not None:
                    required.add(axis_column)
                missing = required - set(batch.schema.names)
                if missing:
                    raise ValueError(
                        f"CDF source is missing columns {sorted(missing)!r}"
                    )
                identities = {
                    name: batch.column(batch.schema.get_field_index(name)).to_pylist()
                    for name in identity_columns
                }
                source_ids = (
                    batch.column(
                        batch.schema.get_field_index(columns.source_id)
                    ).to_pylist()
                    if columns.source_id is not None
                    else [policy.source_id] * batch.num_rows
                )
                value_rows = list(
                    _list_views(
                        batch.column(batch.schema.get_field_index(value_column)),
                        label="samples" if sample_mode else "CDF quantiles",
                        allow_integer=sample_mode,
                    )
                )
                axis_rows: list[np.ndarray[Any, Any] | None] = (
                    list(
                        _list_views(
                            batch.column(batch.schema.get_field_index(axis_column)),
                            label="probability axis",
                            allow_integer=True,
                        )
                    )
                    if axis_column is not None
                    else [None] * batch.num_rows
                )
                if (
                    len(value_rows) != batch.num_rows
                    or len(axis_rows) != batch.num_rows
                ):
                    raise AssertionError(
                        "Arrow list row count changed during conversion"
                    )
                eligibility = {
                    hazard: policy.eligibility_for(str(hazard))
                    for hazard in set(identities[columns.hazard])
                }
                tasks = [
                    _RowTask(
                        value_rows[index],
                        axis_rows[index],
                        eligibility[identities[columns.hazard][index]],
                    )
                    for index in range(batch.num_rows)
                ]
                fit_one = partial(
                    _fit_or_error,
                    shared=shared,
                    axis_kind=axis_kind,
                    sample_mode=sample_mode,
                    policy=policy,
                )
                fitted = (
                    executor.map(fit_one, tasks)
                    if executor is not None
                    else map(fit_one, tasks)
                )
                fit_started = perf_counter()
                output_rows: list[dict[str, Any]] = []
                diagnostic_rows: list[dict[str, Any]] = []
                for index, fitted_value in enumerate(fitted):
                    summary.source_rows += 1
                    hazard_name = identities[columns.hazard][index]
                    summary.rows_by_hazard[hazard_name] += 1
                    identity = {
                        "cell_index": identities[columns.cell][index],
                        "source_id": source_ids[index],
                        "hazard_name": hazard_name,
                        "horizon": identities[columns.horizon][index],
                        "pathway": identities[columns.pathway][index],
                    }
                    floor = tasks[index].eligibility.minimum_informative_value
                    example_identity = {
                        "cell_index": identity["cell_index"],
                        "hazard_name": hazard_name,
                        "horizon": identity["horizon"],
                        "pathway": identity["pathway"],
                    }
                    if isinstance(fitted_value, _ParametricFitSkipped):
                        treatment = "skipped:parametric_failure"
                        summary.skipped_rows += 1
                        summary.treatment_counts[treatment] += 1
                        for family, message in fitted_value.family_errors:
                            summary.family_attempts[family] += 1
                            summary.family_failure_reasons[family][
                                _family_failure_reason(message)
                            ] += 1
                        failed = [family for family, _ in fitted_value.family_errors]
                        if len(summary.examples[treatment]) < 3:
                            summary.examples[treatment].append(
                                {
                                    **example_identity,
                                    "failed_families": failed,
                                    "error": str(fitted_value),
                                }
                            )
                        diagnostic_rows.append(
                            _diagnostic_record(
                                identity,
                                "skipped",
                                attempted=failed,
                                failed=failed,
                                reason="parametric_failure",
                                message=str(fitted_value),
                                treatment=treatment,
                                minimum_informative_value=floor,
                            )
                        )
                        continue
                    if isinstance(fitted_value, ValueError):
                        rejected = isinstance(fitted_value, _RowRejected)
                        reason = (
                            fitted_value.reason
                            if isinstance(fitted_value, _RowRejected)
                            else "invalid_or_unhandled"
                        )
                        treatment = (
                            f"rejected:{reason}" if rejected else "invalid_or_unhandled"
                        )
                        if rejected:
                            summary.rejected_rows += 1
                            summary.rejection_reasons[reason] += 1
                        if len(summary.examples[treatment]) < 3:
                            summary.examples[treatment].append(
                                {**example_identity, "error": str(fitted_value)}
                            )
                        if policy.on_fit_failure == "skip":
                            summary.skipped_rows += 1
                            summary.treatment_counts["skipped"] += 1
                            diagnostic_rows.append(
                                _diagnostic_record(
                                    identity,
                                    "rejected" if rejected else "skipped",
                                    reason=reason,
                                    message=str(fitted_value),
                                    treatment=treatment,
                                    minimum_informative_value=floor,
                                )
                            )
                            continue
                        raise ValueError(
                            f"failed to fit CDF row {summary.source_rows - 1}: "
                            f"{fitted_value}"
                        ) from fitted_value
                    curve = dict(fitted_value)
                    treatment = str(curve.pop("_treatment"))
                    attempts = list(curve.pop("_attempts", []))
                    family_errors = list(curve.pop("_family_errors", []))
                    nrmse = curve.pop("_normalized_rmse", None)
                    max_residual = curve.pop("_maximum_absolute_residual", None)
                    censored = bool(curve.pop("_censored", False))
                    curve.pop("_minimum_informative_value", None)
                    kind = curve["curve_kind"]
                    summary.canonical_rows += 1
                    summary.treatment_counts[treatment] += 1
                    if censored:
                        summary.treatment_counts["bound_censored"] += 1
                    for family in attempts:
                        summary.family_attempts[family] += 1
                    for family, message in family_errors:
                        summary.family_failure_reasons[family][
                            _family_failure_reason(message)
                        ] += 1
                    if kind in {"fitted", "hurdle"}:
                        summary.parametric_rows += 1
                        summary.family_successes[str(curve["curve_type"])] += 1
                    summary.hurdle_rows += int(kind == "hurdle")
                    summary.point_mass_rows += int(kind == "point_mass")
                    summary.tabulated_rows += int(kind == "tabulated")
                    summary.no_data_rows += int(kind == "no_data")
                    if kind == "no_data":
                        summary.no_data_reasons[str(curve["curve_type"])] += 1
                        summary.no_data_by_hazard[hazard_name] += 1
                    if (
                        kind in {"no_data", "tabulated"}
                        and len(summary.examples[treatment]) < 3
                    ):
                        example = {
                            **example_identity,
                            "curve_kind": kind,
                            "curve_type": curve["curve_type"],
                        }
                        if family_errors:
                            example["failed_families"] = [
                                family for family, _ in family_errors
                            ]
                        summary.examples[treatment].append(example)
                    routine = kind in {"fitted", "hurdle", "point_mass"} and not (
                        family_errors or censored
                    )
                    if sidecar is not None and not (
                        policy.diagnostics_rows == "exceptions" and routine
                    ):
                        diagnostic_rows.append(
                            _diagnostic_record(
                                identity,
                                kind,
                                curve_type=str(curve["curve_type"]),
                                normalized_rmse=nrmse,
                                maximum_absolute_residual=max_residual,
                                attempted=attempts,
                                failed=[family for family, _ in family_errors],
                                reason=(
                                    str(curve["curve_type"])
                                    if kind == "no_data"
                                    else "bound_censored"
                                    if censored
                                    else None
                                ),
                                treatment=treatment,
                                minimum_informative_value=floor,
                            )
                        )
                    output_rows.append(
                        {
                            "cell_index": identity["cell_index"],
                            "source_id": identity["source_id"],
                            "source_geometry": None,
                            "hazard_name": hazard_name,
                            "horizon": identity["horizon"],
                            "pathway": identity["pathway"],
                            **curve,
                        }
                    )
                summary.fit_and_canonicalize_seconds += perf_counter() - fit_started
                if sidecar is not None:
                    sidecar.write(diagnostic_rows)
                    summary.diagnostics_rows = sidecar.rows
                check_no_data_rates()
                if output_rows:
                    arrow_started = perf_counter()
                    table = pa.Table.from_pylist(output_rows, schema=canonical_schema)
                    summary.arrow_build_seconds += perf_counter() - arrow_started
                    # The fitter constructs the exact canonical Arrow schema and
                    # validates all curve invariants while producing each row.
                    # The persistence boundary validates the table once before
                    # writing; repeating the row-wise reconstruction here made
                    # every fitted batch pay the same validation cost twice.
                    yield CanonicalHazardBatch(hazard_rows=table)
            if sidecar is not None:
                sidecar.finish()
            completed = True
        finally:
            if executor is not None:
                executor.shutdown()
            if sidecar is not None and not completed:
                sidecar.abort()

    def prefetched_batches() -> Iterator[CanonicalHazardBatch]:
        """Overlap one fitted batch with consumption using bounded memory."""
        yield from _prefetch_one(serial_fitted_batches())

    fitted_batches = prefetched_batches if policy.prefetch else serial_fitted_batches

    return CDFFitResult(
        stream=CanonicalHazardStream(metadata=metadata, batches=fitted_batches()),
        summary=summary,
    )
