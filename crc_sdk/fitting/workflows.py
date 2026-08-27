"""Arrow-batched fitting of tabulated CDF quantiles into canonical curves."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import fit_hurdle_quantiles, fit_quantiles
from crc_framework.distributions import DistributionFamily, HurdleDistribution

from crc_sdk.connectors.adapters import CanonicalHazardBatch, CanonicalHazardStream
from crc_sdk.connectors.duckdb import detected_cpu_count
from crc_sdk.connectors.parquet import hazard_arrow_schema
from crc_sdk.types import CurveFitProvenance, HazardDatasetMetadata, SourceProvenance


@dataclass(frozen=True)
class CDFColumnSchema:
    """Column mapping for one row-per-distribution Arrow source."""

    cell: str = "hex_id"
    hazard: str = "index_name"
    horizon: str = "year"
    pathway: str = "pathway"
    quantiles: str = "cdf_quantiles"
    source_id: str | None = None


@dataclass(frozen=True)
class CDFCurveFitPolicy:
    """Model and quality policy for source CDF quantile rows."""

    h3_resolution: int
    family: DistributionFamily
    value_unit: str
    value_semantics: str
    producer: str
    creation_version: str
    source: SourceProvenance
    source_id: str
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

    def __post_init__(self) -> None:
        if not 0 <= self.h3_resolution <= 15:
            raise ValueError("H3 resolution must be between 0 and 15")
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
    treatment_counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )


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


def _list_views(array: Any) -> Iterator[np.ndarray[Any, Any]]:
    """Yield views over an Arrow list's contiguous child buffer."""
    if isinstance(array, pa.ChunkedArray):
        for chunk in array.chunks:
            yield from _list_views(chunk)
        return
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise TypeError("CDF quantiles column must be an Arrow list of floating values")
    if not pa.types.is_floating(array.values.type):
        raise TypeError("CDF quantile values must be floating point")
    offsets = array.offsets.to_numpy(zero_copy_only=False)
    values = array.values.to_numpy(zero_copy_only=False)
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
    probabilities: np.ndarray[Any, Any], values: np.ndarray[Any, Any]
) -> dict[str, Any]:
    """Remove only redundant plateau interiors from a quantile function."""
    starts = np.concatenate(([True], values[1:] != values[:-1]))
    ends = np.concatenate((values[:-1] != values[1:], [True]))
    keep = starts | ends
    curve = _empty_curve("tabulated", "linear_probability")
    curve["curve_probabilities"] = probabilities[keep].tolist()
    curve["curve_values"] = values[keep].tolist()
    curve["_treatment"] = "tabulated_fallback"
    return curve


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
    )
    return curve


def _fit_parameters(
    probabilities: np.ndarray[Any, Any],
    raw_values: np.ndarray[Any, Any],
    policy: CDFCurveFitPolicy,
) -> dict[str, Any]:
    values = np.asarray(raw_values, dtype=np.float64)
    if len(values) != len(probabilities):
        raise ValueError(
            f"expected {len(probabilities)} CDF quantiles, received {len(values)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("CDF quantiles must be finite")
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
    if len(knot_values) < 4:
        raise ValueError("at least four interior probability knots are required")

    if policy.minimum_informative_value is not None:
        informative = knot_values[knot_values >= policy.minimum_informative_value]
        if len(informative) == 0:
            return _no_data("below_effective_resolution")
    else:
        informative = knot_values
    if len(informative) < policy.minimum_informative_knots:
        return _no_data("insufficient_informative_support")
    if len(np.unique(informative)) < policy.minimum_distinct_informative_values:
        return _no_data("degenerate_effective_range")

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
    values: np.ndarray[Any, Any],
    *,
    probabilities: np.ndarray[Any, Any],
    policy: CDFCurveFitPolicy,
) -> dict[str, Any] | ValueError:
    try:
        return _fit_parameters(probabilities, values, policy)
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


def fit_cdf_quantile_batches(
    batches: Iterable[pa.RecordBatch] | pa.RecordBatchReader | pa.Table,
    probabilities: Sequence[float],
    policy: CDFCurveFitPolicy,
    *,
    columns: CDFColumnSchema = CDFColumnSchema(),
) -> CDFFitResult:
    """Lazily canonicalize Arrow CDF rows into schema-1.2 batches."""
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim != 1 or len(probability_array) < 4:
        raise ValueError("probabilities must be a one-dimensional sequence")
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probabilities must be finite")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must be within [0, 1]")
    if np.any(np.diff(probability_array) <= 0.0):
        raise ValueError("probabilities must be strictly increasing")
    interior = probability_array[(probability_array > 0.0) & (probability_array < 1.0)]
    if len(interior) < 4:
        raise ValueError("at least four interior probabilities are required")

    metadata = HazardDatasetMetadata(
        schema_version="1.2",
        h3_resolution=policy.h3_resolution,
        source_probability_support=(float(interior[0]), float(interior[-1])),
        value_unit=policy.value_unit,
        value_semantics=policy.value_semantics,
        producer=policy.producer,
        source=policy.source,
        fitting=CurveFitProvenance(
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
        ),
        creation_version=policy.creation_version,
    )
    summary = CDFFitSummary()

    def serial_fitted_batches() -> Iterator[CanonicalHazardBatch]:
        canonical_schema = hazard_arrow_schema(metadata)
        workers = policy.max_workers or detected_cpu_count()
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
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
                required = identity_columns | {columns.quantiles}
                if columns.source_id is not None:
                    required.add(columns.source_id)
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
                quantile_array = batch.column(
                    batch.schema.get_field_index(columns.quantiles)
                )
                quantile_rows = list(_list_views(quantile_array))
                if len(quantile_rows) != batch.num_rows:
                    raise AssertionError(
                        "Arrow list row count changed during conversion"
                    )
                fit_one = partial(
                    _fit_or_error,
                    probabilities=probability_array,
                    policy=policy,
                )
                fitted = (
                    executor.map(fit_one, quantile_rows)
                    if executor is not None
                    else map(fit_one, quantile_rows)
                )
                fit_started = perf_counter()
                output_rows: list[dict[str, Any]] = []
                for index, fitted_value in enumerate(fitted):
                    summary.source_rows += 1
                    if isinstance(fitted_value, _ParametricFitSkipped):
                        treatment = "skipped:parametric_failure"
                        summary.skipped_rows += 1
                        summary.treatment_counts[treatment] += 1
                        for family, message in fitted_value.family_errors:
                            summary.family_attempts[family] += 1
                            summary.family_failure_reasons[family][
                                _family_failure_reason(message)
                            ] += 1
                        if len(summary.examples[treatment]) < 3:
                            summary.examples[treatment].append(
                                {
                                    "cell_index": identities[columns.cell][index],
                                    "hazard_name": identities[columns.hazard][index],
                                    "horizon": identities[columns.horizon][index],
                                    "pathway": identities[columns.pathway][index],
                                    "failed_families": [
                                        family
                                        for family, _ in fitted_value.family_errors
                                    ],
                                    "error": str(fitted_value),
                                }
                            )
                        continue
                    if isinstance(fitted_value, ValueError):
                        treatment = "invalid_or_unhandled"
                        if len(summary.examples[treatment]) < 3:
                            summary.examples[treatment].append(
                                {
                                    "cell_index": identities[columns.cell][index],
                                    "hazard_name": identities[columns.hazard][index],
                                    "horizon": identities[columns.horizon][index],
                                    "pathway": identities[columns.pathway][index],
                                    "error": str(fitted_value),
                                }
                            )
                        if policy.on_fit_failure == "skip":
                            summary.skipped_rows += 1
                            summary.treatment_counts["skipped"] += 1
                            continue
                        raise ValueError(
                            f"failed to fit CDF row {summary.source_rows - 1}: "
                            f"{fitted_value}"
                        ) from fitted_value
                    curve = dict(fitted_value)
                    treatment = str(curve.pop("_treatment"))
                    attempts = list(curve.pop("_attempts", []))
                    family_errors = list(curve.pop("_family_errors", []))
                    kind = curve["curve_kind"]
                    summary.canonical_rows += 1
                    summary.treatment_counts[treatment] += 1
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
                    if (
                        kind in {"no_data", "tabulated"}
                        and len(summary.examples[treatment]) < 3
                    ):
                        example = {
                            "cell_index": identities[columns.cell][index],
                            "hazard_name": identities[columns.hazard][index],
                            "horizon": identities[columns.horizon][index],
                            "pathway": identities[columns.pathway][index],
                            "curve_kind": kind,
                            "curve_type": curve["curve_type"],
                        }
                        if family_errors:
                            example["failed_families"] = [
                                family for family, _ in family_errors
                            ]
                        summary.examples[treatment].append(example)
                    output_rows.append(
                        {
                            "cell_index": identities[columns.cell][index],
                            "source_id": source_ids[index],
                            "source_geometry": None,
                            "hazard_name": identities[columns.hazard][index],
                            "horizon": identities[columns.horizon][index],
                            "pathway": identities[columns.pathway][index],
                            **curve,
                        }
                    )
                summary.fit_and_canonicalize_seconds += perf_counter() - fit_started
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
        finally:
            if executor is not None:
                executor.shutdown()

    def prefetched_batches() -> Iterator[CanonicalHazardBatch]:
        """Overlap one fitted batch with consumption using bounded memory."""
        yield from _prefetch_one(serial_fitted_batches())

    fitted_batches = prefetched_batches if policy.prefetch else serial_fitted_batches

    return CDFFitResult(
        stream=CanonicalHazardStream(metadata=metadata, batches=fitted_batches()),
        summary=summary,
    )
