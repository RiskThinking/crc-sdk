"""Arrow-batched fitting of tabulated CDF quantiles into canonical curves."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import fit_hurdle_quantiles, fit_quantiles
from crc_framework.distributions import DistributionFamily, HurdleDistribution

from crc_sdk.connectors.adapters import CanonicalHazardBatch, CanonicalHazardStream
from crc_sdk.connectors.duckdb import detected_cpu_count
from crc_sdk.connectors.parquet import hazard_arrow_schema, validate_hazard_table
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
    atom_policy: Literal["none", "infer_min_plateau"] = "infer_min_plateau"
    maximum_normalized_rmse: float | None = None
    maximum_absolute_residual: float | None = None
    on_fit_failure: Literal["raise", "skip"] = "raise"
    max_workers: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.h3_resolution <= 15:
            raise ValueError("H3 resolution must be between 0 and 15")
        if self.atom_policy not in ("none", "infer_min_plateau"):
            raise ValueError("unknown atom policy")
        if self.on_fit_failure not in ("raise", "skip"):
            raise ValueError("on_fit_failure must be 'raise' or 'skip'")
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        for name, value in (
            ("maximum_normalized_rmse", self.maximum_normalized_rmse),
            ("maximum_absolute_residual", self.maximum_absolute_residual),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class CDFFitSummary:
    """Counters populated as a one-shot fitted stream is consumed."""

    source_rows: int = 0
    fitted_rows: int = 0
    hurdle_rows: int = 0
    point_mass_rows: int = 0
    skipped_rows: int = 0
    failure_examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CDFFitResult:
    stream: CanonicalHazardStream
    summary: CDFFitSummary


def _record_batches(values: Any) -> Iterator[pa.RecordBatch]:
    if isinstance(values, pa.RecordBatchReader):
        yield from values
    elif isinstance(values, pa.Table):
        yield from values.to_batches()
    elif isinstance(values, pa.RecordBatch):
        yield values
    else:
        yield from values


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

    active = (probabilities > 0.0) & (probabilities < 1.0)
    knot_probabilities = probabilities[active]
    knot_values = values[active]
    if len(knot_values) < 4:
        raise ValueError("at least four interior probability knots are required")

    if np.all(knot_values == knot_values[0]):
        location = float(knot_values[0])
        return {
            "curve_kind": "point_mass",
            "curve_type": "point_mass",
            "curve_shape": None,
            "curve_location": location,
            "curve_scale": 0.0,
            "curve_atom_probability": 1.0,
            "curve_atom_location": location,
        }

    plateau_count = int(np.searchsorted(knot_values, knot_values[0], side="right"))
    distribution: Any
    diagnostics: Any
    if policy.atom_policy == "infer_min_plateau" and plateau_count >= 2:
        hurdle_result = fit_hurdle_quantiles(
            knot_probabilities.tolist(),
            knot_values.tolist(),
            family=policy.family,
            atom_probability=float(knot_probabilities[plateau_count - 1]),
            atom_location=float(knot_values[0]),
        )
        distribution = hurdle_result.distribution
        diagnostics = hurdle_result.diagnostics.tail
    else:
        quantile_result = fit_quantiles(
            knot_probabilities.tolist(),
            knot_values.tolist(),
            family=policy.family,
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
    return {
        "curve_kind": (
            "hurdle" if isinstance(distribution, HurdleDistribution) else "fitted"
        ),
        "curve_type": base.family,
        "curve_shape": base.shape,
        "curve_location": base.location,
        "curve_scale": base.scale,
        "curve_atom_probability": (
            distribution.atom_probability
            if isinstance(distribution, HurdleDistribution)
            else None
        ),
        "curve_atom_location": (
            distribution.atom_location
            if isinstance(distribution, HurdleDistribution)
            else None
        ),
    }


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


def fit_cdf_quantile_batches(
    batches: Iterable[pa.RecordBatch] | pa.RecordBatchReader | pa.Table,
    probabilities: Sequence[float],
    policy: CDFCurveFitPolicy,
    *,
    columns: CDFColumnSchema = CDFColumnSchema(),
) -> CDFFitResult:
    """Lazily fit Arrow CDF rows into canonical schema-1.1 batches."""
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
        schema_version="1.1",
        h3_resolution=policy.h3_resolution,
        source_probability_support=(float(interior[0]), float(interior[-1])),
        value_unit=policy.value_unit,
        value_semantics=policy.value_semantics,
        producer=policy.producer,
        source=policy.source,
        fitting=CurveFitProvenance(
            families=(policy.family,),
            atom_policy=policy.atom_policy,
            maximum_normalized_rmse=policy.maximum_normalized_rmse,
            maximum_absolute_residual=policy.maximum_absolute_residual,
            on_fit_failure=policy.on_fit_failure,
        ),
        creation_version=policy.creation_version,
    )
    summary = CDFFitSummary()

    def fitted_batches() -> Iterator[CanonicalHazardBatch]:
        canonical_schema = hazard_arrow_schema(metadata)
        workers = policy.max_workers or detected_cpu_count()
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            for batch in _record_batches(batches):
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
                output_rows: list[dict[str, Any]] = []
                for index, fitted_value in enumerate(fitted):
                    summary.source_rows += 1
                    if isinstance(fitted_value, ValueError):
                        if len(summary.failure_examples) < 100:
                            summary.failure_examples.append(
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
                            continue
                        raise ValueError(
                            f"failed to fit CDF row {summary.source_rows - 1}: "
                            f"{fitted_value}"
                        ) from fitted_value
                    curve = fitted_value
                    kind = curve["curve_kind"]
                    summary.fitted_rows += 1
                    summary.hurdle_rows += int(kind == "hurdle")
                    summary.point_mass_rows += int(kind == "point_mass")
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
                if output_rows:
                    table = pa.Table.from_pylist(output_rows, schema=canonical_schema)
                    yield CanonicalHazardBatch(
                        hazard_rows=validate_hazard_table(
                            table,
                            metadata=metadata,
                            require_unique_keys=False,
                            max_workers=1,
                        )
                    )
        finally:
            if executor is not None:
                executor.shutdown()

    return CDFFitResult(
        stream=CanonicalHazardStream(metadata=metadata, batches=fitted_batches()),
        summary=summary,
    )
