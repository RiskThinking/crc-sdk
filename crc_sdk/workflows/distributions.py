"""Reconstruct and evaluate canonical hazard curve distributions."""

from __future__ import annotations

import math
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, Literal, Union, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from crc_framework import TransformContext
from crc_framework.distributions import (
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
    TabulatedDistribution,
)

from crc_sdk.connectors.duckdb import detected_cpu_count
from crc_sdk.types import CurveParameters, NoDataCurveError
from crc_sdk.types.dataset import TemporalWindow

CURVE_COLUMNS = (
    "curve_kind",
    "curve_type",
    "curve_shape",
    "curve_location",
    "curve_scale",
    "curve_atom_probability",
    "curve_atom_location",
    "curve_probabilities",
    "curve_values",
)

CurveDistribution = Union[
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
    TabulatedDistribution,
]
_MP_CONTEXT = multiprocessing.get_context("spawn")


class ReturnPeriodExtrapolationWarning(UserWarning):
    """Evaluation requested outside the return periods supporting a fit."""


def warn_if_extrapolated(
    return_periods: Sequence[float],
    support: tuple[float, float] | None,
    *,
    stacklevel: int = 2,
) -> tuple[float, ...]:
    """Warn and return periods outside a fitted source-support interval."""
    periods = tuple(float(period) for period in return_periods)
    return_periods_to_probabilities(periods)
    if support is None:
        return ()
    support_min, support_max = support
    extrapolated = tuple(
        period for period in periods if period < support_min or period > support_max
    )
    if extrapolated:
        import warnings

        warnings.warn(
            f"return periods {extrapolated} are outside fitted source support "
            f"{support}; values are extrapolated",
            ReturnPeriodExtrapolationWarning,
            stacklevel=stacklevel,
        )
    return extrapolated


class ProbabilitySemanticsWarning(UserWarning):
    """Return-period evaluation on data whose probability axis is not one."""


class HorizonExtrapolationWarning(UserWarning):
    """A horizon was requested outside the dataset's fitted temporal window."""


# ADR-0003: only these two kinds are not annual-exceedance-like. Absent
# (``None``) means unspecified, and `estimate_confidence` is not refused.
_NOT_RETURN_PERIOD_SEMANTICS = frozenset(
    {"within_period_percentile", "projection_uncertainty"}
)


def _semantics_of(source: Any) -> str | None:
    if source is None or isinstance(source, str):
        return source
    if isinstance(source, Mapping):
        value = source.get("probability_semantics")
    else:
        value = getattr(source, "probability_semantics", None)
    return None if value is None else str(value)


def check_return_period_semantics(
    metadata_or_semantics: Any,
    *,
    strict: bool = False,
    stacklevel: int = 2,
) -> bool:
    """Check that return-period evaluation is meaningful for a dataset (ADR-0003).

    Accepts dataset metadata (anything with a ``probability_semantics``
    attribute, or a mapping with that key) or the semantics string itself.
    On `within_period_percentile` or `projection_uncertainty` data -- a daily
    percentile or an ensemble spread, not a return period -- this warns with
    :class:`ProbabilitySemanticsWarning`, or raises ``ValueError`` when
    ``strict=True``. `annual_exceedance`, `annual_value_distribution`,
    `estimate_confidence` and unspecified (``None``) are silent. Returns
    ``True`` when return-period evaluation is supported.
    """
    semantics = _semantics_of(metadata_or_semantics)
    if semantics not in _NOT_RETURN_PERIOD_SEMANTICS:
        return True
    message = (
        f"probability_semantics={semantics!r} is not a return-period "
        "distribution; return-period values are not meaningful"
    )
    if strict:
        raise ValueError(message)
    import warnings

    warnings.warn(message, ProbabilitySemanticsWarning, stacklevel=stacklevel)
    return False


def warn_if_outside_window(
    horizon: int,
    window: TemporalWindow | None,
    *,
    strict: bool = False,
    stacklevel: int = 2,
) -> bool:
    """Warn when a horizon lies outside a dataset's fitted temporal window.

    A ``window`` dataset covers ``start_year..end_year`` inclusive; a
    ``time_invariant`` one covers only its ``reference_year``. Outside that the
    evaluation is an extrapolation: this warns with
    :class:`HorizonExtrapolationWarning`, or raises ``ValueError`` when
    ``strict=True``. ``window=None`` (no declared window) is silent. Returns
    ``True`` when the horizon is inside the window.
    """
    if window is None:
        return True
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer year")
    if window.kind == "time_invariant":
        inside = horizon == window.reference_year
        described = f"time-invariant at reference year {window.reference_year}"
    else:
        assert window.start_year is not None and window.end_year is not None
        inside = window.start_year <= horizon <= window.end_year
        described = f"window {window.start_year}-{window.end_year}"
    if inside:
        return True
    message = f"horizon {horizon} is outside the dataset's temporal {described}"
    if strict:
        raise ValueError(message)
    import warnings

    warnings.warn(message, HorizonExtrapolationWarning, stacklevel=stacklevel)
    return False


def _python_value(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def curve_parameters_from_row(row: Mapping[str, Any]) -> CurveParameters:
    """Reconstruct validated curve parameters from one canonical row mapping."""
    try:
        values = {
            name: _python_value(row.get(name))
            if name in {"curve_probabilities", "curve_values"}
            else _python_value(row[name])
            for name in CURVE_COLUMNS
        }
    except KeyError as error:
        raise ValueError(
            f"canonical row is missing curve column {error.args[0]!r}"
        ) from error
    return CurveParameters.model_validate(values)


def _row_at(value: Any, row_index: int) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        if row_index != 0:
            raise IndexError("row_index must be 0 when evaluating a row mapping")
        return value
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise TypeError("row_index must be an integer")
    if row_index < 0:
        raise IndexError("row_index must not be negative")
    if not hasattr(value, "num_rows") or not hasattr(value, "slice"):
        raise TypeError("hazard row must be a mapping or Arrow table-like value")
    if row_index >= value.num_rows:
        raise IndexError(
            f"row_index {row_index} is outside a table with {value.num_rows} rows"
        )
    rows = value.slice(row_index, 1).to_pylist()
    return cast(Mapping[str, Any], rows[0])


def distribution_from_hazard_row(
    value: Any, *, row_index: int = 0
) -> CurveDistribution:
    """Reconstruct one framework distribution from a row or Arrow table."""
    return curve_parameters_from_row(_row_at(value, row_index)).to_distribution()


def return_periods_to_probabilities(
    return_periods: Sequence[float],
    *,
    tail: Literal["upper", "lower"] = "upper",
    convention: Literal["one_minus_inverse", "poisson"] = "one_minus_inverse",
) -> tuple[float, ...]:
    """Map unique return periods to the probability their curve was fitted at.

    `convention` is the source's T-to-probability relation (ADR-0003):
    `one_minus_inverse` is `1 - 1/T` (the default) and `poisson` is
    `exp(-1/T)`, i.e. an annual exceedance probability of `1 - exp(-1/T)`.
    Both give non-exceedance probabilities for the upper tail; the lower tail
    is their complement.

    `tail="upper"` (default) returns non-exceedance probabilities
    (`1 - 1/period`) -- the convention for hazards where rarer events are
    worse at *higher* values (e.g. flood depth), matching
    `CurveFitIngestPolicy(tail="upper")` at ingest. `tail="lower"` returns
    `1/period`, the counterpart for hazards where rarer events are worse at
    *lower* values (e.g. drought severity), matching
    `CurveFitIngestPolicy(tail="lower")`. A curve's fitted distribution has
    no memory of which tail it was fitted under -- its `curve_kind`/
    `curve_type`/etc. reconstruct a plain quantile function -- so the caller
    must pass the same `tail` used at ingest to evaluate it at the intended
    return period; passing the wrong one silently reads the *other* tail's
    convention instead of raising.
    """
    if tail not in ("upper", "lower"):
        raise ValueError("tail must be 'upper' or 'lower'")
    periods = tuple(return_periods)
    if not periods:
        raise ValueError("at least one return period is required")
    normalized: list[float] = []
    for period in periods:
        if isinstance(period, bool) or not isinstance(period, (int, float)):
            raise TypeError("return periods must be numeric")
        value = float(period)
        if not math.isfinite(value) or value <= 1.0:
            raise ValueError("return periods must be finite and greater than 1")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("return periods must be unique")
    if convention not in ("one_minus_inverse", "poisson"):
        raise ValueError("convention must be 'one_minus_inverse' or 'poisson'")
    if convention == "poisson":
        upper = tuple(math.exp(-1.0 / period) for period in normalized)
        return upper if tail == "upper" else tuple(1.0 - p for p in upper)
    if tail == "upper":
        return tuple(1.0 - 1.0 / period for period in normalized)
    return tuple(1.0 / period for period in normalized)


def _period_label(period: float) -> str:
    decimal = Decimal(str(period))
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", "_")


def return_period_value_columns(
    return_periods: Sequence[float],
) -> tuple[str, ...]:
    """Return deterministic wide value-column names for return periods."""
    periods = tuple(return_periods)
    return_periods_to_probabilities(periods)
    columns = tuple(f"value_rp{_period_label(float(rp))}" for rp in periods)
    if len(set(columns)) != len(columns):
        raise ValueError("return periods produce colliding output column names")
    return columns


def _curve_quantiles(
    records: Sequence[Mapping[str, Any]],
    probabilities: tuple[float, ...],
    *,
    impact: Any | None = None,
    context_columns: Mapping[str, str] | None = None,
) -> list[tuple[float | None, ...]]:
    results: list[tuple[float | None, ...]] = []
    for row in records:
        try:
            values = (
                curve_parameters_from_row(row)
                .to_distribution()
                .quantiles(probabilities)
            )
        except NoDataCurveError:
            results.append(tuple(None for _ in probabilities))
            continue
        if impact is not None:
            base_context = getattr(impact.function, "context", None)
            if not isinstance(base_context, TransformContext):
                base_context = TransformContext()
            cell = base_context.cell
            country = base_context.country
            continent = base_context.continent
            building_type = base_context.building_type
            historic_mean = base_context.historic_mean
            row_cell = _python_value(row.get("cell_index"))
            if row_cell is not None:
                cell = int(row_cell)
            for field_name, column_name in (context_columns or {}).items():
                value = _python_value(row[column_name])
                if value is None:
                    continue
                if field_name == "country":
                    country = str(value)
                elif field_name == "continent":
                    continent = str(value)
                elif field_name == "building_type":
                    building_type = str(value)
                elif field_name == "historic_mean":
                    historic_mean = float(value)
            context = TransformContext(
                cell=cell,
                country=country,
                continent=continent,
                building_type=building_type,
                historic_mean=historic_mean,
            )
            values = impact.function.evaluate(values, context=context)
        results.append(tuple(float(value) for value in values))
    return results


def _evaluate_in_chunks(
    records: Sequence[Mapping[str, Any]],
    probabilities: tuple[float, ...],
    chunk_rows: int,
    executor: ProcessPoolExecutor | None,
    *,
    impact: Any | None = None,
    context_columns: Mapping[str, str] | None = None,
) -> list[tuple[float | None, ...]]:
    if not records or executor is None or len(records) <= chunk_rows:
        return _curve_quantiles(
            records,
            probabilities,
            impact=impact,
            context_columns=context_columns,
        )
    chunks = [
        records[start : start + chunk_rows]
        for start in range(0, len(records), chunk_rows)
    ]
    results = executor.map(
        partial(
            _curve_quantiles,
            probabilities=probabilities,
            impact=impact,
            context_columns=context_columns,
        ),
        chunks,
    )
    return [value for chunk in results for value in chunk]


def curve_quantiles(
    table: Any,
    probabilities: Sequence[float],
    *,
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> list[tuple[float | None, ...]]:
    """Evaluate multiple non-exceedance probabilities for every curve row."""
    normalized = tuple(float(probability) for probability in probabilities)
    if not normalized:
        raise ValueError("at least one probability is required")
    if any(
        not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in normalized
    ):
        raise ValueError("probabilities must be finite and between 0 and 1")
    records = table.to_pylist()
    if not records:
        return []
    workers = max_workers or detected_cpu_count()
    if workers <= 1 or len(records) <= chunk_rows:
        return _curve_quantiles(records, normalized)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=_MP_CONTEXT,
    ) as executor:
        return _evaluate_in_chunks(
            records,
            normalized,
            chunk_rows,
            executor,
        )


def stream_curve_quantiles_wide_to_parquet(
    con: Any,
    source_sql: str,
    probabilities: Sequence[float],
    output_path: str | Path,
    *,
    passthrough_columns: Sequence[str],
    value_columns: Sequence[str],
    parquet_metadata: Mapping[bytes, bytes] | None = None,
    impact: Any | None = None,
    context_columns: Mapping[str, str] | None = None,
    batch_rows: int = 50_000,
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> int:
    """Stream curve rows to a wide, multi-quantile Parquet dataset."""
    normalized = tuple(float(probability) for probability in probabilities)
    if len(normalized) != len(value_columns):
        raise ValueError("probabilities and value_columns must have equal length")
    if len(set(value_columns)) != len(value_columns):
        raise ValueError("value_columns must be unique")
    if set(passthrough_columns) & set(value_columns):
        raise ValueError("passthrough and value columns must not overlap")
    # Validate probabilities even for an empty source.
    if not normalized or any(
        not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in normalized
    ):
        raise ValueError("probabilities must be finite and between 0 and 1")

    reader = con.execute(source_sql).to_arrow_reader(batch_rows)
    output_schema = pa.schema(
        [reader.schema.field(name) for name in passthrough_columns]
        + [pa.field(name, pa.float64()) for name in value_columns],
        metadata=dict(parquet_metadata or {}),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = max_workers or detected_cpu_count()
    written = 0

    def _drain(executor: ProcessPoolExecutor | None) -> None:
        nonlocal written
        with pq.ParquetWriter(output, output_schema, compression="zstd") as writer:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                evaluation_columns = [
                    name for name in CURVE_COLUMNS if name in batch.schema.names
                ]
                if impact is not None:
                    for name in ("cell_index", *(context_columns or {}).values()):
                        if name not in evaluation_columns:
                            evaluation_columns.append(name)
                evaluation_table = pa.Table.from_arrays(
                    [batch.column(name) for name in evaluation_columns],
                    names=evaluation_columns,
                )
                values = _evaluate_in_chunks(
                    evaluation_table.to_pylist(),
                    normalized,
                    chunk_rows,
                    executor,
                    impact=impact,
                    context_columns=context_columns,
                )
                value_arrays = [
                    pa.array(
                        [row[index] for row in values],
                        type=pa.float64(),
                    )
                    for index in range(len(value_columns))
                ]
                out_batch = pa.RecordBatch.from_arrays(
                    [batch.column(name) for name in passthrough_columns] + value_arrays,
                    schema=output_schema,
                )
                writer.write_batch(out_batch)
                written += out_batch.num_rows

    if workers <= 1:
        _drain(None)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_MP_CONTEXT,
        ) as executor:
            _drain(executor)
    return written
