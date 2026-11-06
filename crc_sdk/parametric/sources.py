"""Outcome samples from fitted curves and from per-year values.

**Monte Carlo from a curve** (`sample_curve`). A canonical curve is a
quantile function over the non-exceedance probability. Each simulated year
draws `u` from `rng.random(n_draws)` and takes `distribution.quantiles(u)`,
the inverse-CDF route of crc-framework's own `sample(size, rng=...)`. A draw
is a year only if the curve describes one year's value, so the curve's
`probability_semantics` must be `annual_exceedance` (the curve of an annual
maximum) or `annual_value_distribution` (an annual value):

- any other declared value raises `ProbabilitySemanticsError`;
- unspecified semantics raise too, unless the caller states one of the two
  with `assume_semantics=`; a stated value that contradicts the declared one
  raises.

A hurdle curve keeps its atom: a dry-year draw gets the atom's value.
`no_data` rows raise; they never become zero. Draws outside the curve's
fitted support are counted on each side: rarer than the longest fitted
return period warns with `ReturnPeriodExtrapolationWarning`; more frequent
than the shortest is recorded and named in the provenance's notes. A
tabulated curve cannot be evaluated outside its knots, so by default its
sample raises; `outside_support="extrapolate"` uses crc-framework's own
extrapolation and `"clamp"` the nearest knot, and the choice is recorded.

**Per-year values** (`AnnualSeries`, `sample_annual`). The years of an
`annual_extremes()` table (ERA5, CMIP6, NEX-GDDP-CMIP6, stations) become the
rows; nothing is assumed about probabilities, and no trend is removed. A
missing or null year raises, unless `missing="drop_year"` drops it and lists
it in the provenance. A second series as `index=` reads the index at another
cell, station or hazard than the loss: the dependence between index and loss
is then the data's own.

**Hazard units.** Before anything is drawn, both functions check the hazard's
unit (the curve's `value_unit`, the hazard series' `unit`) against the loss
model's impact with `LossModel.check_hazard_unit`: an `ImpactExpression`
calibrated for another unit raises `SampleError`. Impacts that declare no
unit are not checked; the loss model's document records `exposure_unit: None`.

Draw order, with one generator per call: the hazard uniforms first, then the
basis-risk model. The provenance records the generator's state before each
step, so a sample can be rebuilt exactly.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, get_args

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from crc_framework.distributions import (
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
    TabulatedDistribution,
)
from numpy.typing import NDArray

from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.basis_risk import BasisRisk, ImpactNoise, NoBasisRisk
from crc_sdk.parametric.errors import (
    ComonotoneIndexWarning,
    ProbabilitySemanticsError,
    SampleError,
)
from crc_sdk.parametric.loss import LossModel
from crc_sdk.parametric.sample import (
    COMONOTONE_MESSAGE,
    OutcomeSample,
    SampleProvenance,
    _arrow_table,
    _check_json,
    _column,
    _freeze,
    _generator_record,
    _identifier,
    _integer_values,
    _licence_entry,
    _require_generator,
    _source_document,
    _thaw,
)
from crc_sdk.types import CurveParameters, HazardDatasetMetadata
from crc_sdk.workflows.distributions import (
    ReturnPeriodExtrapolationWarning,
    curve_parameters_from_row,
    return_periods_to_probabilities,
)

__all__ = [
    "AnnualSemantics",
    "AnnualSeries",
    "MissingYears",
    "OutsideSupport",
    "sample_annual",
    "sample_curve",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

AnnualSemantics = Literal["annual_exceedance", "annual_value_distribution"]
"""The probability semantics under which one curve draw is one year."""

OutsideSupport = Literal["raise", "extrapolate", "clamp"]
"""What a tabulated curve does with draws outside its knots."""

MissingYears = Literal["raise", "drop_year"]
"""What `sample_annual` does with a year without a value."""

_ANNUAL_SEMANTICS: tuple[str, ...] = get_args(AnnualSemantics)
_EARTH_RADIUS_KM = 6371.0088
_ROW_FIELDS = ("hazard_name", "horizon", "pathway", "source_id")
_TABLE_PROVENANCE_COLUMNS = (
    "archive_licence",
    "archive_licence_url",
    "attribution",
    "calendar",
    "datum_epoch",
    "datum_label",
    "grid_label",
    "licence",
    "member",
    "model",
    "original_model_licence",
    "recipe",
    "reference_notice",
    "retrieved_at",
    "scenario",
    "source_url",
    "station",
    "station_id",
    "station_name",
    "version",
)
_MAX_RECORDED_VALUES = 20


# -- shared checks ------------------------------------------------------------


def _check_models(loss_model: Any, basis_risk: Any, asset_id: str) -> None:
    if not isinstance(loss_model, LossModel):
        raise TypeError(
            f"loss_model must be a LossModel, got {type(loss_model).__name__}"
        )
    if basis_risk is None or not isinstance(basis_risk, BasisRisk):
        raise TypeError(
            "basis_risk must be a BasisRisk model; pass NoBasisRisk() to state "
            "that the index is the hazard at the asset"
        )
    if loss_model.asset_id is not None and loss_model.asset_id != asset_id:
        raise SampleError(
            f"the loss model is bound to asset {loss_model.asset_id!r}, "
            f"not {asset_id!r}"
        )


def _apply_basis_risk(
    model: BasisRisk,
    index: FloatArray,
    hazard: FloatArray,
    impact: FloatArray,
    rng: np.random.Generator | None,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Apply a basis-risk model, check its output and summarise its effect."""
    result = model.apply(index.copy(), hazard.copy(), impact.copy(), rng)
    if not isinstance(result, tuple) or len(result) != 2:
        raise SampleError(
            f"basis-risk model {model.name!r} must return (index, impact)"
        )
    try:
        new_index = np.asarray(result[0], dtype=np.float64)
        new_impact = np.asarray(result[1], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SampleError(
            f"basis-risk model {model.name!r} returned non-numbers"
        ) from error
    if new_index.shape != index.shape or new_impact.shape != impact.shape:
        raise SampleError(
            f"basis-risk model {model.name!r} changed the shape of the index or impact"
        )
    if not np.isfinite(new_index).all():
        raise SampleError(
            f"basis-risk model {model.name!r} returned non-finite indices"
        )
    if not np.isfinite(new_impact).all() or (new_impact < 0).any():
        raise SampleError(
            f"basis-risk model {model.name!r} returned negative or non-finite impacts"
        )
    before = math.fsum(impact.tolist())
    summary: dict[str, Any] = {
        "impact_mean_ratio": (
            math.fsum(new_impact.tolist()) / before if before > 0 else None
        ),
        "index_changed": not np.array_equal(new_index, index),
    }
    if isinstance(model, ImpactNoise) and model.upper is not None:
        summary["capped_draws"] = int((new_impact >= model.upper).sum())
    return new_index, new_impact, summary


def _basis_risk_notes(model: BasisRisk, separate_index: bool) -> list[str]:
    if separate_index:
        return ["The index is read from another series than the loss."]
    if isinstance(model, NoBasisRisk):
        return ["No basis-risk model: the index and the loss are comonotone."]
    return []


def _warn_comonotone(model: BasisRisk, separate_index: bool) -> None:
    if isinstance(model, NoBasisRisk) and not separate_index:
        warnings.warn(COMONOTONE_MESSAGE, ComonotoneIndexWarning, stacklevel=3)


# -- Monte Carlo from a curve -------------------------------------------------


def _semantics(
    metadata: HazardDatasetMetadata, assume: Any
) -> tuple[str, Literal["dataset", "caller"]]:
    if assume is not None and assume not in _ANNUAL_SEMANTICS:
        raise ProbabilitySemanticsError(
            f"assume_semantics must be one of {_ANNUAL_SEMANTICS}, got {assume!r}"
        )
    declared = metadata.probability_semantics
    if declared is None:
        if assume is None:
            raise ProbabilitySemanticsError(
                "the dataset declares no probability_semantics, so a draw from its "
                "curve is not known to be a year; state assume_semantics="
                "'annual_exceedance' (an annual maximum) or "
                "'annual_value_distribution' (an annual value) if it is"
            )
        return str(assume), "caller"
    if declared not in _ANNUAL_SEMANTICS:
        raise ProbabilitySemanticsError(
            f"probability_semantics={declared!r} is not a distribution of annual "
            f"values, so a draw from the curve is not a year; sampling needs one of "
            f"{_ANNUAL_SEMANTICS}"
        )
    if assume is not None and assume != declared:
        raise ProbabilitySemanticsError(
            f"assume_semantics={assume!r} contradicts the dataset's declared "
            f"probability_semantics={declared!r}"
        )
    return declared, "dataset"


def _fitted_document(distribution: FittedDistribution) -> dict[str, Any]:
    return {
        "family": distribution.family,
        "location": distribution.location,
        "scale": distribution.scale,
        "shape": None if distribution.shape is None else float(distribution.shape),
    }


def _curve_document(distribution: Any) -> dict[str, Any]:
    if isinstance(distribution, HurdleDistribution):
        return {
            "kind": "hurdle",
            **_fitted_document(distribution.base),
            "atom_probability": distribution.atom_probability,
            "atom_location": distribution.atom_location,
        }
    if isinstance(distribution, FittedDistribution):
        return {"kind": "fitted", **_fitted_document(distribution)}
    if isinstance(distribution, PointMassDistribution):
        return {"kind": "point_mass", "location": distribution.location}
    if isinstance(distribution, TabulatedDistribution):
        return {
            "kind": "tabulated",
            "interpolation": distribution.interpolation,
            "extrapolate": bool(distribution.extrapolate),
            "probabilities": distribution.probabilities.tolist(),
            "values": distribution.values.tolist(),
        }
    return {"kind": "other", "type": type(distribution).__name__}


def _plain(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def _resolve_curve(curve: Any) -> tuple[Any, dict[str, Any] | None]:
    """Return the curve's distribution and, for a canonical row, its key."""
    row: dict[str, Any] | None = None
    if isinstance(curve, CurveParameters):
        parameters: CurveParameters = curve
    elif isinstance(curve, Mapping):
        try:
            parameters = curve_parameters_from_row(curve)
        except ValueError as error:
            raise SampleError(
                f"the canonical row is not a valid curve: {error}"
            ) from error
        row = {name: _plain(curve.get(name)) for name in _ROW_FIELDS}
        cell = _plain(curve.get("cell_index"))
        row["cell_index"] = None if cell is None else format(int(cell), "x")
    elif callable(getattr(curve, "quantiles", None)):
        return curve, None
    else:
        raise TypeError(
            "curve must be a crc-framework distribution, CurveParameters or a "
            f"canonical row mapping, got {type(curve).__name__}"
        )
    if parameters.curve_kind == "no_data":
        raise SampleError(
            f"the curve has no data ({parameters.curve_type}); a no_data row is "
            "never sampled as zero"
        )
    return parameters.to_distribution(), row


def _support(metadata: HazardDatasetMetadata) -> dict[str, Any] | None:
    """The fitted support as non-exceedance probabilities, frequent and rare end."""
    tail = metadata.return_period_tail
    if metadata.return_period_support is not None:
        shortest, longest = metadata.return_period_support
        frequent, rare = return_periods_to_probabilities((shortest, longest), tail=tail)
        return {
            "basis": "return_period_support",
            "return_periods": [shortest, longest],
            "frequent_probability": frequent,
            "rare_probability": rare,
        }
    if metadata.source_probability_support is not None:
        lower, upper = metadata.source_probability_support
        frequent, rare = (lower, upper) if tail == "upper" else (upper, lower)
        return {
            "basis": "source_probability_support",
            "return_periods": None,
            "frequent_probability": frequent,
            "rare_probability": rare,
        }
    return None


def _draw_count(n_draws: Any) -> int:
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise SampleError(f"n_draws must be a positive integer, got {n_draws!r}")
    if int(n_draws) < 1:
        raise SampleError(f"n_draws must be a positive integer, got {n_draws!r}")
    return int(n_draws)


def sample_curve(
    curve: Any,
    *,
    metadata: HazardDatasetMetadata,
    n_draws: int,
    rng: np.random.Generator,
    loss_model: LossModel,
    basis_risk: BasisRisk,
    asset_id: str,
    assume_semantics: AnnualSemantics | None = None,
    outside_support: OutsideSupport = "raise",
    details: Mapping[str, Any] | None = None,
) -> OutcomeSample:
    """Draw `n_draws` simulated years from a fitted curve.

    `curve` is a crc-framework distribution (for example from
    `distribution_from_hazard_row`), `CurveParameters`, or a canonical row
    mapping with the curve columns, whose key (hazard, horizon, pathway,
    cell, source id) is then recorded. `metadata` is the curve's dataset
    metadata (`HazardDatasetMetadata`): its probability semantics, tail,
    fitted support, unit and source. `details` adds caller information to
    the provenance, for example the file's SHA-256.

    Each draw is one year with probability `1/n_draws`; `period` is the draw
    number. The index is the drawn hazard in the dataset's `value_unit`; the
    loss is `loss_model` applied to it, after `basis_risk`. `rng` is
    required: the uniforms are drawn first, then the basis-risk model draws.

    Raises `ProbabilitySemanticsError` under the semantics rules of the
    module docstring; `SampleError` for a `value_unit` the loss model's
    impact is not calibrated for (checked before any draw), a `no_data`
    curve, a non-finite quantile, a tabulated curve sampled outside its
    knots with `outside_support="raise"`, or invalid arguments; `TypeError`
    for wrong types. Warns with `ReturnPeriodExtrapolationWarning` for draws rarer than
    the fitted support, and with `ComonotoneIndexWarning` for
    `NoBasisRisk()`.
    """
    if not isinstance(metadata, HazardDatasetMetadata):
        raise TypeError(
            f"metadata must be HazardDatasetMetadata, got {type(metadata).__name__}"
        )
    n = _draw_count(n_draws)
    rng = _require_generator(rng)
    asset = _identifier("asset_id", asset_id)
    _check_models(loss_model, basis_risk, asset)
    # Before any draw, so a refused call leaves the generator untouched.
    loss_model.check_hazard_unit(metadata.value_unit)
    if outside_support not in get_args(OutsideSupport):
        raise SampleError(
            f"outside_support must be one of {get_args(OutsideSupport)}, "
            f"got {outside_support!r}"
        )
    semantics, semantics_source = _semantics(metadata, assume_semantics)
    distribution, row = _resolve_curve(curve)
    caller_details = _source_document(details)
    curve_document = _curve_document(distribution)
    notes: list[str] = []
    if semantics_source == "caller":
        notes.append(
            f"The dataset declares no probability semantics; the caller stated "
            f"{semantics!r}."
        )

    rng_records = [_generator_record(rng, "hazard")]
    uniforms = rng.random(n)
    evaluated = uniforms
    tabulated: dict[str, Any] | None = None
    if isinstance(distribution, TabulatedDistribution) and not distribution.extrapolate:
        lowest = float(distribution.probabilities[0])
        highest = float(distribution.probabilities[-1])
        below_knots = int((uniforms < lowest).sum())
        above_knots = int((uniforms > highest).sum())
        tabulated = {
            "knots": [lowest, highest],
            "outside_support": outside_support,
            "below_knots": below_knots,
            "above_knots": above_knots,
        }
        outside = below_knots + above_knots
        if outside and outside_support == "raise":
            raise SampleError(
                f"{outside} of {n} draws fall outside the tabulated curve's knots "
                f"[{lowest}, {highest}]; pass outside_support='extrapolate' or "
                "'clamp' to evaluate them"
            )
        if outside and outside_support == "clamp":
            evaluated = np.clip(uniforms, lowest, highest)
            notes.append(
                f"{outside} of {n} draws fell outside the tabulated knots and were "
                "clamped to the nearest knot."
            )
        elif outside:
            distribution = TabulatedDistribution(
                distribution.probabilities,
                distribution.values,
                interpolation=distribution.interpolation,
                extrapolate=True,
            )
            notes.append(
                f"{outside} of {n} draws fell outside the tabulated knots and were "
                "extrapolated by crc-framework."
            )
    try:
        hazard = np.asarray(distribution.quantiles(evaluated), dtype=np.float64)
    except ValueError as error:
        raise SampleError(
            f"the curve cannot be evaluated at a draw: {error}"
        ) from error
    if hazard.shape != (n,) or not np.isfinite(hazard).all():
        raise SampleError("the curve returned non-finite or misshaped quantiles")

    atom: dict[str, Any] | None = None
    if isinstance(distribution, HurdleDistribution):
        atom = {
            "probability": distribution.atom_probability,
            "location": distribution.atom_location,
            "draws": int((uniforms <= distribution.atom_probability).sum()),
        }

    support = _support(metadata)
    counts: dict[str, int] | None = None
    if support is None:
        notes.append(
            "The curve declares no fitted support, so draws outside it are not counted."
        )
    else:
        frequent = support["frequent_probability"]
        rare = support["rare_probability"]
        if metadata.return_period_tail == "upper":
            below = int((uniforms < frequent).sum())
            above = int((uniforms > rare).sum())
        else:
            below = int((uniforms > frequent).sum())
            above = int((uniforms < rare).sum())
        counts = {"below": below, "above": above}
        periods = support["return_periods"]
        if below:
            shortest = f" ({periods[0]:g} years)" if periods is not None else ""
            notes.append(
                f"{below} of {n} draws are more frequent than the shortest fitted "
                f"return period{shortest}: the curve is extrapolated there."
            )
        if above:
            longest = f" ({periods[1]:g} years)" if periods is not None else ""
            notes.append(
                f"{above} of {n} draws are rarer than the longest fitted return "
                f"period{longest}: the curve is extrapolated there."
            )
    pooling = metadata.ensemble.pooling if metadata.ensemble is not None else None
    if pooling == "pooled":
        notes.append(
            "The curve pools an ensemble: a draw mixes model spread with "
            "year-to-year variability."
        )

    impact = loss_model.impact_values(hazard)
    rng_records.append(_generator_record(rng, "basis_risk"))
    index, impact, basis_summary = _apply_basis_risk(
        basis_risk, hazard, hazard, impact, rng
    )
    loss = loss_model.loss_values(impact)
    notes += _basis_risk_notes(basis_risk, separate_index=False)

    source = {
        "role": "hazard",
        "kind": "curve",
        "metadata": metadata.model_dump(mode="json"),
        "curve": curve_document,
        "row": row,
        "details": caller_details,
    }
    _check_json("the curve's source", source)
    provenance = SampleProvenance(
        source_kind="monte_carlo",
        sources=(source,),
        semantics_source=semantics_source,
        probability_semantics=semantics,
        loss_model=loss_model._document(asset),
        basis_risk=basis_risk.to_document(),
        rng=tuple(rng_records),
        licences=(_licence_entry("hazard", metadata.source.model_dump(mode="json")),),
        details={
            "draws": n,
            "tail": metadata.return_period_tail,
            "support": support,
            "outside_support": counts,
            "atom": atom,
            "tabulated": tabulated,
            "ensemble_pooling": pooling,
            "basis_risk": basis_summary,
        },
        notes=tuple(notes),
        lineage=content_hash(
            {
                "kind": "monte_carlo",
                "asset_id": asset,
                "source": source,
                "rng": rng_records[0],
            }
        ),
    )
    sample = OutcomeSample(
        asset_ids=(asset,),
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=loss,
        hazard=None if np.array_equal(index, hazard) else hazard,
        index_unit=metadata.value_unit,
        loss_unit=loss_model.value_unit,
        semantics="annual",
        provenance=provenance,
    )
    if counts is not None and counts["above"]:
        warnings.warn(
            f"{counts['above']} of {n} draws are rarer than the curve's fitted "
            "support; their values are extrapolated",
            ReturnPeriodExtrapolationWarning,
            stacklevel=2,
        )
    _warn_comonotone(basis_risk, separate_index=False)
    return sample


# -- per-year values -----------------------------------------------------------


def _year_array(name: str, value: Any) -> IntArray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.ndim != 1 or array.size == 0:
        raise SampleError(f"{name} must be a non-empty 1-D sequence of integer years")
    years = array.astype(np.int64)
    if years.size > 1 and not (np.diff(years) > 0).all():
        raise SampleError(f"{name} must be strictly increasing")
    return years


def _requested_years(years: Any) -> IntArray | None:
    if years is None:
        return None
    array = np.asarray(years)
    if array.dtype.kind not in "iu" or array.ndim != 1 or array.size == 0:
        raise SampleError("years must be a non-empty sequence of integer years")
    unique = np.unique(array.astype(np.int64))
    if unique.size != array.size:
        raise SampleError("years must not repeat")
    return unique


def _recorded_columns(rows: Any) -> dict[str, Any]:
    recorded: dict[str, Any] = {}
    for name in _TABLE_PROVENANCE_COLUMNS:
        if name not in rows.column_names:
            continue
        values = sorted(
            {str(value) for value in rows.column(name).to_pylist() if value is not None}
        )
        recorded[name] = {
            "values": values[:_MAX_RECORDED_VALUES],
            "distinct": len(values),
        }
    return recorded


@dataclass(frozen=True, eq=False)
class AnnualSeries:
    """One hazard's value per year at one location.

    `years` are strictly increasing integer years; `values` holds one value
    per year in `unit`, with NaN for a year without a value. `provenance`
    describes the series: its source, the table's columns and filter, the
    matched cell or station. Build one with `from_table` or `nearest_cell`
    from an `annual_extremes()` table, or directly.

    Raises `SampleError` for unordered years, mismatched lengths, infinite
    values, an empty unit or a provenance that is not plain JSON.
    """

    years: IntArray
    values: FloatArray
    unit: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        years = _year_array("years", self.years)
        try:
            values = np.array(self.values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise SampleError("values must be numeric") from error
        if values.shape != years.shape:
            raise SampleError(
                f"values must have one entry per year, shape {years.shape}; "
                f"got {values.shape}"
            )
        if np.isinf(values).any():
            raise SampleError("values must be finite, or NaN for a missing year")
        _identifier("unit", self.unit)
        if not isinstance(self.provenance, Mapping):
            raise SampleError("provenance must be a mapping")
        _check_json("provenance", self.provenance)
        years.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "years", years)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "provenance", _freeze(_thaw(self.provenance)))

    def __len__(self) -> int:
        return int(self.years.shape[0])

    @property
    def missing_years(self) -> tuple[int, ...]:
        """The years without a value."""
        return tuple(int(year) for year in self.years[np.isnan(self.values)])

    @classmethod
    def _from_rows(
        cls,
        rows: Any,
        *,
        year: str,
        value: str,
        unit: str,
        years: IntArray | None,
        provenance: dict[str, Any],
    ) -> AnnualSeries:
        what = "the annual table"
        row_years = _integer_values(_column(rows, year, what), year)
        column = _column(rows, value, what)
        if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
            raise SampleError(f"column {value!r} must be numeric, not {column.type}")
        row_values = np.asarray(
            column.cast(pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64
        )
        if np.isinf(row_values).any():
            raise SampleError(f"column {value!r} holds infinite values")
        unique, counts = np.unique(row_years, return_counts=True)
        if (counts > 1).any():
            raise SampleError(
                f"years {unique[counts > 1][:10].tolist()} have several rows; "
                "select one location (where=, nearest_cell) first"
            )
        unit = _identifier("unit", unit)
        if "unit" in rows.column_names:
            stated = {u for u in rows.column("unit").to_pylist() if u is not None}
            if stated and stated != {unit}:
                raise SampleError(
                    f"the table's unit column says {sorted(stated)}, not {unit!r}"
                )
        expected = (
            np.arange(unique[0], unique[-1] + 1, dtype=np.int64)
            if years is None
            else years
        )
        values = np.full(expected.shape[0], np.nan)
        inside = np.isin(row_years, expected)
        values[np.searchsorted(expected, row_years[inside])] = row_values[inside]
        provenance["table_columns"] = _recorded_columns(rows)
        return cls(years=expected, values=values, unit=unit, provenance=provenance)

    @classmethod
    def from_table(
        cls,
        table: Any,
        *,
        year: str = "year",
        value: str = "value",
        where: Mapping[str, Any] | None = None,
        unit: str,
        years: Sequence[int] | None = None,
        source: Any = None,
    ) -> AnnualSeries:
        """Read one location's series from a per-year table.

        `year` and `value` name the columns. `where` keeps the rows whose
        columns equal the given values, for example `{"station":
        "USW00094728"}`; after it, each year may appear once. A null value
        (a station's incomplete year) is a missing year. `years` states the
        years the series must cover; rows outside them are left out, and
        years without a row are missing. By default the series covers every
        year from the first to the last row, so a gap is a missing year.
        `source` describes the table for the provenance: a name, a
        `SourceProvenance` or a mapping such as the plan's
        `explain(format="json")`. Columns such as `attribution`,
        `archive_licence`, `source_url` or `station` are recorded too. If the
        table has a `unit` column, it must equal `unit`.

        `table` is an in-memory Arrow table, record batches (for example
        NEX-GDDP-CMIP6's `annual_extremes()` iterator) or any table Arrow
        reads. Raises `SampleError` for missing columns, unmatched filters,
        repeated years or a unit mismatch.
        """
        data = _arrow_table(table, "an annual table")
        rows = data
        filter_document: dict[str, Any] | None = None
        if where is not None:
            if not isinstance(where, Mapping) or not where:
                raise SampleError("where must be a non-empty mapping of column values")
            mask: Any = None
            for name, wanted in where.items():
                column = _column(data, name, "the annual table")
                if isinstance(wanted, float) and not math.isfinite(wanted):
                    raise SampleError(f"where[{name!r}] must be finite")
                if not isinstance(wanted, (str, int, float, bool)):
                    raise SampleError(
                        f"where[{name!r}] must be a string, number or bool"
                    )
                try:
                    matches = pc.equal(column, pa.scalar(wanted))
                except pa.ArrowException as error:
                    raise SampleError(
                        f"where[{name!r}] cannot be compared with column {name!r}"
                    ) from error
                mask = matches if mask is None else pc.and_(mask, matches)
            rows = data.filter(pc.fill_null(mask, False))
            filter_document = dict(where)
        if rows.num_rows == 0:
            raise SampleError(f"no rows of the annual table match where={where!r}")
        provenance: dict[str, Any] = {
            "kind": "table",
            "source": _source_document(source),
            "columns": {"year": year, "value": value},
            "where": filter_document,
        }
        return cls._from_rows(
            rows,
            year=year,
            value=value,
            unit=unit,
            years=_requested_years(years),
            provenance=provenance,
        )

    @classmethod
    def nearest_cell(
        cls,
        table: Any,
        *,
        longitude: float,
        latitude: float,
        unit: str,
        year: str = "year",
        value: str = "value",
        longitude_column: str = "longitude",
        latitude_column: str = "latitude",
        spacing: tuple[float, float] | None = None,
        years: Sequence[int] | None = None,
        source: Any = None,
    ) -> AnnualSeries:
        """Read the series of the grid cell whose centre is nearest a point.

        For per-cell tables such as ERA5's or CMIP6's `annual_extremes()`
        (`year`, `latitude`, `longitude`, `value`). The cell is accepted only
        if its centre lies within half a grid step of the point in latitude
        and in longitude; otherwise this raises. `spacing` is the grid step
        `(latitude, longitude)` in degrees; by default the smallest step
        between the table's distinct latitudes and longitudes. Longitudes are
        compared modulo 360. The cell centre, the distance in km and the
        step are recorded, not the point itself. By default the series must
        cover every year the table holds: a year the cell lacks is missing.
        `years`, `source` and the other arguments are as in `from_table`.

        Raises `SampleError` if no cell centre is close enough, two are
        equally near, or the spacing cannot be inferred.
        """
        lat0 = _coordinate("latitude", latitude, 90.0)
        lon0 = _coordinate("longitude", longitude, 360.0)
        data = _arrow_table(table, "an annual table")
        what = "the annual table"
        lat = _coordinates(_column(data, latitude_column, what), latitude_column)
        lon = _coordinates(_column(data, longitude_column, what), longitude_column)
        if lat.size == 0:
            raise SampleError("the annual table has no rows")
        if spacing is None:
            step_lat = _smallest_step(lat, latitude_column)
            step_lon = _smallest_step(lon, longitude_column)
        else:
            if not isinstance(spacing, Sequence) or len(spacing) != 2:
                raise SampleError("spacing must be a (latitude, longitude) pair")
            step_lat = _positive_step("spacing[0]", spacing[0])
            step_lon = _positive_step("spacing[1]", spacing[1])
        centres = np.unique(np.stack([lat, lon], axis=1), axis=0)
        d_lat = centres[:, 0] - lat0
        d_lon = (centres[:, 1] - lon0 + 180.0) % 360.0 - 180.0
        slack = 1.0 + 1e-9
        inside = (np.abs(d_lat) <= 0.5 * step_lat * slack) & (
            np.abs(d_lon) <= 0.5 * step_lon * slack
        )
        distance = _great_circle_km(lat0, lon0, centres[:, 0], centres[:, 1])
        if not inside.any():
            nearest = int(np.argmin(distance))
            raise SampleError(
                f"no grid cell centre lies within half a grid step "
                f"({step_lat:g}, {step_lon:g} degrees) of the point; the nearest, "
                f"({centres[nearest, 0]:g}, {centres[nearest, 1]:g}), is "
                f"{distance[nearest]:.1f} km away"
            )
        candidates = np.flatnonzero(inside)
        best = candidates[np.argsort(distance[candidates], kind="stable")]
        if best.size > 1 and math.isclose(
            distance[best[0]], distance[best[1]], rel_tol=1e-12, abs_tol=1e-9
        ):
            raise SampleError(
                "the point is equally near two grid cell centres; choose the cell "
                "with from_table(where=...)"
            )
        cell_lat, cell_lon = (float(v) for v in centres[best[0]])
        rows = data.filter(
            pc.and_(
                pc.equal(data.column(latitude_column), cell_lat),
                pc.equal(data.column(longitude_column), cell_lon),
            )
        )
        requested = _requested_years(years)
        if requested is None:
            all_years = _integer_values(_column(data, year, what), year)
            requested = np.unique(all_years)
        provenance: dict[str, Any] = {
            "kind": "nearest_cell",
            "source": _source_document(source),
            "columns": {
                "year": year,
                "value": value,
                "latitude": latitude_column,
                "longitude": longitude_column,
            },
            "cell": {"latitude": cell_lat, "longitude": cell_lon},
            "distance_km": float(distance[best[0]]),
            "spacing_degrees": {"latitude": step_lat, "longitude": step_lon},
        }
        return cls._from_rows(
            rows,
            year=year,
            value=value,
            unit=unit,
            years=requested,
            provenance=provenance,
        )


def _coordinate(name: str, value: Any, bound: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SampleError(f"{name} must be a number, got {value!r}")
    number = float(value)
    low = -90.0 if bound == 90.0 else -180.0
    if not math.isfinite(number) or not low <= number <= bound:
        raise SampleError(f"{name} must be within [{low:g}, {bound:g}], got {value!r}")
    return number


def _coordinates(column: Any, name: str) -> FloatArray:
    if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
        raise SampleError(f"column {name!r} must be numeric, not {column.type}")
    if column.null_count:
        raise SampleError(f"column {name!r} has {column.null_count} null values")
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
    if not np.isfinite(values).all():
        raise SampleError(f"column {name!r} must be finite")
    return values


def _smallest_step(values: FloatArray, name: str) -> float:
    distinct = np.unique(values)
    if distinct.size < 2:
        raise SampleError(
            f"the table has one distinct {name}, so its grid step cannot be "
            "inferred; pass spacing=(latitude_step, longitude_step)"
        )
    return float(np.diff(distinct).min())


def _positive_step(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SampleError(f"{name} must be a positive number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SampleError(f"{name} must be a positive number, got {value!r}")
    return number


def _great_circle_km(
    lat0: float, lon0: float, lat: FloatArray, lon: FloatArray
) -> FloatArray:
    phi0, phi = np.radians(lat0), np.radians(lat)
    d_phi = phi - phi0
    d_lambda = np.radians(lon - lon0)
    half = (
        np.sin(d_phi / 2) ** 2 + np.cos(phi0) * np.cos(phi) * np.sin(d_lambda / 2) ** 2
    )
    distance: FloatArray = (
        2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(half, 0, 1)))
    )
    return distance


def _series_identity(series: AnnualSeries) -> dict[str, Any]:
    """What identifies a series' source: its provenance, else its data."""
    identity: dict[str, Any] = {"unit": series.unit, **_thaw(series.provenance)}
    if not identity.get("source"):
        digest = hashlib.sha256(
            series.years.astype("<i8").tobytes() + series.values.astype("<f8").tobytes()
        ).hexdigest()
        identity["data"] = f"sha256:{digest}"
    return identity


def _series_licence(role: str, series: AnnualSeries) -> dict[str, Any]:
    provenance = _thaw(series.provenance)
    source = provenance.get("source") or {}
    entry = _licence_entry(role, source if isinstance(source, Mapping) else {})
    recorded = provenance.get("table_columns") or {}

    def joined(*names: str) -> str | None:
        for name in names:
            values = (recorded.get(name) or {}).get("values")
            if values:
                return "; ".join(values)
        return None

    entry["licence"] = entry["licence"] or joined("licence", "archive_licence")
    entry["attribution"] = entry["attribution"] or joined("attribution")
    return entry


def sample_annual(
    hazard: AnnualSeries,
    *,
    loss_model: LossModel,
    basis_risk: BasisRisk,
    asset_id: str,
    index: AnnualSeries | None = None,
    rng: np.random.Generator | None = None,
    missing: MissingYears = "raise",
) -> OutcomeSample:
    """Build a sample whose rows are the years of a per-year series.

    `hazard` is the series the loss model reads. `index`, if given, is the
    series the cover pays on: another cell, station or hazard than the loss
    reads, so the index and the loss differ by the data's own dependence.
    Its years must equal the hazard's. Without it the index is the hazard.

    Each year is one row with probability `1/n`, and `period` is the year.
    A year without a value (in either series) raises, unless
    `missing="drop_year"`; dropped years are listed in the provenance. No
    trend is removed. `rng` is needed only by a basis-risk model that draws,
    such as `ImpactNoise`.

    Raises `SampleError` for a hazard unit the loss model's impact is not
    calibrated for (checked before any draw; the `index` series is not
    checked, as the loss does not read it), missing years, mismatched years
    or invalid arguments, and `TypeError` for wrong types. Warns with
    `ComonotoneIndexWarning` for `NoBasisRisk()` without an `index` series.
    """
    if not isinstance(hazard, AnnualSeries):
        raise TypeError(f"hazard must be an AnnualSeries, got {type(hazard).__name__}")
    if index is not None and not isinstance(index, AnnualSeries):
        raise TypeError(f"index must be an AnnualSeries, got {type(index).__name__}")
    asset = _identifier("asset_id", asset_id)
    _check_models(loss_model, basis_risk, asset)
    # The loss reads the hazard series; the index series may be in another unit.
    loss_model.check_hazard_unit(hazard.unit)
    if rng is not None:
        rng = _require_generator(rng)
    if missing not in get_args(MissingYears):
        raise SampleError(
            f"missing must be one of {get_args(MissingYears)}, got {missing!r}"
        )
    years = hazard.years
    if index is not None and not np.array_equal(index.years, years):
        only_hazard = np.setdiff1d(years, index.years)[:10].tolist()
        only_index = np.setdiff1d(index.years, years)[:10].tolist()
        raise SampleError(
            "the index series' years differ from the hazard's; only in the hazard: "
            f"{only_hazard}, only in the index: {only_index}"
        )
    hazard_missing = np.isnan(hazard.values)
    index_missing = np.isnan(index.values) if index is not None else hazard_missing
    gaps = hazard_missing | index_missing
    dropped = [int(year) for year in years[gaps]]
    if dropped and missing == "raise":
        which = []
        if hazard_missing.any():
            which.append(f"hazard {[int(y) for y in years[hazard_missing]][:10]}")
        if index is not None and index_missing.any():
            which.append(f"index {[int(y) for y in years[index_missing]][:10]}")
        raise SampleError(
            f"{len(dropped)} years have no value ({'; '.join(which)}); pass "
            "missing='drop_year' to drop them, listed in the provenance"
        )
    keep = ~gaps
    if not keep.any():
        raise SampleError("no year has a value")
    kept_years = years[keep]
    hazard_values = np.array(hazard.values[keep])
    index_values = (
        np.array(index.values[keep]) if index is not None else hazard_values.copy()
    )
    n = kept_years.shape[0]

    impact = loss_model.impact_values(hazard_values)
    rng_records = [] if rng is None else [_generator_record(rng, "basis_risk")]
    new_index, impact, basis_summary = _apply_basis_risk(
        basis_risk, index_values, hazard_values, impact, rng
    )
    loss = loss_model.loss_values(impact)

    notes = ["The years are used as recorded; no trend is removed."]
    if dropped:
        noun, verb = ("year", "was") if len(dropped) == 1 else ("years", "were")
        notes.append(
            f"{len(dropped)} {noun} without a value {verb} dropped: "
            f"{', '.join(str(year) for year in dropped[:10])}"
            f"{', ...' if len(dropped) > 10 else ''}."
        )
    notes += _basis_risk_notes(basis_risk, separate_index=index is not None)
    sources: list[dict[str, Any]] = [
        {"role": "hazard", "unit": hazard.unit, "series": _thaw(hazard.provenance)}
    ]
    licences = [_series_licence("hazard", hazard)]
    if index is not None:
        sources.append(
            {"role": "index", "unit": index.unit, "series": _thaw(index.provenance)}
        )
        licences.append(_series_licence("index", index))
    provenance = SampleProvenance(
        source_kind="annual",
        sources=tuple(sources),
        semantics_source="data",
        loss_model=loss_model._document(asset),
        basis_risk=basis_risk.to_document(),
        rng=tuple(rng_records),
        licences=tuple(licences),
        details={
            "years": [int(kept_years[0]), int(kept_years[-1])],
            "n_years": n,
            "missing": missing,
            "dropped_years": dropped,
            "index_from": "series" if index is not None else "hazard",
            "basis_risk": basis_summary,
        },
        notes=tuple(notes),
        lineage=content_hash(
            {
                "kind": "annual",
                "asset_id": asset,
                "hazard": _series_identity(hazard),
                "index": None if index is None else _series_identity(index),
            }
        ),
    )
    sample = OutcomeSample(
        asset_ids=(asset,),
        period=kept_years,
        probability=np.full(n, 1.0 / n),
        index=new_index,
        loss=loss,
        hazard=None if np.array_equal(new_index, hazard_values) else hazard_values,
        index_unit=hazard.unit if index is None else index.unit,
        loss_unit=loss_model.value_unit,
        semantics="annual",
        provenance=provenance,
    )
    _warn_comonotone(basis_risk, separate_index=index is not None)
    return sample
