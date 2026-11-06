"""The parametric plan: from hazard curves or per-year values to compared covers.

A plan is lazy and immutable, as every crc-sdk source is. Builders return a
new plan and do no I/O, and neither does `explain()`. It has two entry
points:

```text
ParametricPlan.for_evaluation(request, value_column=, value_unit=)   # fitted curves
  .monte_carlo(n_draws, seed=)          # required there: simulated years
ParametricPlan.for_annual_extremes(source, assets, impact=, unit=, provenance=)
# then, on either plan:
  .basis_risk(model)                    # required: how index and loss differ
  .covers(*covers)                      # resolved covers and templates
  .objective(*objectives, pricing=, budget=, limit=, options=)
  .split(FreshDraws(n, seed=) | Holdout(0.7) | KFold(5))   # optional
  .metrics(MetricOptions(...))          # optional: levels and thresholds
  .explain()                            # the setup, no I/O
  .samples()                            # execution: one sample per asset
  .designs()                            # execution: the designed covers
  .write_parquet("comparison.parquet")  # execution: written atomically
```

**Fitted curves** (`for_evaluation`). `request` is the `PortfolioEvaluation`
built with `HazardDataset.local(path).for_assets(assets).select(...)
.impact(...)`; the plan reuses its dataset, assets, selection and impact.
At execution, `request.asset_curves()` gives each asset's canonical curve
row (one hazard, horizon and pathway per asset), and `sample_curve` draws
`n_draws` simulated years from it, each with probability 1/n. The
probability-semantics rules of `sample_curve` apply: a curve whose
`probability_semantics` is unspecified is refused unless `.monte_carlo(...,
assume_semantics=...)` states one, and draws beyond the fitted support and a
pooled ensemble are noted in each sample. The impact gets the asset's
matched H3 cell and its context columns, as `request.write_parquet` gives
them.

**Per-year values** (`for_annual_extremes`). `source` is a crc-sdk plan with
an `annual_extremes()` execution point (ERA5 and CMIP6 canonicalization
plans, NEX-GDDP-CMIP6, station plans, NOAA water levels), an Arrow table in
one of their layouts, or a local Parquet file of one. Each asset's series is
the grid cell whose centre is nearest the asset (`NearestCell`, the default,
for per-cell tables), or the station an asset column names
(`StationColumn`). `unit` is the unit of the table's values and
`provenance` (a `SourceProvenance`) the source's licence, attribution and
version: annual-extremes tables carry no licence of their own. Each asset
gets one `OutcomeSample` from `sample_annual`: its years are the rows, each
with probability 1/n, and no trend is removed.

On both routes the loss is `impact(hazard) × value`, with the value from an
asset column or 1 (the damage ratio), and the basis-risk model is required:
`NoBasisRisk()` must be written out.

**Seeding.** The plan holds an integer `seed` (`monte_carlo`'s, or
`for_annual_extremes`'). At execution it builds
`numpy.random.SeedSequence(seed)` and spawns one child generator per asset,
in asset order; each draws the hazard uniforms first, then the basis-risk
model. Fresh draws use `SeedSequence(FreshDraws.seed)` the same way. On the
per-year route only the basis-risk model draws, so `ImpactNoise` needs a
seed there.

**Covers and designs.** A cover written for no asset (`asset_id=None`)
applies to every asset, as a copy written for that asset; one written for an
asset applies to that asset only. Templates, covers with `Free` parameters,
are designed under each objective of `.objective(...)` on each design block;
resolved covers are evaluated as they are. A designed cover's label is the
template's label (or the first 12 hex digits of its hash) followed by the
objective, for example `"step (mse)"`.

**Splits.** Without `.split(...)`, every cover is designed and evaluated on
the whole sample, so every row is in sample. `FreshDraws(n_draws, seed=)`
(curves only) designs on the plan's draws and evaluates on `n_draws` fresh
draws from the same curve. `Holdout(train_fraction)` designs on the first
periods (years or draws) and evaluates on the rest; `KFold(k)` holds out
each of `k` contiguous blocks in turn. Every block gets in-sample rows (on
the design block) and out-of-sample rows (on the held-out block or the fresh
draws). Resolved covers are evaluated on the same blocks, with the design
block as their design sample, so their terms are taken as chosen without
the held-out data.

**Output.** `write_parquet(path)` writes one comparison table, as `compare`
builds it, through a temporary sibling and `os.replace`. Its metadata under
`crc.parametric.evaluation` (`EvaluationMetadata`) holds the setup of every
number: the samples with their source, loss and basis-risk models, the
splits, the pricing, the designs and the covers, plus the source's record
(the canonical file's path and SHA-256, or the annual table's checksum and
the source plan's `explain()`), the licences, the generator states, and the
plan's seed in every evaluation's `details` (`"plan_seed"`, with
`"fresh_draws_seed"` on fresh-draw rows). Asset values are never written;
the metadata holds a SHA-256 of the (asset id, value) pairs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from crc_framework import (
    CallableImpact,
    ClimateImpact,
    ImpactFunction,
    TransformContext,
    impact_registry_manifest,
)
from crc_framework import ImpactExpression as FrameworkImpactExpression

from crc_sdk.impacts.calibration import units_equivalent
from crc_sdk.impacts.expression import ImpactExpression
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.basis_risk import BasisRisk, ImpactNoise
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.design import DesignOptions, DesignResult, design
from crc_sdk.parametric.errors import ProbabilitySemanticsError, SampleError
from crc_sdk.parametric.evaluate import compare, write_comparison
from crc_sdk.parametric.index import AtAsset
from crc_sdk.parametric.loss import LossModel
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY, EvaluationMetadata
from crc_sdk.parametric.metrics import MetricOptions
from crc_sdk.parametric.objectives import (
    MSE,
    ExpectedUncovered,
    Objective,
    UserObjective,
    WeightedCVaR,
)
from crc_sdk.parametric.pricing import CostOfCapitalLoad, Pricing, PurePremium
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _arrow_table,
    _identifier,
    _string_values,
    holdout,
    year_block_kfold,
)
from crc_sdk.parametric.sources import (
    AnnualSemantics,
    AnnualSeries,
    MissingYears,
    OutsideSupport,
    sample_annual,
    sample_curve,
)
from crc_sdk.types import SourceProvenance
from crc_sdk.workflows.distributions import CURVE_COLUMNS
from crc_sdk.workflows.portfolio import (
    AssetPortfolio,
    CellColumn,
    ImpactContextColumns,
    ImpactSpec,
    PointColumns,
    PortfolioEvaluation,
    _ExpressionImpact,
)

__all__ = [
    "FreshDraws",
    "Holdout",
    "KFold",
    "NearestCell",
    "ParametricPlan",
    "ParametricResult",
    "StationColumn",
]

_EXECUTION = (
    "the source is read, the samples are built and the covers are designed and "
    "evaluated only at samples(), designs() and write_parquet()"
)


# -- value objects -----------------------------------------------------------


@dataclass(frozen=True)
class Holdout:
    """Design on the first share of the years; evaluate on the rest.

    The design block is the first `floor(train_fraction * n)` years of each
    asset's sample (`holdout`), so the held-out block holds its latest
    years. Raises `ValueError` unless `train_fraction` lies in (0, 1).
    """

    train_fraction: float = 0.7

    def __post_init__(self) -> None:
        value = self.train_fraction
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
        ):
            raise ValueError(
                f"train_fraction must be a number in (0, 1), got {value!r}"
            )
        object.__setattr__(self, "train_fraction", float(value))

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "holdout", "train_fraction": f}`."""
        return {"kind": "holdout", "train_fraction": self.train_fraction}


@dataclass(frozen=True)
class KFold:
    """Hold out each of `k` contiguous blocks of years in turn.

    Fold `i` designs on the other blocks and evaluates on block `i`
    (`year_block_kfold`), so every year is held out once. Raises
    `ValueError` unless `k` is an integer of at least 2; a sample with fewer
    than `k` years raises `SampleError` at execution.
    """

    k: int

    def __post_init__(self) -> None:
        value = self.k
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"k must be an integer of at least 2, got {value!r}")
        if int(value) < 2:
            raise ValueError(f"k must be an integer of at least 2, got {value!r}")
        object.__setattr__(self, "k", int(value))

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "kfold", "k": k}`."""
        return {"kind": "kfold", "k": self.k}


def _positive_count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _seed_value(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class FreshDraws:
    """Evaluate on fresh Monte Carlo draws from each asset's curve.

    For plans on fitted curves (`for_evaluation`). The designs are made on
    the plan's draws (`monte_carlo`), and every cover is evaluated out of
    sample on `n_draws` fresh draws from the same curve, drawn with one
    child generator of `numpy.random.SeedSequence(seed)` per asset, in asset
    order. Draws are independent simulated years, so fresh draws are the
    plain out-of-sample test. `seed` must differ from the plan's seed; it is
    recorded in every fresh-draw evaluation's `details` as
    `"fresh_draws_seed"`. Raises `ValueError` unless `n_draws` is a positive
    integer and `seed` a non-negative one.
    """

    n_draws: int
    seed: int = field(kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_draws", _positive_count("n_draws", self.n_draws))
        object.__setattr__(self, "seed", _seed_value("seed", self.seed))

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "fresh_draws", "n_draws": n, "seed": seed}`."""
        return {"kind": "fresh_draws", "n_draws": self.n_draws, "seed": self.seed}


@dataclass(frozen=True)
class _MonteCarlo:
    """What `.monte_carlo(...)` states: the draws per asset and the curve rules."""

    n_draws: int
    assume_semantics: AnnualSemantics | None
    outside_support: OutsideSupport

    def to_document(self) -> dict[str, Any]:
        return {
            "assume_semantics": self.assume_semantics,
            "n_draws": self.n_draws,
            "outside_support": self.outside_support,
        }


def _column_name(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty column name, got {value!r}")
    return value


@dataclass(frozen=True)
class NearestCell:
    """Read each asset's series from the grid cell whose centre is nearest it.

    For per-cell annual tables such as ERA5's and CMIP6's `annual_extremes()`
    (`year`, `latitude`, `longitude`, `value`). The assets need points
    (`PointColumns`). The cell is accepted only if its centre lies within
    half a grid step of the asset in latitude and in longitude, as
    `AnnualSeries.nearest_cell` decides; otherwise the plan raises at
    execution. `longitude` and `latitude` name the table's coordinate
    columns; `spacing` is the grid step `(latitude, longitude)` in degrees,
    by default the smallest step between the table's coordinates.
    """

    longitude: str = "longitude"
    latitude: str = "latitude"
    spacing: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _column_name("longitude", self.longitude)
        _column_name("latitude", self.latitude)
        if self.spacing is not None:
            spacing = self.spacing
            if (
                isinstance(spacing, (str, bytes))
                or not isinstance(spacing, Sequence)
                or len(spacing) != 2
                or not all(
                    isinstance(step, Real)
                    and not isinstance(step, bool)
                    and math.isfinite(float(step))
                    and float(step) > 0.0
                    for step in spacing
                )
            ):
                raise ValueError(
                    "spacing must be a (latitude, longitude) pair of positive "
                    f"degrees, got {spacing!r}"
                )
            object.__setattr__(self, "spacing", (float(spacing[0]), float(spacing[1])))

    def to_document(self) -> dict[str, Any]:
        """Return the locator's kind, columns and spacing."""
        return {
            "kind": "nearest_cell",
            "latitude_column": self.latitude,
            "longitude_column": self.longitude,
            "spacing": None if self.spacing is None else list(self.spacing),
        }


@dataclass(frozen=True)
class StationColumn:
    """Read each asset's series from the station an asset column names.

    For per-station annual tables such as the station plans'
    `annual_extremes()` (`station`, `year`, ..., `value`). `name` is the
    asset column holding each asset's station id, and `table_column` the
    table column it is matched against (`"station"`; NOAA water levels use
    `"station_id"`).
    """

    name: str
    table_column: str = "station"

    def __post_init__(self) -> None:
        _column_name("name", self.name)
        _column_name("table_column", self.table_column)

    def to_document(self) -> dict[str, Any]:
        """Return the locator's kind and columns."""
        return {
            "kind": "station",
            "asset_column": self.name,
            "table_column": self.table_column,
        }


@dataclass(frozen=True)
class _DesignSpec:
    """What `.objective(...)` states: the objectives and the design terms."""

    objectives: tuple[Objective, ...]
    pricing: Pricing
    budget: Budget | None
    limit: Limit | None
    options: DesignOptions

    def to_document(self) -> dict[str, Any]:
        return {
            "budget": None if self.budget is None else self.budget.to_document(),
            "limit": None if self.limit is None else self.limit.to_document(),
            "objectives": [objective.to_document() for objective in self.objectives],
            "options": self.options.to_document(),
            "pricing": self.pricing.to_document(),
        }


@dataclass(frozen=True, eq=False)
class ParametricResult:
    """What `ParametricPlan.write_parquet` wrote, and what it was built from.

    - `output`: the Parquet file;
    - `table`: the comparison table as written, with its metadata;
    - `metadata`: its `EvaluationMetadata`, under `crc.parametric.evaluation`;
    - `samples`: one `OutcomeSample` per asset, in asset order (all years,
      before any split);
    - `designs`: every `DesignResult`, in asset, fold, cover and objective
      order.
    """

    output: Path
    table: Any
    metadata: EvaluationMetadata
    samples: Mapping[str, OutcomeSample]
    designs: tuple[DesignResult, ...]

    @property
    def row_count(self) -> int:
        """The number of rows written."""
        return int(self.table.num_rows)


@dataclass(frozen=True, eq=False)
class _Run:
    samples: Mapping[str, OutcomeSample]
    designs: tuple[DesignResult, ...]
    tables: tuple[Any, ...]
    source_key: str
    source: dict[str, Any]
    id_column: str
    values_sha256: str | None


@dataclass(frozen=True, eq=False)
class _Samples:
    """What execution builds before any design: the samples and the source."""

    portfolio: AssetPortfolio
    samples: dict[str, OutcomeSample]
    fresh: dict[str, OutcomeSample] | None
    record_key: str
    record: dict[str, Any]


# -- helpers -----------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Return `value` as plain JSON types; unknown objects become their text."""
    return json.loads(json.dumps(value, default=str))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _table_sha256(table: Any) -> str:
    """SHA-256 of the table's Arrow IPC stream, in one chunk, without metadata."""
    data = table.replace_schema_metadata(None).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, data.schema) as writer:
        writer.write_table(data)
    return f"sha256:{hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()}"


def _source_kind(source: Any) -> Literal["table", "file", "plan"]:
    if isinstance(source, (pa.Table, pa.RecordBatch)):
        return "table"
    if isinstance(source, (str, os.PathLike)):
        return "file"
    if callable(getattr(source, "annual_extremes", None)):
        return "plan"
    raise TypeError(
        "source must be a plan with an annual_extremes() execution point, a "
        "pyarrow.Table in an annual-extremes layout, or a local Parquet file; "
        f"got {type(source).__name__}"
    )


def _assets(assets: Any) -> AssetPortfolio | Path:
    """Return the assets as an `AssetPortfolio`, or a file read at execution."""
    if isinstance(assets, AssetPortfolio):
        return assets
    if isinstance(assets, (str, os.PathLike)):
        return Path(assets)
    return AssetPortfolio(assets)


def _table_columns(assets: AssetPortfolio | Path) -> tuple[str, ...] | None:
    """The asset table's columns, when it is in memory; else `None`."""
    if isinstance(assets, AssetPortfolio) and isinstance(assets.data, pa.Table):
        return tuple(assets.data.column_names)
    return None


def _registry_needs_cell(
    impact: Any, context: Any, assets: AssetPortfolio | Path
) -> str | None:
    """The registry factor that needs an H3 cell the plan cannot give, if any.

    `LossModel.for_asset` fills the cell from an asset's `CellColumn` when
    context columns are given; otherwise only a `TransformContext` gives one.
    """
    if not isinstance(impact, ClimateImpact):
        return None
    if (
        isinstance(assets, AssetPortfolio)
        and isinstance(assets.location, CellColumn)
        and isinstance(context, ImpactContextColumns)
        and context.items()
    ):
        return None
    factor = str(getattr(impact, "factor", ""))
    entry = next(
        (
            item
            for item in impact_registry_manifest()["factors"]
            if item["factor"] == factor
        ),
        None,
    )
    if entry is None or "cell" not in tuple(entry.get("required_context") or ()):
        return None
    for given in (getattr(impact, "context", None), context):
        if isinstance(given, TransformContext) and given.cell is not None:
            return None
    return factor


def _listed(values: Sequence[Any] | None) -> list[Any] | None:
    return None if values is None else list(values)


def _objective_tag(objective: Objective) -> str:
    if isinstance(objective, WeightedCVaR):
        return f"weighted_cvar alpha={objective.alpha:g} beta={objective.beta:g}"
    if isinstance(objective, UserObjective):
        return f"user {objective.name}"
    return str(objective.to_document()["kind"])


def _bound(cover: Cover, asset_id: str, label: str | None = None) -> Cover:
    """`cover` written for `asset_id`, with `label` if given."""
    name = cover.label if label is None else label
    if cover.asset_id == asset_id and name == cover.label:
        return cover
    return Cover(
        cover.index,
        cover.payout,
        payout_unit=cover.payout_unit,
        asset_id=asset_id,
        label=name,
    )


def _design_label(template: Cover, objective: Objective) -> str:
    base = template.label or template.definition_hash.removeprefix("sha256:")[:12]
    return f"{base} ({_objective_tag(objective)})"


_CONTEXT_FIELDS = ("cell", "country", "continent", "building_type", "historic_mean")
_CURVE_ROW_COLUMNS = (
    "cell_index",
    "hazard_name",
    "horizon",
    "pathway",
    "source_id",
    *CURVE_COLUMNS,
)
"""The columns of an `asset_curves()` row that make its canonical curve row."""


def _request_impact(spec: ImpactSpec) -> tuple[Any, ImpactContextColumns | None]:
    """The impact a request's `ImpactSpec` holds, as given, and its context.

    An expression comes back as the `ImpactExpression` itself, so its
    `definition_hash` is kept, and a callable as the callable.
    """
    function: Any = spec.function
    if isinstance(function, _ExpressionImpact):
        function = function.expression
    elif isinstance(function, CallableImpact):
        function = function.function
    context = spec.context_columns if spec.context_columns.items() else None
    return function, context


def _takes_context(impact: Any) -> bool:
    """Whether `LossModel` passes a framework context to `impact`."""
    return not isinstance(
        impact, (ImpactExpression, FrameworkImpactExpression)
    ) and isinstance(impact, ImpactFunction)


def _curve_context(
    impact: Any,
    curve: Mapping[str, Any],
    asset: Mapping[str, Any],
    columns: ImpactContextColumns | None,
) -> TransformContext:
    """The context `PortfolioEvaluation` gives an impact for one matched row.

    The impact's own context, with the matched H3 cell and the asset's
    context columns on top; null asset values keep the impact's own.
    """
    base = getattr(impact, "context", None)
    fields: dict[str, Any] = {
        name: getattr(base, name) if isinstance(base, TransformContext) else None
        for name in _CONTEXT_FIELDS
    }
    if curve.get("cell_index") is not None:
        fields["cell"] = int(curve["cell_index"])
    for field_name, column_name in columns.items() if columns is not None else ():
        if column_name not in asset:
            raise SampleError(f"the asset table has no column {column_name!r}")
        value = asset[column_name]
        if value is None:
            continue
        fields[field_name] = (
            float(value) if field_name == "historic_mean" else str(value)
        )
    return TransformContext(**fields)


# -- the plan ----------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class ParametricPlan:
    """A lazy plan from hazard curves or per-year values to compared covers.

    Start one with `ParametricPlan.for_evaluation(request)` (fitted curves)
    or `ParametricPlan.for_annual_extremes(...)` (per-year values); the
    builders (`monte_carlo`, `basis_risk`, `covers`, `objective`, `split`,
    `metrics`) return new plans, and neither they nor `explain()` do any
    I/O. The execution points `samples()`, `designs()` and
    `write_parquet(path)` read the source and the assets. The module
    docstring describes both routes.

    `entry` names the entry point. On the curve route `source` is the
    request, `unit` and `provenance` are `None` (the dataset's metadata
    states them at execution) and `draws` holds what `monte_carlo` states.
    """

    source: Any
    assets: AssetPortfolio | Path
    loss_model: LossModel
    unit: str | None
    provenance: SourceProvenance | None
    location: NearestCell | StationColumn = field(default_factory=NearestCell)
    year_column: str = "year"
    hazard_column: str = "value"
    years: tuple[int, ...] | None = None
    missing: MissingYears = "raise"
    seed: int | None = None
    basis_risk_model: BasisRisk | None = None
    cover_list: tuple[Cover, ...] = ()
    design_spec: _DesignSpec | None = None
    split_rule: Holdout | KFold | FreshDraws | None = None
    metric_options: MetricOptions = field(default_factory=MetricOptions)
    entry: Literal["for_annual_extremes", "for_evaluation"] = "for_annual_extremes"
    draws: _MonteCarlo | None = None

    # -- the entry points -----------------------------------------------------

    @classmethod
    def for_evaluation(
        cls,
        request: PortfolioEvaluation,
        *,
        value_column: str | None = None,
        value_unit: str = "1",
    ) -> ParametricPlan:
        """Plan covers on each asset's fitted hazard curve.

        `request` is the `PortfolioEvaluation` built with
        `HazardDataset.local(path).for_assets(assets).select(...)
        .impact(...)`; the plan reuses its dataset, assets, selection and
        impact, and changes nothing in it. Each asset needs one curve: the
        selection must leave one hazard, horizon and pathway. At execution,
        `request.asset_curves()` gives the asset's canonical curve row and
        `sample_curve` draws the simulated years that `.monte_carlo(n_draws,
        seed=...)` states, so `.monte_carlo(...)` is required.

        - The loss is `impact(hazard) × value`: the impact is the request's
          (an `ImpactExpression` keeps its `definition_hash`), the value the
          asset column `value_column` in `value_unit`, or 1 (the damage
          ratio, `value_unit="1"`). A framework `ImpactFunction` gets the
          context `request.write_parquet` gives it: its own, with the asset's
          matched H3 cell and the request's context columns on top.
        - The index is the curve's hazard at the asset, in the dataset's
          `value_unit`, so every cover's index must be in that unit
          (checked at execution).
        - The assets are an in-memory table or a local Parquet file (read at
          execution); the asset order is the table's row order.

        Raises `TypeError` for a request of another type (for a remote
        source, materialize it first and build the request on the
        `HazardDataset` it returns) or for SQL assets; `ValueError` for a
        request without an impact or a selection of several hazards,
        horizons or pathways; and `SampleError` for a missing asset column
        or an invalid loss model.
        """
        if not isinstance(request, PortfolioEvaluation):
            raise TypeError(
                "request must be a PortfolioEvaluation, as HazardDataset.local("
                "path).for_assets(assets) builds; for a remote source, "
                "materialize it first (plan.materialize(path) returns the "
                f"HazardDataset), got {type(request).__name__}"
            )
        spec = request.impact_spec
        if spec is None:
            raise ValueError(
                "the request has no impact; add .impact(function, name=..., "
                "value_unit=..., value_semantics=...) to state the loss"
            )
        selection = request.selection
        for name, values in (
            ("hazard_names", selection.hazard_names),
            ("horizons", selection.horizons),
            ("pathways", selection.pathways),
        ):
            if values is not None and len(values) > 1:
                raise ValueError(
                    f"the plan samples one curve per asset, but the request "
                    f"selects {len(values)} {name}; select one with "
                    f".select({name}=[...])"
                )
        portfolio = request.portfolio
        if isinstance(portfolio.data, str):
            raise TypeError(
                "the plan reads each asset's row at execution; pass the assets "
                "as an Arrow table or a Parquet file path, not SQL"
            )
        impact, context = _request_impact(spec)
        loss_model = LossModel(
            impact,
            value_column=value_column,
            value_unit=value_unit,
            context=context,
            name=spec.name,
        )
        columns = _table_columns(portfolio)
        if columns is not None:
            needed = [value_column] if value_column is not None else []
            if context is not None:
                needed += [name for _, name in context.items()]
            absent = [name for name in needed if name not in columns]
            if absent:
                raise SampleError(f"the asset table has no columns {absent}")
        return cls(
            source=request,
            assets=portfolio,
            loss_model=loss_model,
            unit=None,
            provenance=None,
            entry="for_evaluation",
        )

    @classmethod
    def for_annual_extremes(
        cls,
        source: Any,
        assets: Any,
        *,
        impact: Any,
        unit: str,
        provenance: SourceProvenance,
        value_column: str | None = None,
        value_unit: str = "1",
        context: ImpactContextColumns | TransformContext | None = None,
        location: NearestCell | StationColumn = NearestCell(),
        year_column: str = "year",
        hazard_column: str = "value",
        years: Sequence[int] | None = None,
        missing: MissingYears = "raise",
        seed: int | None = None,
    ) -> ParametricPlan:
        """Plan covers on per-year hazard values at each asset.

        `source` is a plan with an `annual_extremes()` execution point (for
        example `HazardDataset.era5(...)...canonicalize()`), an Arrow table
        in an annual-extremes layout, or a local Parquet file of one; it is
        read only at an execution point. `assets` is an `AssetPortfolio`, an
        in-memory table `AssetPortfolio` accepts, or a local Parquet file
        read at execution; the asset order is the table's row order.

        - `impact`, `value_column`, `value_unit` and `context` make the loss
          model, `LossModel(impact, value_column=..., value_unit=...,
          context=...)`: loss = `impact(hazard) × value`, with the value from
          the asset column, or 1 (the damage ratio, `value_unit="1"`).
        - `unit` is the unit of the table's values. It is checked against
          the impact's calibration here, against a `unit` column of the table
          and against the source plan's `explain()` at execution.
        - `provenance` states the source's licence, attribution and version
          (annual-extremes tables carry none); it identifies the series in
          every sample, so one source has one `provenance`.
        - `location` finds each asset's series: `NearestCell()` (the
          default) or `StationColumn(name)`. `year_column` and
          `hazard_column` name the table's year and value columns.
        - `years` states the years every series must cover (by default the
          years the table holds); `missing="drop_year"` drops a year without
          a value and lists it, instead of raising.
        - `seed` seeds the basis-risk draws: one child of
          `numpy.random.SeedSequence(seed)` per asset, in asset order.

        Raises `TypeError` for arguments of the wrong type, `SampleError`
        for a unit the impact is not calibrated for, assets that
        `NearestCell` cannot locate or a missing asset column, and
        `ValueError` for other invalid arguments. A registry impact that
        needs an H3 cell in its context (such as `rx1day`) raises
        `SampleError` unless the impact or `context` is a `TransformContext`
        with a cell: this route reads no cell for an asset.
        """
        _source_kind(source)
        portfolio = _assets(assets)
        unit = _identifier("unit", unit)
        if not isinstance(provenance, SourceProvenance):
            raise TypeError(
                "provenance must be a SourceProvenance stating the source's "
                f"licence and attribution, got {type(provenance).__name__}"
            )
        loss_model = LossModel(
            impact, value_column=value_column, value_unit=value_unit, context=context
        )
        loss_model.check_hazard_unit(unit)
        if not isinstance(location, (NearestCell, StationColumn)):
            raise TypeError(
                "location must be NearestCell() or StationColumn(name), got "
                f"{type(location).__name__}"
            )
        if (
            isinstance(location, NearestCell)
            and isinstance(portfolio, AssetPortfolio)
            and not isinstance(portfolio.location, PointColumns)
        ):
            raise SampleError(
                "NearestCell needs each asset's longitude and latitude; locate "
                "the assets with PointColumns, or name their stations with "
                "StationColumn"
            )
        columns = _table_columns(portfolio)
        if columns is not None:
            needed = [value_column] if value_column is not None else []
            if isinstance(location, StationColumn):
                needed.append(location.name)
            if isinstance(context, ImpactContextColumns):
                needed += [name for _, name in context.items()]
            absent = [name for name in needed if name not in columns]
            if absent:
                raise SampleError(f"the asset table has no columns {absent}")
        factor = _registry_needs_cell(impact, context, portfolio)
        if factor is not None:
            raise SampleError(
                f"impact {factor!r} needs an H3 cell in its context, and this "
                "route reads none for an asset; give the impact a context with "
                "the cell (crc_framework.impacts.for_factor(..., context="
                "TransformContext(cell=...))), or pass context= a "
                "TransformContext with it"
            )
        _column_name("year_column", year_column)
        _column_name("hazard_column", hazard_column)
        window: tuple[int, ...] | None = None
        if years is not None:
            if isinstance(years, (str, bytes)) or not isinstance(years, Sequence):
                raise TypeError("years must be a sequence of integer years")
            if not years or not all(
                isinstance(year, (int, np.integer)) and not isinstance(year, bool)
                for year in years
            ):
                raise ValueError("years must be a non-empty sequence of integer years")
            window = tuple(sorted(int(year) for year in years))
            if len(set(window)) != len(window):
                raise ValueError("years must not repeat")
        if missing not in get_args(MissingYears):
            raise ValueError(
                f"missing must be one of {get_args(MissingYears)}, got {missing!r}"
            )
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
        ):
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        return cls(
            source=source,
            assets=portfolio,
            loss_model=loss_model,
            unit=unit,
            provenance=provenance,
            location=location,
            year_column=year_column,
            hazard_column=hazard_column,
            years=window,
            missing=missing,
            seed=None if seed is None else int(seed),
        )

    # -- builders ---------------------------------------------------------------

    @property
    def _on_curves(self) -> bool:
        return self.entry == "for_evaluation"

    def monte_carlo(
        self,
        n_draws: int,
        *,
        seed: int,
        assume_semantics: AnnualSemantics | None = None,
        outside_support: OutsideSupport = "raise",
    ) -> ParametricPlan:
        """State the simulated years: `n_draws` draws from each asset's curve.

        Required on plans on fitted curves (`for_evaluation`). Each draw is
        one year with probability `1/n_draws`, as `sample_curve` draws it.
        `seed` is the plan's seed: one child of
        `numpy.random.SeedSequence(seed)` per asset, in asset order, draws
        the hazard and then the basis-risk model. It is recorded in every
        evaluation's `details` as `"plan_seed"`.

        A draw is a year only if the curve describes one year's value, so
        the dataset's `probability_semantics` must be `annual_exceedance` or
        `annual_value_distribution`. A dataset that declares none is refused
        at execution unless `assume_semantics` states one of the two; a
        stated value that contradicts the declared one raises.
        `outside_support` is what a tabulated curve does with draws outside
        its knots: `"raise"` (the default), `"extrapolate"` or `"clamp"`
        (`sample_curve`). Replaces any draws stated before.

        Raises `ValueError` on a plan on per-year values, for invalid counts,
        seeds or `outside_support`, or a seed equal to the fresh draws';
        `ProbabilitySemanticsError` for another `assume_semantics`.
        """
        if not self._on_curves:
            raise ValueError(
                "monte_carlo draws simulated years from fitted curves; this plan "
                "reads per-year values (for_annual_extremes)"
            )
        count = _positive_count("n_draws", n_draws)
        plan_seed = _seed_value("seed", seed)
        if assume_semantics is not None and assume_semantics not in get_args(
            AnnualSemantics
        ):
            raise ProbabilitySemanticsError(
                f"assume_semantics must be one of {get_args(AnnualSemantics)}, "
                f"got {assume_semantics!r}"
            )
        if outside_support not in get_args(OutsideSupport):
            raise ValueError(
                f"outside_support must be one of {get_args(OutsideSupport)}, "
                f"got {outside_support!r}"
            )
        rule = self.split_rule
        if isinstance(rule, FreshDraws) and rule.seed == plan_seed:
            raise ValueError(
                f"seed {plan_seed} is also the fresh draws' seed; fresh draws need "
                "another seed, or they repeat the design draws"
            )
        draws = _MonteCarlo(
            n_draws=count,
            assume_semantics=assume_semantics,
            outside_support=outside_support,
        )
        return dataclasses.replace(self, seed=plan_seed, draws=draws)

    def basis_risk(self, model: BasisRisk) -> ParametricPlan:
        """State the basis-risk model every sample is built with.

        `NoBasisRisk()` (the index is the hazard at the asset; samples warn
        with `ComonotoneIndexWarning`), `ImpactNoise(sigma, upper=...)` or a
        user model that implements `BasisRisk`. Required before execution.
        Raises `TypeError` for another type, and `ValueError` for
        `ImpactNoise` on a plan on per-year values without a seed (on fitted
        curves, `monte_carlo` states the seed).
        """
        if not isinstance(model, BasisRisk):
            raise TypeError(
                "model must be a BasisRisk model such as NoBasisRisk() or "
                f"ImpactNoise(sigma), got {type(model).__name__}"
            )
        if isinstance(model, ImpactNoise) and self.seed is None and not self._on_curves:
            raise ValueError(
                "ImpactNoise draws one factor per year; give the plan a seed "
                "(for_annual_extremes(..., seed=...))"
            )
        return dataclasses.replace(self, basis_risk_model=model)

    def covers(self, *covers: Cover) -> ParametricPlan:
        """Set the covers to compare: resolved covers and templates.

        A template (a cover with `Free` parameters) is designed under each
        objective of `.objective(...)`; a resolved cover is evaluated as it
        is. Each cover pays in the loss unit on an index read at the asset
        (`AtAsset()`), in the table's unit on per-year values; on fitted
        curves the index unit is checked against the dataset's `value_unit`
        at execution. Replaces any covers set before.

        Raises `TypeError` for a non-cover, `SampleError` for a cover in
        other units or on an index read elsewhere, and `ValueError` for no
        covers or the same cover twice.
        """
        if not covers:
            raise ValueError("covers needs at least one cover")
        seen: set[tuple[str, str | None]] = set()
        for cover in covers:
            if not isinstance(cover, Cover):
                raise TypeError(f"expected a Cover, got {type(cover).__name__}")
            if cover.payout_unit != self.loss_model.value_unit:
                raise SampleError(
                    f"cover {cover.label or cover.definition_hash!r} pays in "
                    f"{cover.payout_unit!r}, but the loss is in "
                    f"{self.loss_model.value_unit!r}"
                )
            if self.unit is not None and cover.index.unit != self.unit:
                raise SampleError(
                    f"cover {cover.label or cover.definition_hash!r} pays on an "
                    f"index in {cover.index.unit!r}, but the table's values are "
                    f"in {self.unit!r}"
                )
            if not isinstance(cover.index.location, AtAsset):
                raise SampleError(
                    "the plan reads the index at each asset (AtAsset()); for an "
                    "index read elsewhere, build the samples with "
                    "sample_annual(..., index=series)"
                )
            key = (cover.definition_hash, cover.asset_id)
            if key in seen:
                raise ValueError(
                    f"cover {cover.definition_hash} is given twice; each cover "
                    "is compared once"
                )
            seen.add(key)
        return dataclasses.replace(self, cover_list=tuple(covers))

    def objective(
        self,
        *objectives: Objective,
        pricing: Pricing = PurePremium(),
        budget: Budget | None = None,
        limit: Limit | None = None,
        options: DesignOptions = DesignOptions(),
    ) -> ParametricPlan:
        """Design the templates: under each objective, with these terms.

        `objectives` are `MSE()`, `ExpectedUncovered()`,
        `WeightedCVaR(alpha, beta)` or `UserObjective`s; each template is
        designed under each, on each design block. `pricing` prices the
        premium of the designs on their design block, and of the resolved
        covers on each evaluation block; `budget`, `limit` and `options` are
        those of `design`. Without `.objective(...)` resolved covers are
        priced with `PurePremium()` and templates cannot be evaluated.
        Replaces any objective set before.

        Raises `TypeError` for arguments of the wrong type, and `ValueError`
        for no objective or the same objective twice.
        """
        if not objectives:
            raise ValueError("objective needs at least one objective")
        kinds = (MSE, ExpectedUncovered, WeightedCVaR, UserObjective)
        documents: list[str] = []
        for objective in objectives:
            if not isinstance(objective, kinds):
                raise TypeError(
                    "objectives must be MSE(), ExpectedUncovered(), "
                    "WeightedCVaR(alpha, beta) or UserObjective, got "
                    f"{type(objective).__name__}"
                )
            text = json.dumps(objective.to_document(), sort_keys=True)
            if text in documents:
                raise ValueError(f"objective {text} is given twice")
            documents.append(text)
        if not isinstance(pricing, (PurePremium, CostOfCapitalLoad)):
            raise TypeError(
                "pricing must be PurePremium() or CostOfCapitalLoad(rate, q), "
                f"got {type(pricing).__name__}"
            )
        if budget is not None and not isinstance(budget, Budget):
            raise TypeError(f"budget must be a Budget, got {type(budget).__name__}")
        if limit is not None and not isinstance(limit, Limit):
            raise TypeError(f"limit must be a Limit, got {type(limit).__name__}")
        if not isinstance(options, DesignOptions):
            raise TypeError(
                f"options must be DesignOptions, got {type(options).__name__}"
            )
        spec = _DesignSpec(
            objectives=tuple(objectives),
            pricing=pricing,
            budget=budget,
            limit=limit,
            options=options,
        )
        return dataclasses.replace(self, design_spec=spec)

    def split(self, rule: Holdout | KFold | FreshDraws) -> ParametricPlan:
        """Hold out data: `FreshDraws(n_draws, seed=)`, `Holdout(f)` or `KFold(k)`.

        Every design block gets in-sample rows and every held-out block (or
        the fresh draws) out-of-sample rows. `FreshDraws` needs a plan on
        fitted curves; `Holdout` and `KFold` split per-year samples by year
        and Monte Carlo samples by draw. Raises `TypeError` for another rule
        or fresh draws on per-year values, and `ValueError` for fresh draws
        with the plan's seed.
        """
        if not isinstance(rule, (Holdout, KFold, FreshDraws)):
            raise TypeError(
                "rule must be FreshDraws(n_draws, seed=...), Holdout(train_fraction) "
                f"or KFold(k), got {type(rule).__name__}"
            )
        if isinstance(rule, FreshDraws):
            if not self._on_curves:
                raise TypeError(
                    "fresh draws need a Monte Carlo sample from a curve "
                    "(for_evaluation); split per-year values with Holdout or KFold"
                )
            if rule.seed == self.seed:
                raise ValueError(
                    f"seed {rule.seed} is also the plan's seed; fresh draws need "
                    "another seed, or they repeat the design draws"
                )
        return dataclasses.replace(self, split_rule=rule)

    def metrics(self, options: MetricOptions) -> ParametricPlan:
        """Set the metrics' levels and thresholds (`MetricOptions`).

        Raises `TypeError` for another type.
        """
        if not isinstance(options, MetricOptions):
            raise TypeError(
                f"options must be MetricOptions, got {type(options).__name__}"
            )
        return dataclasses.replace(self, metric_options=options)

    # -- explain ----------------------------------------------------------------

    def _pricing(self) -> Pricing:
        return PurePremium() if self.design_spec is None else self.design_spec.pricing

    def _known_asset_ids(self) -> list[str] | None:
        assets = self.assets
        if not isinstance(assets, AssetPortfolio) or not isinstance(
            assets.data, pa.Table
        ):
            return None
        if assets.id_column not in assets.data.column_names:
            return None
        column = assets.data.column(assets.id_column)
        if column.null_count:
            return None
        return [str(value) for value in column.to_pylist()]

    def _seed_details(self) -> dict[str, Any]:
        """`{"plan_seed": seed}` for a plan with a seed, else `{}`."""
        return {} if self.seed is None else {"plan_seed": self.seed}

    def _evaluations(self) -> list[dict[str, Any]]:
        pricing = self._pricing().to_document()
        rule = self.split_rule
        if rule is None:
            folds, method, details = 1, None, {}
        elif isinstance(rule, Holdout):
            folds, method = 1, "holdout"
            details = {"train_fraction": rule.train_fraction}
        elif isinstance(rule, FreshDraws):
            folds, method = 1, "fresh_draws"
            details = {"fresh_draws_seed": rule.seed}
        else:
            folds, method, details = rule.k, "kfold", {"k": rule.k}
        evaluations: list[dict[str, Any]] = []
        for fold in range(folds):
            evaluations.append(
                {
                    "details": self._seed_details(),
                    "fold": fold,
                    "method": "in_sample",
                    "pricing": pricing,
                    "split": "in_sample",
                }
            )
            if method is not None:
                evaluations.append(
                    {
                        "details": {**details, **self._seed_details()},
                        "fold": fold,
                        "method": method,
                        "pricing": pricing,
                        "split": "out_of_sample",
                    }
                )
        return evaluations

    def _source_description(self) -> dict[str, Any]:
        if self._on_curves:
            request: PortfolioEvaluation = self.source
            selection = request.selection
            spec = request.impact_spec
            assert spec is not None
            return {
                "impact": {
                    "name": spec.name,
                    "value_semantics": spec.value_semantics,
                    "value_unit": spec.value_unit,
                },
                "join": "PortfolioEvaluation.asset_curves()",
                "kind": "canonical_dataset",
                "metadata": None,
                "path": os.fspath(request.dataset.provider.source),
                "selection": {
                    "hazard_names": _listed(selection.hazard_names),
                    "horizons": _listed(selection.horizons),
                    "pathways": _listed(selection.pathways),
                },
                "type": type(request.dataset).__name__,
            }
        kind = _source_kind(self.source)
        assert self.provenance is not None
        description: dict[str, Any] = {
            "columns": {"value": self.hazard_column, "year": self.year_column},
            "kind": kind,
            "location": self.location.to_document(),
            "missing": self.missing,
            "provenance": self.provenance.model_dump(mode="json"),
            "type": type(self.source).__name__,
            "unit": self.unit,
            "years": None if self.years is None else list(self.years),
        }
        if kind == "file":
            description["path"] = os.fspath(self.source)
        elif kind == "table":
            description["rows"] = int(self.source.num_rows)
        return description

    def _sample_description(self) -> dict[str, Any]:
        """One asset's sample, as far as it is known before execution."""
        sample: dict[str, Any] = {
            "index_unit": self.unit,
            "loss_model": self.loss_model.to_document(),
            "loss_unit": self.loss_model.value_unit,
            "notes": [],
            "scenario": "baseline",
            "semantics": "annual",
            "semantics_source": "data",
            "source_kind": "annual",
        }
        if self._on_curves:
            draws = self.draws
            assumed = None if draws is None else draws.assume_semantics
            sample.update(
                {
                    "draws": None if draws is None else draws.n_draws,
                    "probability_semantics": assumed,
                    "semantics_source": "dataset" if assumed is None else "caller",
                    "source_kind": "monte_carlo",
                }
            )
        if self.basis_risk_model is not None:
            sample["basis_risk"] = self.basis_risk_model.to_document()
        return sample

    def _rng_description(self) -> dict[str, Any]:
        generators = (
            "one child of numpy.random.SeedSequence(seed) per asset, in asset order"
        )
        if not self._on_curves:
            return {
                "generators": generators,
                "seed": self.seed,
                "steps": ["basis_risk"],
            }
        rule = self.split_rule
        return {
            "fresh_draws_seed": rule.seed if isinstance(rule, FreshDraws) else None,
            "generators": generators
            + "; fresh draws: one child of SeedSequence(fresh_draws_seed) per asset",
            "seed": self.seed,
            "steps": ["hazard", "basis_risk"],
        }

    def _explain_document(self) -> dict[str, Any]:
        ids = self._known_asset_ids()
        sample = self._sample_description()
        samples = (
            [{**sample, "asset_ids": []}]
            if ids is None
            else [{**sample, "asset_ids": [asset_id]} for asset_id in ids]
        )
        spec = self.design_spec
        designs = None
        if spec is not None:
            designs = {
                **spec.to_document(),
                "templates": [
                    cover.definition_hash
                    for cover in self.cover_list
                    if cover.free_parameters()
                ],
            }
        assets = self.assets
        if isinstance(assets, AssetPortfolio):
            location: Any = repr(assets.location)
            id_column: str | None = assets.id_column
        else:
            location, id_column = None, None
        if isinstance(assets, Path):
            path: str | None = os.fspath(assets)
        elif isinstance(assets.data, os.PathLike):
            path = os.fspath(assets.data)
        else:
            path = None
        provenance = self.provenance
        licences: list[dict[str, Any]] | None = (
            None
            if provenance is None
            else [
                {
                    "attribution": provenance.attribution,
                    "licence": provenance.licence,
                    "role": "hazard",
                    "source": provenance.dataset,
                }
            ]
        )
        document: dict[str, Any] = {
            "assets": {
                "count": None if ids is None else len(ids),
                "id_column": id_column,
                "location": location,
                "path": path,
                "value_column": self.loss_model.value_column,
            },
            "covers": [
                {
                    "asset_id": cover.asset_id,
                    "document": cover.to_document(),
                    "free_parameters": list(cover.free_parameters()),
                    "hash": cover.definition_hash,
                    "label": cover.label,
                }
                for cover in self.cover_list
            ],
            "designs": designs,
            "entry": self.entry,
            "execution": _EXECUTION,
            "licences": licences,
            "producer": "crc_sdk.parametric",
            "rng": self._rng_description(),
            "setup": {
                "basis": "annual",
                "evaluations": self._evaluations(),
                "metric_options": self.metric_options.to_document(),
                "samples": samples,
            },
            "sources": [self._source_description()],
        }
        if self._on_curves:
            rule = self.split_rule
            document["monte_carlo"] = {
                **(
                    {"assume_semantics": None, "n_draws": None, "outside_support": None}
                    if self.draws is None
                    else self.draws.to_document()
                ),
                "fresh_draws": (
                    rule.to_document() if isinstance(rule, FreshDraws) else None
                ),
                "seed": self.seed,
            }
        return document

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        """Describe the plan: its setup as far as it is known, without I/O.

        The document mirrors the metadata the plan writes
        (`EvaluationMetadata`): `setup` (the basis, the metric options, one
        sample per asset with its source, loss and basis-risk models, and
        the planned evaluations with their split, fold and pricing),
        `covers`, `designs` (the objectives and design terms), `sources`,
        `assets`, `licences`, `rng` (the seeds) and `entry`; a plan on
        fitted curves adds `monte_carlo` (the draws, the semantics stated by
        the caller, `outside_support` and the fresh draws). Its `setup` has
        the layout of `EvaluationSetup`, so `setup_text(plan.explain(
        format="json")["setup"])` states it before anything is read; the
        plan's seed is in every evaluation's `details` under `"plan_seed"`,
        and the fresh draws' under `"fresh_draws_seed"`. What only execution
        knows is in the written metadata: the years and the matched cells or
        curves, the checksums, the hashes of the samples and designs and, on
        fitted curves, the dataset's metadata (its units, semantics and
        licence, so `licences` is `None` here). `format="json"` returns a
        dict, `"text"` its JSON text.
        """
        document = self._explain_document()
        if format == "json":
            return document
        if format != "text":
            raise ValueError("explain format must be 'text' or 'json'")
        return json.dumps(document, indent=2, sort_keys=True)

    def __repr__(self) -> str:
        ids = self._known_asset_ids()
        assets = "?" if ids is None else str(len(ids))
        objectives = 0 if self.design_spec is None else len(self.design_spec.objectives)
        split = "none" if self.split_rule is None else repr(self.split_rule)
        if self._on_curves:
            source = Path(self.source.dataset.provider.source).name
            draws = "?" if self.draws is None else str(self.draws.n_draws)
            return (
                f"ParametricPlan.for_evaluation({source}, assets={assets}, "
                f"draws={draws}, covers={len(self.cover_list)}, "
                f"objectives={objectives}, split={split})"
            )
        return (
            f"ParametricPlan.for_annual_extremes({type(self.source).__name__}, "
            f"assets={assets}, covers={len(self.cover_list)}, "
            f"objectives={objectives}, split={split})"
        )

    # -- execution --------------------------------------------------------------

    @property
    def _annual_unit(self) -> str:
        """The unit of the annual table's values (per-year route only)."""
        assert self.unit is not None
        return self.unit

    def _require_basis_risk(self) -> BasisRisk:
        if self.basis_risk_model is None:
            raise ValueError(
                "state a basis-risk model with .basis_risk(...); NoBasisRisk() "
                "states that the index is the hazard at the asset"
            )
        return self.basis_risk_model

    def _portfolio(self) -> AssetPortfolio:
        """The assets as an in-memory `AssetPortfolio` (reads a file)."""
        assets = self.assets
        if isinstance(assets, Path):
            return AssetPortfolio(pq.read_table(assets))
        data = assets.data
        if isinstance(data, (str, os.PathLike)):
            table = pq.read_table(Path(data))
        else:
            table = _arrow_table(data, "the asset table")
        return AssetPortfolio(
            table,
            id_column=assets.id_column,
            location=assets.location,
            passthrough_columns=assets.passthrough_columns,
        )

    def _asset_rows(
        self, portfolio: AssetPortfolio
    ) -> list[tuple[str, dict[str, Any]]]:
        table = portfolio.data
        if portfolio.id_column not in table.column_names:
            raise SampleError(
                f"the asset table has no id column {portfolio.id_column!r}"
            )
        ids = _string_values(table.column(portfolio.id_column), portfolio.id_column)
        if not ids:
            raise SampleError("the asset table has no assets")
        repeated = sorted({asset_id for asset_id in ids if ids.count(asset_id) > 1})
        if repeated:
            raise SampleError(f"assets {repeated[:10]} appear more than once")
        return list(zip(ids, table.to_pylist(), strict=True))

    def _read_source(self) -> tuple[Any, dict[str, Any]]:
        """Read the annual table; return it with its record for the metadata."""
        kind = _source_kind(self.source)
        record: dict[str, Any] = {"kind": kind, "type": type(self.source).__name__}
        if kind == "table":
            table = _arrow_table(self.source, "the annual table")
        elif kind == "file":
            path = Path(self.source)
            table = pq.read_table(path)
            record["path"] = os.fspath(path)
            record["file_sha256"] = _file_sha256(path)
        else:
            table = _arrow_table(
                self.source.annual_extremes(), "the source's annual_extremes()"
            )
            explain = getattr(self.source, "explain", None)
            explained = (
                _json_safe(explain(format="json")) if callable(explain) else None
            )
            record["explain"] = explained
            stated = explained.get("unit") if isinstance(explained, dict) else None
            if isinstance(stated, str) and not units_equivalent(
                stated, self._annual_unit
            ):
                raise SampleError(
                    f"the source's values are in {stated!r}, but the plan states "
                    f"unit={self.unit!r}"
                )
        record["columns"] = list(table.column_names)
        record["rows"] = int(table.num_rows)
        record["table_sha256"] = _table_sha256(table)
        return table, record

    def _series_source(self) -> dict[str, Any]:
        """The series' `source`: what identifies the data, stable across runs."""
        assert self.provenance is not None
        document: dict[str, Any] = self.provenance.model_dump(mode="json")
        kind = _source_kind(self.source)
        if kind == "plan":
            document["source_type"] = type(self.source).__name__
        elif kind == "file":
            document["file_sha256"] = _file_sha256(Path(self.source))
        return document

    def _series(
        self,
        table: Any,
        row: Mapping[str, Any],
        asset_id: str,
        portfolio: AssetPortfolio,
        source: Mapping[str, Any],
    ) -> AnnualSeries:
        location = self.location
        if isinstance(location, NearestCell):
            point = portfolio.location
            if not isinstance(point, PointColumns):
                raise SampleError(
                    "NearestCell needs each asset's longitude and latitude; "
                    "locate the assets with PointColumns"
                )
            longitude, latitude = row.get(point.longitude), row.get(point.latitude)
            if longitude is None or latitude is None:
                raise SampleError(f"asset {asset_id!r} has no longitude or latitude")
            return AnnualSeries.nearest_cell(
                table,
                longitude=longitude,
                latitude=latitude,
                unit=self._annual_unit,
                year=self.year_column,
                value=self.hazard_column,
                longitude_column=location.longitude,
                latitude_column=location.latitude,
                spacing=location.spacing,
                years=self.years,
                source=source,
            )
        if location.name not in row:
            raise SampleError(f"the asset table has no column {location.name!r}")
        station = row[location.name]
        if station is None:
            raise SampleError(
                f"asset {asset_id!r} names no station in column {location.name!r}"
            )
        return AnnualSeries.from_table(
            table,
            year=self.year_column,
            value=self.hazard_column,
            where={location.table_column: station},
            unit=self._annual_unit,
            years=self.years,
            source=source,
        )

    def _generators(self, count: int) -> list[np.random.Generator | None]:
        if self.seed is None:
            return [None] * count
        children = np.random.SeedSequence(self.seed).spawn(count)
        return [np.random.default_rng(child) for child in children]

    def _covers_for(self, asset_ids: Sequence[str]) -> dict[str, list[Cover]]:
        known = set(asset_ids)
        unknown = sorted(
            {
                str(cover.asset_id)
                for cover in self.cover_list
                if cover.asset_id is not None and cover.asset_id not in known
            }
        )
        if unknown:
            raise ValueError(
                f"covers are written for assets {unknown}, which the plan's "
                "assets do not hold"
            )
        covers: dict[str, list[Cover]] = {}
        for asset_id in asset_ids:
            chosen = [
                _bound(cover, asset_id)
                for cover in self.cover_list
                if cover.asset_id in (None, asset_id)
            ]
            if not chosen:
                raise ValueError(
                    f"asset {asset_id!r} has no cover; give it one, or leave it "
                    "out of the assets"
                )
            hashes = [cover.definition_hash for cover in chosen]
            repeated = sorted({h for h in hashes if hashes.count(h) > 1})
            if repeated:
                raise ValueError(
                    f"cover {repeated[0]} applies to asset {asset_id!r} twice: once "
                    "written for no asset and once for this one"
                )
            covers[asset_id] = chosen
        return covers

    def _blocks(
        self, sample: OutcomeSample, fresh: OutcomeSample | None
    ) -> list[tuple[int, OutcomeSample, OutcomeSample | None]]:
        rule = self.split_rule
        if rule is None:
            return [(0, sample, None)]
        if isinstance(rule, FreshDraws):
            return [(0, sample, fresh)]
        if isinstance(rule, Holdout):
            train, test = holdout(sample, rule.train_fraction)
            return [(0, train, test)]
        return [
            (fold, train, test)
            for fold, (train, test) in enumerate(year_block_kfold(sample, rule.k))
        ]

    def _build_samples(self, *, fresh: bool = False) -> _Samples:
        """The samples of every asset; with `fresh`, also their fresh draws."""
        if self._on_curves:
            return self._curve_samples(fresh=fresh)
        basis_risk = self._require_basis_risk()
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        table, record = self._read_source()
        source = self._series_source()
        generators = self._generators(len(rows))
        samples: dict[str, OutcomeSample] = {}
        for (asset_id, row), rng in zip(rows, generators, strict=True):
            series = self._series(table, row, asset_id, portfolio, source)
            loss_model = self.loss_model.for_asset(portfolio, asset_id)
            samples[asset_id] = sample_annual(
                series,
                loss_model=loss_model,
                basis_risk=basis_risk,
                asset_id=asset_id,
                rng=rng,
                missing=self.missing,
            )
        return _Samples(portfolio, samples, None, "annual_table", record)

    # -- the curve route ---------------------------------------------------------

    def _require_draws(self) -> _MonteCarlo:
        if self.draws is None:
            raise ValueError(
                "state the simulated years with .monte_carlo(n_draws, seed=...); "
                "the plan draws them from each asset's curve"
            )
        return self.draws

    def _curve_rows(
        self, curves: Any, id_column: str, asset_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        """Each asset's one matched curve row of `asset_curves()`."""
        rows: dict[str, list[dict[str, Any]]] = {}
        ids = _string_values(curves.column(id_column), id_column)
        for asset_id, row in zip(ids, curves.to_pylist(), strict=True):
            rows.setdefault(asset_id, []).append(row)
        matched: dict[str, dict[str, Any]] = {}
        for asset_id in asset_ids:
            found = rows.get(asset_id, [])
            if not found:
                raise SampleError(
                    f"asset {asset_id!r} has no curve in the request's selection"
                )
            if len(found) > 1:
                keys = [
                    (row["hazard_name"], row["horizon"], row["pathway"])
                    for row in found
                ]
                raise SampleError(
                    f"asset {asset_id!r} has {len(found)} curves (hazard, horizon, "
                    f"pathway): {keys}; the plan samples one curve per asset, so "
                    "select one with request.select(hazard_names=[...], "
                    "horizons=[...], pathways=[...])"
                )
            matched[asset_id] = found[0]
        return matched

    def _curve_loss_model(
        self,
        portfolio: AssetPortfolio,
        asset_id: str,
        asset: Mapping[str, Any],
        curve: Mapping[str, Any],
    ) -> LossModel:
        """The loss model for one asset, with the context its matched row gives."""
        model = self.loss_model
        if _takes_context(model.impact):
            context = model.context
            model = LossModel(
                model.impact,
                value_column=model.value_column,
                value_unit=model.value_unit,
                context=_curve_context(
                    model.impact,
                    curve,
                    asset,
                    context if isinstance(context, ImpactContextColumns) else None,
                ),
                name=model.name,
            )
        return model.for_asset(portfolio, asset_id)

    def _curve_samples(self, *, fresh: bool) -> _Samples:
        basis_risk = self._require_basis_risk()
        draws = self._require_draws()
        request: PortfolioEvaluation = self.source
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        metadata = request.dataset.metadata()
        for cover in self.cover_list:
            if cover.index.unit != metadata.value_unit:
                raise SampleError(
                    f"cover {cover.label or cover.definition_hash!r} pays on an "
                    f"index in {cover.index.unit!r}, but the dataset's values are "
                    f"in {metadata.value_unit!r}"
                )
        path = Path(request.dataset.provider.source)
        file_sha256 = _file_sha256(path)
        curves = dataclasses.replace(request, portfolio=portfolio).asset_curves()
        matched = self._curve_rows(
            curves, portfolio.id_column, [asset_id for asset_id, _ in rows]
        )
        rule = self.split_rule if fresh else None
        fresh_seed = rule.seed if isinstance(rule, FreshDraws) else None
        generators = self._generators(len(rows))
        fresh_generators: list[np.random.Generator | None] = (
            [None] * len(rows)
            if fresh_seed is None
            else [
                np.random.default_rng(child)
                for child in np.random.SeedSequence(fresh_seed).spawn(len(rows))
            ]
        )
        samples: dict[str, OutcomeSample] = {}
        fresh_samples: dict[str, OutcomeSample] = {}
        for (asset_id, row), rng, fresh_rng in zip(
            rows, generators, fresh_generators, strict=True
        ):
            curve = matched[asset_id]
            loss_model = self._curve_loss_model(portfolio, asset_id, row, curve)
            canonical = {
                name: curve[name] for name in _CURVE_ROW_COLUMNS if name in curve
            }
            details = {
                "file_sha256": file_sha256,
                "spatial_match": curve["spatial_match"],
            }

            def draw(
                count: int, generator: np.random.Generator | None
            ) -> OutcomeSample:
                assert generator is not None
                return sample_curve(
                    canonical,
                    metadata=metadata,
                    n_draws=count,
                    rng=generator,
                    loss_model=loss_model,
                    basis_risk=basis_risk,
                    asset_id=asset_id,
                    assume_semantics=draws.assume_semantics,
                    outside_support=draws.outside_support,
                    details=details,
                )

            samples[asset_id] = draw(draws.n_draws, rng)
            if isinstance(rule, FreshDraws):
                fresh_samples[asset_id] = draw(rule.n_draws, fresh_rng)
        record = {
            "curves": int(curves.num_rows),
            "file_sha256": file_sha256,
            "kind": "canonical_dataset",
            "path": os.fspath(path),
        }
        return _Samples(
            portfolio,
            samples,
            fresh_samples if isinstance(rule, FreshDraws) else None,
            "canonical_file",
            record,
        )

    def _values_sha256(
        self, portfolio: AssetPortfolio, asset_ids: Sequence[str]
    ) -> str | None:
        column = self.loss_model.value_column
        if column is None:
            return None
        values = dict(
            zip(
                _string_values(
                    portfolio.data.column(portfolio.id_column), portfolio.id_column
                ),
                portfolio.data.column(column).to_pylist(),
                strict=True,
            )
        )
        pairs = sorted(
            ([asset_id, float(values[asset_id])] for asset_id in asset_ids),
            key=lambda pair: str(pair[0]).encode("utf-8"),
        )
        return content_hash({"asset_values": pairs, "value_column": column})

    def _run(self, *, evaluate: bool) -> _Run:
        if not self.cover_list:
            raise ValueError("the plan has no covers; set them with .covers(...)")
        spec = self.design_spec
        templates = [cover for cover in self.cover_list if cover.free_parameters()]
        if templates and spec is None:
            names = ", ".join(
                repr(cover.label or cover.definition_hash) for cover in templates
            )
            raise ValueError(
                f"covers {names} have free parameters; design them with "
                ".objective(...), or resolve them with Cover.with_values"
            )
        built = self._build_samples(fresh=evaluate)
        portfolio, samples = built.portfolio, built.samples
        asset_ids = list(samples)
        covers = self._covers_for(asset_ids)
        pricing = self._pricing()
        designs: list[DesignResult] = []
        tables: list[Any] = []
        for asset_id in asset_ids:
            fresh = None if built.fresh is None else built.fresh[asset_id]
            for fold, train, test in self._blocks(samples[asset_id], fresh):
                items: list[Cover | DesignResult] = []
                for cover in covers[asset_id]:
                    if not cover.free_parameters():
                        items.append(cover)
                        continue
                    assert spec is not None
                    for objective in spec.objectives:
                        result = design(
                            cover,
                            train,
                            objective,
                            pricing=spec.pricing,
                            budget=spec.budget,
                            limit=spec.limit,
                            options=spec.options,
                        )
                        labelled = _bound(
                            result.cover, asset_id, _design_label(cover, objective)
                        )
                        result = dataclasses.replace(result, cover=labelled)
                        designs.append(result)
                        items.append(result)
                if not evaluate:
                    continue
                options = self.metric_options
                tables.append(
                    compare(
                        items,
                        train,
                        split="in_sample",
                        fold=fold,
                        pricing=pricing,
                        options=options,
                        design_sample=train,
                    )
                )
                if test is not None:
                    tables.append(
                        compare(
                            items,
                            test,
                            split="out_of_sample",
                            fold=fold,
                            pricing=pricing,
                            options=options,
                            design_sample=train,
                        )
                    )
        return _Run(
            samples=MappingProxyType(samples),
            designs=tuple(designs),
            tables=tuple(tables),
            source_key=built.record_key,
            source=built.record,
            id_column=portfolio.id_column,
            values_sha256=self._values_sha256(portfolio, asset_ids),
        )

    def samples(self) -> Mapping[str, OutcomeSample]:
        """Execution: build one outcome sample per asset, in asset order.

        Reads the source and the assets. Each sample holds all of the
        asset's draws or years, before any split (fresh draws are drawn by
        `write_parquet()` only). Raises `ValueError` without
        a basis-risk model or, on fitted curves, without `monte_carlo`;
        `SampleError` for an asset that cannot be located, a curve that
        cannot be sampled (`sample_curve`), several curves for one asset, a
        cover whose index unit is not the dataset's, or a series with a
        missing year (unless `missing="drop_year"`); and
        `ProbabilitySemanticsError` for curves whose semantics are not those
        of one year.
        """
        return MappingProxyType(self._build_samples().samples)

    def designs(self) -> tuple[DesignResult, ...]:
        """Execution: design every template, in asset, fold, cover, objective order.

        Each template is designed on each design block (the whole sample
        without a split or with fresh draws; the design block of a holdout;
        each fold's design block) under each objective. Raises `ValueError`
        without covers, an objective or a basis-risk model, the design errors
        of `design`, and `SolverUnavailableError` without
        `crc-sdk[parametric]` on the LP and QP routes.
        """
        if self.design_spec is None:
            raise ValueError(
                "the plan has no objective; state one with .objective(...)"
            )
        return self._run(evaluate=False).designs

    def write_parquet(self, path: str | os.PathLike[str]) -> ParametricResult:
        """Execution: design, evaluate and compare; write the comparison.

        Builds the samples, designs the templates, evaluates every cover and
        design on each block (`compare`) and writes the rows of every asset,
        split and fold to one local Parquet file, through a temporary
        sibling and `os.replace`; the parent directory is created. The
        metadata (`EvaluationMetadata`, under `crc.parametric.evaluation`)
        merges that of every block, and adds to each hazard source the
        source's record: on fitted curves `canonical_file` (the dataset's
        path, its SHA-256 and the number of matched curves); on per-year
        values `annual_table` (its SHA-256, the source plan's `explain()`
        and, for a file, its path and SHA-256). It adds the plan's seed to
        every evaluation's `details` (`"plan_seed"`) and the fresh draws'
        seed to every fresh-draw evaluation's (`"fresh_draws_seed"`), and
        the asset id column and a SHA-256 of the (asset id, value) pairs to
        `assets`.

        Returns a `ParametricResult`. Raises `ValueError` without covers, a
        basis-risk model, or an objective for templates; and as `samples`,
        `design` and `compare` raise.
        """
        run = self._run(evaluate=True)
        merged = EvaluationMetadata.merge(
            *(EvaluationMetadata.from_table(table) for table in run.tables)
        )
        document = merged.model_dump(mode="json")
        document["sources"] = [
            {**entry, run.source_key: run.source}
            if entry.get("role") == "hazard"
            else entry
            for entry in document["sources"]
        ]
        seeded = self._seed_details()
        rule = self.split_rule
        fresh = {"fresh_draws_seed": rule.seed} if isinstance(rule, FreshDraws) else {}
        document["setup"]["evaluations"] = [
            {
                **entry,
                "details": {
                    **entry.get("details", {}),
                    **(fresh if entry.get("method") == "fresh_draws" else {}),
                    **seeded,
                },
            }
            for entry in document["setup"]["evaluations"]
        ]
        document["assets"] = {
            "count": len(run.samples),
            "id_column": run.id_column,
            "values_sha256": run.values_sha256,
        }
        metadata = EvaluationMetadata.model_validate(document)
        table = pa.concat_tables(run.tables)
        existing = dict(table.schema.metadata or {})
        existing[EVALUATION_METADATA_KEY.encode("utf-8")] = metadata.to_json().encode(
            "utf-8"
        )
        table = table.replace_schema_metadata(existing)
        text = os.fspath(path)
        if not isinstance(text, str) or "://" in text:
            raise ValueError(f"write_parquet writes local files, got {path!r}")
        target = Path(text)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_comparison(table, target, metadata=metadata)
        return ParametricResult(
            output=target,
            table=table,
            metadata=metadata,
            samples=run.samples,
            designs=run.designs,
        )
