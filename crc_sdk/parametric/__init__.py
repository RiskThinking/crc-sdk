"""Parametric insurance covers on Climate Risk Commons hazards.

A `Cover` pays on an index, an `IndexDefinition`, through a payout function:
`LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`, `BinaryPayout`, or a
user family that implements `PayoutFunction` and is registered with
`register_payout`. Each payout parameter is a float or `Free`, a value a
design chooses within bounds. Covers round-trip through canonical JSON and
carry a `definition_hash`, as `ImpactExpression` does.

Covers are designed and evaluated on an `OutcomeSample`, one row per year
with an index value and a loss: `sample_curve` draws simulated years from a
fitted curve, `sample_annual` reads the years of an `AnnualSeries`, and
`OutcomeSample.from_table` takes a user table. `LossModel` turns the hazard
into a loss and a `BasisRisk` model states how index and loss differ.

`design` chooses a template's free parameters on an outcome sample for an
objective (`MSE`, `ExpectedUncovered`, `WeightedCVaR` or a `UserObjective`)
under a `Budget` and a `Limit`; its LP and QP routes need the optional extra
`crc-sdk[parametric]`. `evaluate` and `compare` assess covers and designs.

`ParametricPlan` chains these steps as a lazy plan, to a written
comparison: `ParametricPlan.for_evaluation(request)` from the fitted hazard
curves of a `PortfolioEvaluation` (with `.monte_carlo(...)` and, for the
out-of-sample test, `FreshDraws`), or `ParametricPlan.for_annual_extremes(...)`
from an annual-extremes source and assets. Its builders and `explain()` do no
I/O.

The functions work on numpy arrays and Arrow tables in memory and do no I/O,
except `write_comparison`, which writes one Parquet file atomically. The
plan reads its source and assets only at its execution points (`samples`,
`designs`, `write_parquet`), and `write_parquet` writes atomically too. See
`docs/parametric.md`.
"""

from crc_sdk.parametric.basis_risk import BasisRisk, ImpactNoise, NoBasisRisk
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.design import DesignOptions, DesignResult, design
from crc_sdk.parametric.errors import (
    ComonotoneIndexWarning,
    CoverDefinitionError,
    DesignError,
    EvaluationError,
    InfeasibleDesignError,
    LeakageError,
    ParametricError,
    ProbabilitySemanticsError,
    SampleError,
    SolverUnavailableError,
    UnboundedDesignError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from crc_sdk.parametric.evaluate import compare, evaluate, to_long, write_comparison
from crc_sdk.parametric.index import (
    AtAsset,
    AtCell,
    AtPoint,
    AtStation,
    IndexDefinition,
)
from crc_sdk.parametric.loss import LossModel
from crc_sdk.parametric.metadata import EvaluationMetadata, setup_text
from crc_sdk.parametric.metrics import MetricOptions
from crc_sdk.parametric.objectives import (
    MSE,
    ExpectedUncovered,
    UserObjective,
    WeightedCVaR,
)
from crc_sdk.parametric.parameters import Free
from crc_sdk.parametric.payouts import (
    BinaryPayout,
    LinearConstraints,
    LinearInAmounts,
    LinearPayout,
    PayoutFunction,
    PiecewiseLinearPayout,
    StepPayout,
    register_payout,
)
from crc_sdk.parametric.plan import (
    FreshDraws,
    Holdout,
    KFold,
    NearestCell,
    ParametricPlan,
    ParametricResult,
    StationColumn,
)
from crc_sdk.parametric.pricing import CostOfCapitalLoad, PurePremium
from crc_sdk.parametric.sample import (
    OutcomeSample,
    SampleProvenance,
    holdout,
    year_block_kfold,
)
from crc_sdk.parametric.sources import AnnualSeries, sample_annual, sample_curve

__all__ = [
    "AnnualSeries",
    "AtAsset",
    "AtCell",
    "AtPoint",
    "AtStation",
    "BasisRisk",
    "BinaryPayout",
    "Budget",
    "ComonotoneIndexWarning",
    "CostOfCapitalLoad",
    "Cover",
    "CoverDefinitionError",
    "DesignError",
    "DesignOptions",
    "DesignResult",
    "EvaluationError",
    "EvaluationMetadata",
    "ExpectedUncovered",
    "Free",
    "FreshDraws",
    "Holdout",
    "ImpactNoise",
    "IndexDefinition",
    "InfeasibleDesignError",
    "KFold",
    "LeakageError",
    "Limit",
    "LinearConstraints",
    "LinearInAmounts",
    "LinearPayout",
    "LossModel",
    "MSE",
    "MetricOptions",
    "NearestCell",
    "NoBasisRisk",
    "OutcomeSample",
    "ParametricError",
    "ParametricPlan",
    "ParametricResult",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "ProbabilitySemanticsError",
    "PurePremium",
    "SampleError",
    "SampleProvenance",
    "SolverUnavailableError",
    "StationColumn",
    "StepPayout",
    "UnboundedDesignError",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
    "UserObjective",
    "WeightedCVaR",
    "compare",
    "design",
    "evaluate",
    "holdout",
    "register_payout",
    "sample_annual",
    "sample_curve",
    "setup_text",
    "to_long",
    "write_comparison",
    "year_block_kfold",
]
