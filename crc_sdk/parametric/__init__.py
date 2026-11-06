"""Parametric insurance covers on Climate Risk Commons hazards.

A `Cover` pays on an index, an `IndexDefinition`, through a payout function:
`LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`, `BinaryPayout`, or a
user family that implements `PayoutFunction` and is registered with
`register_payout`. Each payout parameter is a float or `Free`, a value a
design chooses within bounds. Covers round-trip through canonical JSON and
carry a `definition_hash`, as `ImpactExpression` does.

`design` chooses a template's free parameters on an outcome sample for an
objective (`MSE`, `ExpectedUncovered`, `WeightedCVaR` or a `UserObjective`)
under a `Budget` and a `Limit`; its LP and QP routes need the optional extra
`crc-sdk[parametric]`. `evaluate` and `compare` assess covers and designs.

The module works on numpy arrays and does no I/O. See `docs/parametric.md`.
"""

from crc_sdk.parametric.basis_risk import BasisRisk, ImpactNoise, NoBasisRisk
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.design import DesignOptions, DesignResult, design
from crc_sdk.parametric.errors import (
    ComonotoneIndexWarning,
    CoverDefinitionError,
    DesignError,
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
from crc_sdk.parametric.metadata import EvaluationMetadata
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
    "EvaluationMetadata",
    "ExpectedUncovered",
    "Free",
    "ImpactNoise",
    "IndexDefinition",
    "InfeasibleDesignError",
    "LeakageError",
    "Limit",
    "LinearConstraints",
    "LinearInAmounts",
    "LinearPayout",
    "LossModel",
    "MSE",
    "MetricOptions",
    "NoBasisRisk",
    "OutcomeSample",
    "ParametricError",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "ProbabilitySemanticsError",
    "PurePremium",
    "SampleError",
    "SampleProvenance",
    "SolverUnavailableError",
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
    "to_long",
    "write_comparison",
    "year_block_kfold",
]
