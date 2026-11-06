"""Parametric insurance covers on Climate Risk Commons hazards.

A `Cover` pays on an index, an `IndexDefinition`, through a payout function:
`LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`, `BinaryPayout`, or a
user family that implements `PayoutFunction` and is registered with
`register_payout`. Each payout parameter is a float or `Free`, a value a
design chooses within bounds. Covers round-trip through canonical JSON and
carry a `definition_hash`, as `ImpactExpression` does.

The module works on numpy arrays and does no I/O. See `docs/parametric.md`.
"""

from crc_sdk.parametric.basis_risk import BasisRisk, ImpactNoise, NoBasisRisk
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import (
    ComonotoneIndexWarning,
    CoverDefinitionError,
    LeakageError,
    ParametricError,
    ProbabilitySemanticsError,
    SampleError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from crc_sdk.parametric.index import (
    AtAsset,
    AtCell,
    AtPoint,
    AtStation,
    IndexDefinition,
)
from crc_sdk.parametric.loss import LossModel
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
    "ComonotoneIndexWarning",
    "Cover",
    "CoverDefinitionError",
    "Free",
    "ImpactNoise",
    "IndexDefinition",
    "LeakageError",
    "LinearConstraints",
    "LinearInAmounts",
    "LinearPayout",
    "LossModel",
    "NoBasisRisk",
    "OutcomeSample",
    "ParametricError",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "ProbabilitySemanticsError",
    "SampleError",
    "SampleProvenance",
    "StepPayout",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
    "holdout",
    "register_payout",
    "sample_annual",
    "sample_curve",
    "year_block_kfold",
]
