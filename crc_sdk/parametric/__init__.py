"""Parametric insurance covers on Climate Risk Commons hazards.

A `Cover` pays on an index, an `IndexDefinition`, through a payout function:
`LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`, `BinaryPayout`, or a
user family that implements `PayoutFunction` and is registered with
`register_payout`. Each payout parameter is a float or `Free`, a value a
design chooses within bounds. Covers round-trip through canonical JSON and
carry a `definition_hash`, as `ImpactExpression` does.

The module works on numpy arrays and does no I/O. See `docs/parametric.md`.
"""

from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import (
    CoverDefinitionError,
    ParametricError,
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

__all__ = [
    "AtAsset",
    "AtCell",
    "AtPoint",
    "AtStation",
    "BinaryPayout",
    "Cover",
    "CoverDefinitionError",
    "Free",
    "IndexDefinition",
    "LinearConstraints",
    "LinearInAmounts",
    "LinearPayout",
    "ParametricError",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "SampleError",
    "StepPayout",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
    "register_payout",
]
