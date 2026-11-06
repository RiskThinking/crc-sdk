"""The public names of crc_sdk.parametric and what importing it loads."""

from __future__ import annotations

import subprocess
import sys

import crc_sdk.parametric as parametric

PUBLIC = [
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


def test_public_names() -> None:
    assert parametric.__all__ == PUBLIC
    for name in PUBLIC:
        assert getattr(parametric, name).__module__.startswith("crc_sdk.parametric")


def test_importing_the_module_loads_no_solver() -> None:
    # The base install has no solver; the module must import without one.
    code = (
        "import sys, crc_sdk.parametric; "
        "print(sorted(m for m in ('highspy', 'scipy', 'cvxpy') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
