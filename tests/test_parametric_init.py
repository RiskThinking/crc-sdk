"""The public names of crc_sdk.parametric and what importing it loads."""

from __future__ import annotations

import subprocess
import sys

import crc_sdk.parametric as parametric

PUBLIC = [
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
