"""The parametric module's errors and warnings."""

from __future__ import annotations

import pytest

from crc_sdk.parametric import errors
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


@pytest.mark.parametrize(
    ("error", "bases"),
    [
        (CoverDefinitionError, (ParametricError, ValueError)),
        (UnknownPayoutKindError, (CoverDefinitionError, ValueError)),
        (UnresolvedParameterError, (ParametricError, ValueError)),
        (SampleError, (ParametricError, ValueError)),
        (ProbabilitySemanticsError, (SampleError, ValueError)),
        (LeakageError, (SampleError, ValueError)),
        (EvaluationError, (ParametricError, ValueError)),
        (DesignError, (ParametricError, RuntimeError)),
        (InfeasibleDesignError, (DesignError, RuntimeError)),
        (UnboundedDesignError, (DesignError, RuntimeError)),
        (SolverUnavailableError, (ParametricError, ImportError)),
    ],
)
def test_error_hierarchy(error: type[Exception], bases: tuple[type, ...]) -> None:
    for base in bases:
        assert issubclass(error, base)


def test_design_errors_carry_the_asset_and_the_solver_status() -> None:
    error = InfeasibleDesignError(
        "no feasible payout",
        asset_id="warehouse-a",
        status="infeasible",
        solver_message="primal infeasible",
    )
    assert str(error) == "no feasible payout"
    assert (error.asset_id, error.status, error.solver_message) == (
        "warehouse-a",
        "infeasible",
        "primal infeasible",
    )
    bare = DesignError("failed")
    assert (bare.asset_id, bare.status, bare.solver_message) == (None, None, None)


def test_solver_unavailable_names_the_extra() -> None:
    error = SolverUnavailableError("install crc-sdk[parametric]")
    assert error.extra == "parametric"
    with pytest.raises(ImportError, match="parametric"):
        raise error


def test_the_comonotone_warning_is_a_user_warning() -> None:
    assert issubclass(ComonotoneIndexWarning, UserWarning)
    assert not issubclass(ComonotoneIndexWarning, ParametricError)


def test_every_error_is_listed() -> None:
    assert sorted(errors.__all__) == sorted(
        name
        for name, value in vars(errors).items()
        if isinstance(value, type) and issubclass(value, (ParametricError, Warning))
    )
