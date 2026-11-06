"""Errors and warnings of the parametric module.

`ParametricError` is the base of every error the module raises. Each subclass
also derives from the built-in exception a caller would expect, so
``except ValueError`` keeps working for invalid inputs.
"""

from __future__ import annotations

__all__ = [
    "ComonotoneIndexWarning",
    "CoverDefinitionError",
    "DesignError",
    "InfeasibleDesignError",
    "LeakageError",
    "ParametricError",
    "ProbabilitySemanticsError",
    "SampleError",
    "SolverUnavailableError",
    "UnboundedDesignError",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
]


class ParametricError(Exception):
    """Base class of the errors raised by `crc_sdk.parametric`."""


class CoverDefinitionError(ParametricError, ValueError):
    """A cover, index definition, payout function or parameter is invalid.

    Raised at construction for wrong lengths, orders, bounds or non-finite
    values, and when a cover document cannot be read.
    """


class UnknownPayoutKindError(CoverDefinitionError):
    """A cover document names a payout `kind` that is not registered."""


class UnresolvedParameterError(ParametricError, ValueError):
    """A template, a cover with `Free` parameters, was evaluated.

    Resolve the free parameters first, with `Cover.with_values` or a design.
    """


class SampleError(ParametricError, ValueError):
    """Index values, losses or an outcome sample are invalid.

    Raised for non-finite or non-numeric index values and wrong array shapes.
    """


class ProbabilitySemanticsError(SampleError):
    """A source's probability semantics do not allow annual outcomes."""


class LeakageError(SampleError):
    """An out-of-sample evaluation shares periods with its design sample."""


class DesignError(ParametricError, RuntimeError):
    """A design could not be found or failed its feasibility check.

    `asset_id`, `status` and `solver_message` carry the asset, the solver's
    status and the solver's own message, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        asset_id: str | None = None,
        status: str | None = None,
        solver_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.asset_id = asset_id
        self.status = status
        self.solver_message = solver_message


class InfeasibleDesignError(DesignError):
    """No payout satisfies the design's constraints."""


class UnboundedDesignError(DesignError):
    """The design's objective is unbounded under its constraints."""


class SolverUnavailableError(ParametricError, ImportError):
    """The solver a design needs is not installed.

    `extra` names the optional dependency set that provides it.
    """

    def __init__(self, message: str, *, extra: str = "parametric") -> None:
        super().__init__(message)
        self.extra = extra


class ComonotoneIndexWarning(UserWarning):
    """The index and the loss move together, so a cover can track the loss.

    Emitted when a sample has no basis-risk model between index and loss.
    """
