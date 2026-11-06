"""Design a cover: choose its free parameters on a design sample.

`design(template, sample, objective, *, pricing, budget, limit, options)`
takes a template, a `Cover` with `Free` parameters, and returns a
`DesignResult`: the resolved cover and its design record. The design
minimises the objective (`MSE`, `ExpectedUncovered`, `WeightedCVaR` or a
`UserObjective`) on the sample, under the premium budget, the payout limit,
the bounds of every `Free` parameter and the payout family's own constraints
(non-negative amounts, non-decreasing levels under `monotone=True`).

**Routes.** A payout linear in its amounts (`LinearInAmounts`, which every
built-in family is) has its amounts solved exactly:

- E[U] and weighted CVaR: one linear programme (`route="lp"`), the CVaR in
  the Rockafellar-Uryasev form;
- MSE: one quadratic programme in the amounts (`route="qp"`);
- a cost-of-capital premium adds the Rockafellar-Uryasev auxiliaries of the
  premium's CVaR to either.

Free thresholds (attachment, exhaustion, breakpoints, knots, trigger) are
searched on their stated grids, with the LP or QP above at each grid point
(`route="grid"`). A `UserObjective`, or a payout that is not linear in its
amounts, is searched on a grid over every free parameter and evaluated in
numpy; candidates that break the budget or the limit are dropped. On a
finite sample the objective is piecewise constant or piecewise smooth in the
thresholds, so a stated grid is used rather than gradients; the objective at
every candidate is kept (`DesignResult.profile`).

**Solver.** The LP and QP routes use HiGHS from the optional extra
`crc-sdk[parametric]` (`pip install crc-sdk[parametric]`), imported only
when needed; without it they raise `SolverUnavailableError`. A grid
evaluated in numpy needs no solver.

**Scaling.** Losses and payouts are divided by the largest loss of a draw
(1 if that is at most 1e-6) before solving; results are in payout units.

**No silent failures.** After a solve, numpy recomputes the payout, the
objective, the premium, the budget and every constraint from the returned
parameters, with the relative tolerance `DesignOptions.feasibility_tolerance`
(relative to the larger of the bound and the scale). A violation, or a
solver status other than optimal, raises `DesignError`,
`InfeasibleDesignError` or `UnboundedDesignError`, with the asset id and the
solver's message. Free amounts below 1e-9 of the scale are set to 0 and
listed (`DesignResult.zeroed`).

**Non-unique optima** are reported, not hidden:

- an amount whose basis column is zero in every design year (a band that no
  year reaches) changes nothing on the sample; it is listed in
  `unidentified` and set by `DesignOptions.fill`: `"lowest"` (the default)
  takes the smallest value the constraints allow, `"highest"` the largest.
  Both keep the design exactly optimal;
- on the LP route, `DesignOptions(ranges=True)` ranges every identified
  amount over the near-optimal designs at two tolerances (1e-9 and 1e-11 of
  `max(1, |optimum|)`, in scaled units). An amount is pinned when its narrow
  range is at most half its wide one, or narrower than 1e-7 of the scale;
  `unique` is true when every identified amount is pinned. Without ranges,
  `unique` is `None` on the LP route;
- on the QP route, `unique` is true when `H = B' diag(p) B` over the
  identified amounts is positive definite (smallest eigenvalue above 1e-10
  of the largest);
- on a grid, `ties` counts the other candidates whose objective lies
  within `DesignOptions.feasibility_tolerance` times `max(1, |best|)` of the
  best, in scaled units; the first in lexicographic order wins. The LP and
  QP routes have one candidate, so `ties` is 0 there. `unique` is false
  with ties, else the inner solve's report
  (`None` for a grid evaluated in numpy).

For a weighted CVaR the design records `p_zero_loss` (the probability of a
zero loss), `tail_outcomes` (`(1 - beta) / p_min`) and `degenerate_tail`:
`"expected_uncovered"` when `beta <= p_zero_loss` (the objective is then a
multiple of E[U]), `"single_outcome"` when the tail holds at most one draw.

Everything here works in memory and does no I/O.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import Strict, model_validator

from crc_sdk.impacts._types import FiniteFloat
from crc_sdk.parametric import _solver
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import (
    CoverDefinitionError,
    DesignError,
    InfeasibleDesignError,
    SampleError,
    UnboundedDesignError,
)
from crc_sdk.parametric.objectives import (
    MSE,
    ExpectedUncovered,
    Objective,
    UserObjective,
    WeightedCVaR,
)
from crc_sdk.parametric.parameters import Free, _DefinitionModel
from crc_sdk.parametric.payouts import LinearConstraints, LinearInAmounts
from crc_sdk.parametric.pricing import CostOfCapitalLoad, Pricing, PurePremium
from crc_sdk.parametric.sample import OutcomeSample

__all__ = [
    "DESIGN_SEMANTICS_VERSION",
    "DesignOptions",
    "DesignResult",
    "Fill",
    "Route",
    "design",
]

FloatArray = NDArray[np.float64]

DESIGN_SEMANTICS_VERSION = "crc-parametric-design/1"
"""The version of the document behind `DesignResult.design_id`."""

Route = Literal["lp", "qp", "grid"]
"""How a design was solved: one LP, one QP, or a grid search."""

Fill = Literal["lowest", "highest"]
"""The rule that sets unidentified amounts."""

SCALE_FLOOR = 1e-6
"""A largest loss at most this gives the scale 1."""

ZERO_AMOUNT = 1e-9
"""Free amounts below this share of the scale are set to 0."""

NEAR_OPTIMAL_TOLERANCES = (1e-9, 1e-11)
"""The wide and the narrow near-optimal tolerance of the ranges."""

PINNED_SHRINK = 0.5
"""An amount whose narrow range is at most this share of its wide one is pinned."""

PINNED_WIDTH = 1e-7
"""An amount whose narrow range is below this share of the scale is pinned."""

MSE_EIGEN_TOLERANCE = 1e-10
"""QP route: unique when H's smallest eigenvalue exceeds this share of its largest."""

DEGENERATE_TAIL_TOLERANCE = 1e-9
"""The rounding allowance of the degenerate-tail comparisons."""

StrictInt = Annotated[int, Strict()]
StrictBool = Annotated[bool, Strict()]


class DesignOptions(_DefinitionModel):
    """How `design` searches, checks and reports.

    - `max_grid_points`: the largest grid (product of the free parameters'
      grids) a design may search; a larger one raises `DesignError`.
    - `feasibility_tolerance`: the relative tolerance of the checks after a
      solve, relative to the larger of the bound and the scale; also the
      tie tolerance of a grid: candidates whose objective is within it times
      `max(1, |best|)` of the best tie with it (in scaled units for the
      built-in objectives, in the objective's own units for a user
      objective, whose scaling is unknown).
    - `fill`: `"lowest"` or `"highest"`, the value of unidentified amounts.
    - `ranges`: on the LP route, range every identified amount over the
      near-optimal designs and report `unique` (four extra LPs per amount).

    Raises `CoverDefinitionError` (a `ValueError`) for invalid values.
    """

    max_grid_points: StrictInt = 10_000
    feasibility_tolerance: FiniteFloat = 1e-7
    fill: Fill = "lowest"
    ranges: StrictBool = False

    def __init__(
        self,
        *,
        max_grid_points: int = 10_000,
        feasibility_tolerance: float = 1e-7,
        fill: Fill = "lowest",
        ranges: bool = False,
    ) -> None:
        super().__init__(
            max_grid_points=max_grid_points,
            feasibility_tolerance=feasibility_tolerance,
            fill=fill,
            ranges=ranges,
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.max_grid_points < 1:
            raise CoverDefinitionError("max_grid_points must be at least 1")
        if not 0.0 < self.feasibility_tolerance < 1.0:
            raise CoverDefinitionError("feasibility_tolerance must lie in (0, 1)")
        return self

    def to_document(self) -> dict[str, Any]:
        """Return every option as plain JSON types."""
        return {
            "feasibility_tolerance": self.feasibility_tolerance,
            "fill": self.fill,
            "max_grid_points": self.max_grid_points,
            "ranges": self.ranges,
        }


@dataclass(frozen=True, eq=False)
class DesignResult:
    """A designed cover and its design record.

    - `cover`: the template with every free parameter resolved;
      `template_hash` the template's `definition_hash`; `design_id` the
      `"sha256:"` hash of the template hash, the objective, the pricing, the
      budget, the limit, the options and the design sample's hash;
      `parameters` the designed value of each free parameter.
    - `objective` (document) and `objective_value`, on the design sample in
      payout units (squared for MSE); `premium`, the design premium (the
      contract price, priced on the design sample with `pricing`);
      `expected_loss`; `budget_bound`, the largest premium allowed, and
      `budget_use`, the premium's share of it.
    - `route`: `"lp"`, `"qp"` or `"grid"`; `amounts`: how the amounts were
      chosen (`"lp"`, `"qp"`, or `"grid"` when they were on the grid too).
    - `solver` (`{"name", "version"}`, or `None` for a grid in numpy),
      `status`, `solver_message`, `iterations`, `solves`, `seconds` (solver
      time) and `solver_options`.
    - The non-uniqueness report: `unidentified` amounts and the `fill` rule
      that set them, `zeroed` amounts, `ranges` (wide ranges in payout
      units, with `ranges=True` on the LP route), `unique`, `ties`, and the
      grid's `profile` (a `pyarrow.Table`: one row per candidate, its values,
      `objective` and `status`) and `grid` counts.
    - The degenerate-tail flags `p_zero_loss`, `tail_outcomes` and
      `degenerate_tail`.
    - `design_sample`, the sample the cover was designed on.

    `evaluate` and `compare` take a `DesignResult` in place of a cover: its
    rows use the design premium and carry `unique`, `ties` and
    `degenerate_tail`, and `to_record()` goes into their metadata.
    """

    cover: Cover
    template_hash: str
    design_id: str
    parameters: Mapping[str, float]
    objective: Mapping[str, Any]
    objective_value: float
    pricing: Mapping[str, Any]
    budget: Mapping[str, Any] | None
    limit: Mapping[str, Any] | None
    options: DesignOptions
    premium: float
    expected_loss: float
    budget_bound: float | None
    budget_use: float | None
    route: Route
    amounts: Literal["lp", "qp", "grid"]
    solver: Mapping[str, str] | None
    status: str
    solver_message: str
    iterations: int
    solves: int
    seconds: float
    solver_options: Mapping[str, Any]
    unidentified: tuple[str, ...]
    fill: Fill
    zeroed: tuple[str, ...]
    ranges: Mapping[str, tuple[float, float]] | None
    unique: bool | None
    ties: int
    profile: Any | None
    grid: Mapping[str, int] | None
    p_zero_loss: float
    tail_outcomes: float | None
    degenerate_tail: str | None
    design_sample: OutcomeSample = field(repr=False)

    @property
    def sample_hash(self) -> str:
        """The design sample's `content_hash`."""
        return self.design_sample.content_hash

    def to_record(self) -> dict[str, Any]:
        """Return the design record as plain JSON types.

        The record holds the ids and hashes, the objective, pricing, budget,
        limit and options, the values, the solve (`solver` with the name and
        version; `solve` with the status, message, iterations, solves, time
        and options), the non-uniqueness report and the degenerate-tail
        flags. `evaluate` writes it to its metadata's `designs`.
        """
        ranges = (
            None
            if self.ranges is None
            else {name: [low, high] for name, (low, high) in self.ranges.items()}
        )
        return {
            "design_id": self.design_id,
            "template_hash": self.template_hash,
            "cover_hash": self.cover.definition_hash,
            "sample_hash": self.sample_hash,
            "objective": copy.deepcopy(dict(self.objective)),
            "objective_value": self.objective_value,
            "pricing": dict(self.pricing),
            "budget": None if self.budget is None else dict(self.budget),
            "limit": None if self.limit is None else dict(self.limit),
            "options": self.options.to_document(),
            "parameters": dict(self.parameters),
            "premium": self.premium,
            "expected_loss": self.expected_loss,
            "budget_bound": self.budget_bound,
            "budget_use": self.budget_use,
            "route": self.route,
            "amounts": self.amounts,
            "solver": None if self.solver is None else dict(self.solver),
            "solve": {
                "status": self.status,
                "message": self.solver_message,
                "iterations": self.iterations,
                "solves": self.solves,
                "seconds": self.seconds,
                "options": copy.deepcopy(dict(self.solver_options)),
            },
            "unidentified": list(self.unidentified),
            "fill": self.fill,
            "zeroed": list(self.zeroed),
            "ranges": ranges,
            "unique": self.unique,
            "ties": self.ties,
            "grid": None if self.grid is None else dict(self.grid),
            "p_zero_loss": self.p_zero_loss,
            "tail_outcomes": self.tail_outcomes,
            "degenerate_tail": self.degenerate_tail,
        }


# -- the problem -------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Context:
    """The design problem in payout units, with its scale."""

    asset_id: str
    index: FloatArray
    loss: FloatArray
    probability: FloatArray
    scale: float
    expected_loss: float
    bound: float | None
    limit: float | None
    objective: Objective
    pricing: Pricing
    options: DesignOptions

    def scaled(self, value: float) -> float:
        """An objective value in scaled units."""
        if isinstance(self.objective, UserObjective):
            return value
        return value / self.scale**self.objective.homogeneity

    def slack(self, bound: float) -> float:
        """The allowance of a check against `bound`, in payout units."""
        return self.options.feasibility_tolerance * max(self.scale, abs(bound))

    def error(
        self,
        kind: type[DesignError],
        message: str,
        *,
        status: str | None = None,
        solver_message: str | None = None,
    ) -> DesignError:
        return kind(
            f"asset {self.asset_id!r}: {message}",
            asset_id=self.asset_id,
            status=status,
            solver_message=solver_message,
        )


@dataclass
class _Solves:
    """The solves of one design, for the record."""

    iterations: int = 0
    count: int = 0
    seconds: float = 0.0
    last: _solver.Solution | None = None

    def add(self, solution: _solver.Solution) -> _solver.Solution:
        self.iterations += solution.iterations
        self.count += 1
        self.seconds += solution.seconds
        self.last = solution
        return solution


@dataclass(frozen=True, eq=False)
class _Amounts:
    """The amounts of one cover, solved by an LP or a QP."""

    cover: Cover
    values: dict[str, float]
    objective_value: float
    route: Literal["lp", "qp"]
    unidentified: tuple[str, ...]
    zeroed: tuple[str, ...]
    unique: bool | None
    ranges: dict[str, tuple[float, float]] | None


class _Builder:
    """Assembles a programme column by column and row by row."""

    def __init__(self) -> None:
        self.cost: list[FloatArray] = []
        self.col_lower: list[FloatArray] = []
        self.col_upper: list[FloatArray] = []
        self.row_lower: list[FloatArray] = []
        self.row_upper: list[FloatArray] = []
        self.rows: list[NDArray[np.int64]] = []
        self.columns: list[NDArray[np.int64]] = []
        self.values: list[FloatArray] = []
        self.n_columns = 0
        self.n_rows = 0

    def variables(self, count: int, lower: Any, upper: Any, cost: Any) -> int:
        start = self.n_columns
        for target, value in (
            (self.col_lower, lower),
            (self.col_upper, upper),
            (self.cost, cost),
        ):
            target.append(np.broadcast_to(np.asarray(value, np.float64), (count,)))
        self.n_columns += count
        return start

    def constraints(self, count: int, lower: Any, upper: Any) -> int:
        start = self.n_rows
        self.row_lower.append(np.broadcast_to(np.asarray(lower, np.float64), (count,)))
        self.row_upper.append(np.broadcast_to(np.asarray(upper, np.float64), (count,)))
        self.n_rows += count
        return start

    def entries(self, rows: Any, columns: Any, values: Any) -> None:
        rows_array = np.asarray(rows, dtype=np.int64).reshape(-1)
        columns_array = np.asarray(columns, dtype=np.int64).reshape(-1)
        values_array = np.broadcast_to(
            np.asarray(values, dtype=np.float64), rows_array.shape
        )
        self.rows.append(rows_array)
        self.columns.append(columns_array)
        self.values.append(np.array(values_array))

    def programme(
        self, *, hessian: _solver.SparseColumns | None = None, offset: float = 0.0
    ) -> _solver.Programme:
        def joined(parts: list[Any], dtype: Any) -> Any:
            return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)

        matrix = _solver.sparse_columns(
            joined(self.rows, np.int64),
            joined(self.columns, np.int64),
            joined(self.values, np.float64),
            (self.n_rows, self.n_columns),
        )
        return _solver.Programme(
            cost=joined(self.cost, np.float64),
            col_lower=joined(self.col_lower, np.float64),
            col_upper=joined(self.col_upper, np.float64),
            matrix=matrix,
            row_lower=joined(self.row_lower, np.float64),
            row_upper=joined(self.row_upper, np.float64),
            offset=offset,
            hessian=hessian,
        )


def _block(builder: _Builder, first_row: int, basis: FloatArray, sign: float) -> None:
    """Add `sign * basis @ theta` to the rows from `first_row` on."""
    rows, columns = np.nonzero(basis)
    builder.entries(first_row + rows, columns, sign * basis[rows, columns])


def _lower_triangle(matrix: FloatArray, n_columns: int) -> _solver.SparseColumns:
    """The lower triangle of a square block in the top left of `n` columns."""
    rows, columns = np.nonzero(np.tril(matrix))
    return _solver.sparse_columns(
        rows, columns, matrix[rows, columns], (n_columns, n_columns)
    )


def _programme(
    basis: FloatArray, constraints: LinearConstraints, context: _Context
) -> tuple[_solver.Programme, Literal["lp", "qp"]]:
    """Build the LP or QP over the amounts, in scaled units.

    The amounts are the first columns, in `amount_names()` order.
    """
    n, k = basis.shape
    scale = context.scale
    p = context.probability
    loss = context.loss / scale
    builder = _Builder()
    builder.variables(k, constraints.lower / scale, constraints.upper / scale, 0.0)
    objective = context.objective
    route: Literal["lp", "qp"]
    hessian: _solver.SparseColumns | None = None
    offset = 0.0
    if isinstance(objective, MSE):
        route = "qp"
    else:
        route = "lp"
        alpha = objective.alpha if isinstance(objective, WeightedCVaR) else 0.0
        uncovered = builder.variables(n, 0.0, math.inf, (1.0 - alpha) * p)
        first = builder.constraints(n, loss, math.inf)
        _block(builder, first, basis, 1.0)
        builder.entries(first + np.arange(n), uncovered + np.arange(n), 1.0)
        if isinstance(objective, WeightedCVaR) and alpha > 0.0:
            beta = objective.beta
            zeta = builder.variables(1, -math.inf, math.inf, alpha)
            excess = builder.variables(n, 0.0, math.inf, alpha * p / (1.0 - beta))
            first = builder.constraints(n, 0.0, math.inf)
            rows = first + np.arange(n)
            builder.entries(rows, excess + np.arange(n), 1.0)
            builder.entries(rows, uncovered + np.arange(n), -1.0)
            builder.entries(rows, np.full(n, zeta), 1.0)
    if context.bound is not None:
        expected = p @ basis
        pricing = context.pricing
        if isinstance(pricing, CostOfCapitalLoad) and pricing.rate > 0.0:
            rate, q = pricing.rate, pricing.q
            eta = builder.variables(1, -math.inf, math.inf, 0.0)
            tail = builder.variables(n, 0.0, math.inf, 0.0)
            first = builder.constraints(n, 0.0, math.inf)
            rows = first + np.arange(n)
            builder.entries(rows, tail + np.arange(n), 1.0)
            _block(builder, first, basis, -1.0)
            builder.entries(rows, np.full(n, eta), 1.0)
            row = builder.constraints(1, -math.inf, context.bound / scale)
            builder.entries(np.full(k, row), np.arange(k), (1.0 - rate) * expected)
            builder.entries([row], [eta], rate)
            builder.entries(np.full(n, row), tail + np.arange(n), rate * p / (1.0 - q))
        else:
            row = builder.constraints(1, -math.inf, context.bound / scale)
            builder.entries(np.full(k, row), np.arange(k), expected)
    m = constraints.a_ub.shape[0]
    if m:
        first = builder.constraints(m, -math.inf, constraints.b_ub / scale)
        _block(builder, first, constraints.a_ub, 1.0)
    if route == "qp":
        weighted = basis * p[:, None]
        gram = basis.T @ weighted
        builder.cost[0] = -2.0 * (weighted.T @ loss)
        hessian = _lower_triangle(2.0 * gram, builder.n_columns)
        offset = math.fsum((p * loss**2).tolist())
    return builder.programme(hessian=hessian, offset=offset), route


# -- solving -----------------------------------------------------------------


def _solve(
    programme: _solver.Programme,
    route: Literal["lp", "qp"],
    context: _Context,
    solves: _Solves,
    options: Mapping[str, Any] | None = None,
) -> _solver.Solution:
    """Solve; raise the typed error of any status other than optimal."""
    run = _solver.solve_qp if route == "qp" else _solver.solve_lp
    return _checked(lambda chosen: run(programme, chosen), context, solves, options)


def _checked(
    run: Callable[[Mapping[str, Any] | None], _solver.Solution],
    context: _Context,
    solves: _Solves,
    options: Mapping[str, Any] | None = None,
) -> _solver.Solution:
    """Call `run(options)`; raise the typed error of any status but optimal."""
    solution = solves.add(run(options))
    if solution.status == "infeasible_or_unbounded":
        # Presolve may not tell the two apart; the solve without it does.
        solution = solves.add(run({**(options or {}), "presolve": "off"}))
    if solution.status == "optimal":
        return solution
    kinds: dict[str, type[DesignError]] = {
        "infeasible": InfeasibleDesignError,
        "unbounded": UnboundedDesignError,
    }
    reasons = {
        "infeasible": "no payout satisfies the constraints",
        "unbounded": "the objective is unbounded under the constraints",
    }
    raise context.error(
        kinds.get(solution.status, DesignError),
        f"{reasons.get(solution.status, 'the solver found no optimal design')} "
        f"(HiGHS: {solution.message})",
        status=solution.status,
        solver_message=solution.message,
    )


def _order_edges(
    constraints: LinearConstraints,
) -> tuple[list[tuple[int, int]], NDArray[np.bool_]]:
    """The rows `a_i <= a_j` of the constraints, and the other rows.

    Returns the pairs `(i, j)` and a mask of the rows that are not of that
    form (all-zero rows are neither).
    """
    edges: list[tuple[int, int]] = []
    other = np.zeros(constraints.a_ub.shape[0], dtype=np.bool_)
    for row, (coefficients, bound) in enumerate(
        zip(constraints.a_ub, constraints.b_ub, strict=True)
    ):
        nonzero = np.flatnonzero(coefficients)
        if nonzero.size == 0:
            continue
        if nonzero.size == 2 and bound == 0.0:
            first, second = (float(coefficients[i]) for i in nonzero)
            if first == -second:
                low, high = (nonzero[0], nonzero[1])
                if first < 0:
                    low, high = high, low
                edges.append((int(low), int(high)))
                continue
        other[row] = True
    return edges, other


def _propagate(
    values: FloatArray,
    edges: Sequence[tuple[int, int]],
    *,
    up: NDArray[np.bool_] | None = None,
    down: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Enforce `a_i <= a_j` by raising `a_j` (where `up`) or lowering `a_i`."""
    result = values.copy()
    for _ in range(result.shape[0] + 1):
        changed = False
        for i, j in edges:
            if result[i] > result[j]:
                if up is not None and up[j]:
                    result[j] = result[i]
                    changed = True
                elif down is not None and down[i]:
                    result[i] = result[j]
                    changed = True
        if not changed:
            break
    return result


def _repair(
    values: FloatArray,
    constraints: LinearConstraints,
    edges: Sequence[tuple[int, int]],
) -> FloatArray:
    """Move solver values onto the bounds and order constraints exactly.

    A solver meets its constraints only to rounding, while a cover checks
    them exactly. The values are clipped to bounds tightened along the order
    constraints, then raised along them, which keeps every bound.
    """
    k = values.shape[0]
    everywhere = np.ones(k, dtype=np.bool_)
    lower = _propagate(constraints.lower.copy(), edges, up=everywhere)
    upper = _propagate(constraints.upper.copy(), edges, down=everywhere)
    clipped = np.minimum(np.maximum(values, lower), upper)
    return _propagate(clipped, edges, up=everywhere)


def _fill(
    values: FloatArray,
    unidentified: NDArray[np.bool_],
    constraints: LinearConstraints,
    edges: Sequence[tuple[int, int]],
    other: NDArray[np.bool_],
    context: _Context,
    solves: _Solves,
    names: Sequence[str],
) -> FloatArray:
    """Set the unidentified amounts by the fill rule.

    `"lowest"` takes the smallest value the constraints allow with the
    identified amounts fixed, `"highest"` the largest. Bounds and order
    constraints are propagated exactly; other constraints on an unidentified
    amount are solved as an LP.
    """
    if not unidentified.any():
        return values
    highest = context.options.fill == "highest"
    touches = (constraints.a_ub[other][:, unidentified] != 0).any()
    if not touches:
        start = values.copy()
        if highest:
            start[unidentified] = constraints.upper[unidentified]
            filled = _propagate(start, edges, down=unidentified)
        else:
            start[unidentified] = constraints.lower[unidentified]
            filled = _propagate(start, edges, up=unidentified)
    else:
        lower = np.where(unidentified, constraints.lower, values) / context.scale
        upper = np.where(unidentified, constraints.upper, values) / context.scale
        builder = _Builder()
        sign = -1.0 if highest else 1.0
        builder.variables(values.shape[0], lower, upper, sign * unidentified)
        m = constraints.a_ub.shape[0]
        first = builder.constraints(m, -math.inf, constraints.b_ub / context.scale)
        _block(builder, first, constraints.a_ub, 1.0)
        solution = _solve(builder.programme(), "lp", context, solves)
        assert solution.values is not None
        filled = np.where(unidentified, solution.values * context.scale, values)
        filled = _repair(filled, constraints, edges)
    if highest and not np.isfinite(filled[unidentified]).all():
        missing = [name for name, free in zip(names, unidentified, strict=True) if free]
        raise context.error(
            DesignError,
            f"fill='highest' needs an upper bound (a Free upper bound or a "
            f"Limit) above the unidentified amounts {missing}",
        )
    return filled


def _ranges(
    programme: _solver.Programme,
    optimum: float,
    identified: Sequence[int],
    context: _Context,
    solves: _Solves,
) -> tuple[dict[int, tuple[float, float]], bool]:
    """Range the identified amounts over two near-optimal sets.

    Each near-optimal set is the programme's feasible set with one more row,
    `cost @ x <= optimum + tolerance * max(1, |optimum|)`. The four LPs per
    amount (its smallest and largest value, at each tolerance) share one
    HiGHS instance: each changes only the objective or that row's bound,
    and every one after the first starts from the previous solve's basis.

    Returns each amount's wide range in scaled units, and whether every one
    is pinned.
    """
    session = _solver.LPSession(programme, _solver.RANGING_OPTIONS)
    near = session.add_row(programme.cost, -math.inf, math.inf)
    found: list[dict[int, tuple[float, float]]] = []
    for tolerance in NEAR_OPTIMAL_TOLERANCES:
        session.change_row_bounds(
            near, -math.inf, optimum + tolerance * max(1.0, abs(optimum))
        )
        bounds: dict[int, tuple[float, float]] = {}
        for column in identified:
            ends: list[float] = []
            for sign in (1.0, -1.0):
                cost = np.zeros(programme.n_columns)
                cost[column] = sign
                session.change_cost(cost)
                solution = _checked(session.solve, context, solves)
                assert solution.values is not None
                ends.append(float(solution.values[column]))
            bounds[column] = (ends[0], ends[1])
        found.append(bounds)
    wide, narrow = found
    pinned = True
    for column in identified:
        width = wide[column][1] - wide[column][0]
        narrow_width = narrow[column][1] - narrow[column][0]
        if not (narrow_width <= PINNED_SHRINK * width or narrow_width < PINNED_WIDTH):
            pinned = False
    return wide, pinned


def _premium(context: _Context, payout: FloatArray) -> float:
    return context.pricing.premium(payout, context.probability)


def _objective_value(context: _Context, payout: FloatArray) -> float:
    return context.objective.value(context.loss, payout, context.probability)


def _verify(
    context: _Context,
    cover: Cover,
    amounts: FloatArray,
    basis: FloatArray | None,
    constraints: LinearConstraints | None,
) -> tuple[FloatArray, float, float]:
    """Recompute a design in numpy; raise `DesignError` on any violation.

    Returns the payout, the objective value and the premium.
    """
    payout = cover.payout_values(context.index)
    problems: list[str] = []
    if basis is not None:
        gap = float(np.max(np.abs(payout - basis @ amounts), initial=0.0))
        if gap > context.slack(0.0):
            problems.append(f"the payout differs from basis @ amounts by {gap:.3g}")
    if constraints is not None:
        below = constraints.lower - amounts
        above = amounts - constraints.upper
        for name, excess, bound in (
            ("a lower bound", below, constraints.lower),
            ("an upper bound", above, constraints.upper),
        ):
            finite = np.isfinite(bound)
            if (
                finite.any()
                and (
                    excess[finite] > np.array([context.slack(b) for b in bound[finite]])
                ).any()
            ):
                problems.append(f"an amount breaks {name}")
        if constraints.a_ub.shape[0]:
            activity = constraints.a_ub @ amounts
            slack = np.array([context.slack(b) for b in constraints.b_ub])
            if (activity - constraints.b_ub > slack).any():
                problems.append("the amounts break a linear constraint of the payout")
    premium = _premium(context, payout)
    if context.bound is not None and premium - context.bound > context.slack(
        context.bound
    ):
        problems.append(
            f"the premium {premium:.10g} exceeds the budget {context.bound:.10g}"
        )
    if context.limit is not None:
        largest = float(payout.max(initial=0.0))
        if largest - context.limit > context.slack(context.limit):
            problems.append(
                f"a payout of {largest:.10g} exceeds the limit {context.limit:.10g}"
            )
    if problems:
        raise context.error(
            DesignError,
            "the design failed its feasibility check: " + "; ".join(problems),
            status="infeasible_solution",
        )
    return payout, _objective_value(context, payout), premium


def _amount_values(
    cover: Cover, names: Sequence[str], amounts: FloatArray
) -> dict[str, float]:
    parameters = cover.payout.parameters()
    return {
        name: float(value) + 0.0
        for name, value in zip(names, amounts, strict=True)
        if isinstance(parameters[name], Free)
    }


def _design_amounts(
    candidate: Cover,
    context: _Context,
    solves: _Solves,
    *,
    report: bool,
) -> _Amounts:
    """Solve the amounts of a cover whose thresholds are resolved.

    With `report`, compute the uniqueness report (ranges on the LP route,
    when asked for; the eigenvalue rule on the QP route).
    """
    payout = cast(LinearInAmounts, candidate.payout)
    names = tuple(payout.amount_names())
    basis = np.asarray(payout.basis(context.index), dtype=np.float64)
    if basis.shape != (context.index.shape[0], len(names)):
        raise context.error(
            DesignError,
            f"the payout's basis has shape {basis.shape}, expected "
            f"{(context.index.shape[0], len(names))}",
        )
    if not np.isfinite(basis).all():
        raise context.error(DesignError, "the payout's basis is not finite")
    constraints = payout.amount_constraints(context.limit)
    if constraints.n_amounts != len(names):
        raise context.error(
            DesignError,
            f"the payout's amount constraints cover {constraints.n_amounts} "
            f"amounts, not {len(names)}",
        )
    empty = [
        name
        for name, low, high in zip(
            names, constraints.lower, constraints.upper, strict=True
        )
        if low > high
    ]
    if empty:
        raise context.error(
            InfeasibleDesignError,
            f"no value of {empty} satisfies its bounds: a fixed amount or a lower "
            "bound lies above the upper bound or the limit",
            status="infeasible",
        )
    parameters = payout.parameters()
    free = np.array([isinstance(parameters[name], Free) for name in names])
    if (
        isinstance(context.objective, ExpectedUncovered | WeightedCVaR)
        and context.bound is None
        and not np.isfinite(constraints.upper[free]).all()
    ):
        raise context.error(
            DesignError,
            "the expected uncovered loss and the weighted CVaR need a budget or "
            "bounded amounts (a Free upper bound or a Limit): without either, any "
            "payout at least as large as the loss is optimal",
        )
    programme, route = _programme(basis, constraints, context)
    solution = _solve(programme, route, context, solves)
    assert solution.values is not None and solution.objective is not None
    k = len(names)
    edges, other = _order_edges(constraints)
    amounts = _repair(solution.values[:k] * context.scale, constraints, edges)
    small = (
        free
        & (constraints.lower <= 0.0)
        & (np.abs(amounts) <= ZERO_AMOUNT * context.scale)
    )
    zeroed = tuple(
        name
        for name, flag, value in zip(names, small, amounts, strict=True)
        if flag and value != 0.0
    )
    amounts = _repair(np.where(small, 0.0, amounts), constraints, edges)
    identified = np.any(basis != 0.0, axis=0)
    unidentified = free & ~identified
    amounts = _fill(
        amounts, unidentified, constraints, edges, other, context, solves, names
    )
    values = _amount_values(candidate, names, amounts)
    try:
        cover = candidate.with_values(values)
    except CoverDefinitionError as error:
        raise context.error(
            DesignError, f"the solved amounts do not make a valid cover: {error}"
        ) from error
    _, value, _ = _verify(context, cover, amounts, basis, constraints)
    scaled = context.scaled(value)
    if abs(scaled - solution.objective) > context.options.feasibility_tolerance * max(
        1.0, abs(scaled)
    ):
        raise context.error(
            DesignError,
            f"the objective recomputed in numpy ({scaled:.12g}, scaled) differs "
            f"from the solver's ({solution.objective:.12g})",
            status="inconsistent_objective",
        )
    unique: bool | None = None
    ranges: dict[str, tuple[float, float]] | None = None
    ranged = [j for j in range(k) if free[j] and identified[j]]
    if report and route == "qp":
        if ranged:
            gram = basis[:, ranged].T @ (
                basis[:, ranged] * context.probability[:, None]
            )
            eigen = np.linalg.eigvalsh(gram)
            unique = bool(eigen.min() > MSE_EIGEN_TOLERANCE * eigen.max())
        else:
            unique = True
    elif report and context.options.ranges:
        wide, unique = _ranges(programme, solution.objective, ranged, context, solves)
        ranges = {
            names[j]: (low * context.scale + 0.0, high * context.scale + 0.0)
            for j, (low, high) in wide.items()
        }
    return _Amounts(
        cover=cover,
        values=values,
        objective_value=value,
        route=route,
        unidentified=tuple(
            name for name, flag in zip(names, unidentified, strict=True) if flag
        ),
        zeroed=zeroed,
        unique=unique,
        ranges=ranges,
    )


# -- the grid ----------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _GridOutcome:
    cover: Cover
    amounts: _Amounts | None
    ties: int
    profile: Any
    counts: dict[str, int]


def _grid(
    template: Cover,
    names: Sequence[str],
    context: _Context,
    solves: _Solves,
    *,
    inner: bool,
) -> _GridOutcome:
    """Search the grids of `names`; solve the amounts at each point if `inner`."""
    parameters = template.payout.parameters()
    grids: list[tuple[float, ...]] = []
    for name in names:
        parameter = parameters[name]
        assert isinstance(parameter, Free)
        if parameter.grid is None:
            raise context.error(
                DesignError,
                f"a grid search needs a grid on every parameter it searches; "
                f"{name} has none (give Free(..., grid=...))",
            )
        grids.append(parameter.grid)
    size = math.prod(len(grid) for grid in grids)
    if size > context.options.max_grid_points:
        raise context.error(
            DesignError,
            f"the grid has {size} points, more than max_grid_points="
            f"{context.options.max_grid_points}",
        )
    linear = isinstance(template.payout, LinearInAmounts)
    rows: list[tuple[tuple[float, ...], float | None, str]] = []
    first_infeasible: str | None = None
    began = time.perf_counter()
    for point in itertools.product(*grids):
        values = dict(zip(names, point, strict=True))
        try:
            candidate = template.with_values(values)
        except CoverDefinitionError:
            rows.append((point, None, "invalid"))
            continue
        if inner:
            try:
                amounts = _design_amounts(candidate, context, solves, report=False)
            except InfeasibleDesignError as error:
                first_infeasible = first_infeasible or str(error)
                rows.append((point, None, "infeasible"))
                continue
            rows.append((point, amounts.objective_value, "optimal"))
            continue
        payout = candidate.payout_values(context.index)
        status = _grid_status(context, candidate, payout, linear)
        value = _objective_value(context, payout) if status == "feasible" else None
        rows.append((point, value, status))
    if not inner:
        solves.seconds += time.perf_counter() - began
    feasible = [
        (position, context.scaled(value))
        for position, (_, value, _) in enumerate(rows)
        if value is not None
    ]
    counts = {
        status: sum(1 for row in rows if row[2] == status)
        for status in ("optimal", "feasible", "infeasible", "invalid")
        + ("over_budget", "over_limit")
    }
    counts = {"candidates": size, **{k: v for k, v in counts.items() if v}}
    if not feasible:
        reason = f"; the first: {first_infeasible}" if first_infeasible else ""
        raise context.error(
            InfeasibleDesignError,
            f"no grid candidate is feasible ({_counts_text(counts)}){reason}",
            status="infeasible",
        )
    best = min(value for _, value in feasible)
    allowance = context.options.feasibility_tolerance * max(1.0, abs(best))
    within = [position for position, value in feasible if value - best <= allowance]
    chosen = within[0]
    point = rows[chosen][0]
    cover = template.with_values(dict(zip(names, point, strict=True)))
    chosen_amounts = (
        _design_amounts(cover, context, solves, report=True) if inner else None
    )
    columns: dict[str, Any] = {
        name: pa.array([row[0][i] for row in rows], pa.float64())
        for i, name in enumerate(names)
    }
    columns["objective"] = pa.array([row[1] for row in rows], pa.float64())
    columns["status"] = pa.array([row[2] for row in rows], pa.string())
    return _GridOutcome(
        cover=cover if chosen_amounts is None else chosen_amounts.cover,
        amounts=chosen_amounts,
        ties=len(within) - 1,
        profile=pa.table(columns),
        counts=counts,
    )


def _counts_text(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}: {count}" for name, count in counts.items())


def _grid_status(
    context: _Context, candidate: Cover, payout: FloatArray, linear: bool
) -> str:
    """Whether a fully resolved candidate meets the budget and the limit."""
    if context.limit is not None:
        if linear:
            family = cast(LinearInAmounts, candidate.payout)
            constraints = family.amount_constraints(context.limit)
            values = np.array(
                [
                    float(cast(float, candidate.payout.parameters()[name]))
                    for name in family.amount_names()
                ]
            )
            over = values - constraints.upper
            if (over > np.array([context.slack(b) for b in constraints.upper])).any():
                return "over_limit"
        elif float(payout.max(initial=0.0)) - context.limit > context.slack(
            context.limit
        ):
            return "over_limit"
    if context.bound is not None and _premium(context, payout) - context.bound > (
        context.slack(context.bound)
    ):
        return "over_budget"
    return "feasible"


# -- the public function -----------------------------------------------------


def _check_inputs(
    template: Any,
    sample: Any,
    objective: Any,
    pricing: Any,
    budget: Any,
    limit: Any,
    options: Any,
) -> None:
    checks: tuple[tuple[str, Any, Any, str], ...] = (
        ("template", template, Cover, "a Cover"),
        ("sample", sample, OutcomeSample, "an OutcomeSample"),
        (
            "objective",
            objective,
            (MSE, ExpectedUncovered, WeightedCVaR, UserObjective),
            "MSE(), ExpectedUncovered(), WeightedCVaR(alpha, beta) or a UserObjective",
        ),
        (
            "pricing",
            pricing,
            (PurePremium, CostOfCapitalLoad),
            "PurePremium() or CostOfCapitalLoad(rate, q)",
        ),
        ("options", options, DesignOptions, "DesignOptions"),
    )
    for name, value, kind, text in checks:
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be {text}, got {type(value).__name__}")
    if budget is not None and not isinstance(budget, Budget):
        raise TypeError(f"budget must be a Budget or None, got {type(budget).__name__}")
    if limit is not None and not isinstance(limit, Limit):
        raise TypeError(f"limit must be a Limit or None, got {type(limit).__name__}")


def _check_template(template: Cover, sample: OutcomeSample) -> None:
    if not template.free_parameters():
        raise ValueError(
            "the template has no free parameters: there is nothing to design; "
            "evaluate the cover with evaluate()"
        )
    if template.payout_unit != sample.loss_unit:
        raise SampleError(
            f"the cover pays in {template.payout_unit!r}, but the sample's loss "
            f"is in {sample.loss_unit!r}"
        )
    if template.index.unit != sample.index_unit:
        raise SampleError(
            f"the cover's index is in {template.index.unit!r}, but the sample's "
            f"index is in {sample.index_unit!r}"
        )
    asset = sample.asset_ids[0]
    if template.asset_id is not None and template.asset_id != asset:
        raise SampleError(
            f"the cover is written for asset {template.asset_id!r}, but the "
            f"sample holds asset {asset!r}"
        )


def _tail_flags(
    objective: Objective, loss: FloatArray, probability: FloatArray
) -> tuple[float, float | None, str | None]:
    p_zero = math.fsum(probability[loss == 0.0].tolist()) + 0.0
    if not isinstance(objective, WeightedCVaR):
        return p_zero, None, None
    tail = (1.0 - objective.beta) / float(probability.min())
    flag: str | None = None
    if objective.beta <= p_zero + DEGENERATE_TAIL_TOLERANCE:
        flag = "expected_uncovered"
    elif tail <= 1.0 + DEGENERATE_TAIL_TOLERANCE:
        flag = "single_outcome"
    return p_zero, tail, flag


def design(
    template: Cover,
    sample: OutcomeSample,
    objective: Objective,
    *,
    pricing: Pricing = PurePremium(),
    budget: Budget | None = None,
    limit: Limit | None = None,
    options: DesignOptions = DesignOptions(),
) -> DesignResult:
    """Design a cover: choose the template's free parameters on `sample`.

    `template` is a `Cover` with `Free` parameters, written for the sample's
    asset (or for none), paying in the sample's loss unit on an index in the
    sample's index unit. `objective` is `MSE()`, `ExpectedUncovered()`,
    `WeightedCVaR(alpha, beta)` or a `UserObjective`. `pricing` prices the
    premium on the sample; `budget` (`Budget(ratio=...)` or
    `Budget(amount=...)`) bounds it; `limit` (`Limit(amount)`) bounds every
    payout. `options` (`DesignOptions`) sets the grid cap, the tolerances,
    the fill rule of unidentified amounts and whether the LP route ranges
    its optimal set. The routes, checks and reports are described in the
    module docstring.

    The expected uncovered loss and the weighted CVaR need a budget or
    bounded amounts; MSE needs neither.

    Returns a `DesignResult` with the resolved cover and its record. Raises
    `SolverUnavailableError` without the `parametric` extra on the LP and QP
    routes; `InfeasibleDesignError` when no payout meets the constraints;
    `UnboundedDesignError` for an unbounded objective; `DesignError` for any
    other failed solve or check; `SampleError` when the template's units or
    asset differ from the sample's; `ValueError` for a template without free
    parameters; and `TypeError` for arguments of the wrong type.
    """
    _check_inputs(template, sample, objective, pricing, budget, limit, options)
    _check_template(template, sample)
    loss = np.array(sample.loss[:, 0], dtype=np.float64)
    index = np.array(sample.index[:, 0], dtype=np.float64)
    probability = np.array(sample.probability, dtype=np.float64)
    largest = float(loss.max())
    scale = largest if largest > SCALE_FLOOR else 1.0
    expected_loss = math.fsum((probability * loss).tolist()) + 0.0
    bound = None if budget is None else budget.bound(expected_loss)
    context = _Context(
        asset_id=sample.asset_ids[0],
        index=index,
        loss=loss,
        probability=probability,
        scale=scale,
        expected_loss=expected_loss,
        bound=bound,
        limit=None if limit is None else limit.amount,
        objective=objective,
        pricing=pricing,
        options=options,
    )
    payout = template.payout
    free = template.free_parameters()
    solves = _Solves()
    linear = isinstance(payout, LinearInAmounts) and not isinstance(
        objective, UserObjective
    )
    ties = 0
    profile: Any = None
    counts: dict[str, int] | None = None
    amounts: _Amounts | None
    if linear:
        _solver.require()
        amount_names = set(cast(LinearInAmounts, payout).amount_names())
        shape = [name for name in free if name not in amount_names]
        if shape:
            outcome = _grid(template, shape, context, solves, inner=True)
            amounts, ties = outcome.amounts, outcome.ties
            profile, counts = outcome.profile, outcome.counts
        else:
            amounts = _design_amounts(template, context, solves, report=True)
        assert amounts is not None
        cover = amounts.cover
        route: Route = "grid" if shape else amounts.route
        amounts_route: Literal["lp", "qp", "grid"] = amounts.route
        unique = amounts.unique
        if ties:
            unique = False
        solver: dict[str, str] | None = _solver.identity()
        ranging = amounts.route == "lp" and options.ranges
        solver_options: dict[str, Any] = {
            "solve": dict(
                _solver.QP_OPTIONS if amounts.route == "qp" else _solver.LP_OPTIONS
            ),
            "ranging": dict(_solver.RANGING_OPTIONS) if ranging else None,
        }
        message = "" if solves.last is None else solves.last.message
    else:
        outcome = _grid(template, free, context, solves, inner=False)
        cover, ties, profile, counts = (
            outcome.cover,
            outcome.ties,
            outcome.profile,
            outcome.counts,
        )
        amounts = None
        route, amounts_route = "grid", "grid"
        unique = False if ties else None
        solver, solver_options, message = None, {}, "best grid candidate"
    _, objective_value, premium = _verify(context, cover, np.zeros(0), None, None)
    p_zero, tail, flag = _tail_flags(objective, loss, probability)
    values = cover.payout.parameters()
    parameters = {name: float(cast(float, values[name])) + 0.0 for name in free}
    budget_document = None if budget is None else budget.to_document()
    limit_document = None if limit is None else limit.to_document()
    design_id = content_hash(
        {
            "budget": budget_document,
            "limit": limit_document,
            "objective": objective.to_document(),
            "options": options.to_document(),
            "pricing": pricing.to_document(),
            "sample_hash": sample.content_hash,
            "semantics_version": DESIGN_SEMANTICS_VERSION,
            "template_hash": template.definition_hash,
        }
    )
    return DesignResult(
        cover=cover,
        template_hash=template.definition_hash,
        design_id=design_id,
        parameters=parameters,
        objective=objective.to_document(),
        objective_value=objective_value,
        pricing=pricing.to_document(),
        budget=budget_document,
        limit=limit_document,
        options=options,
        premium=premium,
        expected_loss=expected_loss,
        budget_bound=bound,
        budget_use=None if bound is None or bound <= 0.0 else premium / bound,
        route=route,
        amounts=amounts_route,
        solver=solver,
        status="optimal",
        solver_message=message,
        iterations=solves.iterations,
        solves=solves.count,
        seconds=solves.seconds,
        solver_options=solver_options,
        unidentified=() if amounts is None else amounts.unidentified,
        fill=options.fill,
        zeroed=() if amounts is None else amounts.zeroed,
        ranges=None if amounts is None else amounts.ranges,
        unique=unique,
        ties=ties,
        profile=profile,
        grid=counts,
        p_zero_loss=p_zero,
        tail_outcomes=tail,
        degenerate_tail=flag,
        design_sample=sample,
    )
