"""The private solver interface of `design`, and its HiGHS backend.

`solve_lp` and `solve_qp` take a `Programme`: plain numpy arrays, with the
constraint matrix and the Hessian in compressed sparse column form
(`SparseColumns`: column pointers, row indices, values). No scipy. They
return a `Solution`: the status, the values, the objective and the solver's
own message, iterations and time. `identity()` names the solver and its
version for the design record. So the backend can change without a change
of the public API.

The backend is HiGHS through `highspy`, from the optional extra
`crc-sdk[parametric]`. It is imported lazily: importing this module needs no
solver, and `require()` raises `SolverUnavailableError`, naming the extra,
when it is missing.

A programme is `minimise 1/2 x'Qx + c'x + offset` subject to `row_lower <=
A x <= row_upper` and `col_lower <= x <= col_upper`, with infinite bounds as
`inf`. `Q` is given as its lower triangle; an LP has none.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from crc_sdk.parametric.errors import SolverUnavailableError

__all__ = [
    "EXTRA",
    "LP_OPTIONS",
    "QP_OPTIONS",
    "RANGING_OPTIONS",
    "SOLVER_NAME",
    "Programme",
    "Solution",
    "SolveStatus",
    "SparseColumns",
    "identity",
    "require",
    "solve_lp",
    "solve_qp",
    "sparse_columns",
]

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int32]

SOLVER_NAME = "highs"
"""The solver's name in design records."""

EXTRA = "parametric"
"""The optional dependency set that installs the solver."""

LP_OPTIONS: Mapping[str, bool | int | float | str] = {"solver": "simplex"}
"""HiGHS options of a design LP: the simplex, so a solution is a vertex."""

QP_OPTIONS: Mapping[str, bool | int | float | str] = {"qp_regularization_value": 1e-12}
"""HiGHS options of a design QP.

HiGHS adds `qp_regularization_value` (1e-7 by default) to the Hessian's
diagonal, which moves a solution by about that share of its scale; 1e-12
keeps the semi-definite design QPs solvable without that bias.
"""

RANGING_OPTIONS: Mapping[str, bool | int | float | str] = {
    "solver": "simplex",
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}
"""HiGHS options of the LPs that range an optimal set (its defaults are 1e-7)."""

SolveStatus = Literal[
    "optimal", "infeasible", "unbounded", "infeasible_or_unbounded", "error"
]
"""A solve's outcome; only `"optimal"` comes with values."""


def require() -> ModuleType:
    """Return the `highspy` module, or raise `SolverUnavailableError`.

    The error is an `ImportError` that names the extra to install.
    """
    try:
        import highspy
    except ImportError as error:
        raise SolverUnavailableError(
            "designing a cover needs the HiGHS solver: "
            f"`pip install crc-sdk[{EXTRA}]` (it installs highspy)",
            extra=EXTRA,
        ) from error
    return highspy


def identity() -> dict[str, str]:
    """Return `{"name": "highs", "version": ...}` of the installed solver."""
    highspy = require()
    return {"name": SOLVER_NAME, "version": str(highspy.Highs().version())}


@dataclass(frozen=True, eq=False)
class SparseColumns:
    """A matrix in compressed sparse column form, as plain numpy arrays.

    Column `j` holds `value[start[j]:start[j + 1]]` in the rows
    `index[start[j]:start[j + 1]]`. `n_rows` is the number of rows.
    """

    start: IndexArray
    index: IndexArray
    value: FloatArray
    n_rows: int

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=np.int32)
        index = np.asarray(self.index, dtype=np.int32)
        value = np.asarray(self.value, dtype=np.float64)
        if start.ndim != 1 or start.shape[0] == 0 or start[0] != 0:
            raise ValueError("start must be a 1-D array that begins with 0")
        if (np.diff(start) < 0).any() or start[-1] != index.shape[0]:
            raise ValueError("start must be non-decreasing and end at the entries")
        if index.shape != value.shape or index.ndim != 1:
            raise ValueError("index and value must be 1-D arrays of one length")
        if index.size and (index.min() < 0 or index.max() >= self.n_rows):
            raise ValueError("row indices must lie in [0, n_rows)")
        if not np.isfinite(value).all():
            raise ValueError("matrix values must be finite")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "value", value)

    @property
    def n_columns(self) -> int:
        """The number of columns."""
        return int(self.start.shape[0] - 1)


def sparse_columns(
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    values: FloatArray,
    shape: tuple[int, int],
) -> SparseColumns:
    """Return the matrix of the entries `(rows[i], columns[i], values[i])`.

    Zero values are dropped; repeated positions are not allowed.
    """
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    keep = values != 0.0
    rows, columns, values = rows[keep], columns[keep], values[keep]
    order = np.lexsort((rows, columns))
    rows, columns, values = rows[order], columns[order], values[order]
    if rows.size > 1:
        same = (np.diff(rows) == 0) & (np.diff(columns) == 0)
        if same.any():
            raise ValueError("a matrix position is given twice")
    counts = np.bincount(columns, minlength=shape[1])
    start = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return SparseColumns(
        start=start,
        index=rows.astype(np.int32),
        value=values,
        n_rows=shape[0],
    )


@dataclass(frozen=True, eq=False)
class Programme:
    """`min 1/2 x'Qx + c'x + offset` s.t. `row_lower <= A x <= row_upper`, bounds.

    `cost` is `c`, one entry per column; `matrix` is `A`; `hessian` is the
    lower triangle of `Q` (square, one row and column per variable), or
    `None` for an LP. Infinite bounds are `inf` or `-inf`.
    """

    cost: FloatArray
    col_lower: FloatArray
    col_upper: FloatArray
    matrix: SparseColumns
    row_lower: FloatArray
    row_upper: FloatArray
    offset: float = 0.0
    hessian: SparseColumns | None = None

    def __post_init__(self) -> None:
        n = int(np.asarray(self.cost).shape[0])
        for name in ("cost", "col_lower", "col_upper"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (n,):
                raise ValueError(f"{name} must have one entry per column")
            object.__setattr__(self, name, array)
        if self.matrix.n_columns != n:
            raise ValueError("the matrix must have one column per variable")
        for name in ("row_lower", "row_upper"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (self.matrix.n_rows,):
                raise ValueError(f"{name} must have one entry per row")
            object.__setattr__(self, name, array)
        if not np.isfinite(self.cost).all() or not math.isfinite(self.offset):
            raise ValueError("cost and offset must be finite")
        if self.hessian is not None and (
            self.hessian.n_columns != n or self.hessian.n_rows != n
        ):
            raise ValueError("the hessian must be square, one row per variable")

    @property
    def n_columns(self) -> int:
        """The number of variables."""
        return int(self.cost.shape[0])


@dataclass(frozen=True, eq=False)
class Solution:
    """The outcome of one solve.

    `status` is `"optimal"` with `values` (one per variable) and `objective`
    (including the offset); otherwise both are `None`. `message` is the
    solver's own status text, `iterations` its iteration count (simplex,
    interior point and QP iterations together) and `seconds` the wall time.
    """

    status: SolveStatus
    values: FloatArray | None
    objective: float | None
    message: str
    iterations: int
    seconds: float


def _options(
    highs: Any, base: Mapping[str, Any], options: Mapping[str, Any] | None
) -> None:
    chosen = {"output_flag": False, **base, **(options or {})}
    for name, value in chosen.items():
        status = highs.setOptionValue(name, value)
        if status != require().HighsStatus.kOk:
            raise ValueError(f"HiGHS rejected the option {name}={value!r}")


def _run(
    programme: Programme, base: Mapping[str, Any], options: Mapping[str, Any] | None
) -> Solution:
    highspy = require()
    highs = highspy.Highs()
    _options(highs, base, options)
    matrix = programme.matrix
    hessian = programme.hessian
    integrality = np.zeros(programme.n_columns, dtype=np.int32)
    colwise = int(highspy.MatrixFormat.kColwise)
    minimise = int(highspy.ObjSense.kMinimize)
    common = (
        programme.cost,
        programme.col_lower,
        programme.col_upper,
        programme.row_lower,
        programme.row_upper,
        matrix.start,
        matrix.index,
        matrix.value,
    )
    if hessian is None:
        passed = highs.passModel(
            programme.n_columns,
            matrix.n_rows,
            int(matrix.index.shape[0]),
            colwise,
            minimise,
            float(programme.offset),
            *common,
            integrality,
        )
    else:
        passed = highs.passModel(
            programme.n_columns,
            matrix.n_rows,
            int(matrix.index.shape[0]),
            int(hessian.index.shape[0]),
            colwise,
            int(highspy.HessianFormat.kTriangular),
            minimise,
            float(programme.offset),
            *common,
            hessian.start,
            hessian.index,
            hessian.value,
            integrality,
        )
    if passed == highspy.HighsStatus.kError:
        return Solution("error", None, None, "HiGHS rejected the model", 0, 0.0)
    began = time.perf_counter()
    ran = highs.run()
    seconds = time.perf_counter() - began
    model_status = highs.getModelStatus()
    message = str(highs.modelStatusToString(model_status))
    info = highs.getInfo()
    iterations = sum(
        max(0, int(getattr(info, name)))
        for name in (
            "simplex_iteration_count",
            "ipm_iteration_count",
            "qp_iteration_count",
            "crossover_iteration_count",
        )
    )
    statuses: dict[Any, SolveStatus] = {
        highspy.HighsModelStatus.kOptimal: "optimal",
        highspy.HighsModelStatus.kInfeasible: "infeasible",
        highspy.HighsModelStatus.kUnbounded: "unbounded",
        highspy.HighsModelStatus.kUnboundedOrInfeasible: "infeasible_or_unbounded",
    }
    status: SolveStatus = statuses.get(model_status, "error")
    if ran == highspy.HighsStatus.kError:
        status = "error"
    if status != "optimal":
        return Solution(status, None, None, message, iterations, seconds)
    values = np.asarray(highs.getSolution().col_value, dtype=np.float64)
    objective = float(info.objective_function_value)
    if values.shape != (programme.n_columns,) or not np.isfinite(values).all():
        return Solution(
            "error",
            None,
            None,
            f"{message}, without finite values",
            iterations,
            seconds,
        )
    return Solution(status, values, objective, message, iterations, seconds)


def solve_lp(
    programme: Programme, options: Mapping[str, bool | int | float | str] | None = None
) -> Solution:
    """Solve a linear programme with HiGHS's simplex.

    `options` are HiGHS options on top of `LP_OPTIONS`. Raises
    `SolverUnavailableError` without highspy, and `ValueError` for a
    programme with a Hessian or an option HiGHS rejects.
    """
    if programme.hessian is not None:
        raise ValueError("solve_lp takes a programme without a hessian")
    return _run(programme, LP_OPTIONS, options)


def solve_qp(
    programme: Programme, options: Mapping[str, bool | int | float | str] | None = None
) -> Solution:
    """Solve a convex quadratic programme with HiGHS.

    The Hessian must be positive semi-definite. `options` are HiGHS options
    on top of `QP_OPTIONS`. Raises `SolverUnavailableError` without highspy,
    and `ValueError` for a programme without a Hessian or an option HiGHS
    rejects.
    """
    if programme.hessian is None:
        raise ValueError("solve_qp takes a programme with a hessian")
    return _run(programme, QP_OPTIONS, options)
