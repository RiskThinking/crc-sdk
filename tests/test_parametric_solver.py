"""The private solver interface and its HiGHS backend."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from crc_sdk.parametric import SolverUnavailableError, _solver


def _matrix(dense: list[list[float]]) -> _solver.SparseColumns:
    array = np.asarray(dense, dtype=np.float64)
    rows, columns = np.nonzero(array)
    return _solver.sparse_columns(rows, columns, array[rows, columns], array.shape)


def _lp(
    cost: list[float],
    dense: list[list[float]],
    upper_rows: list[float],
    *,
    col_upper: float = np.inf,
) -> _solver.Programme:
    n = len(cost)
    return _solver.Programme(
        cost=np.asarray(cost),
        col_lower=np.zeros(n),
        col_upper=np.full(n, col_upper),
        matrix=_matrix(dense),
        row_lower=np.full(len(upper_rows), -np.inf),
        row_upper=np.asarray(upper_rows),
    )


def test_sparse_columns() -> None:
    matrix = _matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])
    assert matrix.start.tolist() == [0, 1, 2, 4]
    assert matrix.index.tolist() == [0, 1, 0, 1]
    assert matrix.value.tolist() == [1.0, 3.0, 2.0, 4.0]
    assert matrix.n_columns == 3 and matrix.n_rows == 2
    with pytest.raises(ValueError, match="twice"):
        _solver.sparse_columns(
            np.array([0, 0]), np.array([1, 1]), np.array([1.0, 2.0]), (1, 2)
        )
    with pytest.raises(ValueError):
        _solver.SparseColumns(
            start=np.array([0, 1]), index=np.array([3]), value=np.array([1.0]), n_rows=2
        )
    with pytest.raises(ValueError):
        _solver.SparseColumns(
            start=np.array([1, 1]), index=np.array([0]), value=np.array([1.0]), n_rows=2
        )


def test_programme_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="one column"):
        _solver.Programme(
            cost=np.zeros(3),
            col_lower=np.zeros(3),
            col_upper=np.zeros(3),
            matrix=_matrix([[1.0, 1.0]]),
            row_lower=np.zeros(1),
            row_upper=np.zeros(1),
        )
    with pytest.raises(ValueError, match="finite"):
        _lp([np.nan, 1.0], [[1.0, 1.0]], [1.0])


def test_an_lp() -> None:
    # max x0 + 2 x1 subject to x0 + x1 <= 1: x = (0, 1).
    solution = _solver.solve_lp(_lp([-1.0, -2.0], [[1.0, 1.0]], [1.0]))
    assert solution.status == "optimal"
    assert solution.values is not None
    assert solution.values.tolist() == pytest.approx([0.0, 1.0])
    assert solution.objective == pytest.approx(-2.0)
    assert solution.message == "Optimal"
    assert solution.iterations >= 0 and solution.seconds >= 0.0


def test_infeasible_and_unbounded_lps() -> None:
    infeasible = _solver.Programme(
        cost=np.array([1.0]),
        col_lower=np.zeros(1),
        col_upper=np.full(1, np.inf),
        matrix=_matrix([[1.0]]),
        row_lower=np.array([2.0]),
        row_upper=np.array([1.0]),
    )
    solution = _solver.solve_lp(infeasible)
    assert solution.status in ("infeasible", "infeasible_or_unbounded")
    assert solution.values is None and solution.objective is None
    unbounded = _solver.solve_lp(_lp([-1.0, 0.0], [[0.0, 1.0]], [1.0]))
    assert unbounded.status in ("unbounded", "infeasible_or_unbounded")
    assert unbounded.values is None
    off = _solver.solve_lp(_lp([-1.0, 0.0], [[0.0, 1.0]], [1.0]), {"presolve": "off"})
    assert off.status == "unbounded"


def test_a_qp() -> None:
    # min x0^2 + x1^2 - 2 x0 - 2 x1 subject to x0 + x1 <= 1: x = (0.5, 0.5).
    hessian = _solver.sparse_columns(
        np.array([0, 1]), np.array([0, 1]), np.array([2.0, 2.0]), (2, 2)
    )
    programme = _solver.Programme(
        cost=np.array([-2.0, -2.0]),
        col_lower=np.zeros(2),
        col_upper=np.full(2, np.inf),
        matrix=_matrix([[1.0, 1.0]]),
        row_lower=np.array([-np.inf]),
        row_upper=np.array([1.0]),
        offset=2.0,
        hessian=hessian,
    )
    solution = _solver.solve_qp(programme)
    assert solution.status == "optimal"
    assert solution.values is not None
    assert solution.values.tolist() == pytest.approx([0.5, 0.5], abs=1e-9)
    assert solution.objective == pytest.approx(0.5)
    with pytest.raises(ValueError, match="hessian"):
        _solver.solve_lp(programme)
    with pytest.raises(ValueError, match="hessian"):
        _solver.solve_qp(_lp([1.0], [[1.0]], [1.0]))


def test_rejected_options_raise() -> None:
    with pytest.raises(ValueError, match="rejected the option"):
        _solver.solve_lp(_lp([1.0], [[1.0]], [1.0]), {"no_such_option": 1})


def test_identity() -> None:
    identity = _solver.identity()
    assert identity["name"] == "highs"
    major, minor, *_ = (int(part) for part in identity["version"].split("."))
    assert (major, minor) >= (1, 13)


def test_a_missing_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "highspy", None)
    with pytest.raises(SolverUnavailableError) as raised:
        _solver.require()
    assert raised.value.extra == "parametric"
    assert "pip install crc-sdk[parametric]" in str(raised.value)
    with pytest.raises(ImportError):
        _solver.solve_lp(_lp([1.0], [[1.0]], [1.0]))


def test_importing_the_interface_loads_no_solver() -> None:
    code = (
        "import sys, crc_sdk.parametric._solver, crc_sdk.parametric.design; "
        "print('highspy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
