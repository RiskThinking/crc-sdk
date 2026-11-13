"""Payout families: values at and around thresholds, bases, constraints."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from crc_sdk.parametric import (
    BinaryPayout,
    CoverDefinitionError,
    Free,
    LinearConstraints,
    LinearInAmounts,
    LinearPayout,
    PayoutFunction,
    PiecewiseLinearPayout,
    SampleError,
    StepPayout,
    UnknownPayoutKindError,
    UnresolvedParameterError,
    register_payout,
)
from crc_sdk.parametric.payouts import _REGISTRY, _payout_reader

FloatArray = NDArray[np.float64]
UNTYPED_STEP: Any = StepPayout  # for calls with deliberately invalid arguments


def below(value: float) -> float:
    """The largest float below `value`."""
    return float(np.nextafter(value, -math.inf))


def above(value: float) -> float:
    """The smallest float above `value`."""
    return float(np.nextafter(value, math.inf))


def pay(payout: Any, values: list[float]) -> list[float]:
    result: list[float] = payout.evaluate(np.asarray(values, dtype=np.float64)).tolist()
    return result


# -- step ------------------------------------------------------------------------

# The core's StepPayout(20, 70, 5, below_attachment="zero") has the edges 20, 30,
# 40, 50, 60, 70; values at or above 70 fall into the last bin, so 70 changes no
# payout and the step is StepPayout(breakpoints=edges[:-1], levels=theta).
STEP = StepPayout([20.0, 30.0, 40.0, 50.0, 60.0], [1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(-50.0, 0.0, id="negative-below-attachment"),
        pytest.param(0.0, 0.0, id="no-hazard"),
        pytest.param(17.5, 0.0, id="below-attachment"),
        pytest.param(below(20.0), 0.0, id="just-below-attachment"),
        pytest.param(20.0, 1.0, id="at-attachment"),
        pytest.param(above(20.0), 1.0, id="just-above-attachment"),
        pytest.param(25.0, 1.0, id="inside-first-band"),
        pytest.param(below(30.0), 1.0, id="just-below-interior-breakpoint"),
        pytest.param(30.0, 2.0, id="at-interior-breakpoint-is-left-closed"),
        pytest.param(above(30.0), 2.0, id="just-above-interior-breakpoint"),
        pytest.param(50.0, 4.0, id="at-middle-breakpoint"),
        pytest.param(59.9, 4.0, id="inside-band-3"),
        pytest.param(below(60.0), 4.0, id="just-below-last-breakpoint"),
        pytest.param(60.0, 5.0, id="at-last-breakpoint"),
        pytest.param(70.0, 5.0, id="at-the-core-limit"),
        pytest.param(1e9, 5.0, id="far-above"),
    ],
)
def test_step_bands_are_half_open(value: float, expected: float) -> None:
    assert pay(STEP, [value]) == [expected]
    row = STEP.basis(np.array([value]))[0]
    band = int(expected) - 1
    assert row.tolist() == [1.0 if k == band else 0.0 for k in range(5)]


def test_step_by_hand_with_explicit_breakpoints() -> None:
    step = StepPayout([20, 25, 40], [0.5, 1.0, 2.0])
    values = [0.0, 19.9, 20.0, 24.9, 25.0, 39.9, 40.0, 70.0, 80.0]
    assert pay(step, values) == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 2.0]


def test_a_single_band_pays_its_level_from_the_attachment_on() -> None:
    step = StepPayout([20.0], [7.0])
    assert pay(step, [-1.0, 19.0, 20.0, 45.0, 1e6]) == [0.0, 0.0, 7.0, 7.0, 7.0]
    assert step.basis(np.array([0.0, 99.0])).tolist() == [[0.0], [1.0]]


def test_step_with_a_negative_attachment() -> None:
    step = StepPayout([-10.0, 0.0], [1.0, 2.0])
    assert pay(step, [-10.5, -10.0, -0.1, 0.0, 12.0]) == [0.0, 1.0, 1.0, 2.0, 2.0]


def test_lower_tail_step() -> None:
    step = StepPayout([-1.0, -1.5, -2.0], [1.0, 2.0, 3.0], tail="lower")
    values = [0.5, -0.5, above(-1.0), -1.0, -1.2, below(-1.5), -1.5, -2.0, -5.0]
    assert pay(step, values) == [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_non_monotone_levels_are_allowed_when_stated() -> None:
    step = StepPayout([1.0, 2.0, 3.0], [5.0, 1.0, 3.0], monotone=False)
    assert pay(step, [0.0, 1.0, 2.0, 3.0]) == [0.0, 5.0, 1.0, 3.0]
    assert step.amount_constraints().a_ub.shape == (0, 3)


# -- linear ----------------------------------------------------------------------

LINEAR = LinearPayout(60.0, 100.0, 1000.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.0, 0.0, id="no-hazard"),
        pytest.param(59.0, 0.0, id="below-attachment"),
        pytest.param(60.0, 0.0, id="at-attachment"),
        pytest.param(70.0, 250.0, id="quarter"),
        pytest.param(80.0, 500.0, id="half"),
        pytest.param(100.0, 1000.0, id="at-exhaustion"),
        pytest.param(120.0, 1000.0, id="beyond-exhaustion"),
    ],
)
def test_linear_values(value: float, expected: float) -> None:
    assert pay(LINEAR, [value]) == [expected]


def test_linear_is_strictly_inside_its_range_between_the_thresholds() -> None:
    low, high = pay(LINEAR, [above(60.0), below(100.0)])
    assert 0.0 < low < 1e-9
    assert 1000.0 - 1e-9 < high < 1000.0


def test_lower_tail_linear() -> None:
    linear = LinearPayout(-1.0, -2.0, 100.0, tail="lower")
    values = [0.0, -1.0, -1.5, -2.0, -3.0]
    assert pay(linear, values) == [0.0, 0.0, 50.0, 100.0, 100.0]


# -- piecewise linear ------------------------------------------------------------

PIECEWISE = PiecewiseLinearPayout([10.0, 20.0, 40.0], [100.0, 200.0, 600.0])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.0, 0.0, id="no-hazard"),
        pytest.param(below(10.0), 0.0, id="just-below-first-knot"),
        pytest.param(10.0, 100.0, id="at-first-knot-jumps-to-first-level"),
        pytest.param(15.0, 150.0, id="first-segment"),
        pytest.param(20.0, 200.0, id="at-interior-knot"),
        pytest.param(30.0, 400.0, id="second-segment"),
        pytest.param(40.0, 600.0, id="at-last-knot"),
        pytest.param(50.0, 600.0, id="beyond-last-knot"),
    ],
)
def test_piecewise_linear_values(value: float, expected: float) -> None:
    assert pay(PIECEWISE, [value]) == pytest.approx([expected], rel=1e-15)


def test_piecewise_linear_is_continuous_at_interior_knots() -> None:
    left, at, right = pay(PIECEWISE, [below(20.0), 20.0, above(20.0)])
    assert left == pytest.approx(200.0) and right == pytest.approx(200.0)
    assert left <= at <= right


def test_piecewise_linear_flat_segment_is_exact() -> None:
    flat = PiecewiseLinearPayout([1.0, 2.0, 4.0], [0.3, 0.3, 0.9])
    values = np.linspace(1.0, 2.0, 101)
    assert pay(flat, values.tolist()) == [0.3] * 101


def test_lower_tail_piecewise_linear() -> None:
    piecewise = PiecewiseLinearPayout(
        [0.0, -1.0, -2.0], [0.0, 10.0, 30.0], tail="lower"
    )
    values = [0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0]
    assert pay(piecewise, values) == pytest.approx([0, 0, 5, 10, 20, 30, 30])


# -- binary ----------------------------------------------------------------------


def test_binary_values() -> None:
    binary = BinaryPayout(50.0, 10.0)
    assert pay(binary, [0.0, 49.999, below(50.0), 50.0, 60.0]) == [0, 0, 0, 10, 10]
    lower = BinaryPayout(-2.0, 10.0, tail="lower")
    assert pay(lower, [0.0, -1.9, above(-2.0), -2.0, -3.0]) == [0, 0, 0, 10, 10]


# -- properties of every family --------------------------------------------------

UPPER = [
    LINEAR,
    STEP,
    PIECEWISE,
    BinaryPayout(50.0, 10.0),
    PiecewiseLinearPayout([10.0, 20.0, 40.0], [0.0, 0.0, 5.0]),
    StepPayout([20.0, 30.0], [3.0, 3.0]),
]


def mirror(payout: Any) -> Any:
    """The same payout on the negated index, with tail="lower"."""
    document = payout.to_document()
    for field in ("attachment", "exhaustion", "trigger"):
        if field in document:
            document[field] = -document[field]
    for field in ("breakpoints", "knots"):
        if field in document:
            document[field] = [-value for value in document[field]]
    document["tail"] = "lower"
    return type(payout).from_document(document)


def grid_with_thresholds(payout: Any) -> FloatArray:
    rng = np.random.default_rng(246)
    names = ("attachment", "exhaustion", "trigger", "breakpoints", "knots")
    thresholds = [
        value
        for name, value in payout.parameters().items()
        if name.split("[")[0] in names
    ]
    near = [f(t) for t in thresholds for f in (below, lambda x: x, above)]
    return np.sort(np.concatenate([rng.uniform(-20.0, 150.0, 2000), near]))


@pytest.mark.parametrize("payout", UPPER, ids=lambda p: p.kind)
def test_payouts_are_monotone_and_start_at_zero(payout: Any) -> None:
    index = grid_with_thresholds(payout)
    paid = payout.evaluate(index)
    assert np.all(np.diff(paid) >= 0)
    assert paid[0] == 0.0
    assert paid[-1] == max(payout.amount_constraints().upper)
    # The lower-tail mirror is monotone in the other direction.
    mirrored = mirror(payout).evaluate(-index)
    np.testing.assert_array_equal(mirrored, paid)


@pytest.mark.parametrize("payout", UPPER, ids=lambda p: p.kind)
def test_evaluate_is_basis_times_amounts(payout: Any) -> None:
    index = grid_with_thresholds(payout)
    basis = payout.basis(index)
    amounts = np.array([payout.parameters()[name] for name in payout.amount_names()])
    assert basis.shape == (index.shape[0], len(payout.amount_names()))
    assert basis.dtype == np.float64
    assert np.all(basis >= 0)
    assert np.all(basis.sum(axis=1) <= 1.0 + 1e-15)
    np.testing.assert_allclose(payout.evaluate(index), basis @ amounts, rtol=1e-14)
    if not isinstance(payout, PiecewiseLinearPayout):
        np.testing.assert_array_equal(payout.evaluate(index), basis @ amounts)


@pytest.mark.parametrize("payout", UPPER, ids=lambda p: p.kind)
def test_evaluate_keeps_the_shape(payout: Any) -> None:
    index = np.array([[0.0, 35.0, 70.0], [20.0, 59.9, 1e3]])
    paid = payout.evaluate(index)
    assert paid.shape == (2, 3)
    assert paid.dtype == np.float64
    np.testing.assert_array_equal(paid.reshape(-1), payout.evaluate(index.reshape(-1)))
    assert payout.evaluate(np.float64(45.0)).shape == ()
    assert payout.evaluate(np.array([], dtype=np.float64)).shape == (0,)
    assert payout.basis(np.array([], dtype=np.float64)).shape == (
        0,
        len(payout.amount_names()),
    )


@pytest.mark.parametrize("payout", UPPER, ids=lambda p: p.kind)
def test_built_in_families_meet_both_protocols(payout: Any) -> None:
    assert isinstance(payout, PayoutFunction)
    assert isinstance(payout, LinearInAmounts)
    assert payout.index_names == ("index",)
    assert _REGISTRY[payout.kind] is type(payout)


def test_protocols_type_check() -> None:
    linear: LinearInAmounts = LinearPayout(1.0, 2.0, 3.0)
    step: LinearInAmounts = StepPayout([1.0], [1.0])
    piecewise: LinearInAmounts = PiecewiseLinearPayout([1.0, 2.0], [1.0, 2.0])
    binary: PayoutFunction = BinaryPayout(1.0, 1.0)
    assert [p.kind for p in (linear, step, piecewise, binary)] == [
        "linear",
        "step",
        "piecewise_linear",
        "binary",
    ]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("payout", UPPER, ids=lambda p: p.kind)
def test_non_finite_index_values_raise(payout: Any, bad: float) -> None:
    with pytest.raises(SampleError, match="finite"):
        payout.evaluate(np.array([30.0, bad]))
    with pytest.raises(SampleError, match="finite"):
        payout.basis(np.array([30.0, bad]))


def test_non_numeric_index_values_and_wrong_shapes_raise() -> None:
    with pytest.raises(SampleError, match="numeric"):
        STEP.evaluate(["thirty"])  # type: ignore[arg-type]
    with pytest.raises(SampleError, match="1-D"):
        STEP.basis(np.zeros((2, 1)))
    with pytest.raises(ValueError):
        STEP.evaluate(np.array([math.nan]))


# -- parameters ------------------------------------------------------------------


def test_parameter_names_are_flat_and_ordered() -> None:
    assert list(LINEAR.parameters()) == ["attachment", "exhaustion", "limit"]
    assert list(StepPayout([1.0, 2.0], [3.0, 4.0]).parameters()) == [
        "breakpoints[0]",
        "breakpoints[1]",
        "levels[0]",
        "levels[1]",
    ]
    assert list(PIECEWISE.parameters())[:3] == ["knots[0]", "knots[1]", "knots[2]"]
    assert list(BinaryPayout(1.0, 2.0).parameters()) == ["trigger", "amount"]
    assert LINEAR.amount_names() == ("limit",)
    assert STEP.amount_names() == tuple(f"levels[{k}]" for k in range(5))
    assert BinaryPayout(1.0, 2.0).amount_names() == ("amount",)


def test_parameters_accept_integers_and_numpy_values() -> None:
    step = StepPayout(np.array([1, 2]), [np.float64(1.0), np.int64(2)])  # type: ignore[list-item]
    assert step.breakpoints == (1.0, 2.0)
    assert step.levels == (1.0, 2.0)
    assert all(type(value) is float for value in step.parameters().values())


TEMPLATE = StepPayout(
    [Free(10.0, 30.0, grid=[10.0, 20.0, 30.0]), 40.0, 60.0],
    [Free(), Free(), Free(upper=500.0)],
)


def test_with_values_resolves_free_parameters() -> None:
    resolved = TEMPLATE.with_values(
        {"breakpoints[0]": 20.0, "levels[0]": 1.0, "levels[1]": 2.0}
    )
    assert resolved.breakpoints == (20.0, 40.0, 60.0)
    assert resolved.levels[:2] == (1.0, 2.0)
    assert resolved.levels[2] == Free(upper=500.0)
    assert TEMPLATE.breakpoints[0] == Free(10.0, 30.0, grid=[10.0, 20.0, 30.0])
    full = resolved.with_values({"levels[2]": 500.0})
    assert pay(full, [19.0, 20.0, 45.0, 70.0]) == [0.0, 1.0, 2.0, 500.0]


def test_with_values_resolves_scalar_parameters() -> None:
    linear = LinearPayout(Free(20.0, 40.0, grid=5), 50.0, Free(upper=1e6))
    assert list(linear.parameters()) == ["attachment", "exhaustion", "limit"]
    resolved = linear.with_values({"attachment": 30.0, "limit": 1e5})
    assert (resolved.attachment, resolved.limit) == (30.0, 1e5)
    assert pay(resolved, [30.0, 40.0, 50.0]) == [0.0, 5e4, 1e5]
    binary = BinaryPayout(Free(-3.0, -1.0, grid=3), Free(), tail="lower")
    resolved_binary = binary.with_values({"trigger": -2.0, "amount": 2.0})
    assert pay(resolved_binary, [-1.0, -2.0, -2.5]) == [0.0, 2.0, 2.0]
    with pytest.raises(CoverDefinitionError, match="strictly increasing"):
        LinearPayout(Free(20.0, 60.0, grid=[20.0, 60.0]), 50.0, 1.0).with_values(
            {"attachment": 55.0}
        )


def test_with_values_accepts_values_off_the_grid_within_the_bounds() -> None:
    assert TEMPLATE.with_values({"breakpoints[0]": 15.5}).breakpoints[0] == 15.5


@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param({"breakpoints[3]": 1.0}, "unknown parameter", id="unknown"),
        pytest.param({"breakpoints[1]": 45.0}, "is fixed", id="fixed"),
        pytest.param({"breakpoints[0]": 35.0}, "outside", id="outside"),
        pytest.param({"levels[2]": 501.0}, "outside", id="above-upper"),
        pytest.param({"levels[0]": -1.0}, "outside", id="negative"),
        pytest.param(
            {"levels[0]": 5.0, "levels[1]": 4.0}, "non-decreasing", id="order"
        ),
    ],
)
def test_with_values_validates(values: dict[str, float], message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        TEMPLATE.with_values(values)


def test_resolved_thresholds_must_be_ordered() -> None:
    template = StepPayout([Free(0.0, 100.0, grid=[10.0, 50.0]), 40.0], [1.0, 2.0])
    assert template.with_values({"breakpoints[0]": 10.0}).breakpoints == (10.0, 40.0)
    with pytest.raises(CoverDefinitionError, match="strictly increasing"):
        template.with_values({"breakpoints[0]": 50.0})


def test_templates_do_not_evaluate() -> None:
    with pytest.raises(UnresolvedParameterError, match="levels"):
        TEMPLATE.evaluate(np.array([1.0]))
    with pytest.raises(UnresolvedParameterError, match="breakpoints"):
        TEMPLATE.basis(np.array([1.0]))
    # Free amounts leave the basis defined.
    amounts_free = TEMPLATE.with_values({"breakpoints[0]": 20.0})
    assert amounts_free.basis(np.array([25.0, 50.0])).tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    with pytest.raises(UnresolvedParameterError):
        LinearPayout(1.0, 2.0, Free()).evaluate(np.array([1.0]))


# -- amount constraints ----------------------------------------------------------


def test_amount_constraints_of_a_monotone_step() -> None:
    constraints = TEMPLATE.amount_constraints()
    assert constraints.n_amounts == 3
    np.testing.assert_array_equal(
        constraints.a_ub, [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]
    )
    np.testing.assert_array_equal(constraints.b_ub, [0.0, 0.0])
    np.testing.assert_array_equal(constraints.lower, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(constraints.upper, [math.inf, math.inf, 500.0])
    capped = TEMPLATE.amount_constraints(limit=300)
    np.testing.assert_array_equal(capped.upper, [300.0, 300.0, 300.0])
    assert not constraints.a_ub.flags.writeable


def test_fixed_amounts_are_bounded_on_both_sides() -> None:
    constraints = StepPayout([1.0, 2.0], [Free(1.0), 4.0]).amount_constraints(3.0)
    np.testing.assert_array_equal(constraints.lower, [1.0, 4.0])
    np.testing.assert_array_equal(constraints.upper, [3.0, 3.0])


def test_amount_constraints_of_single_amount_families() -> None:
    for payout in (LINEAR, BinaryPayout(1.0, Free(2.0, 9.0))):
        constraints = payout.amount_constraints()
        assert constraints.a_ub.shape == (0, 1)
        assert constraints.b_ub.shape == (0,)


@pytest.mark.parametrize("limit", [-1.0, math.nan, math.inf, True, "1"])
def test_invalid_limits_raise(limit: Any) -> None:
    with pytest.raises(CoverDefinitionError, match="limit"):
        STEP.amount_constraints(limit)


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"lower": [0.0], "upper": [1.0, 2.0]}, id="upper"),
        pytest.param({"a_ub": np.zeros((1, 2))}, id="columns"),
        pytest.param({"b_ub": [0.0, 0.0]}, id="rows"),
        pytest.param({"lower": [math.nan]}, id="nan-lower"),
        pytest.param({"upper": [-math.inf]}, id="minus-inf-upper"),
        pytest.param({"upper": [math.nan]}, id="nan-upper"),
        pytest.param({"a_ub": [[math.inf]]}, id="inf-a"),
        pytest.param({"a_ub": [1.0]}, id="1-d-a"),
        pytest.param({"lower": ["x"]}, id="text"),
    ],
)
def test_linear_constraints_validate(arguments: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "a_ub": np.zeros((1, 1)),
        "b_ub": [0.0],
        "lower": [0.0],
        "upper": [math.inf],
    }
    values.update(arguments)
    with pytest.raises(CoverDefinitionError):
        LinearConstraints(**values)


# -- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(lambda: StepPayout([], []), "at least 1", id="step-empty"),
        pytest.param(
            lambda: StepPayout([20.0, 45.0], [1.0]), "one value per", id="step-short"
        ),
        pytest.param(
            lambda: StepPayout([20.0, 50.0, 40.0], [1.0, 2.0, 3.0]),
            "strictly increasing",
            id="step-down",
        ),
        pytest.param(
            lambda: StepPayout([20.0, 40.0, 40.0], [1.0, 2.0, 3.0]),
            "strictly increasing",
            id="step-tie",
        ),
        pytest.param(
            lambda: StepPayout([20.0, math.nan], [1.0, 2.0]), "finite", id="step-nan"
        ),
        pytest.param(
            lambda: UNTYPED_STEP(["a", "b"], [1.0, 2.0]), "number or Free", id="text"
        ),
        pytest.param(lambda: StepPayout([1.0], [True]), "number or Free", id="bool"),
        pytest.param(lambda: UNTYPED_STEP(20.0, [1.0]), "sequence", id="not-sequence"),
        pytest.param(lambda: UNTYPED_STEP("ab", [1.0]), "sequence", id="text-sequence"),
        pytest.param(
            lambda: StepPayout([1.0, 2.0], [2.0, 1.0]), "non-decreasing", id="levels"
        ),
        pytest.param(lambda: StepPayout([1.0], [-1.0]), ">= 0", id="negative-level"),
        pytest.param(
            lambda: StepPayout([1.0, 2.0], [1.0, 2.0], tail="lower"),
            "strictly decreasing",
            id="lower-tail-order",
        ),
        pytest.param(
            lambda: UNTYPED_STEP([1.0], [1.0], monotone=1), "monotone", id="monotone"
        ),
        pytest.param(lambda: UNTYPED_STEP([1.0], [1.0], tail="up"), "tail", id="tail"),
        pytest.param(
            lambda: PiecewiseLinearPayout([1.0], [1.0]), "at least 2", id="pw-one-knot"
        ),
        pytest.param(
            lambda: PiecewiseLinearPayout([1.0, 2.0], [1.0]),
            "one value per",
            id="pw-lengths",
        ),
        pytest.param(
            lambda: PiecewiseLinearPayout([2.0, 1.0], [1.0, 2.0]),
            "strictly increasing",
            id="pw-order",
        ),
        pytest.param(
            lambda: PiecewiseLinearPayout([1.0, 2.0], [3.0, 1.0]),
            "non-decreasing",
            id="pw-levels",
        ),
        pytest.param(
            lambda: LinearPayout(100.0, 60.0, 1.0), "strictly increasing", id="lin"
        ),
        pytest.param(
            lambda: LinearPayout(60.0, 60.0, 1.0), "strictly increasing", id="lin-tie"
        ),
        pytest.param(
            lambda: LinearPayout(-2.0, -1.0, 1.0, tail="lower"),
            "strictly decreasing",
            id="lin-lower",
        ),
        pytest.param(lambda: LinearPayout(1.0, 2.0, -1.0), ">= 0", id="lin-limit"),
        pytest.param(lambda: BinaryPayout(math.inf, 1.0), "finite", id="bin-trigger"),
        pytest.param(lambda: BinaryPayout(1.0, -1.0), ">= 0", id="bin-amount"),
        pytest.param(
            lambda: BinaryPayout(Free(0.0, 10.0), 1.0), "needs a finite", id="no-grid"
        ),
        pytest.param(
            lambda: BinaryPayout(Free(grid=[1.0]), 1.0), "needs a finite", id="no-upper"
        ),
        pytest.param(
            lambda: StepPayout([Free(50.0, 90.0, grid=[50.0, 90.0]), 40.0], [1.0, 2.0]),
            "no choice of the free values is ordered",
            id="no-ordered-choice",
        ),
        pytest.param(
            lambda: StepPayout([1.0], [Free(-1.0)]),
            "lower bound of levels",
            id="free-negative-lower",
        ),
        pytest.param(
            lambda: StepPayout([1.0, 2.0], [5.0, Free(0.0, 3.0)]),
            "non-decreasing",
            id="free-level-too-low",
        ),
        pytest.param(
            lambda: StepPayout([1.0, 2.0], [Free(4.0), 3.0]),
            "non-decreasing",
            id="free-level-too-high",
        ),
        pytest.param(
            lambda: StepPayout([1.0, 2.0], [Free(0.0, 10.0, grid=[6.0, 8.0]), 5.0]),
            "non-decreasing",
            id="free-grid-too-high",
        ),
    ],
)
def test_invalid_payouts_raise(build: Callable[[], Any], message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        build()


def test_feasible_templates_are_accepted() -> None:
    StepPayout([Free(0.0, 50.0, grid=[10.0, 45.0]), 40.0], [Free(), 3.0])
    StepPayout([1.0, 2.0, 3.0], [Free(), 3.0, Free()])
    StepPayout([1.0, 2.0], [Free(0.0, 10.0, grid=[2.0, 8.0]), 5.0])
    StepPayout(
        [Free(0.0, 30.0, grid=3), Free(0.0, 30.0, grid=3), Free(0.0, 30.0, grid=3)],
        [1.0, 2.0, 3.0],
    )
    LinearPayout(Free(-5.0, -1.0, grid=[-5.0, -1.0]), -3.0, 1.0, tail="lower")


# -- documents and the registry --------------------------------------------------

DOCUMENTS = [
    LINEAR,
    LinearPayout(-1.0, -2.0, 100.0, tail="lower"),
    STEP,
    TEMPLATE,
    StepPayout([1.0, 2.0], [2.0, 1.0], monotone=False),
    PIECEWISE,
    BinaryPayout(Free(0.0, 10.0, grid=5), Free(upper=7.0)),
]


@pytest.mark.parametrize("payout", DOCUMENTS, ids=lambda p: p.kind)
def test_documents_round_trip(payout: Any) -> None:
    document = payout.to_document()
    assert document["kind"] == payout.kind
    assert type(payout).from_document(document) == payout
    assert _payout_reader(payout.kind)(document) == payout


def test_the_step_document() -> None:
    assert TEMPLATE.to_document() == {
        "breakpoints": [
            {"free": {"grid": [10.0, 20.0, 30.0], "lower": 10.0, "upper": 30.0}},
            40.0,
            60.0,
        ],
        "kind": "step",
        "levels": [
            {"free": {"grid": None, "lower": 0.0, "upper": None}},
            {"free": {"grid": None, "lower": 0.0, "upper": None}},
            {"free": {"grid": None, "lower": 0.0, "upper": 500.0}},
        ],
        "monotone": True,
        "tail": "upper",
    }


@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param({"kind": "linear"}, "kind 'step'", id="kind"),
        pytest.param(
            {"kind": "step", "breakpoints": [1.0], "levels": [1.0], "cap": 2.0},
            "unknown fields",
            id="unknown-field",
        ),
        pytest.param({"kind": "step", "breakpoints": [1.0]}, "lacks", id="missing"),
        pytest.param(
            {"kind": "step", "breakpoints": [1.0], "levels": [{"lower": 0.0}]},
            "free",
            id="free-not-wrapped",
        ),
        pytest.param([1.0], "object", id="not-an-object"),
    ],
)
def test_invalid_payout_documents_raise(document: Any, message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        StepPayout.from_document(document)


def test_unknown_kinds_raise() -> None:
    with pytest.raises(UnknownPayoutKindError, match="register_payout"):
        _payout_reader("layered")
    with pytest.raises(CoverDefinitionError, match="string kind"):
        _payout_reader(None)


class _Capped:
    """A minimal user family: pays a fraction of the index, up to a cap."""

    kind = "test_capped_fraction"

    def __init__(self, fraction: float, cap: float) -> None:
        self.fraction = fraction
        self.cap = cap

    @property
    def tail(self) -> str:
        return "upper"

    @property
    def index_names(self) -> tuple[str, ...]:
        return ("index",)

    def parameters(self) -> dict[str, float | Free]:
        return {"fraction": self.fraction, "cap": self.cap}

    def with_values(self, values: Any) -> _Capped:
        return self

    def evaluate(self, index: FloatArray) -> FloatArray:
        return np.minimum(np.maximum(index, 0.0) * self.fraction, self.cap)

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.kind, "fraction": self.fraction, "cap": self.cap}

    @classmethod
    def from_document(cls, document: Any) -> _Capped:
        return cls(document["fraction"], document["cap"])


def test_register_payout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(_REGISTRY, _Capped.kind, raising=False)
    assert register_payout(_Capped) is _Capped
    assert register_payout(_Capped) is _Capped  # registering again is a no-op
    restored = _payout_reader(_Capped.kind)(_Capped(0.5, 3.0).to_document())
    assert isinstance(restored, _Capped)
    assert isinstance(restored, PayoutFunction)
    assert not isinstance(restored, LinearInAmounts)
    monkeypatch.delitem(_REGISTRY, _Capped.kind)


def test_register_payout_rejects_clashes_and_non_payouts() -> None:
    class Step(_Capped):
        kind = "step"

    class NoKind:
        pass

    class BlankKind(_Capped):
        kind = " "

    class NoEvaluate:
        kind = "test_no_evaluate"

        def parameters(self) -> None: ...

    with pytest.raises(CoverDefinitionError, match="already registered"):
        register_payout(Step)
    with pytest.raises(CoverDefinitionError, match="kind"):
        register_payout(NoKind)
    with pytest.raises(CoverDefinitionError, match="kind"):
        register_payout(BlankKind)
    with pytest.raises(CoverDefinitionError, match="missing with_values, evaluate"):
        register_payout(NoEvaluate)
    with pytest.raises(CoverDefinitionError, match="takes a class"):
        register_payout(_Capped(0.5, 1.0))  # type: ignore[type-var]
    assert "test_no_evaluate" not in _REGISTRY
    assert _REGISTRY["step"] is StepPayout
