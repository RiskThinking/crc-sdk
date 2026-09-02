"""Validate the vectorized batch classifier against the per-row reference.

`_classify_batch` resolves point-mass and `no_data` rows across a whole
`(n_rows, n_quantiles)` matrix in one pass instead of once per row through
`_fit_or_error` -- see its docstring in `crc_sdk/fitting/workflows.py` for
why. `_fit_or_error`/`_fit_parameters`/`_fit_family` are unchanged, so the
strongest validation is differential: build a batch mixing every category,
run it through the real `fit_cdf_quantile_batches` (which now dispatches
through `_classify_batch` first), and compare every row's result against
calling `_fit_or_error` directly on that same raw row -- the pre-existing,
untouched per-row implementation.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk.fitting import CDFCurveFitPolicy, fit_cdf_quantile_batches
from crc_sdk.fitting.workflows import _classify_batch, _fit_or_error
from crc_sdk.types import SourceProvenance


def _policy(**overrides: object) -> CDFCurveFitPolicy:
    base = dict(
        h3_resolution=5,
        family="gumbel_r",
        value_unit="days",
        value_semantics="annual hot days",
        producer="tests",
        creation_version="1",
        source=SourceProvenance(provider="fixture", dataset="vectorized"),
        source_id="fixture-vectorized",
    )
    base.update(overrides)
    return CDFCurveFitPolicy(**base)  # type: ignore[arg-type]


def _table(rows: list[np.ndarray]) -> pa.Table:
    n = len(rows)
    return pa.table(
        {
            "hex_id": pa.array([599024279241097215] * n, type=pa.uint64()),
            "index_name": ["fixture"] * n,
            "year": pa.array(list(range(2000, 2000 + n)), type=pa.int32()),
            "pathway": ["historic"] * n,
            "cdf_quantiles": pa.array(
                [row.tolist() for row in rows], type=pa.large_list(pa.float64())
            ),
        }
    )


# --- targeted, hand-crafted no_data reasons (21 quantile points, interior = 19) ---

_PROBABILITIES_21 = np.linspace(0.0, 1.0, 21)
_SCREEN_POLICY = _policy(
    minimum_informative_value=1.0,
    minimum_informative_knots=4,
    minimum_distinct_informative_values=3,
)


def test_below_effective_resolution_row() -> None:
    # Monotonic, no informative (>=1.0) knots anywhere.
    row = np.linspace(0.0, 0.9, 21)
    table = _table([row])
    result = fit_cdf_quantile_batches(table, _PROBABILITIES_21, _SCREEN_POLICY)
    out = result.stream.read_all()
    assert out["curve_kind"].to_pylist() == ["no_data"]
    assert out["curve_type"].to_pylist() == ["below_effective_resolution"]


def test_insufficient_informative_support_row() -> None:
    # Exactly 2 informative (>=1.0) interior knots -- below the 4 required.
    row = np.concatenate([np.linspace(0.0, 0.9, 19), [1.0, 1.2]])
    table = _table([row])
    result = fit_cdf_quantile_batches(table, _PROBABILITIES_21, _SCREEN_POLICY)
    out = result.stream.read_all()
    assert out["curve_kind"].to_pylist() == ["no_data"]
    assert out["curve_type"].to_pylist() == ["insufficient_informative_support"]


def test_degenerate_effective_range_row() -> None:
    # 5 informative (>=1.0) interior knots, but only 2 distinct values among
    # them -- below the 3 distinct values required.
    row = np.concatenate([np.linspace(0.0, 0.9, 16), [1.0, 1.0, 1.0, 2.0, 2.0]])
    table = _table([row])
    result = fit_cdf_quantile_batches(table, _PROBABILITIES_21, _SCREEN_POLICY)
    out = result.stream.read_all()
    assert out["curve_kind"].to_pylist() == ["no_data"]
    assert out["curve_type"].to_pylist() == ["degenerate_effective_range"]


def test_sufficient_distinct_informative_values_reaches_fit() -> None:
    # Same shape as the degenerate case but with 3 distinct informative
    # values instead of 2 -- must NOT be classified as no_data.
    row = np.concatenate([np.linspace(0.0, 0.9, 16), [1.0, 1.0, 2.0, 3.0, 4.0]])
    table = _table([row])
    result = fit_cdf_quantile_batches(table, _PROBABILITIES_21, _SCREEN_POLICY)
    out = result.stream.read_all()
    assert out["curve_kind"].to_pylist()[0] != "no_data"


# --- direct unit tests of `_classify_batch` covering multiple rows at once ---


def test_classify_batch_mixed_matrix_matches_expected_categories() -> None:
    probabilities = _PROBABILITIES_21
    rows = [
        np.full(21, 3.0),  # 0: point_mass
        np.linspace(0.0, 0.9, 21),  # 1: below_effective_resolution
        np.concatenate([np.linspace(0.0, 0.9, 19), [1.0, 1.2]]),  # 2: insufficient
        np.concatenate(
            [np.linspace(0.0, 0.9, 16), [1.0, 1.0, 1.0, 2.0, 2.0]]
        ),  # 3: degenerate
        np.concatenate(
            [np.linspace(0.0, 0.9, 16), [1.0, 1.0, 2.0, 3.0, 4.0]]
        ),  # 4: needs_fit
    ]
    nan_row = np.linspace(0.0, 10.0, 21)
    nan_row[10] = np.nan
    rows.append(nan_row)  # 5: invalid (finite)
    nonmonotonic = np.linspace(0.0, 10.0, 21)
    nonmonotonic[5], nonmonotonic[6] = nonmonotonic[6], nonmonotonic[5]
    rows.append(nonmonotonic)  # 6: invalid (non-decreasing)

    matrix = np.stack(rows)
    resolved, needs_fit = _classify_batch(matrix, probabilities, _SCREEN_POLICY)

    curve_0, curve_1, curve_2, curve_3 = (
        resolved[0],
        resolved[1],
        resolved[2],
        resolved[3],
    )
    assert isinstance(curve_0, dict) and curve_0["curve_kind"] == "point_mass"
    assert (
        isinstance(curve_1, dict)
        and curve_1["curve_type"] == "below_effective_resolution"
    )
    assert (
        isinstance(curve_2, dict)
        and curve_2["curve_type"] == "insufficient_informative_support"
    )
    assert (
        isinstance(curve_3, dict)
        and curve_3["curve_type"] == "degenerate_effective_range"
    )
    assert resolved[4] is None and needs_fit[4]
    assert isinstance(resolved[5], ValueError)
    assert "finite" in str(resolved[5])
    assert isinstance(resolved[6], ValueError)
    assert "non-decreasing" in str(resolved[6])
    assert list(needs_fit) == [False, False, False, False, True, False, False]


# --- large randomized differential test against the untouched per-row path ---


def _random_point_mass_row(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.full(n, float(rng.uniform(-5.0, 5.0)))


def _random_fittable_row(rng: np.random.Generator, n: int) -> np.ndarray:
    # A smooth, strictly-increasing-enough curve with real variance --
    # should survive every screen and reach `_fit_family`.
    base = np.sort(rng.uniform(0.0, 50.0, n))
    base[0] = min(base[0], 0.0)
    return np.round(base, 3)


def _random_invalid_row(rng: np.random.Generator, n: int) -> np.ndarray:
    row = np.linspace(0.0, 10.0, n)
    if rng.random() < 0.5:
        row[rng.integers(1, n - 1)] = np.nan
    else:
        i = rng.integers(1, n - 2)
        row[i], row[i + 1] = row[i + 1], row[i]
    return row


@pytest.mark.parametrize("minimum_informative_value", [None, 1.0])
def test_large_random_batch_matches_per_row_reference(
    minimum_informative_value: float | None,
) -> None:
    rng = np.random.default_rng(20260901)
    n_quantiles = 41
    probabilities = np.linspace(0.0, 1.0, n_quantiles)
    policy = _policy(
        minimum_informative_value=minimum_informative_value,
        minimum_informative_knots=4,
        minimum_distinct_informative_values=3,
        parametric_failure_action="tabulated",
        on_fit_failure="skip",
    )

    generators = [_random_point_mass_row, _random_fittable_row, _random_invalid_row]
    rows = [
        generators[rng.integers(0, len(generators))](rng, n_quantiles)
        for _ in range(300)
    ]

    reference = [
        _fit_or_error(row, probabilities=probabilities, policy=policy) for row in rows
    ]

    table = _table(rows)
    result = fit_cdf_quantile_batches(table, probabilities, policy)
    out = result.stream.read_all()

    out_index = 0
    for row_index, ref in enumerate(reference):
        if isinstance(ref, ValueError):
            # Routed to skipped/invalid handling depending on which check
            # failed and policy.on_fit_failure -- either way it produces no
            # canonical row, exactly like the reference call raising/being
            # a ValueError with nothing written.
            continue
        actual_kind = out["curve_kind"][out_index].as_py()
        actual_type = out["curve_type"][out_index].as_py()
        ref_treatment = ref.pop("_treatment")
        assert actual_kind == ref["curve_kind"], (row_index, rows[row_index])
        assert actual_type == ref["curve_type"], (row_index, rows[row_index])
        for field in (
            "curve_shape",
            "curve_location",
            "curve_scale",
            "curve_atom_probability",
            "curve_atom_location",
        ):
            actual_value = out[field][out_index].as_py()
            ref_value = ref[field]
            if ref_value is None:
                assert actual_value is None, (row_index, field)
            else:
                assert actual_value == pytest.approx(ref_value), (
                    row_index,
                    field,
                    ref_treatment,
                )
        out_index += 1
    assert out_index == out.num_rows
