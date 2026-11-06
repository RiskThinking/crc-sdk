"""Free parameters: bounds, grids, documents and validation."""

from __future__ import annotations

import math
import pickle
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from crc_sdk.parametric import CoverDefinitionError, Free
from crc_sdk.parametric.parameters import _resolve_values


def test_defaults_are_a_non_negative_unbounded_amount() -> None:
    free = Free()
    assert (free.lower, free.upper, free.grid) == (0.0, None, None)
    assert free.to_document() == {"free": {"grid": None, "lower": 0.0, "upper": None}}


def test_positional_bounds_and_keyword_grid() -> None:
    free = Free(10, 20.5, grid=[10, 15, 20.5])
    assert (free.lower, free.upper, free.grid) == (10.0, 20.5, (10.0, 15.0, 20.5))
    assert all(isinstance(value, float) for value in free.grid or ())


def test_a_grid_count_is_stored_as_its_values() -> None:
    free = Free(50.0, 100.0, grid=5)
    assert free.grid == (50.0, 62.5, 75.0, 87.5, 100.0)
    assert free.to_document()["free"]["grid"] == [50.0, 62.5, 75.0, 87.5, 100.0]
    assert Free(0.0, 1.0, grid=np.int64(3)).grid == (0.0, 0.5, 1.0)


def test_a_grid_may_be_a_numpy_array() -> None:
    assert Free(0.0, 3.0, grid=np.array([1.0, 2.0])).grid == (1.0, 2.0)


def test_negative_bounds_are_allowed_for_thresholds() -> None:
    free = Free(-3.0, -1.0, grid=3)
    assert free.grid == (-3.0, -2.0, -1.0)


def test_negative_zero_is_normalized() -> None:
    assert math.copysign(1.0, Free(-0.0).lower) == 1.0


def test_contains_checks_the_bounds_only() -> None:
    free = Free(1.0, 2.0, grid=[1.0, 2.0])
    assert free.contains(1.5)
    assert free.contains(1.0) and free.contains(2.0)
    assert not free.contains(0.5) and not free.contains(2.5)
    assert Free().contains(1e300)


def test_document_round_trip_equality_and_pickle() -> None:
    free = Free(1.0, 5.0, grid=[1.0, 3.0])
    assert Free.from_document(free.to_document()) == free
    assert pickle.loads(pickle.dumps(free)) == free
    assert hash(free) == hash(Free(1.0, 5.0, grid=[1.0, 3.0]))


def test_free_is_frozen() -> None:
    free = Free()
    with pytest.raises(ValidationError, match="frozen"):
        free.lower = 1.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        pytest.param({"lower": math.nan}, "finite", id="nan-lower"),
        pytest.param({"lower": 0.0, "upper": math.inf}, "finite", id="inf-upper"),
        pytest.param({"lower": True}, "lower", id="bool-lower"),
        pytest.param({"lower": "1"}, "lower", id="text-lower"),
        pytest.param({"lower": 2.0, "upper": 1.0}, "must be <= upper", id="order"),
        pytest.param({"grid": 3}, "count needs finite bounds", id="count-unbounded"),
        pytest.param(
            {"lower": 1.0, "upper": 1.0, "grid": 2}, "lower < upper", id="count-flat"
        ),
        pytest.param({"lower": 0.0, "upper": 1.0, "grid": 1}, "2 points", id="count-1"),
        pytest.param({"lower": 0.0, "upper": 1.0, "grid": []}, "at least", id="empty"),
        pytest.param(
            {"lower": 0.0, "upper": 1.0, "grid": [0.5, 0.5]}, "strictly", id="repeat"
        ),
        pytest.param(
            {"lower": 0.0, "upper": 1.0, "grid": [0.8, 0.2]}, "strictly", id="down"
        ),
        pytest.param(
            {"lower": 0.0, "upper": 1.0, "grid": [0.5, 1.5]}, "within", id="above"
        ),
        pytest.param({"lower": 1.0, "grid": [0.5]}, "within", id="below"),
        pytest.param(
            {"lower": 0.0, "upper": 1.0, "grid": [0.5, math.nan]}, "grid", id="nan-grid"
        ),
        pytest.param({"grid": "abc"}, "grid", id="text-grid"),
    ],
)
def test_invalid_free_parameters_raise(arguments: dict[str, Any], message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        Free(**arguments)


def test_errors_are_value_errors() -> None:
    with pytest.raises(ValueError):
        Free(2.0, 1.0)


@pytest.mark.parametrize(
    "document",
    [
        pytest.param({"lower": 0.0}, id="not-wrapped"),
        pytest.param({"free": {}, "other": 1}, id="extra-key"),
        pytest.param({"free": {"lower": 0.0, "upper": None, "step": 1}}, id="unknown"),
        pytest.param({"free": [0.0]}, id="not-an-object"),
        pytest.param([0.0], id="list"),
    ],
)
def test_invalid_documents_raise(document: Any) -> None:
    with pytest.raises(CoverDefinitionError):
        Free.from_document(document)


def test_from_document_fills_defaults() -> None:
    assert Free.from_document({"free": {}}) == Free()


PARAMETERS: dict[str, float | Free] = {
    "attachment": 50.0,
    "levels[0]": Free(),
    "levels[1]": Free(1.0, 10.0),
}


def test_resolve_values_checks_names_bounds_and_types() -> None:
    assert _resolve_values(PARAMETERS, {"levels[0]": 3, "levels[1]": 10.0}) == {
        "levels[0]": 3.0,
        "levels[1]": 10.0,
    }
    assert _resolve_values(PARAMETERS, {}) == {}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param({"levels[2]": 1.0}, "unknown parameter", id="unknown"),
        pytest.param({"attachment": 60.0}, "is fixed", id="fixed"),
        pytest.param({"levels[0]": -1.0}, "outside", id="below"),
        pytest.param({"levels[1]": 11.0}, "outside", id="above"),
        pytest.param({"levels[0]": math.nan}, "finite number", id="nan"),
        pytest.param({"levels[0]": True}, "finite number", id="bool"),
        pytest.param({"levels[0]": "1"}, "finite number", id="text"),
    ],
)
def test_resolve_values_rejects_bad_values(values: Any, message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        _resolve_values(PARAMETERS, values)


def test_resolve_values_needs_a_mapping() -> None:
    with pytest.raises(CoverDefinitionError, match="map parameter names"):
        _resolve_values(PARAMETERS, [("levels[0]", 1.0)])  # type: ignore[arg-type]
