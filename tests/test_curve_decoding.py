"""Columnar curve decoding admits and rejects exactly what row-wise decoding does."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    Distribution,
    DistributionFamily,
    QuantileCurve,
    TailCompletion,
)

from crc_sdk._schema.dataset import (
    _curve_arrow,
    _curve_from_arrow,
    _curves_from_arrow,
    dataset_arrow_schema,
)

_CURVE = dataset_arrow_schema().field("curve").type

_GEV = Distribution.parametric(
    DistributionFamily.GENERALIZED_EXTREME_VALUE, location=1.0, scale=2.0, shape=-0.1
)


def _valid() -> list[Distribution | QuantileCurve]:
    curve = QuantileCurve([0.5, 0.9, 0.99], [1.0, 2.0, 4.0])
    return [
        _GEV,
        Distribution.parametric(
            DistributionFamily.GUMBEL_RIGHT, location=-1.5, scale=0.25
        ),
        Distribution.point_mass(0.0),
        Distribution.hurdle(_GEV, atom_probability=0.4, atom_location=0.0),
        Distribution.hurdle(Distribution.point_mass(2.0), atom_probability=1.0),
        Distribution.hurdle(
            Distribution.empirical([1.0, 2.0, 2.0]), atom_probability=0.2
        ),
        curve,
        curve.complete(TailCompletion(lower=0.0, upper=10.0)),
        Distribution.empirical([1.0, 3.0, 3.0, 7.0]),
        Distribution.histogram([0.0, 1.0, 3.0], [0.25, 0.75]),
    ]


def _branch(kind: str, **values: object) -> dict[str, object]:
    row: dict[str, object] = {
        name: None
        for name in (
            "parametric",
            "discrete",
            "histogram",
            "point_mass",
            "hurdle",
            "quantile",
            "transformed",
            "quantile_curve",
        )
    }
    row.update(kind=kind, evidence=None)
    row[kind] = values
    return row


def _base(text: str) -> dict[str, object]:
    return _branch("hurdle", base=text, atom_probability=0.5, atom_location=0.0)


_GEV_TEXT = (
    '{"family":"genextreme","kind":"parametric","location":1.0,"scale":2.0,'
    '"shape":-0.1}'
)

_INVALID = [
    _branch("parametric", family="genextreme", location=math.nan, scale=1.0, shape=0.1),
    _branch("parametric", family="genextreme", location=0.0, scale=-1.0, shape=0.1),
    _branch("parametric", family="genextreme", location=0.0, scale=1.0, shape=None),
    _branch("parametric", family="gumbel_r", location=0.0, scale=1.0, shape=0.1),
    _branch("parametric", family="weibull_min", location=0.0, scale=1.0, shape=-1.0),
    _branch("parametric", family="unknown", location=0.0, scale=1.0, shape=0.1),
    _branch("point_mass", value=math.inf),
    _branch("quantile_curve", probabilities=[0.5, 0.4], values=[1.0, 2.0]),
    _branch("quantile_curve", probabilities=[0.5, 1.5], values=[1.0, 2.0]),
    _branch("quantile_curve", probabilities=[0.5, 0.9], values=[2.0, 1.0]),
    _branch("quantile_curve", probabilities=[0.5], values=[1.0]),
    _branch("quantile_curve", probabilities=[0.5, None], values=[1.0, 2.0]),
    _branch("quantile_curve", probabilities=[0.5, 0.9], values=[1.0, math.nan]),
    _base(_GEV_TEXT.replace('"scale":2.0', '"scale":2.0,"scale":3.0')),
    _base(_GEV_TEXT.replace('"scale":2.0', '"scale":0.0')),
    _base(_GEV_TEXT.replace('"location":1.0', '"location":NaN')),
    _base(_GEV_TEXT.replace('"shape":-0.1', '"extra":1.0,"shape":-0.1')),
    _base(_GEV_TEXT.replace('"family":"genextreme"', '"family":"gumbel_r"')),
    _base("[]"),
    _branch("hurdle", base=_GEV_TEXT, atom_probability=1.5, atom_location=0.0),
    _branch("hurdle", base=_GEV_TEXT, atom_probability=0.5, atom_location=math.inf),
    {**_branch("point_mass", value=1.0), "kind": "parametric"},
    {**_branch("point_mass", value=1.0), "parametric": {"family": "gumbel_r"}},
    {**_branch("point_mass", value=1.0), "evidence": "{}"},
    {**_branch("point_mass", value=1.0), "kind": "unknown"},
]

# Readable by the framework but not written canonically by this module: the
# typed path declines them and the JSON path decides.
_NONCANONICAL = [
    _base(_GEV_TEXT.replace(",", ", ")),
    _base(_GEV_TEXT.replace('"location":1.0', '"location":1')),
    _base(
        '{"kind":"parametric","family":"genextreme","location":1.0,"scale":2.0,'
        '"shape":-0.1}'
    ),
]


def _outcome(decode: Callable[[], str]) -> tuple[str, str]:
    """A decoding's JSON text, or its error type and message."""
    try:
        return "ok", decode()
    except (TypeError, ValueError) as error:
        return type(error).__name__, str(error)


def _text(curves: Iterable[Distribution | QuantileCurve | None]) -> str:
    return "\n".join("null" if c is None else c.to_json() for c in curves)


@pytest.mark.parametrize("row", _INVALID + _NONCANONICAL)
def test_each_row_has_its_row_wise_outcome(row: dict[str, object]) -> None:
    array = pa.array([row], type=_CURVE)
    assert _outcome(lambda: _text(_curves_from_arrow(array))) == _outcome(
        lambda: _text([_curve_from_arrow(array[0].as_py())])
    )


def test_valid_rows_decode_to_identical_framework_objects() -> None:
    valid = _valid()
    random.seed(7)
    curves = [random.choice([None, *valid]) for _ in range(400)]
    array = pa.array(
        [None if c is None else _curve_arrow(c) for c in curves], type=_CURVE
    )
    chunked = pa.chunked_array([array.slice(0, 150), array.slice(150)])
    for column in (array, chunked, array.slice(37, 200)):
        decoded = _curves_from_arrow(column)
        expected = [
            None if v is None else _curve_from_arrow(v) for v in column.to_pylist()
        ]
        assert _text(decoded) == _text(expected)
        assert all(type(a) is type(b) for a, b in zip(decoded, expected, strict=True))


@pytest.mark.parametrize("row", _INVALID)
def test_a_batch_has_its_row_wise_outcome(row: dict[str, object]) -> None:
    mixed = pa.array([_curve_arrow(_GEV), None, row], type=_CURVE)
    assert _outcome(lambda: _text(_curves_from_arrow(mixed))) == _outcome(
        lambda: _text(
            None if v is None else _curve_from_arrow(v) for v in mixed.to_pylist()
        )
    )


def test_noncanonical_hurdle_bases_still_decode() -> None:
    for row in _NONCANONICAL:
        (curve,) = _curves_from_arrow(pa.array([row], type=_CURVE))
        assert isinstance(curve, Distribution)
        assert curve.kind.value == "hurdle"
