"""Canonical JSON and hashes agree with crc-framework's canonical profile."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
import sys
from typing import Any

import numpy as np
import pytest
from crc_framework.contracts import load_impact_contract_file, load_impact_fixtures

from crc_sdk.impacts import ImpactExpression
from crc_sdk.parametric import OutcomeSample, SampleProvenance
from crc_sdk.parametric import _canonical as canonical
from crc_sdk.parametric._canonical import (
    canonical_json,
    content_hash,
    format_number,
    format_numbers,
)

CANONICAL = load_impact_contract_file("fixtures/canonical-json.json")
DEFINITIONS = load_impact_fixtures("impact_definition")
CALIBRATION = {
    "exposure_variable": "rx1day",
    "exposure_unit": "mm/day",
    "impact_metric": "damage_ratio",
    "impact_unit": "1",
}
# Small and large exponents on both sides of every switch between positional
# and exponent notation, plus the extremes of binary64.
EXPONENT_VALUES = [
    1e-5,
    1.5e-5,
    9.99e-6,
    1e-6,
    1.2345678e-8,
    1e-7,
    -2.5e-10,
    1e-100,
    5e-324,
    2.2250738585072014e-308,
    0.000123,
    0.1,
    3.0,
    123456789012.5,
    999999999999999.9,
    1e15,
    1.5e15,
    9007199254740992.0,
    1e16,
    1.2345678901234568e16,
    1e21,
    6.02214076e23,
    1e100,
    1.7976931348623157e308,
    -1e16,
    -0.0,
]
# Both sides of each layout switch, and the binary64 neighbours of the
# switching values: decimal exponents -5/-6 and 15/16, subnormals, and values
# whose shortest round-trip spelling needs 17 significant digits.
BOUNDARY_VALUES = [
    0.0,
    -0.0,
    5e-324,
    1e-323,
    2.225073858507201e-308,
    2.2250738585072014e-308,
    1e-10,
    9.999999999999999e-10,
    1e-9,
    9.99999999999999e-7,
    1e-6,
    9.999999999999999e-6,
    float(np.nextafter(1e-5, 0.0)),
    1e-5,
    float(np.nextafter(1e-5, 1.0)),
    1.0000000000000001e-5,
    5.5e-5,
    9.999999999999999e-5,
    float(np.nextafter(1e-4, 0.0)),
    1e-4,
    float(np.nextafter(1e-4, 1.0)),
    0.30000000000000004,
    1.2345678901234567,
    33.333333333333336,
    999999999999999.9,
    float(np.nextafter(1e15, 0.0)),
    1e15,
    9999999999999998.0,
    float(np.nextafter(1e16, 0.0)),
    1e16,
    float(np.nextafter(1e16, 1e17)),
    1.2345678901234568e16,
    1.7976931348623157e308,
]
BOUNDARY_VALUES += [-value for value in BOUNDARY_VALUES]


def _item_by_item(items: list[Any]) -> str:
    """The canonical text of a list, spelled one item at a time."""
    return "[" + ",".join(canonical_json(item) for item in items) + "]"


def test_profile_is_the_one_this_module_follows() -> None:
    assert CANONICAL["profile"] == "crc-canonical-json/1"


@pytest.mark.parametrize(
    "vector", CANONICAL["numbers"], ids=[v["name"] for v in CANONICAL["numbers"]]
)
def test_number_vectors_of_the_framework_profile(vector: dict[str, str]) -> None:
    value = float(vector["input"])
    # The profile reads negative zero as zero.
    assert struct.pack(">d", value + 0.0).hex() == vector["bits"]
    assert format_number(value) == vector["canonical"]
    assert canonical_json(value) == vector["canonical"]


@pytest.mark.parametrize(
    "vector", CANONICAL["strings"], ids=[v["name"] for v in CANONICAL["strings"]]
)
def test_string_vectors_of_the_framework_profile(vector: dict[str, str]) -> None:
    text = canonical_json(vector["value"])
    assert text == vector["canonical"]
    assert text.encode("utf-8").hex() == vector["utf8_hex"]


@pytest.mark.parametrize(
    "case", DEFINITIONS["accepted"], ids=[c["name"] for c in DEFINITIONS["accepted"]]
)
def test_framework_definition_fixtures_hash_the_same(case: dict[str, Any]) -> None:
    document = json.loads(case["canonical_json"])
    assert canonical_json(document) == case["canonical_json"]
    assert content_hash(document) == case["content_hash"]


def test_exponents_hash_as_the_framework_hashes_them() -> None:
    points = [(float(i), value) for i, value in enumerate(EXPONENT_VALUES)]
    expression = ImpactExpression.tabulated(points, calibration=CALIBRATION)
    text = expression.to_json()
    document = json.loads(text)
    assert canonical_json(document) == text
    assert content_hash(document) == expression.definition_hash
    # The framework spells every value as format_number does.
    for position, value in enumerate(EXPONENT_VALUES):
        assert f'{{"x":{format_number(position)},"y":{format_number(value)}}}' in text


def test_hash_is_sha256_of_the_canonical_bytes() -> None:
    document = {"b": [1, 2.5, None, True, False], "a": {"z": "é", "y": -0.0}}
    text = canonical_json(document)
    assert text == '{"a":{"y":0.0,"z":"é"},"b":[1.0,2.5,null,true,false]}'
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert content_hash(document) == f"sha256:{digest}"


def test_keys_sort_by_utf8_bytes() -> None:
    document = {"é": 1.0, "b": 1.0, "Z": 1.0, "aa": 1.0, "a": 1.0, "ÿ": 1.0}
    assert canonical_json(document) == (
        '{"Z":1.0,"a":1.0,"aa":1.0,"b":1.0,"é":1.0,"ÿ":1.0}'
    )


def test_integers_are_written_as_binary64_floats() -> None:
    assert canonical_json([0, 1, -7, 2**53 + 1]) == "[0.0,1.0,-7.0,9007199254740992.0]"


def test_tuples_and_nested_documents() -> None:
    assert canonical_json({"k": ((1.0,), [])}) == '{"k":[[1.0],[]]}'


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 10**400])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": value})


def test_unpaired_surrogates_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unicode"):
        canonical_json("\ud800")
    with pytest.raises(ValueError, match="Unicode"):
        canonical_json({"\ud800": 1.0})


@pytest.mark.parametrize("value", [{1: 1.0}, b"bytes", object(), {1.0}])
def test_other_types_are_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        canonical_json(value)


# -- the vectorised path -----------------------------------------------------


def test_format_numbers_spells_random_bit_patterns_as_format_number() -> None:
    rng = np.random.default_rng(20261106)
    values = rng.integers(0, 2**64, size=200_000, dtype=np.uint64).view(np.float64)
    values = values[np.isfinite(values)]
    assert format_numbers(values) == [format_number(v) for v in values.tolist()]


def test_format_numbers_spells_decimal_values_as_format_number() -> None:
    rng = np.random.default_rng(7)
    exponents = rng.integers(-12, 20, size=100_000)
    values = rng.random(100_000) * 10.0 ** exponents.astype(np.float64)
    rounded = np.array([round(v, int(d)) for v, d in zip(values, exponents % 7)])
    for array in (values, -values, rounded, np.floor(values)):
        assert format_numbers(array) == [format_number(v) for v in array.tolist()]


@pytest.mark.parametrize("value", BOUNDARY_VALUES, ids=repr)
def test_format_numbers_at_the_layout_boundaries(value: float) -> None:
    expected = format_number(value)
    assert format_numbers([value]) == [expected]
    # Among other values, and at every position of a long sequence.
    rng = np.random.default_rng(3)
    for position in (0, 17, 39):
        items = rng.normal(0.0, 1e3, 40).tolist()
        items[position] = value
        spelled = format_numbers(items)
        assert spelled[position] == expected
        assert spelled == [format_number(item) for item in items]
        assert canonical_json(items) == _item_by_item(items)


def test_format_numbers_follows_the_framework_vectors() -> None:
    vectors = CANONICAL["numbers"]
    values = [float(vector["input"]) for vector in vectors]
    expected = [vector["canonical"] for vector in vectors]
    assert format_numbers(values) == expected
    assert canonical_json(values * 3) == "[" + ",".join(expected * 3) + "]"


def test_long_number_lists_hash_as_the_framework_hashes_them() -> None:
    rng = np.random.default_rng(11)
    magnitudes = 10.0 ** rng.integers(-11, 19, 300).astype(np.float64)
    edges = np.unique(np.r_[rng.random(300) * magnitudes, BOUNDARY_VALUES])
    values = rng.random(edges.size - 1) * 10.0 ** rng.integers(-9, 1, edges.size - 1)
    expression = ImpactExpression.binned(
        edges.tolist(), values.tolist(), calibration=CALIBRATION
    )
    text = expression.to_json()
    assert canonical_json(json.loads(text)) == text
    assert content_hash(json.loads(text)) == expression.definition_hash


@pytest.mark.parametrize("length", [1, 31, 32, 33, 1000])
def test_long_sequences_are_spelled_as_item_by_item(length: int) -> None:
    rng = np.random.default_rng(length)
    strings = [vector["value"] for vector in CANONICAL["strings"]]
    sequences: list[list[Any]] = [
        rng.normal(0.0, 1e-5, length).tolist(),
        rng.integers(-(2**62), 2**62, length).tolist(),
        [2**53 + 1, 2**64 + 1, -(2**70) - 3, 10**300, 0, -7] * length,
        [1, 2.5, -0.0, 3] * length,
        [1.0 / 3.0] * length,
        [-0.0] * length,
        [strings[i % len(strings)] for i in range(length)],
        ["warehouse-a"] * length,
        [1.0, True, None, "x", [2.0] * 40, {"k": 1}] * length,
        [np.float64(0.5), 1.0] * length,
    ]
    for items in sequences:
        assert canonical_json(items) == _item_by_item(items)
        assert canonical_json(tuple(items)) == _item_by_item(items)


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        (math.nan, "no spelling"),
        (-math.inf, "no spelling"),
        (10**400, "binary64"),
        ("\ud800", "Unicode"),
    ],
)
def test_long_sequences_reject_what_single_values_reject(
    tail: Any, message: str
) -> None:
    head: list[Any] = ["a"] * 40 if isinstance(tail, str) else [1.0] * 40
    with pytest.raises(ValueError, match=message):
        canonical_json([*head, tail])


@pytest.mark.parametrize(
    "values", [[1.0, math.nan], [math.inf], [10**400], [[1.0, 2.0]]], ids=repr
)
def test_format_numbers_rejects_what_has_no_spelling(values: Any) -> None:
    with pytest.raises(ValueError):
        format_numbers(values)


def test_format_numbers_of_nothing_is_nothing() -> None:
    assert format_numbers([]) == []
    assert format_numbers(np.arange(3)) == ["0.0", "1.0", "2.0"]


def test_a_large_sample_hashes_as_item_by_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(5)
    n = 20_000
    index = rng.gumbel(50.0, 10.0, n)
    sample = OutcomeSample(
        asset_ids=("warehouse-a",),
        period=np.arange(n),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=np.maximum(index - 60.0, 0.0) * 1e4,
        hazard=index * 1e-7,  # decimal exponents -6 and -5
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table", sources=(), semantics_source="caller"
        ),
    )
    vectorised = sample.content_hash
    monkeypatch.setattr(canonical, "_BULK_LENGTH", sys.maxsize)
    # A fresh copy, so the cached hash is not reused.
    assert dataclasses.replace(sample).content_hash == vectorised
