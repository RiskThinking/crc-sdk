"""Canonical JSON and hashes agree with crc-framework's canonical profile."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

import pytest
from crc_framework.contracts import load_impact_contract_file, load_impact_fixtures

from crc_sdk.impacts import ImpactExpression
from crc_sdk.parametric._canonical import canonical_json, content_hash, format_number

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
