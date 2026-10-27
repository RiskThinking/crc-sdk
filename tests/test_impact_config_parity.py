"""SDK impact models against the pinned framework contract artifact."""

from __future__ import annotations

import json
import pickle
from typing import Any, get_args

import pytest
from crc_framework import ImpactError, impact_registry_manifest

from crc_sdk.impacts import ImpactDefinition, ImpactExpression, RegistryImpactReference
from crc_sdk.impacts._contracts import impact_contracts
from crc_sdk.impacts.config import RegistryFactor, RegistryOverrides

# json.loads collapses duplicate keys, and an overflowing float literal becomes
# infinity before the native parser can report number_overflow. These cases are
# checked through from_json, which sees the original text.
DOCUMENT_EXCEPTIONS = frozenset(
    {
        "duplicate-key-root",
        "duplicate-key-model",
        "overflow-float-literal",
    }
)

_SCALAR_KEYS = (
    "type",
    "const",
    "enum",
    "default",
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "exclusiveMaximum",
    "pattern",
    "minItems",
    "minLength",
    "minProperties",
    "uniqueItems",
    "additionalProperties",
    "propertyNames",
    "items",
    "required",
    "properties",
    "anyOf",
)


def _deref(node: Any, root: dict[str, Any]) -> Any:
    if isinstance(node, dict) and "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        merged = dict(root["$defs"][name])
        merged.update({key: value for key, value in node.items() if key != "$ref"})
        return _deref(merged, root)
    return node


def normalize_schema(schema: dict[str, Any], root: dict[str, Any] | None = None) -> Any:
    """Compare shape, constraints, defaults, and unions. Drop prose."""
    if root is None:
        root = schema
    node = _deref(schema, root)
    if isinstance(node, list):
        return [normalize_schema(item, root) for item in node]
    if not isinstance(node, dict):
        return node
    union = node.get("anyOf", node.get("oneOf"))
    if union is not None:
        branches = sorted(
            (normalize_schema(branch, root) for branch in union),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
        normalized: dict[str, Any] = {"anyOf": branches}
        for key, value in node.items():
            if key in {"anyOf", "oneOf"}:
                continue
            if key in _SCALAR_KEYS:
                normalized[key] = normalize_schema(value, root)
        return normalized
    normalized = {}
    for key, value in node.items():
        if key not in _SCALAR_KEYS:
            continue
        if key == "enum":
            normalized[key] = sorted(value)
        elif key == "required":
            normalized[key] = sorted(value)
        elif key == "properties":
            normalized[key] = {
                name: normalize_schema(prop, root) for name, prop in value.items()
            }
        else:
            normalized[key] = normalize_schema(value, root)
    return normalized


def _cases(contract: str) -> dict[str, Any]:
    fixtures: dict[str, Any] = impact_contracts().load_impact_fixtures(contract)
    return fixtures


@pytest.mark.parametrize(
    ("contract", "model", "digest_name"),
    [
        ("impact_definition", ImpactDefinition, "definition_hash"),
        ("registry_impact_reference", RegistryImpactReference, "reference_hash"),
    ],
)
def test_accepted_fixtures_round_trip(
    contract: str, model: type[Any], digest_name: str
) -> None:
    for case in _cases(contract)["accepted"]:
        parsed = model.from_json(case["input_json"])
        assert parsed.to_json() == case["canonical_json"], case["name"]
        assert getattr(parsed, digest_name) == case["content_hash"], case["name"]
        validated = model.model_validate(json.loads(case["input_json"]))
        assert getattr(validated, digest_name) == case["content_hash"], case["name"]
        assert parsed.model_dump(mode="json") == json.loads(case["canonical_json"])
        assert pickle.loads(pickle.dumps(parsed)).to_json() == case["canonical_json"]
        if model is ImpactDefinition:
            restored = ImpactExpression.from_config(parsed).to_config()
            assert restored == parsed


def test_rejected_fixtures_report_the_contract_error() -> None:
    seen_exceptions: set[str] = set()
    for model, contract in (
        (ImpactDefinition, "impact_definition"),
        (RegistryImpactReference, "registry_impact_reference"),
    ):
        for case in _cases(contract)["rejected"]:
            with pytest.raises(ImpactError) as raised:
                model.from_json(case["input_json"])
            assert (raised.value.code, raised.value.path) == (
                case["error"]["code"],
                case["error"]["path"],
            ), case["name"]
            try:
                loaded = json.loads(case["input_json"])
            except json.JSONDecodeError:
                continue
            if case["name"] in DOCUMENT_EXCEPTIONS:
                seen_exceptions.add(case["name"])
                continue
            with pytest.raises(ImpactError) as raised:
                model.model_validate(loaded)
            assert (raised.value.code, raised.value.path) == (
                case["error"]["code"],
                case["error"]["path"],
            ), case["name"]
    assert seen_exceptions == DOCUMENT_EXCEPTIONS


def test_schema_valid_rejections_come_from_the_native_parser() -> None:
    for model, contract in (
        (ImpactDefinition, "impact_definition"),
        (RegistryImpactReference, "registry_impact_reference"),
    ):
        native_only = [
            case
            for case in _cases(contract)["rejected"]
            if case["schema_valid"] is True
        ]
        assert native_only
        for case in native_only:
            with pytest.raises(ImpactError) as raised:
                model.from_json(case["input_json"])
            assert raised.value.code == case["error"]["code"]


@pytest.mark.parametrize(
    ("model", "contract"),
    [
        (ImpactDefinition, "impact_definition"),
        (RegistryImpactReference, "registry_impact_reference"),
    ],
)
def test_emitted_schema_matches_the_artifact(model: type[Any], contract: str) -> None:
    contracts = impact_contracts()
    emitted = normalize_schema(model.model_json_schema(mode="validation"))
    published = normalize_schema(contracts.load_impact_schema(contract))
    assert emitted == published


def test_factor_and_override_names_match_the_artifact() -> None:
    contracts = impact_contracts()
    schema = contracts.load_impact_schema("impact_definition")
    origin = schema["$defs"]["origin_registry"]["properties"]["factor"]["enum"]
    assert set(get_args(RegistryFactor)) == set(origin)
    catalog = contracts.load_impact_contract_file("catalogs/registry-overrides.json")
    keys: set[str] = set()
    for factor in catalog["factors"].values():
        keys.update(factor["overrides"])
    assert set(RegistryOverrides.model_fields) == keys


def test_pinned_reference_uses_the_installed_registry() -> None:
    pinned = RegistryImpactReference.pinned("rx1day")
    manifest = impact_registry_manifest()
    document = json.loads(pinned.to_json())
    assert document["registry_id"] == manifest["registry_id"]
    assert document["registry_revision"] == manifest["registry_revision"]
    assert (
        document["resolver_semantics_version"] == manifest["resolver_semantics_version"]
    )
    factors = {item["factor"]: item for item in manifest["factors"]}
    assert factors["rx1day"]["calibration"]["exposure_unit"] == "mm/day"
    exposures = impact_contracts().load_impact_contract_file(
        "catalogs/hazard-exposures.json"
    )
    assert {entry["factor"] for entry in exposures["entries"]} <= set(factors)
