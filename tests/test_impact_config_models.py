"""Construction, immutability, and identity of impact config models."""

from __future__ import annotations

import json
import pickle

import pytest
from crc_framework import ImpactError, TransformContext
from pydantic import ValidationError

from crc_sdk.impacts import (
    ImpactContext,
    ImpactDefinition,
    ImpactModelError,
    LinearModel,
    RegistryImpactReference,
    RegistryOverrides,
)

CALIBRATION = {
    "exposure_variable": "flood_depth",
    "exposure_unit": "m",
    "impact_metric": "damage_ratio",
    "impact_unit": "1",
}


def _definition(**changes: object) -> ImpactDefinition:
    document: dict[str, object] = {
        "schema_version": "1.0",
        "semantics_version": "crc-impact/1",
        "calibration": CALIBRATION,
        "model": {"kind": "linear", "slope": 1.0},
        "provenance": {"origin": {"kind": "user_defined"}},
    }
    document.update(changes)
    return ImpactDefinition.model_validate(document)


def test_models_are_deeply_immutable() -> None:
    definition = _definition()
    with pytest.raises(ValidationError):
        definition.calibration.exposure_unit = "cm"
    models = (
        definition,
        definition.calibration,
        definition.model,
        definition.provenance,
    )
    for model in models:
        for name in type(model).model_fields:
            value = getattr(model, name)
            assert not isinstance(value, (dict, list))


def test_changes_revalidate() -> None:
    definition = _definition()
    changed = definition.with_changes(model={"kind": "linear", "slope": 2.0})
    assert changed.definition_hash != definition.definition_hash
    assert isinstance(changed.model, LinearModel)
    assert changed.model.slope == 2.0
    with pytest.raises(ImpactError) as raised:
        definition.model_copy(update={"model": {"kind": "linear", "slope": "no"}})
    assert raised.value.code == "invalid_type"


def test_derive_replaces_a_registry_origin() -> None:
    reference = RegistryImpactReference.pinned("fwi")
    resolved = reference.resolve()
    derived = resolved.to_config().derive(
        "scaled", model={"kind": "linear", "slope": 3.0}
    )
    origin = derived.provenance.origin
    assert origin.kind == "derived"
    assert origin.description == "scaled"
    assert resolved.definition_hash in derived.provenance.derived_from
    assert derived.definition_hash != resolved.definition_hash


def test_equality_follows_the_content_hash() -> None:
    left = _definition()
    right = ImpactDefinition.from_json(left.to_json())
    assert left == right
    assert hash(left) == hash(right)
    assert left != _definition(model={"kind": "linear", "slope": -1.0})


def test_context_absence_is_not_explicit_null() -> None:
    omitted = ImpactContext(continent="Europe")
    cleared = ImpactContext(continent="Europe", country=None)
    assert omitted.supplied() == {"continent": "Europe"}
    assert cleared.supplied() == {"continent": "Europe", "country": None}


def test_override_absence_is_not_explicit_null() -> None:
    omitted = RegistryOverrides(midpoint=1.0)
    cleared = RegistryOverrides(midpoint=1.0, zero_below=None)
    assert omitted.model_dump(mode="json") == {"midpoint": 1.0}
    assert cleared.model_dump(mode="json") == {"midpoint": 1.0, "zero_below": None}
    document = {
        "schema_version": "1.0",
        "registry_id": "crc-climate-impacts",
        "registry_revision": "2026.10.0",
        "registry_manifest_sha256": "33" * 32,
        "resolver_semantics_version": "crc-impact-registry/1",
        "factor": "cyclone",
        "overrides": {"midpoint": None},
    }
    with pytest.raises(ImpactError) as raised:
        RegistryImpactReference.model_validate(document)
    assert (raised.value.code, raised.value.path) == (
        "invalid_value",
        "/overrides/midpoint",
    )


def test_integer_cell_is_rejected_on_the_wire_and_accepted_from_context() -> None:
    context = ImpactContext.from_transform_context(TransformContext(cell=0xABC))
    assert context.cell == f"{0xABC:015x}"
    document = {
        "schema_version": "1.0",
        "semantics_version": "crc-impact/1",
        "calibration": CALIBRATION,
        "model": {"kind": "linear", "slope": 1},
        "provenance": {
            "origin": {
                "kind": "registry",
                "registry_id": "crc-climate-impacts",
                "registry_revision": "2026.10.0",
                "registry_manifest_sha256": "ab" * 32,
                "resolver_semantics_version": "crc-impact-registry/1",
                "factor": "fwi",
                "context": {"cell": 1},
                "overrides": {},
                "fallback_policy": "error",
                "lookups": [],
            }
        },
    }
    with pytest.raises(ImpactError) as raised:
        ImpactDefinition.model_validate(document)
    assert raised.value.code == "invalid_type"
    assert raised.value.path == "/provenance/origin/context/cell"


def test_model_validate_json_rejects_duplicate_keys() -> None:
    text = '{"schema_version":"1.0","schema_version":"1.0"}'
    with pytest.raises(ImpactError) as raised:
        ImpactDefinition.model_validate_json(text)
    assert raised.value.code == "duplicate_key"
    assert json.loads(text)["schema_version"] == "1.0"


def test_pickle_round_trip_uses_canonical_json() -> None:
    definition = _definition()
    restored = pickle.loads(pickle.dumps(definition))
    assert restored.to_json() == definition.to_json()
    assert not isinstance(restored, ImpactModelError)
