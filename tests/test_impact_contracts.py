from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from crc_sdk.impacts._contracts import (
    IMPACT_CONTRACTS,
    REQUIRED_IMPACT_CONTRACT_MANIFEST_SHA256,
    REQUIRED_IMPACT_CONTRACT_VERSION,
    ImpactContractUnavailable,
    impact_contracts,
)


def _installed_contracts() -> ModuleType:
    return impact_contracts()


def test_installed_framework_matches_pinned_artifact() -> None:
    contracts = _installed_contracts()
    manifest = contracts.impact_contract_manifest()
    assert manifest["artifact_version"] == REQUIRED_IMPACT_CONTRACT_VERSION
    assert contracts.impact_contract_manifest_sha256() == (
        REQUIRED_IMPACT_CONTRACT_MANIFEST_SHA256
    )
    assert tuple(contracts.impact_contract_names()) == IMPACT_CONTRACTS


def test_artifact_exposes_schemas_and_fixtures_for_every_contract() -> None:
    contracts = _installed_contracts()
    for name in IMPACT_CONTRACTS:
        schema = contracts.load_impact_schema(name)
        fixtures = contracts.load_impact_fixtures(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert fixtures["contract"] == name
        assert fixtures["accepted"] and fixtures["rejected"]


def _fake_contracts(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "IMPACT_CONTRACT_ARTIFACT": "crc-impact-contracts",
        "IMPACT_CONTRACT_VERSION": REQUIRED_IMPACT_CONTRACT_VERSION,
        "impact_contract_manifest_sha256": lambda: (
            REQUIRED_IMPACT_CONTRACT_MANIFEST_SHA256
        ),
        "verify_impact_contracts": lambda: None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_import(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    real = importlib.import_module

    def fake(name: str, package: Any = None) -> Any:
        if name == "crc_framework.contracts":
            if module is None:
                raise ModuleNotFoundError(name)
            return module
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake)


def test_missing_artifact_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import(monkeypatch, None)
    with pytest.raises(ImpactContractUnavailable, match="does not provide"):
        impact_contracts()


def test_other_artifact_version_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import(monkeypatch, _fake_contracts(IMPACT_CONTRACT_VERSION="2.0.0"))
    with pytest.raises(ImpactContractUnavailable, match="requires"):
        impact_contracts()


def test_changed_manifest_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_contracts(impact_contract_manifest_sha256=lambda: "0" * 64)
    _patch_import(monkeypatch, fake)
    with pytest.raises(ImpactContractUnavailable, match="differs from the pinned"):
        impact_contracts()


def test_matching_artifact_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    verified = []
    fake = _fake_contracts(verify_impact_contracts=lambda: verified.append(True))
    _patch_import(monkeypatch, fake)
    contracts: object = impact_contracts()
    assert contracts is fake
    assert verified == [True]


@pytest.mark.parametrize(
    "member", ["impact_contract_manifest_sha256", "verify_impact_contracts"]
)
@pytest.mark.parametrize("error_type", [FileNotFoundError, ValueError])
def test_unreadable_or_corrupt_artifact_is_reported(
    monkeypatch: pytest.MonkeyPatch, member: str, error_type: type[Exception]
) -> None:
    error = error_type("artifact file is missing or corrupted")

    def fail() -> None:
        raise error

    _patch_import(monkeypatch, _fake_contracts(**{member: fail}))
    with pytest.raises(ImpactContractUnavailable, match="could not be verified") as exc:
        impact_contracts()
    assert exc.value.__cause__ is error


def test_operation_fixtures_pin_acceptance_values() -> None:
    contracts = _installed_contracts()
    operations = contracts.load_impact_contract_file("fixtures/operations.json")
    expected = {case["name"]: case for case in operations["expected_value"]}
    assert expected["flood-aligned-histogram"]["expected"] == 0.1275
    assert expected["uniform-bin-threshold"]["expected"] == 0.75
    assert expected["misaligned-bins"]["expected"] == 0.75
    assert expected["linear-histogram"]["expected"] == 3.5
    assert expected["atom-on-threshold"]["expected"] == 0.5
    assert expected["discrete-weighted-sum"]["expected"] == 2.8
    assert expected["heavy-tail-divergent"]["error"] == "not_integrable"
    assert expected["incomplete-support"]["error"] == "missing_probability_support"
    assert expected["histogram-mass-mismatch"]["error"] == "invalid_histogram"
    assert expected["exact-unavailable"]["error"] == "exact_method_unavailable"
    assert expected["quadrature-budget"]["error"] == "convergence_failed"
    transforms = {case["name"]: case for case in operations["transform_distribution"]}
    assert transforms["decreasing-uniform"]["quantiles"] == [1.0, 5.0, 8.0]
    assert transforms["flat-segment"]["quantiles"] == [0.5, 1.0, 1.0, 1.75]
    assert transforms["binary-threshold"]["atoms"] == [
        {"value": 0.0, "probability": 0.5},
        {"value": 1.0, "probability": 0.5},
    ]
    assert transforms["non-monotone"]["error"] == "monotonicity_required"
    assert transforms["partial-support"]["error"] == "missing_probability_support"
