from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from crc_sdk.registry import (
    CATALOG_PUBLIC_FIELDS,
    EligibilityDefaults,
    HazardRegistry,
    HazardSpec,
    os_climate_registry,
    public_registry,
)

ETCCDI_ADDED = {
    "fd": ("days", "upper"),
    "su": ("days", "upper"),
    "tr": ("days", "upper"),
    "dtr": ("degC", "upper"),
    "prcptot": ("mm", "upper"),
    "r10mm": ("days", "upper"),
    "r20mm": ("days", "upper"),
    "r95p": ("mm", "upper"),
}


def test_etccdi_additions_resolve_by_name_and_alias() -> None:
    registry = public_registry()
    for name, (unit, _) in ETCCDI_ADDED.items():
        assert registry.resolve(name).unit == unit
        alias = {"r10mm": "R10mm", "r20mm": "R20mm", "r95p": "R95p"}.get(
            name, name.upper()
        )
        assert registry.resolve(alias).name == name
        assert registry.resolve(name).tail == ETCCDI_ADDED[name][1]
    assert registry.resolve("R95p").name == "r95p"


def test_internal_names_are_not_aliased_to_open_names() -> None:
    registry = public_registry(os_climate=True)
    for internal in ("tx_max", "tn_min", "consecutive_dry_days", "hot_days"):
        assert internal not in registry


def test_os_climate_seed_is_opt_in() -> None:
    assert "Wind/max_speed" not in public_registry()
    registry = public_registry(os_climate=True)
    assert registry.resolve("Wind/max_speed").unit == "m/s"
    assert registry.resolve("ChronicHeat/days_tas/above/{temp_c}c").unit == "days/year"
    assert len(os_climate_registry()) == 3
    assert "txx" in registry


def _private_registry() -> HazardRegistry:
    registry = HazardRegistry()
    registry.register(
        HazardSpec(
            "secret_hazard",
            "mm",
            "one-liner",
            block="annual",
            aliases=("sh",),
            eligibility=EligibilityDefaults(
                minimum_informative_value=0.123456, minimum_informative_knots=7
            ),
            preferred_families=("gumbel_secret",),
        )
    )
    return registry


def test_export_catalog_default_fields_and_json_ready() -> None:
    catalog = public_registry().export_catalog()
    assert [entry["name"] for entry in catalog] == sorted(
        entry["name"] for entry in catalog
    )
    assert all(set(entry) == set(CATALOG_PUBLIC_FIELDS) for entry in catalog)
    txx = next(entry for entry in catalog if entry["name"] == "txx")
    assert txx["aliases"] == ["TXx"] and txx["tail"] == "upper"
    json.dumps(catalog)


def test_export_catalog_does_not_leak_private_fields() -> None:
    text = json.dumps(_private_registry().export_catalog())
    for secret in ("eligibility", "preferred_families", "gumbel_secret", "0.123456"):
        assert secret not in text
    assert "secret_hazard" in text


def test_new_spec_fields_default_to_private() -> None:
    @dataclass(frozen=True)
    class ExtendedSpec(HazardSpec):
        campaign_note: str = "launch-policy-secret"

    registry = HazardRegistry()
    registry.register(ExtendedSpec("x", "mm", "v"))
    entry = registry.export_catalog()[0]
    assert set(entry) <= set(CATALOG_PUBLIC_FIELDS)
    assert "launch-policy-secret" not in json.dumps(entry)
    with pytest.raises(ValueError, match="campaign_note"):
        registry.export_catalog(fields=("name", "campaign_note"))


def test_explicit_opt_in_and_unknown_fields() -> None:
    registry = _private_registry()
    entry = registry.export_catalog(fields=("name", "eligibility"))[0]
    assert entry["eligibility"]["minimum_informative_knots"] == 7
    assert registry.export_catalog(fields=("name", "preferred_families"))[0][
        "preferred_families"
    ] == ["gumbel_secret"]
    with pytest.raises(ValueError, match="unknown catalogue fields"):
        registry.export_catalog(fields=("nope",))
