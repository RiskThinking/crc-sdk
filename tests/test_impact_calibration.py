"""Exposure and response checks against the framework catalogs."""

from __future__ import annotations

import pytest

from crc_sdk.impacts import (
    ExposureDeclaration,
    ImpactCalibration,
    RegistryImpactReference,
    check_exposure,
    check_response_labels,
    exposure_declaration,
    normalize_unit,
    units_equivalent,
)
from crc_sdk.impacts.errors import ImpactCalibrationError


def test_unit_aliases_are_case_sensitive() -> None:
    assert normalize_unit("days") == "d"
    assert normalize_unit("Days") == "Days"
    assert units_equivalent("metres", "m")
    assert units_equivalent("fraction", "1")
    assert not units_equivalent("mm", "mm/day")
    assert not units_equivalent("MM", "mm")


def test_catalog_entries_match_factor_calibrations() -> None:
    rx1day = exposure_declaration("Rx1day")
    assert rx1day == ExposureDeclaration("rx1day", "mm/day", "hazard_catalog")
    frost = exposure_declaration("fd")
    assert frost.variable == "frost_days"
    assert frost.unit == "d"
    dry = exposure_declaration("CDD")
    assert dry.variable == "maximum_consecutive_dry_days"
    assert dry.unit == "d"
    matched = check_exposure(
        ImpactCalibration(
            exposure_variable="frost_days",
            exposure_unit="d",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
        ExposureDeclaration("frost_days", "days", "hazard_catalog"),
    )
    assert matched.basis == "unit_alias"
    exact = check_exposure(
        ImpactCalibration(
            exposure_variable="rx1day",
            exposure_unit="mm/day",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
        exposure_declaration("rx1day"),
    )
    assert exact.basis == "exact"


def test_explicit_mapping_overrides_the_catalog() -> None:
    declared = exposure_declaration(
        "rx1day",
        mapping={"rx1day": ExposureDeclaration("stage", "m", "hazard_catalog")},
    )
    assert declared.basis == "explicit"
    assert declared.variable == "stage"
    match = check_exposure(
        ImpactCalibration(
            exposure_variable="stage",
            exposure_unit="m",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
        declared,
    )
    assert match.basis == "explicit_mapping"


def test_unknown_hazards_and_disagreements_are_rejected() -> None:
    with pytest.raises(ImpactCalibrationError) as raised:
        exposure_declaration("txx")
    assert raised.value.code == "unknown_exposure_semantics"
    calibration = ImpactCalibration(
        exposure_variable="rx1day",
        exposure_unit="mm/day",
        impact_metric="damage_ratio",
        impact_unit="1",
    )
    with pytest.raises(ImpactCalibrationError) as raised:
        check_exposure(
            calibration, ExposureDeclaration("rx1day", "mm", "hazard_catalog")
        )
    assert raised.value.code == "unit_mismatch"
    with pytest.raises(ImpactCalibrationError) as raised:
        check_exposure(
            calibration,
            ExposureDeclaration("precipitation", "mm/day", "hazard_catalog"),
        )
    assert raised.value.code == "variable_mismatch"


def test_response_labels_keep_free_text_semantics() -> None:
    calibration = ImpactCalibration(
        exposure_variable="flood_depth",
        exposure_unit="m",
        impact_metric="damage_ratio",
        impact_unit="1",
    )
    labels = check_response_labels(
        calibration, value_unit="fraction", value_semantics="building damage"
    )
    assert labels.impact_metric == "damage_ratio"
    assert labels.value_semantics == "building damage"
    with pytest.raises(ImpactCalibrationError) as raised:
        check_response_labels(calibration, value_unit="m")
    assert raised.value.code == "unit_mismatch"


def test_missing_manifest_factor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = RegistryImpactReference.pinned("rx1day")
    monkeypatch.setattr(
        "crc_sdk.impacts.config.impact_registry_manifest",
        lambda: {"factors": []},
    )
    with pytest.raises(ImpactCalibrationError) as raised:
        reference.calibration()
    assert raised.value.code == "unknown_exposure_semantics"
    assert "rx1day" in str(raised.value)


def test_cooling_calibration_requires_a_branch() -> None:
    reference = RegistryImpactReference.pinned("cooling_degree_days")
    with pytest.raises(ImpactCalibrationError) as raised:
        reference.calibration()
    assert raised.value.code == "branch_required"
    agricultural = reference.calibration("agricultural")
    assert agricultural.impact_metric == "crop_yield_change"
    assert agricultural.exposure_unit == "K.d"
    other = reference.calibration("non_agricultural")
    assert other.impact_metric == "normalized_cooling_cost"
    assert other.exposure_variable == agricultural.exposure_variable
