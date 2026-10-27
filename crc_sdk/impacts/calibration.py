"""Compare a dataset's declared exposure with an impact calibration.

Unit comparison uses the framework alias catalog. Hazard identity comes from
the framework hazard-exposure catalog. An explicit declaration overrides it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from crc_framework import impact_registry_manifest

from crc_sdk.impacts._contracts import hazard_exposures, unit_aliases
from crc_sdk.impacts._types import thaw
from crc_sdk.impacts.config import ImpactCalibration
from crc_sdk.impacts.errors import ImpactCalibrationError


@dataclass(frozen=True)
class ExposureDeclaration:
    """Exposure variable and unit established for one hazard."""

    variable: str
    unit: str
    basis: Literal["hazard_catalog", "explicit"]


@dataclass(frozen=True)
class CalibrationMatch:
    """How a calibration agreed with a declaration."""

    basis: Literal["exact", "unit_alias", "explicit_mapping"]


@dataclass(frozen=True)
class ResponseLabels:
    """Authoritative metric plus an optional caller label."""

    impact_metric: str
    impact_unit: str
    value_semantics: str | None


def normalize_unit(token: str) -> str:
    """Apply a documented alias, otherwise return the token unchanged."""
    return unit_aliases().get(token, token)


def units_equivalent(left: str, right: str) -> bool:
    """Exact equality after alias normalization. Case sensitive, no conversion."""
    return normalize_unit(left) == normalize_unit(right)


def exposure_declaration(
    hazard: str,
    *,
    mapping: Mapping[str, ExposureDeclaration] | None = None,
) -> ExposureDeclaration:
    """Resolve a hazard name to the exposure a definition must accept."""
    if mapping is not None and hazard in mapping:
        declared = mapping[hazard]
        return ExposureDeclaration(declared.variable, declared.unit, "explicit")
    for entry in hazard_exposures():
        if hazard == entry["id"] or hazard in entry["aliases"]:
            variable, unit = _factor_exposure(str(entry["factor"]))
            return ExposureDeclaration(variable, unit, "hazard_catalog")
    raise ImpactCalibrationError(
        "unknown_exposure_semantics",
        f"no exposure declaration for {hazard!r}",
    )


def check_exposure(
    calibration: ImpactCalibration, declared: ExposureDeclaration
) -> CalibrationMatch:
    """Reject a contradictory variable or unit before evaluation."""
    if calibration.exposure_variable != declared.variable:
        raise ImpactCalibrationError(
            "variable_mismatch",
            f"hazard variable {declared.variable!r} does not match "
            f"calibration variable {calibration.exposure_variable!r}",
        )
    if not units_equivalent(calibration.exposure_unit, declared.unit):
        raise ImpactCalibrationError(
            "unit_mismatch",
            f"hazard unit {declared.unit!r} does not match "
            f"calibration unit {calibration.exposure_unit!r}",
        )
    if declared.basis == "explicit":
        basis: Literal["exact", "unit_alias", "explicit_mapping"] = "explicit_mapping"
    elif calibration.exposure_unit != declared.unit:
        basis = "unit_alias"
    else:
        basis = "exact"
    return CalibrationMatch(basis=basis)


def check_response_labels(
    calibration: ImpactCalibration,
    *,
    value_unit: str | None = None,
    value_semantics: str | None = None,
) -> ResponseLabels:
    """Check a caller unit and keep free-text semantics as a label."""
    if value_unit is not None and not units_equivalent(
        value_unit, calibration.impact_unit
    ):
        raise ImpactCalibrationError(
            "unit_mismatch",
            f"value unit {value_unit!r} does not match "
            f"calibration impact unit {calibration.impact_unit!r}",
        )
    return ResponseLabels(
        calibration.impact_metric,
        calibration.impact_unit,
        value_semantics,
    )


def _factor_exposure(factor: str) -> tuple[str, str]:
    manifest = impact_registry_manifest()
    for item in manifest["factors"]:
        if item["factor"] != factor:
            continue
        calibration = thaw(item["calibration"])
        return str(calibration["exposure_variable"]), str(calibration["exposure_unit"])
    raise ImpactCalibrationError(
        "unknown_exposure_semantics",
        f"factor {factor!r} is not in the installed registry manifest",
    )
