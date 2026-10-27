"""Pinned impact contract artifact published by ``crc-framework``.

The SDK does not maintain its own copy of the impact schemas or conformance
fixtures. It pins the framework artifact by manifest digest so that a schema,
default or error-code change in the framework is detected rather than
silently diverging from SDK config models.
"""

from __future__ import annotations

import importlib
from functools import cache
from types import MappingProxyType, ModuleType
from typing import Any

REQUIRED_IMPACT_CONTRACT_ARTIFACT = "crc-impact-contracts"
REQUIRED_IMPACT_CONTRACT_VERSION = "1.0.0"
REQUIRED_IMPACT_CONTRACT_MANIFEST_SHA256 = (
    "9f09f489ed72891455d9eeffa8e17e9226b73c121b6e0e0132d766ed21101aed"
)
IMPACT_CONTRACTS = (
    "impact_definition",
    "impact_evaluation_record",
    "registry_impact_reference",
    "spatial_combination_policy",
    "spatial_impact_evaluation_record",
)


class ImpactContractUnavailable(RuntimeError):
    """The framework cannot supply the pinned, verified impact contract artifact."""


def impact_contracts() -> ModuleType:
    """Return the verified framework contract module.

    Raises ``ImpactContractUnavailable`` when the installed framework does not
    ship the artifact, ships a different version or manifest, or has unreadable
    or corrupted artifact files.
    """
    try:
        contracts = importlib.import_module("crc_framework.contracts")
    except ImportError as error:
        raise ImpactContractUnavailable(
            "the installed crc-framework does not provide impact contracts "
            f"{REQUIRED_IMPACT_CONTRACT_VERSION}"
        ) from error
    artifact = getattr(contracts, "IMPACT_CONTRACT_ARTIFACT", None)
    version = getattr(contracts, "IMPACT_CONTRACT_VERSION", None)
    if (artifact, version) != (
        REQUIRED_IMPACT_CONTRACT_ARTIFACT,
        REQUIRED_IMPACT_CONTRACT_VERSION,
    ):
        raise ImpactContractUnavailable(
            f"crc-framework provides {artifact} {version}; the SDK requires "
            f"{REQUIRED_IMPACT_CONTRACT_ARTIFACT} {REQUIRED_IMPACT_CONTRACT_VERSION}"
        )
    try:
        digest = contracts.impact_contract_manifest_sha256()
        if digest != REQUIRED_IMPACT_CONTRACT_MANIFEST_SHA256:
            raise ImpactContractUnavailable(
                f"crc-framework impact contract manifest {digest} differs from the "
                f"pinned {REQUIRED_IMPACT_CONTRACT_MANIFEST_SHA256}"
            )
        contracts.verify_impact_contracts()
    except (OSError, ValueError) as error:
        raise ImpactContractUnavailable(
            f"crc-framework impact contract artifact could not be verified: {error}"
        ) from error
    return contracts


@cache
def unit_aliases() -> MappingProxyType[str, str]:
    """Documented unit tokens that compare equal, such as ``days`` and ``d``."""
    catalog = impact_contracts().load_impact_contract_file("catalogs/unit-aliases.json")
    aliases = {str(key): str(value) for key, value in catalog["aliases"].items()}
    return MappingProxyType(aliases)


@cache
def hazard_exposures() -> tuple[MappingProxyType[str, Any], ...]:
    """Framework catalog of hazard ids that denote a registry factor."""
    catalog = impact_contracts().load_impact_contract_file(
        "catalogs/hazard-exposures.json"
    )
    entries = []
    for entry in catalog["entries"]:
        frozen = {
            "id": str(entry["id"]),
            "aliases": tuple(str(alias) for alias in entry["aliases"]),
            "factor": str(entry["factor"]),
        }
        entries.append(MappingProxyType(frozen))
    return tuple(entries)
