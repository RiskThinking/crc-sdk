"""NOAA GHCNh station observation workflow."""

from dataclasses import dataclass
from typing import Literal

from crc_sdk.providers.ghcnh import GHCNhCompleteness
from crc_sdk.providers.isd import ISDCompleteness

from .stations import StationObservationPlan


@dataclass(frozen=True)
class GHCNhPlan(StationObservationPlan):
    """GHCNh temperatures with GHCN identifiers and source-aware quality rules."""

    provider: Literal["isd", "ghcnh"] = "ghcnh"
    completeness: ISDCompleteness | GHCNhCompleteness = GHCNhCompleteness()
