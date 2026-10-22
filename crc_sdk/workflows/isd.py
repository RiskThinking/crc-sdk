"""NOAA ISD historical station observation workflow."""

from dataclasses import dataclass

from .stations import StationObservationPlan


@dataclass(frozen=True)
class ISDPlan(StationObservationPlan):
    """Historical ISD temperatures with numeric USAF + WBAN identifiers."""
