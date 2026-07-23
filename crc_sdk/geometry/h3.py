"""H3 conversion and resolution-selection interfaces."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionEstimate:
    """Measured trade-off for one candidate H3 resolution."""

    resolution: int
    coverage_error: float
    cell_count: int
