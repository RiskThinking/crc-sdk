"""Cross-module analytical workflows."""

from .distributions import (
    CURVE_COLUMNS,
    CurveSample,
    HazardPointSample,
    curve_parameters_from_row,
    distribution_from_hazard_row,
    sample_hazard_at_point,
    sample_hazard_row,
)
from .tiling import (
    OSClimateSelectionSpec,
    curve_quantiles_at,
    run_tiled_canonicalization,
    stream_curve_quantiles_to_parquet,
    tile_bounds,
)

__all__ = [
    "CURVE_COLUMNS",
    "CurveSample",
    "HazardPointSample",
    "OSClimateSelectionSpec",
    "curve_parameters_from_row",
    "curve_quantiles_at",
    "distribution_from_hazard_row",
    "run_tiled_canonicalization",
    "sample_hazard_at_point",
    "sample_hazard_row",
    "stream_curve_quantiles_to_parquet",
    "tile_bounds",
]
