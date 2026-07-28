"""Cross-module analytical workflows."""

from .tiling import (
    OSClimateSelectionSpec,
    curve_quantiles_at,
    run_tiled_canonicalization,
    stream_curve_quantiles_to_parquet,
    tile_bounds,
)

__all__ = [
    "OSClimateSelectionSpec",
    "curve_quantiles_at",
    "run_tiled_canonicalization",
    "stream_curve_quantiles_to_parquet",
    "tile_bounds",
]
