"""Cross-module analytical workflows."""

from .tiling import (
    OSClimateSelectionSpec,
    curve_quantiles_at,
    run_tiled_canonicalization,
    tile_bounds,
)

__all__ = [
    "OSClimateSelectionSpec",
    "curve_quantiles_at",
    "run_tiled_canonicalization",
    "tile_bounds",
]
