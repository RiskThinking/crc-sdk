"""GeoParquet -> PMTiles, streamed through DuckDB directly into tippecanoe.

Prefers one streaming tiling pass over an entire source (a single file or a
Hive-partitioned dataset glob); raises a clear, actionable error instead of
auto-sharding when a pre-flight budget check estimates the source won't fit
available scratch disk. Requires ``tippecanoe``/``tile-join`` on ``PATH`` --
see :func:`require_tippecanoe`/:func:`require_tile_join` -- these are OS
binaries assumed present on the runtime image, not a pip extra.
"""

from __future__ import annotations

from .archive import PMTilesBuild, PMTilesLayer, PMTilesResult, ZoomRange
from .binaries import require_tile_join, require_tippecanoe
from .budget import TilingBudget, check_tiling_budget, measure_source
from .presets import (
    AREAS,
    POINTS,
    POLYGONS,
    POLYGONS_CAPPED,
    TippecanoePreset,
    coerce_zoom_range,
    tippecanoe_command,
)

__all__ = [
    "AREAS",
    "POINTS",
    "POLYGONS",
    "POLYGONS_CAPPED",
    "PMTilesBuild",
    "PMTilesLayer",
    "PMTilesResult",
    "TilingBudget",
    "TippecanoePreset",
    "ZoomRange",
    "check_tiling_budget",
    "coerce_zoom_range",
    "measure_source",
    "require_tile_join",
    "require_tippecanoe",
    "tippecanoe_command",
]
