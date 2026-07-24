"""Geometry conversion and H3 resolution utilities."""

from crc_framework import lookup_continent, lookup_geography, lookup_ipcc_region

from .formats import FormatAdapter, GeoFormat
from .h3 import (
    H3Indexer,
    PolyfillMode,
    ResolutionEstimate,
    cell_polygon,
    estimate_resolutions,
    intersecting_cells,
    point_to_cell,
)

__all__ = [
    "FormatAdapter",
    "GeoFormat",
    "H3Indexer",
    "PolyfillMode",
    "ResolutionEstimate",
    "cell_polygon",
    "estimate_resolutions",
    "intersecting_cells",
    "lookup_continent",
    "lookup_geography",
    "lookup_ipcc_region",
    "point_to_cell",
]
