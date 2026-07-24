"""Geometry conversion and H3 resolution utilities."""

from crc_framework import lookup_continent, lookup_geography, lookup_ipcc_region

from .h3 import (
    ResolutionEstimate,
    cell_polygon,
    estimate_resolutions,
    intersecting_cells,
    point_to_cell,
)

__all__ = [
    "ResolutionEstimate",
    "cell_polygon",
    "estimate_resolutions",
    "intersecting_cells",
    "lookup_continent",
    "lookup_geography",
    "lookup_ipcc_region",
    "point_to_cell",
]
