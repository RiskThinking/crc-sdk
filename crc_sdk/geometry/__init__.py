"""Geometry conversion and H3 resolution utilities."""

from crc_framework import lookup_continent, lookup_geography, lookup_ipcc_region

from .admin import (
    LookupCatalog,
    enrich_adm2_with_adm1_sql,
    write_lookup_contract,
    write_partitioned_lookup,
)
from .coverage import (
    build_border_hexes_sql,
    build_candidates_sql,
    build_coverage_sql,
    materialize_cell_geometries_sql,
    missing_cell_wkt_sql,
)
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
    "LookupCatalog",
    "PolyfillMode",
    "ResolutionEstimate",
    "build_border_hexes_sql",
    "build_candidates_sql",
    "build_coverage_sql",
    "cell_polygon",
    "enrich_adm2_with_adm1_sql",
    "estimate_resolutions",
    "intersecting_cells",
    "lookup_continent",
    "lookup_geography",
    "lookup_ipcc_region",
    "materialize_cell_geometries_sql",
    "missing_cell_wkt_sql",
    "point_to_cell",
    "write_lookup_contract",
    "write_partitioned_lookup",
]
