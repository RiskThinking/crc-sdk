"""Geometry conversion and H3 resolution utilities."""

from crc_framework import lookup_continent, lookup_geography, lookup_ipcc_region

from .admin import (
    LookupCatalog,
    enrich_adm2_with_adm1,
    enrich_adm2_with_adm1_sql,
    write_lookup_contract,
    write_partitioned_lookup,
)
from .coverage import (
    build_border_hexes_sql,
    build_candidates_sql,
    build_coverage_sql,
    build_edge_hexes_sql,
    build_hex_counts_sql,
    materialize_cell_geometries_sql,
    materialize_edge_hexes,
    missing_cell_wkt_sql,
    recommend_parent_resolution,
    write_exploded_coverage,
    write_hierarchical_coverage,
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
from .vector import VectorContainment, expand_polygon_candidates, polyfill_wkb

__all__ = [
    "FormatAdapter",
    "GeoFormat",
    "H3Indexer",
    "LookupCatalog",
    "PolyfillMode",
    "ResolutionEstimate",
    "VectorContainment",
    "build_border_hexes_sql",
    "build_candidates_sql",
    "build_coverage_sql",
    "build_edge_hexes_sql",
    "build_hex_counts_sql",
    "cell_polygon",
    "enrich_adm2_with_adm1",
    "enrich_adm2_with_adm1_sql",
    "estimate_resolutions",
    "expand_polygon_candidates",
    "intersecting_cells",
    "lookup_continent",
    "lookup_geography",
    "lookup_ipcc_region",
    "materialize_cell_geometries_sql",
    "materialize_edge_hexes",
    "missing_cell_wkt_sql",
    "point_to_cell",
    "polyfill_wkb",
    "recommend_parent_resolution",
    "write_exploded_coverage",
    "write_hierarchical_coverage",
    "write_lookup_contract",
    "write_partitioned_lookup",
]
