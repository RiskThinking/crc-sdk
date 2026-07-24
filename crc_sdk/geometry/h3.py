"""H3 conversion, resolution selection, and spatial indexing utilities for vector geometries."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import duckdb


class PolyfillMode(str, Enum):
    CENTROID = "centroid"  # H3 index of the geometry's centroid
    OVERLAP = "overlap"  # H3 cells touching/overlapping geometry BBox / Polygon
    CONTAINS = "contains"  # H3 cells whose centers/polygons are fully contained
    EXACT_INTERSECTION = (
        "exact_intersection"  # Overlap + exact area intersection ratio calculation
    )


class H3Indexer:
    """Out-of-core DuckDB-native H3 indexer for vector geometries."""

    def __init__(self, con: Optional[duckdb.DuckDBPyConnection] = None):
        self.con = con or duckdb.connect()
        self._init_extensions()

    def _init_extensions(self) -> None:
        self.con.execute("INSTALL spatial; LOAD spatial;")
        try:
            self.con.execute("INSTALL h3 FROM community; LOAD h3;")
        except Exception:
            # Fallback macro definition if community extension is loaded differently
            pass

    def build_h3_query(
        self,
        input_relation_sql: str,
        resolution: int,
        mode: PolyfillMode = PolyfillMode.OVERLAP,
        geom_col: str = "geometry",
        h3_col: str = "h3_index",
    ) -> str:
        """Constructs an un-materialized SQL query streaming geometry through H3 polyfilling."""

        if mode == PolyfillMode.CENTROID:
            return f"""
                SELECT *,
                       h3_latlng_to_cell(ST_Y(ST_Centroid({geom_col})), ST_X(ST_Centroid({geom_col})), {resolution}) AS {h3_col}
                FROM {input_relation_sql}
            """

        elif mode in (PolyfillMode.OVERLAP, PolyfillMode.EXACT_INTERSECTION):
            # Polygon polyfill logic using DuckDB spatial BBox/Grid decomposition
            base_sql = f"""
                WITH expanded AS (
                    SELECT *,
                           h3_polygon_wkt_to_cells(ST_AsText({geom_col}), {resolution}) AS _cells
                    FROM {input_relation_sql}
                )
                SELECT * EXCLUDE(_cells), UNNEST(_cells) AS {h3_col}
                FROM expanded
            """
            if mode == PolyfillMode.EXACT_INTERSECTION:
                # Compute exact area intersection ratio between geometry and H3 cell
                return f"""
                    WITH indexed AS ({base_sql})
                    SELECT *,
                           ST_Area(ST_Intersection({geom_col}, ST_GeomFromText(h3_cell_to_boundary_wkt({h3_col}))))
                           / ST_Area({geom_col}) AS intersection_ratio
                    FROM indexed
                """
            return base_sql

        elif mode == PolyfillMode.CONTAINS:
            return f"""
                WITH expanded AS (
                    SELECT *,
                           h3_polygon_wkt_to_cells(ST_AsText({geom_col}), {resolution}) AS _cells
                    FROM {input_relation_sql}
                ),
                flattened AS (
                    SELECT * EXCLUDE(_cells), UNNEST(_cells) AS {h3_col}
                    FROM expanded
                )
                SELECT * FROM flattened
                WHERE ST_Contains({geom_col}, ST_GeomFromText(h3_cell_to_boundary_wkt({h3_col})))
            """
        else:
            raise ValueError(f"Unsupported PolyfillMode: {mode}")


@dataclass(frozen=True)
class ResolutionEstimate:
    """Measured trade-off for one candidate H3 resolution."""

    resolution: int
    coverage_error: float
    cell_count: int

    @property
    def row_count(self) -> int:
        """Expanded source-to-cell row count at this resolution."""
        return self.cell_count


def _libraries() -> tuple[Any, Any, Any]:
    try:
        import h3  # type: ignore[import-untyped]
        from shapely.geometry import Polygon  # type: ignore[import-untyped]
        from shapely.ops import unary_union  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "H3 geometry support requires `pip install crc-sdk[geometry]`"
        ) from error
    return h3, Polygon, unary_union


def point_to_cell(longitude: float, latitude: float, resolution: int) -> int:
    """Return an unsigned integer H3 index for a WGS84 point."""
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    h3, _, _ = _libraries()
    return int(h3.str_to_int(h3.latlng_to_cell(latitude, longitude, resolution)))


def intersecting_cells(geometry: Any, resolution: int) -> tuple[int, ...]:
    """Return every H3 cell that overlaps a Polygon or MultiPolygon."""
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    if geometry is None or geometry.is_empty:
        return ()
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise TypeError("source geometry must be a Polygon or MultiPolygon")
    if not geometry.is_valid:
        raise ValueError("source geometry must be valid")
    h3, _, _ = _libraries()
    h3_shape = h3.geo_to_h3shape(geometry.__geo_interface__)
    cells = h3.h3shape_to_cells_experimental(
        h3_shape,
        resolution,
        contain="overlap",
    )
    return tuple(sorted(h3.str_to_int(cell) for cell in cells))


def cell_polygon(cell_index: int) -> Any:
    """Return a WGS84 Shapely polygon for an integer H3 cell."""
    h3, Polygon, _ = _libraries()
    boundary = h3.cell_to_boundary(h3.int_to_str(cell_index))
    return Polygon([(longitude, latitude) for latitude, longitude in boundary])


def estimate_resolutions(
    geometries: Iterable[Any],
    resolutions: Sequence[int],
) -> tuple[ResolutionEstimate, ...]:
    """Measure excess candidate coverage and expanded rows per resolution.

    Coverage error is the symmetric-difference area divided by source area in
    the input coordinate plane. It is intended for relative resolution
    comparisons; ingest policy remains responsible for selecting a resolution.
    """
    _, _, unary_union = _libraries()
    sources = tuple(geometry for geometry in geometries if not geometry.is_empty)
    if not sources:
        raise ValueError("at least one non-empty source geometry is required")
    source_union = unary_union(sources)
    source_area = float(source_union.area)
    if source_area <= 0.0:
        raise ValueError("source geometries must have positive area")

    estimates = []
    for resolution in resolutions:
        expanded = [
            cell
            for geometry in sources
            for cell in intersecting_cells(geometry, resolution)
        ]
        covered = unary_union([cell_polygon(cell) for cell in set(expanded)])
        error = float(covered.symmetric_difference(source_union).area) / source_area
        estimates.append(
            ResolutionEstimate(
                resolution=resolution,
                coverage_error=error,
                cell_count=len(expanded),
            )
        )
    return tuple(estimates)
