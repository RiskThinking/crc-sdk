"""H3 conversion, resolution selection, and DuckDB spatial indexing utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from crc_sdk.connectors.duckdb.connection import (
    DuckDBConnection,
    default_work_dir,
    ensure_extensions,
)
from crc_sdk.geometry.formats import FormatAdapter, GeoFormat

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


class PolyfillMode(str, Enum):
    CENTROID = "centroid"
    OVERLAP = "overlap"
    CONTAINS = "contains"
    EXACT_INTERSECTION = "exact_intersection"


_CONTAINMENT = {
    PolyfillMode.OVERLAP: "overlap",
    PolyfillMode.CONTAINS: "full",
    PolyfillMode.EXACT_INTERSECTION: "overlap",
}


class H3Indexer:
    """Out-of-core DuckDB-native H3 indexer for vector geometries."""

    def __init__(
        self,
        con: DuckDBPyConnection | None = None,
        *,
        work_dir: str | Path | None = None,
    ) -> None:
        # An explicit connection means the caller is already in control; only
        # build (and resource-tune) one when they didn't supply their own.
        self.con = con or DuckDBConnection.for_analytics(
            work_dir or default_work_dir(), extensions=("spatial", "h3")
        ).connect()
        ensure_extensions(self.con, "spatial", "h3")

    @staticmethod
    def _polyfill_sql(
        input_relation_sql: str,
        resolution: int,
        geom_col: str,
        h3_col: str,
        containment: str,
        *,
        as_string: bool = False,
        preserve_geom: bool = True,
    ) -> str:
        """Dump multiparts, then polyfill each polygon with the requested containment.

        DuckDB's polygon polyfill does not reliably handle MULTIPOLYGON WKT, so
        geometries are decomposed with ``ST_Dump`` first. Geometry is dropped from
        the expansion path unless ``preserve_geom`` is set, so cell unnest does not
        replicate large polygons onto every output row.
        """
        keep_geom = f", src.{geom_col}" if preserve_geom else ""
        cell_expr = (
            f"h3_h3_to_string(CAST(cell AS UBIGINT)) AS {h3_col}"
            if as_string
            else f"CAST(cell AS UBIGINT) AS {h3_col}"
        )
        return f"""
            WITH parts AS (
                SELECT src.* EXCLUDE ({geom_col}){keep_geom},
                       (unnest(ST_Dump(src.{geom_col}))).geom AS _poly
                FROM {input_relation_sql} AS src
            ),
            expanded AS (
                SELECT * EXCLUDE (_poly),
                       h3_polygon_wkt_to_cells_experimental(
                           ST_AsText(_poly), {resolution}, '{containment}'
                       ) AS _cells
                FROM parts
            ),
            flattened AS (
                SELECT e.* EXCLUDE (_cells), {cell_expr}
                FROM expanded AS e, UNNEST(e._cells) AS _u(cell)
            )
            SELECT DISTINCT * FROM flattened
        """

    @staticmethod
    def build_h3_query(
        input_relation_sql: str,
        resolution: int,
        mode: PolyfillMode = PolyfillMode.OVERLAP,
        geom_col: str = "geometry",
        h3_col: str = "h3_index",
        *,
        as_string: bool = False,
        preserve_geom: bool = True,
    ) -> str:
        """Construct un-materialized SQL streaming geometry through H3 polyfill."""
        mode = PolyfillMode(mode.lower()) if isinstance(mode, str) else mode

        if mode == PolyfillMode.CENTROID:
            lat = f"ST_Y(ST_Centroid({geom_col}))"
            lng = f"ST_X(ST_Centroid({geom_col}))"
            cell_expr = (
                f"h3_h3_to_string(h3_latlng_to_cell({lat}, {lng}, {resolution}))"
                if as_string
                else f"h3_latlng_to_cell({lat}, {lng}, {resolution})"
            )
            return f"""
                SELECT *,
                       {cell_expr} AS {h3_col}
                FROM {input_relation_sql}
            """

        if mode not in _CONTAINMENT:
            raise ValueError(f"Unsupported PolyfillMode: {mode}")

        # Exact intersection / contains need the source geom for predicates.
        keep_geom = preserve_geom or mode in (
            PolyfillMode.EXACT_INTERSECTION,
            PolyfillMode.CONTAINS,
        )
        base_sql = H3Indexer._polyfill_sql(
            input_relation_sql,
            resolution,
            geom_col,
            h3_col,
            _CONTAINMENT[mode],
            as_string=as_string,
            preserve_geom=keep_geom,
        )
        if mode == PolyfillMode.EXACT_INTERSECTION:
            boundary = (
                f"ST_GeomFromText(h3_cell_to_boundary_wkt(h3_string_to_h3({h3_col})))"
                if as_string
                else f"ST_GeomFromText(h3_cell_to_boundary_wkt({h3_col}))"
            )
            return f"""
                WITH indexed AS ({base_sql})
                SELECT *,
                       CASE
                           WHEN ST_Area({geom_col}) > 0 THEN
                               ST_Area(ST_Intersection({geom_col}, {boundary}))
                               / ST_Area({geom_col})
                           ELSE 0.0
                       END AS intersection_ratio
                FROM indexed
            """
        return base_sql

    def build_h3_query_from_file(
        self,
        file_path: str,
        fmt: GeoFormat,
        resolution: int,
        mode: PolyfillMode = PolyfillMode.OVERLAP,
        *,
        geometry_column: str = "geometry",
        h3_col: str = "h3_index",
        as_string: bool = False,
        preserve_geom: bool = True,
    ) -> str:
        """Polyfill any :class:`GeoFormat` source straight to H3 cells.

        Composes :meth:`FormatAdapter.build_read_relation` with
        :meth:`build_h3_query` for the common case of indexing one vector
        file (GeoJSON, Shapefile, GeoParquet, ...) without hand-wiring the
        two SQL builders together. Still un-materialized, streamed SQL.
        """
        relation_sql = FormatAdapter.build_read_relation(
            self.con,
            file_path,
            fmt,
            geometry_column=geometry_column,
            preserve_source_geom=False,
        )
        return self.build_h3_query(
            relation_sql,
            resolution,
            mode,
            geom_col=geometry_column,
            h3_col=h3_col,
            as_string=as_string,
            preserve_geom=preserve_geom,
        )


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


# Average hexagon edge lengths in km (res 0-15), excluding pentagons.
# https://h3geo.org/docs/core-library/restable#edge-lengths
_H3_AVERAGE_EDGE_LENGTH_KM = (
    1281.256011,
    483.0568391,
    182.5129565,
    68.97922179,
    26.07175968,
    9.854090990,
    3.724532667,
    1.406475763,
    0.531414010,
    0.200786148,
    0.075863783,
    0.028663897,
    0.010830188,
    0.004092010,
    0.001546100,
    0.000584169,
)

# Absorbs H3 edge-length variation around the tabulated averages and residual
# projection distortion in max_pixel_spacing_m's coverage guarantee.
_COVERAGE_SAFETY = 0.85


def average_edge_length_m(resolution: int) -> float:
    """Average H3 hexagon edge length in meters at `resolution` (0-15)."""
    try:
        return _H3_AVERAGE_EDGE_LENGTH_KM[resolution] * 1000.0
    except IndexError as error:
        raise ValueError(
            f"unsupported H3 resolution {resolution} (supported: 0-15)"
        ) from error


def max_pixel_spacing_m(resolution: int) -> float:
    """Largest per-axis sample spacing that still guarantees a point in every
    cell of `resolution` (see :func:`pixel_grid_resolution` for the proof)."""
    return math.sqrt(6) / 2 * average_edge_length_m(resolution) * _COVERAGE_SAFETY


def pixel_grid_resolution(pixel_size_m: float, *, max_subsample: float = 2.0) -> int:
    """Finest H3 resolution covered by a pixel grid of `pixel_size_m` spacing.

    A rectangular sample grid with per-axis spacing s contains a point in any
    region enclosing a disc of radius r >= s*sqrt(2)/2. A hexagon of edge e
    encloses a disc of radius (sqrt(3)/2)*e, so s <= (sqrt(6)/2)*e guarantees
    at least one sample point per cell. `max_subsample` bounds how many
    subsample points per pixel axis the caller is willing to place to reach a
    finer resolution than the pixel spacing alone would allow.
    """
    if max_subsample <= 0:
        raise ValueError("max_subsample must be positive")
    for resolution in range(len(_H3_AVERAGE_EDGE_LENGTH_KM) - 1, -1, -1):
        if max_pixel_spacing_m(resolution) * max_subsample >= pixel_size_m:
            return resolution
    return 0


def subsample_offsets(count: int) -> np.ndarray:
    """Centered fractional pixel offsets for `count` samples along one axis."""
    if count < 1:
        raise ValueError("count must be positive")
    return -0.5 + (np.arange(count, dtype=np.float64) + 0.5) / count


def reduce_h3_values(
    cells: Any,
    values: Any,
    *,
    reduce: Literal["max", "min"] = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated H3 cell ids to one value per cell.

    `reduce` is applied associatively, so this is safe to call once per
    partial batch and again to merge those partial results.
    """
    if reduce not in ("max", "min"):
        raise ValueError(f"unsupported reduce {reduce!r}; use 'max' or 'min'")
    cells = np.asarray(cells, dtype=np.uint64)
    values = np.asarray(values, dtype=np.float64)
    if len(cells) == 0:
        return cells, values
    order = np.argsort(cells, kind="stable")
    cells = cells[order]
    values = values[order]
    starts = np.flatnonzero(np.r_[True, cells[1:] != cells[:-1]])
    ufunc = np.maximum if reduce == "max" else np.minimum
    return cells[starts], ufunc.reduceat(values, starts)


def sample_grid_to_h3(
    longitudes: Any,
    latitudes: Any,
    values: Any,
    *,
    resolution: int,
    reduce: Literal["max", "min"] = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce WGS84 point samples to one value per H3 cell.

    Vectorized via h3ronpy; requires `pip install crc-sdk[geometry]`.
    """
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    try:
        from h3ronpy.vector import coordinates_to_cells  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "Grid-to-H3 sampling requires `pip install crc-sdk[geometry]`"
        ) from error
    cells = coordinates_to_cells(
        np.asarray(latitudes, dtype=np.float64),
        np.asarray(longitudes, dtype=np.float64),
        resolution,
    )
    return reduce_h3_values(cells, values, reduce=reduce)
