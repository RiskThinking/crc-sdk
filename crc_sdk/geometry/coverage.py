"""DuckDB SQL builders for H3 cell coverage against polygon relations."""

from __future__ import annotations

from pathlib import Path

from crc_sdk.connectors.duckdb.connection import sql_quote


def build_candidates_sql(
    polygon_relation: str,
    resolution: int,
    *,
    geom_col: str = "geom",
    id_col: str = "poly_rid",
    hex_col: str = "hex_id",
) -> str:
    """Overlap-polyfill polygons into string H3 cell candidates.

    Only the polygon id is carried through the expansion. Geometry is dumped to
    parts for polyfill and then discarded so cell unnest does not replicate
    large multipolygons onto every candidate row.
    """
    return f"""
        WITH parts AS (
            SELECT src.{id_col} AS poly_rid,
                   (unnest(ST_Dump(src.{geom_col}))).geom AS _poly
            FROM {polygon_relation} AS src
        ),
        expanded AS (
            SELECT poly_rid,
                   h3_polygon_wkt_to_cells_experimental(
                       ST_AsText(_poly), {int(resolution)}, 'overlap'
                   ) AS _cells
            FROM parts
        )
        SELECT DISTINCT
               h3_h3_to_string(CAST(cell AS UBIGINT)) AS {hex_col},
               poly_rid
        FROM expanded AS e, UNNEST(e._cells) AS _u(cell)
    """


def materialize_cell_geometries_sql(
    candidates_relation: str,
    *,
    hex_col: str = "hex_id",
    cache_parquet: str | Path | None = None,
) -> str:
    """SQL that yields (hex_id, geom) for distinct candidate cells.

    When ``cache_parquet`` is provided, cached WKT rows are joined with predicate
    pushdown and only missing cells are generated.
    """
    boundary = f"ST_GeomFromText(h3_cell_to_boundary_wkt(h3_string_to_h3({hex_col})))"
    if cache_parquet is None:
        return f"""
            SELECT {hex_col}, {boundary} AS geom
            FROM (SELECT DISTINCT {hex_col} FROM {candidates_relation}) AS cells
        """

    cached = sql_quote(cache_parquet)
    return f"""
        WITH needed AS (
            SELECT DISTINCT {hex_col} FROM {candidates_relation}
        ),
        from_cache AS (
            SELECT n.{hex_col}, ST_GeomFromText(c.hex_wkt) AS geom
            FROM needed AS n
            JOIN read_parquet({cached}) AS c USING ({hex_col})
        ),
        missing AS (
            SELECT n.{hex_col}
            FROM needed AS n
            LEFT JOIN from_cache AS f USING ({hex_col})
            WHERE f.{hex_col} IS NULL
        ),
        generated AS (
            SELECT {hex_col}, {boundary} AS geom
            FROM missing
        )
        SELECT * FROM from_cache
        UNION ALL
        SELECT * FROM generated
    """


def missing_cell_wkt_sql(
    candidates_relation: str,
    *,
    hex_col: str = "hex_id",
    present_relation: str | None = None,
) -> str:
    """SQL yielding (hex_id, hex_wkt) for cells not already present."""
    if present_relation is None:
        source = f"SELECT DISTINCT {hex_col} FROM {candidates_relation}"
    else:
        source = f"""
            SELECT DISTINCT c.{hex_col}
            FROM {candidates_relation} AS c
            LEFT JOIN {present_relation} AS p USING ({hex_col})
            WHERE p.{hex_col} IS NULL
        """
    return f"""
        SELECT {hex_col},
               h3_cell_to_boundary_wkt(h3_string_to_h3({hex_col})) AS hex_wkt
        FROM ({source}) AS missing
    """


def build_coverage_sql(
    candidates_relation: str,
    polygons_relation: str,
    hex_geoms_relation: str,
    *,
    optimize_single_cell: bool = True,
    border_hexes_relation: str | None = None,
    hex_col: str = "hex_id",
    poly_id_col: str = "poly_rid",
    polygon_id_col: str = "adm2_rid",
    adm0_col: str = "adm0_iso",
    adm1_id_col: str = "adm1_id",
    adm1_name_col: str = "adm1_name",
    adm2_id_col: str = "adm2_id",
    adm2_name_col: str = "adm2_name",
) -> str:
    """Coverage fraction of each hex covered by each intersecting admin polygon."""
    attrs = f"""
        a.{adm0_col} AS adm0_iso,
        a.{adm1_id_col} AS adm1_id,
        a.{adm1_name_col} AS adm1_name,
        a.{adm2_id_col} AS adm2_id,
        a.{adm2_name_col} AS adm2_name
    """
    if not optimize_single_cell:
        return f"""
            SELECT
                c.{hex_col},
                {attrs},
                ST_Area(ST_Intersection(ST_MakeValid(h.geom), ST_MakeValid(a.geom)))
                    / ST_Area(h.geom) AS pct
            FROM {candidates_relation} AS c
            JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
            JOIN {hex_geoms_relation} AS h ON c.{hex_col} = h.{hex_col}
        """

    counts = f"""
        SELECT {hex_col}, COUNT(*) AS cnt
        FROM {candidates_relation}
        GROUP BY {hex_col}
    """
    if border_hexes_relation is None:
        single_pred = "hc.cnt = 1"
        exact_pred = "hc.cnt > 1"
    else:
        single_pred = (
            f"hc.cnt = 1 AND c.{hex_col} NOT IN "
            f"(SELECT {hex_col} FROM {border_hexes_relation})"
        )
        exact_pred = (
            f"hc.cnt > 1 OR c.{hex_col} IN "
            f"(SELECT {hex_col} FROM {border_hexes_relation})"
        )

    return f"""
        WITH hex_counts AS ({counts})
        SELECT
            c.{hex_col},
            {attrs},
            1.0 AS pct
        FROM {candidates_relation} AS c
        JOIN hex_counts AS hc USING ({hex_col})
        JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
        WHERE {single_pred}
        UNION ALL
        SELECT
            c.{hex_col},
            {attrs},
            ST_Area(ST_Intersection(ST_MakeValid(h.geom), ST_MakeValid(a.geom)))
                / ST_Area(h.geom) AS pct
        FROM {candidates_relation} AS c
        JOIN hex_counts AS hc USING ({hex_col})
        JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
        JOIN {hex_geoms_relation} AS h ON c.{hex_col} = h.{hex_col}
        WHERE {exact_pred}
    """


def build_border_hexes_sql(
    candidates_relation: str,
    polygons_relation: str,
    hex_geoms_relation: str,
    country_iso: str,
    *,
    hex_col: str = "hex_id",
    adm0_col: str = "adm0_iso",
) -> str:
    """Single-candidate hexes that still intersect a foreign ADM0 polygon."""
    return f"""
        WITH hex_counts AS (
            SELECT {hex_col}, COUNT(*) AS cnt
            FROM {candidates_relation}
            GROUP BY {hex_col}
        )
        SELECT DISTINCT hc.{hex_col}
        FROM hex_counts AS hc
        JOIN {hex_geoms_relation} AS h USING ({hex_col})
        WHERE hc.cnt = 1
          AND EXISTS (
              SELECT 1
              FROM {polygons_relation} AS a
              WHERE ST_Intersects(h.geom, a.geom)
                AND a.{adm0_col} != {sql_quote(country_iso)}
          )
    """
