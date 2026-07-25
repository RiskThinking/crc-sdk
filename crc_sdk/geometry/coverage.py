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

    Only polygon ids are carried through expansion. Prefer
    ``expand_polygon_candidates`` / ``polyfill_wkb`` with ``COVERS`` when
    CGAZ admin-lookup parity is required (DuckDB overlap can under-fill).
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
    """SQL yielding (hex_id, geom) for distinct candidate cells."""
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


def build_hex_counts_sql(
    candidates_relation: str,
    *,
    hex_col: str = "hex_id",
) -> str:
    """Per-cell candidate counts for single-cell coverage optimization."""
    return f"""
        SELECT {hex_col}, COUNT(*) AS cnt
        FROM {candidates_relation}
        GROUP BY {hex_col}
    """


def _clipped_intersection_pct() -> str:
    """Intersection fraction after clipping admin geom to the hex envelope."""
    return """
        ST_Area(
            ST_Intersection(
                ST_MakeValid(h.geom),
                ST_MakeValid(ST_Intersection(a.geom, ST_Envelope(h.geom)))
            )
        ) / ST_Area(h.geom) AS pct
    """


def _with_hex_counts_cte(
    candidates_relation: str,
    body: str,
    *,
    hex_col: str,
    hex_counts_relation: str | None,
) -> str:
    if hex_counts_relation is not None:
        return body
    counts_sql = build_hex_counts_sql(candidates_relation, hex_col=hex_col)
    return f"""
        WITH hex_counts AS ({counts_sql})
        {body}
    """


def _hex_counts_prefix(
    candidates_relation: str,
    *,
    hex_col: str,
    hex_counts_relation: str | None,
) -> tuple[str, str]:
    """Return (optional WITH prefix including trailing comma, counts relation)."""
    if hex_counts_relation is not None:
        return "", hex_counts_relation
    return (
        f"""
            hex_counts AS (
                SELECT {hex_col}, COUNT(*) AS cnt
                FROM {candidates_relation}
                GROUP BY {hex_col}
            ),
        """,
        "hex_counts",
    )


def build_edge_hexes_sql(
    candidates_relation: str,
    polygons_relation: str,
    hex_geoms_relation: str,
    *,
    hex_col: str = "hex_id",
    poly_id_col: str = "poly_rid",
    polygon_id_col: str = "adm2_rid",
    hex_counts_relation: str | None = None,
) -> str:
    """IDs of single-candidate hexes not fully contained in their polygon."""
    counts_cte, counts = _hex_counts_prefix(
        candidates_relation,
        hex_col=hex_col,
        hex_counts_relation=hex_counts_relation,
    )
    return f"""
        WITH {counts_cte}
        single AS (
            SELECT c.{hex_col}, c.{poly_id_col}
            FROM {candidates_relation} AS c
            JOIN {counts} AS hc USING ({hex_col})
            WHERE hc.cnt = 1
        )
        SELECT DISTINCT s.{hex_col}
        FROM single AS s
        JOIN {hex_geoms_relation} AS h USING ({hex_col})
        JOIN {polygons_relation} AS a
          ON s.{poly_id_col} = a.{polygon_id_col}
        WHERE NOT ST_Contains(a.geom, h.geom)
    """


def build_coverage_sql(
    candidates_relation: str,
    polygons_relation: str,
    hex_geoms_relation: str,
    *,
    optimize_single_cell: bool = True,
    border_hexes_relation: str | None = None,
    hex_counts_relation: str | None = None,
    hex_col: str = "hex_id",
    poly_id_col: str = "poly_rid",
    polygon_id_col: str = "adm2_rid",
    adm0_col: str = "adm0_iso",
    adm1_id_col: str = "adm1_id",
    adm1_name_col: str = "adm1_name",
    adm2_id_col: str = "adm2_id",
    adm2_name_col: str = "adm2_name",
) -> str:
    """Coverage fraction of each hex covered by each intersecting admin polygon.

    With ``optimize_single_cell``, pass ``border_hexes_relation`` (edge/border
    hex IDs from ``build_edge_hexes_sql`` / ``build_border_hexes_sql``) so
    interior singles can use ``pct=1.0`` without joining hex geometries.
    """
    attrs = f"""
        a.{adm0_col} AS adm0_iso,
        a.{adm1_id_col} AS adm1_id,
        a.{adm1_name_col} AS adm1_name,
        a.{adm2_id_col} AS adm2_id,
        a.{adm2_name_col} AS adm2_name
    """
    intersection_pct = _clipped_intersection_pct()

    if not optimize_single_cell or border_hexes_relation is None:
        return f"""
            SELECT
                c.{hex_col},
                {attrs},
                {intersection_pct}
            FROM {candidates_relation} AS c
            JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
            JOIN {hex_geoms_relation} AS h ON c.{hex_col} = h.{hex_col}
        """

    counts_join = hex_counts_relation or "hex_counts"
    body = f"""
        SELECT
            c.{hex_col},
            {attrs},
            1.0 AS pct
        FROM {candidates_relation} AS c
        JOIN {counts_join} AS hc ON c.{hex_col} = hc.{hex_col}
        JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
        LEFT JOIN {border_hexes_relation} AS p ON c.{hex_col} = p.{hex_col}
        WHERE hc.cnt = 1 AND p.{hex_col} IS NULL
        UNION ALL
        SELECT
            c.{hex_col},
            {attrs},
            {intersection_pct}
        FROM {candidates_relation} AS c
        JOIN {counts_join} AS hc ON c.{hex_col} = hc.{hex_col}
        JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
        JOIN {hex_geoms_relation} AS h ON c.{hex_col} = h.{hex_col}
        WHERE hc.cnt > 1
        UNION ALL
        SELECT
            c.{hex_col},
            {attrs},
            {intersection_pct}
        FROM {candidates_relation} AS c
        JOIN {border_hexes_relation} AS p ON c.{hex_col} = p.{hex_col}
        JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
        JOIN {hex_geoms_relation} AS h ON c.{hex_col} = h.{hex_col}
    """
    return _with_hex_counts_cte(
        candidates_relation,
        body,
        hex_col=hex_col,
        hex_counts_relation=hex_counts_relation,
    )


def build_border_hexes_sql(
    candidates_relation: str,
    polygons_relation: str,
    hex_geoms_relation: str,
    country_iso: str,
    *,
    hex_col: str = "hex_id",
    adm0_col: str = "adm0_iso",
    poly_id_col: str = "poly_rid",
    polygon_id_col: str = "adm2_rid",
    hex_counts_relation: str | None = None,
) -> str:
    """Single-candidate hex IDs needing exact coverage in country-partition mode.

    Coastal/edge (not contained) plus hexes whose envelope intersects a foreign
    ADM0 envelope. Returns IDs only.
    """
    iso = sql_quote(country_iso)
    counts_cte, counts = _hex_counts_prefix(
        candidates_relation,
        hex_col=hex_col,
        hex_counts_relation=hex_counts_relation,
    )
    return f"""
        WITH {counts_cte}
        single AS (
            SELECT c.{hex_col}, c.{poly_id_col}
            FROM {candidates_relation} AS c
            JOIN {counts} AS hc USING ({hex_col})
            WHERE hc.cnt = 1
        ),
        foreign_adm0 AS (
            SELECT
                a.{adm0_col} AS adm0_iso,
                ST_Envelope(ST_Extent_Agg(a.geom)) AS bbox
            FROM {polygons_relation} AS a
            WHERE a.{adm0_col} != {iso}
            GROUP BY a.{adm0_col}
        )
        SELECT DISTINCT s.{hex_col}
        FROM single AS s
        JOIN {hex_geoms_relation} AS h USING ({hex_col})
        JOIN {polygons_relation} AS a
          ON s.{poly_id_col} = a.{polygon_id_col}
        WHERE NOT ST_Contains(a.geom, h.geom)
           OR EXISTS (
              SELECT 1
              FROM foreign_adm0 AS f
              WHERE ST_Intersects(h.geom, f.bbox)
          )
    """
