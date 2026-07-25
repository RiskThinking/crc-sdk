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

    For complex real-world multipolygons (e.g. CGAZ ADM), DuckDB's experimental
    overlap polyfill can under-fill relative to h3ronpy Covers. Prefer
    ``expand_polygon_candidates`` / ``polyfill_wkb`` with ``COVERS`` when
    published-lookup parity is required.
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


def build_hex_counts_sql(
    candidates_relation: str,
    *,
    hex_col: str = "hex_id",
) -> str:
    """Per-cell candidate counts used by single-cell coverage optimization."""
    return f"""
        SELECT {hex_col}, COUNT(*) AS cnt
        FROM {candidates_relation}
        GROUP BY {hex_col}
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

    Admin polygons are assumed already validated (e.g. by
    ``enrich_adm2_with_adm1_sql``). Hex geometries are validated once in the
    joined row set. Single-candidate full coverage requires ``ST_Contains``.
    """
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
                ST_Area(
                    ST_Intersection(ST_MakeValid(h.geom), ST_MakeValid(a.geom))
                ) / ST_Area(h.geom) AS pct
            FROM {candidates_relation} AS c
            JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
            JOIN {hex_geoms_relation} AS h ON c.{hex_col} = h.{hex_col}
        """

    counts_join = hex_counts_relation or "hex_counts"
    border_pred = (
        f"c.{hex_col} IN (SELECT {hex_col} FROM {border_hexes_relation})"
        if border_hexes_relation is not None
        else "FALSE"
    )
    # Single pass: evaluate containment once, then CASE for pct.
    body = f"""
        SELECT
            j.{hex_col},
            j.adm0_iso,
            j.adm1_id,
            j.adm1_name,
            j.adm2_id,
            j.adm2_name,
            CASE
                WHEN j.cnt = 1 AND j.contained AND NOT j.is_border THEN 1.0
                ELSE ST_Area(
                    ST_Intersection(
                        ST_MakeValid(j.hex_geom),
                        ST_MakeValid(j.poly_geom)
                    )
                ) / ST_Area(j.hex_geom)
            END AS pct
        FROM (
            SELECT
                c.{hex_col},
                {attrs},
                hc.cnt,
                a.geom AS poly_geom,
                h.geom AS hex_geom,
                ST_Contains(a.geom, h.geom) AS contained,
                ({border_pred}) AS is_border
            FROM {candidates_relation} AS c
            JOIN {counts_join} AS hc ON c.{hex_col} = hc.{hex_col}
            JOIN {polygons_relation} AS a ON c.{poly_id_col} = a.{polygon_id_col}
            JOIN {hex_geoms_relation} AS h ON c.{hex_col} = h.{hex_col}
        ) AS j
    """
    if hex_counts_relation is not None:
        return body

    counts_sql = build_hex_counts_sql(candidates_relation, hex_col=hex_col)
    return f"""
        WITH hex_counts AS ({counts_sql})
        {body}
    """


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
) -> str:
    """Single-candidate hexes that need exact intersection coverage.

    Includes coastal / dataset-edge hexes that are not fully contained in their
    candidate polygon, and hexes whose envelope intersects a foreign ADM0
    envelope (cheap probe vs scanning every foreign ADM2 polygon).
    """
    iso = sql_quote(country_iso)
    return f"""
        WITH hex_counts AS (
            SELECT {hex_col}, COUNT(*) AS cnt
            FROM {candidates_relation}
            GROUP BY {hex_col}
        ),
        single AS (
            SELECT c.{hex_col}, c.{poly_id_col}
            FROM {candidates_relation} AS c
            JOIN hex_counts AS hc USING ({hex_col})
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
