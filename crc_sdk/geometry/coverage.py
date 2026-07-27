"""DuckDB SQL builders for H3 cell coverage against polygon relations."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from crc_sdk.connectors.duckdb.connection import sql_quote

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


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


def recommend_parent_resolution(resolution: int) -> int | None:
    """Parent resolution for hierarchical exact coverage, or None to skip.

    Prefer a parent fine enough that typical ADM2 polygons can fully contain
    parent cells (r3 is usually too coarse for ADM2).
    """
    if resolution < 6:
        return None
    return max(0, resolution - 2)


def _drop_relation(con: DuckDBPyConnection, relation: str) -> None:
    """Drop a table or view by name (DuckDB rejects cross-type IF EXISTS)."""
    try:
        con.execute(f"DROP VIEW {relation}")
    except Exception:
        pass
    try:
        con.execute(f"DROP TABLE IF EXISTS {relation}")
    except Exception:
        pass


def _spill_relation_to_parquet_view(
    con: DuckDBPyConnection,
    relation: str,
    parquet_path: Path,
) -> None:
    """Persist ``relation`` to Parquet and replace it with a streaming view.

    Frees DuckDB buffer-pool pressure from large candidate tables before
    GEOS-heavy stages. ``relation`` must be a table or view name.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (SELECT * FROM {relation}) TO {sql_quote(parquet_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
        """
    )
    _drop_relation(con, relation)
    con.execute(
        f"""
        CREATE VIEW {relation} AS
        SELECT * FROM read_parquet({sql_quote(parquet_path)})
        """
    )


def write_exploded_coverage(
    con: DuckDBPyConnection,
    candidates_relation: str,
    polygons_relation: str,
    output_parquet: str | Path,
    *,
    resolution: int,
    parent_resolution: int | None = None,
    hex_geoms_parquet: str | Path | None = None,
    work_dir: str | Path | None = None,
    spill_candidates: bool = True,
    edge_batch_rows: int = 250_000,
    hex_col: str = "hex_id",
    poly_id_col: str = "poly_rid",
    polygon_id_col: str = "adm2_rid",
    adm0_col: str = "adm0_iso",
    adm1_id_col: str = "adm1_id",
    adm1_name_col: str = "adm1_name",
    adm2_id_col: str = "adm2_id",
    adm2_name_col: str = "adm2_name",
) -> dict[str, int | str]:
    """Write the canonical exploded ``(hex_id, adm*, pct)`` Parquet artifact.

    This is the primary crc-sdk coverage product. Nested lookup contracts are
    a pure derivation via :func:`crc_sdk.geometry.write_lookup_contract` and
    should run in a separate session/process after pipeline tables are gone.

    For ``resolution >= 6``, uses hierarchical parent containment so interior
    singles skip child geometries. Lower resolutions use batched edge
    detection plus envelope-clipped intersection.
    """
    out = Path(output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(work_dir) if work_dir is not None else out.parent
    stage_dir = stage_root / f".{out.stem}_exploded_tmp"
    stage_dir.mkdir(parents=True, exist_ok=True)
    spilled = False
    parent_res = (
        parent_resolution
        if parent_resolution is not None
        else recommend_parent_resolution(resolution)
    )
    try:
        if spill_candidates:
            logger.info("exploded: spill candidates to parquet view")
            _spill_relation_to_parquet_view(
                con,
                candidates_relation,
                stage_dir / "candidates.parquet",
            )
            spilled = True
        if parent_res is not None:
            return write_hierarchical_coverage(
                con,
                candidates_relation,
                polygons_relation,
                out,
                resolution=resolution,
                parent_resolution=parent_res,
                hex_geoms_parquet=hex_geoms_parquet,
                work_dir=stage_dir,
                edge_batch_rows=edge_batch_rows,
                hex_col=hex_col,
                poly_id_col=poly_id_col,
                polygon_id_col=polygon_id_col,
                adm0_col=adm0_col,
                adm1_id_col=adm1_id_col,
                adm1_name_col=adm1_name_col,
                adm2_id_col=adm2_id_col,
                adm2_name_col=adm2_name_col,
            )
        return _write_flat_exploded_coverage(
            con,
            candidates_relation,
            polygons_relation,
            out,
            hex_geoms_parquet=hex_geoms_parquet,
            work_dir=stage_dir,
            edge_batch_rows=edge_batch_rows,
            hex_col=hex_col,
            poly_id_col=poly_id_col,
            polygon_id_col=polygon_id_col,
            adm0_col=adm0_col,
            adm1_id_col=adm1_id_col,
            adm1_name_col=adm1_name_col,
            adm2_id_col=adm2_id_col,
            adm2_name_col=adm2_name_col,
        )
    finally:
        if spilled:
            _drop_relation(con, candidates_relation)
        shutil.rmtree(stage_dir, ignore_errors=True)


def _write_flat_exploded_coverage(
    con: DuckDBPyConnection,
    candidates_relation: str,
    polygons_relation: str,
    output_parquet: Path,
    *,
    hex_geoms_parquet: str | Path | None,
    work_dir: Path,
    edge_batch_rows: int,
    hex_col: str,
    poly_id_col: str,
    polygon_id_col: str,
    adm0_col: str,
    adm1_id_col: str,
    adm1_name_col: str,
    adm2_id_col: str,
    adm2_name_col: str,
) -> dict[str, int | str]:
    """Batched edge detection + coverage for resolutions without hierarchy."""
    logger.info("exploded: flat hex_counts")
    con.execute("DROP TABLE IF EXISTS hex_counts")
    con.execute(
        f"""
        CREATE TEMPORARY TABLE hex_counts AS
        SELECT {hex_col}, COUNT(*) AS cnt
        FROM {candidates_relation}
        GROUP BY {hex_col}
        """
    )
    stats = con.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE cnt = 1),
            COUNT(*) FILTER (WHERE cnt > 1),
            COALESCE(SUM(cnt) FILTER (WHERE cnt > 1), 0)
        FROM hex_counts
        """
    ).fetchone()
    single_count = int(stats[0])
    competing_count = int(stats[1])
    competing_ops = int(stats[2])

    logger.info("exploded: flat edge batch classify")
    edge_count = materialize_edge_hexes(
        con,
        candidates_relation,
        polygons_relation,
        hex_counts_relation="hex_counts",
        output_table="partial_hexes",
        hex_geoms_parquet=hex_geoms_parquet,
        batch_rows=edge_batch_rows,
        hex_col=hex_col,
        poly_id_col=poly_id_col,
        polygon_id_col=polygon_id_col,
    )
    con.execute("DROP TABLE IF EXISTS needed_hexes")
    con.execute(
        f"""
        CREATE TEMPORARY TABLE needed_hexes AS
        SELECT {hex_col} FROM hex_counts WHERE cnt > 1
        UNION
        SELECT {hex_col} FROM partial_hexes
        """
    )
    needed_count = int(con.execute("SELECT COUNT(*) FROM needed_hexes").fetchone()[0])
    con.execute("DROP TABLE IF EXISTS hex_geoms")
    con.execute(f"CREATE TABLE hex_geoms ({hex_col} VARCHAR, geom GEOMETRY)")
    if needed_count:
        if hex_geoms_parquet is not None and Path(hex_geoms_parquet).exists():
            con.execute(
                f"""
                INSERT INTO hex_geoms
                SELECT n.{hex_col},
                       COALESCE(
                           ST_GeomFromText(c.hex_wkt),
                           ST_GeomFromText(
                               h3_cell_to_boundary_wkt(h3_string_to_h3(n.{hex_col}))
                           )
                       ) AS geom
                FROM needed_hexes AS n
                LEFT JOIN read_parquet({sql_quote(hex_geoms_parquet)}) AS c
                  USING ({hex_col})
                """
            )
        else:
            con.execute(
                f"""
                INSERT INTO hex_geoms
                SELECT
                    {hex_col},
                    ST_GeomFromText(
                        h3_cell_to_boundary_wkt(h3_string_to_h3({hex_col}))
                    ) AS geom
                FROM needed_hexes
                """
            )
    coverage_sql = build_coverage_sql(
        candidates_relation,
        polygons_relation,
        "hex_geoms",
        optimize_single_cell=True,
        border_hexes_relation="partial_hexes",
        hex_counts_relation="hex_counts",
        hex_col=hex_col,
        poly_id_col=poly_id_col,
        polygon_id_col=polygon_id_col,
        adm0_col=adm0_col,
        adm1_id_col=adm1_id_col,
        adm1_name_col=adm1_name_col,
        adm2_id_col=adm2_id_col,
        adm2_name_col=adm2_name_col,
    )
    logger.info("exploded: flat stream coverage")
    con.execute(
        f"""
        COPY ({coverage_sql}) TO {sql_quote(output_parquet)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
        """
    )
    row_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_quote(output_parquet)})"
        ).fetchone()[0]
    )
    partial_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_quote(output_parquet)}) "
            f"WHERE pct < 1.0"
        ).fetchone()[0]
    )
    for table in ("needed_hexes", "partial_hexes", "hex_geoms"):
        con.execute(f"DROP TABLE IF EXISTS {table}")
    return {
        "mode": "flat",
        "single": single_count,
        "competing": competing_count,
        "competing_ops": competing_ops,
        "edge_singles": edge_count,
        "needed_hexes": needed_count,
        "coverage_rows": row_count,
        "partial_pct_rows": partial_count,
    }


def write_hierarchical_coverage(
    con: DuckDBPyConnection,
    candidates_relation: str,
    polygons_relation: str,
    output_parquet: str | Path,
    *,
    resolution: int,
    parent_resolution: int,
    hex_geoms_parquet: str | Path | None = None,
    work_dir: str | Path | None = None,
    edge_batch_rows: int = 250_000,
    hex_col: str = "hex_id",
    poly_id_col: str = "poly_rid",
    polygon_id_col: str = "adm2_rid",
    adm0_col: str = "adm0_iso",
    adm1_id_col: str = "adm1_id",
    adm1_name_col: str = "adm1_name",
    adm2_id_col: str = "adm2_id",
    adm2_name_col: str = "adm2_name",
) -> dict[str, int | str]:
    """Stream exact exploded coverage via parent-level containment classification.

    Single-candidate children under a parent fully contained by their ADM
    polygon get ``pct=1.0`` without child geometries. Boundary-parent children
    are edge-classified in bounded batches; only true edge singles and
    competing hexes pay for envelope-clipped intersection.
    """
    if not 0 <= parent_resolution < resolution <= 15:
        raise ValueError("require 0 <= parent_resolution < resolution <= 15")

    attrs = f"""
        a.{adm0_col} AS adm0_iso,
        a.{adm1_id_col} AS adm1_id,
        a.{adm1_name_col} AS adm1_name,
        a.{adm2_id_col} AS adm2_id,
        a.{adm2_name_col} AS adm2_name
    """
    parent_of = (
        "h3_h3_to_string(h3_cell_to_parent("
        f"h3_string_to_h3({{col}}), {int(parent_resolution)}))"
    )
    intersection_pct = _clipped_intersection_pct()
    out = Path(output_parquet)
    stage_root = Path(work_dir) if work_dir is not None else out.parent
    stage_dir = stage_root / f".{out.stem}_hier_tmp"
    stage_dir.mkdir(parents=True, exist_ok=True)
    interior_path = stage_dir / "interior.parquet"
    contained_path = stage_dir / "boundary_contained.parquet"
    exact_path = stage_dir / "exact.parquet"

    try:
        logger.info("hierarchical: hex_counts")
        con.execute("DROP TABLE IF EXISTS hex_counts")
        con.execute(
            f"""
            CREATE TEMPORARY TABLE hex_counts AS
            SELECT {hex_col}, COUNT(*) AS cnt
            FROM {candidates_relation}
            GROUP BY {hex_col}
            """
        )
        stats = con.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE cnt = 1),
                COUNT(*) FILTER (WHERE cnt > 1),
                COALESCE(SUM(cnt) FILTER (WHERE cnt > 1), 0)
            FROM hex_counts
            """
        ).fetchone()
        single_count = int(stats[0])
        competing_count = int(stats[1])
        competing_ops = int(stats[2])
        logger.info(
            "hierarchical: hex_counts single=%s competing=%s ops=%s",
            single_count,
            competing_count,
            competing_ops,
        )

        logger.info("hierarchical: materialize single cells with parents")
        singles_path = stage_dir / "singles.parquet"
        con.execute(
            f"""
            COPY (
                SELECT
                    c.{hex_col},
                    c.{poly_id_col},
                    {parent_of.format(col=f"c.{hex_col}")} AS parent_id
                FROM {candidates_relation} AS c
                JOIN hex_counts AS hc USING ({hex_col})
                WHERE hc.cnt = 1
            ) TO {sql_quote(singles_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
            """
        )
        con.execute("DROP TABLE IF EXISTS _crc_single_cells")
        con.execute(
            f"""
            CREATE VIEW _crc_single_cells AS
            SELECT * FROM read_parquet({sql_quote(singles_path)})
            """
        )

        logger.info("hierarchical: parent touch + geoms + classify")
        con.execute("DROP TABLE IF EXISTS _crc_parent_touch")
        con.execute(
            f"""
            CREATE TEMPORARY TABLE _crc_parent_touch AS
            SELECT DISTINCT parent_id, {poly_id_col}
            FROM _crc_single_cells
            """
        )
        parent_touch_count = int(
            con.execute("SELECT COUNT(*) FROM _crc_parent_touch").fetchone()[0]
        )

        con.execute("DROP TABLE IF EXISTS _crc_parent_geoms")
        con.execute(
            """
            CREATE TEMPORARY TABLE _crc_parent_geoms AS
            SELECT
                parent_id,
                ST_GeomFromText(
                    h3_cell_to_boundary_wkt(h3_string_to_h3(parent_id))
                ) AS geom
            FROM (SELECT DISTINCT parent_id FROM _crc_parent_touch)
            """
        )
        parent_geom_count = int(
            con.execute("SELECT COUNT(*) FROM _crc_parent_geoms").fetchone()[0]
        )

        con.execute("DROP TABLE IF EXISTS _crc_parent_class")
        con.execute(
            f"""
            CREATE TEMPORARY TABLE _crc_parent_class AS
            SELECT
                t.parent_id,
                t.{poly_id_col},
                ST_Contains(a.geom, p.geom) AS interior
            FROM _crc_parent_touch AS t
            JOIN _crc_parent_geoms AS p USING (parent_id)
            JOIN {polygons_relation} AS a
              ON t.{poly_id_col} = a.{polygon_id_col}
            """
        )
        con.execute("DROP TABLE IF EXISTS _crc_parent_geoms")
        con.execute("DROP TABLE IF EXISTS _crc_parent_touch")
        class_stats = con.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE interior),
                COUNT(*) FILTER (WHERE NOT interior)
            FROM _crc_parent_class
            """
        ).fetchone()
        interior_parents = int(class_stats[0])
        boundary_parents = int(class_stats[1])
        logger.info(
            "hierarchical: parent_touch=%s parent_geoms=%s interior=%s boundary=%s",
            parent_touch_count,
            parent_geom_count,
            interior_parents,
            boundary_parents,
        )

        logger.info("hierarchical: write interior singles")
        con.execute(
            f"""
            COPY (
                SELECT
                    s.{hex_col},
                    {attrs},
                    1.0 AS pct
                FROM _crc_single_cells AS s
                JOIN _crc_parent_class AS pc
                  ON s.parent_id = pc.parent_id
                 AND s.{poly_id_col} = pc.{poly_id_col}
                JOIN {polygons_relation} AS a
                  ON s.{poly_id_col} = a.{polygon_id_col}
                WHERE pc.interior
            ) TO {sql_quote(interior_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
            """
        )

        logger.info("hierarchical: boundary children edge batch classify")
        con.execute("DROP TABLE IF EXISTS _crc_boundary_children")
        con.execute(
            f"""
            CREATE TEMPORARY TABLE _crc_boundary_children AS
            SELECT DISTINCT s.{hex_col}
            FROM _crc_single_cells AS s
            JOIN _crc_parent_class AS pc
              ON s.parent_id = pc.parent_id
             AND s.{poly_id_col} = pc.{poly_id_col}
            WHERE NOT pc.interior
            """
        )
        boundary_child_count = int(
            con.execute("SELECT COUNT(*) FROM _crc_boundary_children").fetchone()[0]
        )
        edge_single_count = materialize_edge_hexes(
            con,
            candidates_relation,
            polygons_relation,
            hex_counts_relation="hex_counts",
            output_table="_crc_edge_singles",
            hex_geoms_parquet=hex_geoms_parquet,
            batch_rows=edge_batch_rows,
            single_hexes_relation="_crc_boundary_children",
            hex_col=hex_col,
            poly_id_col=poly_id_col,
            polygon_id_col=polygon_id_col,
        )
        logger.info(
            "hierarchical: boundary_children=%s edge_singles=%s",
            boundary_child_count,
            edge_single_count,
        )

        logger.info("hierarchical: write boundary-contained singles")
        con.execute(
            f"""
            COPY (
                SELECT
                    s.{hex_col},
                    {attrs},
                    1.0 AS pct
                FROM _crc_single_cells AS s
                JOIN _crc_parent_class AS pc
                  ON s.parent_id = pc.parent_id
                 AND s.{poly_id_col} = pc.{poly_id_col}
                JOIN {polygons_relation} AS a
                  ON s.{poly_id_col} = a.{polygon_id_col}
                LEFT JOIN _crc_edge_singles AS e USING ({hex_col})
                WHERE NOT pc.interior AND e.{hex_col} IS NULL
            ) TO {sql_quote(contained_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
            """
        )
        con.execute("DROP TABLE IF EXISTS _crc_boundary_children")

        con.execute("DROP TABLE IF EXISTS needed_hexes")
        con.execute(
            f"""
            CREATE TEMPORARY TABLE needed_hexes AS
            SELECT {hex_col} FROM _crc_edge_singles
            UNION
            SELECT {hex_col} FROM hex_counts WHERE cnt > 1
            """
        )
        needed_count = int(
            con.execute("SELECT COUNT(*) FROM needed_hexes").fetchone()[0]
        )
        logger.info("hierarchical: needed_hexes=%s", needed_count)

        logger.info("hierarchical: write exact edge+competing")
        if needed_count == 0:
            con.execute(
                f"""
                COPY (
                    SELECT
                        CAST(NULL AS VARCHAR) AS {hex_col},
                        CAST(NULL AS VARCHAR) AS adm0_iso,
                        CAST(NULL AS VARCHAR) AS adm1_id,
                        CAST(NULL AS VARCHAR) AS adm1_name,
                        CAST(NULL AS VARCHAR) AS adm2_id,
                        CAST(NULL AS VARCHAR) AS adm2_name,
                        CAST(NULL AS DOUBLE) AS pct
                    WHERE FALSE
                ) TO {sql_quote(exact_path)}
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
                """
            )
        else:
            con.execute("DROP TABLE IF EXISTS hex_geoms")
            con.execute(f"CREATE TABLE hex_geoms ({hex_col} VARCHAR, geom GEOMETRY)")
            if hex_geoms_parquet is not None and Path(hex_geoms_parquet).exists():
                con.execute(
                    f"""
                    INSERT INTO hex_geoms
                    SELECT n.{hex_col},
                           COALESCE(
                               ST_GeomFromText(c.hex_wkt),
                               ST_GeomFromText(
                                   h3_cell_to_boundary_wkt(
                                       h3_string_to_h3(n.{hex_col})
                                   )
                               )
                           ) AS geom
                    FROM needed_hexes AS n
                    LEFT JOIN read_parquet({sql_quote(hex_geoms_parquet)}) AS c
                      USING ({hex_col})
                    """
                )
            else:
                con.execute(
                    f"""
                    INSERT INTO hex_geoms
                    SELECT
                        {hex_col},
                        ST_GeomFromText(
                            h3_cell_to_boundary_wkt(h3_string_to_h3({hex_col}))
                        ) AS geom
                    FROM needed_hexes
                    """
                )
            con.execute(
                f"""
                COPY (
                    SELECT
                        s.{hex_col},
                        {attrs},
                        {intersection_pct}
                    FROM _crc_single_cells AS s
                    JOIN _crc_edge_singles AS e USING ({hex_col})
                    JOIN {polygons_relation} AS a
                      ON s.{poly_id_col} = a.{polygon_id_col}
                    JOIN hex_geoms AS h USING ({hex_col})
                    UNION ALL
                    SELECT
                        c.{hex_col},
                        {attrs},
                        {intersection_pct}
                    FROM {candidates_relation} AS c
                    JOIN hex_counts AS hc USING ({hex_col})
                    JOIN {polygons_relation} AS a
                      ON c.{poly_id_col} = a.{polygon_id_col}
                    JOIN hex_geoms AS h USING ({hex_col})
                    WHERE hc.cnt > 1
                ) TO {sql_quote(exact_path)}
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
                """
            )

        logger.info("hierarchical: merge coverage parquet")
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet({sql_quote(interior_path)})
                UNION ALL
                SELECT * FROM read_parquet({sql_quote(contained_path)})
                UNION ALL
                SELECT * FROM read_parquet({sql_quote(exact_path)})
            ) TO {sql_quote(out)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
            """
        )
        row_count = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet({sql_quote(out)})"
            ).fetchone()[0]
        )
        partial_count = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet({sql_quote(out)}) "
                f"WHERE pct < 1.0"
            ).fetchone()[0]
        )
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        for table in (
            "_crc_parent_class",
            "_crc_single_cells",
            "_crc_parent_touch",
            "_crc_parent_geoms",
            "_crc_edge_singles",
            "_crc_boundary_children",
            "needed_hexes",
            "hex_geoms",
        ):
            _drop_relation(con, table)
    return {
        "mode": "hierarchical",
        "single": single_count,
        "competing": competing_count,
        "competing_ops": competing_ops,
        "parent_touch": parent_touch_count,
        "parent_geoms": parent_geom_count,
        "interior_parents": interior_parents,
        "boundary_parents": boundary_parents,
        "boundary_children": boundary_child_count,
        "edge_singles": edge_single_count,
        "needed_hexes": needed_count,
        "coverage_rows": row_count,
        "partial_pct_rows": partial_count,
    }


def build_edge_hexes_sql(
    candidates_relation: str,
    polygons_relation: str,
    hex_geoms_relation: str,
    *,
    hex_col: str = "hex_id",
    poly_id_col: str = "poly_rid",
    polygon_id_col: str = "adm2_rid",
    hex_counts_relation: str | None = None,
    single_hexes_relation: str | None = None,
) -> str:
    """IDs of single-candidate hexes not fully contained in their polygon.

    When ``single_hexes_relation`` is set it should already be the ``cnt = 1``
    hex id list (optionally a batch), and ``hex_counts_relation`` is ignored.
    """
    if single_hexes_relation is not None:
        single_cte = f"""
            single AS (
                SELECT c.{hex_col}, c.{poly_id_col}
                FROM {candidates_relation} AS c
                JOIN {single_hexes_relation} AS s USING ({hex_col})
            )
        """
        prefix = ""
    else:
        counts_cte, counts = _hex_counts_prefix(
            candidates_relation,
            hex_col=hex_col,
            hex_counts_relation=hex_counts_relation,
        )
        prefix = counts_cte
        single_cte = f"""
            single AS (
                SELECT c.{hex_col}, c.{poly_id_col}
                FROM {candidates_relation} AS c
                JOIN {counts} AS hc USING ({hex_col})
                WHERE hc.cnt = 1
            )
        """
    return f"""
        WITH {prefix}
        {single_cte}
        SELECT DISTINCT s.{hex_col}
        FROM single AS s
        JOIN {hex_geoms_relation} AS h USING ({hex_col})
        JOIN {polygons_relation} AS a
          ON s.{poly_id_col} = a.{polygon_id_col}
        WHERE NOT ST_Contains(a.geom, h.geom)
    """


def materialize_edge_hexes(
    con: DuckDBPyConnection,
    candidates_relation: str,
    polygons_relation: str,
    *,
    hex_counts_relation: str = "hex_counts",
    output_table: str = "partial_hexes",
    hex_geoms_relation: str | None = None,
    hex_geoms_parquet: str | Path | None = None,
    batch_rows: int = 250_000,
    single_hexes_relation: str | None = None,
    hex_col: str = "hex_id",
    poly_id_col: str = "poly_rid",
    polygon_id_col: str = "adm2_rid",
) -> int:
    """Build edge-hex ID table without one giant ST_Contains over all singles.

    Batches single-candidate hexes and resolves geometries from an in-memory
    relation, a parquet cache, or on-the-fly H3 boundaries. Peak memory stays
    proportional to ``batch_rows`` instead of the full single-hex set (critical
    at r7+ where singles can exceed 20M).

    When ``single_hexes_relation`` is set it is used as the hex id universe
    (e.g. boundary-parent children only); otherwise ``hex_counts`` ``cnt = 1``
    rows are used.
    """
    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(f"CREATE TEMPORARY TABLE {output_table} ({hex_col} VARCHAR)")
    con.execute("DROP TABLE IF EXISTS _crc_single_hex_ids")
    if single_hexes_relation is not None:
        con.execute(
            f"""
            CREATE TEMPORARY TABLE _crc_single_hex_ids AS
            SELECT DISTINCT {hex_col}
            FROM {single_hexes_relation}
            ORDER BY {hex_col}
            """
        )
    else:
        con.execute(
            f"""
            CREATE TEMPORARY TABLE _crc_single_hex_ids AS
            SELECT {hex_col}
            FROM {hex_counts_relation}
            WHERE cnt = 1
            ORDER BY {hex_col}
            """
        )
    total_singles = int(
        con.execute("SELECT COUNT(*) FROM _crc_single_hex_ids").fetchone()[0]
    )
    if total_singles == 0:
        con.execute("DROP TABLE IF EXISTS _crc_single_hex_ids")
        return 0

    batch_rows = max(1, int(batch_rows))
    parquet_path = (
        Path(hex_geoms_parquet)
        if hex_geoms_parquet is not None and Path(hex_geoms_parquet).exists()
        else None
    )
    last_hex: str | None = None
    while True:
        con.execute("DROP TABLE IF EXISTS _crc_edge_batch")
        if last_hex is None:
            con.execute(
                f"""
                CREATE TEMPORARY TABLE _crc_edge_batch AS
                SELECT {hex_col}
                FROM _crc_single_hex_ids
                ORDER BY {hex_col}
                LIMIT {batch_rows}
                """
            )
        else:
            con.execute(
                f"""
                CREATE TEMPORARY TABLE _crc_edge_batch AS
                SELECT {hex_col}
                FROM _crc_single_hex_ids
                WHERE {hex_col} > {sql_quote(last_hex)}
                ORDER BY {hex_col}
                LIMIT {batch_rows}
                """
            )
        batch_count = int(
            con.execute("SELECT COUNT(*) FROM _crc_edge_batch").fetchone()[0]
        )
        if batch_count == 0:
            break

        con.execute("DROP TABLE IF EXISTS _crc_edge_batch_geoms")
        if hex_geoms_relation is not None:
            con.execute(
                f"""
                CREATE TEMPORARY TABLE _crc_edge_batch_geoms AS
                SELECT h.{hex_col}, h.geom
                FROM {hex_geoms_relation} AS h
                JOIN _crc_edge_batch AS b USING ({hex_col})
                """
            )
        elif parquet_path is not None:
            con.execute(
                f"""
                CREATE TEMPORARY TABLE _crc_edge_batch_geoms AS
                SELECT
                    b.{hex_col},
                    COALESCE(
                        ST_GeomFromText(c.hex_wkt),
                        ST_GeomFromText(
                            h3_cell_to_boundary_wkt(h3_string_to_h3(b.{hex_col}))
                        )
                    ) AS geom
                FROM _crc_edge_batch AS b
                LEFT JOIN read_parquet({sql_quote(parquet_path)}) AS c
                  USING ({hex_col})
                """
            )
        else:
            con.execute(
                f"""
                CREATE TEMPORARY TABLE _crc_edge_batch_geoms AS
                SELECT
                    b.{hex_col},
                    ST_GeomFromText(
                        h3_cell_to_boundary_wkt(h3_string_to_h3(b.{hex_col}))
                    ) AS geom
                FROM _crc_edge_batch AS b
                """
            )
        edge_sql = build_edge_hexes_sql(
            candidates_relation,
            polygons_relation,
            "_crc_edge_batch_geoms",
            hex_col=hex_col,
            poly_id_col=poly_id_col,
            polygon_id_col=polygon_id_col,
            single_hexes_relation="_crc_edge_batch",
        )
        con.execute(f"INSERT INTO {output_table} {edge_sql}")
        last_hex = con.execute(
            f"SELECT MAX({hex_col}) FROM _crc_edge_batch"
        ).fetchone()[0]
        if batch_count < batch_rows:
            break

    con.execute("DROP TABLE IF EXISTS _crc_edge_batch_geoms")
    con.execute("DROP TABLE IF EXISTS _crc_edge_batch")
    con.execute("DROP TABLE IF EXISTS _crc_single_hex_ids")
    return int(con.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0])


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
