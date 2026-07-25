import duckdb

from crc_sdk.connectors.duckdb import ensure_extensions
from crc_sdk.geometry import (
    build_border_hexes_sql,
    build_coverage_sql,
    build_hex_counts_sql,
)


def test_build_coverage_sql_shares_hex_counts_cte_by_default() -> None:
    sql = build_coverage_sql("candidates", "admins", "hex_geoms")
    assert "WITH hex_counts AS" in sql
    assert "singles AS" in sql
    assert sql.count("GROUP BY hex_id") == 1
    assert build_hex_counts_sql("candidates").count("GROUP BY hex_id") == 1


def test_build_coverage_sql_uses_provided_hex_counts_relation() -> None:
    sql = build_coverage_sql(
        "candidates",
        "admins",
        "hex_geoms",
        hex_counts_relation="precomputed_counts",
    )
    assert "hex_counts AS" not in sql or "JOIN precomputed_counts" in sql
    assert "JOIN precomputed_counts AS hc" in sql
    assert "singles AS" in sql


def test_build_coverage_sql_keeps_contains_off_competing_path() -> None:
    sql = build_coverage_sql("candidates", "admins", "hex_geoms")
    assert "ST_Contains(a.geom, h.geom)" in sql
    assert "WHERE hc.cnt > 1" in sql
    # Competing branch must not wrap intersection in CASE over all rows.
    assert "CASE" not in sql
    assert sql.count("UNION ALL") == 2


def test_build_border_hexes_sql_includes_non_contained_edges() -> None:
    sql = build_border_hexes_sql("candidates", "admins", "hex_geoms", "LUX")
    assert "NOT ST_Contains(a.geom, h.geom)" in sql
    assert "foreign_adm0" in sql
    assert "ST_Extent_Agg" in sql
    assert "!= 'LUX'" in sql


def test_single_candidate_interior_hex_gets_full_coverage() -> None:
    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    con.execute(
        """
        CREATE TABLE admins AS
        SELECT
            1::BIGINT AS adm2_rid,
            'LUX' AS adm0_iso,
            'a1' AS adm1_id,
            'Canton' AS adm1_name,
            'a2' AS adm2_id,
            'Commune' AS adm2_name,
            ST_GeomFromText('POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))') AS geom
        """
    )
    con.execute(
        """
        CREATE TABLE hex_geoms AS
        SELECT
            'inside' AS hex_id,
            ST_GeomFromText('POLYGON((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))')
                AS geom
        """
    )
    con.execute(
        "CREATE TABLE candidates AS SELECT 'inside' AS hex_id, 1::BIGINT AS poly_rid"
    )
    sql = build_coverage_sql("candidates", "admins", "hex_geoms")
    pct = con.execute(f"SELECT pct FROM ({sql})").fetchone()[0]
    assert pct == 1.0


def test_single_candidate_edge_hex_gets_partial_coverage() -> None:
    """Coastal/edge hex with one candidate must not be forced to pct=1.0."""
    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    con.execute(
        """
        CREATE TABLE admins AS
        SELECT
            1::BIGINT AS adm2_rid,
            'LUX' AS adm0_iso,
            'a1' AS adm1_id,
            'Canton' AS adm1_name,
            'a2' AS adm2_id,
            'Commune' AS adm2_name,
            ST_GeomFromText('POLYGON((0 0, 2 0, 2 1, 0 1, 0 0))') AS geom
        """
    )
    # Hex spans land (y<=1) and water (y>1); only one candidate polygon.
    con.execute(
        """
        CREATE TABLE hex_geoms AS
        SELECT
            'edge' AS hex_id,
            ST_GeomFromText('POLYGON((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))')
                AS geom
        """
    )
    con.execute(
        "CREATE TABLE candidates AS SELECT 'edge' AS hex_id, 1::BIGINT AS poly_rid"
    )
    sql = build_coverage_sql("candidates", "admins", "hex_geoms")
    pct = con.execute(f"SELECT pct FROM ({sql})").fetchone()[0]
    assert 0.0 < pct < 1.0


def test_competing_and_interior_rows_coexist() -> None:
    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    con.execute(
        """
        CREATE TABLE admins AS
        SELECT 1::BIGINT AS adm2_rid, 'LUX' AS adm0_iso,
               'a1' AS adm1_id, 'Canton' AS adm1_name,
               'a2' AS adm2_id, 'Commune' AS adm2_name,
               ST_GeomFromText('POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))') AS geom
        UNION ALL
        SELECT 2, 'BEL', 'b1', 'Prov', 'b2', 'Town',
               ST_GeomFromText('POLYGON((1 0, 3 0, 3 2, 1 2, 1 0))')
        """
    )
    con.execute(
        """
        CREATE TABLE hex_geoms AS
        SELECT 'inside' AS hex_id,
               ST_GeomFromText('POLYGON((0.2 0.2, 0.8 0.2, 0.8 0.8, 0.2 0.8, 0.2 0.2))')
                   AS geom
        UNION ALL
        SELECT 'shared',
               ST_GeomFromText('POLYGON((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))')
        """
    )
    con.execute(
        """
        CREATE TABLE candidates AS
        SELECT 'inside' AS hex_id, 1::BIGINT AS poly_rid
        UNION ALL SELECT 'shared', 1
        UNION ALL SELECT 'shared', 2
        """
    )
    sql = build_coverage_sql("candidates", "admins", "hex_geoms")
    rows = con.execute(
        f"SELECT hex_id, COUNT(*), MAX(pct) FROM ({sql}) GROUP BY 1 ORDER BY 1"
    ).fetchall()
    by_hex = {hex_id: (count, max_pct) for hex_id, count, max_pct in rows}
    assert by_hex["inside"] == (1, 1.0)
    assert by_hex["shared"][0] == 2
    shared_pcts = con.execute(
        f"SELECT pct FROM ({sql}) WHERE hex_id = 'shared' ORDER BY pct"
    ).fetchall()
    assert all(row[0] > 0 for row in shared_pcts)
