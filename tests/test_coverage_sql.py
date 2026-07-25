import duckdb

from crc_sdk.connectors.duckdb import ensure_extensions
from crc_sdk.geometry import (
    build_border_hexes_sql,
    build_coverage_sql,
    build_edge_hexes_sql,
    build_hex_counts_sql,
)


def test_build_coverage_sql_shares_hex_counts_cte_by_default() -> None:
    sql = build_coverage_sql(
        "candidates",
        "admins",
        "hex_geoms",
        border_hexes_relation="partial_hexes",
    )
    assert "WITH hex_counts AS" in sql
    assert sql.count("GROUP BY hex_id") == 1
    assert build_hex_counts_sql("candidates").count("GROUP BY hex_id") == 1


def test_build_coverage_sql_omits_hex_counts_without_border() -> None:
    sql = build_coverage_sql("candidates", "admins", "hex_geoms")
    assert "hex_counts" not in sql
    assert "UNION ALL" not in sql
    assert "ST_Envelope(h.geom)" in sql


def test_build_coverage_sql_splits_competing_and_edge_branches() -> None:
    sql = build_coverage_sql(
        "candidates",
        "admins",
        "hex_geoms",
        border_hexes_relation="partial_hexes",
        hex_counts_relation="hex_counts",
    )
    assert sql.count("UNION ALL") == 2
    assert "WHERE hc.cnt > 1" in sql
    assert "LEFT JOIN partial_hexes AS p" in sql
    assert "JOIN partial_hexes AS p" in sql
    # Hex geoms only on the two exact branches.
    assert sql.count("JOIN hex_geoms AS h") == 2
    assert "cnt > 1 OR" not in sql


def test_build_edge_hexes_sql_is_id_only() -> None:
    sql = build_edge_hexes_sql("candidates", "admins", "hex_geoms")
    assert "NOT ST_Contains(a.geom, h.geom)" in sql
    assert "SELECT DISTINCT s.hex_id" in sql


def test_build_border_hexes_sql_includes_non_contained_edges() -> None:
    sql = build_border_hexes_sql("candidates", "admins", "hex_geoms", "LUX")
    assert "NOT ST_Contains(a.geom, h.geom)" in sql
    assert "foreign_adm0" in sql


def test_build_coverage_sql_clips_admin_to_hex_envelope() -> None:
    sql = build_coverage_sql(
        "candidates",
        "admins",
        "hex_geoms",
        border_hexes_relation="partial_hexes",
        hex_counts_relation="hex_counts",
    )
    assert "ST_Intersection(a.geom, ST_Envelope(h.geom))" in sql


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
    con.execute(
        "CREATE TEMPORARY TABLE hex_counts AS "
        "SELECT hex_id, COUNT(*) AS cnt FROM candidates GROUP BY hex_id"
    )
    con.execute(
        "CREATE TEMPORARY TABLE partial_hexes AS "
        f"{build_edge_hexes_sql('candidates', 'admins', 'hex_geoms', hex_counts_relation='hex_counts')}"
    )
    assert con.execute("SELECT COUNT(*) FROM partial_hexes").fetchone() == (0,)
    sql = build_coverage_sql(
        "candidates",
        "admins",
        "hex_geoms",
        border_hexes_relation="partial_hexes",
        hex_counts_relation="hex_counts",
    )
    pct = con.execute(f"SELECT pct FROM ({sql})").fetchone()[0]
    assert pct == 1.0


def test_single_candidate_edge_hex_gets_partial_coverage() -> None:
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
    con.execute(
        "CREATE TEMPORARY TABLE hex_counts AS "
        "SELECT hex_id, COUNT(*) AS cnt FROM candidates GROUP BY hex_id"
    )
    con.execute(
        "CREATE TEMPORARY TABLE partial_hexes AS "
        f"{build_edge_hexes_sql('candidates', 'admins', 'hex_geoms', hex_counts_relation='hex_counts')}"
    )
    assert con.execute("SELECT COUNT(*) FROM partial_hexes").fetchone() == (1,)
    sql = build_coverage_sql(
        "candidates",
        "admins",
        "hex_geoms",
        border_hexes_relation="partial_hexes",
        hex_counts_relation="hex_counts",
    )
    pct = con.execute(f"SELECT pct FROM ({sql})").fetchone()[0]
    assert 0.0 < pct < 1.0
