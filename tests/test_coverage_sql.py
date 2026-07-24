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
    assert sql.count("GROUP BY hex_id") == 1
    assert sql.count("JOIN hex_counts AS hc") == 2
    # Inlining the subquery would duplicate this GROUP BY fragment.
    assert build_hex_counts_sql("candidates").count("GROUP BY hex_id") == 1


def test_build_coverage_sql_uses_provided_hex_counts_relation() -> None:
    sql = build_coverage_sql(
        "candidates",
        "admins",
        "hex_geoms",
        hex_counts_relation="precomputed_counts",
    )
    assert "WITH hex_counts AS" not in sql
    assert sql.count("JOIN precomputed_counts AS hc") == 2


def test_build_coverage_sql_requires_containment_for_full_coverage() -> None:
    sql = build_coverage_sql("candidates", "admins", "hex_geoms")
    assert "ST_Contains(ST_MakeValid(a.geom), h.geom)" in sql
    assert "1.0 AS pct" in sql
    assert "NOT (ST_Contains(ST_MakeValid(a.geom), h.geom))" in sql


def test_build_border_hexes_sql_includes_non_contained_edges() -> None:
    sql = build_border_hexes_sql("candidates", "admins", "hex_geoms", "LUX")
    assert "NOT ST_Contains(ST_MakeValid(a.geom), h.geom)" in sql
    assert "foreign_a.adm0_iso != 'LUX'" in sql


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
