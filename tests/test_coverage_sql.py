import duckdb
from pathlib import Path

from crc_sdk.connectors.duckdb import ensure_extensions, sql_quote
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


def test_materialize_edge_hexes_batches_without_full_hex_geoms() -> None:
    from crc_sdk.geometry import materialize_edge_hexes

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
        SELECT * FROM (VALUES
            (
                'edge',
                ST_GeomFromText('POLYGON((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))')
            ),
            (
                'inside',
                ST_GeomFromText('POLYGON((0.25 0.25, 0.75 0.25, 0.75 0.75, 0.25 0.75, 0.25 0.25))')
            )
        ) AS t(hex_id, geom)
        """
    )
    con.execute(
        """
        CREATE TABLE candidates AS
        SELECT * FROM (VALUES
            ('edge', 1::BIGINT),
            ('inside', 1::BIGINT)
        ) AS t(hex_id, poly_rid)
        """
    )
    con.execute(
        "CREATE TEMPORARY TABLE hex_counts AS "
        "SELECT hex_id, COUNT(*) AS cnt FROM candidates GROUP BY hex_id"
    )
    count = materialize_edge_hexes(
        con,
        "candidates",
        "admins",
        hex_counts_relation="hex_counts",
        hex_geoms_relation="hex_geoms",
        batch_rows=1,
    )
    assert count == 1
    assert con.execute("SELECT hex_id FROM partial_hexes").fetchone() == ("edge",)


def test_recommend_parent_resolution() -> None:
    from crc_sdk.geometry import recommend_parent_resolution

    assert recommend_parent_resolution(5) is None
    assert recommend_parent_resolution(6) == 4
    assert recommend_parent_resolution(7) == 5


def test_hierarchical_coverage_interior_parent_skips_child_geoms(
    tmp_path: Path,
) -> None:
    from crc_sdk.geometry import write_hierarchical_coverage

    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    cell = con.execute(
        "SELECT h3_h3_to_string(h3_latlng_to_cell(49.6, 6.1, 6))"
    ).fetchone()[0]
    parent = con.execute(
        f"SELECT h3_h3_to_string(h3_cell_to_parent(h3_string_to_h3('{cell}'), 4))"
    ).fetchone()[0]
    parent_wkt = con.execute(
        f"SELECT h3_cell_to_boundary_wkt(h3_string_to_h3('{parent}'))"
    ).fetchone()[0]
    con.execute(
        f"""
        CREATE TABLE admins AS
        SELECT
            1::BIGINT AS adm2_rid,
            'LUX' AS adm0_iso,
            'a1' AS adm1_id,
            'Canton' AS adm1_name,
            'a2' AS adm2_id,
            'Commune' AS adm2_name,
            ST_Buffer(ST_GeomFromText({sql_quote(parent_wkt)}), 1.0) AS geom
        """
    )
    con.execute(
        f"CREATE TABLE candidates AS SELECT '{cell}' AS hex_id, 1::BIGINT AS poly_rid"
    )
    out = tmp_path / "cov.parquet"
    stats = write_hierarchical_coverage(
        con,
        "candidates",
        "admins",
        out,
        resolution=6,
        parent_resolution=4,
    )
    assert stats["coverage_rows"] == 1
    assert stats["interior_parents"] >= 1
    assert stats["needed_hexes"] == 0
    pct = con.execute(f"SELECT pct FROM read_parquet('{out}')").fetchone()[0]
    assert pct == 1.0


def test_write_exploded_coverage_dispatches_hierarchical(
    tmp_path: Path,
) -> None:
    from crc_sdk.geometry import write_exploded_coverage

    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    cell = con.execute(
        "SELECT h3_h3_to_string(h3_latlng_to_cell(49.6, 6.1, 6))"
    ).fetchone()[0]
    parent = con.execute(
        f"SELECT h3_h3_to_string(h3_cell_to_parent(h3_string_to_h3('{cell}'), 4))"
    ).fetchone()[0]
    parent_wkt = con.execute(
        f"SELECT h3_cell_to_boundary_wkt(h3_string_to_h3('{parent}'))"
    ).fetchone()[0]
    con.execute(
        f"""
        CREATE TABLE admins AS
        SELECT
            1::BIGINT AS adm2_rid,
            'LUX' AS adm0_iso,
            'a1' AS adm1_id,
            'Canton' AS adm1_name,
            'a2' AS adm2_id,
            'Commune' AS adm2_name,
            ST_Buffer(ST_GeomFromText({sql_quote(parent_wkt)}), 1.0) AS geom
        """
    )
    con.execute(
        f"CREATE TABLE candidates AS SELECT '{cell}' AS hex_id, 1::BIGINT AS poly_rid"
    )
    out = tmp_path / "cov.parquet"
    stats = write_exploded_coverage(
        con,
        "candidates",
        "admins",
        out,
        resolution=6,
        work_dir=tmp_path / "work",
    )
    assert stats["mode"] == "hierarchical"
    assert stats["coverage_rows"] == 1
    assert Path(out).exists()


def test_hierarchical_coverage_boundary_parent_gets_partial_pct(
    tmp_path: Path,
) -> None:
    from crc_sdk.geometry import write_hierarchical_coverage

    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    cell = con.execute(
        "SELECT h3_h3_to_string(h3_latlng_to_cell(49.6, 6.1, 6))"
    ).fetchone()[0]
    # Tiny polygon that intersects the cell but does not contain its parent.
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
            ST_GeomFromText(
                'POLYGON((6.09 49.59, 6.11 49.59, 6.11 49.61, 6.09 49.61, 6.09 49.59))'
            ) AS geom
        """
    )
    con.execute(
        f"CREATE TABLE candidates AS SELECT '{cell}' AS hex_id, 1::BIGINT AS poly_rid"
    )
    out = tmp_path / "cov.parquet"
    stats = write_hierarchical_coverage(
        con,
        "candidates",
        "admins",
        out,
        resolution=6,
        parent_resolution=4,
    )
    assert stats["coverage_rows"] == 1
    assert stats["boundary_parents"] >= 1
    pct = con.execute(f"SELECT pct FROM read_parquet('{out}')").fetchone()[0]
    assert 0.0 <= pct <= 1.0
