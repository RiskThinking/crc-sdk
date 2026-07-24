from pathlib import Path

import duckdb

from crc_sdk.geometry import (
    LookupCatalog,
    write_lookup_contract,
    write_partitioned_lookup,
)


def _stub_h3(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE MACRO h3_string_to_h3(value) AS value")
    con.execute(
        "CREATE MACRO h3_cell_to_parent(value, resolution) "
        "AS value || '_' || resolution"
    )
    con.execute("CREATE MACRO h3_h3_to_string(value) AS value")


def test_contract_marks_only_complete_single_path_as_interior(
    tmp_path: Path,
) -> None:
    con = duckdb.connect()
    _stub_h3(con)
    con.execute(
        """
        CREATE TABLE coverage(
            hex_id VARCHAR, adm0_iso VARCHAR,
            adm1_id VARCHAR, adm1_name VARCHAR,
            adm2_id VARCHAR, adm2_name VARCHAR, pct DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO coverage VALUES
          ('h1', 'LUX', 'a1', 'Canton', 'a2', 'Commune', 1.0),
          ('h2', 'LUX', 'a1', 'Canton', 'a2', 'Commune', 0.6),
          ('h2', 'BEL', 'b1', 'Province', 'b2', 'Town', 0.4)
        """
    )
    exploded = tmp_path / "r8_exploded.parquet"
    lookup = tmp_path / "r8.parquet"
    con.execute(f"COPY coverage TO '{exploded}' (FORMAT PARQUET)")
    write_lookup_contract(
        con,
        exploded,
        lookup,
        resolution=8,
        include_coverage=False,
    )
    assert con.execute(
        f"""
        SELECT hex_id, best_adm0, candidate_count, is_adm2_interior
        FROM '{lookup}' ORDER BY hex_id
        """
    ).fetchall() == [
        ("h1", "LUX", 1, True),
        ("h2", "LUX", 2, False),
    ]


def test_catalog_exposes_stable_artifact_names() -> None:
    catalog = LookupCatalog("gs://bucket/h3-lookup/")
    assert catalog.lookup_uri(8) == "gs://bucket/h3-lookup/r8.parquet"
    assert catalog.exploded_uri(8) == "gs://bucket/h3-lookup/r8_exploded.parquet"
    assert catalog.partitioned_lookup_root(7) == "gs://bucket/h3-lookup/r7"
    assert catalog.partitioned_lookup_uri(7, 1, "81") == (
        "gs://bucket/h3-lookup/r7/h3_r1=81/*.parquet"
    )
    assert "x.adm0_iso" in catalog.country_cells_sql("lux")
    assert "best_adm0='LUX'" in catalog.country_cells_sql("lux")

    no_coverage = catalog.country_cells_sql("lux", resolution=7, has_coverage=False)
    assert "r7_exploded.parquet" in no_coverage
    assert "coverage" not in no_coverage

    r8_sql = catalog.country_cells_sql("lux", resolution=8)
    assert "r8_exploded.parquet" in r8_sql
    assert "adm0_iso='LUX'" in r8_sql
    assert "pct > 0" in r8_sql


def test_contract_accepts_adm0_only_coverage(tmp_path: Path) -> None:
    con = duckdb.connect()
    _stub_h3(con)
    exploded = tmp_path / "adm0_exploded.parquet"
    lookup = tmp_path / "adm0.parquet"
    con.execute(
        """
        CREATE TABLE coverage(
            hex_id VARCHAR, adm0_iso VARCHAR, pct DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO coverage VALUES
          ('h1', 'LUX', 1.0),
          ('h2', 'LUX', 0.6),
          ('h2', 'BEL', 0.4)
        """
    )
    con.execute(f"COPY coverage TO '{exploded}' (FORMAT PARQUET)")
    write_lookup_contract(
        con,
        exploded,
        lookup,
        resolution=5,
        include_coverage=True,
    )
    rows = con.execute(
        f"""
        SELECT hex_id, best_adm0, best_adm1_id, best_adm1, best_adm2_id, best_adm2,
               is_adm2_interior
        FROM '{lookup}' ORDER BY hex_id
        """
    ).fetchall()
    assert rows == [
        ("h1", "LUX", None, None, None, None, True),
        ("h2", "LUX", None, None, None, None, False),
    ]


def test_nested_lookup_can_be_partitioned_without_exploded_rows(tmp_path: Path) -> None:
    con = duckdb.connect()
    _stub_h3(con)
    source = tmp_path / "r7.parquet"
    output = tmp_path / "r7"
    con.execute(
        "CREATE TABLE lookup(hex_id VARCHAR, best_adm0 VARCHAR, "
        "coverage STRUCT(adm0_iso VARCHAR, pct DOUBLE)[])"
    )
    con.execute(
        "INSERT INTO lookup VALUES "
        "('a', 'LUX', [{'adm0_iso': 'LUX', 'pct': 1.0}]), "
        "('b', 'BEL', [{'adm0_iso': 'BEL', 'pct': 1.0}])"
    )
    con.execute(f"COPY lookup TO '{source}' (FORMAT PARQUET)")
    write_partitioned_lookup(con, source, output, resolution=7, partition_resolution=1)
    assert con.execute(
        f"SELECT count(*) FROM read_parquet('{output}/**/*.parquet')"
    ).fetchone() == (2,)
