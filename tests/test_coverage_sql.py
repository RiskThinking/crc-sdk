from crc_sdk.geometry import build_coverage_sql, build_hex_counts_sql


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
