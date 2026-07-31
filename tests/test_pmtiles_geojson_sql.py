import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from crc_sdk.geometry.pmtiles._geojson_sql import (
    LayerSource,
    build_combined_query,
    build_layer_query,
)


def _connection() -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    return con


def _write_points(con: "duckdb.DuckDBPyConnection", path: Path, count: int = 5) -> None:
    con.execute(
        f"""
        COPY (
            SELECT i AS id, 'name_' || i AS name, ST_Point(i * 0.1, i * 0.1) AS geometry
            FROM range(1, {count + 1}) t(i)
        ) TO '{path}' (FORMAT PARQUET)
        """
    )


def _parse_feature(raw: str) -> dict[str, Any]:
    """Each `feature` row carries the RFC 8142 record separator/newline
    baked in (so the caller's write loop needs no per-row Python
    formatting) -- strip them before parsing as JSON.
    """
    assert raw[0] == "\x1e"
    assert raw[-1] == "\n"
    parsed: dict[str, Any] = json.loads(raw[1:-1])
    return parsed


def test_build_layer_query_produces_valid_feature_rows(tmp_path: Path) -> None:
    con = _connection()
    path = tmp_path / "points.parquet"
    _write_points(con, path, count=3)

    query = build_layer_query(
        con, LayerSource(source=str(path), layer="places", minzoom=0, maxzoom=10)
    )
    rows = con.execute(query).fetchall()
    assert len(rows) == 3
    for (feature_json,) in rows:
        feature = _parse_feature(feature_json)
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"
        assert feature["tippecanoe"] == {"layer": "places", "minzoom": 0, "maxzoom": 10}
        assert "id" in feature["properties"]
        assert "name" in feature["properties"]
        assert "geometry" not in feature["properties"]


def test_build_layer_query_honors_precision(tmp_path: Path) -> None:
    con = _connection()
    path = tmp_path / "points.parquet"
    con.execute(
        f"""
        COPY (SELECT 1 AS id, ST_Point(1.123456789, 2.987654321) AS geometry)
        TO '{path}' (FORMAT PARQUET)
        """
    )
    query = build_layer_query(
        con,
        LayerSource(source=str(path), layer="p", minzoom=0, maxzoom=1, precision=2),
    )
    row = con.execute(query).fetchone()
    assert row is not None
    (feature_json,) = row
    coordinates = _parse_feature(feature_json)["geometry"]["coordinates"]
    assert coordinates == [1.12, 2.99]


def test_build_layer_query_handles_zero_row_geoparquet(tmp_path: Path) -> None:
    """A 0-row GeoParquet file loses its native GEOMETRY type on read back
    (falls back to raw BLOB/WKB) -- confirmed empirically. The query must
    still build and execute (producing zero rows), not raise a type error.
    """
    con = _connection()
    path = tmp_path / "empty.parquet"
    con.execute(
        f"""
        COPY (SELECT 1 AS id, ST_Point(0, 0) AS geometry FROM range(0))
        TO '{path}' (FORMAT PARQUET)
        """
    )
    query = build_layer_query(
        con, LayerSource(source=str(path), layer="p", minzoom=0, maxzoom=1)
    )
    assert con.execute(query).fetchall() == []


def test_build_layer_query_reprojects_the_cast_expression_not_the_raw_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: a non-native (BLOB/WKB) geometry column that also
    needs reprojection must have ST_Transform applied to the ST_GeomFromWKB
    cast, not the raw column -- feeding a raw BLOB straight to ST_Transform
    fails to bind (confirmed empirically) rather than silently mistiling.

    DuckDB auto-promotes any column with complete GeoParquet 'geo' metadata
    to native GEOMETRY on read regardless of row count, which makes "BLOB
    column + known non-WGS84 CRS" hard to reach through a real file today
    (the only naturally-occurring BLOB case, a 0-row file, carries no 'geo'
    metadata at all, hence no CRS). Writing the geometry column as a plain
    BLOB (via ST_AsWKB, no 'geo' metadata attached) reproduces a genuinely
    non-native column; ``geo_metadata`` is mocked only to supply the CRS
    that column's own file doesn't carry.
    """
    import crc_sdk.geometry.pmtiles._geojson_sql as geojson_sql

    con = _connection()
    path = tmp_path / "utm.parquet"
    # A point at (500000, 4649776) in EPSG:32633 (UTM 33N) is approximately
    # (15.0 E, 42.0 N) in EPSG:4326.
    con.execute(
        f"""
        COPY (SELECT 1 AS id, ST_AsWKB(ST_Point(500000, 4649776)) AS geometry)
        TO '{path}' (FORMAT PARQUET)
        """
    )
    assert (
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()[1][1]
        == "BLOB"
    )
    monkeypatch.setattr(
        geojson_sql,
        "geo_metadata",
        lambda con, source: {
            "primary_column": "geometry",
            "columns": {
                "geometry": {"crs": {"id": {"authority": "EPSG", "code": "32633"}}}
            },
        },
    )

    query = build_layer_query(
        con, LayerSource(source=str(path), layer="p", minzoom=0, maxzoom=1, precision=2)
    )
    assert 'ST_Transform(ST_GeomFromWKB("geometry")' in query
    row = con.execute(query).fetchone()
    assert row is not None
    coordinates = _parse_feature(row[0])["geometry"]["coordinates"]
    assert coordinates == pytest.approx([15.0, 42.0], abs=0.05)


def test_build_layer_query_reads_a_glob_with_schema_drift_via_union_by_name(
    tmp_path: Path,
) -> None:
    """Regression test: a Hive dataset accumulated per-partition (e.g. one
    file per country) can have genuine column drift -- one partition simply
    lacking a scenario/pathway column another has. Without `union_by_name`,
    DuckDB silently keeps only the first file's columns and drops the rest
    from every other file (confirmed empirically, not just documented) --
    not an error, just quietly missing data in the tiled output.
    """
    con = _connection()
    con.execute(
        f"""COPY (SELECT 1 AS id, 10 AS perc_50, ST_Point(0, 0) AS geometry)
        TO '{tmp_path / "a.parquet"}' (FORMAT PARQUET)"""
    )
    con.execute(
        f"""COPY (SELECT 2 AS id, 20 AS perc_50, 30 AS perc_90,
        ST_Point(1, 1) AS geometry) TO '{tmp_path / "b.parquet"}' (FORMAT PARQUET)"""
    )
    glob = str(tmp_path / "*.parquet")
    query = build_layer_query(
        con, LayerSource(source=glob, layer="p", minzoom=0, maxzoom=1)
    )
    features = [_parse_feature(row[0]) for row in con.execute(query).fetchall()]
    by_id = {f["properties"]["id"]: f["properties"] for f in features}
    assert by_id[1].get("perc_90") is None
    assert by_id[2]["perc_90"] == 30


def test_build_layer_query_raises_when_no_geometry_column_found(tmp_path: Path) -> None:
    con = _connection()
    path = tmp_path / "no_geom.parquet"
    con.execute(f"COPY (SELECT 1 AS id, 'x' AS label) TO '{path}' (FORMAT PARQUET)")
    with pytest.raises(ValueError, match="geometry column"):
        build_layer_query(
            con, LayerSource(source=str(path), layer="p", minzoom=0, maxzoom=1)
        )


def test_build_combined_query_unions_multiple_layers(tmp_path: Path) -> None:
    con = _connection()
    points_path = tmp_path / "points.parquet"
    polygons_path = tmp_path / "polygons.parquet"
    _write_points(con, points_path, count=2)
    con.execute(
        f"""
        COPY (
            SELECT i AS building_id, ST_Buffer(ST_Point(i, i), 0.5) AS geometry
            FROM range(1, 3) t(i)
        ) TO '{polygons_path}' (FORMAT PARQUET)
        """
    )
    query = build_combined_query(
        con,
        [
            LayerSource(source=str(points_path), layer="points", minzoom=0, maxzoom=10),
            LayerSource(
                source=str(polygons_path), layer="buildings", minzoom=11, maxzoom=14
            ),
        ],
    )
    rows = con.execute(query).fetchall()
    assert len(rows) == 4
    layers = {_parse_feature(row[0])["tippecanoe"]["layer"] for row in rows}
    assert layers == {"points", "buildings"}


def test_build_combined_query_single_layer_is_not_wrapped_in_union(
    tmp_path: Path,
) -> None:
    con = _connection()
    path = tmp_path / "points.parquet"
    _write_points(con, path, count=1)
    single = build_layer_query(
        con, LayerSource(source=str(path), layer="p", minzoom=0, maxzoom=1)
    )
    combined = build_combined_query(
        con, [LayerSource(source=str(path), layer="p", minzoom=0, maxzoom=1)]
    )
    assert combined == single


def test_build_combined_query_requires_at_least_one_layer() -> None:
    con = _connection()
    with pytest.raises(ValueError):
        build_combined_query(con, [])
