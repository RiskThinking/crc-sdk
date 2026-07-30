import json
from pathlib import Path

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
        feature = json.loads(feature_json)
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
    coordinates = json.loads(feature_json)["geometry"]["coordinates"]
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
    layers = {json.loads(row[0])["tippecanoe"]["layer"] for row in rows}
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
