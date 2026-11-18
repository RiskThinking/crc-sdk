"""Asset locations, retained data, ownership, and bounded explicit SQL inputs."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely import from_wkb  # type: ignore[import-untyped]
from shapely.geometry import Point  # type: ignore[import-untyped]

from crc_sdk._assets import Assets, GeometryColumn, PointColumns
from crc_sdk._errors import CapabilityError
from crc_sdk._graph import ExecutionScope
from crc_sdk._runtime import Cache, Runtime


def _points() -> dict[str, object]:
    return {
        "asset_id": ["0001"],
        "longitude": [-79.4],
        "latitude": [43.7],
        "country": ["CA"],
    }


def test_constructors_and_column_bindings_do_no_io(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    source = Assets(path)
    assert source.column("country").name == "country"
    sql = Assets.sql("SELECT * FROM not_yet_created", location=PointColumns())
    assert sql.column("country").name == "country"
    with ExecutionScope() as scope, pytest.raises(FileNotFoundError):
        source._materialize(scope)
    with pytest.raises(AttributeError, match="immutable"):
        source.id_column = "changed"  # type: ignore[misc]


def test_owned_table_snapshot_retention_and_scope_memoization() -> None:
    longitude = np.array([-79.4])
    source = Assets(
        {**_points(), "longitude": longitude}, columns=["country", "asset_id"]
    )
    longitude[0] = 100.0
    with ExecutionScope() as scope:
        data = source._materialize(scope)
        assert source._materialize(scope) is data
    assert data.table.column_names == ["asset_id", "country"]
    assert from_wkb(data.geometries[0]).x == -79.4
    assert data.table.to_pylist()[0]["country"] == "CA"


def test_custom_mapping_and_multiple_locations_require_a_choice() -> None:
    table = {**_points(), "geometry": [Point(-80, 44).wkb]}
    with ExecutionScope() as scope:
        with pytest.raises(ValueError, match="ambiguous"):
            Assets(table)._materialize(scope)
        explicit = Assets(table, location=GeometryColumn())._materialize(scope)
        assert from_wkb(explicit.geometries[0]).x == -80
        custom = Assets(
            {"key": [7], "x": [-79.4], "y": [43.7]},
            id_column="key",
            location=PointColumns(longitude="x", latitude="y"),
        )._materialize(scope)
        assert custom.id_column == "key"
        with pytest.raises(ValueError, match="missing asset location"):
            Assets(
                _points(), location=PointColumns(longitude="x", latitude="y")
            )._materialize(scope)


@pytest.mark.parametrize(
    "ids", [["a", "a"], [None, "b"], [True, False], [1.0, 2.0], ["", "b"]]
)
def test_invalid_and_duplicate_ids_are_rejected(ids: list[object]) -> None:
    with ExecutionScope() as scope, pytest.raises(ValueError, match="asset ID"):
        Assets({"asset_id": ids, "longitude": [0, 0], "latitude": [0, 0]})._materialize(
            scope
        )


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(0, 179.999), (89.999, 20), (-89.999, -120)],
)
def test_cell_assets_keep_dateline_and_polar_footprints_local(
    latitude: float,
    longitude: float,
) -> None:
    cell = h3.latlng_to_cell(latitude, longitude, 5)
    with ExecutionScope() as scope:
        data = Assets({"asset_id": ["a"], "cell_id": [cell]})._materialize(scope)
    shape = from_wkb(data.geometries[0])
    assert shape.is_valid
    assert shape.covers(Point(longitude, latitude))
    assert not shape.covers(Point(0, 0))


def test_cell_location_uses_native_h3_footprint() -> None:
    cell = h3.latlng_to_cell(43.7, -79.4, 9)
    with ExecutionScope() as scope:
        data = Assets({"asset_id": [1], "cell_id": [cell]})._materialize(scope)
    shape = from_wkb(data.geometries[0])
    assert shape.geom_type == "Polygon"
    assert shape.contains(Point(-79.4, 43.7))


@pytest.mark.parametrize(
    "lon,lat", [(181, 0), (0, -91), (float("inf"), 0), (True, 0), (None, 0)]
)
def test_points_require_finite_wgs84(lon: object, lat: object) -> None:
    with ExecutionScope() as scope, pytest.raises(ValueError, match="WGS84"):
        Assets({"asset_id": [1], "longitude": [lon], "latitude": [lat]})._materialize(
            scope
        )


@pytest.mark.parametrize(
    "geometry",
    [
        b"bad",
        struct.pack("<BII2d", 1, 2, 1, 1, 2),
        Point(1, 2, 3).wkb,
        struct.pack("<BI3d", 1, 2001, 1, 2, 3),
        struct.pack("<BI3d", 1, 0x40000001, 1, 2, 3),
        struct.pack("<BII", 1, 7, 1) + struct.pack("<BI3d", 1, 2001, 1, 2, 3),
        struct.pack("<BII2d", 1, 0x20000001, 3857, 1, 2),
        Point().wkb,
        Point(181, 0).wkb,
    ],
)
def test_wkb_rejects_invalid_dimensions_crs_and_empty_geometry(geometry: bytes) -> None:
    with ExecutionScope() as scope, pytest.raises(ValueError, match="geometry"):
        Assets({"asset_id": [1], "geometry": [geometry]})._materialize(scope)


@pytest.mark.parametrize(
    "crs", [None, "EPSG:3857", {"id": {"authority": "EPSG", "code": 3857}}]
)
def test_declared_non_wgs84_or_unknown_crs_is_rejected(crs: object) -> None:
    table = pa.table(
        {"asset_id": [1], "geometry": [Point(1, 2).wkb]}
    ).replace_schema_metadata(
        {
            b"geo": json.dumps(
                {"columns": {"geometry": {"encoding": "WKB", "crs": crs}}}
            ).encode()
        }
    )
    with ExecutionScope() as scope, pytest.raises(ValueError, match="WGS84 CRS"):
        Assets(table)._materialize(scope)


@pytest.mark.parametrize(
    "crs", ["EPSG:4326", "OGC:CRS84", {"id": {"authority": "EPSG", "code": 4326}}]
)
def test_wgs84_geoparquet_and_wkb_are_preserved(tmp_path: Path, crs: object) -> None:
    geometry = cast(bytes, Point(1, 2).wkb)
    table = pa.table({"asset_id": [1], "geometry": [geometry]}).replace_schema_metadata(
        {
            b"geo": json.dumps(
                {"columns": {"geometry": {"encoding": "WKB", "crs": crs}}}
            ).encode()
        }
    )
    path = tmp_path / "assets.parquet"
    pq.write_table(table, path)
    with ExecutionScope() as scope:
        data = Assets(path)._materialize(scope)
    assert data.geometries == (geometry,)
    assert data.source is not None and data.source.resources[0].sha256


def test_csv_ids_preserve_leading_zeroes(tmp_path: Path) -> None:
    path = tmp_path / "assets.csv"
    path.write_text(
        "asset_id,longitude,latitude,unused\n0001,-79.4,43.7,x\n0002,-80,44,y\n"
    )
    with ExecutionScope() as scope:
        data = Assets(path, columns=[])._materialize(scope)
    assert data.table.to_pydict() == {"asset_id": ["0001", "0002"]}


def test_sql_is_explicit_bounded_and_keeps_external_connection_settings() -> None:
    connection = duckdb.connect(config={"threads": 2})
    try:
        before = connection.execute("SELECT current_setting('threads')").fetchone()
        source = Assets.sql(
            "SELECT i AS asset_id, 1.0 AS longitude, 2.0 AS latitude, "
            "repeat('x',1000) AS extra FROM range(100) t(i)",
            location=PointColumns(),
            columns=[],
        )
        with ExecutionScope(
            Runtime(connection=connection, batch_rows=3, memory_limit=10000)
        ) as scope:
            data = source._materialize(scope)
        assert data.table.num_rows == 100
        assert data.table.column_names == ["asset_id"]
        assert data.source is not None
        assert data.source.resources[0].uri.startswith("duckdb:query:")
        assert data.source.resources[0].sha256.startswith("sha256:")
        assert (
            connection.execute("SELECT current_setting('threads')").fetchone() == before
        )
        with ExecutionScope(
            Runtime(connection=connection, batch_rows=3, memory_limit=100)
        ) as scope:
            with pytest.raises(MemoryError, match="Assets exceed"):
                source._materialize(scope)
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "query",
    ["SELECT 1; SELECT 2", "CREATE TABLE accidental(x INTEGER)", "SET threads=1"],
)
def test_sql_rejects_mutations_and_multiple_statements(query: str) -> None:
    with ExecutionScope() as scope, pytest.raises(ValueError, match="one SELECT"):
        Assets.sql(query, location=PointColumns())._materialize(scope)


def test_strings_are_paths_and_sql_requires_explicit_mapping() -> None:
    with ExecutionScope() as scope, pytest.raises(FileNotFoundError):
        Assets("SELECT 1 AS asset_id")._materialize(scope)
    with pytest.raises(TypeError, match="explicit"):
        Assets.sql("SELECT 1", location=None)  # type: ignore[arg-type]


def test_empty_sql_has_owned_stable_columns_after_scope_closes() -> None:
    with ExecutionScope() as scope:
        data = Assets.sql(
            "SELECT 1 AS asset_id, 1.0 AS longitude, 2.0 AS latitude WHERE false",
            location=PointColumns(),
        )._materialize(scope)
    assert data.table.num_rows == 0
    assert data.table.column_names == ["asset_id", "longitude", "latitude"]
    assert data.geometries == ()


def test_offline_sql_cannot_open_an_external_reader(tmp_path: Path) -> None:
    assets = Assets.sql(
        "SELECT * FROM read_csv_auto('http://127.0.0.1:1/assets.csv')",
        location=PointColumns(),
    )
    with (
        ExecutionScope(Runtime(cache=Cache.offline(tmp_path))) as scope,
        pytest.raises(CapabilityError, match="cannot access external resources"),
    ):
        assets._materialize(scope)


def test_offline_sql_uses_an_owned_connection_without_reconfiguring_borrowed_sql(
    tmp_path: Path,
) -> None:
    assets = Assets.sql(
        "SELECT 'A' AS asset_id, 0.0 AS longitude, 0.0 AS latitude",
        location=PointColumns(),
    )
    cache = Cache.offline(tmp_path)
    with ExecutionScope(Runtime(cache=cache)) as scope:
        data = assets._materialize(scope)
        assert data.table["asset_id"].to_pylist() == ["A"]
        assert data.source is not None
    with duckdb.connect() as database:
        runtime = Runtime(cache=cache, connection=database)
        with (
            ExecutionScope(runtime) as scope,
            pytest.raises(CapabilityError, match="SDK-owned DuckDB"),
        ):
            assets._materialize(scope)
        assert database.execute(
            "SELECT current_setting('enable_external_access')"
        ).fetchone() == (True,)


def test_duplicate_arrow_column_names_are_rejected() -> None:
    table = pa.Table.from_arrays(
        [pa.array(["a"]), pa.array(["b"]), pa.array([1.0]), pa.array([2.0])],
        names=["asset_id", "asset_id", "longitude", "latitude"],
    )
    with (
        ExecutionScope() as scope,
        pytest.raises(ValueError, match="names must be unique"),
    ):
        Assets(table)._materialize(scope)


def test_geoarrow_declared_crs_is_checked() -> None:
    field = pa.field(
        "geometry",
        pa.binary(),
        metadata={
            b"ARROW:extension:name": b"geoarrow.wkb",
            b"ARROW:extension:metadata": b'{"crs":"EPSG:3857"}',
        },
    )
    table = pa.Table.from_arrays(
        [pa.array(["a"]), pa.array([Point(1, 2).wkb])],
        schema=pa.schema([pa.field("asset_id", pa.string()), field]),
    )
    with ExecutionScope() as scope, pytest.raises(ValueError, match="WGS84 CRS"):
        Assets(table)._materialize(scope)
