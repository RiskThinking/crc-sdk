from pathlib import Path

import duckdb
import pyarrow as pa
from shapely.geometry import Polygon, box
from shapely.wkb import dumps

from crc_sdk.connectors.duckdb import ensure_extensions
from crc_sdk.geometry import (
    VectorContainment,
    expand_polygon_candidates,
    polyfill_wkb,
)
from crc_sdk.geometry.vector import _to_pyarrow_array


def test_to_pyarrow_array_uses_arrow_bridge_for_h3ronpy_cells() -> None:
    cells = polyfill_wkb(
        [dumps(box(0, 0, 1, 1))],
        2,
        containment=VectorContainment.COVERS,
        flatten=False,
    )
    converted = _to_pyarrow_array(cells, pa)
    assert isinstance(converted, pa.Array)
    assert len(converted) == 1
    assert converted.to_pylist()[0]


def test_to_pyarrow_array_falls_back_to_pylist() -> None:
    calls: list[str] = []

    class _ListOnly:
        def to_pylist(self):
            calls.append("to_pylist")
            return [[1, 2], [3]]

    converted = _to_pyarrow_array(_ListOnly(), pa)
    assert calls == ["to_pylist"]
    assert converted.to_pylist() == [[1, 2], [3]]


def test_covers_polyfill_finds_cells_for_small_and_adjacent_polygons() -> None:
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])
    small = box(0, 0, 0.001, 0.001)
    cells = polyfill_wkb(
        [dumps(poly_a), dumps(poly_b), dumps(small)],
        2,
        containment=VectorContainment.COVERS,
    )
    rows = cells.to_pylist()
    assert rows[0]
    assert rows[1]
    assert set(rows[0]) & set(rows[1])
    assert rows[2]


def test_expand_polygon_candidates_one_shot_and_batched() -> None:
    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    con.execute(
        """
        CREATE TABLE polys AS
        SELECT 1::BIGINT AS poly_rid,
               ST_AsWKB(ST_GeomFromText(?)) AS wkb
        UNION ALL
        SELECT 2::BIGINT,
               ST_AsWKB(ST_GeomFromText(?))
        """,
        [
            "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON((1 0, 2 0, 2 1, 1 1, 1 0))",
        ],
    )
    one_shot = expand_polygon_candidates(
        con,
        "SELECT poly_rid, wkb FROM polys",
        2,
        containment=VectorContainment.COVERS,
        batch_rows=None,
    )
    assert one_shot > 0
    assert con.execute(
        "SELECT COUNT(DISTINCT poly_rid) FROM candidates"
    ).fetchone() == (2,)

    batched = expand_polygon_candidates(
        con,
        "SELECT poly_rid, wkb FROM polys",
        2,
        containment=VectorContainment.COVERS,
        batch_rows=1,
        candidates_table="candidates_batched",
    )
    assert batched == one_shot
    assert con.execute(
        "SELECT COUNT(DISTINCT poly_rid) FROM candidates_batched"
    ).fetchone() == (2,)
