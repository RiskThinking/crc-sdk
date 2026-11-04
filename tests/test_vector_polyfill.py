import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Polygon, box  # type: ignore[import-untyped]
from shapely.wkb import dumps  # type: ignore[import-untyped]

import crc_sdk.geometry.vector as vector_module
from crc_sdk.connectors.duckdb import ensure_extensions
from crc_sdk.geometry import (
    VectorContainment,
    expand_polygon_candidates,
    polyfill_wkb,
)
from crc_sdk.geometry.vector import _to_pyarrow_array


def test_to_pyarrow_array_passes_polyfill_cells_through() -> None:
    cells = polyfill_wkb(
        [dumps(box(0, 0, 1, 1))],
        2,
        containment=VectorContainment.COVERS,
        flatten=False,
    )
    converted = _to_pyarrow_array(cells)
    assert isinstance(converted, pa.Array)
    assert len(converted) == 1
    assert converted.to_pylist()[0]


def test_to_pyarrow_array_falls_back_to_pylist() -> None:
    calls: list[str] = []

    class _ListOnly:
        def to_pylist(self) -> list[list[int]]:
            calls.append("to_pylist")
            return [[1, 2], [3]]

    converted = _to_pyarrow_array(_ListOnly())
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


def test_expand_polygon_candidates_auto_batches_past_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """batch_rows=None picks one-shot vs. chunked based on a row-count check."""
    calls: list[int] = []
    original_expand_batched = vector_module._expand_batched

    def _spy_expand_batched(*args: object, **kwargs: object) -> int:
        calls.append(kwargs["batch_rows"])  # type: ignore[arg-type]
        return original_expand_batched(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(vector_module, "_expand_batched", _spy_expand_batched)
    monkeypatch.setattr(vector_module, "_AUTO_BATCH_THRESHOLD_ROWS", 1)

    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    con.execute(
        """
        CREATE TABLE polys AS
        SELECT 1::BIGINT AS poly_rid, ST_AsWKB(ST_GeomFromText(?)) AS wkb
        UNION ALL
        SELECT 2::BIGINT, ST_AsWKB(ST_GeomFromText(?))
        """,
        [
            "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON((1 0, 2 0, 2 1, 1 1, 1 0))",
        ],
    )
    count = vector_module.expand_polygon_candidates(
        con,
        "SELECT poly_rid, wkb FROM polys",
        2,
        containment=VectorContainment.COVERS,
        batch_rows=None,
    )
    assert count > 0
    # 2 rows > the monkeypatched threshold of 1, so it auto-selected the
    # default chunk size and went through the batched path, not one-shot.
    assert calls == [vector_module._AUTO_BATCH_ROWS]


def test_expand_polygon_candidates_stays_one_shot_under_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    original_expand_one_shot = vector_module._expand_one_shot

    def _spy_expand_one_shot(*args: object, **kwargs: object) -> int:
        calls.append(True)
        return original_expand_one_shot(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(vector_module, "_expand_one_shot", _spy_expand_one_shot)

    con = duckdb.connect()
    ensure_extensions(con, "spatial", "h3")
    con.execute(
        "CREATE TABLE polys AS SELECT 1::BIGINT AS poly_rid, "
        "ST_AsWKB(ST_GeomFromText(?)) AS wkb",
        ["POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"],
    )
    count = vector_module.expand_polygon_candidates(
        con,
        "SELECT poly_rid, wkb FROM polys",
        2,
        containment=VectorContainment.COVERS,
        batch_rows=None,
    )
    assert count > 0
    assert calls == [True]


def test_list_offsets_refuse_to_wrap_past_int32() -> None:
    import numpy as np

    offsets = vector_module._list_offsets(np.array([3, 0, 2], dtype=np.int64))
    assert offsets.to_pylist() == [0, 3, 3, 5]
    with pytest.raises(OverflowError, match="flatten=True"):
        vector_module._list_offsets(np.array([2**31 - 1, 1], dtype=np.int64))


def _connected(cells: set[int]) -> bool:
    from h3.api import basic_int  # type: ignore[import-untyped]

    start = next(iter(cells))
    seen, frontier = {start}, [start]
    while frontier:
        cell = frontier.pop()
        for neighbor in basic_int.grid_disk(cell, 1):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return seen == cells


def test_line_polyfill_traces_segments_h3_cannot_path_directly() -> None:
    from h3 import H3FailedError  # type: ignore[import-untyped]
    from h3.api import basic_int
    from shapely.geometry import LineString

    pentagon = basic_int.get_pentagons(2)[0]
    ring = list(basic_int.grid_ring(pentagon, 2))
    start, end = next(
        (a, b)
        for a in ring
        for b in ring
        if not _path_exists(basic_int, a, b, H3FailedError)
    )
    lines = [
        # Opposite sides of a pentagon.
        LineString([basic_int.cell_to_latlng(c)[::-1] for c in (start, end)]),
        # Too far apart for one grid path.
        LineString([(0.0, 0.0), (170.0, 0.0)]),
    ]
    resolutions = [2, 3]
    for line, resolution in zip(lines, resolutions):
        cells = polyfill_wkb([dumps(line)], resolution).to_pylist()[0]
        lng_lat = list(line.coords)
        endpoints = {
            basic_int.latlng_to_cell(lat, lng, resolution)
            for lng, lat in (lng_lat[0], lng_lat[-1])
        }
        assert endpoints <= set(cells)
        assert _connected(set(cells))


def _path_exists(h3: object, a: int, b: int, error: type[Exception]) -> bool:
    try:
        h3.grid_path_cells(a, b)  # type: ignore[attr-defined]
    except error:
        return False
    return True


def test_line_polyfill_matches_grid_path_cells_when_it_succeeds() -> None:
    from h3.api import basic_int
    from shapely.geometry import LineString

    line = LineString([(-79.4, 43.6), (-79.0, 43.9), (-78.6, 43.7)])
    expected: set[int] = set()
    cells_at = [basic_int.latlng_to_cell(lat, lng, 8) for lng, lat in line.coords]
    for a, b in zip(cells_at, cells_at[1:]):
        expected.update(basic_int.grid_path_cells(a, b))
    assert polyfill_wkb([dumps(line)], 8).to_pylist()[0] == sorted(expected)
