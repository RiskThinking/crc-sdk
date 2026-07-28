import json
from pathlib import Path

import pytest
from shapely.geometry import (  # type: ignore[import-untyped]
    MultiPolygon,
    Polygon,
    box,
    mapping,
)
from shapely.ops import unary_union  # type: ignore[import-untyped]

from crc_sdk.geometry import (
    GeoFormat,
    H3Indexer,
    PolyfillMode,
    cell_polygon,
    estimate_resolutions,
    intersecting_cells,
    point_to_cell,
)


def test_intersecting_cells_conservatively_cover_source_geometry() -> None:
    source = box(-0.25, 51.25, 0.25, 51.75)
    cells = intersecting_cells(source, 5)
    coverage = unary_union([cell_polygon(cell) for cell in cells])

    assert cells
    assert source.difference(coverage).area == pytest.approx(0.0, abs=1e-10)
    assert all(cell_polygon(cell).intersects(source) for cell in cells)
    assert point_to_cell(0.0, 51.5, 5) in cells


def test_resolution_estimates_report_error_and_expanded_rows() -> None:
    source = box(-0.25, 51.25, 0.25, 51.75)
    estimates = estimate_resolutions([source], [4, 5])

    assert [estimate.resolution for estimate in estimates] == [4, 5]
    assert all(estimate.coverage_error >= 0.0 for estimate in estimates)
    assert all(estimate.row_count == estimate.cell_count for estimate in estimates)
    assert estimates[1].cell_count >= estimates[0].cell_count


def test_h3_indexer_polyfills_multipolygon_and_polygon() -> None:
    polygon = box(-0.25, 51.25, 0.25, 51.75)
    multipolygon = MultiPolygon(
        [
            box(-0.25, 51.25, -0.05, 51.45),
            box(0.05, 51.55, 0.25, 51.75),
        ]
    )
    indexer = H3Indexer()
    indexer.con.execute(
        """
        CREATE OR REPLACE TABLE geoms AS
        SELECT ST_GeomFromText(?) AS geometry, 'polygon' AS kind
        UNION ALL
        SELECT ST_GeomFromText(?), 'multipolygon'
        """,
        [polygon.wkt, multipolygon.wkt],
    )

    sql = indexer.build_h3_query(
        "geoms",
        resolution=5,
        mode=PolyfillMode.OVERLAP,
    )
    result = indexer.con.execute(sql)
    columns = [col[0] for col in result.description]
    rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    by_kind: dict[str, list[int]] = {kind: [] for kind in ("polygon", "multipolygon")}
    for row in rows:
        by_kind[str(row["kind"])].append(int(row["h3_index"]))

    assert by_kind["polygon"]
    assert by_kind["multipolygon"]
    assert len(by_kind["multipolygon"]) == len(set(by_kind["multipolygon"]))
    assert set(by_kind["multipolygon"]).issubset(set(by_kind["polygon"]))


def test_h3_indexer_overlap_covers_adjacent_and_small_polygons() -> None:
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])
    small = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)])
    indexer = H3Indexer()
    indexer.con.execute(
        """
        CREATE OR REPLACE TABLE geoms AS
        SELECT ST_GeomFromText(?) AS geometry, 'a' AS kind
        UNION ALL
        SELECT ST_GeomFromText(?), 'b'
        UNION ALL
        SELECT ST_GeomFromText(?), 'small'
        """,
        [poly_a.wkt, poly_b.wkt, small.wkt],
    )
    sql = indexer.build_h3_query(
        "geoms",
        resolution=2,
        mode=PolyfillMode.OVERLAP,
        as_string=True,
    )
    result = indexer.con.execute(sql)
    columns = [col[0] for col in result.description]
    rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    by_kind: dict[str, set[str]] = {"a": set(), "b": set(), "small": set()}
    for row in rows:
        by_kind[str(row["kind"])].add(str(row["h3_index"]))

    assert by_kind["a"]
    assert by_kind["b"]
    assert by_kind["a"] & by_kind["b"]

    small_sql = indexer.build_h3_query(
        "(SELECT * FROM geoms WHERE kind = 'small')",
        resolution=5,
        mode=PolyfillMode.OVERLAP,
        as_string=True,
    )
    small_rows = indexer.con.execute(small_sql).fetchall()
    assert small_rows


def test_build_h3_query_from_file_composes_format_and_polyfill(
    tmp_path: Path,
) -> None:
    polygon = box(-0.25, 51.25, 0.25, 51.75)
    geojson_path = tmp_path / "aoi.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "aoi"},
                        "geometry": mapping(polygon),
                    }
                ],
            }
        )
    )
    indexer = H3Indexer()
    sql = indexer.build_h3_query_from_file(
        str(geojson_path), GeoFormat.GEOJSON, resolution=5
    )
    result = indexer.con.execute(sql)
    columns = [col[0] for col in result.description]
    cells = {
        int(dict(zip(columns, row, strict=True))["h3_index"])
        for row in result.fetchall()
    }
    assert cells == set(intersecting_cells(polygon, 5))
