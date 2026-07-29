import json
from pathlib import Path

import duckdb
import numpy as np
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
    average_edge_length_m,
    cell_polygon,
    estimate_resolutions,
    intersecting_cells,
    max_pixel_spacing_m,
    pixel_grid_resolution,
    point_to_cell,
    reduce_h3_values,
    sample_grid_to_h3,
    subsample_offsets,
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


def test_h3_indexer_default_connection_is_resource_tuned(tmp_path: Path) -> None:
    indexer = H3Indexer(work_dir=tmp_path)
    settings = dict(
        indexer.con.execute("SELECT name, value FROM duckdb_settings()").fetchall()
    )
    assert int(settings["threads"]) > 0
    assert (tmp_path / "duckdb-temp").is_dir()


def test_h3_indexer_explicit_connection_is_not_overridden() -> None:
    explicit = duckdb.connect()
    indexer = H3Indexer(explicit)
    assert indexer.con is explicit


def test_average_edge_length_m_matches_known_table_values() -> None:
    assert average_edge_length_m(0) == pytest.approx(1_281_256.011, rel=1e-6)
    assert average_edge_length_m(15) == pytest.approx(0.584169, rel=1e-6)
    with pytest.raises(ValueError):
        average_edge_length_m(16)


def test_max_pixel_spacing_m_shrinks_with_finer_resolution() -> None:
    assert max_pixel_spacing_m(5) > max_pixel_spacing_m(10)


def test_pixel_grid_resolution_picks_finest_covered_resolution() -> None:
    # A pixel much larger than any cell at high resolution should still
    # resolve to a coarse, valid resolution rather than overshooting.
    coarse = pixel_grid_resolution(500_000.0)
    assert 0 <= coarse <= 3

    # A tiny pixel should resolve to a much finer resolution.
    fine = pixel_grid_resolution(5.0)
    assert fine > coarse

    with pytest.raises(ValueError):
        pixel_grid_resolution(10.0, max_subsample=0.0)


def test_subsample_offsets_are_centered_and_span_unit_interval() -> None:
    offsets = subsample_offsets(4)
    assert len(offsets) == 4
    assert offsets[0] == pytest.approx(-0.375)
    assert offsets[-1] == pytest.approx(0.375)
    assert np.all(np.diff(offsets) > 0)

    with pytest.raises(ValueError):
        subsample_offsets(0)


def test_reduce_h3_values_collapses_duplicate_cells() -> None:
    cells = np.array([5, 1, 5, 3, 1], dtype=np.uint64)
    values = np.array([1.0, 2.0, 4.0, 3.0, 9.0], dtype=np.float64)

    max_cells, max_values = reduce_h3_values(cells, values, reduce="max")
    assert list(max_cells) == [1, 3, 5]
    assert list(max_values) == [9.0, 3.0, 4.0]

    min_cells, min_values = reduce_h3_values(cells, values, reduce="min")
    assert list(min_values) == [2.0, 3.0, 1.0]

    with pytest.raises(ValueError):
        reduce_h3_values(cells, values, reduce="sum")  # type: ignore[arg-type]


def test_reduce_h3_values_handles_empty_input() -> None:
    cells, values = reduce_h3_values(
        np.array([], dtype=np.uint64), np.array([], dtype=np.float64)
    )
    assert len(cells) == 0
    assert len(values) == 0


def test_sample_grid_to_h3_reduces_points_landing_in_one_cell() -> None:
    # Two points close enough together to land in the same coarse cell.
    lons = [-0.001, 0.001, 10.0]
    lats = [51.5, 51.5, -33.0]
    values = [1.0, 5.0, 2.0]

    cells, reduced = sample_grid_to_h3(lons, lats, values, resolution=2, reduce="max")
    assert len(cells) == 2
    assert set(np.round(reduced, 3)) == {5.0, 2.0}

    with pytest.raises(ValueError):
        sample_grid_to_h3(lons, lats, values, resolution=16)
