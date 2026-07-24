import pytest
from shapely.geometry import box  # type: ignore[import-untyped]
from shapely.ops import unary_union  # type: ignore[import-untyped]

from crc_sdk.geometry import (
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
