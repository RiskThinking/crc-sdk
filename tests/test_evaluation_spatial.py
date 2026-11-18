"""Spatial evaluation uses actual footprints and explicit point distances."""

from __future__ import annotations

import math
from collections.abc import Sequence

import crc_framework as cf
import pytest
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely.geometry import Point, Polygon, box  # type: ignore[import-untyped]

from crc_sdk._evaluation_spatial import (
    Distance,
    SpatialIndex,
    SpatialMatch,
    SpatialMatchError,
    spatial_match_from_document,
)
from crc_sdk._schema.dataset import DatasetRow


def _point(name: str, longitude: float, latitude: float = 0) -> DatasetRow:
    return DatasetRow(
        variable_id="depth",
        support_id=name,
        cell_id=h3.latlng_to_cell(latitude, longitude, 4),
        geometry=bytes(Point(longitude, latitude).wkb),
        curve=cf.QuantileCurve([0.2, 0.8], [1, 3]),
    )


def _index(rows: Sequence[DatasetRow]) -> SpatialIndex:
    return SpatialIndex(rows, SpatialMatch.covering())


def test_distance_policy_is_typed_and_round_trips() -> None:
    policy = SpatialMatch.nearest(max_distance=Distance.kilometres(1.25))
    assert policy.max_distance == Distance.metres(1250)
    assert spatial_match_from_document(policy._document()) == policy
    with pytest.raises(ValueError, match="nonnegative"):
        Distance.kilometres(-1)
    for value in (math.nan, math.inf, True):
        with pytest.raises(ValueError, match="finite"):
            Distance.metres(value)
    with pytest.raises(ValueError, match="invalid spatial"):
        spatial_match_from_document({"kind": "covering", "max_distance_metres": 5})


def test_point_match_does_not_expand_over_its_delivery_cell() -> None:
    point = _point("a", 12, 64)
    beside = Point(12.00001, 64)
    assert h3.latlng_to_cell(beside.y, beside.x, 4) == point.cell_id
    assert _index([point]).match(Point(12, 64).wkb).row is point
    missing = _index([point]).match(beside.wkb)
    assert missing.row is None and missing.failure_code == "no_covering_support"


def test_covering_uses_polygon_holes_and_full_asset_geometry() -> None:
    polygon = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
    )
    row = DatasetRow(
        variable_id="depth",
        support_id="footprint",
        cell_id=h3.latlng_to_cell(2, 2, 4),
        geometry=polygon.wkb,
        curve=cf.QuantileCurve([0.2, 0.8], [1, 3]),
    )
    spatial = _index([row])
    assert spatial.match(Point(1.5, 1.5).wkb).row is None
    assert spatial.match(box(3.7, 3.6, 4.1, 4.2).wkb).row is None
    assert spatial.match(box(3, 3, 4, 4).wkb).row is row
    assert spatial.match(Point(0, 1).wkb).row is row
    with pytest.raises(SpatialMatchError) as error:
        SpatialIndex([row], SpatialMatch.nearest(max_distance=Distance.metres(10)))
    assert error.value.code == "nearest_requires_points"


@pytest.mark.parametrize(("latitude", "longitude"), [(0, 179.999), (89.999, 0)])
def test_cell_support_uses_actual_spherical_cell_membership(
    latitude: float, longitude: float
) -> None:
    cell = h3.latlng_to_cell(latitude, longitude, 4)
    row = DatasetRow(
        variable_id="depth",
        support_id="cell",
        cell_id=cell,
        curve=cf.QuantileCurve([0.2, 0.8], [1, 3]),
    )
    spatial = _index([row])
    assert spatial.match(Point(longitude, latitude).wkb).row is row
    assert spatial.match(Point(0, 0).wkb).row is None


def test_duplicate_covering_supports_fail_without_implied_priority() -> None:
    with pytest.raises(SpatialMatchError) as error:
        _index([_point("a", -180), _point("b", 180)]).match(Point(180, 0).wkb)
    assert error.value.code == "overlapping_supports"
    assert "'a'" in str(error.value) and "'b'" in str(error.value)


def test_covering_priority_uses_only_matching_supports() -> None:
    first, second, elsewhere = (
        _point("first", 0),
        _point("second", 0),
        _point("elsewhere", 10),
    )
    policy = SpatialMatch.covering(preferred_supports=["elsewhere", "second", "first"])
    assert policy.preferred_supports == ("elsewhere", "second", "first")
    assert (
        spatial_match_from_document(
            {"kind": "covering", "preferred_supports": ["elsewhere", "second", "first"]}
        )
        == policy
    )
    asset = Point(0, 0).wkb
    for rows in ([first, second, elsewhere], [elsewhere, second, first]):
        assert SpatialIndex(rows, policy).match(asset).row is second
    with pytest.raises(SpatialMatchError, match="highest priority support"):
        SpatialIndex(
            [first, second], SpatialMatch.covering(preferred_supports=["elsewhere"])
        ).match(asset)
    with pytest.raises(ValueError, match="unique support IDs"):
        SpatialMatch.covering(preferred_supports=["first", "first"])


def test_nearest_uses_great_circle_metres_across_antimeridian() -> None:
    asset = Point(-179.99, 30)
    match = _point("regional", 179.99, 30)
    rows = [_point("distant", -179.95, 30), match, _point("other", 0)]
    spatial = SpatialIndex(
        rows, SpatialMatch.nearest(max_distance=Distance.kilometres(2.5))
    )
    result = spatial.match(asset.wkb)
    expected = h3.great_circle_distance((30, -179.99), (30, 179.99), unit="m")
    assert result.row is match
    assert result.distance_metres == pytest.approx(expected)
    missing = SpatialIndex(
        rows, SpatialMatch.nearest(max_distance=Distance.kilometres(1))
    ).match(asset.wkb)
    assert missing.row is None and missing.failure_code == "outside_max_distance"


def test_nearest_equal_distances_and_nonpoint_assets_fail_explicitly() -> None:
    spatial = SpatialIndex(
        [_point("a", -0.001), _point("b", 0.001)],
        SpatialMatch.nearest(max_distance=Distance.kilometres(2)),
    )
    with pytest.raises(SpatialMatchError) as tie:
        spatial.match(Point(0, 0).wkb)
    assert tie.value.code == "nearest_support_tie"
    with pytest.raises(SpatialMatchError) as incompatible:
        spatial.match(box(0, 0, 1, 1).wkb)
    assert incompatible.value.code == "nearest_requires_point_assets"
