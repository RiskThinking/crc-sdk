"""Matching against the declared spatial support of a scientific series."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import cast

from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely import STRtree, from_wkb  # type: ignore[import-untyped]
from shapely.affinity import translate  # type: ignore[import-untyped]
from shapely.geometry import MultiPolygon, Point, Polygon, box  # type: ignore[import-untyped]
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]

from ._dataset_schema import _array, _identifier, _number, _object
from ._errors import CapabilityError
from ._geometry import cell_polygon
from ._schema.dataset import DatasetRow
from ._selection import _angle, _vector

# H3 uses an authalic sphere. Recover its radius from a quarter-circle arc.
_RADIUS_METRES = float(h3.great_circle_distance((0, 0), (0, 90), unit="m")) / (
    math.pi / 2
)


@dataclass(frozen=True, slots=True, init=False)
class Distance:
    """An explicit nonnegative physical distance."""

    value_metres: float

    def __init__(self) -> None:
        raise TypeError("use Distance.metres(value) or Distance.kilometres(value)")

    @classmethod
    def metres(cls, value: float) -> Distance:
        value = _number(value, "distance")
        if value < 0:
            raise ValueError("distance must be nonnegative")
        result = object.__new__(cls)
        object.__setattr__(result, "value_metres", 0.0 if value == 0 else value)
        return result

    @classmethod
    def kilometres(cls, value: float) -> Distance:
        return cls.metres(_number(value, "distance") * 1000)


class SpatialMatchKind(Enum):
    COVERING = "covering"
    NEAREST = "nearest"


@dataclass(frozen=True, slots=True, init=False)
class SpatialMatch:
    """Match declared coverage or request an explicit bounded point search."""

    kind: SpatialMatchKind
    max_distance: Distance | None
    preferred_supports: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("use SpatialMatch.covering() or SpatialMatch.nearest(...)")

    @classmethod
    def covering(cls, *, preferred_supports: Sequence[str] = ()) -> SpatialMatch:
        """Match declared coverage, resolving overlaps by explicit support priority."""
        if not isinstance(preferred_supports, Sequence) or isinstance(
            preferred_supports, (str, bytes)
        ):
            raise TypeError(
                "preferred_supports must be a sequence of exact support IDs"
            )
        supports = tuple(_identifier(id_, "support ID") for id_ in preferred_supports)
        if len(set(supports)) != len(supports):
            raise ValueError("preferred_supports must contain unique support IDs")
        result = object.__new__(cls)
        object.__setattr__(result, "kind", SpatialMatchKind.COVERING)
        object.__setattr__(result, "max_distance", None)
        object.__setattr__(result, "preferred_supports", supports)
        return result

    @classmethod
    def nearest(cls, *, max_distance: Distance) -> SpatialMatch:
        if not isinstance(max_distance, Distance):
            raise TypeError(
                "max_distance must be Distance.metres(...) or .kilometres(...)"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "kind", SpatialMatchKind.NEAREST)
        object.__setattr__(result, "max_distance", max_distance)
        object.__setattr__(result, "preferred_supports", ())
        return result

    def _document(self) -> dict[str, object]:
        document: dict[str, object] = {"kind": self.kind.value}
        if self.max_distance is not None:
            document["max_distance_metres"] = self.max_distance.value_metres
        if self.preferred_supports:
            document["preferred_supports"] = list(self.preferred_supports)
        return document


def spatial_match_from_document(value: object) -> SpatialMatch:
    fields = _object(value, {"kind"}, {"max_distance_metres", "preferred_supports"})
    if (
        fields["kind"] == SpatialMatchKind.COVERING.value
        and "max_distance_metres" not in fields
    ):
        if "preferred_supports" not in fields:
            return SpatialMatch.covering()
        order = _array(fields["preferred_supports"], "preferred_supports")
        if not order:
            raise ValueError(
                "omit preferred_supports when no priorities are configured"
            )
        return SpatialMatch.covering(
            preferred_supports=tuple(_identifier(id_, "support ID") for id_ in order)
        )
    if fields["kind"] == SpatialMatchKind.NEAREST.value and fields.keys() == {
        "kind",
        "max_distance_metres",
    }:
        return SpatialMatch.nearest(
            max_distance=Distance.metres(
                _number(fields["max_distance_metres"], "max_distance_metres")
            )
        )
    raise ValueError("invalid spatial matching policy")


class SpatialMatchError(CapabilityError):
    """A matching policy cannot resolve the requested support."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class MatchResult:
    row: DatasetRow | None
    distance_metres: float | None = None
    failure_code: str | None = None
    failure_message: str | None = None


def _point_key(point: Point) -> tuple[float, float]:
    latitude, longitude = float(point.y), float(point.x)
    if abs(latitude) == 90:
        return latitude, 0.0
    return latitude, -180.0 if longitude == 180 else longitude


def candidate_cells(geometries: Iterable[bytes]) -> frozenset[int]:
    """H3 cells, at every resolution, that can cover the given assets.

    A point is covered only by the cell containing it, as `SpatialIndex` finds
    it. A cell covering a polygon contains the polygon's representative point;
    the one-ring absorbs the difference between planar cell polygons and H3
    assignment near cell edges.
    """
    cells: set[int] = set()
    for value in geometries:
        asset = cast(BaseGeometry, from_wkb(value))
        if asset.is_empty:
            continue
        if isinstance(asset, Point):
            latitude, longitude = _point_key(asset)
            cells.update(
                int(h3.latlng_to_cell(latitude, longitude, resolution))
                for resolution in range(16)
            )
            continue
        point = asset.representative_point()
        for resolution in range(16):
            cell = h3.latlng_to_cell(float(point.y), float(point.x), resolution)
            cells.update(int(item) for item in h3.grid_disk(cell, 1))
    return frozenset(cells)


def _rectangular_quadrants(point: Point, shape: BaseGeometry) -> set[tuple[bool, bool]]:
    """Return the local (north, east) quadrants inside an axis-aligned cell."""
    if not isinstance(shape, (Polygon, MultiPolygon)):
        return set()
    pieces = shape.geoms if isinstance(shape, MultiPolygon) else (shape,)
    x, y = point.x, point.y
    longitudes = (x, -x) if abs(x) == 180 else (x,)
    quadrants: set[tuple[bool, bool]] = set()
    for piece in pieces:
        west, south, east, north = piece.bounds
        if not piece.equals(box(west, south, east, north)):
            return set()
        for longitude in longitudes:
            if west <= longitude <= east and south <= y <= north:
                vertical = [(False, south < y), (True, y < north)]
                horizontal = [(False, west < longitude), (True, longitude < east)]
                quadrants.update(
                    (above, right)
                    for above, has_height in vertical
                    for right, has_width in horizontal
                    if has_height and has_width
                )
    return quadrants


def _source_supports(rows: Sequence[DatasetRow]) -> tuple[DatasetRow, ...]:
    supports: dict[str, DatasetRow] = {}
    for row in rows:
        existing = supports.get(row.support_id)
        if existing is not None:
            if row.geometry is None or existing.geometry is None:
                same_support = (
                    row.geometry == existing.geometry
                    and row.cell_id == existing.cell_id
                )
            else:
                same_support = row.geometry == existing.geometry or from_wkb(
                    row.geometry
                ).equals(from_wkb(existing.geometry))
            delivery_fields = {"cell_id", "geometry"}
            if not same_support or {
                key: value
                for key, value in row._document().items()
                if key not in delivery_fields
            } != {
                key: value
                for key, value in existing._document().items()
                if key not in delivery_fields
            }:
                raise SpatialMatchError(
                    "conflicting_support_delivery",
                    "Delivery cells disagree on the geometry or model of support "
                    f"{row.support_id!r}.",
                )
            if existing.cell_id <= row.cell_id:
                continue
        supports[row.support_id] = row
    return tuple(supports.values())


class SpatialIndex:
    """Spatial candidates for one member and one series."""

    def __init__(self, rows: Sequence[DatasetRow], policy: SpatialMatch) -> None:
        self._policy = policy
        self._support_priority = {
            support: rank for rank, support in enumerate(policy.preferred_supports)
        }
        self._rows = _source_supports(rows)
        # Matching reads only supports, so series with this layout match alike.
        self.layout = tuple((r.support_id, r.cell_id, r.geometry) for r in self._rows)
        self._positions = {id(row): index for index, row in enumerate(self._rows)}
        self._points: dict[tuple[float, float], list[int]] = {}
        self._cells: dict[int, dict[int, list[int]]] = {}
        self._areas: list[BaseGeometry] = []
        self._area_rows: list[int] = []
        self._point_tree: STRtree | None = None
        self._cell_tree: STRtree | None = None
        self._cell_shapes: list[BaseGeometry] = []
        self._cell_rows: list[int] = []
        if policy.kind is SpatialMatchKind.NEAREST:
            self._prepare_nearest()
            return
        for index, row in enumerate(self._rows):
            if row.geometry is None:
                resolution = int(h3.get_resolution(row.cell_id))
                self._cells.setdefault(resolution, {}).setdefault(
                    row.cell_id, []
                ).append(index)
                continue
            shape = from_wkb(row.geometry)
            if isinstance(shape, Point):
                self._points.setdefault(_point_key(shape), []).append(index)
            else:
                self._areas.append(shape)
                self._area_rows.append(index)
        self._area_tree = STRtree(self._areas)

    def _prepare_nearest(self) -> None:
        points: list[Point] = []
        coordinates: list[tuple[float, float]] = []
        for row in self._rows:
            shape = None if row.geometry is None else from_wkb(row.geometry)
            if not isinstance(shape, Point):
                raise SpatialMatchError(
                    "nearest_requires_points",
                    "SpatialMatch.nearest(...) requires point-supported series. "
                    "Use SpatialMatch.covering() for polygon or H3 support.",
                )
            latitude, longitude = _point_key(shape)
            points.append(Point(longitude, latitude))
            coordinates.append((latitude, longitude))
        self._nearest_coordinates = tuple(coordinates)
        self._nearest_vectors = tuple(_vector(*point) for point in coordinates)
        self._point_tree = STRtree(points)

    def _prepare_cells(self) -> None:
        for cells in self._cells.values():
            for cell, indices in cells.items():
                shape = cell_polygon(cell)
                for index in indices:
                    self._cell_shapes.append(shape)
                    self._cell_rows.append(index)
        self._cell_tree = STRtree(self._cell_shapes)

    @staticmethod
    def _query_covering(
        tree: STRtree,
        shapes: Sequence[BaseGeometry],
        indices: Sequence[int],
        asset: BaseGeometry,
    ) -> set[int]:
        west, _, east, _ = asset.bounds
        candidates = [asset]
        if west == -180:
            candidates.append(translate(asset, xoff=360))
        if east == 180:
            candidates.append(translate(asset, xoff=-360))
        return {
            indices[int(index)]
            for candidate in candidates
            for index in tree.query(candidate)
            if shapes[int(index)].covers(candidate)
        }

    def _covering(self, asset: BaseGeometry) -> MatchResult:
        indices = self._query_covering(
            self._area_tree, self._areas, self._area_rows, asset
        )
        if isinstance(asset, Point):
            latitude, longitude = _point_key(asset)
            indices.update(self._points.get((latitude, longitude), ()))
            for resolution, cells in self._cells.items():
                cell = int(h3.latlng_to_cell(latitude, longitude, resolution))
                indices.update(cells.get(cell, ()))
        elif self._cells:
            if self._cell_tree is None:
                self._prepare_cells()
            assert self._cell_tree is not None
            indices.update(
                self._query_covering(
                    self._cell_tree, self._cell_shapes, self._cell_rows, asset
                )
            )
        if len(indices) > 1:
            ranked = [
                (self._support_priority[support], index)
                for index in indices
                if (support := self._rows[index].support_id) in self._support_priority
            ]
            if ranked:
                rank, preferred = min(ranked)
                if sum(priority == rank for priority, _ in ranked) == 1:
                    return MatchResult(self._rows[preferred])
            if isinstance(asset, Point):
                owner = self._grid_edge_owner(asset, indices)
                if owner is not None:
                    return MatchResult(self._rows[owner])
            raise self._ambiguity("overlapping_supports", indices)
        if not indices:
            return MatchResult(None, failure_code="no_covering_support")
        return MatchResult(self._rows[next(iter(indices))])

    def _grid_edge_owner(self, point: Point, indices: set[int]) -> int | None:
        owners: dict[tuple[bool, bool], int] = {}
        for index in indices:
            geometry = self._rows[index].geometry
            if geometry is None:
                return None
            quadrants = _rectangular_quadrants(point, from_wkb(geometry))
            if not quadrants or not quadrants.isdisjoint(owners):
                return None
            owners.update(dict.fromkeys(quadrants, index))
        # Shared edges belong to the cell to the north or east. Where the
        # preferred cell is absent, keep coverage from the remaining cells.
        return owners[max(owners)]

    def _nearest(self, asset: BaseGeometry) -> MatchResult:
        if not isinstance(asset, Point):
            raise SpatialMatchError(
                "nearest_requires_point_assets",
                "SpatialMatch.nearest(...) requires point asset locations. Supply "
                "point geometry or evaluate with SpatialMatch.covering().",
            )
        assert self._point_tree is not None and self._policy.max_distance is not None
        latitude, longitude = _point_key(asset)
        limit = self._policy.max_distance.value_metres
        angle = limit / _RADIUS_METRES
        degrees = math.degrees(min(math.pi, angle)) + 1e-12
        south, north = max(-90.0, latitude - degrees), min(90.0, latitude + degrees)
        if south == -90 or north == 90:
            windows = (box(-180, south, 180, north),)
        else:
            longitude_delta = (
                math.degrees(
                    math.asin(
                        min(1.0, math.sin(angle) / math.cos(math.radians(latitude)))
                    )
                )
                + 1e-12
            )
            windows = tuple(
                box(
                    longitude + offset - longitude_delta,
                    south,
                    longitude + offset + longitude_delta,
                    north,
                )
                for offset in (-360, 0, 360)
            )
        candidates = {
            int(index) for window in windows for index in self._point_tree.query(window)
        }
        vector = _vector(latitude, longitude)
        distances = [
            (
                _RADIUS_METRES * _angle(vector, self._nearest_vectors[index]),
                index,
            )
            for index in candidates
        ]
        within = [
            (distance, index) for distance, index in distances if distance <= limit
        ]
        if not within:
            return MatchResult(None, failure_code="outside_max_distance")
        distance, best = min(within)
        tied = [
            index
            for candidate, index in within
            if math.isclose(candidate, distance, rel_tol=1e-12, abs_tol=1e-7)
        ]
        if len(tied) > 1:
            raise self._ambiguity("nearest_support_tie", tied)
        return MatchResult(self._rows[best], distance)

    def _ambiguity(
        self, code: str, candidates: Sequence[int] | set[int]
    ) -> SpatialMatchError:
        names = sorted(
            (self._rows[index].support_id, self._rows[index].cell_id)
            for index in candidates
        )
        reason = (
            "Multiple declared supports cover the asset"
            if code == "overlapping_supports"
            else "Multiple point supports are tied at the nearest distance"
        )
        return SpatialMatchError(
            code,
            f"{reason}: {names!r}. "
            + (
                "Set SpatialMatch.covering(preferred_supports=[...]) with a unique "
                "highest priority support."
                if code == "overlapping_supports"
                else "Select a series with unambiguous spatial support."
            ),
        )

    def translated(self, result: MatchResult, source: SpatialIndex) -> MatchResult:
        """The same match in this index, for a `source` index with this layout."""
        if result.row is None or source is self:
            return result
        return replace(result, row=self._rows[source._positions[id(result.row)]])

    def match(self, asset_geometry: bytes) -> MatchResult:
        asset = cast(BaseGeometry, from_wkb(asset_geometry))
        if self._policy.kind is SpatialMatchKind.NEAREST:
            return self._nearest(asset)
        return self._covering(asset)
