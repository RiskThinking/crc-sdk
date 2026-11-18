"""Geographic footprints for H3-indexed assets and data support."""

from __future__ import annotations

from typing import cast

from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely.affinity import translate  # type: ignore[import-untyped]
from shapely.geometry import Polygon, box  # type: ignore[import-untyped]
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]
from shapely.ops import unary_union  # type: ignore[import-untyped]


def cell_polygon(cell_index: int) -> BaseGeometry:
    """Return the H3 footprint in WGS84, split at the antimeridian."""
    points: list[tuple[float, float]] = []
    for latitude, longitude in h3.cell_to_boundary(cell_index):
        if points:
            longitude += 360 * round((points[-1][0] - longitude) / 360)
        points.append((longitude, latitude))
    resolution = h3.get_resolution(cell_index)
    north = h3.latlng_to_cell(90, 0, resolution) == cell_index
    south = h3.latlng_to_cell(-90, 0, resolution) == cell_index
    if north or south:
        first_longitude, first_latitude = points[0]
        closing_longitude = first_longitude + 360 * round(
            (points[-1][0] - first_longitude) / 360
        )
        pole = 90.0 if north else -90.0
        points.extend(
            [
                (closing_longitude, first_latitude),
                (closing_longitude, pole),
                (first_longitude, pole),
            ]
        )
    polygon = Polygon(points)
    world = box(-180, -90, 180, 90)
    parts: list[BaseGeometry] = []
    for offset in (-720, -360, 0, 360, 720):
        piece = translate(polygon, xoff=offset).intersection(world)
        parts.extend(
            part
            for part in getattr(piece, "geoms", [piece])
            if isinstance(part, Polygon) and not part.is_empty and part.area > 0
        )
    return cast(BaseGeometry, unary_union(parts))
