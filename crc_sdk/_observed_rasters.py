"""Bounded native raster decoding with actual pixel support geometry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from ._inspection import Bounds
from ._selection import support_intersects


def raster_records(
    path: Path,
    *,
    provider: str,
    variable_id: str,
    period_id: str | None,
    area: Bounds,
    time: str | None = None,
    categorical: bool = False,
    block_pixels: int = 65536,
) -> Iterator[dict[str, object]]:
    """Yield pixels from verified local bytes; no network handles outlive a scope."""
    try:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.warp import transform, transform_bounds  # type: ignore[import-untyped]
        from rasterio.windows import Window, from_bounds  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "Raster sources require `pip install crc-sdk[raster]`"
        ) from error

    pieces: tuple[Bounds, ...]
    if area.west > area.east:
        pieces = (
            Bounds(west=area.west, south=area.south, east=180, north=area.north),
            Bounds(west=-180, south=area.south, east=area.east, north=area.north),
        )
    else:
        pieces = (area,)
    seen: set[tuple[int, int]] = set()
    with rasterio.open(path) as raster:
        if raster.crs is None or raster.count != 1:
            raise ValueError("Source raster requires one band and an explicit CRS")
        names = raster.tags(1)
        tags = raster.tags()
        source_row_offset = int(tags.get("crc_source_row", 0))
        source_column_offset = int(tags.get("crc_source_column", 0))
        source_transform = (
            raster.transform
            if "crc_source_transform" not in tags
            else rasterio.Affine(*json.loads(tags["crc_source_transform"]))
        )
        for piece in pieces:
            native_bounds = transform_bounds(
                "EPSG:4326",
                raster.crs,
                piece.west,
                piece.south,
                piece.east,
                piece.north,
                densify_pts=21,
            )
            requested = from_bounds(*native_bounds, transform=raster.transform)
            column_start = max(0, int(requested.col_off // 1))
            row_start = max(0, int(requested.row_off // 1))
            column_stop = min(
                raster.width, int(-(-(requested.col_off + requested.width) // 1))
            )
            row_stop = min(
                raster.height, int(-(-(requested.row_off + requested.height) // 1))
            )
            width = column_stop - column_start
            if width <= 0 or row_stop <= row_start:
                continue
            columns_per_block = min(width, max(1, block_pixels))
            rows_per_block = max(1, block_pixels // columns_per_block)
            for first_row in range(row_start, row_stop, rows_per_block):
                height = min(rows_per_block, row_stop - first_row)
                for first_column in range(column_start, column_stop, columns_per_block):
                    count = min(columns_per_block, column_stop - first_column)
                    values = raster.read(
                        1,
                        window=Window(first_column, first_row, count, height),
                        masked=True,
                    )
                    for row_offset in range(height):
                        for column_offset in range(count):
                            row, column = (
                                first_row + row_offset,
                                first_column + column_offset,
                            )
                            if (row, column) in seen:
                                continue
                            seen.add((row, column))
                            source_row = row + source_row_offset
                            source_column = column + source_column_offset
                            corners = [
                                source_transform @ pair
                                for pair in (
                                    (source_column, source_row),
                                    (source_column + 1, source_row),
                                    (source_column + 1, source_row + 1),
                                    (source_column, source_row + 1),
                                )
                            ]
                            xs, ys = transform(
                                raster.crs,
                                "EPSG:4326",
                                [v[0] for v in corners],
                                [v[1] for v in corners],
                            )
                            geometry = Polygon(zip(xs, ys, strict=True))
                            if geometry.is_empty or not geometry.is_valid:
                                raise ValueError(
                                    "Source raster has invalid WGS84 pixel support"
                                )
                            wkb = cast(bytes, geometry.wkb)
                            centre = geometry.centroid
                            cell = h3.latlng_to_cell(centre.y, centre.x, 9)
                            if not support_intersects(area, geometry=wkb, cell_id=cell):
                                continue
                            sample = values[row_offset, column_offset]
                            missing = (
                                bool(values.mask[row_offset, column_offset])
                                if values.mask.ndim
                                else bool(values.mask)
                            )
                            value = None if missing else float(sample)
                            record: dict[str, object] = {
                                "variable_id": variable_id,
                                "scenario_id": None,
                                "period_id": period_id,
                                "support_id": (
                                    f"{provider}:pixel:"
                                    f"{hashlib.sha256(wkb).hexdigest()}"
                                ),
                                "cell_id": cell,
                                "geometry": wkb,
                                "value": value,
                                "failure": {
                                    "code": "source_no_data",
                                    "message": "The source pixel is masked.",
                                }
                                if missing
                                else None,
                            }
                            if time is None:
                                record["label"] = (
                                    None
                                    if value is None
                                    else names.get(str(int(value)))
                                    if categorical
                                    else None
                                )
                            else:
                                record.update(
                                    time=time,
                                    station_id=None,
                                    group_id=None,
                                    coverage=None,
                                )
                            yield record
