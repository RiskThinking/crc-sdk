"""Lazy Arrow bridge from chunked Zarr rasters into DuckDB."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Literal, Optional

import numpy as np

from .connection import DuckDBConnection

Bounds = tuple[float, float, float, float]
Point = tuple[float, float]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class RasterMetadata:
    """Metadata needed to expose an OS-Climate raster as rows."""

    hazard_type: str
    indicator_id: str
    scenario: str
    year: int
    units: str
    path: str


@dataclass(frozen=True)
class RasterCurve:
    """One source raster pixel and all values along its leading axis."""

    row: int
    column: int
    boundary: tuple[Point, Point, Point, Point]
    axis_values: np.ndarray[Any, np.dtype[np.float64]]
    values: np.ndarray[Any, np.dtype[np.float64]]


class ZarrRaster:
    """A remote or local Zarr raster with lazy DuckDB scan helpers."""

    def __init__(
        self,
        array: Any,
        metadata: RasterMetadata,
        *,
        connection: Optional[DuckDBConnection] = None,
    ) -> None:
        if len(array.shape) not in (2, 3):
            raise ValueError(
                f"expected a two- or three-dimensional raster, got {array.shape}"
            )
        self.array = array
        self.metadata = metadata
        self.connection = connection or DuckDBConnection()
        self._axis_name, self._axis_values = self._read_axis()
        self._transform = self._read_transform()

    @property
    def axis_name(self) -> str:
        return self._axis_name

    @property
    def axis_values(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._axis_values.copy()

    @property
    def shape(self) -> tuple[int, int, int]:
        if len(self.array.shape) == 2:
            return (1, int(self.array.shape[0]), int(self.array.shape[1]))
        return (
            int(self.array.shape[0]),
            int(self.array.shape[1]),
            int(self.array.shape[2]),
        )

    @property
    def bounds(self) -> Bounds:
        _, height, width = self.shape
        corners = [
            self._pixel_to_world(column, row)
            for column, row in ((0, 0), (width, 0), (0, height), (width, height))
        ]
        longitudes, latitudes = zip(*corners)
        return (
            min(longitudes),
            min(latitudes),
            max(longitudes),
            max(latitudes),
        )

    def scan(
        self,
        bounds: Optional[Bounds] = None,
        *,
        batch_rows: int = 262_144,
    ) -> "ZarrScan":
        """Describe a reusable, out-of-core raster scan."""
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        return ZarrScan(self, bounds or self.bounds, batch_rows)

    def points(
        self,
        coordinates: Sequence[Point],
        *,
        connection: Optional[DuckDBConnection] = None,
    ) -> Any:
        """Read requested lon/lat points and return a DuckDB relation."""
        if not coordinates:
            raise ValueError("at least one coordinate is required")
        columns = self._point_columns(coordinates)
        try:
            import pyarrow as pa  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "Arrow support requires `pip install crc-sdk[connectors]`"
            ) from error
        arrow_columns = {
            **columns,
            "value": pa.array(
                columns["value"],
                mask=~np.isfinite(columns["value"]),
            ),
        }
        table = pa.table(arrow_columns)
        active = (connection or self.connection).connect()
        return self._project_metadata(active.from_arrow(table))

    def point_values(
        self, longitude: float, latitude: float
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]:
        """Materialize only the leading-axis values for one raster cell."""
        columns = self._point_columns([(longitude, latitude)])
        return (
            np.asarray(columns["axis_value"], dtype=np.float64),
            np.asarray(columns["value"], dtype=np.float64),
        )

    def return_period_distribution(
        self,
        longitude: float,
        latitude: float,
        *,
        tail: Literal["upper", "lower"] = "upper",
    ) -> Any:
        """Build a core distribution when the raster axis is return period."""
        if "return period" not in self.axis_name.lower():
            raise ValueError(
                f"{self.metadata.path} has axis {self.axis_name!r}, not return periods"
            )
        from crc_framework import TabulatedDistribution

        periods, values = self.point_values(longitude, latitude)
        valid = np.isfinite(periods) & np.isfinite(values)
        periods, values = periods[valid], values[valid]
        if len(periods) < 2:
            raise ValueError(
                f"{self.metadata.path} has fewer than two finite return levels "
                "at the requested point"
            )
        return TabulatedDistribution.from_return_periods(
            periods,
            values,
            tail=tail,
        )

    def pixel_boundary(
        self, row: int, column: int
    ) -> tuple[Point, Point, Point, Point]:
        """Return a source pixel boundary in counter-clockwise WGS84 order."""
        _, height, width = self.shape
        if not 0 <= row < height or not 0 <= column < width:
            raise ValueError("pixel coordinates fall outside the raster")
        return (
            self._pixel_to_world(column, row),
            self._pixel_to_world(column, row + 1),
            self._pixel_to_world(column + 1, row + 1),
            self._pixel_to_world(column + 1, row),
        )

    def iter_curves(self, bounds: Optional[Bounds] = None) -> Iterator[RasterCurve]:
        """Stream source pixels with complete leading-axis curves."""
        column_start, column_stop, row_start, row_stop = self._pixel_window(
            bounds or self.bounds
        )
        chunk_shape = getattr(self.array, "chunks", self.shape)
        chunk_height = int(chunk_shape[-2])
        chunk_width = int(chunk_shape[-1])
        for row in range(row_start, row_stop, chunk_height):
            row_end = min(row + chunk_height, row_stop)
            for column in range(column_start, column_stop, chunk_width):
                column_end = min(column + chunk_width, column_stop)
                if len(self.array.shape) == 2:
                    values = np.asarray(
                        self.array[row:row_end, column:column_end],
                        dtype=np.float64,
                    )[np.newaxis, :, :]
                else:
                    values = np.asarray(
                        self.array[:, row:row_end, column:column_end],
                        dtype=np.float64,
                    )
                for source_row in range(row, row_end):
                    for source_column in range(column, column_end):
                        yield RasterCurve(
                            row=source_row,
                            column=source_column,
                            boundary=self.pixel_boundary(source_row, source_column),
                            axis_values=self._axis_values.copy(),
                            values=values[
                                :,
                                source_row - row,
                                source_column - column,
                            ].copy(),
                        )

    def _read_axis(
        self,
    ) -> tuple[str, np.ndarray[Any, np.dtype[np.float64]]]:
        attrs = dict(self.array.attrs)
        axis_count = 1 if len(self.array.shape) == 2 else int(self.array.shape[0])
        dimensions = list(
            attrs.get("dimensions") or attrs.get("_ARRAY_DIMENSIONS") or ()
        )
        dimension = (
            str(dimensions[0]) if len(dimensions) == len(self.array.shape) else ""
        )
        axis_name = str(attrs.get("index_name") or dimension or "index")
        values = attrs.get("index_values")
        if values is None and dimension:
            values = attrs.get(f"{dimension}_values")
        if values is None:
            values = range(axis_count)
        result = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(result) != axis_count:
            raise ValueError(
                f"axis metadata has {len(result)} values for {axis_count} slices"
            )
        return axis_name, result

    def _read_transform(self) -> tuple[float, float, float, float, float, float]:
        values = self.array.attrs.get("transform_mat3x3")
        if values is None or len(values) < 6:
            raise ValueError("Zarr raster is missing transform_mat3x3")
        return (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5]),
        )

    def _pixel_to_world(self, column: float, row: float) -> Point:
        a, b, c, d, e, f = self._transform
        return (a * column + b * row + c, d * column + e * row + f)

    def _world_to_pixel(self, longitude: float, latitude: float) -> Point:
        a, b, c, d, e, f = self._transform
        determinant = a * e - b * d
        if determinant == 0.0:
            raise ValueError("raster transform is not invertible")
        x, y = longitude - c, latitude - f
        return ((e * x - b * y) / determinant, (-d * x + a * y) / determinant)

    def _pixel_window(self, bounds: Bounds) -> tuple[int, int, int, int]:
        minimum_x, minimum_y, maximum_x, maximum_y = bounds
        if minimum_x > maximum_x or minimum_y > maximum_y:
            raise ValueError("bounds must be (min_lon, min_lat, max_lon, max_lat)")
        pixels = [
            self._world_to_pixel(longitude, latitude)
            for longitude, latitude in (
                (minimum_x, minimum_y),
                (minimum_x, maximum_y),
                (maximum_x, minimum_y),
                (maximum_x, maximum_y),
            )
        ]
        columns, rows = zip(*pixels)
        _, height, width = self.shape
        column_start = max(0, floor(min(columns)))
        column_stop = min(width, ceil(max(columns)))
        row_start = max(0, floor(min(rows)))
        row_stop = min(height, ceil(max(rows)))
        if column_start >= column_stop or row_start >= row_stop:
            raise ValueError("bounds do not intersect the raster")
        return column_start, column_stop, row_start, row_stop

    def _point_columns(self, coordinates: Sequence[Point]) -> dict[str, Any]:
        axis_count, height, width = self.shape
        pixel_coordinates = [
            self._world_to_pixel(longitude, latitude)
            for longitude, latitude in coordinates
        ]
        columns = np.asarray([floor(value[0]) for value in pixel_coordinates])
        rows = np.asarray([floor(value[1]) for value in pixel_coordinates])
        if (
            np.any(columns < 0)
            or np.any(columns >= width)
            or np.any(rows < 0)
            or np.any(rows >= height)
        ):
            raise ValueError("one or more coordinates fall outside the raster")

        point_count = len(coordinates)
        axis_indices = np.tile(np.arange(axis_count, dtype=np.int32), point_count)
        repeated_rows = np.repeat(rows, axis_count)
        repeated_columns = np.repeat(columns, axis_count)
        if len(self.array.shape) == 2:
            values = self.array.get_coordinate_selection((rows, columns))
        else:
            values = self.array.get_coordinate_selection(
                (axis_indices, repeated_rows, repeated_columns)
            )
        centers = [
            self._pixel_to_world(column + 0.5, row + 0.5)
            for column, row in zip(columns, rows)
        ]
        return {
            "longitude": np.repeat(
                np.asarray([point[0] for point in centers], dtype=np.float64),
                axis_count,
            ),
            "latitude": np.repeat(
                np.asarray([point[1] for point in centers], dtype=np.float64),
                axis_count,
            ),
            "axis_index": axis_indices,
            "axis_value": np.tile(self._axis_values, point_count),
            "value": np.asarray(values, dtype=np.float64).reshape(-1),
        }

    def _project_metadata(self, relation: Any) -> Any:
        values = self.metadata
        expression = ", ".join(
            [
                "*",
                f"{_sql_string(values.hazard_type)} AS hazard_type",
                f"{_sql_string(values.indicator_id)} AS indicator_id",
                f"{_sql_string(values.scenario)} AS scenario",
                f"{values.year}::INTEGER AS year",
                f"{_sql_string(values.units)} AS units",
                f"{_sql_string(self.axis_name)} AS axis_name",
                f"{_sql_string(values.path)} AS source_path",
            ]
        )
        return relation.project(expression)


@dataclass(frozen=True)
class ZarrScan:
    """Reusable description of a one-pass Zarr-to-DuckDB scan."""

    raster: ZarrRaster
    bounds: Bounds
    batch_rows: int

    def relation(self, *, connection: Optional[DuckDBConnection] = None) -> Any:
        """Create a fresh lazy DuckDB relation for one query execution."""
        try:
            import pyarrow as pa
        except ImportError as error:
            raise ImportError(
                "Arrow support requires `pip install crc-sdk[connectors]`"
            ) from error

        schema = pa.schema(
            [
                ("longitude", pa.float64()),
                ("latitude", pa.float64()),
                ("axis_index", pa.int32()),
                ("axis_value", pa.float64()),
                ("value", pa.float64()),
            ]
        )
        reader = pa.RecordBatchReader.from_batches(schema, self._batches(pa))
        active = (connection or self.raster.connection).connect()
        return self.raster._project_metadata(active.from_arrow(reader))

    def _batches(self, pa: Any) -> Iterator[Any]:
        raster = self.raster
        axis_count, _, _ = raster.shape
        column_start, column_stop, row_start, row_stop = raster._pixel_window(
            self.bounds
        )
        chunk_shape = getattr(raster.array, "chunks", raster.shape)
        chunk_height = int(chunk_shape[-2])
        chunk_width = int(chunk_shape[-1])
        target_pixels = max(1, self.batch_rows // axis_count)

        tile_width = min(chunk_width, column_stop - column_start, target_pixels)
        tile_height = min(
            chunk_height,
            row_stop - row_start,
            max(1, target_pixels // tile_width),
        )

        for row in range(row_start, row_stop, tile_height):
            row_end = min(row + tile_height, row_stop)
            for column in range(column_start, column_stop, tile_width):
                column_end = min(column + tile_width, column_stop)
                if len(raster.array.shape) == 2:
                    values = np.asarray(
                        raster.array[row:row_end, column:column_end],
                        dtype=np.float64,
                    )[np.newaxis, :, :]
                else:
                    values = np.asarray(
                        raster.array[:, row:row_end, column:column_end],
                        dtype=np.float64,
                    )
                yield self._record_batch(
                    pa,
                    values,
                    row,
                    row_end,
                    column,
                    column_end,
                )

    def _record_batch(
        self,
        pa: Any,
        values: np.ndarray[Any, np.dtype[np.float64]],
        row_start: int,
        row_stop: int,
        column_start: int,
        column_stop: int,
    ) -> Any:
        raster = self.raster
        axis_count = values.shape[0]
        rows, columns = np.meshgrid(
            np.arange(row_start, row_stop, dtype=np.float64) + 0.5,
            np.arange(column_start, column_stop, dtype=np.float64) + 0.5,
            indexing="ij",
        )
        a, b, c, d, e, f = raster._transform
        longitude = a * columns + b * rows + c
        latitude = d * columns + e * rows + f
        pixel_count = longitude.size
        return pa.record_batch(
            [
                pa.array(np.tile(longitude.reshape(-1), axis_count)),
                pa.array(np.tile(latitude.reshape(-1), axis_count)),
                pa.array(np.repeat(np.arange(axis_count, dtype=np.int32), pixel_count)),
                pa.array(np.repeat(raster._axis_values, pixel_count)),
                pa.array(
                    values.reshape(-1),
                    mask=~np.isfinite(values.reshape(-1)),
                ),
            ],
            names=[
                "longitude",
                "latitude",
                "axis_index",
                "axis_value",
                "value",
            ],
        )
