"""Native calendar and spatial decoding for readable annual climate evidence."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import groupby
from typing import Any

import numpy as np
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely.geometry import MultiPolygon, box  # type: ignore[import-untyped]

from ._declarations import Calendar, Period
from ._inspection import Bounds
from ._providers.cmip6 import CFCalendar


@dataclass(frozen=True, slots=True)
class GridCell:
    row: int
    column: int
    latitude: float
    longitude: float
    geometry: bytes
    cell_id: int
    source_row: int | None = None
    source_column: int | None = None


@dataclass(frozen=True, slots=True)
class _GridWindow:
    rows: range
    columns: range
    output_indices: tuple[int, ...]
    row_offsets: tuple[int, ...]
    column_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GridReadPlan:
    windows: tuple[_GridWindow, ...]
    cell_count: int
    bytes_per_step: int

    @classmethod
    def from_cells(cls, cells: tuple[GridCell, ...]) -> GridReadPlan:
        by_column: dict[int, list[int]] = {}
        for index, cell in enumerate(cells):
            by_column.setdefault(cell.column, []).append(index)
        windows: list[_GridWindow] = []
        bytes_per_step = 0
        # Adjacent storage columns have the same difference from their position
        # in the sorted selection. A wrap on the native axis creates two runs.
        for _, run in groupby(
            enumerate(sorted(by_column)), key=lambda item: item[1] - item[0]
        ):
            columns = [column for _, column in run]
            output_indices = tuple(
                index for column in columns for index in by_column[column]
            )
            first_row = min(cells[index].row for index in output_indices)
            last_row = max(cells[index].row for index in output_indices)
            first_column = columns[0]
            rows = range(first_row, last_row + 1)
            window_columns = range(first_column, columns[-1] + 1)
            windows.append(
                _GridWindow(
                    rows,
                    window_columns,
                    output_indices,
                    tuple(cells[index].row - first_row for index in output_indices),
                    tuple(
                        cells[index].column - first_column for index in output_indices
                    ),
                )
            )
            bytes_per_step += len(rows) * len(window_columns) * 8
        return cls(tuple(windows), len(cells), bytes_per_step)


def calendar_name(value: object) -> Calendar:
    names = {
        "standard": Calendar.GREGORIAN,
        "gregorian": Calendar.GREGORIAN,
        "proleptic_gregorian": Calendar.PROLEPTIC_GREGORIAN,
        "noleap": Calendar.NO_LEAP,
        "365_day": Calendar.NO_LEAP,
        "all_leap": Calendar.ALL_LEAP,
        "366_day": Calendar.ALL_LEAP,
        "360_day": Calendar.DAYS_360,
        "julian": Calendar.JULIAN,
    }
    # CF explicitly defines its default calendar as 'standard'.
    label = "standard" if value is None else str(value)
    if label not in names:
        raise ValueError(f"unsupported source CF calendar {label!r}")
    return names[label]


def cf_calendar(value: Calendar) -> CFCalendar:
    names = {
        Calendar.GREGORIAN: "proleptic_gregorian",
        Calendar.PROLEPTIC_GREGORIAN: "proleptic_gregorian",
        Calendar.NO_LEAP: "noleap",
        Calendar.ALL_LEAP: "all_leap",
        Calendar.DAYS_360: "360_day",
    }
    if value not in names:
        raise ValueError("this source grid does not support Julian calendar decoding")
    return CFCalendar(names[value])


def _coordinate_bounds(
    group: Any, name: str, *, fallback: float | None
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    axis = group[name]
    values = np.asarray(axis[:], dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or values.size == 0:
        raise ValueError(
            "climate source coordinates must be finite one-dimensional axes"
        )
    bounds_name = axis.attrs.get("bounds")
    if bounds_name is not None and bounds_name in group:
        bounds = np.asarray(group[bounds_name][:], dtype=np.float64)
        if bounds.shape != (values.size, 2) or not np.isfinite(bounds).all():
            raise ValueError(
                "source CF coordinate bounds contradict the coordinate axis"
            )
        return values, bounds
    if values.size < 2:
        if fallback is None:
            raise ValueError("a one-cell climate grid must declare its cell bounds")
        spacing = fallback
    else:
        differences = np.diff(values)
        if np.any(differences == 0) or not np.allclose(
            differences, differences[0], rtol=1e-6, atol=1e-8
        ):
            raise ValueError(
                "an irregular climate grid must declare its CF cell bounds"
            )
        spacing = abs(float(differences[0]))
    # Neighbouring cells share one computed edge, so a point on it is not left
    # in a rounding gap or overlap between their polygons.
    step = spacing if values.size < 2 or values[1] > values[0] else -spacing
    edges = np.concatenate(
        (
            [values[0] - step / 2],
            (values[:-1] + values[1:]) / 2,
            [values[-1] + step / 2],
        )
    )
    lower, upper = edges[:-1], edges[1:]
    bounds = np.column_stack((np.minimum(lower, upper), np.maximum(lower, upper)))
    return values, bounds


def cell_polygon(west: float, south: float, east: float, north: float) -> Any:
    west, east = sorted((west, east))
    south, north = max(-90, min(south, north)), min(90, max(south, north))
    width = east - west
    if width <= 0 or width > 360 or north <= south:
        raise ValueError("source grid cell has invalid WGS84 boundaries")
    # Shift both edges by the same multiple of 360; the subtraction is exact,
    # so cells that share an edge in the source still share it here.
    turns = 360 * math.floor((west + 180) / 360)
    left, right = west - turns, east - turns
    if right <= 180:
        return box(left, south, right, north)
    return MultiPolygon(
        (box(left, south, 180, north), box(-180, south, right - 360, north))
    )


def grid_cells(
    group: Any,
    *,
    lat_name: str,
    lon_name: str,
    area: Bounds,
    fallback_resolution: float | None = None,
) -> tuple[GridCell, ...]:
    from ._climate_io import _parts

    latitude, lat_bounds = _coordinate_bounds(
        group, lat_name, fallback=fallback_resolution
    )
    longitude, lon_bounds = _coordinate_bounds(
        group, lon_name, fallback=fallback_resolution
    )
    parts = _parts(area)
    result = []
    for row, (lat, lat_bound) in enumerate(zip(latitude, lat_bounds, strict=True)):
        if max(lat_bound) < area.south or min(lat_bound) > area.north:
            continue
        for column, (lon, lon_bound) in enumerate(
            zip(longitude, lon_bounds, strict=True)
        ):
            geometry = cell_polygon(
                float(lon_bound[0]),
                float(lat_bound[0]),
                float(lon_bound[1]),
                float(lat_bound[1]),
            )
            if not any(part.intersects(geometry) for part in parts):
                continue
            lon = (float(lon) + 180) % 360 - 180
            if not -90 <= lat <= 90:
                raise ValueError("source grid latitude lies outside WGS84")
            result.append(
                GridCell(
                    row,
                    column,
                    float(lat),
                    lon,
                    geometry.wkb,
                    int(h3.latlng_to_cell(float(lat), lon, 9)),
                    getattr(group, "source_rows", list(range(len(latitude))))[row],
                    getattr(group, "source_columns", list(range(len(longitude))))[
                        column
                    ],
                )
            )
    return tuple(result)


def _attribute_values(attributes: Mapping[str, object], name: str) -> tuple[Any, ...]:
    raw = attributes.get(name)
    return () if raw is None else tuple(np.asarray(raw).reshape(-1).tolist())


def _unpack_grid_values(
    raw: np.ndarray[Any, Any], attributes: Mapping[str, object]
) -> np.ndarray[Any, Any]:
    """Mask CF sentinels on stored values before applying scale and offset."""
    stored = np.asarray(raw)
    invalid = np.zeros(stored.shape, dtype=bool)
    sentinels = _attribute_values(attributes, "_FillValue") + _attribute_values(
        attributes, "missing_value"
    )
    for sentinel in dict.fromkeys(sentinels):
        invalid |= stored == sentinel
    values = stored.astype(np.float64)
    scale = _attribute_values(attributes, "scale_factor")
    offset = _attribute_values(attributes, "add_offset")
    if scale:
        values *= float(scale[0])
    if offset:
        values += float(offset[0])
    values[invalid] = np.nan
    return values


def grid_values(
    array: Any,
    *,
    time_name: str,
    lat_name: str,
    lon_name: str,
    start: int,
    stop: int,
    plan: GridReadPlan,
    memory_limit: int | None = None,
) -> np.ndarray[Any, Any]:
    dimensions = tuple(
        array.attrs.get("_ARRAY_DIMENSIONS", getattr(array, "dimensions", ()))
    )
    wanted = (time_name, lat_name, lon_name)
    if len(dimensions) != 3 or set(dimensions) != set(wanted):
        raise ValueError(
            f"source variable dimensions {dimensions!r} do not match {wanted!r}"
        )
    count = stop - start
    required = count * plan.bytes_per_step
    if memory_limit is not None and required > memory_limit:
        raise MemoryError(
            f"Source grid reduction needs at least {required} bytes; "
            "select a smaller area or increase Runtime(memory_limit=...)."
        )
    result = np.empty((count, plan.cell_count), dtype=np.float64)
    if not count:
        return result
    time_axis, lat_axis, lon_axis = (dimensions.index(name) for name in wanted)
    selection = [slice(None)] * 3
    selection[time_axis] = slice(start, stop)
    for window in plan.windows:
        selection[lat_axis] = slice(window.rows.start, window.rows.stop)
        selection[lon_axis] = slice(window.columns.start, window.columns.stop)
        raw = array[tuple(selection)]
        unpacked = _unpack_grid_values(raw, array.attrs)
        values = np.transpose(unpacked, (time_axis, lat_axis, lon_axis))
        result[:, window.output_indices] = values[
            :, window.row_offsets, window.column_offsets
        ]
    return result


def annual_records(
    *,
    times: np.ndarray[Any, Any],
    values: np.ndarray[Any, Any],
    cells: tuple[GridCell, ...],
    calendar: Calendar,
    selected: Period,
    native: str,
    hourly: bool,
    interval_end: bool,
    support_prefix: str,
    variable_id: str,
    scenario_id: str | None,
    minimum_fraction: float,
) -> Iterator[dict[str, object]]:
    """Reduce complete native days and retain every requested annual outcome."""
    assert selected.start is not None and selected.end is not None
    cal = cf_calendar(calendar)
    precipitation = native in ("rx1day", "rx5day")
    if precipitation and np.any(np.isfinite(values) & (values < 0)):
        raise ValueError("Precipitation totals require nonnegative source intervals")
    days = np.floor(times / (24 if hourly else 1)).astype(np.int64)
    if interval_end:
        days = np.floor((times - 1) / 24).astype(np.int64)
    for year in range(selected.start, selected.end + 1):
        first, stop = cal.to_ordinal(year, 1, 1), cal.to_ordinal(year + 1, 1, 1)
        count = stop - first
        indices = np.flatnonzero((days >= first) & (days < stop))
        daily = np.full((count, len(cells)), np.nan, dtype=np.float64)
        for day in range(first, stop):
            selected_indices = indices[days[indices] == day]
            if selected_indices.size != (24 if hourly else 1):
                continue
            sample = values[selected_indices]
            valid = np.isfinite(sample).all(axis=0)
            if precipitation:
                reduced = sample.sum(axis=0)
            elif native == "tnn":
                reduced = sample.min(axis=0)
            else:
                reduced = sample.max(axis=0)
            daily[day - first] = np.where(valid, reduced, np.nan)
        for column, cell in enumerate(cells):
            evidence = daily[:, column]
            candidates = evidence
            if native == "rx5day":
                windows = np.lib.stride_tricks.sliding_window_view(evidence, 5)
                candidates = np.where(
                    np.isfinite(windows).all(axis=1), windows.sum(axis=1), np.nan
                )
            expected = len(candidates)
            valid_count = int(np.isfinite(candidates).sum())
            usable = valid_count > 0 and valid_count / expected >= minimum_fraction
            value = None
            if usable:
                value = float(
                    np.nanmin(candidates) if native == "tnn" else np.nanmax(candidates)
                )
                if not math.isfinite(value):
                    usable, value = False, None
            source_row = cell.row if cell.source_row is None else cell.source_row
            source_column = (
                cell.column if cell.source_column is None else cell.source_column
            )
            yield {
                "variable_id": variable_id,
                "scenario_id": scenario_id,
                "period_id": selected.id,
                "support_id": (f"{support_prefix}:{source_row}:{source_column}"),
                "cell_id": cell.cell_id,
                "geometry": cell.geometry,
                "time": f"{year:04d}",
                "value": value,
                "failure": None
                if usable
                else {
                    "code": "incomplete_year",
                    "message": (
                        f"{valid_count} of {expected} expected "
                        f"{'five-day totals' if native == 'rx5day' else 'source days'} "
                        "were valid."
                    ),
                },
                "coverage": {
                    "valid": valid_count,
                    "expected": expected,
                    "minimum_fraction": minimum_fraction,
                },
            }
