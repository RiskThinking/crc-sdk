"""Exact scientific selection and recorded identity/value transformations."""

from __future__ import annotations

import json
import math
import re
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

import crc_framework
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import shapely  # type: ignore[import-untyped]
from crc_framework import _core
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely import from_wkb
from shapely.affinity import translate  # type: ignore[import-untyped]
from shapely.geometry import box  # type: ignore[import-untyped]

from ._aggregation import (
    Sampling,
    TemporalReduction,
    TimestampPosition,
    _date,
    _position,
)
from ._dataset_schema import (
    IdentityKind,
    IdentityRename,
    ObservationTimeSelection,
    SourceIdentity,
    Transformation,
    UnitConversion,
    _identifier,
)
from ._declarations import (
    Calendar,
    DataSpec,
    Period,
    PeriodKind,
    Scenario,
    SeriesSpec,
    Variable,
    scientific_identity,
)
from ._evidence import EvidenceMetadata
from ._inspection import Bounds, SeriesReference, StationReference
from ._schema.dataset import DatasetRow

if TYPE_CHECKING:
    from ._dataset_schema import DatasetMetadata

MetadataT = TypeVar("MetadataT", bound="DatasetMetadata | EvidenceMetadata")

VariableSelector: TypeAlias = Variable | str | Sequence[Variable | str] | None
ScenarioSelector: TypeAlias = Scenario | str | Sequence[Scenario | str] | None
PeriodSelector: TypeAlias = Period | Sequence[Period] | None
SeriesSelector: TypeAlias = SeriesReference | Sequence[SeriesReference] | None
IdentifierSelector: TypeAlias = str | Sequence[str] | None
StationSelector: TypeAlias = (
    StationReference | str | Sequence[StationReference | str] | None
)
Unit: TypeAlias = (
    crc_framework.LengthUnit
    | crc_framework.TimeUnit
    | crc_framework.RatioUnit
    | crc_framework.TemperatureUnit
    | crc_framework.SpeedUnit
    | crc_framework.PrecipitationRateUnit
    | crc_framework.DegreeDayUnit
)

SeriesKey: TypeAlias = tuple[str, str | None, str | None]
_SelectionValue = TypeVar(
    "_SelectionValue", Variable, Scenario, Period, SeriesReference
)


def _series_key(value: SeriesSpec | SeriesReference) -> SeriesKey:
    return (
        value.variable.id,
        None if value.scenario is None else value.scenario.id,
        None if value.period is None else value.period.id,
    )


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError("a selector requires one reference or a sequence of references")
    return tuple(value)


def _identities(
    value: object, expected: type[_SelectionValue], *, strings: bool
) -> tuple[str | _SelectionValue, ...] | None:
    if value is None:
        return None
    values: tuple[object, ...]
    if isinstance(value, expected) or (strings and isinstance(value, str)):
        values = (value,)
    else:
        values = _sequence(value)
    result: list[str | _SelectionValue] = []
    for item in values:
        if strings and isinstance(item, str):
            # A public provider enum may subclass str; its actual exact ID is its value.
            identifier = item.value if isinstance(item, Enum) else item
            result.append(_identifier(identifier, "selection id"))
        elif isinstance(item, expected):
            result.append(item)
        else:
            allowed = f"{expected.__name__} or str" if strings else expected.__name__
            raise TypeError(f"selector entries must be {allowed}")
    return tuple(result)


def _stations(value: StationSelector) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = (
        (value,) if isinstance(value, (str, StationReference)) else _sequence(value)
    )
    return tuple(
        _identifier(
            item.id if isinstance(item, StationReference) else item, "station id"
        )
        for item in values
    )


def _reference_document(
    value: str | Variable | Scenario | Period | SeriesReference,
) -> object:
    if isinstance(value, str):
        return {"id": value}
    if isinstance(value, SeriesReference):
        result: dict[str, object] = {
            "variable": _reference_document(value.variable),
            "scenario": None
            if value.scenario is None
            else _reference_document(value.scenario),
            "period": None
            if value.period is None
            else _reference_document(value.period),
        }
        if value.release is not None:
            result["release"] = {
                "id": value.release.id,
                "sha256": value.release.sha256,
            }
        if value.probability is not None:
            result["probability"] = value.probability._document()
        if value.model is not None:
            result["model"] = value.model
        if value.member is not None:
            result["member"] = value.member
        return result
    return {"id": value.id, "science": scientific_identity(value)}


def _selection_document(
    values: Sequence[str | Variable | Scenario | Period | SeriesReference] | None,
) -> object:
    if values is None:
        return None
    documents = [_reference_document(value) for value in values]
    return sorted(documents, key=lambda value: json.dumps(value, sort_keys=True))


@dataclass(frozen=True, slots=True)
class SelectionState:
    variables: tuple[str | Variable, ...] | None = None
    scenarios: tuple[str | Scenario, ...] | None = None
    periods: tuple[Period, ...] | None = None
    series: tuple[SeriesReference, ...] | None = None
    area: Bounds | None = None
    stations: tuple[str, ...] | None = None
    time: Period | None = None
    # Internal restriction to the H3 cells that requested assets can match.
    # It removes only rows that cannot match, so it is not a scientific
    # selection and never appears in metadata or provenance.
    cells: frozenset[int] | None = None

    def updated(
        self,
        *,
        variables: VariableSelector | types.EllipsisType = ...,
        scenarios: ScenarioSelector | types.EllipsisType = ...,
        periods: PeriodSelector | types.EllipsisType = ...,
        series: SeriesSelector | types.EllipsisType = ...,
        area: Bounds | None | types.EllipsisType = ...,
        stations: StationSelector | types.EllipsisType = ...,
        time: Period | None | types.EllipsisType = ...,
    ) -> SelectionState:
        if area is not ... and area is not None and not isinstance(area, Bounds):
            raise TypeError("area must be Bounds or None")
        if (
            time is not ...
            and time is not None
            and (not isinstance(time, Period) or time.kind is not PeriodKind.WINDOW)
        ):
            raise TypeError("observation time requires a Period.window or None")
        return SelectionState(
            variables=self.variables
            if variables is ...
            else _identities(variables, Variable, strings=True),
            scenarios=self.scenarios
            if scenarios is ...
            else _identities(scenarios, Scenario, strings=True),
            periods=self.periods
            if periods is ...
            else cast(
                tuple[Period, ...] | None, _identities(periods, Period, strings=False)
            ),
            series=self.series
            if series is ...
            else cast(
                tuple[SeriesReference, ...] | None,
                _identities(series, SeriesReference, strings=False),
            ),
            area=self.area if area is ... else area,
            stations=self.stations if stations is ... else _stations(stations),
            time=self.time if time is ... else time,
            cells=self.cells,
        )

    def restricted(self, cells: frozenset[int]) -> SelectionState:
        """Also keep only cell-supported rows in `cells`; geometry rows are kept."""
        return replace(self, cells=cells if self.cells is None else self.cells & cells)

    def document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "variables": _selection_document(self.variables),
            "scenarios": _selection_document(self.scenarios),
            "periods": _selection_document(self.periods),
            "series": _selection_document(self.series),
            "area": None
            if self.area is None
            else {
                "west": self.area.west,
                "south": self.area.south,
                "east": self.area.east,
                "north": self.area.north,
            },
            "stations": None if self.stations is None else sorted(set(self.stations)),
            "time": None if self.time is None else self.time._document(),
        }
        if self.cells is not None:
            document["cells"] = sorted(self.cells)
        return document

    def resolve(
        self, spec: DataSpec, source: SourceIdentity | None = None
    ) -> ResolvedSelection:
        if not isinstance(spec, DataSpec):
            raise TypeError("selection requires DataSpec")
        variables = _resolve_ids(self.variables, spec.variables)
        scenarios = _resolve_ids(self.scenarios, spec.scenarios)
        periods = _resolve_ids(self.periods, spec.periods)
        series_keys: set[SeriesKey] | None = None
        if self.series is not None:
            available = {_series_key(item): item for item in spec.series}
            series_keys = set()
            for reference in self.series:
                key = _series_key(reference)
                item = available.get(key)
                if item is None:
                    raise ValueError(f"unknown series identity {key!r}")
                for actual, requested in (
                    (item.variable, reference.variable),
                    (item.scenario, reference.scenario),
                    (item.period, reference.period),
                    (item.probability, reference.probability),
                ):
                    if actual is not None and requested is not None:
                        if scientific_identity(actual) != scientific_identity(
                            requested
                        ):
                            raise ValueError(
                                f"conflicting scientific series definition {key!r}"
                            )
                if reference.release is not None and (
                    source is None
                    or source.release is None
                    or (source.release.id, source.release.sha256)
                    != (reference.release.id, reference.release.sha256)
                ):
                    raise ValueError(
                        "selected series release differs from the source snapshot"
                    )
                for name in ("model", "member"):
                    requested = getattr(reference, name)
                    if requested is not None and (
                        source is None or requested != getattr(source, name)
                    ):
                        raise ValueError(
                            f"selected series {name} differs from the source snapshot"
                        )
                series_keys.add(key)
        selected = tuple(
            item
            for item in spec.series
            if (variables is None or item.variable.id in variables)
            and (
                scenarios is None
                or (item.scenario is not None and item.scenario.id in scenarios)
            )
            and (
                periods is None
                or (item.period is not None and item.period.id in periods)
            )
            and (series_keys is None or _series_key(item) in series_keys)
        )
        time = self.time
        if time is not None:
            time = _observation_time(
                time,
                {
                    item.period.calendar
                    for item in selected
                    if item.period is not None and item.period.calendar is not None
                },
            )
        selected_spec = participating_spec(spec, selected)
        return ResolvedSelection(
            spec=selected_spec,
            keys=frozenset(_series_key(item) for item in selected),
            area=self.area,
            stations=self.stations,
            time=time,
            cells=self.cells,
        )


def _validate_static_selection(
    state: SelectionState, variables: Sequence[Variable]
) -> None:
    """Validate known variable and series science before catalogue acquisition."""
    spec = DataSpec(variables, series=[SeriesSpec(v) for v in variables])
    # The catalogue establishes the actual coverage periods and release.
    series = (
        None
        if state.series is None
        else tuple(replace(ref, period=None, release=None) for ref in state.series)
    )
    SelectionState(
        variables=state.variables, scenarios=state.scenarios, series=series
    ).resolve(spec)


def _resolve_ids(
    requested: Sequence[str | Variable | Scenario | Period] | None,
    descriptors: Sequence[Variable | Scenario | Period],
) -> frozenset[str] | None:
    if requested is None:
        return None
    available = {item.id: item for item in descriptors}
    selected: set[str] = set()
    for item in requested:
        identifier = item if isinstance(item, str) else item.id
        actual = available.get(identifier)
        if actual is None:
            raise ValueError(f"unknown requested identity {identifier!r}")
        if not isinstance(item, str) and scientific_identity(
            actual
        ) != scientific_identity(item):
            raise ValueError(f"conflicting scientific definition for {identifier!r}")
        selected.add(identifier)
    return frozenset(selected)


def participating_spec(spec: DataSpec, series: Sequence[SeriesSpec]) -> DataSpec:
    selected = tuple(series)
    variables = {item.variable.id for item in selected}
    scenarios = {item.scenario.id for item in selected if item.scenario is not None}
    periods = {item.period.id for item in selected if item.period is not None}
    return DataSpec(
        [item for item in spec.variables if item.id in variables],
        series=selected,
        scenarios=[item for item in spec.scenarios if item.id in scenarios],
        periods=[item for item in spec.periods if item.id in periods],
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedSelection:
    spec: DataSpec
    keys: frozenset[SeriesKey]
    area: Bounds | None = None
    stations: tuple[str, ...] | None = None
    time: Period | None = None
    cells: frozenset[int] | None = None

    @property
    def empty(self) -> bool:
        return not self.keys or self.stations == ()

    def matches(self, row: DatasetRow) -> bool:
        return self.matches_record(
            {
                "variable_id": row.variable_id,
                "scenario_id": row.scenario_id,
                "period_id": row.period_id,
                "support_id": row.support_id,
                "cell_id": row.cell_id,
                "geometry": row.geometry,
            }
        )

    def record_mask(
        self, batch: pa.RecordBatch, cache: dict[object, bool] | None = None
    ) -> pa.BooleanArray:
        """`matches_record` for every row of a Dataset batch, without per-row Python
        for the series keys and stations. The exact support/area test runs once per
        distinct cell or geometry; pass the same `cache` across a dataset's batches.
        """
        if self.time is not None:
            return pa.array(
                [self.matches_record(row) for row in batch.to_pylist()],
                type=pa.bool_(),
            )
        mask = _series_mask(batch, self.keys)
        if self.stations is not None:
            stations = pa.array(self.stations, type=pa.string())
            mask = pc.and_(mask, pc.is_in(batch.column("support_id"), stations))
        if self.cells is not None:
            cell_ids = batch.column("cell_id")
            outside = pc.and_(
                pc.and_(pc.is_null(batch.column("geometry")), pc.is_valid(cell_ids)),
                pc.invert(
                    pc.is_in(
                        pc.cast(cell_ids, pa.uint64()),
                        pa.array(sorted(self.cells), type=pa.uint64()),
                    )
                ),
            )
            mask = pc.and_not(mask, outside)
        if self.area is None or not pc.any(mask).as_py():
            return mask
        cache = {} if cache is None else cache
        area = self.area
        cell_rows = pc.and_(
            pc.is_null(batch.column("geometry")), pc.is_valid(batch.column("cell_id"))
        )
        mask = pc.and_not(
            mask, _outside_ancestors(area, batch, pc.and_(mask, cell_rows), cache)
        )
        if not pc.any(mask).as_py():
            return mask
        geometries = batch.column("geometry").to_pylist()
        cells = batch.column("cell_id").to_pylist()
        selected = mask.to_pylist()
        for index, keep in enumerate(selected):
            if not keep:
                continue
            geometry, cell = geometries[index], cells[index]
            token = ("g", geometry) if geometry is not None else ("c", cell)
            if token not in cache:
                cache[token] = support_intersects(area, geometry=geometry, cell_id=cell)
            selected[index] = cache[token]
        return pa.array(selected, type=pa.bool_())

    def matches_record(
        self, row: Mapping[str, object], *, sampling: Sampling | None = None
    ) -> bool:
        key = (row.get("variable_id"), row.get("scenario_id"), row.get("period_id"))
        if key not in self.keys:
            return False
        if (
            self.stations is not None
            and (row.get("station_id") or row.get("support_id")) not in self.stations
        ):
            return False
        if (
            self.cells is not None
            and row.get("geometry") is None
            and row.get("cell_id") is not None
            and int(cast(int, row["cell_id"])) not in self.cells
        ):
            return False
        if self.time is not None:
            time = row.get("time")
            coordinate = row.get("coordinate")
            if isinstance(coordinate, Mapping) and coordinate.get("kind") == "time":
                time = coordinate.get("time")
            if sampling is not None and sampling.timestamp is TimestampPosition.END:
                ordinal, _ = _position(str(time), sampling)
                time = _date(ordinal, sampling.calendar)[0]
            if not _in_time(time, self.time):
                return False
        return self.area is None or support_intersects(
            self.area,
            geometry=cast(bytes | None, row.get("geometry")),
            cell_id=cast(int | None, row.get("cell_id")),
        )


def _series_mask(batch: pa.RecordBatch, keys: frozenset[SeriesKey]) -> pa.BooleanArray:
    """Rows whose `(variable_id, scenario_id, period_id)` is one of `keys`.

    Each column maps to integer codes: the position of its value among the
    selected values, one code for null and one for any other value. One
    membership test of the combined codes then decides every row, however
    many keys are selected.
    """
    if not keys:
        return pa.array([False] * batch.num_rows, type=pa.bool_())
    combined: pa.Array | None = None
    key_codes = [0] * len(keys)
    ordered = list(keys)
    for position, name in enumerate(("variable_id", "scenario_id", "period_id")):
        column = batch.column(name)
        values = sorted(
            {value for key in ordered if (value := key[position]) is not None}
        )
        null, other = len(values), len(values) + 1
        found = pc.index_in(column, value_set=pa.array(values, type=column.type))
        codes = pc.if_else(
            pc.is_null(column),
            pa.scalar(null, pa.int64()),
            pc.fill_null(pc.cast(found, pa.int64()), other),
        )
        index = {value: code for code, value in enumerate(values)}
        for i, key in enumerate(ordered):
            value = key[position]
            key_codes[i] = key_codes[i] * (other + 1) + (
                null if value is None else index[value]
            )
        combined = (
            codes
            if combined is None
            else pc.add(pc.multiply(combined, other + 1), codes)
        )
    return pc.is_in(combined, value_set=pa.array(key_codes, type=pa.int64()))


def _observation_time(time: Period, calendars: set[Calendar]) -> Period:
    if time.calendar is not None:
        if calendars - {time.calendar}:
            raise ValueError(
                "observation time calendar contradicts the source calendar"
            )
    elif len(calendars) == 1:
        assert time.start is not None and time.end is not None
        return Period.window(time.start, time.end, calendar=next(iter(calendars)))
    return time


def _in_time(value: object, period: Period) -> bool:
    if isinstance(value, (datetime, date)):
        year = value.year
    elif type(value) is int:
        year = value
    elif isinstance(value, str) and (
        match := re.match(r"^([+-]?\d{4,})(?:-|$)", value)
    ):
        year = int(match[1])
    else:
        raise ValueError(
            "observation time filtering requires a calendar year or timestamp"
        )
    assert period.start is not None and period.end is not None
    return period.start <= year <= period.end


_H3_RESOLUTION_SHIFT = 52
_H3_RESOLUTION_MASK = 0xF << _H3_RESOLUTION_SHIFT
# Ancestors this many resolutions coarser bound a cell's descendants within
# their one-ring neighbourhood (checked for every resolution 0 and 2 cell).
_ANCESTOR_LEVELS = 3


def _ancestor_cells(cells: np.ndarray, resolution: int) -> np.ndarray:
    """H3 ancestors at `resolution`, by rewriting the resolution and digit bits."""
    unused_digits = np.uint64((1 << (3 * (15 - resolution))) - 1)
    return (
        (cells & np.uint64(~_H3_RESOLUTION_MASK & (2**64 - 1)))
        | np.uint64(resolution << _H3_RESOLUTION_SHIFT)
        | unused_digits
    )


def _outside_ancestors(
    area: Bounds,
    batch: pa.RecordBatch,
    candidates: pa.BooleanArray,
    cache: dict[object, bool],
) -> pa.BooleanArray:
    """Mark cell-supported candidate rows whose coarse ancestor's one-ring cannot
    reach `area`, so the exact cell test runs only near the area. A cell lies
    within the one-ring of its ancestor three resolutions coarser, so this never
    excludes a cell that intersects the area."""
    outside = np.zeros(batch.num_rows, dtype=bool)
    rows = np.flatnonzero(candidates.to_numpy(zero_copy_only=False))
    if not rows.size:
        return pa.array(outside)
    cells = (
        pc.cast(batch.column("cell_id"), pa.uint64())
        .take(pa.array(rows))
        .to_numpy(zero_copy_only=False)
        .astype(np.uint64)
    )
    resolutions = (cells >> np.uint64(_H3_RESOLUTION_SHIFT)) & np.uint64(0xF)
    for resolution in np.unique(resolutions):
        levels = sorted({0, max(0, int(resolution) - _ANCESTOR_LEVELS)})
        selected = resolutions == resolution
        for level in levels:
            if level >= resolution:
                continue
            ancestors = _ancestor_cells(cells, level)
            reachable = []
            for ancestor in np.unique(ancestors[selected]).tolist():
                token = ("ring", ancestor)
                if token not in cache:
                    cache[token] = any(
                        _cell_intersects(area, cell)
                        for cell in h3.grid_disk(ancestor, 1)
                    )
                if cache[token]:
                    reachable.append(ancestor)
            selected &= np.isin(ancestors, np.array(reachable, dtype=np.uint64))
        outside[rows[(resolutions == resolution) & ~selected]] = True
    return pa.array(outside)


# Margin, in degrees, for floating-point error in the cell extent.
_EXTENT_MARGIN = 1e-6


def _cell_box(cell: int) -> tuple[float, float, float, float]:
    """South, north, west, east of an H3 cell on the sphere. A cell crossing the
    antimeridian or containing a pole spans every longitude."""
    resolution = h3.get_resolution(cell)
    for pole in (90.0, -90.0):
        if h3.latlng_to_cell(pole, 0.0, resolution) == cell:
            latitudes = [latitude for latitude, _ in h3.cell_to_boundary(cell)]
            return (
                (min(latitudes) - _EXTENT_MARGIN, 90.0, -180.0, 180.0)
                if pole > 0
                else (-90.0, max(latitudes) + _EXTENT_MARGIN, -180.0, 180.0)
            )
    vertices = h3.cell_to_boundary(cell)
    latitudes = [latitude for latitude, _ in vertices]
    longitudes = [longitude for _, longitude in vertices]
    boundary = [_vector(latitude, longitude) for latitude, longitude in vertices]
    for first, last in zip(boundary, [*boundary[1:], boundary[0]], strict=True):
        # An edge's latitude peaks where its great circle is closest to a pole.
        normal = _cross(first, last)
        horizontal = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if horizontal <= 1e-15:
            continue
        scale = math.sqrt(_dot(normal, normal))
        for sign in (1, -1):
            extreme = cast(
                _Vector,
                tuple(
                    sign * value
                    for value in (
                        -normal[0] * normal[2] / (scale * horizontal),
                        -normal[1] * normal[2] / (scale * horizontal),
                        horizontal / scale,
                    )
                ),
            )
            if _on_arc(extreme, first, last):
                latitudes.append(
                    math.degrees(math.asin(max(-1.0, min(1.0, extreme[2]))))
                )
    south = max(-90.0, min(latitudes) - _EXTENT_MARGIN)
    north = min(90.0, max(latitudes) + _EXTENT_MARGIN)
    if max(longitudes) - min(longitudes) > 180:
        return south, north, -180.0, 180.0
    margin = _EXTENT_MARGIN / max(
        math.cos(math.radians(max(abs(south), abs(north)))), 1e-6
    )
    return (
        south,
        north,
        max(-180.0, min(longitudes) - margin),
        min(180.0, max(longitudes) + margin),
    )


def cell_bounds(cell: int) -> Bounds:
    """Conservative bounds of one H3 cell."""
    south, north, west, east = _cell_box(cell)
    return Bounds(west=west, south=south, east=east, north=north)


def support_extent(batches: Iterable[pa.RecordBatch]) -> Bounds | None:
    """Conservative bounds of every row support in Dataset batches.

    Cell rows are bounded by the one-ring of their ancestor three resolutions
    coarser (or by the cell itself at resolution 3 and coarser), so the result
    contains each cell without bounding every cell. Geometry rows use their
    planar bounds. Returns None when there are no rows.
    """
    boxes: list[tuple[float, float, float, float]] = []
    regions: set[int] = set()
    for batch in batches:
        geometry = batch.column("geometry")
        cells = batch.column("cell_id")
        cell_rows = pc.and_(pc.is_null(geometry), pc.is_valid(cells))
        values = (
            pc.cast(pc.filter(cells, cell_rows), pa.uint64())
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64)
        )
        if values.size:
            values = np.unique(values)
            resolutions = (values >> np.uint64(_H3_RESOLUTION_SHIFT)) & np.uint64(0xF)
            for resolution in np.unique(resolutions).tolist():
                selected = values[resolutions == resolution]
                if resolution <= _ANCESTOR_LEVELS:
                    regions.update(selected.tolist())
                    continue
                for ancestor in np.unique(
                    _ancestor_cells(selected, resolution - _ANCESTOR_LEVELS)
                ).tolist():
                    regions.update(h3.grid_disk(ancestor, 1))
        shapes = pc.filter(geometry, pc.is_valid(geometry)).to_numpy(
            zero_copy_only=False
        )
        if shapes.size:
            west, south, east, north = shapely.total_bounds(shapely.from_wkb(shapes))
            if west < -180 or east > 180:
                west, east = -180.0, 180.0
            boxes.append((south, north, west, east))
        missing = pc.and_(pc.is_null(geometry), pc.is_null(cells))
        if pc.any(missing).as_py():
            raise ValueError(
                "support extent requires actual support geometry or an H3 cell"
            )
    boxes.extend(map(_cell_box, sorted(regions)))
    if not boxes:
        return None
    return Bounds(
        west=min(item[2] for item in boxes),
        south=min(item[0] for item in boxes),
        east=max(item[3] for item in boxes),
        north=max(item[1] for item in boxes),
    )


def bounds_intersect(first: Bounds, second: Bounds) -> bool:
    """Inclusive intersection of two bounds, either of which may cross the
    antimeridian."""
    if first.south > second.north or second.south > first.north:
        return False

    def spans(bounds: Bounds) -> tuple[tuple[float, float], ...]:
        if bounds.west <= bounds.east:
            return ((bounds.west, bounds.east),)
        return ((bounds.west, 180.0), (-180.0, bounds.east))

    return any(
        a_west <= b_east and b_west <= a_east
        for a_west, a_east in spans(first)
        for b_west, b_east in spans(second)
    )


def support_intersects(
    area: Bounds, *, geometry: bytes | None, cell_id: int | None
) -> bool:
    """Select actual source support, using cell area only for cell-supported rows."""
    if geometry is not None:
        support = from_wkb(geometry)
    elif cell_id is not None:
        return _cell_intersects(area, cell_id)
    else:
        raise ValueError(
            "area selection requires actual support geometry or an H3 cell"
        )
    bounds = (
        (box(area.west, area.south, area.east, area.north),)
        if area.west <= area.east
        else (
            box(area.west, area.south, 180, area.north),
            box(-180, area.south, area.east, area.north),
        )
    )
    return any(
        translate(support, xoff=offset).intersects(bound)
        for offset in (-360, 0, 360)
        for bound in bounds
    )


_Vector: TypeAlias = tuple[float, float, float]


def _cross(a: _Vector, b: _Vector) -> _Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: _Vector, b: _Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _vector(latitude: float, longitude: float) -> _Vector:
    latitude, longitude = math.radians(latitude), math.radians(longitude)
    return (
        math.cos(latitude) * math.cos(longitude),
        math.cos(latitude) * math.sin(longitude),
        math.sin(latitude),
    )


def _angle(a: _Vector, b: _Vector) -> float:
    cross = _cross(a, b)
    return math.atan2(math.sqrt(_dot(cross, cross)), _dot(a, b))


def _on_arc(point: _Vector, first: _Vector, last: _Vector) -> bool:
    return (
        abs(_angle(first, point) + _angle(point, last) - _angle(first, last)) <= 1e-10
    )


def _longitude_inside(area: Bounds, longitude: float) -> bool:
    return (
        area.west - 1e-10 <= longitude <= area.east + 1e-10
        if area.west <= area.east
        else longitude >= area.west - 1e-10 or longitude <= area.east + 1e-10
    )


def _cell_intersects(area: Bounds, cell_id: int) -> bool:
    # H3 cell edges are spherical arcs. A planar polygon between their vertex
    # coordinates distorts long edges near the poles and antimeridian. Test
    # containment and the rectangle/arc intersections on the sphere instead.
    resolution = h3.get_resolution(cell_id)
    for latitude in (area.south, area.north):
        for longitude in (area.west, area.east):
            if h3.latlng_to_cell(latitude, longitude, resolution) == cell_id:
                return True
    boundary = h3.cell_to_boundary(cell_id)
    for latitude, longitude in boundary:
        if area.south <= latitude <= area.north and _longitude_inside(area, longitude):
            return True
    vertices = [_vector(latitude, longitude) for latitude, longitude in boundary]
    for first, last in zip(vertices, [*vertices[1:], vertices[0]], strict=True):
        normal = _cross(first, last)
        for longitude in (area.west, area.east):
            meridian = math.radians(longitude)
            intersections = _cross(
                normal, (-math.sin(meridian), math.cos(meridian), 0.0)
            )
            norm = math.sqrt(_dot(intersections, intersections))
            if norm <= 1e-15:
                continue
            for sign in (-1, 1):
                point = cast(
                    _Vector, tuple(sign * value / norm for value in intersections)
                )
                latitude = math.degrees(math.asin(max(-1.0, min(1.0, point[2]))))
                candidate_longitude = math.degrees(math.atan2(point[1], point[0]))
                delta = (candidate_longitude - longitude + 180) % 360 - 180
                if (
                    area.south - 1e-10 <= latitude <= area.north + 1e-10
                    and (abs(delta) <= 1e-8 or abs(latitude) >= 90 - 1e-8)
                    and _on_arc(point, first, last)
                ):
                    return True
        horizontal = normal[0] ** 2 + normal[1] ** 2
        if horizontal <= 1e-30:
            continue
        for latitude in (area.south, area.north):
            z = math.sin(math.radians(latitude))
            constant = -normal[2] * z
            radius_squared = max(0.0, 1 - z * z)
            remaining = radius_squared - constant * constant / horizontal
            if remaining < -1e-15:
                continue
            offset = math.sqrt(max(0.0, remaining) / horizontal)
            for sign in (-1, 1):
                x = constant * normal[0] / horizontal - sign * normal[1] * offset
                y = constant * normal[1] / horizontal + sign * normal[0] * offset
                point = (x, y, z)
                longitude = math.degrees(math.atan2(y, x))
                if _longitude_inside(area, longitude) and _on_arc(point, first, last):
                    return True
    return False


def _rename_mapping(
    requested: Mapping[str, str] | None, available: Sequence[str], dimension: str
) -> dict[str, str]:
    if requested is None:
        return {}
    if not isinstance(requested, Mapping):
        raise TypeError(f"{dimension} renaming requires a mapping")
    mapping = {
        _identifier(key, f"old {dimension} id"): _identifier(
            value, f"new {dimension} id"
        )
        for key, value in requested.items()
    }
    unknown = set(mapping) - set(available)
    if unknown:
        raise ValueError(f"unknown {dimension} identities {sorted(unknown)!r}")
    effective = [mapping.get(identifier, identifier) for identifier in available]
    if len(set(effective)) != len(effective):
        raise ValueError(
            f"{dimension} renaming collides with another participating identity"
        )
    return {key: value for key, value in mapping.items() if key != value}


def rename_spec(
    spec: DataSpec,
    *,
    variables: Mapping[str, str] | None = None,
    scenarios: Mapping[str, str] | None = None,
) -> tuple[DataSpec, tuple[IdentityRename, ...]]:
    variable_mapping = _rename_mapping(
        variables, [v.id for v in spec.variables], "variable"
    )
    scenario_mapping = _rename_mapping(
        scenarios, [s.id for s in spec.scenarios], "scenario"
    )
    new_variables = {
        variable.id: Variable(
            variable_mapping.get(variable.id, variable.id),
            measurement=variable.measurement,
            label=variable.label,
        )
        for variable in spec.variables
    }
    new_scenarios = {
        scenario.id: Scenario(
            scenario_mapping.get(scenario.id, scenario.id),
            definition=scenario.definition,
            schema=scenario.schema,
            label=scenario.label,
        )
        for scenario in spec.scenarios
    }
    result = DataSpec(
        tuple(new_variables.values()),
        series=[
            SeriesSpec(
                new_variables[item.variable.id],
                scenario=None
                if item.scenario is None
                else new_scenarios[item.scenario.id],
                period=item.period,
                probability=item.probability,
            )
            for item in spec.series
        ],
        scenarios=tuple(new_scenarios.values()),
        periods=spec.periods,
    )
    provenance = tuple(
        IdentityRename(dimension=dimension, mapping=mapping)
        for dimension, mapping in (
            (IdentityKind.VARIABLE, variable_mapping),
            (IdentityKind.SCENARIO, scenario_mapping),
        )
        if mapping
    )
    return result, provenance


def rename_row(
    row: DatasetRow,
    *,
    variables: Mapping[str, str] | None = None,
    scenarios: Mapping[str, str] | None = None,
) -> DatasetRow:
    return replace(
        row,
        variable_id=(variables or {}).get(row.variable_id, row.variable_id),
        scenario_id=(scenarios or {}).get(row.scenario_id, row.scenario_id)
        if row.scenario_id is not None
        else None,
    )


def rename_metadata(
    metadata: MetadataT,
    *,
    variables: Mapping[str, str] | None = None,
    scenarios: Mapping[str, str] | None = None,
) -> MetadataT:
    spec, provenance = rename_spec(
        metadata.spec, variables=variables, scenarios=scenarios
    )
    return replace(metadata, spec=spec, provenance=(*metadata.provenance, *provenance))


def selected_metadata(metadata: MetadataT, selected: ResolvedSelection) -> MetadataT:
    variables = {item.id for item in selected.spec.variables}
    scenarios = {item.id for item in selected.spec.scenarios}
    retained: list[Transformation] = []
    for step in reversed(metadata.provenance):
        if isinstance(step, TemporalReduction):
            retained.append(
                replace(
                    step,
                    measurements=tuple(
                        change
                        for change in step.measurements
                        if change.variable_id in variables
                    ),
                )
            )
            continue
        if isinstance(step, ObservationTimeSelection):
            retained.append(step)
            continue
        if isinstance(step, UnitConversion):
            if step.variable_id in variables:
                retained.append(step)
            continue
        participating = (
            variables if step.dimension is IdentityKind.VARIABLE else scenarios
        )
        mapping = {
            old: new for old, new in step.mapping.items() if new in participating
        }
        if mapping:
            retained.append(IdentityRename(dimension=step.dimension, mapping=mapping))
            inverse = {new: old for old, new in mapping.items()}
            before = {
                inverse.get(identifier, identifier) for identifier in participating
            }
            participating.clear()
            participating.update(before)
    provenance = tuple(reversed(retained))
    if selected.time is not None:
        time = selected.time
        if isinstance(metadata, EvidenceMetadata) and metadata.sampling is not None:
            time = _observation_time(time, {metadata.sampling.calendar})
        selection = ObservationTimeSelection(time=time)
        if selection not in provenance:
            provenance = (*provenance, selection)
    if isinstance(metadata, EvidenceMetadata) and metadata._empty_supports:
        empty_supports = (
            ()
            if selected.empty
            else tuple(
                support
                for support in metadata._empty_supports
                if (selected.stations is None or support.id in selected.stations)
                and (
                    selected.area is None
                    or support_intersects(
                        selected.area,
                        geometry=support.geometry,
                        cell_id=support.cell_id,
                    )
                )
            )
        )
        return replace(
            metadata,
            spec=selected.spec,
            provenance=provenance,
            _empty_supports=empty_supports,
        )
    return replace(metadata, spec=selected.spec, provenance=provenance)


_UNIT_TYPES = (
    crc_framework.LengthUnit,
    crc_framework.TimeUnit,
    crc_framework.RatioUnit,
    crc_framework.TemperatureUnit,
    crc_framework.SpeedUnit,
    crc_framework.PrecipitationRateUnit,
    crc_framework.DegreeDayUnit,
)


def conversion_spec(
    spec: DataSpec,
    *,
    variable: Variable | str,
    unit: Unit,
) -> tuple[DataSpec, UnitConversion | None]:
    if not isinstance(unit, _UNIT_TYPES):
        raise TypeError("unit must be a typed framework unit")
    requested = _resolve_ids((variable,), spec.variables)
    assert requested is not None
    identifier = next(iter(requested))
    original = next(item for item in spec.variables if item.id == identifier)
    measurement = original.measurement
    if measurement is None or measurement.unit is None:
        raise ValueError(f"variable {identifier!r} has no measurement unit to convert")
    source_unit = (
        measurement.unit.value
        if isinstance(measurement.unit, Enum)
        else measurement.unit
    )
    if source_unit == unit.value:
        return spec, None
    scale, offset = _core.unit_conversion(source_unit, unit.value)
    document = json.loads(measurement.to_json())
    document["unit"] = unit.value
    try:
        converted = crc_framework.Measurement.from_json(json.dumps(document))
    except ValueError as error:
        raise ValueError(
            f"{type(measurement).__name__} does not permit unit {unit.value!r}"
        ) from error
    replacement = Variable(identifier, measurement=converted, label=original.label)
    result = DataSpec(
        [replacement if item.id == identifier else item for item in spec.variables],
        series=[
            SeriesSpec(
                replacement if item.variable.id == identifier else item.variable,
                scenario=item.scenario,
                period=item.period,
                probability=item.probability,
            )
            for item in spec.series
        ],
        scenarios=spec.scenarios,
        periods=spec.periods,
    )
    return result, UnitConversion(
        variable_id=identifier,
        source_unit=source_unit,
        target_unit=unit.value,
        scale=scale,
        offset=offset,
    )


def conversion_row(
    row: DatasetRow,
    conversion: UnitConversion,
    *,
    input: crc_framework.Measurement,
    output: crc_framework.Measurement,
) -> DatasetRow:
    if row.variable_id != conversion.variable_id or row.curve is None:
        return row
    if isinstance(row.curve, crc_framework.QuantileCurve):
        curve: crc_framework.Distribution | crc_framework.QuantileCurve = (
            crc_framework.QuantileCurve(
                row.curve.probabilities,
                row.curve.values * conversion.scale + conversion.offset,
            )
        )
    else:
        curve = crc_framework.Impact.linear(
            slope=conversion.scale,
            intercept=conversion.offset,
            input=input,
            output=output,
        ).pushforward(row.curve)
    return replace(row, curve=curve)


def conversion_metadata(
    metadata: MetadataT,
    *,
    variable: Variable | str,
    unit: Unit,
) -> tuple[MetadataT, UnitConversion | None]:
    spec, conversion = conversion_spec(metadata.spec, variable=variable, unit=unit)
    return (
        replace(
            metadata,
            spec=spec,
            provenance=metadata.provenance
            if conversion is None
            else (*metadata.provenance, conversion),
        ),
        conversion,
    )
