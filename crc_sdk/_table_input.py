"""Typed long-form table inputs and owned, canonical snapshots."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute  # type: ignore[import-untyped]  # noqa: F401
from crc_framework import QuantileCurve
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from numpy.typing import NDArray
from shapely import from_wkb  # type: ignore[import-untyped]

from ._aggregation import Sampling, _annual_sampling
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionReport,
    SpatialContract,
    SupportKind,
    _identifier,
    _number,
)
from ._declarations import (
    DataSpec,
    Period,
    Probability,
    Scenario,
    SeriesSpec,
    Variable,
)
from ._evidence import EvidenceKind, EvidenceMetadata, _sample_time, evidence_table
from ._schema.dataset import _geometry, dataset_arrow_schema

Scalar: TypeAlias = str | int | float | bool | bytes | date | None


@runtime_checkable
class ArrowInterchange(Protocol):
    def __arrow_c_stream__(self, requested_schema: object = None) -> object: ...


@runtime_checkable
class _DataFrameInterchange(Protocol):
    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> object: ...


TableInput: TypeAlias = (
    Mapping[str, Sequence[Scalar] | NDArray[np.generic]]
    | pa.Table
    | pa.RecordBatch
    | ArrowInterchange
    | _DataFrameInterchange
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TimeSeriesColumns:
    value: str = "value"
    time: str = "time"
    support_id: str = "support_id"
    longitude: str = "longitude"
    latitude: str = "latitude"
    variable_id: str | None = None
    scenario_id: str | None = None
    period_id: str | None = None
    station_id: str | None = None
    group_id: str | None = None

    def __post_init__(self) -> None:
        _columns(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantileColumns:
    support_id: str = "support_id"
    longitude: str = "longitude"
    latitude: str = "latitude"
    probability: str = "probability"
    value: str = "value"
    variable_id: str | None = None
    scenario_id: str | None = None
    period_id: str | None = None
    cell: str | None = None
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.probabilities is not None:
            levels = (
                None
                if isinstance(self.probabilities, (str, bytes))
                else np.asarray(self.probabilities)
            )
            if levels is None or levels.ndim != 1 or levels.dtype.kind not in "iuf":
                raise TypeError("probabilities must be a sequence of numbers")
            if not np.isfinite(levels).all():
                raise ValueError("probabilities must be finite")
            object.__setattr__(self, "probabilities", tuple(float(p) for p in levels))
            if not self.probabilities:
                raise ValueError("probabilities must not be empty")
            if len(set(self.probabilities)) != len(self.probabilities):
                raise ValueError("probabilities must be distinct")
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None and name != "probabilities":
                _identifier(value, "column name")
        read = self._read()
        if len(set(read)) != len(read):
            raise ValueError("column mappings must name distinct columns")

    def _read(self) -> tuple[str, ...]:
        """The input columns this mapping reads."""
        names = [self.support_id, self.value]
        names += (
            [self.cell] if self.cell is not None else [self.longitude, self.latitude]
        )
        if self.probabilities is None:
            names.append(self.probability)
        names += [
            name
            for name in (self.variable_id, self.scenario_id, self.period_id)
            if name is not None
        ]
        return tuple(names)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectionColumns:
    support_id: str = "support_id"
    longitude: str = "longitude"
    latitude: str = "latitude"
    time: str | None = None
    return_period: str | None = None
    value: str | None = "value"
    probability: str | None = None
    lower: str | None = None
    upper: str | None = None
    confidence: str | None = None
    variable_id: str | None = None
    scenario_id: str | None = None
    period_id: str | None = None
    station_id: str | None = None

    def __post_init__(self) -> None:
        if self.time is None and self.return_period is None:
            object.__setattr__(self, "time", "time")
        _columns(self)
        if (self.time is None) == (self.return_period is None):
            raise ValueError("map exactly one of time or return_period")
        if any(
            v is not None for v in (self.lower, self.upper, self.confidence)
        ) and any(v is None for v in (self.lower, self.upper, self.confidence)):
            raise ValueError("interval columns require lower, upper, and confidence")
        if self.value is None and self.lower is None:
            raise ValueError("map a point value, an interval, or both")


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerColumns:
    support_id: str = "support_id"
    longitude: str | None = None
    latitude: str | None = None
    geometry: str | None = None
    value: str | None = "value"
    label: str | None = None
    variable_id: str | None = None
    scenario_id: str | None = None
    period_id: str | None = None

    def __post_init__(self) -> None:
        if self.geometry is None:
            if self.longitude is None:
                object.__setattr__(self, "longitude", "longitude")
            if self.latitude is None:
                object.__setattr__(self, "latitude", "latitude")
        _columns(self)
        point = self.longitude is not None and self.latitude is not None
        if point == (self.geometry is not None):
            raise ValueError("map point coordinates or WKB geometry, exclusively")
        if (self.longitude is None) != (self.latitude is None):
            raise ValueError("longitude and latitude must be mapped together")


def _columns(
    record: TimeSeriesColumns | ProjectionColumns | LayerColumns,
) -> None:
    names = [
        getattr(record, name)
        for name in record.__dataclass_fields__
        if getattr(record, name) is not None
    ]
    for name in names:
        _identifier(name, "column name")
    if len(set(names)) != len(names):
        raise ValueError("column mappings must name distinct columns")


def snapshot_table(table: TableInput) -> pa.Table:
    """Take ownership even when an Arrow input borrows mutable NumPy buffers."""
    if isinstance(table, pa.RecordBatch):
        result = pa.Table.from_batches([table])
    elif isinstance(table, pa.Table):
        result = table
    elif isinstance(table, Mapping):
        if any(not isinstance(name, str) for name in table):
            raise TypeError("table column names must be strings")
        result = pa.table(table)
    elif hasattr(table, "__arrow_c_stream__"):
        result = pa.table(table)
    elif hasattr(table, "__dataframe__"):
        from pyarrow.interchange import from_dataframe  # type: ignore[import-untyped]

        result = from_dataframe(table)
    else:
        raise TypeError(
            "table must be a column mapping, Arrow table or record batch, "
            "or a supported Arrow/dataframe interchange object"
        )
    if len(set(result.column_names)) != len(result.column_names):
        raise ValueError("table column names must be unique")
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, result.schema) as writer:
        writer.write_table(result)
    return pa.ipc.open_stream(sink.getvalue()).read_all()


def input_spec(
    *,
    variable: Variable | None = None,
    probability: Probability | None = None,
    scenario: Scenario | None = None,
    period: Period | None = None,
    spec: DataSpec | None = None,
) -> DataSpec:
    if spec is not None:
        if not isinstance(spec, DataSpec):
            raise TypeError("spec must be DataSpec")
        if any(v is not None for v in (variable, probability, scenario, period)):
            raise ValueError("supply spec or direct scientific declarations, not both")
        return spec
    if not isinstance(variable, Variable):
        raise TypeError("supply variable=Variable(...) or spec=DataSpec(...)")
    return DataSpec(
        [variable],
        series=[
            SeriesSpec(
                variable, scenario=scenario, period=period, probability=probability
            )
        ],
    )


def _point(
    record: dict[str, object], longitude: str, latitude: str
) -> tuple[int, bytes]:
    lon = _number(record[longitude], longitude)
    lat = _number(record[latitude], latitude)
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError("point coordinates must be finite WGS84 longitude/latitude")
    return h3.latlng_to_cell(lat, lon, 9), struct.pack("<BIdd", 1, 1, lon, lat)


def _require_columns(table: pa.Table, names: Sequence[str]) -> None:
    missing = set(names) - set(table.column_names)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")


def _series_identity(
    record: dict[str, object],
    spec: DataSpec,
    variable_column: str | None,
    scenario_column: str | None,
    period_column: str | None,
) -> tuple[str, str | None, str | None]:
    declared = {
        (
            s.variable.id,
            None if s.scenario is None else s.scenario.id,
            None if s.period is None else s.period.id,
        )
        for s in spec.series
    }
    if not declared:
        raise ValueError("input requires at least one declared scientific series")
    identity: list[str | None] = []
    for index, column in enumerate((variable_column, scenario_column, period_column)):
        if column is None:
            values = {item[index] for item in declared}
            if len(values) != 1:
                raise ValueError(
                    "multiple scientific identities require an explicit column mapping"
                )
            identity.append(next(iter(values)))
        else:
            value = record[column]
            if value is not None:
                _identifier(value, column)
            identity.append(cast(str | None, value))
    result = cast(tuple[str, str | None, str | None], tuple(identity))
    if result not in declared:
        raise ValueError(f"input refers to an undeclared scientific series: {result!r}")
    return result


def _report(count: int) -> ExecutionReport:
    return ExecutionReport(
        complete=True,
        valid=True,
        counts=ExecutionCounts(requested=count, produced=count),
    )


def normalize_samples(
    table: TableInput,
    *,
    variable: Variable | None = None,
    probability: Probability | None = None,
    scenario: Scenario | None = None,
    period: Period | None = None,
    spec: DataSpec | None = None,
    columns: TimeSeriesColumns | None = None,
    sampling: Sampling | None = None,
) -> tuple[pa.Table, EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]]:
    declarations = input_spec(
        variable=variable,
        probability=probability,
        scenario=scenario,
        period=period,
        spec=spec,
    )
    columns = TimeSeriesColumns() if columns is None else columns
    if not isinstance(columns, TimeSeriesColumns):
        raise TypeError("columns must be TimeSeriesColumns")
    source = snapshot_table(table)
    _require_columns(
        source,
        [
            getattr(columns, name)
            for name in columns.__dataclass_fields__
            if getattr(columns, name) is not None
        ],
    )
    records: list[dict[str, object]] = []
    for record in source.to_pylist():
        identity = _series_identity(
            record,
            declarations,
            columns.variable_id,
            columns.scenario_id,
            columns.period_id,
        )
        cell, geometry = _point(record, columns.longitude, columns.latitude)
        records.append(
            {
                "variable_id": identity[0],
                "scenario_id": identity[1],
                "period_id": identity[2],
                "support_id": _identifier(
                    record[columns.support_id], columns.support_id
                ),
                "cell_id": cell,
                "geometry": geometry,
                "time": _sample_time(record[columns.time]),
                "value": None
                if record[columns.value] is None
                else _number(record[columns.value], columns.value),
                "failure": {
                    "code": "missing_observation",
                    "message": "The supplied observation is missing",
                }
                if record[columns.value] is None
                else None,
                "station_id": None
                if columns.station_id is None
                else record[columns.station_id],
                "group_id": None
                if columns.group_id is None
                else record[columns.group_id],
            }
        )
    missing = sum(r["failure"] is not None for r in records)
    annual = _annual_sampling(sampling)
    metadata: EvidenceMetadata[Literal[EvidenceKind.SAMPLES]] = EvidenceMetadata(
        kind=EvidenceKind.SAMPLES,
        sampling=sampling,
        spec=declarations,
        spatial=SpatialContract(supports=(SupportKind.POINT,), index_resolution=9),
        report=ExecutionReport(
            True,
            True,
            counts=ExecutionCounts(
                requested=len(records),
                produced=len(records),
                no_data=missing,
                usable_years=len(records) - missing if annual else 0,
                incomplete_years=missing if annual else 0,
            ),
        ),
    )
    return evidence_table(records, metadata), metadata


def normalize_observations(
    table: TableInput,
    *,
    variable: Variable | None = None,
    probability: Probability | None = None,
    scenario: Scenario | None = None,
    period: Period | None = None,
    spec: DataSpec | None = None,
    columns: TimeSeriesColumns | None = None,
    sampling: Sampling | None = None,
) -> tuple[pa.Table, EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]]:
    values, samples = normalize_samples(
        table,
        variable=variable,
        probability=probability,
        scenario=scenario,
        period=period,
        spec=spec,
        columns=columns,
        sampling=sampling,
    )
    metadata: EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]] = EvidenceMetadata(
        kind=EvidenceKind.OBSERVATIONS,
        spec=samples.spec,
        spatial=samples.spatial,
        report=replace(
            samples.report,
            counts=replace(samples.report.counts, usable_years=0, incomplete_years=0),
        ),
        source=samples.source,
        provenance=samples.provenance,
        sampling=samples.sampling,
    )
    from ._evidence import evidence_arrow_schema

    values = values.replace_schema_metadata(evidence_arrow_schema(metadata).metadata)
    return values, metadata


def normalize_projections(
    table: TableInput,
    *,
    variable: Variable | None = None,
    probability: Probability | None = None,
    scenario: Scenario | None = None,
    period: Period | None = None,
    spec: DataSpec | None = None,
    columns: ProjectionColumns | None = None,
) -> tuple[pa.Table, EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]]]:
    declarations = input_spec(
        variable=variable,
        probability=probability,
        scenario=scenario,
        period=period,
        spec=spec,
    )
    columns = ProjectionColumns() if columns is None else columns
    if not isinstance(columns, ProjectionColumns):
        raise TypeError("columns must be ProjectionColumns")
    source = snapshot_table(table)
    _require_columns(
        source,
        [
            getattr(columns, name)
            for name in columns.__dataclass_fields__
            if getattr(columns, name) is not None
        ],
    )
    records: list[dict[str, object]] = []
    for row in source.to_pylist():
        identity = _series_identity(
            row,
            declarations,
            columns.variable_id,
            columns.scenario_id,
            columns.period_id,
        )
        cell, geometry = _point(row, columns.longitude, columns.latitude)
        coordinate = (
            {"kind": "time", "time": _sample_time(row[columns.time])}
            if columns.time is not None
            else {
                "kind": "return_period",
                "years": _number(
                    row[cast(str, columns.return_period)], "return_period"
                ),
            }
        )
        interval = (
            None
            if columns.lower is None
            else {
                "confidence": _number(row[cast(str, columns.confidence)], "confidence"),
                "lower": _number(row[columns.lower], "lower"),
                "upper": _number(row[cast(str, columns.upper)], "upper"),
            }
        )
        records.append(
            {
                "variable_id": identity[0],
                "scenario_id": identity[1],
                "period_id": identity[2],
                "support_id": _identifier(row[columns.support_id], columns.support_id),
                "cell_id": cell,
                "geometry": geometry,
                "coordinate": coordinate,
                "value": None
                if columns.value is None
                else _number(row[columns.value], columns.value),
                "probability": None
                if columns.probability is None
                else _number(row[columns.probability], columns.probability),
                "interval": interval,
                "station_id": None
                if columns.station_id is None
                else row[columns.station_id],
                "failure": None,
            }
        )
    metadata: EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]] = EvidenceMetadata(
        kind=EvidenceKind.PROJECTIONS,
        spec=declarations,
        spatial=SpatialContract(supports=(SupportKind.POINT,), index_resolution=9),
        report=_report(len(records)),
    )
    return evidence_table(records, metadata), metadata


def normalize_layer(
    table: TableInput,
    *,
    variable: Variable | None = None,
    scenario: Scenario | None = None,
    period: Period | None = None,
    spec: DataSpec | None = None,
    columns: LayerColumns | None = None,
) -> tuple[pa.Table, EvidenceMetadata[Literal[EvidenceKind.LAYER]]]:
    declarations = input_spec(
        variable=variable, scenario=scenario, period=period, spec=spec
    )
    columns = LayerColumns() if columns is None else columns
    if not isinstance(columns, LayerColumns):
        raise TypeError("columns must be LayerColumns")
    source = snapshot_table(table)
    _require_columns(
        source,
        [
            getattr(columns, name)
            for name in columns.__dataclass_fields__
            if getattr(columns, name) is not None
        ],
    )
    records: list[dict[str, object]] = []
    supports: set[SupportKind] = set()
    for row in source.to_pylist():
        identity = _series_identity(
            row,
            declarations,
            columns.variable_id,
            columns.scenario_id,
            columns.period_id,
        )
        if columns.geometry is None:
            cell, geometry = _point(
                row, cast(str, columns.longitude), cast(str, columns.latitude)
            )
            supports.add(SupportKind.POINT)
        else:
            geometry = row[columns.geometry]
            if not isinstance(geometry, bytes):
                raise TypeError("geometry columns contain WKB bytes")
            supports.add(_geometry(geometry))
            point = from_wkb(geometry).representative_point()
            cell = h3.latlng_to_cell(point.y, point.x, 9)
        records.append(
            {
                "variable_id": identity[0],
                "scenario_id": identity[1],
                "period_id": identity[2],
                "support_id": _identifier(row[columns.support_id], columns.support_id),
                "cell_id": cell,
                "geometry": geometry,
                "value": None
                if columns.value is None
                else _number(row[columns.value], columns.value),
                "label": None if columns.label is None else row[columns.label],
                "failure": None,
            }
        )
    metadata: EvidenceMetadata[Literal[EvidenceKind.LAYER]] = EvidenceMetadata(
        kind=EvidenceKind.LAYER,
        spec=declarations,
        spatial=SpatialContract(
            supports=tuple(sorted(supports, key=lambda k: k.value))
            if supports
            else (
                SupportKind.POLYGON
                if columns.geometry is not None
                else SupportKind.POINT,
            ),
            index_resolution=9,
        ),
        report=_report(len(records)),
    )
    return evidence_table(records, metadata), metadata


def _check_curves(
    q: NDArray[np.float64], values: NDArray[np.float64], offsets: NDArray[np.int64]
) -> None:
    """Check every curve's knots as `QuantileCurve` does, all curves at once.

    The first invalid curve is rebuilt as a `QuantileCurve`, so the error is
    the framework's own.
    """
    counts = np.diff(offsets)
    bad = counts < 2
    starts, ends = offsets[:-1], offsets[1:]
    present = counts > 0
    bad[present] |= (q[starts[present]] < 0) | (q[ends[present] - 1] > 1)
    knots = ~np.isfinite(q) | ~np.isfinite(values)
    if len(q) > 1:
        within = np.ones(len(q) - 1, dtype=bool)
        within[offsets[1:-1][(offsets[1:-1] > 0) & (offsets[1:-1] < len(q))] - 1] = (
            False
        )
        broken = within & ((np.diff(q) <= 0) | (np.diff(values) < 0))
        knots[:-1] |= broken
    if knots.any():
        curve = np.searchsorted(offsets, np.flatnonzero(knots), side="right") - 1
        bad[curve] = True
    if bad.any():
        index = int(np.flatnonzero(bad)[0])
        start, end = offsets[index], offsets[index + 1]
        QuantileCurve(q[start:end].tolist(), values[start:end].tolist())
        raise AssertionError(f"curve {index} failed vectorized knot checks only")


def normalize_quantiles(
    table: TableInput,
    *,
    variable: Variable | None = None,
    probability: Probability | None = None,
    scenario: Scenario | None = None,
    period: Period | None = None,
    spec: DataSpec | None = None,
    columns: QuantileColumns | None = None,
) -> tuple[pa.Table, DatasetMetadata]:
    """Group supplied knots into canonical quantile-curve rows.

    Rows keep the order in which each curve first appears, and knots are
    sorted by probability. Values are checked column by column; each curve is
    then validated once by the framework.
    """
    declarations = input_spec(
        variable=variable,
        probability=probability,
        scenario=scenario,
        period=period,
        spec=spec,
    )
    columns = QuantileColumns() if columns is None else columns
    if not isinstance(columns, QuantileColumns):
        raise TypeError("columns must be QuantileColumns")
    source = snapshot_table(table)
    _require_columns(source, list(columns._read()))
    count = source.num_rows
    identities, identity_codes = _series_codes(source, declarations, columns)
    support_names, support_codes = _codes(source.column(columns.support_id))
    for name in support_names:
        _identifier(name, columns.support_id)
    if columns.cell is None:
        longitude = _finite(source.column(columns.longitude), columns.longitude)
        latitude = _finite(source.column(columns.latitude), columns.latitude)
        if (np.abs(longitude) > 180).any() or (np.abs(latitude) > 90).any():
            raise ValueError(
                "point coordinates must be finite WGS84 longitude/latitude"
            )
        location: NDArray[np.float64] | NDArray[np.uint64] = np.stack(
            [longitude, latitude], axis=1
        )
    else:
        location = _cells(source.column(columns.cell), columns.cell)[:, None]
    places, place_codes = np.unique(location, axis=0, return_inverse=True)
    groups, first, group_codes = np.unique(
        np.stack([identity_codes, support_codes, place_codes.reshape(-1)], axis=1),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    group_codes = group_codes.reshape(-1)
    # Rows follow the first appearance of their curve; knots are sorted by
    # probability, then value.
    by_appearance = np.argsort(first, kind="stable")
    ordered = groups[by_appearance]
    if columns.probabilities is not None and len(groups) == count:
        # One list per curve: rows keep their order and every row shares one
        # permutation of the declared probabilities, which are distinct, so
        # no knot ties on probability.
        size = len(columns.probabilities)
        _, flat = _knot_lists(source.column(columns.value), columns.value, size)
        levels = np.asarray(columns.probabilities, dtype=np.float64)
        permutation = np.argsort(levels, kind="stable")
        matrix = flat.reshape(count, size)[:, permutation]
        q = np.tile(levels[permutation], count)
        values = matrix.reshape(-1)
        offsets = np.arange(count + 1, dtype=np.int64) * size
    else:
        if columns.probabilities is None:
            knots = np.ones(count, dtype=np.int64)
            q = _finite(source.column(columns.probability), columns.probability)
            values = _finite(source.column(columns.value), columns.value)
        else:
            knots, values = _knot_lists(
                source.column(columns.value), columns.value, len(columns.probabilities)
            )
            q = np.tile(np.asarray(columns.probabilities, dtype=np.float64), count)
        rank = np.empty(len(groups), dtype=np.int64)
        rank[by_appearance] = np.arange(len(groups))
        curve_rank = np.repeat(rank[group_codes], knots)
        order = np.lexsort((values, q, curve_rank))
        q, values = q[order], values[order]
        offsets = np.concatenate(
            [[0], np.cumsum(np.bincount(curve_rank, minlength=len(groups)))]
        )
    _check_curves(q, values, offsets)
    if columns.cell is None:
        coordinates = places[ordered[:, 2]]
        cells = np.fromiter(
            (h3.latlng_to_cell(lat, lon, 9) for lon, lat in coordinates),
            dtype=np.uint64,
            count=len(coordinates),
        )
        geometry: list[bytes | None] = [
            struct.pack("<BIdd", 1, 1, lon, lat) for lon, lat in coordinates
        ]
        resolution: int | None = 9
        kind = SupportKind.POINT
    else:
        cells = places[ordered[:, 2], 0].astype(np.uint64)
        geometry = [None] * len(cells)
        resolutions = {h3.get_resolution(int(cell)) for cell in np.unique(cells)}
        if len(resolutions) > 1:
            raise ValueError(f"{columns.cell} cells must share one H3 resolution")
        resolution = next(iter(resolutions), None)
        kind = SupportKind.CELL
    _unique_rows(ordered, cells, identities, support_names)
    metadata = DatasetMetadata(
        spec=declarations,
        spatial=SpatialContract(supports=(kind,), index_resolution=resolution),
        report=_report(len(ordered)),
    )
    schema = dataset_arrow_schema(metadata)
    series = [identities[code] for code in ordered[:, 0]]
    curve_type = schema.field("curve").type
    knots_type = curve_type.field("quantile_curve").type
    rows = len(ordered)
    bounds = pa.array(offsets, pa.int32())
    branches = {
        "kind": pa.array(["quantile_curve"] * rows, pa.string()),
        "quantile_curve": pa.StructArray.from_arrays(
            [
                pa.ListArray.from_arrays(bounds, pa.array(q, pa.float64())),
                pa.ListArray.from_arrays(bounds, pa.array(values, pa.float64())),
            ],
            fields=list(knots_type),
        ),
    }
    curve = pa.StructArray.from_arrays(
        [branches.get(field.name, _absent(rows, field.type)) for field in curve_type],
        fields=list(curve_type),
    )
    table = pa.Table.from_arrays(
        [
            pa.array([item[0] for item in series], pa.string()),
            pa.array([item[1] for item in series], pa.string()),
            pa.array([item[2] for item in series], pa.string()),
            pa.array(support_names[ordered[:, 1]].tolist(), pa.string()),
            pa.array(cells, pa.uint64()),
            pa.array(geometry, pa.binary()),
            curve,
            _absent(rows, schema.field("fit").type),
            _absent(rows, schema.field("failure").type),
        ],
        schema=schema,
    )
    return table, metadata


def _absent(rows: int, kind: pa.DataType) -> pa.Array:
    """Nulls whose struct children still satisfy their non-null fields."""
    return pa.array([None] * rows, type=kind)


def _codes(column: pa.ChunkedArray) -> tuple[NDArray[np.object_], NDArray[np.int64]]:
    """Distinct values, with null last, and each row's code into them."""
    encoded = column.combine_chunks().dictionary_encode()
    names = np.array(encoded.dictionary.to_pylist() + [None], dtype=object)
    codes = encoded.indices.fill_null(len(names) - 1).to_numpy().astype(np.int64)
    used = np.unique(codes)
    remap = np.full(len(names), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return names[used], remap[codes]


def _finite(column: pa.ChunkedArray | pa.Array, name: str) -> NDArray[np.float64]:
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    kind = column.type
    if not (pa.types.is_integer(kind) or pa.types.is_floating(kind)):
        raise ValueError(f"{name} must be finite")
    if column.null_count:
        raise ValueError(f"{name} must be finite")
    values: NDArray[np.float64] = column.to_numpy(zero_copy_only=False).astype(
        np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _knot_lists(
    column: pa.ChunkedArray, name: str, size: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    lists = column.combine_chunks()
    if not (
        pa.types.is_list(lists.type)
        or pa.types.is_large_list(lists.type)
        or pa.types.is_fixed_size_list(lists.type)
    ):
        raise ValueError(f"{name} must be a list column when probabilities are given")
    if lists.null_count:
        raise ValueError(f"{name} must hold one value per probability")
    lengths = pa.compute.list_value_length(lists).to_numpy(zero_copy_only=False)
    if (lengths != size).any():
        raise ValueError(f"{name} must hold one value per probability")
    return np.full(len(lists), size, dtype=np.int64), _finite(lists.flatten(), name)


def _cells(column: pa.ChunkedArray, name: str) -> NDArray[np.uint64]:
    cells = column.combine_chunks()
    if not pa.types.is_integer(cells.type) or cells.null_count:
        raise ValueError(f"{name} must hold valid H3 cell integers")
    raw = cells.to_numpy(zero_copy_only=False)
    if raw.dtype.kind == "i" and (raw < 0).any():
        raise ValueError(f"{name} must hold valid H3 cell integers")
    values: NDArray[np.uint64] = raw.astype(np.uint64)
    for cell in np.unique(values):
        if not h3.is_valid_cell(int(cell)):
            raise ValueError(f"{name} must hold valid H3 cell integers")
    return values


def _series_codes(
    source: pa.Table, spec: DataSpec, columns: QuantileColumns
) -> tuple[list[tuple[str, str | None, str | None]], NDArray[np.int64]]:
    """Each row's declared scientific series, as a code into the distinct ones."""
    declared = {
        (
            s.variable.id,
            None if s.scenario is None else s.scenario.id,
            None if s.period is None else s.period.id,
        )
        for s in spec.series
    }
    if not declared:
        raise ValueError("input requires at least one declared scientific series")
    parts: list[NDArray[np.object_]] = []
    codes: list[NDArray[np.int64]] = []
    for index, column in enumerate(
        (columns.variable_id, columns.scenario_id, columns.period_id)
    ):
        if column is None:
            options = {item[index] for item in declared}
            if len(options) != 1:
                raise ValueError(
                    "multiple scientific identities require an explicit column mapping"
                )
            parts.append(np.array([next(iter(options))], dtype=object))
            codes.append(np.zeros(source.num_rows, dtype=np.int64))
        else:
            names, row_codes = _codes(source.column(column))
            for value in names:
                if value is not None:
                    _identifier(value, column)
            parts.append(names)
            codes.append(row_codes)
    if source.num_rows == 0:
        return [], np.zeros(0, dtype=np.int64)
    combined, inverse = np.unique(np.stack(codes, axis=1), axis=0, return_inverse=True)
    identities = [
        cast(
            tuple[str, str | None, str | None],
            tuple(parts[i][code] for i, code in enumerate(row)),
        )
        for row in combined
    ]
    for identity in identities:
        if identity not in declared:
            raise ValueError(
                f"input refers to an undeclared scientific series: {identity!r}"
            )
    return identities, inverse.reshape(-1)


def _unique_rows(
    curves: NDArray[np.int64],
    cells: NDArray[np.uint64],
    identities: list[tuple[str, str | None, str | None]],
    supports: NDArray[np.object_],
) -> None:
    """Reject repeated row keys and supports that name two locations."""
    if not len(curves):
        return
    keys = np.stack([curves[:, 0], curves[:, 1], cells.astype(np.int64)], axis=1)
    distinct, first, counts = np.unique(
        keys, axis=0, return_index=True, return_counts=True
    )
    if (counts > 1).any():
        identity, support, cell = curves[first[counts > 1][0]][:2].tolist() + [
            int(cells[first[counts > 1][0]])
        ]
        key = (*identities[identity], supports[support], cell)
        raise ValueError(f"duplicate dataset row key: {key!r}")
    places = np.unique(curves[:, [1, 2]], axis=0)
    if len(np.unique(places[:, 0])) != len(places):
        raise ValueError("one support_id cannot identify different source supports")
