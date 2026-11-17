"""Typed long-form table inputs and owned, canonical snapshots."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import QuantileCurve
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from numpy.typing import NDArray
from shapely import from_wkb  # type: ignore[import-untyped]

from ._aggregation import Sampling
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
from ._schema.dataset import DatasetRow, _geometry, validate_rows

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

    def __post_init__(self) -> None:
        _columns(self)


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
    record: TimeSeriesColumns | QuantileColumns | ProjectionColumns | LayerColumns,
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
                no_data=sum(r["failure"] is not None for r in records),
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
        report=samples.report,
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


def normalize_quantiles(
    table: TableInput,
    *,
    variable: Variable | None = None,
    probability: Probability | None = None,
    scenario: Scenario | None = None,
    period: Period | None = None,
    spec: DataSpec | None = None,
    columns: QuantileColumns | None = None,
) -> tuple[tuple[DatasetRow, ...], DatasetMetadata]:
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
    _require_columns(
        source,
        [
            getattr(columns, name)
            for name in columns.__dataclass_fields__
            if getattr(columns, name) is not None
        ],
    )
    groups: dict[
        tuple[str, str | None, str | None, str, int, bytes], list[tuple[float, float]]
    ] = {}
    for record in source.to_pylist():
        identity = _series_identity(
            record,
            declarations,
            columns.variable_id,
            columns.scenario_id,
            columns.period_id,
        )
        cell, geometry = _point(record, columns.longitude, columns.latitude)
        support = _identifier(record[columns.support_id], columns.support_id)
        q = _number(record[columns.probability], columns.probability)
        value = _number(record[columns.value], columns.value)
        groups.setdefault((*identity, support, cell, geometry), []).append((q, value))
    rows: list[DatasetRow] = []
    for (
        variable_id,
        scenario_id,
        period_id,
        support_id,
        cell_id,
        geometry,
    ), knots in groups.items():
        knots.sort()
        curve = QuantileCurve([q for q, _ in knots], [value for _, value in knots])
        rows.append(
            DatasetRow(
                variable_id=variable_id,
                scenario_id=scenario_id,
                period_id=period_id,
                support_id=support_id,
                cell_id=cell_id,
                geometry=geometry,
                curve=curve,
            )
        )
    metadata = DatasetMetadata(
        spec=declarations,
        spatial=SpatialContract(supports=(SupportKind.POINT,), index_resolution=9),
        report=_report(len(rows)),
    )
    validate_rows(rows, metadata)
    return tuple(rows), metadata
