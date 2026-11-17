"""Typed long-form table inputs and owned, canonical snapshots."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import QuantileCurve
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from numpy.typing import NDArray

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
from ._schema.dataset import DatasetRow, validate_rows

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
class SampleColumns:
    value: str = "value"
    time: str = "time"
    support_id: str = "support_id"
    longitude: str = "longitude"
    latitude: str = "latitude"
    variable_id: str | None = None
    scenario_id: str | None = None
    period_id: str | None = None

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


def _columns(record: SampleColumns | QuantileColumns) -> None:
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
    columns: SampleColumns | None = None,
) -> tuple[pa.Table, EvidenceMetadata]:
    declarations = input_spec(
        variable=variable,
        probability=probability,
        scenario=scenario,
        period=period,
        spec=spec,
    )
    columns = SampleColumns() if columns is None else columns
    if not isinstance(columns, SampleColumns):
        raise TypeError("columns must be SampleColumns")
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
                "value": _number(record[columns.value], columns.value),
            }
        )
    metadata = EvidenceMetadata(
        kind=EvidenceKind.SAMPLES,
        spec=declarations,
        spatial=SpatialContract(supports=(SupportKind.POINT,), index_resolution=9),
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
