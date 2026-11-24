"""Canonical dataset rows and their strictly validated Arrow representation."""

from __future__ import annotations

import math
import os
import re
import struct
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from crc_framework import Distribution, FitRecord, QuantileCurve, _core
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely import from_wkb, is_valid, is_valid_reason  # type: ignore[import-untyped]

from .._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionReport,
    RetainedFailure,
    SourceUnavailable,
    SupportKind,
    _boolean,
    _dump,
    _failure,
    _identifier,
    _integer,
    _load,
    _object,
    _wire_integer,
)

_METADATA_KEY = b"crc.dataset"
_ROW_KEY = ("variable_id", "scenario_id", "period_id", "support_id", "cell_id")
_VARIANTS = (
    "parametric",
    "discrete",
    "histogram",
    "point_mass",
    "hurdle",
    "quantile",
    "transformed",
    "quantile_curve",
)
_SHAPED = {"genextreme", "weibull_min", "weibull_max", "skewnorm", "genpareto"}


def _cell(value: object) -> int:
    index = _integer(value, "cell_id")
    if index >= 1 << 64 or not h3.is_valid_cell(index):
        raise ValueError("cell_id must be a valid H3 cell integer")
    return index


def _geometry(value: bytes) -> SupportKind:
    """Validate canonical 2D WKB structure and WGS84 coordinates."""
    if not isinstance(value, bytes):
        raise TypeError("geometry must be immutable WKB bytes")
    return _geometry_kind(value)


# A support's geometry repeats across rows and validation passes; the check is a
# pure function of its bytes, and failures are not cached.
@lru_cache(maxsize=65536)
def _geometry_kind(value: bytes) -> SupportKind:
    position = 0

    def read(fmt: str) -> tuple[object, ...]:
        nonlocal position
        size = struct.calcsize(fmt)
        if position + size > len(value):
            raise ValueError("truncated WKB geometry")
        result = struct.unpack_from(fmt, value, position)
        position += size
        return result

    def point(endian: str) -> tuple[float, float]:
        x, y = cast(tuple[float, float], read(endian + "dd"))
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not -180 <= x <= 180
            or not -90 <= y <= 90
        ):
            raise ValueError("geometry must contain finite WGS84 coordinates")
        return x, y

    def polygon(endian: str) -> None:
        count = cast(int, read(endian + "I")[0])
        if count < 1:
            raise ValueError("empty polygon support is not allowed")
        for _ in range(count):
            vertices = cast(int, read(endian + "I")[0])
            if vertices < 4:
                raise ValueError("polygon rings require at least four coordinates")
            coords = [point(endian) for _ in range(vertices)]
            if coords[0] != coords[-1]:
                raise ValueError("polygon rings must be closed")
            ox, oy = coords[0]
            area = math.fsum(
                (a[0] - ox) * (b[1] - oy) - (b[0] - ox) * (a[1] - oy)
                for a, b in zip(coords, coords[1:])
            )
            if area == 0:
                raise ValueError("polygon rings must have nonzero area")

    def geometry() -> SupportKind:
        marker = read("B")[0]
        if marker not in (0, 1):
            raise ValueError("invalid WKB byte order")
        endian = "<" if marker else ">"
        kind = read(endian + "I")[0]
        if kind == 1:
            point(endian)
            return SupportKind.POINT
        if kind == 3:
            polygon(endian)
            return SupportKind.POLYGON
        if kind == 6:
            count = cast(int, read(endian + "I")[0])
            if count < 1:
                raise ValueError("empty multipolygon support is not allowed")
            for _ in range(count):
                if geometry() is not SupportKind.POLYGON:
                    raise ValueError("multipolygon members must be polygons")
            return SupportKind.POLYGON
        raise ValueError(
            "geometry must be canonical 2D Point, Polygon, or MultiPolygon WKB"
        )

    result = geometry()
    if position != len(value):
        raise ValueError("geometry has trailing WKB data")
    shape = from_wkb(value)
    if not is_valid(shape):
        raise ValueError(f"invalid source support: {is_valid_reason(shape)}")
    return result


def _curve(value: object) -> Distribution | QuantileCurve:
    if not isinstance(value, dict):
        raise ValueError("curve must be a framework numerical object")
    text = _dump(value)
    if value.get("object") == "distribution":
        return Distribution.from_json(text)
    if value.get("object") == "quantile_curve":
        return QuantileCurve.from_json(text)
    raise ValueError("curve must declare a current framework numerical object")


@dataclass(frozen=True, slots=True)
class DatasetRow:
    variable_id: str
    support_id: str
    cell_id: int
    scenario_id: str | None = None
    period_id: str | None = None
    geometry: bytes | None = None
    curve: Distribution | QuantileCurve | None = None
    fit: FitRecord | None = None
    failure: RetainedFailure | SourceUnavailable | None = None

    def __post_init__(self) -> None:
        _identifier(self.variable_id, "variable_id")
        _identifier(self.support_id, "support_id")
        _cell(self.cell_id)
        for name in ("scenario_id", "period_id"):
            if getattr(self, name) is not None:
                _identifier(getattr(self, name), name)
        if self.geometry is not None:
            _geometry(self.geometry)
        if (self.curve is None) == (self.failure is None):
            raise ValueError("a row requires exactly one of curve or failure")
        if self.curve is not None and not isinstance(
            self.curve, (Distribution, QuantileCurve)
        ):
            raise TypeError("curve must be a framework Distribution or QuantileCurve")
        if self.failure is not None and not isinstance(
            self.failure, (RetainedFailure, SourceUnavailable)
        ):
            raise TypeError("failure must be RetainedFailure or SourceUnavailable")
        if self.fit is not None:
            if (
                not isinstance(self.fit, FitRecord)
                or not self.fit.converged
                or not isinstance(self.curve, Distribution)
            ):
                raise ValueError(
                    "fit records require a successfully fitted distribution"
                )

    def _document(self) -> dict[str, object]:
        result: dict[str, object] = {
            "variable_id": self.variable_id,
            "scenario_id": self.scenario_id,
            "period_id": self.period_id,
            "support_id": self.support_id,
            "cell_id": self.cell_id,
        }
        if self.geometry is not None:
            result["geometry"] = self.geometry.hex()
        if self.curve is not None:
            result["curve"] = _load(self.curve.to_json())
        if self.fit is not None:
            result["fit"] = _load(self.fit.to_json())
        if self.failure is not None:
            result["failure"] = self.failure._document()
        return result


def _row(value: object) -> DatasetRow:
    v = _object(value, set(_ROW_KEY), {"geometry", "curve", "fit", "failure"})
    geometry = v.get("geometry")
    if "geometry" in v and (
        not isinstance(geometry, str)
        or re.fullmatch(r"(?:[0-9a-f]{2}){21,}", geometry) is None
    ):
        raise ValueError("geometry must be lowercase hexadecimal WKB")
    return DatasetRow(
        variable_id=_identifier(v["variable_id"], "variable_id"),
        scenario_id=cast(str | None, v["scenario_id"]),
        period_id=cast(str | None, v["period_id"]),
        support_id=_identifier(v["support_id"], "support_id"),
        cell_id=_cell(_wire_integer(v["cell_id"], "cell_id")),
        geometry=None if geometry is None else bytes.fromhex(cast(str, geometry)),
        curve=None if "curve" not in v else _curve(v["curve"]),
        fit=None if "fit" not in v else FitRecord.from_json(_dump(v["fit"])),
        failure=None if "failure" not in v else _failure(v["failure"]),
    )


def validate_rows(
    rows: Sequence[DatasetRow],
    metadata: DatasetMetadata,
    *,
    stored_fits: Sequence[str | None] | None = None,
) -> ExecutionReport:
    """Validate rows against their metadata.

    ``stored_fits`` holds each row's fit record exactly as stored, when the rows
    were decoded from it, so the stored text is validated without re-encoding.
    """
    if not isinstance(metadata, DatasetMetadata):
        raise TypeError("metadata must be DatasetMetadata")
    if stored_fits is not None and len(stored_fits) != len(rows):
        raise ValueError("stored_fits must have one entry per row")
    series = {
        (
            s.variable.id,
            None if s.scenario is None else s.scenario.id,
            None if s.period is None else s.period.id,
        )
        for s in metadata.spec.series
    }
    keys: set[tuple[str, str | None, str | None, str, int]] = set()
    supports: dict[str, tuple[bytes | None, int | None]] = {}
    provenance: str | None = None
    failed = 0
    for position, row in enumerate(rows):
        if not isinstance(row, DatasetRow):
            raise TypeError("rows must contain DatasetRow objects")
        identity = (row.variable_id, row.scenario_id, row.period_id)
        if identity not in series:
            raise ValueError(
                f"row refers to an undeclared scientific series: {identity!r}"
            )
        key = (*identity, row.support_id, row.cell_id)
        if key in keys:
            raise ValueError(f"duplicate dataset row key: {key!r}")
        keys.add(key)
        if (
            metadata.spatial.index_resolution is not None
            and (row.cell_id >> 52) & 15 != metadata.spatial.index_resolution
        ):
            raise ValueError("cell_id resolution contradicts the spatial contract")
        kind = SupportKind.CELL if row.geometry is None else _geometry(row.geometry)
        if kind not in metadata.spatial.supports:
            raise ValueError("source geometry contradicts the spatial support contract")
        support = (row.geometry, row.cell_id if row.geometry is None else None)
        if row.support_id in supports and supports[row.support_id] != support:
            original_geometry, original_cell = supports[row.support_id]
            same_geometry = (
                original_geometry is not None
                and row.geometry is not None
                and from_wkb(original_geometry).equals(from_wkb(row.geometry))
            )
            if not same_geometry or original_cell != support[1]:
                raise ValueError(
                    "one support_id cannot identify different source supports"
                )
        supports[row.support_id] = support
        failed += int(row.failure is not None)
        if row.fit is not None:
            assert isinstance(row.curve, Distribution)
            if provenance is None:
                provenance = _dump([step._document() for step in metadata.provenance])
            stored = None if stored_fits is None else stored_fits[position]
            _core.validate_fitted_curve(
                row.fit.to_json() if stored is None else stored,
                row.curve.to_json(),
                row.variable_id,
                provenance,
            )
    if metadata.report.complete:
        if metadata.report.counts.produced != len(rows):
            raise ValueError("complete dataset produced count disagrees with its rows")
        if metadata.report.counts.no_data != failed:
            raise ValueError(
                "complete dataset no_data count disagrees with failure rows"
            )
    return ExecutionReport(
        complete=True,
        valid=True,
        counts=ExecutionCounts(requested=len(rows), produced=len(rows), no_data=failed),
        source_identities=() if metadata.source is None else (metadata.source,),
    )


def _curve_arrow(value: Distribution | QuantileCurve) -> dict[str, object]:
    data = _load(value.to_json())
    kind = (
        "quantile_curve" if data.pop("object") == "quantile_curve" else data.pop("kind")
    )
    data.pop("schema_version")
    evidence = data.pop("evidence", None)
    if kind == "parametric":
        pass
    elif kind == "hurdle":
        data["base"] = _dump(data["base"])
    elif kind == "transformed":
        data["source"] = _dump(data["source"])
        data["impact"] = _dump(data["impact"])
    return {
        "kind": kind,
        **{variant: data if variant == kind else None for variant in _VARIANTS},
        "evidence": None if evidence is None else _dump(evidence),
    }


def _curve_from_arrow(value: object) -> Distribution | QuantileCurve:
    v = _object(value, {"kind", "evidence", *_VARIANTS})
    kind = v["kind"]
    if kind not in _VARIANTS:
        raise ValueError("unknown Arrow numerical variant")
    if (
        any(v[name] is not None for name in _VARIANTS if name != kind)
        or v[kind] is None
    ):
        raise ValueError("only the active Arrow numerical branch may be populated")
    branch = v[kind]
    if not isinstance(branch, dict):
        raise ValueError("active Arrow numerical branch must be an object")
    data = {k: x for k, x in branch.items() if x is not None}
    if kind == "hurdle":
        data["base"] = _load(cast(str, data["base"]))
    elif kind == "transformed":
        data["source"] = _load(cast(str, data["source"]))
        data["impact"] = _load(cast(str, data["impact"]))
    document: dict[str, object] = {
        "schema_version": "1.0",
        "object": "quantile_curve" if kind == "quantile_curve" else "distribution",
        **data,
    }
    if kind != "quantile_curve":
        document["kind"] = kind
    if v["evidence"] is not None:
        document["evidence"] = _load(cast(str, v["evidence"]))
    return _curve(document)


def dataset_arrow_schema(metadata: DatasetMetadata | None = None) -> pa.Schema:
    numbers = pa.list_(pa.float64())
    curve_type = pa.struct(
        [
            pa.field("kind", pa.string(), nullable=False),
            pa.field(
                "parametric",
                pa.struct(
                    [
                        pa.field("family", pa.string(), nullable=False),
                        pa.field("location", pa.float64(), nullable=False),
                        pa.field("scale", pa.float64(), nullable=False),
                        pa.field("shape", pa.float64()),
                    ]
                ),
            ),
            pa.field(
                "discrete",
                pa.struct(
                    [
                        pa.field("values", numbers, nullable=False),
                        pa.field("probabilities", numbers, nullable=False),
                        # Sample counts of empirical laws keep their exact ranks.
                        pa.field("counts", pa.list_(pa.uint64())),
                    ]
                ),
            ),
            pa.field(
                "histogram",
                pa.struct(
                    [
                        pa.field("edges", numbers, nullable=False),
                        pa.field("probabilities", numbers, nullable=False),
                    ]
                ),
            ),
            pa.field(
                "point_mass",
                pa.struct([pa.field("value", pa.float64(), nullable=False)]),
            ),
            pa.field(
                "hurdle",
                pa.struct(
                    [
                        pa.field("base", pa.large_string(), nullable=False),
                        pa.field("atom_probability", pa.float64(), nullable=False),
                        pa.field("atom_location", pa.float64(), nullable=False),
                    ]
                ),
            ),
            pa.field(
                "quantile",
                pa.struct(
                    [
                        pa.field("probabilities", numbers, nullable=False),
                        pa.field("values", numbers, nullable=False),
                    ]
                ),
            ),
            pa.field(
                "transformed",
                pa.struct(
                    [
                        pa.field("source", pa.large_string(), nullable=False),
                        pa.field("impact", pa.large_string(), nullable=False),
                    ]
                ),
            ),
            pa.field(
                "quantile_curve",
                pa.struct(
                    [
                        pa.field("probabilities", numbers, nullable=False),
                        pa.field("values", numbers, nullable=False),
                    ]
                ),
            ),
            pa.field("evidence", pa.large_string()),
        ]
    )
    schema = pa.schema(
        [
            pa.field("variable_id", pa.string(), nullable=False),
            pa.field("scenario_id", pa.string()),
            pa.field("period_id", pa.string()),
            pa.field("support_id", pa.string(), nullable=False),
            pa.field("cell_id", pa.uint64(), nullable=False),
            pa.field("geometry", pa.binary()),
            pa.field("curve", curve_type),
            pa.field("fit", pa.large_string()),
            pa.field(
                "failure",
                pa.struct(
                    [
                        pa.field("code", pa.string(), nullable=False),
                        pa.field("message", pa.string(), nullable=False),
                        pa.field("record", pa.large_string()),
                    ]
                ),
            ),
        ]
    )
    return (
        schema
        if metadata is None
        else schema.with_metadata({_METADATA_KEY: metadata.to_json().encode("utf-8")})
    )


def rows_to_table(rows: Sequence[DatasetRow], metadata: DatasetMetadata) -> pa.Table:
    validate_rows(rows, metadata)
    values = []
    for row in rows:
        values.append(
            {
                "variable_id": row.variable_id,
                "scenario_id": row.scenario_id,
                "period_id": row.period_id,
                "support_id": row.support_id,
                "cell_id": row.cell_id,
                "geometry": row.geometry,
                "curve": None if row.curve is None else _curve_arrow(row.curve),
                "fit": None if row.fit is None else row.fit.to_json(),
                "failure": None
                if row.failure is None
                else {
                    "code": row.failure.code,
                    "message": row.failure.message,
                    "record": row.failure.record.to_json()
                    if isinstance(row.failure, RetainedFailure)
                    else None,
                },
            }
        )
    return pa.Table.from_pylist(values, schema=dataset_arrow_schema(metadata))


def metadata_from_table(table: pa.Table) -> DatasetMetadata:
    value = (table.schema.metadata or {}).get(_METADATA_KEY)
    if value is None:
        raise ValueError("canonical dataset metadata is missing")
    return DatasetMetadata.from_json(value.decode("utf-8"))


def table_to_rows(
    table: pa.Table, metadata: DatasetMetadata | None = None
) -> tuple[DatasetRow, ...]:
    if not isinstance(table, pa.Table):
        raise TypeError("table must be an Arrow Table")
    if not table.schema.equals(dataset_arrow_schema(), check_metadata=False):
        raise ValueError("Arrow fields do not match canonical dataset schema 2.0")
    stored = metadata_from_table(table)
    if metadata is not None and metadata.to_json() != stored.to_json():
        raise ValueError("supplied and stored metadata disagree")
    rows = []
    fits: list[str | None] = []
    for value in table.to_pylist():
        failure = value["failure"]
        fits.append(value["fit"])
        rows.append(
            DatasetRow(
                variable_id=value["variable_id"],
                scenario_id=value["scenario_id"],
                period_id=value["period_id"],
                support_id=value["support_id"],
                cell_id=value["cell_id"],
                geometry=value["geometry"],
                curve=None
                if value["curve"] is None
                else _curve_from_arrow(value["curve"]),
                fit=None if value["fit"] is None else FitRecord.from_json(value["fit"]),
                failure=None
                if failure is None
                else SourceUnavailable(failure["code"], failure["message"])
                if failure["record"] is None
                else RetainedFailure(
                    failure["code"],
                    failure["message"],
                    FitRecord.from_json(failure["record"]),
                ),
            )
        )
    validate_rows(rows, stored, stored_fits=fits)
    return tuple(rows)


def dataset_to_json(rows: Sequence[DatasetRow], metadata: DatasetMetadata) -> str:
    validate_rows(rows, metadata)
    return _dump(
        {"metadata": metadata._document(), "rows": [row._document() for row in rows]}
    )


def dataset_from_json(text: str) -> tuple[DatasetMetadata, tuple[DatasetRow, ...]]:
    v = _object(_load(text), {"metadata", "rows"})
    metadata = DatasetMetadata.from_json(_dump(v["metadata"]))
    if not isinstance(v["rows"], list):
        raise ValueError("rows must be an array")
    rows = tuple(_row(value) for value in v["rows"])
    validate_rows(rows, metadata)
    return metadata, rows


def write_parquet(
    path: str | os.PathLike[str], table: pa.Table, *, overwrite: bool = False
) -> Path:
    _boolean(overwrite, "overwrite")
    table_to_rows(table)
    if isinstance(path, str) and "://" in path:
        raise ValueError("canonical dataset storage accepts local filesystem paths")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".parquet", dir=destination.parent
    )
    os.close(descriptor)
    try:
        pq.write_table(table, temporary)
        # The table is validated above; a read-back with the same columns, values
        # and metadata proves the file holds that same valid table. Parquet may
        # rename list element fields, so columns are compared, not field names.
        stored = pq.read_table(temporary)
        if (
            stored.column_names != table.column_names
            or stored.schema.metadata != table.schema.metadata
            or not all(
                stored.column(name).equals(table.column(name))
                for name in table.column_names
            )
        ):
            raise OSError(f"{destination} did not read back as written")
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def read_parquet(
    path: str | os.PathLike[str],
) -> tuple[DatasetMetadata, tuple[DatasetRow, ...]]:
    if isinstance(path, str) and "://" in path:
        raise ValueError("canonical dataset storage accepts local filesystem paths")
    table = pq.read_table(path)
    return metadata_from_table(table), table_to_rows(table)
