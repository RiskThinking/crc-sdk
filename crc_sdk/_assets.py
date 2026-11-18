"""Explicit, immutable asset input and location mappings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TypeAlias, cast

import duckdb
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as csv  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from shapely import from_wkb, get_coordinates, to_wkb  # type: ignore[import-untyped]
from shapely.errors import GEOSException  # type: ignore[import-untyped]
from shapely.geometry import Point  # type: ignore[import-untyped]

from ._dataset_schema import ResourceIdentity, SourceIdentity, _identifier
from ._errors import CapabilityError
from ._geometry import cell_polygon
from ._graph import ExecutionScope
from ._runtime import CacheMode, LocalPath, _local_path, _memory_bytes
from ._schema.dataset import _cell
from ._table_input import TableInput, snapshot_table


@dataclass(frozen=True, slots=True, kw_only=True)
class PointColumns:
    longitude: str = "longitude"
    latitude: str = "latitude"

    def __post_init__(self) -> None:
        _identifier(self.longitude, "longitude column")
        _identifier(self.latitude, "latitude column")
        if self.longitude == self.latitude:
            raise ValueError("longitude and latitude columns must be distinct")


@dataclass(frozen=True, slots=True, kw_only=True)
class CellColumn:
    name: str = "cell_id"

    def __post_init__(self) -> None:
        _identifier(self.name, "cell column")


@dataclass(frozen=True, slots=True, kw_only=True)
class GeometryColumn:
    name: str = "geometry"

    def __post_init__(self) -> None:
        _identifier(self.name, "geometry column")


@dataclass(frozen=True, slots=True)
class ColumnBinding:
    name: str

    def __post_init__(self) -> None:
        _identifier(self.name, "asset column")


AssetLocation: TypeAlias = PointColumns | CellColumn | GeometryColumn


def _location_columns(location: AssetLocation) -> tuple[str, ...]:
    if isinstance(location, PointColumns):
        return (location.longitude, location.latitude)
    return (location.name,)


def _location(table: pa.Schema, requested: AssetLocation | None) -> AssetLocation:
    names = set(table.names)
    if len(names) != len(table.names):
        raise ValueError("asset column names must be unique")
    if requested is None:
        candidates: list[AssetLocation] = []
        if {"longitude", "latitude"} <= names:
            candidates.append(PointColumns())
        if "cell_id" in names:
            candidates.append(CellColumn())
        if "geometry" in names:
            candidates.append(GeometryColumn())
        if len(candidates) != 1:
            raise ValueError(
                "asset location is missing or ambiguous; pass "
                "location=PointColumns(...), CellColumn(...), or GeometryColumn(...)"
            )
        requested = candidates[0]
    missing = set(_location_columns(requested)) - names
    if missing:
        raise ValueError(f"missing asset location columns: {sorted(missing)}")
    return requested


def _wgs84(value: object) -> bool:
    if isinstance(value, str):
        return value in {"OGC:CRS84", "EPSG:4326", "urn:ogc:def:crs:OGC::CRS84"}
    if isinstance(value, dict):
        identifier = value.get("id")
        if isinstance(identifier, dict):
            authority, code = identifier.get("authority"), identifier.get("code")
            return (authority == "EPSG" and code in (4326, "4326")) or (
                authority == "OGC" and code == "CRS84"
            )
    return False


def _geometry_crs(schema: pa.Schema, column: str) -> None:
    """Unannotated WKB uses WGS84; explicit declarations must agree."""
    try:
        geo = (schema.metadata or {}).get(b"geo")
        if geo is not None:
            document = json.loads(geo)
            if not isinstance(document, dict) or not isinstance(
                document.get("columns"), dict
            ):
                raise ValueError("invalid GeoParquet geometry metadata")
            descriptor = document["columns"].get(column)
            if descriptor is not None:
                if (
                    not isinstance(descriptor, dict)
                    or descriptor.get("encoding") != "WKB"
                ):
                    raise ValueError("asset geometry requires GeoParquet WKB encoding")
                if "crs" in descriptor and not _wgs84(descriptor["crs"]):
                    raise ValueError("asset geometry must declare a WGS84 CRS")
        field = schema.field(column)
        extension = (field.metadata or {}).get(b"ARROW:extension:metadata")
        if extension is not None:
            descriptor = json.loads(extension)
            if not isinstance(descriptor, dict):
                raise ValueError("invalid asset geometry extension metadata")
            if "crs" in descriptor and not _wgs84(descriptor["crs"]):
                raise ValueError("asset geometry must declare a WGS84 CRS")
    except (TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid asset geometry CRS metadata") from error


def _wkb_2d(value: bytes) -> None:
    """Validate dimensions before GEOS, which can discard M on older releases."""

    def geometry(offset: int, depth: int) -> int:
        if depth > 64 or offset + 5 > len(value) or value[offset] not in (0, 1):
            raise ValueError("asset geometry must contain valid two-dimensional WKB")
        endian = "<" if value[offset] else ">"
        kind = struct.unpack_from(endian + "I", value, offset + 1)[0]
        offset += 5
        if kind & 0xC0000000 or 1000 <= (kind & 0x1FFFFFFF) < 4000:
            raise ValueError(
                "asset geometry must be two-dimensional; Z and M are unsupported"
            )
        if kind & 0x20000000:
            if offset + 4 > len(value):
                raise ValueError("asset geometry contains truncated WKB")
            srid = struct.unpack_from(endian + "I", value, offset)[0]
            if srid not in (0, 4326):
                raise ValueError("asset geometry must use a WGS84 CRS")
            offset += 4
        kind &= 0x1FFFFFFF

        def count(position: int) -> tuple[int, int]:
            if position + 4 > len(value):
                raise ValueError("asset geometry contains truncated WKB")
            result = struct.unpack_from(endian + "I", value, position)[0]
            return result, position + 4

        if kind == 1:
            offset += 16
        elif kind == 2:
            points, offset = count(offset)
            offset += 16 * points
        elif kind == 3:
            rings, offset = count(offset)
            if rings > len(value) // 4:
                raise ValueError("asset geometry contains truncated WKB")
            for _ in range(rings):
                points, offset = count(offset)
                offset += 16 * points
        elif kind in (4, 5, 6, 7):
            children, offset = count(offset)
            if children > len(value) // 5:
                raise ValueError("asset geometry contains truncated WKB")
            for _ in range(children):
                offset = geometry(offset, depth + 1)
        else:
            raise ValueError("asset geometry contains an unsupported WKB type")
        if offset > len(value):
            raise ValueError("asset geometry contains truncated WKB")
        return offset

    if geometry(0, 0) != len(value):
        raise ValueError("asset geometry contains trailing WKB bytes")


@dataclass(frozen=True, slots=True)
class AssetData:
    table: pa.Table
    context: pa.Table
    id_column: str
    location: AssetLocation
    geometries: tuple[bytes, ...]
    source: SourceIdentity | None


class Assets:
    __slots__ = (
        "_table",
        "_path",
        "_query",
        "_id_column",
        "_location",
        "_columns",
        "_key",
    )
    _table: pa.Table | None
    _path: Path | None
    _query: str | None
    _id_column: str
    _location: AssetLocation | None
    _columns: tuple[str, ...] | None
    _key: str

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("asset inputs are immutable")

    def __init__(
        self,
        data: TableInput | LocalPath,
        *,
        id_column: str = "asset_id",
        location: AssetLocation | None = None,
        columns: Sequence[str] | None = None,
    ) -> None:
        self._initialize(id_column, location, columns)
        if isinstance(data, (str, os.PathLike)):
            path = _local_path(data)
            object.__setattr__(self, "_path", path)
            object.__setattr__(self, "_table", None)
            identity = str(path)
        else:
            table = snapshot_table(data)
            object.__setattr__(self, "_table", table)
            object.__setattr__(self, "_path", None)
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            identity = hashlib.sha256(sink.getvalue()).hexdigest()
        object.__setattr__(self, "_query", None)
        self._set_key(identity)

    def _initialize(
        self,
        id_column: str,
        location: AssetLocation | None,
        columns: Sequence[str] | None,
    ) -> None:
        _identifier(id_column, "id_column")
        if location is not None and not isinstance(
            location, (PointColumns, CellColumn, GeometryColumn)
        ):
            raise TypeError(
                "location must be PointColumns, CellColumn, or GeometryColumn"
            )
        if isinstance(columns, (str, bytes)):
            raise TypeError("columns must be a sequence of column names")
        retained = (
            None
            if columns is None
            else tuple(_identifier(c, "asset column") for c in columns)
        )
        if retained is not None and len(set(retained)) != len(retained):
            raise ValueError("retained asset columns must be unique")
        object.__setattr__(self, "_id_column", id_column)
        object.__setattr__(self, "_location", location)
        object.__setattr__(self, "_columns", retained)

    def _set_key(self, identity: str) -> None:
        data = (identity, self._id_column, repr(self._location), self._columns)
        object.__setattr__(
            self,
            "_key",
            "assets:" + hashlib.sha256(json.dumps(data).encode()).hexdigest(),
        )

    @classmethod
    def sql(
        cls,
        query: str,
        *,
        location: AssetLocation,
        id_column: str = "asset_id",
        columns: Sequence[str] | None = None,
    ) -> Assets:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty SQL query")
        if not isinstance(location, (PointColumns, CellColumn, GeometryColumn)):
            raise TypeError("Assets.sql requires an explicit asset location mapping")
        result = object.__new__(cls)
        result._initialize(id_column, location, columns)
        object.__setattr__(result, "_query", query)
        object.__setattr__(result, "_path", None)
        object.__setattr__(result, "_table", None)
        result._set_key("sql:" + query)
        return result

    @property
    def id_column(self) -> str:
        return self._id_column

    @property
    def location(self) -> AssetLocation | None:
        return self._location

    def column(self, name: str) -> ColumnBinding:
        return ColumnBinding(name)

    def _input_columns(
        self, schema: pa.Schema, required: tuple[str, ...]
    ) -> tuple[str, ...]:
        location = _location(schema, self._location)
        missing = set(required) - set(schema.names)
        if missing:
            raise ValueError(
                f"Missing bound asset columns: {sorted(missing)!r}. "
                "Include them in the asset input or supply typed context constants."
            )
        keep = (
            tuple(schema.names)
            if self._columns is None
            else tuple(
                dict.fromkeys(
                    (
                        self._id_column,
                        *self._columns,
                        *_location_columns(location),
                        *required,
                    )
                )
            )
        )
        missing = set(keep) - set(schema.names)
        if missing:
            raise ValueError(f"unknown requested asset columns: {sorted(missing)}")
        return keep

    def _retained_columns(self, schema: pa.Schema) -> tuple[str, ...]:
        if self._columns is None:
            return tuple(schema.names)
        keep = tuple(dict.fromkeys((self._id_column, *self._columns)))
        missing = set(keep) - set(schema.names)
        if missing:
            raise ValueError(f"unknown retained asset columns: {sorted(missing)}")
        return keep

    def _retained_schema(self) -> pa.Schema | None:
        if self._table is None:
            return None
        schema = self._table.schema
        return pa.schema(
            [schema.field(name) for name in self._retained_columns(schema)]
        )

    def _missing_bound_columns(self, required: Sequence[str]) -> tuple[str, ...]:
        if self._table is None:
            return ()
        return tuple(sorted(set(required) - set(self._table.column_names)))

    def _collect(
        self,
        batches: Iterable[pa.RecordBatch],
        schema: pa.Schema,
        keep: tuple[str, ...],
        budget: int | None,
    ) -> pa.Table:
        selected_schema = pa.schema(
            [schema.field(name) for name in keep], metadata=schema.metadata
        )
        owned = []
        size = 0
        for batch in batches:
            batch = batch.select(keep)
            size += batch.nbytes
            if budget is not None and size > budget:
                raise MemoryError(
                    "Assets exceed Runtime(memory_limit=...). Reduce retained "
                    "columns or increase the limit."
                )
            owned.append(batch)
        return pa.Table.from_batches(owned, schema=selected_schema)

    def _materialize(
        self, scope: ExecutionScope, *, required_columns: Sequence[str] = ()
    ) -> AssetData:
        scope._active()
        required = tuple(
            sorted(
                {_identifier(name, "asset context column") for name in required_columns}
            )
        )
        key = self._key if not required else self._key + ":" + json.dumps(required)
        if key in scope._asset_results:
            return cast(AssetData, scope._asset_results[key])
        source = None
        budget = _memory_bytes(scope.runtime.memory_limit)
        if self._table is not None:
            table = self._table.select(
                self._input_columns(self._table.schema, required)
            )
        elif self._path is not None:
            pinned, resource = scope.pin_local(self._path)
            source = SourceIdentity(
                provider="user", product="assets", resources=(resource,)
            )
            if self._path.suffix.lower() == ".csv":
                with csv.open_csv(
                    pinned,
                    convert_options=csv.ConvertOptions(
                        column_types={self._id_column: pa.string()}
                    ),
                ) as reader:
                    keep = self._input_columns(reader.schema, required)
                    table = self._collect(reader, reader.schema, keep, budget)
            elif self._path.suffix.lower() in {".parquet", ".pq"}:
                with pq.ParquetFile(pinned) as parquet:
                    keep = self._input_columns(parquet.schema_arrow, required)
                    batches = (
                        parquet.iter_batches(
                            batch_size=scope.runtime.batch_rows, columns=list(keep)
                        )
                        if parquet.num_row_groups
                        else ()
                    )
                    table = self._collect(
                        batches,
                        parquet.schema_arrow,
                        keep,
                        budget,
                    )
            else:
                raise ValueError("asset paths must name CSV or Parquet files")
        else:
            assert self._query is not None
            offline = scope.runtime.cache.mode is CacheMode.OFFLINE
            if offline and scope.runtime._connection is not None:
                raise CapabilityError(
                    "Offline SQL assets require an SDK-owned DuckDB connection. "
                    "Pass a table or local file from your borrowed connection instead."
                )
            connection = scope.connection
            statements = connection.extract_statements(self._query)
            if (
                len(statements) != 1
                or statements[0].type != duckdb.StatementType.SELECT
            ):
                raise ValueError("Assets.sql accepts one SELECT query")
            try:
                with connection.sql(self._query).to_arrow_reader(
                    batch_size=scope.runtime.batch_rows
                ) as reader:
                    keep = self._input_columns(reader.schema, required)
                    table = self._collect(reader, reader.schema, keep, budget)
            except duckdb.PermissionException as error:
                if not offline:
                    raise
                raise CapabilityError(
                    "Offline SQL assets cannot access external resources. "
                    "Supply an owned asset table or a local asset file."
                ) from error
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            snapshot = sink.getvalue()
            table = pa.ipc.open_stream(snapshot).read_all()
            query_digest = hashlib.sha256(self._query.encode()).hexdigest()
            source = SourceIdentity(
                provider="user",
                product="sql_assets",
                resources=(
                    ResourceIdentity(
                        f"duckdb:query:{query_digest}",
                        "sha256:" + hashlib.sha256(snapshot).hexdigest(),
                        retrieved_at=datetime.now(UTC),
                    ),
                ),
            )
        if len(set(table.column_names)) != len(table.column_names):
            raise ValueError("asset column names must be unique")
        if self._id_column not in table.column_names:
            raise ValueError(f"missing asset id column {self._id_column!r}")
        ids = table[self._id_column].to_pylist()
        if any(
            not isinstance(value, (str, int)) or isinstance(value, bool)
            for value in ids
        ):
            raise ValueError("asset IDs must be non-null strings or integers")
        for value in ids:
            if isinstance(value, str):
                _identifier(value, "asset ID")
        if len(set(ids)) != len(ids):
            raise ValueError("asset IDs must be unique")
        location = _location(table.schema, self._location)
        geometry: list[bytes] = []
        if isinstance(location, PointColumns):
            for longitude, latitude in zip(
                table[location.longitude].to_pylist(),
                table[location.latitude].to_pylist(),
                strict=True,
            ):
                if any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float, Decimal))
                    or not math.isfinite(v)
                    for v in (longitude, latitude)
                ) or not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                    raise ValueError(
                        "asset locations require finite WGS84 longitude/latitude"
                    )
                geometry.append(bytes(to_wkb(Point(longitude, latitude))))
        elif isinstance(location, CellColumn):
            for value in table[location.name].to_pylist():
                cell = _cell(value)
                geometry.append(bytes(to_wkb(cell_polygon(cell))))
        else:
            _geometry_crs(table.schema, location.name)
            for value in table[location.name].to_pylist():
                if not isinstance(value, bytes):
                    raise ValueError("asset geometry must contain WKB bytes")
                _wkb_2d(value)
                try:
                    shape = from_wkb(value)
                except (GEOSException, ValueError) as error:
                    raise ValueError("asset geometry must contain valid WKB") from error
                if (
                    shape.is_empty
                    or not shape.is_valid
                    or shape.has_z
                    or not np.isfinite(get_coordinates(shape)).all()
                    or not (
                        -180 <= shape.bounds[0] <= shape.bounds[2] <= 180
                        and -90 <= shape.bounds[1] <= shape.bounds[3] <= 90
                    )
                ):
                    raise ValueError(
                        "asset geometry must be valid two-dimensional WGS84 geometry"
                    )
                geometry.append(value)
        context = table.select(required)
        if self._columns is not None:
            table = table.select(self._retained_columns(table.schema))
        context_only = context.select(
            [name for name in required if name not in table.column_names]
        )
        if (
            budget is not None
            and table.nbytes + context_only.nbytes + sum(map(len, geometry)) > budget
        ):
            raise MemoryError(
                "Assets exceed Runtime(memory_limit=...). Reduce retained columns "
                "or increase the limit."
            )
        result = AssetData(
            table, context, self._id_column, location, tuple(geometry), source
        )
        scope._asset_results[key] = result
        return result
