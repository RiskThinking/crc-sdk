"""Validated Arrow and Parquet I/O for canonical hazard datasets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from crc_sdk.schema import HAZARD_FIELDS, HAZARD_ROW_KEY, HAZARD_SORT_ORDER
from crc_sdk.types import (
    PARQUET_METADATA_KEY,
    CurveParameters,
    HazardDatasetMetadata,
    HazardQuery,
)


def _pyarrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.dataset as ds  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "Parquet support requires `pip install crc-sdk[connectors]`"
        ) from error
    return pa, ds, pq


def hazard_arrow_schema(
    metadata: HazardDatasetMetadata | None = None,
) -> Any:
    """Return the canonical physical hazard-row schema."""
    pa, _, _ = _pyarrow()
    types = {
        "uint64": pa.uint64(),
        "binary": pa.binary(),
        "string": pa.string(),
        "int32": pa.int32(),
        "float64": pa.float64(),
    }
    schema = pa.schema(
        [
            pa.field(field.name, types[field.data_type], nullable=field.nullable)
            for field in HAZARD_FIELDS
        ]
    )
    if metadata is not None:
        schema = schema.with_metadata(metadata.to_parquet_metadata())
    return schema


def _as_table(value: Any) -> Any:
    pa, _, _ = _pyarrow()
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if isinstance(value, pa.RecordBatchReader):
        return value.read_all()
    if isinstance(value, Iterable):
        return pa.Table.from_batches(list(value))
    raise TypeError("hazard data must be an Arrow table, batch, reader, or batches")


def validate_hazard_table(
    value: Any,
    *,
    metadata: HazardDatasetMetadata | None = None,
    require_unique_keys: bool = True,
) -> Any:
    """Cast and validate canonical columns, curves, nullability, and row keys."""
    table = _as_table(value)
    expected = hazard_arrow_schema(metadata)
    expected_names = [field.name for field in HAZARD_FIELDS]
    if table.column_names != expected_names:
        raise ValueError(
            f"canonical hazard columns must be exactly {expected_names!r}"
        )
    try:
        table = table.cast(expected, safe=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"canonical hazard column types are invalid: {error}"
        ) from error

    for field in HAZARD_FIELDS:
        if not field.nullable and table[field.name].null_count:
            raise ValueError(f"{field.name} must not contain null values")

    string_columns = (
        "source_id",
        "hazard_name",
        "pathway",
        "curve_kind",
        "curve_type",
    )
    for name in string_columns:
        if any(not value for value in table[name].to_pylist()):
            raise ValueError(f"{name} must contain non-empty strings")

    curves = zip(
        table["curve_kind"].to_pylist(),
        table["curve_type"].to_pylist(),
        table["curve_shape"].to_pylist(),
        table["curve_location"].to_pylist(),
        table["curve_scale"].to_pylist(),
        table["curve_atom_probability"].to_pylist(),
        table["curve_atom_location"].to_pylist(),
    )
    for (
        kind,
        curve_type,
        shape,
        location,
        scale,
        atom_probability,
        atom_location,
    ) in curves:
        CurveParameters(
            curve_kind=kind,
            curve_type=curve_type,
            curve_shape=shape,
            curve_location=location,
            curve_scale=scale,
            curve_atom_probability=atom_probability,
            curve_atom_location=atom_location,
        )

    if require_unique_keys:
        key_columns = [table[name].to_pylist() for name in HAZARD_ROW_KEY]
        keys = list(zip(*key_columns))
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate canonical row key {HAZARD_ROW_KEY!r}")
    return table


def sort_hazard_table(value: Any) -> Any:
    """Return a table in the canonical filter and merge-join order."""
    table = _as_table(value)
    return table.sort_by([(name, "ascending") for name in HAZARD_SORT_ORDER])


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _duckdb_connection(connection: Any | None) -> tuple[Any, bool]:
    if connection is not None:
        if not hasattr(connection, "execute") or not hasattr(connection, "register"):
            raise TypeError("connection must be a DuckDBPyConnection")
        return connection, False
    try:
        import duckdb
    except ImportError as error:
        raise ImportError(
            "Parquet writes require `pip install crc-sdk[connectors]`"
        ) from error
    return duckdb.connect(), True


def write_hazard_dataset(
    value: Any,
    destination: str | Path,
    metadata: HazardDatasetMetadata,
    *,
    connection: Any | None = None,
    compression: str = "zstd",
) -> str | Path:
    """Write one self-describing Parquet file through DuckDB."""
    normalized_compression = compression.lower()
    supported_compression = {
        "uncompressed",
        "snappy",
        "gzip",
        "zstd",
        "lz4_raw",
    }
    if normalized_compression not in supported_compression:
        raise ValueError(
            f"compression must be one of {sorted(supported_compression)!r}"
        )
    table = validate_hazard_table(value, metadata=metadata)
    table = sort_hazard_table(table)
    target = str(destination)
    relation_name = f"_crc_hazard_{uuid4().hex}"
    duckdb_connection, owned = _duckdb_connection(connection)
    metadata_json = metadata.to_json_bytes().decode("utf-8")
    copy_sql = (
        f"COPY {_sql_identifier(relation_name)} TO {_sql_string(target)} "
        f"(FORMAT PARQUET, COMPRESSION {normalized_compression.upper()}, "
        f"KV_METADATA "
        f"{{{_sql_string(PARQUET_METADATA_KEY)}: "
        f"{_sql_string(metadata_json)}}})"
    )
    try:
        duckdb_connection.register(relation_name, table)
        duckdb_connection.execute(copy_sql)
    finally:
        try:
            duckdb_connection.unregister(relation_name)
        finally:
            if owned:
                duckdb_connection.close()
    return destination


def write_hazard_stream(
    stream: Any,
    destination: str | Path,
    *,
    connection: Any | None = None,
    compression: str = "zstd",
) -> str | Path:
    """Consume a canonical stream and write the hazard dataset."""
    return write_hazard_dataset(
        stream.read_all(),
        destination,
        stream.metadata,
        connection=connection,
        compression=compression,
    )


def read_hazard_metadata(source: str | Path) -> HazardDatasetMetadata:
    """Read the complete canonical payload from Parquet key-value metadata."""
    _, _, pq = _pyarrow()
    parquet_schema = pq.read_schema(source)
    embedded = HazardDatasetMetadata.from_parquet_metadata(
        parquet_schema.metadata
    )
    expected = hazard_arrow_schema()
    if parquet_schema.names != expected.names or any(
        actual.type != wanted.type
        for actual, wanted in zip(parquet_schema, expected)
    ):
        raise ValueError("Parquet file does not use the canonical hazard schema")
    return embedded


def read_hazard_dataset(
    source: str | Path,
    query: HazardQuery | None = None,
    *,
    columns: list[str] | None = None,
) -> Any:
    """Read a validated canonical table with predicate pushdown."""
    _, ds, _ = _pyarrow()
    metadata = read_hazard_metadata(source)
    expression = None
    if query is not None:
        expression = ds.field("hazard_name") == query.hazard_name
        if query.horizon is not None:
            expression = expression & (ds.field("horizon") == query.horizon)
        if query.pathway is not None:
            expression = expression & (ds.field("pathway") == query.pathway)
    table = ds.dataset(source, format="parquet").to_table(
        columns=columns,
        filter=expression,
    )
    if columns is None:
        return validate_hazard_table(
            table,
            metadata=metadata,
            require_unique_keys=True,
        )
    return table
