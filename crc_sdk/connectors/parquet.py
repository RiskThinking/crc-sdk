"""Validated Arrow and Parquet I/O for canonical hazard datasets."""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from crc_sdk.connectors.duckdb import DuckDBConnection, detected_cpu_count
from crc_sdk.schema import HAZARD_FIELDS, HAZARD_ROW_KEY, HAZARD_SORT_ORDER
from crc_sdk.types import (
    PARQUET_METADATA_KEY,
    CurveParameters,
    HazardDatasetMetadata,
    HazardQuery,
)


def hazard_arrow_schema(
    metadata: HazardDatasetMetadata | None = None,
) -> Any:
    """Return the canonical physical hazard-row schema."""
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
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if isinstance(value, pa.RecordBatchReader):
        return value.read_all()
    if isinstance(value, Iterable):
        return pa.Table.from_batches(list(value))
    raise TypeError("hazard data must be an Arrow table, batch, reader, or batches")


_CURVE_COLUMNS = (
    "curve_kind",
    "curve_type",
    "curve_shape",
    "curve_location",
    "curve_scale",
    "curve_atom_probability",
    "curve_atom_location",
)


def _validate_curves(records: Sequence[Mapping[str, Any]]) -> None:
    for row in records:
        CurveParameters(
            curve_kind=row["curve_kind"],
            curve_type=row["curve_type"],
            curve_shape=row["curve_shape"],
            curve_location=row["curve_location"],
            curve_scale=row["curve_scale"],
            curve_atom_probability=row["curve_atom_probability"],
            curve_atom_location=row["curve_atom_location"],
        )


def validate_hazard_table(
    value: Any,
    *,
    metadata: HazardDatasetMetadata | None = None,
    require_unique_keys: bool = True,
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> Any:
    """Cast and validate canonical columns, curves, nullability, and row keys.

    Every write and (unless a ``columns`` projection is requested) every read
    of a canonical hazard dataset runs this. The nullability, empty-string,
    and duplicate-key checks are pure Arrow/vectorized — no Python loop.
    Reconstructing each row's ``CurveParameters`` is the one check that
    cannot be vectorized (Pydantic validation plus a per-row Rust call, same
    as curve evaluation elsewhere in the SDK), so it's chunked across a
    process pool for tables above ``chunk_rows``; smaller tables validate
    in-process with no pool overhead.
    """
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
        if pc.any(pc.equal(pc.utf8_length(table[name]), 0)).as_py():
            raise ValueError(f"{name} must contain non-empty strings")

    curve_table = table.select(_CURVE_COLUMNS)
    records = curve_table.to_pylist()
    workers = max_workers or detected_cpu_count()
    if records:
        if workers <= 1 or len(records) <= chunk_rows:
            _validate_curves(records)
        else:
            chunks = [
                records[start : start + chunk_rows]
                for start in range(0, len(records), chunk_rows)
            ]
            mp_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp_context
            ) as executor:
                for _ in executor.map(_validate_curves, chunks):
                    pass

    if require_unique_keys:
        key_columns = list(HAZARD_ROW_KEY)
        distinct = table.select(key_columns).group_by(key_columns).aggregate([])
        if distinct.num_rows != table.num_rows:
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


def _duckdb_connection(
    connection: Any | None, *, work_dir: Path | None = None
) -> tuple[Any, bool]:
    """Return (connection, owned). An owned connection is resource-tuned.

    Without an explicit ``connection``, a bare ``duckdb.connect()`` would
    skip the SDK's own thread/memory detection entirely (DuckDB's own
    defaults aren't cgroup-aware and aren't tuned for this SDK's spatial
    workloads) — every out-of-the-box write should get that tuning for free,
    not only callers who remember to build a connection themselves.
    """
    if connection is not None:
        if not hasattr(connection, "execute") or not hasattr(connection, "register"):
            raise TypeError("connection must be a DuckDBPyConnection")
        return connection, False
    if work_dir is not None:
        return DuckDBConnection.for_analytics(work_dir, extensions=()).connect(), True
    return DuckDBConnection().connect(), True


def write_hazard_dataset(
    value: Any,
    destination: str | Path,
    metadata: HazardDatasetMetadata,
    *,
    connection: Any | None = None,
    compression: str = "zstd",
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> str | Path:
    """Write one self-describing Parquet file through DuckDB.

    ``max_workers``/``chunk_rows`` pass straight through to
    :func:`validate_hazard_table`'s curve-reconstruction pass. A caller
    already running inside its own worker process (e.g. one tile of
    :func:`crc_sdk.workflows.run_tiled_canonicalization`) should pass
    ``max_workers=1`` here — otherwise validation may open a nested
    ``ProcessPoolExecutor`` per call, fanning out ``tile_workers ×
    validation_workers`` processes instead of just using the outer pool.
    """
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
    table = validate_hazard_table(
        value, metadata=metadata, max_workers=max_workers, chunk_rows=chunk_rows
    )
    table = sort_hazard_table(table)
    target = str(destination)
    relation_name = f"_crc_hazard_{uuid4().hex}"
    duckdb_connection, owned = _duckdb_connection(
        connection, work_dir=Path(destination).parent
    )
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
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> str | Path:
    """Consume a canonical stream and write the hazard dataset.

    ``max_workers``/``chunk_rows`` pass through to
    :func:`write_hazard_dataset` — see its docstring re: nested process pools.
    """
    return write_hazard_dataset(
        stream.read_all(),
        destination,
        stream.metadata,
        connection=connection,
        compression=compression,
        max_workers=max_workers,
        chunk_rows=chunk_rows,
    )


def read_hazard_metadata(source: str | Path) -> HazardDatasetMetadata:
    """Read the complete canonical payload from Parquet key-value metadata."""
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
    metadata = read_hazard_metadata(source)
    expression = None
    if query is not None:
        expression = ds.field("hazard_name") == query.hazard_name
        if query.horizon is not None:
            expression = expression & (ds.field("horizon") == query.horizon)
        if query.pathway is not None:
            expression = expression & (ds.field("pathway") == query.pathway)
        if query.cell_index is not None:
            expression = expression & (
                ds.field("cell_index") == query.cell_index
            )
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
