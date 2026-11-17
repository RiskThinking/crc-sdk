"""Canonical scientific dataset validation and storage primitives."""

from .dataset import (
    DatasetRow,
    dataset_arrow_schema,
    dataset_from_json,
    dataset_to_json,
    metadata_from_table,
    read_parquet,
    rows_to_table,
    table_to_rows,
    validate_rows,
    write_parquet,
)

__all__ = [
    "DatasetRow",
    "dataset_arrow_schema",
    "dataset_from_json",
    "dataset_to_json",
    "metadata_from_table",
    "read_parquet",
    "rows_to_table",
    "table_to_rows",
    "validate_rows",
    "write_parquet",
]
