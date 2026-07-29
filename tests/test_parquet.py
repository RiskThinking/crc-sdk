from pathlib import Path

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import HurdleDistribution

from crc_sdk.connectors.parquet import (
    hazard_arrow_schema,
    read_hazard_dataset,
    read_hazard_metadata,
    validate_hazard_table,
    write_hazard_dataset,
)
from crc_sdk.providers import LocalProvider
from crc_sdk.types import (
    PARQUET_METADATA_KEY,
    CurveParameters,
    HazardDatasetMetadata,
    HazardQuery,
    SourceProvenance,
)


def _metadata() -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="metres",
        value_semantics="flood depth",
        producer="tests",
        source=SourceProvenance(provider="fixture", dataset="flood"),
        creation_version="1",
    )


def _table() -> pa.Table:
    rows = [
        {
            "cell_index": 2,
            "source_id": "source-b",
            "source_geometry": None,
            "hazard_name": "flood",
            "horizon": 2050,
            "pathway": "ssp585",
            "curve_kind": "hurdle",
            "curve_type": "gumbel_r",
            "curve_shape": None,
            "curve_location": 2.0,
            "curve_scale": 3.0,
            "curve_atom_probability": 0.5,
            "curve_atom_location": 0.0,
        },
        {
            "cell_index": 1,
            "source_id": "source-a",
            "source_geometry": None,
            "hazard_name": "flood",
            "horizon": 2030,
            "pathway": "ssp245",
            "curve_kind": "fitted",
            "curve_type": "genextreme",
            "curve_shape": 0.1,
            "curve_location": 1.0,
            "curve_scale": 2.0,
            "curve_atom_probability": None,
            "curve_atom_location": None,
        },
    ]
    return pa.Table.from_pylist(rows, schema=hazard_arrow_schema())


def test_parquet_round_trip_sorts_filters_and_reconstructs(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    destination = tmp_path / "custom hazard '2050'.parquet"
    connection = duckdb.connect()
    written = write_hazard_dataset(
        _table(),
        destination,
        metadata,
        connection=connection,
    )

    assert written == destination
    assert not (tmp_path / "manifest.json").exists()
    assert set(pq.read_schema(destination).metadata or {}) == {
        PARQUET_METADATA_KEY.encode()
    }
    assert read_hazard_metadata(destination) == metadata
    table = read_hazard_dataset(
        destination,
        HazardQuery(hazard_name="flood", horizon=2050, pathway="ssp585"),
    )
    assert table["cell_index"].to_pylist() == [2]
    parameters = CurveParameters(
        curve_kind=table["curve_kind"][0].as_py(),
        curve_type=table["curve_type"][0].as_py(),
        curve_shape=table["curve_shape"][0].as_py(),
        curve_location=table["curve_location"][0].as_py(),
        curve_scale=table["curve_scale"][0].as_py(),
        curve_atom_probability=table["curve_atom_probability"][0].as_py(),
        curve_atom_location=table["curve_atom_location"][0].as_py(),
    )
    assert isinstance(parameters.to_distribution(), HurdleDistribution)
    assert LocalProvider(destination).list_hazards() == ("flood",)


def test_validation_rejects_duplicate_row_keys() -> None:
    table = _table()
    duplicate = pa.concat_tables([table.slice(0, 1), table.slice(0, 1)])

    with pytest.raises(ValueError, match="duplicate canonical row key"):
        validate_hazard_table(duplicate)


def test_validation_rejects_empty_required_strings() -> None:
    rows = _table().to_pylist()
    rows[0]["hazard_name"] = ""
    table = pa.Table.from_pylist(rows, schema=hazard_arrow_schema())

    with pytest.raises(ValueError, match="hazard_name must contain non-empty strings"):
        validate_hazard_table(table)


def _many_rows(count: int) -> pa.Table:
    rows = [
        {
            "cell_index": index,
            "source_id": f"source-{index}",
            "source_geometry": None,
            "hazard_name": "flood",
            "horizon": 1980,
            "pathway": "historical",
            "curve_kind": "fitted",
            "curve_type": "gumbel_r",
            "curve_shape": None,
            "curve_location": float(index % 100) / 10.0,
            "curve_scale": 2.0,
            "curve_atom_probability": None,
            "curve_atom_location": None,
        }
        for index in range(count)
    ]
    return pa.Table.from_pylist(rows, schema=hazard_arrow_schema())


def test_validation_parallel_matches_sequential_above_chunk_threshold() -> None:
    table = _many_rows(50)

    sequential = validate_hazard_table(table, max_workers=1)
    parallel = validate_hazard_table(table, max_workers=2, chunk_rows=10)

    assert sequential.equals(parallel)
    assert sequential.num_rows == 50


def test_validation_still_raises_when_parallelized() -> None:
    rows = _many_rows(50).to_pylist()
    rows[25]["curve_scale"] = -1.0  # invalid: scale must be positive
    table = pa.Table.from_pylist(rows, schema=hazard_arrow_schema())

    with pytest.raises(Exception):  # noqa: B017 - Pydantic ValidationError, cross-process
        validate_hazard_table(table, max_workers=2, chunk_rows=10)


def test_write_hazard_dataset_default_connection_is_resource_tuned(
    tmp_path: Path,
) -> None:
    """No explicit connection still gets RuntimeResources tuning applied."""
    destination = tmp_path / "tuned.parquet"
    write_hazard_dataset(_table(), destination, _metadata())

    assert destination.exists()
    # DuckDBConnection.for_analytics(work_dir) stages its spill directory
    # under work_dir/duckdb-temp; its presence confirms the tuned path ran
    # rather than a bare duckdb.connect().
    assert (tmp_path / "duckdb-temp").is_dir()
