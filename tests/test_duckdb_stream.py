from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk.connectors.duckdb import (
    ArrowBatchSource,
    DuckDBConnection,
    DuckDBStreamEngine,
)


def test_arrow_source_and_pipeline_remain_lazy(tmp_path: Path) -> None:
    reads: list[int] = []

    def batches() -> Any:
        reads.append(1)
        yield pa.record_batch([[1, 2, 3]], names=["value"])

    source = ArrowBatchSource(pa.schema([("value", pa.int64())]), batches)
    pipeline = (
        DuckDBStreamEngine.from_source(
            source,
            connection=DuckDBConnection.for_analytics(tmp_path, extensions=()),
        )
        .where("value >= 2")
        .select("value, value * 10 AS scaled")
    )

    assert reads == []
    assert pipeline.relation().fetchall() == [(2, 20), (3, 30)]
    assert reads == [1]


def test_arrow_source_default_connection_is_resource_tuned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRC_DUCKDB_WORK_DIR", str(tmp_path))
    source = ArrowBatchSource(
        pa.schema([("value", pa.int64())]),
        lambda: iter([pa.record_batch([[1]], names=["value"])]),
    )

    assert source.pipeline().relation().fetchall() == [(1,)]
    assert (tmp_path / "duckdb-temp").is_dir()


def test_pipeline_streams_arrow_batches_and_parquet(tmp_path: Path) -> None:
    source = ArrowBatchSource(
        pa.schema([("value", pa.int64())]),
        lambda: iter(
            [
                pa.record_batch([[1, 2]], names=["value"]),
                pa.record_batch([[3, 4]], names=["value"]),
            ]
        ),
    )
    pipeline = source.pipeline(
        connection=DuckDBConnection.for_analytics(tmp_path, extensions=())
    )

    reader = pipeline.to_arrow_reader(batch_rows=2)
    assert [batch.num_rows for batch in reader] == [2, 2]

    output = pipeline.write_parquet(tmp_path / "values.parquet")
    assert output.is_file()


def test_pipeline_aggregate_retains_group_columns(tmp_path: Path) -> None:
    source = ArrowBatchSource(
        pa.schema([("kind", pa.string()), ("value", pa.int64())]),
        lambda: iter(
            [pa.record_batch([["a", "a", "b"], [1, 2, 3]], names=["kind", "value"])]
        ),
    )
    rows = (
        source.pipeline(
            connection=DuckDBConnection.for_analytics(tmp_path, extensions=())
        )
        .aggregate("sum(value) AS total", groups="kind")
        .order_by("kind")
        .relation()
        .fetchall()
    )
    assert rows == [("a", 3), ("b", 3)]


@pytest.mark.xfail(
    reason="DuckDB 1.5.5 returns no rows for windows over Arrow batch readers",
    strict=True,
)
def test_pipeline_window_over_arrow_batches(tmp_path: Path) -> None:
    source = ArrowBatchSource(
        pa.schema([("kind", pa.string()), ("value", pa.int64())]),
        lambda: iter(
            [pa.record_batch([["a", "a", "b"], [1, 2, 3]], names=["kind", "value"])]
        ),
    )
    rows = (
        source.pipeline(
            connection=DuckDBConnection.for_analytics(tmp_path, extensions=())
        )
        .select("*, sum(value) OVER (PARTITION BY kind) AS total")
        .order_by("kind, value")
        .relation()
        .fetchall()
    )
    assert rows == [("a", 1, 3), ("a", 2, 3), ("b", 3, 3)]


def test_pipeline_rejects_non_source() -> None:
    try:
        DuckDBStreamEngine.from_source(object())
    except TypeError as error:
        assert "relation" in str(error)
    else:
        raise AssertionError("expected a TypeError")
