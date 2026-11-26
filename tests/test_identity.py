"""Row keys and supports stay exact when a long stream checks them out of core."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import Tail
from shapely import Point, to_wkb  # type: ignore[import-untyped]

from crc_sdk import Dataset, Probability, Runtime, ValidationError, Variable, _identity
from crc_sdk._graph import ExecutionScope
from crc_sdk._identity import RowIdentities

_KEYS = pa.schema(
    [
        pa.field("variable_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("period_id", pa.string()),
        pa.field("support_id", pa.string()),
        pa.field("cell_id", pa.int64()),
    ]
)
_SUPPORTS = pa.schema(
    [
        pa.field("support_id", pa.string()),
        pa.field("geometry", pa.binary()),
        pa.field("cell_id", pa.int64()),
    ]
)


def _batch(
    supports: list[str],
    *,
    cells: list[int] | None = None,
    geometry: list[bytes | None] | None = None,
) -> tuple[pa.Table, pa.Table]:
    cell_ids = cells or [7] * len(supports)
    keys = pa.Table.from_pydict(
        {
            "variable_id": ["v"] * len(supports),
            "scenario_id": [None] * len(supports),
            "period_id": ["p"] * len(supports),
            "support_id": supports,
            "cell_id": cell_ids,
        },
        schema=_KEYS,
    )
    rows = pa.Table.from_pydict(
        {
            "support_id": supports,
            "geometry": geometry or [None] * len(supports),
            "cell_id": cell_ids,
        },
        schema=_SUPPORTS,
    )
    return keys, rows


def test_long_streams_move_keys_to_the_execution_directory() -> None:
    with ExecutionScope() as scope:
        identities = RowIdentities(scope, in_memory_rows=2)
        identities.add(*_batch(["a", "b"]))
        assert not list(scope.directory.glob("identity-*.duckdb"))
        identities.add(*_batch(["c", "d"]))
        identities.add(*_batch(["e"]))
        assert list(scope.directory.glob("identity-*.duckdb"))
        identities.finish()


def test_a_key_repeated_after_the_spill_fails_when_the_stream_ends() -> None:
    with ExecutionScope() as scope:
        identities = RowIdentities(scope, in_memory_rows=2)
        identities.add(*_batch(["a", "b"]))
        identities.add(*_batch(["c", "a"]))
        with pytest.raises(
            ValueError,
            match=r'duplicate scientific row key: \["v", null, "p", "a", 7\]',
        ):
            identities.finish()


def test_small_streams_fail_on_the_batch_that_repeats_a_key() -> None:
    with ExecutionScope() as scope:
        identities = RowIdentities(scope)
        identities.add(*_batch(["a", "b"]))
        with pytest.raises(ValueError, match="duplicate scientific row key"):
            identities.add(*_batch(["b"]))


def test_supports_are_compared_exactly_after_the_spill() -> None:
    point = Point(1.0, 2.0)
    little = to_wkb(point, byte_order=1)
    big = to_wkb(point, byte_order=0)
    other = to_wkb(Point(1.0, 3.0))
    with ExecutionScope() as scope:
        identities = RowIdentities(scope, in_memory_rows=1)
        identities.add(*_batch(["s"], geometry=[little]))
        # The same point in another byte order identifies the same support.
        identities.add(*_batch(["t", "s"], cells=[7, 8], geometry=[None, big]))
        identities.add(*_batch(["u"], cells=[9]))
        identities.finish()
    with ExecutionScope() as scope:
        identities = RowIdentities(scope, in_memory_rows=1)
        identities.add(*_batch(["s"], geometry=[little]))
        identities.add(*_batch(["t", "s"], cells=[7, 8], geometry=[None, other]))
        with pytest.raises(ValueError, match="'s' identifies conflicting source"):
            identities.finish()
    with ExecutionScope() as scope:
        identities = RowIdentities(scope, in_memory_rows=1)
        identities.add(*_batch(["s"], cells=[7]))
        identities.add(*_batch(["t", "s"], cells=[7, 8]))
        with pytest.raises(ValueError, match="'s' identifies conflicting source"):
            identities.finish()


def test_reads_reject_keys_repeated_across_spilled_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = (
        Dataset.from_quantiles(
            {
                "support_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
                "longitude": [6.96] * 3 + [7.1] * 3 + [7.2] * 3,
                "latitude": [50.94] * 9,
                "probability": [0.5, 0.9, 0.99] * 3,
                "value": [0.0, 0.3, 1.2] * 3,
            },
            variable=Variable("company:flood-depth"),
            probability=Probability.annual(tail=Tail.UPPER),
        )
        .read()
        .table
    )
    path = tmp_path / "rows.parquet"
    pq.write_table(rows, path)
    monkeypatch.setattr(_identity, "_IN_MEMORY_ROWS", 1)
    runtime = Runtime(batch_rows=1)
    assert Dataset.open(path).read(runtime=runtime).report.valid
    records = rows.to_pylist()
    records[2] = records[0]
    pq.write_table(pa.Table.from_pylist(records, schema=rows.schema), path)
    with pytest.raises(ValidationError, match="duplicate scientific row key"):
        Dataset.open(path).read(runtime=runtime)
