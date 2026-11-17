"""Composable lazy sources for the DuckDB streaming process engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa  # type: ignore[import-untyped]

from .connection import DuckDBConnection, default_work_dir


@runtime_checkable
class DuckDBRelationSource(Protocol):
    """An adapter that can create a fresh lazy DuckDB relation.

    Existing raster scans satisfy this protocol structurally. New adapters can
    either implement ``relation`` directly (ideal for Parquet/SQL sources) or
    use :class:`ArrowBatchSource` to feed bounded record batches into DuckDB.
    """

    def relation(self, *, connection: DuckDBConnection | None = None) -> Any: ...


BatchFactory = Callable[[], Iterable[Any]]
RelationTransform = Callable[[Any], Any]


class _LockedBatchIterator:
    """Serialize pulls when DuckDB scans a Python Arrow source in parallel."""

    def __init__(self, batches: Iterable[Any]) -> None:
        self._iterator = iter(batches)
        self._lock = Lock()

    def __iter__(self) -> _LockedBatchIterator:
        return self

    def __next__(self) -> Any:
        with self._lock:
            return next(self._iterator)


@dataclass(frozen=True)
class ArrowBatchSource:
    """Reusable lazy bridge from a batch factory to a DuckDB relation."""

    schema: Any
    batches: BatchFactory
    connection: DuckDBConnection | None = None

    def relation(self, *, connection: DuckDBConnection | None = None) -> Any:
        reader = pa.RecordBatchReader.from_batches(
            self.schema, _LockedBatchIterator(self.batches())
        )
        active_config = connection or self.connection
        if active_config is None:
            active_config = DuckDBConnection.for_analytics(
                default_work_dir(), extensions=()
            )
        active = active_config.connect()
        return active.from_arrow(reader)

    def pipeline(self, *, connection: DuckDBConnection | None = None) -> DuckDBPipeline:
        return DuckDBPipeline(self, connection=connection)


@dataclass(frozen=True)
class DuckDBPipeline:
    """An immutable, lazily executed chain over any relation source."""

    source: DuckDBRelationSource
    connection: DuckDBConnection | None = None
    transforms: tuple[RelationTransform, ...] = ()

    def transform(self, operation: RelationTransform) -> DuckDBPipeline:
        """Append a relational operation without executing the source."""
        if not callable(operation):
            raise TypeError("operation must be callable")
        return replace(self, transforms=(*self.transforms, operation))

    def select(self, expression: str) -> DuckDBPipeline:
        return self.transform(lambda relation: relation.project(expression))

    def where(self, expression: str) -> DuckDBPipeline:
        return self.transform(lambda relation: relation.filter(expression))

    def aggregate(self, expression: str, groups: str = "") -> DuckDBPipeline:
        projection = f"{groups}, {expression}" if groups else expression
        return self.transform(lambda relation: relation.aggregate(projection, groups))

    def order_by(self, expression: str) -> DuckDBPipeline:
        return self.transform(lambda relation: relation.order(expression))

    def limit(self, count: int, *, offset: int = 0) -> DuckDBPipeline:
        if count < 0 or offset < 0:
            raise ValueError("count and offset must be non-negative")
        return self.transform(lambda relation: relation.limit(count, offset=offset))

    def relation(self) -> Any:
        relation = self.source.relation(connection=self.connection)
        for operation in self.transforms:
            relation = operation(relation)
        return relation

    def to_arrow_reader(self, batch_rows: int = 65_536) -> Any:
        """Execute once and stream bounded Arrow batches from DuckDB."""
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        return self.relation().to_arrow_reader(batch_rows)

    def write_parquet(
        self,
        destination: str | Path,
        *,
        compression: str = "zstd",
        overwrite: bool = False,
    ) -> Path:
        """Stream the pipeline result directly to Parquet through DuckDB."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.relation().write_parquet(
            str(path), compression=compression, overwrite=overwrite
        )
        return path
