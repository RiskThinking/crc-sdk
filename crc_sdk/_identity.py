"""Exact row-key uniqueness and support consistency over streams of any size."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
from shapely import from_wkb  # type: ignore[import-untyped]

from ._graph import ExecutionScope
from ._runtime import _memory_bytes

# Streams up to this many rows keep their keys in Python sets and fail on the
# batch that repeats a key. Longer streams move them to a scratch database.
_IN_MEMORY_ROWS = 1 << 15
# The scratch database follows a smaller runtime limit down to what DuckDB needs.
_DATABASE_MEMORY = 128 * 1024 * 1024
_DATABASE_MEMORY_FLOOR = 16 * 1024 * 1024
SUPPORT_COLUMNS = ("support_id", "geometry", "cell_id")
_SUPPORTS = pa.schema(
    [
        pa.field("support_id", pa.string()),
        pa.field("geometry", pa.binary()),
        pa.field("cell", pa.string()),
    ]
)

SupportValue = tuple[bytes | None, str | None]


def _rows(table: pa.Table) -> Iterator[tuple[object, ...]]:
    return zip(*(column.to_pylist() for column in table.columns), strict=True)


def _support_rows(table: pa.Table) -> Iterator[tuple[str, SupportValue]]:
    """`(support_id, (geometry, cell))`; a cell identifies only cell supports."""
    for support_id, geometry, cell_id in _rows(table):
        yield (
            str(support_id),
            (
                None if geometry is None else bytes(cast(bytes, geometry)),
                str(cell_id) if geometry is None else None,
            ),
        )


def _conflict(support_id: str, old: SupportValue, new: SupportValue) -> None:
    (old_geometry, old_cell), (geometry, cell) = old, new
    if (
        old_cell != cell
        or old_geometry is None
        or geometry is None
        or not from_wkb(old_geometry).equals(from_wkb(geometry))
    ):
        raise ValueError(
            f"support_id {support_id!r} identifies conflicting source geometry"
        )


class RowIdentities:
    """Reject repeated row keys and support IDs naming different geometry.

    Keys of small streams stay in memory, and the batch that repeats one fails.
    Longer streams append keys and supports to a scratch DuckDB database in the
    execution directory, which stores them compressed and spills to disk under
    a fixed memory limit. `finish` then groups their 64-bit hashes and checks
    only the colliding rows exactly. Either way every key and support is
    compared exactly; only the time of failure differs.
    """

    def __init__(
        self, scope: ExecutionScope, *, in_memory_rows: int | None = None
    ) -> None:
        self._scope = scope
        self._limit = _IN_MEMORY_ROWS if in_memory_rows is None else in_memory_rows
        self._keys: set[tuple[object, ...]] = set()
        self._supports: dict[str, SupportValue] = {}
        self._key_schema: pa.Schema | None = None
        self._db: duckdb.DuckDBPyConnection | None = None
        self._files: tuple[Path, Path] | None = None

    def add(self, keys: pa.Table, supports: pa.Table) -> None:
        """Record one batch: key columns, and `support_id, geometry, cell_id`."""
        if self._key_schema is None:
            self._key_schema = keys.schema
        elif not keys.schema.equals(self._key_schema):
            raise TypeError("row keys changed their columns within one stream")
        if self._db is None and len(self._keys) + keys.num_rows <= self._limit:
            self._remember(_rows(keys), _support_rows(supports))
            return
        if self._db is None:
            self._spill()
        assert self._db is not None
        self._db.register("batch_keys", keys)
        self._db.register("batch_supports", supports)
        try:
            self._db.execute("INSERT INTO keys SELECT * FROM batch_keys")
            self._db.execute(
                "INSERT INTO supports SELECT DISTINCT support_id, geometry, "
                "CASE WHEN geometry IS NULL THEN CAST(cell_id AS VARCHAR) END "
                "FROM batch_supports"
            )
        finally:
            self._db.unregister("batch_keys")
            self._db.unregister("batch_supports")

    def finish(self) -> None:
        """Raise for a repeated key or conflicting support not yet reported.

        The scratch database is removed afterwards, even when this raises.
        """
        if self._db is None:
            return
        try:
            self._check_database()
        finally:
            self._release()

    def _check_database(self) -> None:
        assert self._db is not None
        columns = ", ".join(
            '"' + name.replace('"', '""') + '"'
            for name in (self._key_schema or pa.schema([])).names
        )
        # Rows whose key hashes collide are the only possible repeats.
        candidates = self._db.execute(
            f"SELECT {columns} FROM keys WHERE hash({columns}) IN ("
            f"SELECT h FROM (SELECT hash({columns}) AS h FROM keys) "
            "GROUP BY h HAVING count(*) > 1)"
        )
        self._remember(self._fetch(candidates), (), memory=False)
        # Supports listed with more than one value are the only possible
        # conflicts; equal geometry in another WKB encoding is not one.
        conflicts = self._db.execute(
            "SELECT support_id, geometry, cell FROM supports "
            "WHERE hash(support_id) IN ("
            "SELECT hash(support_id) AS h FROM supports GROUP BY h "
            "HAVING min(hash(geometry, cell)) <> max(hash(geometry, cell)))"
        )
        supports: dict[str, SupportValue] = {}
        for support_id, geometry, cell in self._fetch(conflicts):
            value = (geometry, cell)
            old = supports.setdefault(str(support_id), value)  # type: ignore[arg-type]
            if old != value:
                _conflict(str(support_id), old, value)  # type: ignore[arg-type]

    def _release(self) -> None:
        db, self._db = self._db, None
        if db is None:
            return
        db.close()
        assert self._files is not None
        database, spill = self._files
        database.unlink(missing_ok=True)
        Path(f"{database}.wal").unlink(missing_ok=True)
        shutil.rmtree(spill, ignore_errors=True)

    def _remember(
        self,
        keys: Iterable[tuple[object, ...]],
        supports: Iterable[tuple[str, SupportValue]],
        *,
        memory: bool = True,
    ) -> None:
        seen = self._keys if memory else set()
        for key in keys:
            if key in seen:
                raise ValueError(f"duplicate scientific row key: {self._text(key)}")
            seen.add(key)
        for support_id, value in supports:
            old = self._supports.setdefault(support_id, value)
            if old != value:
                _conflict(support_id, old, value)

    @staticmethod
    def _text(key: tuple[object, ...]) -> str:
        # Evidence keys arrive as one canonical JSON text; dataset keys as columns.
        if len(key) == 1 and isinstance(key[0], str):
            return key[0]
        return json.dumps(list(key), ensure_ascii=False)

    @staticmethod
    def _fetch(
        cursor: duckdb.DuckDBPyConnection,
    ) -> Iterator[tuple[object, ...]]:
        while rows := cursor.fetchmany(4096):
            yield from rows

    def _spill(self) -> None:
        assert self._key_schema is not None
        directory = self._scope.directory
        name = f"identity-{uuid.uuid4().hex}"
        limit = _memory_bytes(self._scope.runtime.memory_limit) or _DATABASE_MEMORY
        memory = max(min(limit, _DATABASE_MEMORY), _DATABASE_MEMORY_FLOOR)
        db = duckdb.connect(
            str(directory / f"{name}.duckdb"),
            config={
                "threads": 1,
                "memory_limit": f"{memory}B",
                "temp_directory": str(directory / name),
                "preserve_insertion_order": False,
            },
        )
        self._db = db
        self._files = (directory / f"{name}.duckdb", directory / name)
        self._scope.callback(self._release)
        keys = pa.Table.from_pylist(
            [dict(zip(self._key_schema.names, key, strict=True)) for key in self._keys],
            schema=self._key_schema,
        )
        supports = pa.Table.from_pylist(
            [
                {"support_id": support_id, "geometry": geometry, "cell": cell}
                for support_id, (geometry, cell) in self._supports.items()
            ],
            schema=_SUPPORTS,
        )
        db.register("held_keys", keys)
        db.register("held_supports", supports)
        try:
            db.execute("CREATE TABLE keys AS SELECT * FROM held_keys")
            db.execute("CREATE TABLE supports AS SELECT * FROM held_supports")
        finally:
            db.unregister("held_keys")
            db.unregister("held_supports")
        self._keys = set()
        self._supports = {}
