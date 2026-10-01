"""Row-level fit diagnostics sidecar (ADR-0007)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fsspec
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

DIAGNOSTICS_SCHEMA = pa.schema(
    [
        pa.field("cell_index", pa.uint64()),
        pa.field("source_id", pa.string()),
        pa.field("hazard_name", pa.string()),
        pa.field("horizon", pa.int32()),
        pa.field("pathway", pa.string()),
        pa.field("outcome", pa.string()),
        pa.field("curve_type", pa.string()),
        pa.field("normalized_rmse", pa.float64()),
        pa.field("maximum_absolute_residual", pa.float64()),
        pa.field("attempted_families", pa.list_(pa.string())),
        pa.field("failed_families", pa.list_(pa.string())),
        pa.field("fallback", pa.bool_()),
        pa.field("reason", pa.string()),
        pa.field("message", pa.string()),
        pa.field("treatment", pa.string()),
        pa.field("minimum_informative_value", pa.float64()),
    ]
)


class DiagnosticsWriter:
    """Append diagnostics batches to a partial file, publish on ``finish``.

    A half-written sidecar is never mistaken for a result: rows go to
    ``<path>.partial`` and are moved into place only by :meth:`finish`.
    """

    def __init__(self, destination: str | Path) -> None:
        self.destination = str(destination)
        self._fs, self._path = fsspec.core.url_to_fs(self.destination)
        self._partial = f"{self._path}.partial"
        parent = str(Path(self._path).parent)
        self._fs.makedirs(parent, exist_ok=True)
        self._sink = self._fs.open(self._partial, "wb")
        self._writer = pq.ParquetWriter(
            self._sink, DIAGNOSTICS_SCHEMA, compression="zstd"
        )
        self.rows = 0
        self._closed = False

    def write(self, records: Sequence[dict[str, Any]]) -> None:
        if not records:
            return
        self._writer.write_table(
            pa.Table.from_pylist(list(records), DIAGNOSTICS_SCHEMA)
        )
        self.rows += len(records)

    def _close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._writer.close()
            finally:
                self._sink.close()

    def finish(self) -> str:
        self._close()
        if self._fs.protocol in ("file", "local") or (
            isinstance(self._fs.protocol, tuple) and "file" in self._fs.protocol
        ):
            os.replace(self._partial, self._path)
        else:
            self._fs.mv(self._partial, self._path)
        return self.destination

    def abort(self) -> None:
        self._close()
        try:
            self._fs.rm(self._partial)
        except FileNotFoundError:
            pass
