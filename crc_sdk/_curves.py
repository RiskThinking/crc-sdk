"""Framework objects for the numerical payload of canonical dataset rows."""

from __future__ import annotations

import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import Distribution, QuantileCurve

from ._schema.dataset import _curve_from_arrow


def decode_curves(
    rows: pa.Table | pa.RecordBatch,
) -> tuple[Distribution | QuantileCurve | None, ...]:
    """Return each row's `curve` as a framework object, or None for a failure row."""
    if not isinstance(rows, (pa.Table, pa.RecordBatch)):
        raise TypeError("rows must be a pyarrow Table or RecordBatch")
    if "curve" not in rows.schema.names:
        raise ValueError("rows have no curve column; pass Dataset 2.0 rows")
    return tuple(
        None if value is None else _curve_from_arrow(value)
        for value in rows.column("curve").to_pylist()
    )
