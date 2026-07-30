"""Pre-flight core/disk budget for one tippecanoe tiling pass.

Generalizes gen_pmtiles_v2's ``polygon_shards.py`` scratch-size math into a
pass/fail check, not a shard planner: :func:`check_tiling_budget` raises
before spawning tippecanoe if the estimated scratch requirement exceeds what's
available, rather than silently degrading into a slower shard-then-merge
fallback. An oversized source is the operator's problem to resolve --
provision more disk, or narrow the run's scope -- not something this
primitive works around automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crc_sdk.connectors.duckdb import (
    DuckDBConnection,
    default_work_dir,
    detected_cpu_count,
    sql_quote,
)

_GIB = 1024**3

# Conservative observation from large building-footprint tiling runs:
# tippecanoe's own sorted geometry/index scratch can be around an order of
# magnitude larger than the compressed GeoParquet input it's fed. This is an
# execution guard, not a product knob -- first calibrated in
# gen_pmtiles_v2/polygon_shards.py, reused here unchanged.
DEFAULT_TEMP_TO_INPUT_FACTOR = 14
_DEFAULT_SCRATCH_FRACTION = 0.25
_MIN_SAFE_SCRATCH_BYTES = 2 * _GIB


@dataclass(frozen=True)
class TilingBudget:
    """Detected core count and safe scratch-disk budget for a tiling pass."""

    tippecanoe_threads: int
    duckdb_threads: int
    free_disk_bytes: int
    safe_scratch_bytes: int

    @classmethod
    def detect(
        cls,
        work_dir: str | Path | None = None,
        *,
        tippecanoe_threads: int | None = None,
        duckdb_threads: int | None = None,
        scratch_fraction: float = _DEFAULT_SCRATCH_FRACTION,
    ) -> TilingBudget:
        """Detect available cores and scratch disk for one tiling pass.

        Threads default to every detected CPU for *both* tippecanoe and
        DuckDB -- unlike DuckDB's own GEOS-throttled default elsewhere in
        this SDK, tippecanoe's tile-building has no documented per-thread
        memory ceiling, and conservatively capping it (gen_pmtiles_v2's prior
        ``min(6, ...)``-style default) is exactly what produced its observed
        one-third-utilization, day-long tiling runs. Override via
        ``CRC_TIPPECANOE_THREADS``/``CRC_DUCKDB_THREADS``, or the keyword
        arguments here.
        """
        root = Path(work_dir) if work_dir is not None else default_work_dir()
        root.mkdir(parents=True, exist_ok=True)
        cpus = detected_cpu_count()
        free_disk_bytes = _disk_free_bytes(root)
        safe_scratch = max(
            _MIN_SAFE_SCRATCH_BYTES, int(free_disk_bytes * scratch_fraction)
        )
        threads = tippecanoe_threads or _env_int("CRC_TIPPECANOE_THREADS", cpus, cpus)
        db_threads = duckdb_threads or _env_int("CRC_DUCKDB_THREADS", cpus, cpus)
        return cls(
            tippecanoe_threads=threads,
            duckdb_threads=db_threads,
            free_disk_bytes=free_disk_bytes,
            safe_scratch_bytes=safe_scratch,
        )


def measure_source(
    source: str | Path,
    *,
    con: Any | None = None,
    work_dir: str | Path | None = None,
) -> tuple[int, int]:
    """Return ``(total_bytes, feature_count)`` for a GeoParquet file or glob.

    Byte size is measured via ``fsspec`` (uniform across local/``s3://``/
    ``gs://``); feature count is a ``count(*)`` through DuckDB's own
    ``read_parquet`` (which needs no extension for plain Parquet metadata),
    reusing the caller's connection if one is already open and configured
    with remote credentials, or building a default one otherwise.
    """
    owns_connection = con is None
    connection = con or DuckDBConnection.for_analytics(
        work_dir or default_work_dir(), extensions=("httpfs",)
    ).connect()
    try:
        source_ref = f"read_parquet({sql_quote(str(source))})"
        row = connection.execute(f"SELECT count(*) FROM {source_ref}").fetchone()
        feature_count = int(row[0]) if row else 0
        return _source_bytes(str(source)), feature_count
    finally:
        if owns_connection:
            connection.close()


def check_tiling_budget(
    input_bytes: int,
    feature_count: int,
    budget: TilingBudget,
    *,
    temp_to_input_factor: int = DEFAULT_TEMP_TO_INPUT_FACTOR,
) -> None:
    """Raise ``ValueError`` if tiling ``input_bytes`` likely exceeds the budget.

    A pure function over already-measured numbers -- no I/O -- so it's
    testable without mocking anything. The message states the estimated
    scratch requirement, what's actually available, and the two levers an
    operator has (more disk, or a narrower run), rather than attempting to
    auto-shard.
    """
    estimated_scratch_bytes = input_bytes * temp_to_input_factor
    if estimated_scratch_bytes <= budget.safe_scratch_bytes:
        return
    raise ValueError(
        "tiling this source needs an estimated ~"
        f"{estimated_scratch_bytes / _GIB:.1f} GiB of scratch disk "
        f"({input_bytes / _GIB:.2f} GiB input x {temp_to_input_factor}), but "
        f"only ~{budget.safe_scratch_bytes / _GIB:.1f} GiB is considered safe "
        f"to use (~{budget.free_disk_bytes / _GIB:.1f} GiB free) -- provision "
        "more disk, or narrow this run's scope (currently "
        f"{feature_count:,} features)."
    )


def _disk_free_bytes(path: Path) -> int:
    import shutil as _shutil

    return int(_shutil.disk_usage(path).free)


def _source_bytes(source: str) -> int:
    import fsspec  # type: ignore[import-untyped]

    filesystem, path = fsspec.core.url_to_fs(source)
    matches = filesystem.glob(path) if any(ch in path for ch in "*?[") else [path]
    if not matches:
        raise FileNotFoundError(f"no files matched source: {source!r}")
    return sum(int(filesystem.info(match)["size"]) for match in matches)


def _env_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(1, min(int(raw), maximum))
    except ValueError:
        return default
