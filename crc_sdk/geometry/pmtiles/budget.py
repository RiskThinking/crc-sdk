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

import math
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

# Directly measured, not assumed: tippecanoe's own scratch is almost entirely
# anonymous temporary files (created, then immediately unlinked, kept open
# via file descriptor -- confirmed by its maintainer and reproduced here), so
# `du` on its `-t` temp directory reports it as empty regardless of real
# usage; only the containing filesystem's free-space delta (`df`, polled
# during the run) reveals the true figure. Measured this way three times
# (145k lossless LUX/frost_days polygons, ~415 wide percentile-score
# properties each, 4 and 8 threads): 527MiB, 554MiB, 483MiB peak scratch
# against a ~22MiB compressed GeoParquet input -- a 22-26x ratio, higher
# than the previous ``14`` this replaces (an unvalidated estimate carried
# over unchanged from gen_pmtiles_v2/polygon_shards.py). Getting this
# constant too low is the dangerous direction -- a mid-run disk-exhaustion
# failure wastes all compute already spent and can leave a corrupt partial
# output, versus a too-high value only costing an operator a pre-flight
# rejection they can act on -- so this errs upward from the measured range
# rather than sitting at its low end. Still calibrated at one small AOI on
# one workload shape (lossless + very wide attributes); narrower/lossy
# schemas (e.g. the ``POINTS`` preset) very likely need proportionally less,
# but this stays one constant for every preset until there's comparable
# measured evidence to split it. Validating against a larger, still
# wide-attribute country is the concrete next step.
DEFAULT_TEMP_TO_INPUT_FACTOR = 30
_DEFAULT_SCRATCH_FRACTION = 0.25
_MIN_SAFE_SCRATCH_BYTES = 2 * _GIB


def resolve_temp_to_input_factor(explicit: float | None = None) -> float:
    """``explicit`` if given, else ``CRC_TIPPECANOE_TEMP_TO_INPUT_FACTOR``, else
    :data:`DEFAULT_TEMP_TO_INPUT_FACTOR` -- same precedence as
    :meth:`TilingBudget.detect`'s ``scratch_fraction`` resolution, kept as a
    small standalone function (rather than a bare env read inline at the
    call site) so other callers can resolve the same value consistently.
    """
    if explicit is not None:
        return explicit
    raw = os.getenv("CRC_TIPPECANOE_TEMP_TO_INPUT_FACTOR")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_TEMP_TO_INPUT_FACTOR


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
        scratch_fraction: float | None = None,
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

        ``scratch_fraction`` -- how much of free disk is considered safe to
        gamble on one tiling pass, independent of
        ``DEFAULT_TEMP_TO_INPUT_FACTOR``'s per-workload estimate -- follows
        the same precedence as the thread arguments: this keyword wins if
        given, else ``CRC_TIPPECANOE_SCRATCH_FRACTION``, else the conservative
        25% default. This default deliberately assumes nothing about the
        caller: no guarantee that nothing else is using this disk, and no
        guarantee that a previous or concurrent build's scratch has been
        cleaned up. A caller that *can* guarantee those things (e.g. one
        tiling pass at a time, each fully cleaned up before the next starts,
        as gen_pmtiles_v2's finalize stage does) has a real case for passing
        something looser here -- that's a property of the caller's own
        orchestration, not something this primitive can assume on its own.
        """
        root = Path(work_dir) if work_dir is not None else default_work_dir()
        root.mkdir(parents=True, exist_ok=True)
        cpus = detected_cpu_count()
        free_disk_bytes = _disk_free_bytes(root)
        fraction = (
            scratch_fraction
            if scratch_fraction is not None
            else _env_float(
                "CRC_TIPPECANOE_SCRATCH_FRACTION", _DEFAULT_SCRATCH_FRACTION
            )
        )
        safe_scratch = max(_MIN_SAFE_SCRATCH_BYTES, int(free_disk_bytes * fraction))
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
    connection = (
        con
        or DuckDBConnection.for_analytics(
            work_dir or default_work_dir(), extensions=("httpfs",)
        ).connect()
    )
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
    temp_to_input_factor: float = DEFAULT_TEMP_TO_INPUT_FACTOR,
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


def nearest_power_of_two(value: float) -> int:
    """Round ``value`` to the nearest power of 2 -- at least 1.

    tippecanoe's own ``init_cpus()`` unconditionally rounds whatever thread
    count it's given *down* to the nearest power of 2 (confirmed in its
    source), regardless of whether that count came from auto-detection or
    ``TIPPECANOE_MAX_THREADS``. That's harmless for a raw, untransformed core
    count -- tippecanoe was always going to floor it to the same bracket
    either way -- but it compounds badly the moment a caller scales or
    transforms the core count *before* choosing a thread target: half of a
    31-core budget is 15, which tippecanoe's own floor then takes down to 8 --
    a 4x cut from the real core count, not the intended 2x. Snapping to the
    nearest power of 2 *before* handing tippecanoe a computed target (e.g.
    ``nearest_power_of_two(cpu / 2)``) makes tippecanoe's own rounding a
    no-op instead of a second, compounding one -- for any transform, not just
    halving.
    """
    if value <= 1:
        return 1
    return 1 << round(math.log2(value))


def _disk_free_bytes(path: Path) -> int:
    import shutil as _shutil

    return int(_shutil.disk_usage(path).free)


def _source_bytes(source: str) -> int:
    import fsspec  # type: ignore[import-untyped]

    filesystem, path = fsspec.core.url_to_fs(source)
    if any(ch in path for ch in "*?["):
        # One bulk listing call (`detail=True` returns {path: info} directly)
        # instead of `glob()` (list) + one `.info()` round trip per match --
        # for a remote (s3://, gs://) Hive glob spanning ~200 per-country
        # files at USA scale, that's ~200 avoidable network round trips on
        # every pre-flight budget check, not just the first.
        details = filesystem.glob(path, detail=True)
        if not details:
            raise FileNotFoundError(f"no files matched source: {source!r}")
        return sum(int(info["size"]) for info in details.values())
    return int(filesystem.info(path)["size"])


def _env_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(1, min(int(raw), maximum))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0 < value <= 1 else default
