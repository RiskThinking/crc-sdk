"""DuckDB connection and extension lifecycle."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import psutil  # type: ignore[import-untyped]
from duckdb import DuckDBPyConnection

_GIB = 1024**3
_COMMUNITY_EXTENSIONS = frozenset({"h3"})
# GEOS-backed spatial operations regress when over-threaded relative to
# available memory; DuckDB's core engine has no such pathology and is
# already bounded by memory_limit + spill-to-disk regardless of thread
# count, so this ceiling applies only when "spatial" is actually requested.
_SPATIAL_BYTES_PER_THREAD_GIB = 2.5


def default_work_dir() -> Path:
    """The SDK-wide default work/spill directory for resource-tuned connections.

    Constructors that need :meth:`DuckDBConnection.for_analytics` tuning but
    have no caller-supplied directory of their own (unlike, say,
    ``write_hazard_dataset``, which derives one from its destination path)
    use this — a stable, discoverable location under the system temp
    directory, not a fresh random one per call. Override globally with
    ``CRC_DUCKDB_WORK_DIR``, or per-call via each constructor's own
    ``work_dir`` parameter.
    """
    override = os.getenv("CRC_DUCKDB_WORK_DIR")
    return Path(override) if override else Path(tempfile.gettempdir()) / "crc-sdk"


# Applied to every connection, even a bare DuckDBConnection() with no config
# at all — DuckDB's own defaults preserve insertion order (limits some
# parallel query plans) and cache object metadata unboundedly (unbounded
# memory growth across repeated reads). self.config always wins if a caller
# sets either explicitly.
_BASE_CONFIG: dict[str, Any] = {
    "preserve_insertion_order": False,
    "enable_object_cache": False,
}


def detected_cpu_count() -> int:
    """CPU count clamped to cgroup quota and process affinity, whichever is tightest.

    Unlike ``os.cpu_count()``, this reflects container CPU quotas and
    affinity masks. Anything sizing a process pool — not just DuckDB's own
    thread count — should use this instead, or it can spawn far more workers
    than a constrained host/container is actually allotted.
    """
    cpu_limits = [psutil.cpu_count(logical=True) or 1]
    try:
        affinity = psutil.Process().cpu_affinity()
        if affinity:
            cpu_limits.append(len(affinity))
    except (AttributeError, OSError, Exception):
        pass
    quota = _cpu_quota()
    if quota is not None:
        cpu_limits.append(quota)
    return max(1, min(cpu_limits))


# Empirically derived, not a documented DuckDB recommendation: a live
# production ``COPY ... PARTITION_BY`` write (248 partitions, USA-scale
# building data) showed three regimes as this setting varied relative to the
# actual partition count -- a fixed value far below it (16) throttled the
# whole write to ~1/30th of available cores via constant file-handle
# open/close churn; setting it to exactly match the partition count (248)
# was ~3x faster initially but the write *collapsed* to a near-standstill
# crawl partway through a sustained multi-GB write (confirmed via /proc
# thread inspection to still be doing real, if minimal, work -- not
# deadlocked -- but at a rate that would have taken hours longer); a capped
# middle ground (64) gave up some of that initial speed but sailed straight
# through the point where the uncapped run collapsed and finished reliably.
# This value has not been swept for an optimum between 64 and the full
# partition count -- treat it as a reasoned, tested-safe default, not a
# proven-best one.
DEFAULT_PARTITIONED_WRITE_MAX_OPEN_FILES_CEILING = 64


def partitioned_write_open_files_hint(
    partition_count: int,
    *,
    ceiling: int = DEFAULT_PARTITIONED_WRITE_MAX_OPEN_FILES_CEILING,
) -> int:
    """Suggest a ``partitioned_write_max_open_files`` for one ``PARTITION_BY`` write.

    Call this per-write (``con.execute(f"SET partitioned_write_max_open_files={hint}")``
    right before the ``COPY``), not once at connection-creation time -- the
    right value depends on how many distinct partition values *that write*
    is about to touch, which :class:`RuntimeResources`'s connection-wide
    default (a fixed, conservative ``8``) has no way to know in advance. See
    the module-level comment above
    :data:`DEFAULT_PARTITIONED_WRITE_MAX_OPEN_FILES_CEILING` for how that
    ceiling was chosen. Override the ceiling via
    ``CRC_DUCKDB_PARTITIONED_WRITE_MAX_OPEN_FILES`` for a quick global tweak
    without touching call sites.
    """
    raw = os.getenv("CRC_DUCKDB_PARTITIONED_WRITE_MAX_OPEN_FILES")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, min(int(partition_count), ceiling))


@dataclass(frozen=True)
class RuntimeResources:
    """Detected host/container limits and recommended DuckDB settings."""

    cpus: int
    memory_bytes: int
    disk_free_bytes: int
    threads: int
    memory_limit: str
    temp_directory: Path
    max_temp_directory_size: str

    @classmethod
    def detect(
        cls,
        work_dir: str | Path,
        *,
        bytes_per_thread_gib: float | None = _SPATIAL_BYTES_PER_THREAD_GIB,
    ) -> RuntimeResources:
        """Detect host/container limits and recommend DuckDB settings.

        ``bytes_per_thread_gib`` throttles thread count against usable
        memory — appropriate for GEOS-backed spatial work (the default).
        Pass ``None`` to skip that throttle entirely and use every detected
        CPU (DuckDB's core engine bounds total memory via ``memory_limit`` +
        spill-to-disk regardless of thread count, so nothing else needs it).
        :meth:`for_analytics` picks this automatically from whether
        ``"spatial"`` is requested; call this directly only for finer control.
        ``CRC_DUCKDB_BYTES_PER_THREAD_GIB``, if set to a positive value,
        overrides either case.
        """
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        temp_directory = root / "duckdb-temp"
        temp_directory.mkdir(parents=True, exist_ok=True)

        cpus = detected_cpu_count()
        memory_bytes = _memory_limit_bytes(int(psutil.virtual_memory().total))
        disk_free_bytes = int(psutil.disk_usage(str(root)).free)
        usable_memory = max(_GIB, int(memory_bytes * 0.60))
        per_thread = _bytes_per_thread(bytes_per_thread_gib)
        if per_thread is None:
            safe_threads = cpus
        else:
            safe_threads = max(1, min(cpus, usable_memory // per_thread))
        threads = _env_int("CRC_DUCKDB_THREADS", safe_threads, cpus)
        memory_gib = max(1, usable_memory // _GIB)
        memory_limit = os.getenv("CRC_DUCKDB_MEMORY", f"{memory_gib}GiB")
        spill = max(_GIB, min(64 * _GIB, disk_free_bytes // 2))
        return cls(
            cpus=cpus,
            memory_bytes=memory_bytes,
            disk_free_bytes=disk_free_bytes,
            threads=threads,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
            max_temp_directory_size=f"{spill}B",
        )

    def as_duckdb_config(self) -> dict[str, Any]:
        return {
            "threads": self.threads,
            "memory_limit": self.memory_limit,
            "temp_directory": str(self.temp_directory),
            "max_temp_directory_size": self.max_temp_directory_size,
            "preserve_insertion_order": False,
            "enable_object_cache": False,
            "partitioned_write_max_open_files": 8,
        }

    def as_dict(self) -> dict[str, int | str | float]:
        result = asdict(self)
        result["memory_gib"] = round(self.memory_bytes / _GIB, 2)
        result["disk_free_gib"] = round(self.disk_free_bytes / _GIB, 2)
        result["temp_directory"] = str(self.temp_directory)
        return result


@dataclass(frozen=True)
class DuckDBConnection:
    """Configuration for a lazily created DuckDB connection.

    ``setup_sql`` runs after extensions load, before the connection is
    handed back — the natural place for ``CREATE OR REPLACE SECRET`` (or any
    other per-connection DDL/``SET``/``ATTACH``) a caller needs. This is
    deliberately just a sequence of raw SQL strings, not a typed secret
    builder: DuckDB's own secret DDL is already the idiomatic, fully
    documented interface (https://duckdb.org/docs/configuration/secrets_manager),
    varies per provider/type in ways a wrapper would have to keep chasing
    (new provider keywords, new options), and a caller writing
    ``f"CREATE OR REPLACE SECRET g (TYPE GCS, KEY_ID {sql_quote(key)}, "
    f"SECRET {sql_quote(secret)})"`` already gets everything a builder would
    add except one bug class -- DuckDB's ``PROVIDER`` option being a bare
    keyword rather than a quoted literal, unlike every other secret option --
    which is a one-line reminder, not a reason to maintain a whole builder.
    :func:`sql_quote`/:func:`sql_identifier` stay exported for exactly this:
    quoting values safely in caller-authored statements.
    """

    database: str | None = None
    read_only: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)
    extensions: Sequence[str] = field(default_factory=tuple)
    setup_sql: Sequence[str] = field(default_factory=tuple)

    def connect(self) -> DuckDBPyConnection:
        """Create a connection, load requested extensions, then run ``setup_sql``."""
        con = duckdb.connect(
            self.database or ":memory:",
            read_only=self.read_only,
            config={**_BASE_CONFIG, **self.config},
        )
        if self.extensions:
            ensure_extensions(con, *self.extensions)
        for statement in self.setup_sql:
            con.execute(statement)
        return con

    @classmethod
    def for_analytics(
        cls,
        work_dir: str | Path,
        *,
        extensions: Sequence[str] = ("spatial", "httpfs", "h3"),
        setup_sql: Sequence[str] = (),
        config: Mapping[str, Any] | None = None,
        database: str | None = None,
        read_only: bool = False,
    ) -> DuckDBConnection:
        """Build a connection config from detected resources with optional overrides.

        Thread count is throttled to ``_SPATIAL_BYTES_PER_THREAD_GIB`` only
        when ``"spatial"`` is among ``extensions`` — that ceiling exists for
        GEOS's per-thread memory footprint, not a general DuckDB concern.
        Without it, threads default to every detected CPU.
        """
        bytes_per_thread_gib = (
            _SPATIAL_BYTES_PER_THREAD_GIB if "spatial" in extensions else None
        )
        resources = RuntimeResources.detect(
            work_dir, bytes_per_thread_gib=bytes_per_thread_gib
        )
        merged = resources.as_duckdb_config()
        if config:
            merged.update(dict(config))
        return cls(
            database=database,
            read_only=read_only,
            config=merged,
            extensions=tuple(extensions),
            setup_sql=tuple(setup_sql),
        )


def ensure_extensions(con: DuckDBPyConnection, *names: str) -> None:
    """Install (if needed) and load DuckDB extensions.

    Falls back from ``LOAD`` to ``INSTALL`` to ``FORCE INSTALL`` in turn --
    the last step matters for a stale or offline extension cache, where a
    plain ``INSTALL`` raises ``duckdb.IOException`` even though a
    re-fetched copy would succeed.
    """
    for name in names:
        try:
            con.execute(f"LOAD {name}")
            continue
        except Exception:
            pass
        origin = " FROM community" if name in _COMMUNITY_EXTENSIONS else ""
        statement = f"INSTALL {name}{origin}".strip()
        try:
            con.execute(statement)
        except duckdb.IOException:
            con.execute(statement.replace("INSTALL", "FORCE INSTALL", 1))
        con.execute(f"LOAD {name}")


def sql_identifier(name: str) -> str:
    """Quote ``name`` as a DuckDB identifier, doubling any embedded quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def sql_quote(value: object) -> str:
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


class DuckDBStreamEngine:
    """Streaming helpers over analytics-tuned DuckDB connections."""

    @staticmethod
    def create_streaming_connection(
        work_dir: str | Path | None = None,
        *,
        extensions: Sequence[str] = (),
        config: Mapping[str, Any] | None = None,
    ) -> DuckDBPyConnection:
        if work_dir is None:
            merged: dict[str, Any] = {"preserve_insertion_order": False}
            if config:
                merged.update(dict(config))
            return DuckDBConnection(
                config=merged,
                extensions=tuple(extensions),
            ).connect()
        return DuckDBConnection.for_analytics(
            work_dir,
            extensions=extensions,
            config=config,
        ).connect()

    @staticmethod
    def execute_to_parquet_stream(
        con: DuckDBPyConnection,
        query_sql: str,
        output_parquet_path: str | Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Execute a query directly to Parquet using streaming out-of-core write."""
        opts: dict[str, Any] = {"COMPRESSION": "SNAPPY"}
        if options:
            opts.update({str(key).upper(): value for key, value in options.items()})
        option_sql = ", ".join(
            f"{key} {sql_quote(value) if isinstance(value, str) else value}"
            for key, value in opts.items()
        )
        con.execute(
            f"""
            COPY ({query_sql})
            TO {sql_quote(output_parquet_path)}
            (FORMAT PARQUET, {option_sql})
            """
        )


_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _container_cgroup_root() -> Path:
    """Resolve the container's own cgroup v2 path, not the mounted root.

    On some container runtimes (confirmed live on a GKE/containerd node),
    ``/sys/fs/cgroup`` is mounted as the *host's* root cgroup hierarchy, not
    the container's own -- reading ``cpu.max``/``memory.max`` directly from
    that root then silently returns the host's limits instead of the
    container's enforced ones (measured: reported 32 cores/251.9GiB vs. the
    real 31-core/235GiB cgroup-enforced limit, which is what actually gets
    SIGKILLed by the kernel OOM-killer). ``/proc/self/cgroup``'s unified
    (cgroup v2) entry -- ``0::<path>`` -- names the container's own path
    relative to that mount; resolving through it fixes this regardless of
    whether the mount happens to already be container-scoped (same file
    either way in that case, so this is a no-op there).
    """
    try:
        content = Path("/proc/self/cgroup").read_text()
    except OSError:
        return _CGROUP_ROOT
    for line in content.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            relative = parts[2].lstrip("/")
            return (_CGROUP_ROOT / relative) if relative else _CGROUP_ROOT
    return _CGROUP_ROOT


def _memory_limit_bytes(host_total: int) -> int:
    limits = [host_total]
    cgroup_root = _container_cgroup_root()
    for name in (
        cgroup_root / "memory.max",
        _CGROUP_ROOT / "memory/memory.limit_in_bytes",
    ):
        try:
            raw = name.read_text().strip()
            if raw != "max" and 0 < int(raw) < 1 << 60:
                limits.append(int(raw))
        except (OSError, ValueError):
            pass
    return min(limits)


def _cpu_quota() -> int | None:
    cgroup_root = _container_cgroup_root()
    try:
        raw_quota, raw_period = (cgroup_root / "cpu.max").read_text().split()[:2]
        if raw_quota != "max":
            return max(1, math.ceil(int(raw_quota) / int(raw_period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        cfs_quota = int((_CGROUP_ROOT / "cpu/cpu.cfs_quota_us").read_text())
        cfs_period = int((_CGROUP_ROOT / "cpu/cpu.cfs_period_us").read_text())
        if cfs_quota > 0:
            return max(1, math.ceil(cfs_quota / cfs_period))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return None


def _env_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(1, min(int(raw), maximum))
    except ValueError:
        return default


def _bytes_per_thread(default_gib: float | None) -> int | None:
    """Bytes-per-thread budget, or ``None`` for no per-thread cap.

    ``CRC_DUCKDB_BYTES_PER_THREAD_GIB``, if set to a positive number,
    overrides ``default_gib`` unconditionally — including forcing a cap
    where the caller otherwise requested none.
    """
    raw = os.getenv("CRC_DUCKDB_BYTES_PER_THREAD_GIB")
    if raw:
        try:
            gib = float(raw)
        except ValueError:
            gib = None
        if gib is not None and gib > 0:
            return max(_GIB // 4, int(gib * _GIB))
    return int(default_gib * _GIB) if default_gib is not None else None
