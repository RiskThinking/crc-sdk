"""DuckDB connection and extension lifecycle."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

_GIB = 1024**3
_COMMUNITY_EXTENSIONS = frozenset({"h3"})


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
    def detect(cls, work_dir: str | Path) -> RuntimeResources:
        try:
            import psutil  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "Runtime resource detection requires `pip install crc-sdk[connectors]`"
            ) from error

        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        temp_directory = root / "duckdb-temp"
        temp_directory.mkdir(parents=True, exist_ok=True)

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
        cpus = max(1, min(cpu_limits))

        memory_bytes = _memory_limit_bytes(int(psutil.virtual_memory().total))
        disk_free_bytes = int(psutil.disk_usage(str(root)).free)
        usable_memory = max(_GIB, int(memory_bytes * 0.60))
        safe_threads = max(1, min(cpus, usable_memory // (3 * _GIB)))
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
    """Configuration for a lazily created DuckDB connection."""

    database: str | None = None
    read_only: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)
    extensions: Sequence[str] = field(default_factory=tuple)

    def connect(self) -> DuckDBPyConnection:
        """Create a connection and optionally load requested extensions."""
        try:
            import duckdb
        except ImportError as error:
            raise ImportError(
                "DuckDB support requires `pip install crc-sdk[connectors]`"
            ) from error

        con = duckdb.connect(
            self.database or ":memory:",
            read_only=self.read_only,
            config=dict(self.config),
        )
        if self.extensions:
            ensure_extensions(con, *self.extensions)
        return con

    @classmethod
    def for_analytics(
        cls,
        work_dir: str | Path,
        *,
        extensions: Sequence[str] = ("spatial", "httpfs", "h3"),
        config: Mapping[str, Any] | None = None,
        database: str | None = None,
        read_only: bool = False,
    ) -> DuckDBConnection:
        """Build a connection config from detected resources with optional overrides."""
        resources = RuntimeResources.detect(work_dir)
        merged = resources.as_duckdb_config()
        if config:
            merged.update(dict(config))
        return cls(
            database=database,
            read_only=read_only,
            config=merged,
            extensions=tuple(extensions),
        )


def ensure_extensions(con: DuckDBPyConnection, *names: str) -> None:
    """Install (if needed) and load DuckDB extensions."""
    for name in names:
        try:
            con.execute(f"LOAD {name}")
            continue
        except Exception:
            pass
        origin = " FROM community" if name in _COMMUNITY_EXTENSIONS else ""
        con.execute(f"INSTALL {name}{origin}")
        con.execute(f"LOAD {name}")


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


def _memory_limit_bytes(host_total: int) -> int:
    limits = [host_total]
    for name in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            raw = Path(name).read_text().strip()
            if raw != "max" and 0 < int(raw) < 1 << 60:
                limits.append(int(raw))
        except (OSError, ValueError):
            pass
    return min(limits)


def _cpu_quota() -> int | None:
    try:
        raw_quota, raw_period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if raw_quota != "max":
            return max(1, math.ceil(int(raw_quota) / int(raw_period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        cfs_quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        cfs_period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
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
