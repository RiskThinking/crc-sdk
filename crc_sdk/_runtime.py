"""Immutable execution settings and verified source-cache policy."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Self

import duckdb

from ._dataset_schema import Stage

LocalPath = str | os.PathLike[str]


def _local_path(value: LocalPath) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("path must be a local string or filesystem path")
    text = os.fspath(value)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise ValueError("path must be a nonempty local filesystem path")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", text) and not re.match(
        r"^[A-Za-z]:[\\/]", text
    ):
        raise ValueError("remote URI publication is unsupported; use a local path")
    return Path(text).absolute()


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _memory_bytes(value: str | int | None) -> int | None:
    if value is None:
        return None
    if type(value) is int:
        return _positive(value, "memory_limit")
    if not isinstance(value, str):
        raise TypeError("memory_limit must be bytes or a size such as '512MB'")
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s*", value, re.I
    )
    if match is None:
        raise ValueError("memory_limit must be bytes or a size such as '512MB'")
    unit = match.group(2).upper()
    powers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    result = int(float(match.group(1)) * powers[unit])
    return _positive(result, "memory_limit")


class CacheMode(Enum):
    EPHEMERAL = "ephemeral"
    REUSE = "reuse"
    OFFLINE = "offline"
    REFRESH = "refresh"


@dataclass(frozen=True, slots=True, init=False)
class Cache:
    mode: CacheMode
    directory: Path | None

    def __init__(self) -> None:
        raise TypeError(
            "use Cache.ephemeral(), reuse(path), offline(path), or refresh(path)"
        )

    @classmethod
    def _create(cls, mode: CacheMode, directory: LocalPath | None) -> Self:
        value = object.__new__(cls)
        object.__setattr__(value, "mode", mode)
        object.__setattr__(
            value, "directory", None if directory is None else _local_path(directory)
        )
        return value

    @classmethod
    def ephemeral(cls) -> Cache:
        return cls._create(CacheMode.EPHEMERAL, None)

    @classmethod
    def reuse(cls, directory: LocalPath) -> Cache:
        return cls._create(CacheMode.REUSE, directory)

    @classmethod
    def offline(cls, directory: LocalPath) -> Cache:
        return cls._create(CacheMode.OFFLINE, directory)

    @classmethod
    def refresh(cls, directory: LocalPath) -> Cache:
        return cls._create(CacheMode.REFRESH, directory)


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: Stage
    completed: int
    total: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage):
            raise TypeError("stage must be Stage")
        if type(self.completed) is not int or self.completed < 0:
            raise ValueError("completed must be a nonnegative integer")
        if self.total is not None and (
            type(self.total) is not int or self.total < self.completed
        ):
            raise ValueError("total must be an integer at least completed")


@dataclass(frozen=True, slots=True, init=False)
class Runtime:
    cache: Cache
    workers: int
    batch_rows: int
    memory_limit: str | int | None
    _connection: duckdb.DuckDBPyConnection | None = field(repr=False)
    _progress: Callable[[ProgressEvent], None] | None = field(repr=False)

    def __init__(
        self,
        *,
        cache: Cache | None = None,
        connection: duckdb.DuckDBPyConnection | None = None,
        workers: int = 1,
        batch_rows: int = 65536,
        memory_limit: str | int | None = None,
        progress: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        if cache is not None and not isinstance(cache, Cache):
            raise TypeError("cache must be Cache")
        if connection is not None and not isinstance(
            connection, duckdb.DuckDBPyConnection
        ):
            raise TypeError("connection must be a DuckDB connection")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")
        _positive(workers, "workers")
        _positive(batch_rows, "batch_rows")
        _memory_bytes(memory_limit)
        for name, value in {
            "cache": Cache.ephemeral() if cache is None else cache,
            "workers": workers,
            "batch_rows": batch_rows,
            "memory_limit": memory_limit,
            "_connection": connection,
            "_progress": progress,
        }.items():
            object.__setattr__(self, name, value)
