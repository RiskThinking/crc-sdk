import os
from pathlib import Path

from crc_sdk.connectors.duckdb.connection import (
    RuntimeResources,
    _bytes_per_thread,
    _env_int,
)


def test_env_int_falls_back_on_non_numeric(monkeypatch) -> None:
    monkeypatch.setenv("CRC_DUCKDB_THREADS", "eight")
    assert _env_int("CRC_DUCKDB_THREADS", default=4, maximum=16) == 4


def test_detect_tolerates_invalid_thread_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CRC_DUCKDB_THREADS", "not-a-number")
    monkeypatch.delenv("CRC_DUCKDB_MEMORY", raising=False)
    resources = RuntimeResources.detect(tmp_path)
    assert resources.threads >= 1
    assert os.getenv("CRC_DUCKDB_THREADS") == "not-a-number"


def test_bytes_per_thread_defaults_to_2_5_gib(monkeypatch) -> None:
    monkeypatch.delenv("CRC_DUCKDB_BYTES_PER_THREAD_GIB", raising=False)
    assert _bytes_per_thread() == int(2.5 * 1024**3)


def test_geo_profile_thread_budget_is_memory_derived(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("CRC_DUCKDB_THREADS", raising=False)
    monkeypatch.delenv("CRC_DUCKDB_MEMORY", raising=False)
    monkeypatch.delenv("CRC_DUCKDB_BYTES_PER_THREAD_GIB", raising=False)
    resources = RuntimeResources.detect(tmp_path)
    usable = max(1024**3, int(resources.memory_bytes * 0.60))
    expected = max(1, min(resources.cpus, usable // int(2.5 * 1024**3)))
    assert resources.threads == expected
