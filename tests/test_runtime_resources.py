import os
import tempfile
from pathlib import Path

from crc_sdk.connectors.duckdb.connection import (
    DuckDBConnection,
    RuntimeResources,
    _bytes_per_thread,
    _env_int,
    default_work_dir,
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


def test_bytes_per_thread_defaults_to_given_value(monkeypatch) -> None:
    monkeypatch.delenv("CRC_DUCKDB_BYTES_PER_THREAD_GIB", raising=False)
    assert _bytes_per_thread(2.5) == int(2.5 * 1024**3)


def test_bytes_per_thread_none_means_no_cap(monkeypatch) -> None:
    monkeypatch.delenv("CRC_DUCKDB_BYTES_PER_THREAD_GIB", raising=False)
    assert _bytes_per_thread(None) is None


def test_bytes_per_thread_env_overrides_none(monkeypatch) -> None:
    monkeypatch.setenv("CRC_DUCKDB_BYTES_PER_THREAD_GIB", "1.0")
    assert _bytes_per_thread(None) == 1024**3


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


def test_detect_with_no_per_thread_cap_uses_full_cpu_count(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("CRC_DUCKDB_THREADS", raising=False)
    monkeypatch.delenv("CRC_DUCKDB_BYTES_PER_THREAD_GIB", raising=False)
    resources = RuntimeResources.detect(tmp_path, bytes_per_thread_gib=None)
    assert resources.threads == resources.cpus


def test_for_analytics_uncaps_threads_without_spatial_extension(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("CRC_DUCKDB_THREADS", raising=False)
    monkeypatch.delenv("CRC_DUCKDB_BYTES_PER_THREAD_GIB", raising=False)
    spatial = DuckDBConnection.for_analytics(tmp_path, extensions=("spatial",))
    non_spatial = DuckDBConnection.for_analytics(
        tmp_path, extensions=(), database=None
    )
    cpus = RuntimeResources.detect(tmp_path).cpus
    assert non_spatial.config["threads"] == cpus
    assert spatial.config["threads"] <= non_spatial.config["threads"]


def test_default_work_dir_is_a_stable_location_under_system_temp(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CRC_DUCKDB_WORK_DIR", raising=False)
    expected = Path(tempfile.gettempdir()) / "crc-sdk"
    assert default_work_dir() == expected
    # Stable, not a fresh mkdtemp-style path each call.
    assert default_work_dir() == expected


def test_default_work_dir_honors_env_override(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "custom-work-dir"
    monkeypatch.setenv("CRC_DUCKDB_WORK_DIR", str(override))
    assert default_work_dir() == override
