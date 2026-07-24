import os
from pathlib import Path

from crc_sdk.connectors.duckdb.connection import RuntimeResources, _env_int


def test_env_int_falls_back_on_non_numeric(monkeypatch) -> None:
    monkeypatch.setenv("CRC_DUCKDB_THREADS", "eight")
    assert _env_int("CRC_DUCKDB_THREADS", default=4, maximum=16) == 4


def test_detect_tolerates_invalid_thread_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CRC_DUCKDB_THREADS", "not-a-number")
    monkeypatch.delenv("CRC_DUCKDB_MEMORY", raising=False)
    resources = RuntimeResources.detect(tmp_path)
    assert resources.threads >= 1
    assert os.getenv("CRC_DUCKDB_THREADS") == "not-a-number"
