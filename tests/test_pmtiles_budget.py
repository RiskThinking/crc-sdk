from pathlib import Path

import duckdb
import pytest

from crc_sdk.geometry.pmtiles.budget import (
    DEFAULT_TEMP_TO_INPUT_FACTOR,
    TilingBudget,
    check_tiling_budget,
    measure_source,
)

_GIB = 1024**3


def _budget(free_disk_bytes: int, safe_scratch_bytes: int) -> TilingBudget:
    return TilingBudget(
        tippecanoe_threads=4,
        duckdb_threads=4,
        free_disk_bytes=free_disk_bytes,
        safe_scratch_bytes=safe_scratch_bytes,
    )


def test_check_tiling_budget_passes_when_estimate_fits() -> None:
    budget = _budget(free_disk_bytes=100 * _GIB, safe_scratch_bytes=20 * _GIB)
    check_tiling_budget(input_bytes=1 * _GIB // 2, feature_count=1_000, budget=budget)


def test_check_tiling_budget_raises_with_actionable_message_when_exceeded() -> None:
    budget = _budget(free_disk_bytes=100 * _GIB, safe_scratch_bytes=1 * _GIB)
    with pytest.raises(ValueError) as excinfo:
        check_tiling_budget(
            input_bytes=10 * _GIB, feature_count=1_000_000, budget=budget
        )
    message = str(excinfo.value)
    assert "300.0 GiB" in message  # 10 GiB * 30
    assert "1.0 GiB is considered safe" in message
    assert "1,000,000 features" in message


def test_check_tiling_budget_uses_the_provided_factor() -> None:
    budget = _budget(free_disk_bytes=100 * _GIB, safe_scratch_bytes=5 * _GIB)
    # 1 GiB * 4 = 4 GiB, within a 5 GiB budget -> no raise.
    check_tiling_budget(
        input_bytes=1 * _GIB,
        feature_count=1,
        budget=budget,
        temp_to_input_factor=4,
    )
    with pytest.raises(ValueError):
        check_tiling_budget(
            input_bytes=1 * _GIB,
            feature_count=1,
            budget=budget,
            temp_to_input_factor=DEFAULT_TEMP_TO_INPUT_FACTOR,
        )


def test_tiling_budget_detect_honors_thread_overrides(tmp_path: Path) -> None:
    budget = TilingBudget.detect(tmp_path, tippecanoe_threads=2, duckdb_threads=3)
    assert budget.tippecanoe_threads == 2
    assert budget.duckdb_threads == 3


def test_tiling_budget_detect_honors_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRC_TIPPECANOE_THREADS", "2")
    monkeypatch.setenv("CRC_DUCKDB_THREADS", "3")
    budget = TilingBudget.detect(tmp_path)
    assert budget.tippecanoe_threads == 2
    assert budget.duckdb_threads == 3


def _fix_free_disk(monkeypatch: pytest.MonkeyPatch, free_bytes: int) -> None:
    """Pin ``_disk_free_bytes`` so scratch-fraction math is deterministic --
    a real ``tmp_path`` filesystem's free space varies by machine/CI, and at
    the wrong (small) size both a 25% and a 90% fraction floor to the same
    ``_MIN_SAFE_SCRATCH_BYTES``, masking any difference the fraction makes.
    """
    import crc_sdk.geometry.pmtiles.budget as budget_module

    monkeypatch.setattr(budget_module, "_disk_free_bytes", lambda path: free_bytes)


def test_tiling_budget_detect_honors_scratch_fraction_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CRC_TIPPECANOE_SCRATCH_FRACTION", raising=False)
    _fix_free_disk(monkeypatch, 100 * _GIB)
    default_budget = TilingBudget.detect(tmp_path)
    monkeypatch.setenv("CRC_TIPPECANOE_SCRATCH_FRACTION", "0.9")
    wider_budget = TilingBudget.detect(tmp_path)
    assert wider_budget.safe_scratch_bytes > default_budget.safe_scratch_bytes


def test_tiling_budget_detect_ignores_invalid_scratch_fraction_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CRC_TIPPECANOE_SCRATCH_FRACTION", raising=False)
    _fix_free_disk(monkeypatch, 100 * _GIB)
    default_budget = TilingBudget.detect(tmp_path)
    for invalid in ("0", "-0.5", "1.5", "not-a-number", ""):
        monkeypatch.setenv("CRC_TIPPECANOE_SCRATCH_FRACTION", invalid)
        budget = TilingBudget.detect(tmp_path)
        assert budget.safe_scratch_bytes == default_budget.safe_scratch_bytes


def test_tiling_budget_detect_uses_full_cpu_count_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crc_sdk.connectors.duckdb import detected_cpu_count

    monkeypatch.delenv("CRC_TIPPECANOE_THREADS", raising=False)
    monkeypatch.delenv("CRC_DUCKDB_THREADS", raising=False)
    budget = TilingBudget.detect(tmp_path)
    cpus = detected_cpu_count()
    assert budget.tippecanoe_threads == cpus
    assert budget.duckdb_threads == cpus


def test_tiling_budget_detect_scratch_has_a_floor(tmp_path: Path) -> None:
    budget = TilingBudget.detect(tmp_path, scratch_fraction=0.0)
    assert budget.safe_scratch_bytes >= 2 * _GIB


def test_measure_source_counts_rows_and_bytes(tmp_path: Path) -> None:
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    parquet_path = tmp_path / "points.parquet"
    con.execute(
        f"""
        COPY (SELECT i AS id, ST_Point(i, i) AS geometry FROM range(1, 11) t(i))
        TO '{parquet_path}' (FORMAT PARQUET)
        """
    )
    total_bytes, feature_count = measure_source(str(parquet_path), con=con)
    assert feature_count == 10
    assert total_bytes == parquet_path.stat().st_size


def test_measure_source_raises_on_no_match(tmp_path: Path) -> None:
    con = duckdb.connect()
    # DuckDB's own `read_parquet` rejects the glob before the byte-size
    # probe (`fsspec`-based, in `_source_bytes`) is ever reached.
    with pytest.raises(duckdb.Error):
        measure_source(str(tmp_path / "does-not-exist-*.parquet"), con=con)
