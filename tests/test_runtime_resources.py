import os
import tempfile
from pathlib import Path

from crc_sdk.connectors.duckdb import connection as connection_module
from crc_sdk.connectors.duckdb.connection import (
    DuckDBConnection,
    RuntimeResources,
    _bytes_per_thread,
    _container_cgroup_root,
    _cpu_quota,
    _env_int,
    _memory_limit_bytes,
    default_work_dir,
    partitioned_write_open_files_hint,
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
    non_spatial = DuckDBConnection.for_analytics(tmp_path, extensions=(), database=None)
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


def _patch_proc_self_cgroup(monkeypatch, content: str | None) -> None:
    """Intercept only ``Path("/proc/self/cgroup").read_text()``; anything
    else falls through to the real implementation so other cgroup/file
    reads in the same test aren't accidentally short-circuited."""
    original_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if str(self) == "/proc/self/cgroup":
            if content is None:
                raise OSError("no such file")
            return content
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)


def test_container_cgroup_root_resolves_v2_unified_line(monkeypatch) -> None:
    _patch_proc_self_cgroup(
        monkeypatch, "0::/kubepods/besteffort/pod123/container456\n"
    )
    assert _container_cgroup_root() == Path(
        "/sys/fs/cgroup/kubepods/besteffort/pod123/container456"
    )


def test_container_cgroup_root_falls_back_when_proc_missing(monkeypatch) -> None:
    _patch_proc_self_cgroup(monkeypatch, None)
    assert _container_cgroup_root() == Path("/sys/fs/cgroup")


def test_container_cgroup_root_falls_back_without_unified_line(monkeypatch) -> None:
    # cgroup v1-style content -- no "0::" unified entry to resolve through.
    _patch_proc_self_cgroup(
        monkeypatch, "4:memory:/docker/abc123\n5:cpu,cpuacct:/docker/abc123\n"
    )
    assert _container_cgroup_root() == Path("/sys/fs/cgroup")


def test_container_cgroup_root_is_a_noop_when_already_at_mount_root(
    monkeypatch,
) -> None:
    _patch_proc_self_cgroup(monkeypatch, "0::/\n")
    assert _container_cgroup_root() == Path("/sys/fs/cgroup")


def test_memory_limit_bytes_reads_resolved_container_cgroup(
    monkeypatch, tmp_path: Path
) -> None:
    """Regression test for the mounted-root-vs-container-cgroup bug: reading
    straight from ``/sys/fs/cgroup/memory.max`` can silently return the
    *host's* limit rather than the container's own on some runtimes."""
    container_root = tmp_path / "container-scope"
    container_root.mkdir()
    (container_root / "memory.max").write_text("123456789")
    monkeypatch.setattr(
        connection_module, "_container_cgroup_root", lambda: container_root
    )
    assert _memory_limit_bytes(host_total=10**15) == 123456789


def test_cpu_quota_reads_resolved_container_cgroup(monkeypatch, tmp_path: Path) -> None:
    container_root = tmp_path / "container-scope"
    container_root.mkdir()
    (container_root / "cpu.max").write_text("400000 100000")
    monkeypatch.setattr(
        connection_module, "_container_cgroup_root", lambda: container_root
    )
    assert _cpu_quota() == 4


def test_partitioned_write_open_files_hint_passes_through_below_ceiling(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CRC_DUCKDB_PARTITIONED_WRITE_MAX_OPEN_FILES", raising=False)
    assert partitioned_write_open_files_hint(3) == 3


def test_partitioned_write_open_files_hint_caps_at_default_ceiling(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CRC_DUCKDB_PARTITIONED_WRITE_MAX_OPEN_FILES", raising=False)
    assert partitioned_write_open_files_hint(248) == 64


def test_partitioned_write_open_files_hint_honors_custom_ceiling(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CRC_DUCKDB_PARTITIONED_WRITE_MAX_OPEN_FILES", raising=False)
    assert partitioned_write_open_files_hint(248, ceiling=128) == 128


def test_partitioned_write_open_files_hint_env_override_bypasses_ceiling(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CRC_DUCKDB_PARTITIONED_WRITE_MAX_OPEN_FILES", "200")
    assert partitioned_write_open_files_hint(248) == 200


def test_partitioned_write_open_files_hint_ignores_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("CRC_DUCKDB_PARTITIONED_WRITE_MAX_OPEN_FILES", "not-a-number")
    assert partitioned_write_open_files_hint(3) == 3
