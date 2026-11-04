from __future__ import annotations

import multiprocessing
from pathlib import Path

from crc_sdk.workflows._remote import (
    manifest_lock,
    merge_manifest_entries,
    read_manifest,
    read_plan_manifest,
    write_plan_manifest,
)

_WRITERS = 4
_UPDATES = 15


def _merge_many(cache_dir: str, writer: int) -> None:
    directory = Path(cache_dir)
    for update in range(_UPDATES):
        merge_manifest_entries(
            directory, {f"w{writer}-{update}": update}, provider="test"
        )
        write_plan_manifest(directory, f"plan-{writer}-{update}", {"n": update})


def test_concurrent_manifest_merges_keep_every_entry(tmp_path: Path) -> None:
    # Spawn (the Windows/macOS default) so every writer is a real process.
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_merge_many, args=(str(tmp_path), writer))
        for writer in range(_WRITERS)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    manifest = read_manifest(tmp_path)
    assert manifest is not None
    assert set(manifest["entries"]) == {
        f"w{writer}-{update}"
        for writer in range(_WRITERS)
        for update in range(_UPDATES)
    }
    assert len(manifest["plans"]) == _WRITERS * _UPDATES
    assert read_plan_manifest(tmp_path, "plan-0-3") == {"n": 3}


def test_manifest_lock_can_be_taken_repeatedly_and_creates_the_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "new" / "cache"
    with manifest_lock(directory):
        assert directory.is_dir()
    with manifest_lock(directory):
        pass
    merge_manifest_entries(directory, {"a": 1})
    assert read_manifest(directory) == {"entries": {"a": 1}}
