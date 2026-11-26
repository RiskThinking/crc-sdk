"""Execution identity, source pinning, streaming replay, and resource lifetimes."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import DiagnosticSeverity

from crc_sdk._dataset_schema import (
    ExecutionDiagnostic,
    ExecutionReport,
    ResourceIdentity,
    Stage,
)
from crc_sdk._graph import DataValidationError, ExecutionScope, Node, NodeStream
from crc_sdk._runtime import Cache, CacheMode, ProgressEvent, Runtime


@dataclass(frozen=True)
class Metadata:
    name: str
    report: ExecutionReport = ExecutionReport(False, None)


class CountingNode(Node[Metadata]):
    def __init__(self, key: str, events: list[str], *, fail: bool = False) -> None:
        self.key = key
        self.events = events
        self.fail = fail

    def open(self, scope: ExecutionScope) -> NodeStream[Metadata]:
        self.events.append("open:" + self.key)
        schema = pa.schema([pa.field("value", pa.int64())])

        def batches() -> Iterator[pa.RecordBatch]:
            try:
                for number in range(3):
                    self.events.append("batch:" + str(number))
                    if self.fail and number == 1:
                        raise DataValidationError(
                            (
                                ExecutionDiagnostic(
                                    "bad_value",
                                    DiagnosticSeverity.ERROR,
                                    Stage.VALIDATION,
                                    "The next batch is invalid.",
                                ),
                            )
                        )
                    yield pa.record_batch([[number]], schema=schema)
            finally:
                self.events.append("close:" + self.key)

        return NodeStream(schema, Metadata(self.key), batches())


def identity(
    data: bytes, uri: str = "https://example.invalid/object"
) -> ResourceIdentity:
    return ResourceIdentity(uri, "sha256:" + hashlib.sha256(data).hexdigest())


def test_configuration_is_pure_immutable_and_has_one_default_policy(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "does-not-exist"
    cache = Cache.reuse(directory)
    runtime = Runtime(cache=cache, workers=2, batch_rows=5, memory_limit="1GiB")
    assert not directory.exists()
    assert runtime.cache is cache
    assert runtime.workers == 2
    assert Runtime().cache == Cache.ephemeral()
    assert Cache.ephemeral().mode is CacheMode.EPHEMERAL
    assert Cache.ephemeral().directory is None
    with pytest.raises(FrozenInstanceError):
        runtime.workers = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        Cache()
    for value in (0, -1, True):
        with pytest.raises(ValueError):
            Runtime(workers=value)
    with pytest.raises(ValueError):
        Runtime(memory_limit="unbounded")
    with pytest.raises(ValueError):
        Cache.reuse("s3://bucket/cache")


def test_scope_creation_and_metadata_do_not_compute_batches() -> None:
    events: list[str] = []
    node = CountingNode("source", events)
    scope = ExecutionScope()
    assert scope._temporary is None
    assert scope.metadata(node).name == "source"
    assert events == ["open:source"]
    assert scope.report(node).complete is False
    assert scope.report(node).valid is None
    scope.close()
    assert scope.metadata(node).name == "source"
    assert scope.report(node).complete is False


def test_interleaved_consumers_replay_one_execution_and_own_tables_after_close() -> (
    None
):
    events: list[str] = []
    node = CountingNode("source", events)
    with ExecutionScope() as scope:
        first = scope.iter_batches(node)
        second = scope.iter_batches(CountingNode("source", events))
        one = next(first)
        assert one.column(0).to_pylist() == [0]
        assert events == ["open:source", "batch:0"]
        assert next(second).column(0).to_pylist() == [0]
        assert next(second).column(0).to_pylist() == [1]
        assert next(first).column(0).to_pylist() == [1]
        assert [b.column(0).to_pylist() for b in first] == [[2]]
        assert [b.column(0).to_pylist() for b in second] == [[2]]
        assert events.count("open:source") == 1
        assert events.count("batch:0") == 1
        assert scope.complete(node)
        assert scope.report(node).valid is True
        assert scope.report(node).counts.produced == 3
        owned_directory = scope.directory
        final_report = scope.report(node)
    assert not owned_directory.exists()
    assert scope.report(node) == final_report
    assert one.column(0).to_pylist() == [0]
    with pytest.raises(RuntimeError, match="closed"):
        list(scope.iter_batches(node))


def test_early_exit_closes_reader_and_scope_owned_files() -> None:
    events: list[str] = []
    node = CountingNode("source", events)
    with ExecutionScope() as scope:
        cursor = scope.iter_batches(node)
        next(cursor)
        directory = scope.directory
        assert not scope.report(node).complete
    assert events == ["open:source", "batch:0", "close:source"]
    assert not directory.exists()
    assert not scope.report(node).complete
    assert scope.report(node).valid is None


def test_expected_data_failure_has_same_report_and_is_not_recomputed() -> None:
    events: list[str] = []
    node = CountingNode("source", events, fail=True)
    with ExecutionScope() as scope:
        for _ in range(2):
            iterator = scope.iter_batches(node)
            assert next(iterator).column(0).to_pylist() == [0]
            with pytest.raises(DataValidationError, match="invalid"):
                next(iterator)
        report = scope.report(node)
        assert not report.complete
        assert report.valid is False
        assert report.diagnostics[0].code == "bad_value"
        assert events.count("batch:1") == 1
    assert scope.report(node) == report


def test_two_model_nodes_share_one_source_and_each_distinct_model_once() -> None:
    events: list[str] = []
    parent = CountingNode("source", events)

    class Model(Node[Metadata]):
        def __init__(self, amount: int) -> None:
            self.key = "model:" + str(amount)
            self.amount = amount

        def open(self, scope: ExecutionScope) -> NodeStream[Metadata]:
            events.append("model-open:" + self.key)
            schema = pa.schema([pa.field("value", pa.int64())])

            def batches() -> Iterator[pa.RecordBatch]:
                for batch in scope.iter_batches(parent):
                    values = [
                        value + self.amount for value in batch.column(0).to_pylist()
                    ]
                    yield pa.record_batch([values], schema=schema)

            return NodeStream(schema, Metadata(self.key), batches())

    with ExecutionScope() as scope:
        assert [b.column(0).to_pylist() for b in scope.iter_batches(Model(2))] == [
            [2],
            [3],
            [4],
        ]
        assert [b.column(0).to_pylist() for b in scope.iter_batches(Model(4))] == [
            [4],
            [5],
            [6],
        ]
        assert [b.column(0).to_pylist() for b in scope.iter_batches(Model(2))] == [
            [2],
            [3],
            [4],
        ]
    assert events.count("open:source") == 1
    assert events.count("model-open:model:2") == 1
    assert events.count("model-open:model:4") == 1
    assert events.count("batch:0") == 1


def test_recipes_do_not_qualify_by_having_reader_methods() -> None:
    class Pretender:
        key = "pretender"

        def inspect(self) -> Metadata:
            return Metadata("not-a-reader")

        def batches(self) -> Iterator[pa.RecordBatch]:
            return iter(())

    with ExecutionScope() as scope:
        with pytest.raises(TypeError, match="explicit Node"):
            scope.metadata(Pretender())  # type: ignore[arg-type]


def test_schema_mismatch_is_a_data_error_before_any_batch_is_yielded() -> None:
    class WrongSchema(Node[Metadata]):
        key = "bad-schema"

        def open(self, scope: ExecutionScope) -> NodeStream[Metadata]:
            return NodeStream(
                pa.schema([("value", pa.int64())]),
                Metadata("bad"),
                iter([pa.record_batch([["wrong"]], names=["value"])]),
            )

    with ExecutionScope() as scope:
        node = WrongSchema()
        with pytest.raises(DataValidationError, match="schema"):
            next(scope.iter_batches(node))
        assert scope.report(node).valid is False
        assert scope.report(node).counts.produced == 0


def test_resource_cache_verifies_hits_and_protects_consumed_bytes(
    tmp_path: Path,
) -> None:
    content = b"source content"
    resource = identity(content)
    calls: list[Path] = []

    def acquire(path: Path) -> None:
        calls.append(path)
        path.write_bytes(content)

    cache = tmp_path / "cache"
    with ExecutionScope(Runtime(cache=Cache.reuse(cache))) as scope:
        pinned = scope.pin(resource, acquire)
        assert scope.pin(resource, acquire) == pinned
        assert pinned.read_bytes() == content
    assert len(calls) == 1
    cached = cache / "source-objects-1.0" / resource.sha256.removeprefix("sha256:")
    with ExecutionScope(Runtime(cache=Cache.offline(cache))) as scope:
        pinned = scope.pin(resource, acquire)
        cached.write_bytes(b"changed cache object")
        assert pinned.read_bytes() == content
    assert len(calls) == 1
    with ExecutionScope(Runtime(cache=Cache.reuse(cache))) as scope:
        with pytest.raises(OSError, match="identity mismatch"):
            scope.pin(resource, acquire)
    assert len(calls) == 1


def test_offline_cache_miss_never_acquires_and_old_cache_layout_is_unsupported(
    tmp_path: Path,
) -> None:
    data = b"object"
    calls: list[Path] = []
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / identity(data).sha256.removeprefix("sha256:")).write_bytes(data)
    with ExecutionScope(Runtime(cache=Cache.offline(cache))) as scope:
        with pytest.raises(FileNotFoundError, match="offline"):
            scope.pin(identity(data), lambda target: calls.append(target))
    assert calls == []


def test_refresh_always_reacquires_once_per_scope_and_repairs_corrupt_entry(
    tmp_path: Path,
) -> None:
    data = b"new object"
    resource = identity(data)
    cached = tmp_path / "source-objects-1.0" / resource.sha256.removeprefix("sha256:")
    cached.parent.mkdir()
    cached.write_bytes(b"corrupt")
    calls: list[Path] = []

    def acquire(target: Path) -> None:
        calls.append(target)
        target.write_bytes(data)

    with ExecutionScope(Runtime(cache=Cache.refresh(tmp_path))) as scope:
        assert scope.pin(resource, acquire).read_bytes() == data
        scope.pin(resource, acquire)
    assert len(calls) == 1
    assert cached.read_bytes() == data


def test_failed_or_cancelled_acquisition_never_replaces_cache(tmp_path: Path) -> None:
    resource = identity(b"correct")
    cached = tmp_path / "source-objects-1.0" / resource.sha256.removeprefix("sha256:")
    cached.parent.mkdir()
    cached.write_bytes(b"previous verified object")

    def wrong(target: Path) -> None:
        target.write_bytes(b"incorrect")

    with ExecutionScope(Runtime(cache=Cache.refresh(tmp_path))) as scope:
        with pytest.raises(OSError, match="identity mismatch"):
            scope.pin(resource, wrong)
    assert cached.read_bytes() == b"previous verified object"

    def cancel(event: ProgressEvent) -> None:
        if event.completed:
            raise KeyboardInterrupt

    def correct(target: Path) -> None:
        target.write_bytes(b"correct")

    with ExecutionScope(
        Runtime(cache=Cache.refresh(tmp_path), progress=cancel)
    ) as scope:
        with pytest.raises(KeyboardInterrupt):
            scope.pin(resource, correct)
        directory = scope.directory
    assert not directory.exists()
    assert cached.read_bytes() == b"previous verified object"


def test_local_snapshot_does_not_reopen_mutable_source_under_earlier_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    with ExecutionScope() as scope:
        pinned, resource = scope.pin_local(source)
        source.write_bytes(b"changed")
        pinned_again, resource_again = scope.pin_local(source)
        assert pinned == pinned_again
        assert pinned.read_bytes() == b"original"
        assert resource.sha256 == resource_again.sha256 == identity(b"original").sha256


def test_borrowed_connection_is_not_configured_or_closed() -> None:
    connection = duckdb.connect(config={"threads": 2})
    before = connection.execute("SELECT current_setting('threads')").fetchone()
    with ExecutionScope(
        Runtime(connection=connection, workers=7, memory_limit="1GB")
    ) as scope:
        assert scope.connection is connection
        assert (
            scope.connection.execute("SELECT current_setting('threads')").fetchone()
            == before
        )
    assert connection.execute("SELECT 42").fetchone() == (42,)
    connection.close()


def test_owned_connection_is_configured_and_closed() -> None:
    with ExecutionScope(Runtime(workers=2, memory_limit="512MB")) as scope:
        connection = scope.connection
        assert connection.execute("SELECT current_setting('threads')").fetchone() == (
            2,
        )
        assert connection.execute(
            "SELECT current_setting('memory_limit')"
        ).fetchone() == ("488.2 MiB",)
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_registered_context_resources_close_on_exception() -> None:
    events: list[str] = []

    class Resource:
        def __enter__(self) -> Resource:
            events.append("entered")
            return self

        def __exit__(self, *args: object) -> None:
            events.append("closed")

    with pytest.raises(OSError, match="failure"):
        with ExecutionScope() as scope:
            scope.enter_context(Resource())
            raise OSError("failure")
    assert events == ["entered", "closed"]


def test_unopened_reports_and_completion_are_pure_even_after_close() -> None:
    events: list[str] = []
    node = CountingNode("unopened", events)
    scope = ExecutionScope()
    assert scope.report(node) == ExecutionReport(False, None)
    assert not scope.complete(node)
    assert scope._temporary is None
    scope.close()
    assert scope.report(node) == ExecutionReport(False, None)
    assert not scope.complete(node)
    assert events == []


def test_open_validation_failure_is_cached_without_reopening() -> None:
    events: list[str] = []
    diagnostics = (
        ExecutionDiagnostic(
            "metadata_invalid",
            DiagnosticSeverity.ERROR,
            Stage.VALIDATION,
            "The metadata is invalid.",
        ),
    )

    class InvalidMetadata(Node[Metadata]):
        key = "invalid-metadata"

        def open(self, scope: ExecutionScope) -> NodeStream[Metadata]:
            events.append("open")
            raise DataValidationError(diagnostics)

    node = InvalidMetadata()
    scope = ExecutionScope()
    assert scope.report(node).valid is None
    for _ in range(2):
        with pytest.raises(DataValidationError, match="metadata"):
            scope.metadata(node)
    assert events == ["open"]
    expected = ExecutionReport(False, False, diagnostics)
    assert scope.report(node) == expected
    assert not scope.complete(node)
    scope.close()
    assert scope.report(node) == expected


def test_operational_open_failure_preserves_exception_category() -> None:
    events: list[str] = []

    class Unavailable(Node[Metadata]):
        key = "unavailable"

        def open(self, scope: ExecutionScope) -> NodeStream[Metadata]:
            events.append("open")
            raise PermissionError("cannot read source")

    with ExecutionScope() as scope:
        node = Unavailable()
        for _ in range(2):
            with pytest.raises(PermissionError, match="cannot read"):
                scope.metadata(node)
        assert events == ["open"]
        assert scope.report(node) == ExecutionReport(False, None)


def test_graph_cycles_fail_before_recursive_acquisition() -> None:
    class Cycle(Node[Metadata]):
        key = "cycle"

        def open(self, scope: ExecutionScope) -> NodeStream[Metadata]:
            return NodeStream(pa.schema([]), scope.metadata(self), iter(()))

    with ExecutionScope() as scope:
        node = Cycle()
        with pytest.raises(RuntimeError, match="cyclic"):
            scope.metadata(node)
        assert scope.report(node).valid is None


def test_pinning_owned_handle_uses_discovered_fd_not_replaced_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"original source")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement source")
    with source.open("rb") as reader, ExecutionScope() as scope:
        reader.seek(3)
        try:
            replacement.replace(source)
        except PermissionError:
            pytest.skip("this operating system disallows replacing an open file")
        pinned, resource = scope.pin_handle(reader, uri=str(source))
        assert pinned.read_bytes() == b"original source"
        assert resource.sha256 == identity(b"original source").sha256
        assert reader.tell() == 3
        assert not reader.closed
    assert reader.closed
    assert source.read_bytes() == b"replacement source"


def test_pinning_handle_retains_verified_object_for_offline_use(tmp_path: Path) -> None:
    import io

    content = b"pinned object"
    with ExecutionScope(Runtime(cache=Cache.reuse(tmp_path))) as scope:
        reader = io.BytesIO(content)
        pinned, resource = scope.pin_handle(reader, uri="memory:explicit-local")
        assert pinned.read_bytes() == content
        assert not reader.closed
    cached = tmp_path / "source-objects-1.0" / resource.sha256.removeprefix("sha256:")
    assert cached.read_bytes() == content
    with ExecutionScope(Runtime(cache=Cache.offline(tmp_path))) as scope:
        changed = io.BytesIO(b"changed source")
        changed.seek(4)
        pinned, restored = scope.pin_handle(
            changed,
            uri="memory:explicit-local",
            expected_sha256=resource.sha256,
        )
        assert pinned.read_bytes() == content
        assert restored.sha256 == resource.sha256
        assert changed.tell() == 4
        assert not changed.closed


def test_wrong_handle_digest_restores_position_and_leaves_handle_open() -> None:
    import io

    reader = io.BytesIO(b"incorrect")
    reader.seek(2)
    with ExecutionScope() as scope:
        with pytest.raises(OSError, match="identity mismatch"):
            scope.pin_handle(
                reader, uri="memory:input", expected_sha256=identity(b"correct").sha256
            )
        assert reader.tell() == 2
        assert not reader.closed
        assert list(scope.directory.iterdir()) == []


def test_shared_computation_keeps_requested_branch_metadata_and_reports() -> None:
    from dataclasses import replace

    from crc_sdk._dataset_schema import SourceIdentity

    events: list[str] = []

    class View(CountingNode):
        def __init__(self, label: str) -> None:
            super().__init__("same-scientific-computation", events)
            self.label = label

        def _view_metadata(
            self,
            scope: ExecutionScope,
            replay_metadata: Metadata,
        ) -> Metadata:
            return replace(replay_metadata, name=self.label)

        def _view_report(
            self,
            scope: ExecutionScope,
            replay_report: ExecutionReport,
        ) -> ExecutionReport:
            return replace(
                replay_report,
                source_identities=(
                    SourceIdentity(
                        provider="user",
                        product=self.label,
                    ),
                ),
            )

        def _view_batch(
            self,
            scope: ExecutionScope,
            batch: pa.RecordBatch,
        ) -> pa.RecordBatch:
            return batch.replace_schema_metadata({b"label": self.label.encode()})

    first, second = View("first-attribution"), View("second-attribution")
    with ExecutionScope() as scope:
        assert scope.computation_report(first) == ExecutionReport(False, None)
        assert events == []
        first_batches = list(scope.iter_batches(first))
        second_batches = list(scope.iter_batches(second))
        assert scope.metadata(first).name == "first-attribution"
        assert scope.metadata(second).name == "second-attribution"
        assert first_batches[0].schema.metadata == {b"label": b"first-attribution"}
        assert second_batches[0].schema.metadata == {b"label": b"second-attribution"}
        report_first = scope.report(first)
        report_second = scope.report(second)
        raw_report = scope.computation_report(second)
        assert raw_report.complete and raw_report.valid is True
        assert raw_report.source_identities == ()
        assert report_first.source_identities[0].product == "first-attribution"
        assert report_second.source_identities[0].product == "second-attribution"
        assert report_first.complete and report_second.complete
    assert scope.metadata(second).name == "second-attribution"
    assert scope.report(second) == report_second
    assert scope.computation_report(second) == raw_report
    assert report_first.source_identities[0].product == "first-attribution"
    assert events.count("open:same-scientific-computation") == 1
    assert events.count("batch:0") == 1


def test_prefetched_acquires_ahead_on_threads_and_reports_in_order() -> None:
    import threading
    import time

    from crc_sdk._dataset_schema import Stage

    consumer = threading.get_ident()
    callbacks: list[int] = []
    workers: set[int] = set()
    running = 0
    peak = 0
    lock = threading.Lock()

    def acquire(item: int) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        workers.add(threading.get_ident())
        time.sleep(0.01 * ((7 * item) % 3))
        scope.emit(Stage.ACQUISITION, item, 10)
        with lock:
            running -= 1
        if item == 6:
            raise OSError("item 6 is unavailable")
        return item * item

    def progress(event: ProgressEvent) -> None:
        callbacks.append(threading.get_ident())

    runtime = Runtime(workers=3, progress=progress)
    with ExecutionScope(runtime) as scope:
        results = []
        with pytest.raises(OSError, match="item 6"):
            for result in scope.prefetched(list(range(10)), acquire):
                results.append(result)
        assert results == [i * i for i in range(6)]
        assert consumer not in workers and 1 < peak <= 3
        assert callbacks and set(callbacks) == {consumer}


def test_closing_prefetched_early_waits_for_running_acquisitions() -> None:
    import time

    started: list[int] = []
    finished: list[int] = []

    def acquire(item: int) -> int:
        started.append(item)
        time.sleep(0.02)
        finished.append(item)
        return item

    with ExecutionScope(Runtime(workers=2)) as scope:
        stream = scope.prefetched(list(range(20)), acquire)
        assert next(stream) == 0
        stream.close()
        assert sorted(started) == sorted(finished)
        assert len(started) < 20


@pytest.mark.parametrize("clone", [True, False])
def test_verified_copies_clone_where_supported_and_copy_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clone: bool
) -> None:
    import hashlib

    from crc_sdk import _graph

    if not clone:
        monkeypatch.setattr(_graph, "_clone", lambda source, destination: False)
    source = tmp_path / "source"
    source.write_bytes(b"verified bytes" * 1000)
    digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    copy = tmp_path / "copy"
    assert _graph._copy_verified(source, copy, digest) == digest
    assert copy.read_bytes() == source.read_bytes()
    # The copy is independent of later writes to its source.
    source.write_bytes(b"changed")
    assert copy.read_bytes() == b"verified bytes" * 1000
    wrong = tmp_path / "wrong"
    with pytest.raises(_graph.SourceContentChanged):
        _graph._copy_verified(copy, wrong, "sha256:" + "0" * 64)
    assert not wrong.exists()


def test_downloaded_bytes_become_the_snapshot_without_a_copy(tmp_path: Path) -> None:
    import hashlib

    from crc_sdk import _graph

    acquired = tmp_path / "acquiring"
    acquired.write_bytes(b"payload")
    digest = "sha256:" + hashlib.sha256(b"payload").hexdigest()
    with pytest.raises(_graph.SourceContentChanged):
        _graph._move_verified(acquired, tmp_path / "bad", "sha256:" + "0" * 64)
    assert acquired.exists() and not (tmp_path / "bad").exists()
    inode = acquired.stat().st_ino
    _graph._move_verified(acquired, tmp_path / "snapshot", digest)
    assert (tmp_path / "snapshot").stat().st_ino == inode
    assert not acquired.exists()
