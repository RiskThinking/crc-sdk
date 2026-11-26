"""Scope-owned graph execution and incremental, replayable Arrow streams."""

from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import sys
import tempfile
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Generic, Self, TypeVar, cast

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import DiagnosticSeverity

from ._dataset_schema import (
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    ResourceIdentity,
    Stage,
    _identifier,
)
from ._runtime import (
    CacheMode,
    LocalPath,
    ProgressEvent,
    Runtime,
    _local_path,
    _memory_bytes,
    _positive,
)

MetadataT = TypeVar("MetadataT")
ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")
# Set on threads that acquire sources ahead of the consuming thread.
_acquiring = threading.local()
ResourceT = TypeVar("ResourceT")


class DataValidationError(Exception):
    """An expected scientific-data failure, separate from operational exceptions."""

    def __init__(self, diagnostics: tuple[ExecutionDiagnostic, ...]) -> None:
        if (
            not isinstance(diagnostics, tuple)
            or not diagnostics
            or not all(isinstance(d, ExecutionDiagnostic) for d in diagnostics)
        ):
            raise TypeError(
                "diagnostics must be a nonempty tuple of ExecutionDiagnostic"
            )
        self.diagnostics = diagnostics
        super().__init__("; ".join(d.message for d in diagnostics))


class SourceContentChanged(OSError):
    """Acquired bytes differ from the content identity they were requested by."""

    def __init__(self, expected: str, received: str, *, uri: str | None = None) -> None:
        self.expected = expected
        self.received = received
        self.uri = uri
        served = "" if uri is None else f"{uri} served {received}, "
        super().__init__(
            f"source content identity mismatch: {served}"
            f"expected {expected}, received {received}"
            if uri is None
            else f"source content identity mismatch: {served}expected {expected}"
        )


@dataclass(slots=True)
class NodeStream(Generic[MetadataT]):
    schema: pa.Schema
    metadata: MetadataT
    batches: Iterator[pa.RecordBatch]
    report: Callable[[], ExecutionReport] | None = None
    final_metadata: Callable[[], MetadataT] | None = None
    close: Callable[[], None] | None = None
    checked: bool = False
    verify_before_yield: bool = False


class Node(ABC, Generic[MetadataT]):
    """A recipe node. Only its explicit open method can produce a stable stream."""

    key: str

    def _view_metadata(
        self, scope: ExecutionScope, replay_metadata: MetadataT
    ) -> MetadataT:
        return replay_metadata

    def _view_report(
        self, scope: ExecutionScope, replay_report: ExecutionReport
    ) -> ExecutionReport:
        return replay_report

    def _view_batch(
        self, scope: ExecutionScope, batch: pa.RecordBatch
    ) -> pa.RecordBatch:
        return batch

    @abstractmethod
    def open(self, scope: ExecutionScope) -> NodeStream[MetadataT]:
        raise NotImplementedError


class _Replay(Generic[MetadataT]):
    def __init__(self, stream: NodeStream[MetadataT], directory: Path) -> None:
        if not isinstance(stream.schema, pa.Schema):
            raise TypeError("node stream schema must be an Arrow schema")
        self.stream = stream
        self.directory = directory
        self.paths: list[Path] = []
        self.rows = 0
        self.exhausted = False
        self.closed = False
        self.pulling = False
        self.failure: BaseException | None = None
        self._metadata_snapshot = stream.metadata
        self._report_snapshot = ExecutionReport(False, None)

    def metadata(self) -> MetadataT:
        if not self.closed and self.stream.final_metadata is not None:
            self._metadata_snapshot = self.stream.final_metadata()
        return self._metadata_snapshot

    def report(self) -> ExecutionReport:
        if self.closed:
            return self._report_snapshot
        if self.stream.report is not None:
            base = self.stream.report()
            if not isinstance(base, ExecutionReport):
                raise TypeError("node report callback must return ExecutionReport")
        else:
            metadata_report = getattr(self.stream.metadata, "report", None)
            base = ExecutionReport(
                False,
                None,
                diagnostics=metadata_report.diagnostics
                if isinstance(metadata_report, ExecutionReport)
                else (),
                counts=ExecutionCounts(produced=self.rows),
                source_identities=metadata_report.source_identities
                if isinstance(metadata_report, ExecutionReport)
                else (),
            )
        diagnostics = base.diagnostics
        if isinstance(self.failure, DataValidationError):
            diagnostics += tuple(
                d for d in self.failure.diagnostics if d not in diagnostics
            )
        has_errors = any(d.severity is DiagnosticSeverity.ERROR for d in diagnostics)
        complete = self.exhausted and self.failure is None
        self._report_snapshot = ExecutionReport(
            complete,
            False if has_errors else True if complete else None,
            diagnostics,
            base.counts,
            base.source_identities,
        )
        return self._report_snapshot

    def _pull(self) -> bool:
        if self.pulling:
            raise RuntimeError("cyclic or reentrant graph computation")
        if self.closed:
            raise RuntimeError("execution scope is closed")
        if self.failure is not None:
            raise self.failure
        if self.exhausted:
            return False
        self.pulling = True
        try:
            batch = next(self.stream.batches)
            if not isinstance(batch, pa.RecordBatch):
                raise TypeError("graph nodes must yield Arrow RecordBatch values")
            if not batch.schema.equals(self.stream.schema, check_metadata=False):
                raise DataValidationError(
                    (
                        ExecutionDiagnostic(
                            "batch_schema_mismatch",
                            DiagnosticSeverity.ERROR,
                            Stage.VALIDATION,
                            "The batch schema differs from the declared stream schema.",
                        ),
                    )
                )
            batch = batch.replace_schema_metadata(self.stream.schema.metadata)
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{len(self.paths):012d}.arrow"
            with (
                pa.OSFile(str(path), "wb") as sink,
                pa.ipc.new_file(sink, batch.schema) as writer,
            ):
                writer.write_batch(batch)
            self.paths.append(path)
            self.rows += batch.num_rows
            return True
        except StopIteration:
            self.exhausted = True
            try:
                self.metadata()
                self.report()
            except BaseException as error:
                self.failure = error
                raise
            return False
        except BaseException as error:
            self.failure = error
            try:
                self.report()
            except BaseException:
                pass
            raise
        finally:
            self.pulling = False

    def batches(self, batch_rows: int) -> Iterator[pa.RecordBatch]:
        if self.stream.verify_before_yield:
            # A content pin can cover more source objects than the first batch.
            # The producer checks its digest before this replay can be read.
            while self._pull():
                pass
        index = 0
        while True:
            if self.closed:
                raise RuntimeError("execution scope is closed")
            if index == len(self.paths) and not self._pull():
                return
            with pa.OSFile(str(self.paths[index]), "rb") as source:
                batch = pa.ipc.open_file(source).get_batch(0)
            index += 1
            for offset in range(0, batch.num_rows, batch_rows):
                yield batch.slice(offset, batch_rows)

    def close(self) -> None:
        if self.closed:
            return
        first_error: BaseException | None = None
        try:
            close = getattr(self.stream.batches, "close", None)
            if close is not None:
                close()
        except BaseException as error:
            first_error = error
        try:
            if self.stream.close is not None:
                self.stream.close()
        except BaseException as error:
            first_error = first_error or error
        finally:
            try:
                self.metadata()
                self.report()
            finally:
                self.closed = True
        if first_error is not None:
            raise first_error


def _copy_handle_verified(
    reader: BinaryIO, destination: Path, expected: str | None
) -> str:
    digest = hashlib.sha256()
    position = reader.tell()
    created = False
    try:
        reader.seek(0)
        with destination.open("xb") as writer:
            created = True
            while chunk := reader.read(1024 * 1024):
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        actual = "sha256:" + digest.hexdigest()
        if expected is not None and actual != expected:
            raise SourceContentChanged(expected, actual)
        return actual
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        reader.seek(position)


def _clone(source: Path, destination: Path) -> bool:
    """Make `destination` a copy-on-write clone of `source`, where supported.

    A clone is an independent file, like a copy, but shares the source's
    blocks until either is written: no data is written and no space is used.
    APFS (macOS) and Btrfs/XFS (Linux) support it; elsewhere this returns
    False and nothing is created.
    """
    try:
        if sys.platform == "darwin":
            clonefile = _clonefile()
            if clonefile is None:
                return False
            if clonefile(os.fsencode(source), os.fsencode(destination), 0) != 0:
                return False
            return True
        if sys.platform.startswith("linux"):
            import fcntl

            with source.open("rb") as reader, destination.open("xb") as writer:
                try:
                    fcntl.ioctl(writer.fileno(), _FICLONE, reader.fileno())
                except OSError:
                    destination.unlink(missing_ok=True)
                    return False
            return True
    except OSError:
        return False
    return False


_FICLONE = 0x40049409


@functools.cache
def _clonefile() -> Any:
    import ctypes

    try:
        function = ctypes.CDLL(None, use_errno=True).clonefile
    except (AttributeError, OSError):
        return None
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
    function.restype = ctypes.c_int
    return function


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_verified(source: Path, destination: Path, expected: str | None) -> str:
    """Copy `source` to a new `destination` and return its verified checksum.

    A clone is used where the filesystem supports one; the clone itself is
    then read and verified, so the checksum is of the file that remains.
    """
    if _clone(source, destination):
        try:
            with destination.open("rb") as clone:
                os.fsync(clone.fileno())
            actual = _sha256(destination)
            if expected is not None and actual != expected:
                raise SourceContentChanged(expected, actual)
            return actual
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    with source.open("rb") as reader:
        return _copy_handle_verified(reader, destination, expected)


def _move_verified(source: Path, destination: Path, expected: str) -> None:
    """Verify `source` and rename it to a new `destination` in the same directory."""
    actual = _sha256(source)
    if actual != expected:
        raise SourceContentChanged(expected, actual)
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(source, destination)


class ExecutionScope:
    """Own source snapshots, lazy node replays, and optional database resources."""

    def __init__(self, runtime: Runtime | None = None) -> None:
        if runtime is not None and not isinstance(runtime, Runtime):
            raise TypeError("runtime must be Runtime")
        self.runtime = Runtime() if runtime is None else runtime
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._replays: dict[str, _Replay[object]] = {}
        self._open_failures: dict[str, BaseException] = {}
        self._opening: set[str] = set()
        self._pins: dict[str, Path] = {}
        self._source_memo: dict[
            tuple[str, str | None], tuple[Path, ResourceIdentity, int]
        ] = {}
        self._asset_results: dict[str, object] = {}
        self._local_identities: dict[str, ResourceIdentity] = {}
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._resources = ExitStack()
        self._closed = False
        self._entered = False
        self._callback_error: BaseException | None = None
        self._deferred_events: deque[ProgressEvent] = deque()
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def _key_lock(self, key: str) -> threading.Lock:
        """One lock per owned file, for acquisition threads sharing a scope."""
        with self._key_locks_guard:
            return self._key_locks.setdefault(key, threading.Lock())

    def __enter__(self) -> Self:
        self._active()
        if self._entered:
            raise RuntimeError("an execution scope cannot be entered twice")
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException:
            if exc is None:
                raise

    def _active(self) -> None:
        if self._closed:
            raise RuntimeError("execution scope is closed")

    @property
    def directory(self) -> Path:
        self._active()
        if self._temporary is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="crc-execution-")
        return Path(self._temporary.name)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        self._active()
        if self.runtime._connection is not None:
            return self.runtime._connection
        if self._connection is None:
            config: dict[str, str | bool | int | float | list[str]] = {
                "threads": self.runtime.workers,
                "temp_directory": str(self.directory / "duckdb"),
            }
            offline = self.runtime.cache.mode is CacheMode.OFFLINE
            if offline:
                config.update(
                    autoinstall_known_extensions=False,
                    autoload_known_extensions=False,
                )
            memory = _memory_bytes(self.runtime.memory_limit)
            if memory is not None:
                config["memory_limit"] = f"{memory}B"
            self._connection = duckdb.connect(config=config)
            if offline:
                self._connection.execute("SET enable_external_access = false")
        return self._connection

    def emit(self, stage: Stage, completed: int, total: int | None = None) -> None:
        self._active()
        event = ProgressEvent(stage, completed, total)
        if getattr(_acquiring, "active", False):
            # Progress callbacks run on the consuming thread, in order.
            self._deferred_events.append(event)
            return
        self.deliver_events()
        self._deliver(event)

    def deliver_events(self) -> None:
        """Report progress that acquisition threads recorded."""
        while self._deferred_events:
            self._deliver(self._deferred_events.popleft())

    def _deliver(self, event: ProgressEvent) -> None:
        if self.runtime._progress is not None:
            try:
                self.runtime._progress(event)
            except BaseException as error:
                self._callback_error = error
                raise

    def prefetched(
        self, items: Sequence[ItemT], acquire: Callable[[ItemT], ResultT]
    ) -> Generator[ResultT, None, None]:
        """`acquire(item)` for each item in order, acquiring ahead on threads.

        Up to `Runtime(workers=...)` items are acquired at once, never more than
        that many ahead of the consumer. Results and exceptions arrive in item
        order on the consuming thread, and progress callbacks run there too.
        Close the generator to stop early: waiting items are cancelled at
        once, and the scope waits for running ones when it closes.
        """
        workers = min(self.runtime.workers, len(items))
        if workers <= 1:
            for item in items:
                yield acquire(item)
            return
        self.directory  # noqa: B018 - created once, before threads use it

        def run(item: ItemT) -> ResultT:
            _acquiring.active = True
            try:
                return acquire(item)
            finally:
                _acquiring.active = False

        pool = ThreadPoolExecutor(workers, thread_name_prefix="crc-acquire")
        # Closing this generator never waits for its threads. The garbage
        # collector can close an abandoned generator on any thread, including
        # one that holds a lock the pool's threads need. The scope waits for
        # them when it closes instead.
        self._resources.callback(pool.shutdown)
        pending: deque[Future[ResultT]] = deque()
        submitted = 0
        try:
            for index in range(len(items)):
                while submitted < len(items) and submitted < index + workers:
                    pending.append(pool.submit(run, items[submitted]))
                    submitted += 1
                future = pending.popleft()
                try:
                    result = future.result()
                finally:
                    self.deliver_events()
                yield result
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def enter_context(self, resource: AbstractContextManager[ResourceT]) -> ResourceT:
        self._active()
        return self._resources.enter_context(resource)

    def callback(self, close: Callable[[], None]) -> None:
        self._active()
        self._resources.callback(close)

    def _node_key(self, node: Node[MetadataT]) -> str:
        if not isinstance(node, Node):
            raise TypeError("graph execution requires an explicit Node recipe")
        if not isinstance(node.key, str) or not node.key:
            raise ValueError("graph node key must be a nonempty computation identity")
        return node.key

    def _replay(self, node: Node[MetadataT]) -> _Replay[MetadataT]:
        key = self._node_key(node)
        if key in self._replays:
            return cast(_Replay[MetadataT], self._replays[key])
        if key in self._open_failures:
            raise self._open_failures[key]
        self._active()
        if key in self._opening:
            raise RuntimeError("cyclic graph dependency while opening a node")
        self._opening.add(key)
        try:
            stream = node.open(self)
            if not isinstance(stream, NodeStream):
                raise TypeError("Node.open must return NodeStream")
            from ._source_io import SourceNode

            if isinstance(node, SourceNode):
                stream = node.checked(self, stream)
            directory = (
                self.directory / "nodes" / hashlib.sha256(key.encode()).hexdigest()
            )
            replay = _Replay(stream, directory)
            self._replays[key] = cast(_Replay[object], replay)
            return replay
        except BaseException as error:
            self._open_failures[key] = error
            raise
        finally:
            self._opening.remove(key)

    def iter_batches(
        self, node: Node[MetadataT], *, batch_rows: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        self._active()
        rows = (
            self.runtime.batch_rows
            if batch_rows is None
            else _positive(batch_rows, "batch_rows")
        )
        for batch in self._replay(node).batches(rows):
            viewed = node._view_batch(self, batch)
            if not isinstance(viewed, pa.RecordBatch):
                raise TypeError("node batch views must return Arrow RecordBatch values")
            if not viewed.schema.equals(batch.schema, check_metadata=False):
                raise TypeError("node batch views cannot change the computation schema")
            yield viewed

    def metadata(self, node: Node[MetadataT]) -> MetadataT:
        replay = self._replay(node)
        return node._view_metadata(self, replay.metadata())

    def computation_metadata(self, node: Node[MetadataT]) -> MetadataT:
        """Read scope-local producer metadata without applying branch views."""
        return self._replay(node).metadata()

    def computation_report(self, node: Node[MetadataT]) -> ExecutionReport:
        """Read shared row-validation state without applying branch assertions."""
        key = self._node_key(node)
        if key in self._replays:
            return self._replays[key].report()
        error = self._open_failures.get(key)
        if isinstance(error, DataValidationError):
            return ExecutionReport(False, False, error.diagnostics)
        return ExecutionReport(False, None)

    def report(self, node: Node[MetadataT]) -> ExecutionReport:
        key = self._node_key(node)
        report = self.computation_report(node)
        if key in self._replays:
            report = node._view_report(self, report)
            if not isinstance(report, ExecutionReport):
                raise TypeError("node report views must return ExecutionReport")
        return report

    def complete(self, node: Node[MetadataT]) -> bool:
        key = self._node_key(node)
        replay = self._replays.get(key)
        return replay is not None and replay.exhausted and replay.failure is None

    def pin(self, resource: ResourceIdentity, acquire: Callable[[Path], None]) -> Path:
        self._active()
        if not isinstance(resource, ResourceIdentity) or not callable(acquire):
            raise TypeError("pin requires ResourceIdentity and an acquisition callable")
        if resource.sha256 in self._pins:
            return self._pins[resource.sha256]
        # Threads pinning equal content share its owned paths: one acquires,
        # the others then reuse its pin.
        with self._key_lock(resource.sha256):
            if resource.sha256 in self._pins:
                return self._pins[resource.sha256]
            return self._pin(resource, acquire)

    def _pin(self, resource: ResourceIdentity, acquire: Callable[[Path], None]) -> Path:
        digest = resource.sha256.removeprefix("sha256:")
        private = self.directory / f"source-{digest}"
        cache = self.runtime.cache
        cached = (
            None
            if cache.directory is None
            else cache.directory / "source-objects-1.0" / digest
        )
        if (
            cache.mode in (CacheMode.REUSE, CacheMode.OFFLINE)
            and cached is not None
            and cached.exists()
        ):
            _copy_verified(cached, private, resource.sha256)
        elif cache.mode is CacheMode.OFFLINE:
            raise FileNotFoundError(
                f"verified source is unavailable offline: {resource.uri} "
                f"({resource.sha256})"
            )
        else:
            acquired = self.directory / f"acquiring-{digest}"
            try:
                self.emit(Stage.ACQUISITION, 0, 1)
                acquire(acquired)
                try:
                    # Downloaded bytes become the snapshot without a copy.
                    _move_verified(acquired, private, resource.sha256)
                except SourceContentChanged as error:
                    raise SourceContentChanged(
                        error.expected, error.received, uri=resource.uri
                    ) from None
                self.emit(Stage.ACQUISITION, 1, 1)
                if cached is not None:
                    self._retain_snapshot(private, resource.sha256)
            except BaseException:
                private.unlink(missing_ok=True)
                raise
            finally:
                acquired.unlink(missing_ok=True)
        self._pins[resource.sha256] = private
        return private

    def _retain_snapshot(self, private: Path, digest: str) -> None:
        cache = self.runtime.cache
        if cache.directory is None or cache.mode is CacheMode.OFFLINE:
            return
        cached = cache.directory / "source-objects-1.0" / digest.removeprefix("sha256:")
        cached.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".acquiring-", dir=cached.parent)
        os.close(descriptor)
        staged = Path(name)
        staged.unlink()
        try:
            _copy_verified(private, staged, digest)
            os.replace(staged, cached)
        finally:
            staged.unlink(missing_ok=True)

    def pin_handle(
        self,
        reader: BinaryIO,
        *,
        uri: str,
        expected_sha256: str | None = None,
    ) -> tuple[Path, ResourceIdentity]:
        self._active()
        uri = _identifier(uri, "uri")
        if expected_sha256 is not None:
            if re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256) is None:
                raise ValueError("expected_sha256 must be a lowercase sha256: digest")
            local_key = f"{uri}:{expected_sha256}"
            resource = self._local_identities.get(local_key) or ResourceIdentity(
                uri, expected_sha256, retrieved_at=datetime.now(UTC)
            )

            def acquire(target: Path) -> None:
                _copy_handle_verified(reader, target, expected_sha256)

            pinned = self.pin(resource, acquire)
            self._local_identities[local_key] = resource
            return pinned, resource
        name = hashlib.sha256(uri.encode()).hexdigest()
        key = f"local:{name}"
        with self._key_lock(key):
            return self._pin_unverified(reader, uri, key, name)

    def _pin_unverified(
        self, reader: BinaryIO, uri: str, key: str, name: str
    ) -> tuple[Path, ResourceIdentity]:
        if key in self._pins:
            return self._pins[key], self._local_identities[key]
        private = self.directory / f"local-{name}"
        digest = _copy_handle_verified(reader, private, None)
        try:
            resource = ResourceIdentity(uri, digest, retrieved_at=datetime.now(UTC))
            self._retain_snapshot(private, digest)
        except BaseException:
            private.unlink(missing_ok=True)
            raise
        self._pins[key] = private
        self._local_identities[key] = resource
        return private, resource

    def pin_local(
        self, path: LocalPath, *, expected_sha256: str | None = None
    ) -> tuple[Path, ResourceIdentity]:
        self._active()
        source = _local_path(path)
        if expected_sha256 is not None:
            if re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256) is None:
                raise ValueError("expected_sha256 must be a lowercase sha256: digest")
            local_key = f"{source}:{expected_sha256}"
            identity = self._local_identities.get(local_key) or ResourceIdentity(
                str(source), expected_sha256, retrieved_at=datetime.now(UTC)
            )

            def acquire(target: Path) -> None:
                shutil.copyfile(source, target)

            pinned = self.pin(identity, acquire)
            self._local_identities[local_key] = identity
            return pinned, identity
        name = hashlib.sha256(str(source).encode()).hexdigest()
        key = f"local:{name}"
        if key in self._pins:
            return self._pins[key], self._local_identities[key]
        with source.open("rb") as reader:
            return self.pin_handle(reader, uri=str(source))

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        for replay in reversed(tuple(self._replays.values())):
            try:
                replay.close()
            except BaseException as error:
                first_error = first_error or error
        try:
            self._resources.close()
        except BaseException as error:
            first_error = first_error or error
        if self._connection is not None:
            try:
                self._connection.close()
            except BaseException as error:
                first_error = first_error or error
        if self._temporary is not None:
            try:
                self._temporary.cleanup()
            except BaseException as error:
                first_error = first_error or error
        self._closed = True
        if first_error is not None:
            raise first_error
