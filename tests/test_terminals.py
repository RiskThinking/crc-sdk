"""Terminal consumption, immutable results, and single-pass atomic publication."""

from __future__ import annotations

import base64
import errno
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    DiagnosticSeverity,
    Distribution,
    DistributionFamily,
    SampleFit,
)

from crc_sdk import Assets, Dataset, Evaluation, ReadResult, SpatialMatch
from crc_sdk._dataset_schema import (
    DatasetMetadata,
    ExecutionDiagnostic,
    ExecutionReport,
    SpatialContract,
    Stage,
    SupportKind,
)
from crc_sdk._declarations import DataSpec, SeriesSpec, Variable
from crc_sdk._errors import CapabilityError, ValidationError
from crc_sdk._graph import DataValidationError, ExecutionScope, Node, NodeStream
from crc_sdk._runtime import ProgressEvent, Runtime
from crc_sdk._schema import DatasetRow, metadata_from_table, rows_to_table
from crc_sdk._terminals import BatchStream, Execution, _schema

CELL = 594475228122316799


def metadata() -> DatasetMetadata:
    variable = Variable("user:quantity")
    return DatasetMetadata(
        spec=DataSpec([variable], series=[SeriesSpec(variable)]),
        spatial=SpatialContract(supports=(SupportKind.CELL,), index_resolution=4),
        report=ExecutionReport(False, None),
    )


def failure() -> DataValidationError:
    return DataValidationError(
        (
            ExecutionDiagnostic(
                "invalid_evidence",
                DiagnosticSeverity.ERROR,
                Stage.VALIDATION,
                "Scientific evidence is invalid.",
            ),
        )
    )


class InputNode(Node[DatasetMetadata]):
    key = "terminal-input"

    def __init__(
        self,
        events: list[str],
        *,
        fail_open: bool = False,
        fail_iteration: bool = False,
        operational: bool = False,
    ) -> None:
        self.events = events
        self.fail_open = fail_open
        self.fail_iteration = fail_iteration
        self.operational = operational

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        self.events.append("open")
        scope.callback(lambda: self.events.append("resource-close"))
        if self.operational:
            raise PermissionError("source cannot be read")
        if self.fail_open:
            raise failure()
        planned = metadata()
        schema = _schema(planned)

        def batches() -> Iterator[pa.RecordBatch]:
            try:
                for index in range(3):
                    self.events.append("batch:" + str(index))
                    if self.fail_iteration and index == 1:
                        raise failure()
                    table = rows_to_table(
                        [
                            DatasetRow(
                                "user:quantity",
                                f"support:{index}",
                                CELL,
                                curve=Distribution.point_mass(index),
                            ),
                        ],
                        planned,
                    )
                    yield table.to_batches()[0]
            finally:
                self.events.append("iterator-close")

        return NodeStream(schema, planned, batches())


@dataclass(frozen=True)
class Saved:
    path: Path
    metadata: DatasetMetadata
    sha256: str


class Recipe:
    def __init__(
        self, node: Node[DatasetMetadata], *, fail_saved: bool = False
    ) -> None:
        self._node = node
        self.fail_saved = fail_saved

    def _saved(self, path: Path, value: DatasetMetadata, digest: str) -> Saved:
        if self.fail_saved:
            raise RuntimeError("handle construction failed")
        return Saved(path, value, digest)


def test_workflow_annotations_resolve_through_public_imports() -> None:
    dataset = get_type_hints(Dataset.evaluate)
    assert dataset["assets"] is Assets
    assert dataset["spatial"] == SpatialMatch | None
    assert dataset["return"] is Evaluation
    assert get_type_hints(ReadResult.fit)["return"] is Dataset


def test_read_result_and_report_survive_scope_cleanup() -> None:
    events: list[str] = []
    execution = Execution(Recipe(InputNode(events)))
    with execution:
        result = execution.read()
        directory = execution._scope.directory
        report = execution.report
        assert result.table.num_rows == 3
        assert report.complete and report.valid is True
        assert result.metadata.report == result.report == report
        assert metadata_from_table(result.table) == result.metadata
    assert result.table.num_rows == 3
    assert result.report == execution.report == report
    assert result.metadata.spec.variables[0].id == "user:quantity"
    assert not directory.exists()
    assert events[-1] == "resource-close"


def test_validate_read_and_write_reuse_the_same_completed_graph(tmp_path: Path) -> None:
    events: list[str] = []
    with Execution(Recipe(InputNode(events))) as execution:
        assert execution.validate().valid is True
        assert execution.read().table.num_rows == 3
        saved = execution.write(tmp_path / "result.parquet")
        assert saved.path == tmp_path / "result.parquet"
    assert events.count("open") == 1
    for index in range(3):
        assert events.count("batch:" + str(index)) == 1


def test_unconsumed_execution_exits_without_source_acquisition() -> None:
    events: list[str] = []
    execution = Execution(Recipe(InputNode(events)))
    assert execution.report == ExecutionReport(False, None)
    with execution:
        assert execution.report == ExecutionReport(False, None)
    assert events == []
    assert execution.report == ExecutionReport(False, None)


@pytest.mark.parametrize("terminal", ["read", "validate", "write", "batches"])
def test_metadata_validation_failures_use_one_public_report(
    terminal: str,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    node = InputNode(events, fail_open=True)
    expected = ExecutionReport(False, False, failure().diagnostics)
    execution = Execution(Recipe(node))
    with execution:
        if terminal == "validate":
            assert execution.validate() == expected
        else:
            with pytest.raises(ValidationError) as caught:
                if terminal == "read":
                    execution.read()
                elif terminal == "write":
                    execution.write(tmp_path / "invalid.parquet")
                else:
                    with execution.batches():
                        pass
            assert caught.value.report == expected
        assert execution.report == expected
    assert events == ["open", "resource-close"]
    assert not list(tmp_path.iterdir())


def test_iteration_failure_report_survives_batch_stream_exit() -> None:
    events: list[str] = []
    stream = BatchStream(ExecutionScope(), InputNode(events, fail_iteration=True))
    with pytest.raises(ValidationError) as caught:
        with stream:
            iterator = iter(stream)
            first = next(iterator)
            assert first.num_rows == 1
            next(iterator)
    assert stream.report.valid is False
    assert not stream.report.complete
    assert caught.value.report == stream.report
    assert stream.metadata.report == stream.report
    assert events[-1] == "resource-close"
    assert first.num_rows == 1


def test_batch_stream_entry_failure_closes_owned_resources() -> None:
    events: list[str] = []
    stream = BatchStream(ExecutionScope(), InputNode(events, fail_open=True))
    with pytest.raises(ValidationError) as caught:
        with stream:
            pass
    assert stream.report == caught.value.report
    assert stream.report.valid is False
    assert events == ["open", "resource-close"]
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = stream.metadata


def test_streaming_yields_before_computing_remaining_batches_and_closes_early() -> None:
    events: list[str] = []
    stream = BatchStream(ExecutionScope(), InputNode(events))
    with stream:
        assert events == ["open"]
        assert stream.metadata.report.complete is False
        iterator = iter(stream)
        first = next(iterator)
        assert first.num_rows == 1
        assert events == ["open", "batch:0"]
        snapshot = stream.report
        assert snapshot.complete is False
        assert snapshot.valid is None
    assert stream.report.complete is False
    assert stream.report.valid is None
    assert events == ["open", "batch:0", "iterator-close", "resource-close"]
    assert snapshot.complete is False


def test_full_stream_is_complete_only_after_successful_exhaustion() -> None:
    events: list[str] = []
    stream = BatchStream(ExecutionScope(), InputNode(events), batch_rows=1)
    with stream:
        assert len(list(stream)) == 3
        assert stream.report.complete
        assert stream.report.valid is True
    assert stream.report.complete
    assert stream.metadata.report == stream.report


def test_truncated_replay_is_an_incomplete_result_and_stream() -> None:
    events: list[str] = []
    with Execution(Recipe(InputNode(events))) as execution:
        execution.validate()
        preview = execution.read(limit=1)
        assert preview.table.num_rows == 1
        assert preview.report.complete is False
        assert preview.report.valid is None
        spec = SampleFit(family=DistributionFamily.GUMBEL_RIGHT)
        with pytest.raises(CapabilityError, match="preview"):
            preview.fit(spec)
        with execution.batches() as stream:
            next(iter(stream))
        assert stream.report.complete is False
        assert stream.report.valid is None
        assert execution.read(limit=3).report.complete
    assert events.count("open") == 1


def test_first_boundary_preview_remains_incomplete_without_lookahead() -> None:
    events: list[str] = []
    with Execution(Recipe(InputNode(events))) as execution:
        result = execution.read(limit=1)
        assert result.report.complete is False
        assert result.report.valid is None
        assert events == ["open", "batch:0"]


def test_collected_output_memory_limit_keeps_streaming_available() -> None:
    events: list[str] = []
    with Execution(
        Recipe(InputNode(events)), runtime=Runtime(memory_limit=1)
    ) as execution:
        with pytest.raises(MemoryError, match="batches"):
            execution.read()
        with execution.batches() as stream:
            assert sum(batch.num_rows for batch in stream) == 3
        assert stream.report.valid is True
    assert events.count("open") == 1


@pytest.mark.parametrize("path_kind", ["remote", "exists", "directory", "broken-link"])
def test_publication_preflight_precedes_source_work(
    tmp_path: Path, path_kind: str
) -> None:
    events: list[str] = []
    path: str | Path
    overwrite = False
    error: type[Exception]
    if path_kind == "remote":
        path = "s3://bucket/result.parquet"
        error = ValueError
    elif path_kind == "exists":
        path = tmp_path / "exists.parquet"
        path.write_bytes(b"original")
        error = FileExistsError
    elif path_kind == "directory":
        path = tmp_path / "directory"
        path.mkdir()
        overwrite = True
        error = IsADirectoryError
    else:
        path = tmp_path / "broken.parquet"
        try:
            path.symlink_to(tmp_path / "absent")
        except NotImplementedError:
            pytest.skip("this operating system does not support symbolic links")
        except OSError as caught:
            if (
                caught.errno in (errno.EACCES, errno.EPERM, errno.ENOSYS)
                or getattr(caught, "winerror", None) == 1314
            ):
                pytest.skip("symbolic-link creation is unavailable for this user")
            raise
        error = FileExistsError
    with Execution(Recipe(InputNode(events))) as execution:
        with pytest.raises(error):
            execution.write(path, overwrite=overwrite)
    assert events == []


def test_write_finalizes_arrow_and_parquet_metadata_without_output_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []
    original_writer = pq.ParquetWriter
    original_batch = original_writer.write_batch
    original_metadata = original_writer.add_key_value_metadata

    def writer_factory(*args: object, **kwargs: object) -> pq.ParquetWriter:
        writes.append("writer")
        return original_writer(*args, **kwargs)

    def write_batch(self: object, *args: object, **kwargs: object) -> None:
        writes.append("batch")
        original_batch(self, *args, **kwargs)

    def write_metadata(self: object, *args: object, **kwargs: object) -> None:
        writes.append("metadata")
        original_metadata(self, *args, **kwargs)

    monkeypatch.setattr(original_writer, "write_batch", write_batch)
    monkeypatch.setattr(original_writer, "add_key_value_metadata", write_metadata)
    monkeypatch.setattr(pq, "ParquetWriter", writer_factory)
    events: list[str] = []
    destination = tmp_path / "result.parquet"
    with Execution(Recipe(InputNode(events))) as execution:
        saved = execution.write(destination)
        report = execution.report
    assert report.complete and report.valid is True
    assert writes.count("writer") == writes.count("metadata") == 1
    assert writes.count("batch") == 3
    table = pq.read_table(destination)
    stored = metadata_from_table(table)
    assert stored.report == report
    assert saved.metadata == stored
    assert table.num_rows == 3
    recovered = pq.read_schema(destination)
    assert recovered.equals(_schema(stored), check_metadata=False)
    assert recovered.metadata == _schema(stored).metadata
    footer = pq.read_metadata(destination).metadata
    embedded = pa.ipc.read_schema(
        pa.BufferReader(base64.b64decode(footer[b"ARROW:schema"]))
    )
    assert embedded.equals(_schema(stored), check_metadata=True)
    assert events.count("open") == 1
    assert len([event for event in events if event.startswith("batch:")]) == 3
    assert not list(tmp_path.glob(".result.parquet.*"))
    assert (
        saved.sha256 == "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest()
    )


def test_write_validation_or_cancellation_preserves_existing_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.parquet"
    destination.write_bytes(b"original bytes")
    events: list[str] = []
    with Execution(Recipe(InputNode(events, fail_iteration=True))) as execution:
        with pytest.raises(ValidationError):
            execution.write(destination, overwrite=True)
    assert destination.read_bytes() == b"original bytes"
    assert not list(tmp_path.glob(".result.parquet.*"))

    def cancel(event: ProgressEvent) -> None:
        if event.stage is Stage.WRITING and event.completed:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with Execution(
            Recipe(InputNode([])), runtime=Runtime(progress=cancel)
        ) as execution:
            execution.write(destination, overwrite=True)
    assert destination.read_bytes() == b"original bytes"
    assert not list(tmp_path.glob(".result.parquet.*"))


def test_handle_construction_failure_precedes_publication(tmp_path: Path) -> None:
    destination = tmp_path / "result.parquet"
    destination.write_bytes(b"original bytes")
    with Execution(Recipe(InputNode([]), fail_saved=True)) as execution:
        with pytest.raises(RuntimeError, match="handle"):
            execution.write(destination, overwrite=True)
    assert destination.read_bytes() == b"original bytes"
    assert not list(tmp_path.glob(".result.parquet.*"))


def test_operational_failure_category_is_retained() -> None:
    events: list[str] = []
    with Execution(Recipe(InputNode(events, operational=True))) as execution:
        with pytest.raises(PermissionError, match="source"):
            execution.read()
        assert execution.report.valid is None
    assert events == ["open", "resource-close"]


@pytest.mark.parametrize("terminal", ["read", "validate", "batches", "write"])
def test_final_branch_validation_report_is_never_accepted_as_success(
    terminal: str,
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    class InvalidView(InputNode):
        def _view_report(
            self,
            scope: ExecutionScope,
            replay_report: ExecutionReport,
        ) -> ExecutionReport:
            if replay_report.complete:
                return replace(
                    replay_report, valid=False, diagnostics=failure().diagnostics
                )
            return replay_report

    events: list[str] = []
    with Execution(Recipe(InvalidView(events))) as execution:
        if terminal == "validate":
            report = execution.validate()
            assert report.complete and report.valid is False
        else:
            with pytest.raises(ValidationError) as caught:
                if terminal == "read":
                    execution.read()
                elif terminal == "batches":
                    with execution.batches() as stream:
                        list(stream)
                else:
                    execution.write(tmp_path / "invalid.parquet")
            assert caught.value.report.valid is False
            assert caught.value.report.diagnostics == failure().diagnostics
    assert not list(tmp_path.iterdir())
