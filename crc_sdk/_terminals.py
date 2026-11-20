"""Owned result terminals for the shared lazy execution graph."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import types
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import crc_framework
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

import crc_sdk

from ._aggregation import Aggregation, AnnualMaximum, AnnualMinimum, DailyTotal
from ._dataset_schema import DatasetMetadata, ExecutionReport, Stage
from ._errors import CapabilityError, ValidationError
from ._evidence import EvidenceKind, EvidenceMetadata, FitFailure
from ._graph import DataValidationError, ExecutionScope, Node
from ._inspection import ArtifactMetadata
from ._runtime import (
    LocalPath,
    Runtime,
    _arrow_nbytes,
    _local_path,
    _memory_bytes,
    _positive,
)
from ._schema import dataset_arrow_schema

if TYPE_CHECKING:
    import pandas  # type: ignore[import-untyped]

MetadataT = TypeVar("MetadataT", bound=ArtifactMetadata)
ArtifactT = TypeVar("ArtifactT", covariant=True)
SampleFitSpec: TypeAlias = crc_framework.SampleFit | crc_framework.ModelSelection
QuantileFitSpec: TypeAlias = (
    crc_framework.QuantileFit
    | crc_framework.CensoredQuantileFit
    | crc_framework.HurdleQuantileFit
    | crc_framework.ModelSelection
)


def _with_report(metadata: MetadataT, report: ExecutionReport) -> MetadataT:
    return replace(metadata, report=report)


def _schema(metadata: ArtifactMetadata) -> pa.Schema:
    if isinstance(metadata, DatasetMetadata):
        return dataset_arrow_schema(metadata)
    if isinstance(metadata, EvidenceMetadata):
        from ._evidence import evidence_arrow_schema

        return evidence_arrow_schema(metadata)
    from ._collection_schema import CollectionMetadata, collection_arrow_schema
    from ._evaluation_schema import evaluation_arrow_schema

    if isinstance(metadata, CollectionMetadata):
        return collection_arrow_schema(metadata)
    return evaluation_arrow_schema(metadata)


def _unit(measurement: crc_framework.Measurement | None) -> str:
    if measurement is None or measurement.unit is None:
        return "unknown"
    return str(getattr(measurement.unit, "value", measurement.unit))


def _limit(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    return value


class _Recipe(Protocol[MetadataT, ArtifactT]):
    @property
    def _node(self) -> Node[MetadataT]: ...

    def _saved(self, path: Path, metadata: MetadataT, digest: str) -> ArtifactT: ...


@dataclass(frozen=True, slots=True)
class ReadResult(Generic[MetadataT]):
    table: pa.Table
    metadata: MetadataT
    report: ExecutionReport

    @overload
    def fit(
        self: ReadResult[DatasetMetadata],
        spec: QuantileFitSpec,
        *,
        on_failure: FitFailure = FitFailure.RAISE,
    ) -> crc_sdk.Dataset: ...

    @overload
    def fit(
        self: ReadResult[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]],
        spec: SampleFitSpec,
        *,
        on_failure: FitFailure = FitFailure.RAISE,
    ) -> crc_sdk.Dataset: ...

    def fit(
        self: ReadResult[DatasetMetadata]
        | ReadResult[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]],
        spec: QuantileFitSpec | SampleFitSpec,
        *,
        on_failure: FitFailure = FitFailure.RAISE,
    ) -> crc_sdk.Dataset:
        if not self.report.complete:
            raise CapabilityError(
                "A bounded preview is incomplete evidence. Call read() without a "
                "limit, then fit the completed result."
            )
        from ._recipes import Dataset, Samples

        if isinstance(self.metadata, EvidenceMetadata):
            if self.metadata.kind is not EvidenceKind.SAMPLES:
                raise CapabilityError(
                    "Only genuine sample evidence can be fitted; "
                    "aggregate raw observations first"
                )
            return Samples._collected(self.table, self.metadata).fit(
                cast(SampleFitSpec, spec), on_failure=on_failure
            )
        if isinstance(self.metadata, DatasetMetadata):
            return Dataset._collected(self.table, self.metadata).fit(
                cast(QuantileFitSpec, spec), on_failure=on_failure
            )
        raise CapabilityError("Fitting requires sample or quantile evidence")

    @overload
    def aggregate(
        self: ReadResult[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]],
        spec: DailyTotal,
    ) -> crc_sdk.Observations: ...

    @overload
    def aggregate(
        self: ReadResult[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]],
        spec: AnnualMaximum | AnnualMinimum,
    ) -> crc_sdk.Samples: ...

    @overload
    def aggregate(
        self: ReadResult[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]],
        spec: Aggregation,
    ) -> crc_sdk.Samples | crc_sdk.Observations: ...

    def aggregate(
        self: ReadResult[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]],
        spec: Aggregation,
    ) -> crc_sdk.Samples | crc_sdk.Observations:
        if not self.report.complete:
            raise CapabilityError(
                "A bounded preview is incomplete evidence. "
                "Call read() without a limit before aggregation."
            )
        from ._evidence_recipes import Observations

        if isinstance(self.metadata, EvidenceMetadata):
            if self.metadata.kind is EvidenceKind.OBSERVATIONS:
                return Observations._collected(self.table, self.metadata).aggregate(
                    spec
                )
        raise CapabilityError("Temporal aggregation requires observations")

    def to_pandas(self) -> pandas.DataFrame:
        frame = self.table.to_pandas()
        frame.attrs["crc"] = json.loads(self.metadata.to_json())
        frame.attrs["crc_report"] = self.report._document()
        return frame

    def evaluate(
        self: ReadResult[DatasetMetadata] | ReadResult[crc_sdk.CollectionMetadata],
        assets: crc_sdk.Assets,
        *,
        spatial: crc_sdk.SpatialMatch | None = None,
    ) -> crc_sdk.Evaluation:
        if not self.report.complete:
            raise CapabilityError(
                "Call read() without a limit before evaluating collected data."
            )
        from ._collection import DatasetCollection
        from ._collection_schema import CollectionMetadata
        from ._recipes import Dataset

        if isinstance(self.metadata, CollectionMetadata):
            return DatasetCollection._collected(self.table, self.metadata).evaluate(
                assets, spatial=spatial
            )
        if isinstance(self.metadata, DatasetMetadata):
            return Dataset._collected(self.table, self.metadata).evaluate(
                assets, spatial=spatial
            )
        raise CapabilityError("Evaluation requires a dataset or dataset collection")

    def __repr__(self) -> str:
        from ._evaluation_schema import EvaluationMetadata, QueryKind

        if isinstance(self.metadata, EvaluationMetadata):
            query = self.metadata.query.kind.value
            if self.metadata.query.kind is QueryKind.RETURN_PERIOD:
                query += " (years)"
            units: dict[str | None, str] = {
                response.name: _unit(response.output)
                for response in self.metadata.responses
            }
            if self.metadata.include_exposure:
                exposure = {_unit(v.measurement) for v in self.metadata.spec.variables}
                units[None] = (
                    next(iter(exposure))
                    if len(exposure) == 1
                    else "mixed"
                    if exposure
                    else "unknown"
                )
            return (
                f"ReadResult(rows={self.table.num_rows}, query={query!r}, "
                f"units={units!r}, "
                f"complete={self.report.complete}, valid={self.report.valid})\n"
                f"{self.table!r}"
            )
        value_units = {
            _unit(v.measurement)
            for v in self.metadata.spec.variables
            if v.measurement is not None and v.measurement.unit is not None
        }
        unit = (
            next(iter(value_units))
            if len(value_units) == 1
            else "mixed"
            if value_units
            else "unknown"
        )
        return (
            f"ReadResult(rows={self.table.num_rows}, unit={unit!r}, "
            f"complete={self.report.complete}, valid={self.report.valid})\n"
            f"{self.table!r}"
        )


class BatchStream(Generic[MetadataT]):
    def __init__(
        self,
        scope: ExecutionScope,
        node: Node[MetadataT],
        *,
        batch_rows: int | None = None,
        owned: bool = True,
    ) -> None:
        if batch_rows is not None:
            _positive(batch_rows, "batch_rows")
        self._scope = scope
        self._node: Node[MetadataT] = node
        self._batch_rows = batch_rows
        self._owned = owned
        self._iterator: Iterator[pa.RecordBatch] | None = None
        self._entered = False
        self._closed = False
        self._metadata: MetadataT | None = None
        self._report = ExecutionReport(False, None)
        self._exhausted = False
        self._iterating = False

    def __enter__(self) -> Self:
        if self._entered or self._closed:
            raise RuntimeError("a batch stream can only be entered once")
        self._entered = True
        try:
            self._metadata = self._scope.metadata(self._node)
            self._iterator = self._scope.iter_batches(
                self._node, batch_rows=self._batch_rows
            )
            return self
        except BaseException as error:
            self._report = self._scope.report(self._node)
            self._closed = True
            if self._owned:
                try:
                    self._scope.close()
                except BaseException:
                    pass
            if isinstance(error, DataValidationError):
                raise ValidationError(report=self._report) from error
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            if self._iterator is not None:
                close = getattr(self._iterator, "close", None)
                if close is not None:
                    close()
        except BaseException as caught:
            error = caught
        if self._owned:
            try:
                self._scope.close()
            except BaseException as caught:
                error = error or caught
        try:
            self._report = self._snapshot_report()
            if self._metadata is not None:
                self._metadata = _with_report(
                    self._scope.metadata(self._node), self._report
                )
        except BaseException as caught:
            error = error or caught
        self._closed = True
        if error is not None and exc is None:
            if isinstance(error, DataValidationError):
                raise ValidationError(report=self._report) from error
            raise error

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        if not self._entered or self._closed or self._iterator is None:
            raise RuntimeError(
                "iterate batches inside 'with recipe.batches() as stream:'"
            )
        if self._iterating:
            raise RuntimeError("a batch stream has one active iterator")
        self._iterating = True
        try:
            yield from self._iterator
            self._exhausted = True
            if self.report.valid is False:
                raise ValidationError(report=self.report)
        except DataValidationError as error:
            raise ValidationError(report=self.report) from error
        finally:
            self._iterating = False

    def _snapshot_report(self) -> ExecutionReport:
        report = self._scope.report(self._node)
        if not self._exhausted:
            return replace(
                report,
                complete=False,
                valid=False if report.valid is False else None,
            )
        return report

    @property
    def metadata(self) -> MetadataT:
        if not self._entered:
            raise RuntimeError("enter the batch stream to inspect its metadata")
        if self._closed:
            if self._metadata is None:
                raise RuntimeError(
                    "stream metadata is unavailable because opening failed"
                )
            return self._metadata
        return _with_report(self._scope.metadata(self._node), self.report)

    @property
    def report(self) -> ExecutionReport:
        if self._closed:
            return self._report
        if not self._entered:
            return self._report
        return self._snapshot_report()


class Execution(Generic[MetadataT, ArtifactT]):
    def __init__(
        self, recipe: _Recipe[MetadataT, ArtifactT], *, runtime: Runtime | None = None
    ) -> None:
        self._recipe = recipe
        self._scope = ExecutionScope(runtime)
        self._entered = False
        self._closed = False
        self._report = ExecutionReport(False, None)

    def __enter__(self) -> Self:
        if self._entered or self._closed:
            raise RuntimeError("an execution can only be entered once")
        self._scope.__enter__()
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        try:
            self._report = self.report
        finally:
            self._closed = True
            self._scope.__exit__(exc_type, exc, traceback)

    def _active(self) -> None:
        if not self._entered or self._closed:
            raise RuntimeError("use execution terminals inside 'with recipe.execute()'")

    @property
    def report(self) -> ExecutionReport:
        if self._closed:
            return self._report
        if not self._entered:
            return self._report
        return self._scope.report(self._recipe._node)

    def read(self, *, limit: int | None = None) -> ReadResult[MetadataT]:
        self._active()
        _limit(limit)
        node = self._recipe._node
        batches: list[pa.RecordBatch] = []
        count = 0
        collected_bytes = 0
        memory_limit = _memory_bytes(self._scope.runtime.memory_limit)
        try:
            metadata = self._scope.metadata(node)
            if limit != 0:
                iterator = self._scope.iter_batches(node)
                try:
                    for batch in iterator:
                        if limit is not None:
                            batch = batch.slice(0, limit - count)
                        if memory_limit is not None:
                            collected_bytes += _arrow_nbytes(batch)
                            if collected_bytes > memory_limit:
                                raise MemoryError(
                                    "Collected output exceeds "
                                    "Runtime(memory_limit=...). "
                                    "Use batches() or write() for streaming output."
                                )
                        batches.append(batch)
                        count += batch.num_rows
                        if limit is not None and count >= limit:
                            break
                finally:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
        except DataValidationError as error:
            raise ValidationError(report=self.report) from error
        report = self.report
        if report.valid is False:
            raise ValidationError(report=report)
        # A replay may be complete while this particular collection is bounded.
        if limit is not None and count < report.counts.produced:
            report = replace(
                report, complete=False, valid=None if report.valid else report.valid
            )
        try:
            metadata = _with_report(self._scope.metadata(node), report)
        except DataValidationError as error:
            raise ValidationError(report=self.report) from error
        table = pa.Table.from_batches(
            batches, schema=_schema(metadata)
        ).replace_schema_metadata(_schema(metadata).metadata)
        return ReadResult(table, metadata, report)

    def batches(self, *, batch_rows: int | None = None) -> BatchStream[MetadataT]:
        self._active()
        return BatchStream(
            self._scope, self._recipe._node, batch_rows=batch_rows, owned=False
        )

    def validate(self) -> ExecutionReport:
        self._active()
        try:
            for _ in self._scope.iter_batches(self._recipe._node):
                pass
        except DataValidationError:
            pass
        return self.report

    def write(self, path: LocalPath, *, overwrite: bool = False) -> ArtifactT:
        from ._evaluation_schema import (
            EvaluationMetadata,
            _check_evaluation_fields,
            _convert_parquet_assets,
            _evaluation_parquet_schema,
        )

        self._active()
        destination = _publication_path(path, overwrite)
        node = self._recipe._node
        try:
            metadata = self._scope.metadata(node)
        except DataValidationError as error:
            raise ValidationError(report=self.report) from error
        evaluation = isinstance(metadata, EvaluationMetadata)
        schema = _schema(metadata)
        storage_schema = _evaluation_parquet_schema(schema) if evaluation else schema
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".parquet", dir=destination.parent
        )
        os.close(descriptor)
        staging = Path(name)
        try:
            try:
                # Metadata is finalized in the footer after the only output pass.
                with pq.ParquetWriter(
                    staging, storage_schema.remove_metadata()
                ) as writer:
                    written = 0
                    for batch in self._scope.iter_batches(node):
                        if evaluation:
                            batch = _convert_parquet_assets(
                                batch, storage_schema, writing=True
                            )
                        writer.write_batch(batch.replace_schema_metadata(None))
                        written += batch.num_rows
                        self._scope.emit(Stage.WRITING, written)
                    report = self.report
                    if report.valid is not True or not report.complete:
                        raise ValidationError(report=report)
                    metadata = _with_report(self._scope.metadata(node), report)
                    schema = _schema(metadata)
                    footer = dict(schema.metadata or {})
                    storage_schema = storage_schema.with_metadata(schema.metadata)
                    footer[b"ARROW:schema"] = base64.b64encode(
                        storage_schema.serialize().to_pybytes()
                    )
                    writer.add_key_value_metadata(footer)
                    self._scope.emit(Stage.WRITING, written, written)
            except DataValidationError as error:
                raise ValidationError(report=self.report) from error
            if evaluation:
                with pq.ParquetFile(staging) as stored:
                    _check_evaluation_fields(stored.schema_arrow, schema, parquet=True)
            with staging.open("rb") as handle:
                os.fsync(handle.fileno())
                digest = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
            result = self._recipe._saved(destination, metadata, digest)
            if overwrite:
                os.replace(staging, destination)
            else:
                os.link(staging, destination)
                staging.unlink()
            return result
        finally:
            staging.unlink(missing_ok=True)


def _publication_path(path: LocalPath, overwrite: bool) -> Path:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    destination = _local_path(path)
    if os.path.lexists(destination) and not overwrite:
        raise FileExistsError(destination)
    if destination.is_dir():
        raise IsADirectoryError(destination)
    existing_parent = destination.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise NotADirectoryError(existing_parent)
    return destination
