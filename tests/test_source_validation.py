"""Scientific sources validate globally and retain operational error boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk import _source_io
from crc_sdk._aggregation import Sampling
from crc_sdk._dataset_schema import (
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    ResourceIdentity,
    SourceIdentity,
    Stage,
)
from crc_sdk._declarations import Calendar, Period
from crc_sdk._errors import CapabilityError, ValidationError
from crc_sdk._evidence import evidence_arrow_schema, evidence_table
from crc_sdk._graph import DataValidationError, ExecutionScope, NodeStream
from crc_sdk._observed_decoders import observed_record
from crc_sdk._observed_sources import GHCNh, _ObservedNode, _Prepared
from crc_sdk._runtime import ProgressEvent, Runtime
from crc_sdk._source_io import native_data


@pytest.fixture
def native_station(monkeypatch: pytest.MonkeyPatch) -> GHCNh:
    def download(uri: str, path: Path) -> int:
        if uri.endswith("ghcnh-station-list.csv"):
            path.write_text(
                "GHCN_ID,NAME,LATITUDE,LONGITUDE,ELEVATION\nSTATION0001,Station,43.7,-79.4,100\n"
            )
        else:
            row = {
                "STATION": "STATION0001",
                "DATE": "2020-01-01T00:00:00",
                "LATITUDE": 43.7,
                "LONGITUDE": -79.4,
                "temperature": 10.0,
                "temperature_Quality_Code": "1",
                "temperature_Source_Code": "220",
            }
            pq.write_table(pa.Table.from_pylist([row, row]), path)
        return 200

    monkeypatch.setattr(_source_io, "_download", download)
    return GHCNh().select(stations="STATION0001", time=Period.window(2020, 2020))


def test_duplicate_native_records_across_batches_produce_typed_failed_reports(
    native_station: GHCNh,
) -> None:
    recipe = native_station.observations()
    report = recipe.validate(runtime=Runtime(batch_rows=1))
    assert report.valid is False and report.complete is False
    assert report.counts.produced == 1
    assert "duplicate" in report.diagnostics[-1].message
    assert report.diagnostics[-1].stage is Stage.VALIDATION
    assert report.source_identities and report.source_identities[0].resources
    with pytest.raises(ValidationError, match="duplicate") as error:
        recipe.read(runtime=Runtime(batch_rows=1))
    assert error.value.report.valid is False
    assert error.value.report.counts.produced == 1


def test_invalid_native_stream_keeps_existing_publication_and_closes_resources(
    native_station: GHCNh, tmp_path: Path
) -> None:
    destination = tmp_path / "existing.parquet"
    destination.write_bytes(b"existing artifact")
    execution = native_station.observations().execute(runtime=Runtime(batch_rows=1))
    directory: Path
    with pytest.raises(ValidationError, match="duplicate"):
        with execution:
            directory = execution._scope.directory
            execution.write(destination, overwrite=True)
    assert destination.read_bytes() == b"existing artifact"
    assert not directory.exists()
    assert list(tmp_path.iterdir()) == [destination]


def test_global_support_geometry_cannot_change_between_native_batches() -> None:
    node = _ObservedNode("usgs")
    metadata = node._metadata(_Prepared((), None, ()))
    first = observed_record(
        "usgs:daily_mean_discharge", "station", -79.4, 43.7, "2020-01-01", 1.0
    )
    second = observed_record(
        "usgs:daily_mean_discharge", "station", -78.4, 43.7, "2020-01-02", 2.0
    )
    first["support_id"] = second["support_id"] = "source-support"

    def batches() -> Iterator[pa.RecordBatch]:
        for row in (first, second):
            yield from evidence_table([row], metadata).to_batches()

    with ExecutionScope() as scope:
        checked = node.checked(
            scope, NodeStream(evidence_arrow_schema(metadata), metadata, batches())
        )
        with pytest.raises(DataValidationError, match="conflicting source geometry"):
            list(checked.batches)
        assert checked.report is not None and checked.report().counts.produced == 1


def test_source_check_preserves_final_science_diagnostics_counts_identities_and_close(
    tmp_path: Path,
) -> None:
    node = _ObservedNode("usgs")
    path = tmp_path / "native-data"
    path.write_bytes(b"native bytes")
    resource = ResourceIdentity(
        path.as_uri(),
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        license="public-domain",
        attribution="fixture source",
        retrieved_at=datetime.now(UTC),
    )
    source = SourceIdentity(
        provider="local", product="native_measurements", resources=(resource,)
    )
    warning = ExecutionDiagnostic(
        "source_note",
        cf.DiagnosticSeverity.WARNING,
        Stage.DECODING,
        "A native source note.",
    )
    original_report = ExecutionReport(
        False,
        None,
        (warning,),
        ExecutionCounts(
            requested=7,
            produced=999,
            no_data=999,
            skipped=3,
            usable_years=2,
            incomplete_years=4,
        ),
        (source,),
    )
    metadata = replace(
        node._metadata(_Prepared((), source, ())), report=original_report
    )
    final = replace(
        metadata,
        sampling=Sampling(
            calendar=Calendar.GREGORIAN, step=cf.Duration(1, unit=cf.TimeUnit.DAY)
        ),
    )
    closed: list[str] = []

    def batches() -> Iterator[pa.RecordBatch]:
        try:
            for when, value in (("2020-01-01", 1.0), ("2020-01-02", None)):
                row = observed_record(
                    "usgs:daily_mean_discharge", "station", -79.4, 43.7, when, value
                )
                yield from evidence_table([row], metadata).to_batches()
        finally:
            closed.append("generator")

    with ExecutionScope() as scope:
        checked = node.checked(
            scope,
            NodeStream(
                evidence_arrow_schema(metadata),
                metadata,
                batches(),
                report=lambda: original_report,
                final_metadata=lambda: final,
                close=lambda: closed.append("stream"),
            ),
        )
        list(checked.batches)
        assert checked.report is not None
        report = checked.report()
        assert report.counts == ExecutionCounts(
            requested=7,
            produced=2,
            no_data=1,
            skipped=3,
            usable_years=2,
            incomplete_years=4,
        )
        assert report.diagnostics == (warning,)
        assert report.source_identities == (source,)
        assert checked.final_metadata is not None and checked.final_metadata() is final
        assert checked.close is not None
        checked.close()
    assert closed == ["generator", "stream"]


def test_already_checked_sources_still_translate_native_decoder_errors() -> None:
    node = _ObservedNode("usgs")
    metadata = node._metadata(_Prepared((), None, ()))

    def malformed() -> Iterator[pa.RecordBatch]:
        raise ValueError("malformed native header")
        yield  # pragma: no cover

    with ExecutionScope() as scope:
        checked = node.checked(
            scope,
            NodeStream(
                evidence_arrow_schema(metadata), metadata, malformed(), checked=True
            ),
        )
        with pytest.raises(DataValidationError, match="malformed native header"):
            list(checked.batches)


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("bad native value"),
        TypeError("bad native type"),
        KeyError("missing_native_field"),
        IndexError("short_native_record"),
        ParseError("bad XML"),
        BadZipFile("bad archive"),
    ],
)
def test_native_parse_errors_have_a_typed_scientific_category(
    failure: Exception,
) -> None:
    with pytest.raises(DataValidationError) as error:
        with native_data(source="native-source"):
            raise failure
    diagnostic = error.value.diagnostics[0]
    assert diagnostic.code == "invalid_source_data"
    assert diagnostic.stage is Stage.DECODING
    assert diagnostic.source == "native-source"
    assert error.value.__cause__ is failure


@pytest.mark.parametrize(
    "failure",
    [
        CapabilityError("choose an indicator"),
        OSError("offline cache unavailable"),
        MemoryError("source memory limit"),
        KeyboardInterrupt(),
    ],
)
def test_configuration_io_memory_and_cancellation_are_not_scientific_data_errors(
    failure: BaseException,
) -> None:
    with pytest.raises(type(failure)) as error:
        with native_data(source="native-source"):
            raise failure
    assert error.value is failure


@pytest.mark.parametrize("stage", [Stage.ACQUISITION, Stage.VALIDATION])
@pytest.mark.parametrize(
    "failure", [ValueError("callback cancelled"), TypeError("callback cancelled")]
)
def test_source_callback_errors_preserve_identity_and_cleanup(
    native_station: GHCNh, stage: Stage, failure: Exception
) -> None:
    def progress(event: ProgressEvent) -> None:
        if event.stage is stage:
            raise failure

    execution = native_station.observations().execute(
        runtime=Runtime(batch_rows=1, progress=progress)
    )
    directory: Path
    with pytest.raises(type(failure)) as error:
        with execution:
            directory = execution._scope.directory
            execution.read()
    assert error.value is failure
    assert not directory.exists()
    assert vars(failure) == {}


def test_native_context_preserves_callback_errors_inside_acquisition_context() -> None:
    failure = ValueError("callback cancelled acquisition")

    def progress(event: ProgressEvent) -> None:
        raise failure

    with ExecutionScope(Runtime(progress=progress)) as scope:
        with pytest.raises(ValueError) as error:
            with native_data(scope=scope, source="native-source"):
                scope.emit(Stage.ACQUISITION, 0)
        assert error.value is failure
