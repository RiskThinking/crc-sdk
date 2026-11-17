"""Owned long-form evidence and explicit sample modelling."""

from __future__ import annotations

import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import TypedDict, cast

import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import (
    DiagnosticSeverity,
    FitError,
    FitSpec,
    ModelSelection,
    SampleFit,
    fit_samples,
)
from referencing.jsonschema import DRAFT202012
from shapely import from_wkb  # type: ignore[import-untyped]

from ._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    FittingDiagnostic,
    RetainedFailure,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
    Transformation,
    _dump,
    _identifier,
    _load,
    _number,
    _object,
    _report,
    _source,
    _spatial,
    _transformation,
)
from ._declarations import Calendar, DataSpec, _data_spec
from ._schema.dataset import DatasetRow, _cell, _geometry, validate_rows

_METADATA_KEY = b"crc.evidence"
_IDENTITY = ("variable_id", "scenario_id", "period_id", "support_id", "cell_id")


class _RowIdentity(TypedDict):
    variable_id: str
    scenario_id: str | None
    period_id: str | None
    support_id: str
    cell_id: int
    geometry: bytes | None


def _sample_time(value: object, calendar: Calendar | None = None) -> str:
    if isinstance(value, bool):
        raise ValueError("sample time must identify an observation year or date")
    if isinstance(value, int):
        if not 1 <= value <= 9999:
            raise ValueError("sample year must be between 1 and 9999")
        return f"{value:04d}"
    if isinstance(value, (date, datetime)):
        value = value.isoformat()
    if not isinstance(value, str):
        raise ValueError("sample time must identify an observation year or date")
    match = re.fullmatch(
        r"(?P<year>\d{4})(?:-(?P<month>\d{2})-(?P<day>\d{2})"
        r"(?:[Tt](?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
        r"(?:\.\d+)?(?:[Zz]|[+-](?P<offset_hour>\d{2}):"
        r"(?P<offset_minute>\d{2}))?)?)?",
        value,
    )
    if match is None or int(match["year"]) == 0:
        raise ValueError("sample time must identify an observation year or date")
    if match["month"] is not None:
        year, month, day = (int(match[k]) for k in ("year", "month", "day"))
        if not 1 <= month <= 12:
            raise ValueError("sample time has an invalid month")
        if calendar is Calendar.DAYS_360:
            maximum = 30
        elif calendar is None:
            # Accept dates possible in at least one supported calendar without
            # claiming Gregorian semantics for an undeclared source calendar.
            maximum = (31, 30, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]
        else:
            leap = (
                calendar is Calendar.ALL_LEAP
                or (calendar is Calendar.JULIAN and year % 4 == 0)
                or (
                    calendar in (Calendar.GREGORIAN, Calendar.PROLEPTIC_GREGORIAN)
                    and year % 4 == 0
                    and (year % 100 != 0 or year % 400 == 0)
                )
            )
            maximum = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[
                month - 1
            ]
        if not 1 <= day <= maximum:
            raise ValueError("sample time has an invalid day for its declared calendar")
    for field, maximum in (
        ("hour", 23),
        ("minute", 59),
        ("second", 59),
        ("offset_hour", 23),
        ("offset_minute", 59),
    ):
        if match[field] is not None and int(match[field]) > maximum:
            raise ValueError("sample time has an invalid clock or UTC offset")
    return value.replace("t", "T").replace("z", "Z")


class EvidenceKind(Enum):
    SAMPLES = "samples"
    OBSERVATIONS = "observations"
    PROJECTIONS = "projections"
    LAYER = "layer"


class FitFailure(Enum):
    RAISE = "raise"
    KEEP_NO_DATA = "keep_no_data"
    SKIP = "skip"


class _SampleFitError(FitError):
    """Numerical fitting failure with its known data-group identity."""

    def __init__(
        self,
        error: FitError,
        *,
        variable_id: str,
        support_id: str,
        source: str | None,
    ) -> None:
        super().__init__(
            error.message, record=error.record, diagnostics=error.diagnostics
        )
        self.code = error.code
        self.variable_id = _identifier(variable_id, "variable_id")
        self.support_id = _identifier(support_id, "support_id")
        self.source = None if source is None else _identifier(source, "source")


def evidence_metadata_schema() -> dict[str, object]:
    """Generate the persisted header schema from the shared scientific metadata."""
    from ._schema.contract import dataset_json_schema

    dataset = dataset_json_schema()
    definitions = cast(dict[str, dict[str, object]], dataset["$defs"])
    metadata = deepcopy(definitions["metadata"])
    properties = cast(dict[str, object], metadata["properties"])
    properties["schema_version"] = {"const": "1.0"}
    properties["kind"] = {"enum": [kind.value for kind in EvidenceKind]}
    cast(list[str], metadata["required"]).append("kind")
    required: set[str] = set()

    def references(schema: object) -> None:
        if not isinstance(schema, dict):
            return
        reference = schema.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference[len("#/$defs/") :].split("/", 1)[0]
            if name not in required:
                required.add(name)
                references(definitions[name])
        for child in DRAFT202012.subresources_of(schema):
            references(child)

    references(metadata)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/evidence/1.0",
        "title": "Canonical evidence metadata",
        **metadata,
        "$defs": {name: definitions[name] for name in sorted(required)},
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceMetadata:
    kind: EvidenceKind
    spec: DataSpec
    spatial: SpatialContract
    report: ExecutionReport
    source: SourceIdentity | None = None
    provenance: tuple[Transformation, ...] = ()

    @property
    def schema_version(self) -> str:
        return "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceKind):
            raise TypeError("kind must be EvidenceKind")
        # Scientific declarations and transformation history have one validator.
        DatasetMetadata(
            spec=self.spec,
            spatial=self.spatial,
            report=self.report,
            source=self.source,
            provenance=self.provenance,
        )

    def to_json(self) -> str:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "spec": self.spec._document(),
            "spatial": self.spatial._document(),
            "report": self.report._document(),
            "provenance": [p._document() for p in self.provenance],
        }
        if self.source is not None:
            value["source"] = self.source._document()
        return _dump(value)

    @classmethod
    def from_json(cls, text: str) -> EvidenceMetadata:
        value = _object(
            _load(text),
            {"schema_version", "kind", "spec", "spatial", "report", "provenance"},
            {"source"},
        )
        if value["schema_version"] != "1.0":
            raise ValueError("expected evidence schema 1.0")
        provenance = value["provenance"]
        if not isinstance(provenance, list):
            raise ValueError("provenance must be an array")
        return cls(
            kind=EvidenceKind(value["kind"]),
            spec=_data_spec(value["spec"]),
            spatial=_spatial(value["spatial"]),
            report=_report(value["report"]),
            source=None if "source" not in value else _source(value["source"]),
            provenance=tuple(_transformation(p) for p in provenance),
        )


def evidence_arrow_schema(metadata: EvidenceMetadata | None = None) -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("variable_id", pa.string(), nullable=False),
            pa.field("scenario_id", pa.string()),
            pa.field("period_id", pa.string()),
            pa.field("support_id", pa.string(), nullable=False),
            pa.field("cell_id", pa.uint64(), nullable=False),
            pa.field("geometry", pa.binary()),
            pa.field("time", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
        ]
    )
    if metadata is not None:
        return schema.with_metadata({_METADATA_KEY: metadata.to_json().encode()})
    return schema


def metadata_from_evidence_table(table: pa.Table) -> EvidenceMetadata:
    value = (table.schema.metadata or {}).get(_METADATA_KEY)
    if value is None:
        raise ValueError("canonical evidence metadata is missing")
    return EvidenceMetadata.from_json(value.decode("utf-8"))


def evidence_table(
    records: Sequence[dict[str, object]], metadata: EvidenceMetadata
) -> pa.Table:
    fields = set(evidence_arrow_schema().names)
    if any(set(record) != fields for record in records):
        raise ValueError("evidence records must contain exactly the canonical fields")
    table = pa.Table.from_pylist(list(records), schema=evidence_arrow_schema(metadata))
    validate_evidence(table, metadata)
    return table


def validate_evidence(
    table: pa.Table,
    metadata: EvidenceMetadata | None = None,
    *,
    check_counts: bool = True,
) -> ExecutionReport:
    if not isinstance(table, pa.Table):
        raise TypeError("evidence must be an Arrow Table")
    if not table.schema.equals(evidence_arrow_schema(), check_metadata=False):
        raise ValueError("Arrow fields do not match canonical evidence schema 1.0")
    stored = metadata_from_evidence_table(table)
    if metadata is not None and stored.to_json() != metadata.to_json():
        raise ValueError("supplied and stored evidence metadata disagree")
    metadata = stored
    if metadata.kind is not EvidenceKind.SAMPLES:
        raise ValueError("this evidence layout contains samples")
    series = {
        (
            s.variable.id,
            None if s.scenario is None else s.scenario.id,
            None if s.period is None else s.period.id,
        ): s
        for s in metadata.spec.series
    }
    keys: set[tuple[object, ...]] = set()
    supports: dict[str, tuple[bytes | None, int | None]] = {}
    for record in table.to_pylist():
        for field in ("variable_id", "support_id", "time"):
            _identifier(record[field], field)
        for field in ("scenario_id", "period_id"):
            if record[field] is not None:
                _identifier(record[field], field)
        identity = tuple(record[name] for name in _IDENTITY[:3])
        if identity not in series:
            raise ValueError(f"sample refers to an undeclared series: {identity!r}")
        declared_period = series[identity].period
        calendar = None if declared_period is None else declared_period.calendar
        if _sample_time(record["time"], calendar) != record["time"]:
            raise ValueError("sample time must use canonical uppercase T and Z")
        cell = _cell(record["cell_id"])
        if metadata.spatial.index_resolution is not None and (
            (cell >> 52) & 15 != metadata.spatial.index_resolution
        ):
            raise ValueError("cell_id resolution contradicts the spatial contract")
        geometry = record["geometry"]
        kind = SupportKind.CELL if geometry is None else _geometry(geometry)
        if kind not in metadata.spatial.supports:
            raise ValueError("geometry contradicts the declared spatial support")
        support = (geometry, cell if geometry is None else None)
        previous = supports.setdefault(record["support_id"], support)
        if previous != support:
            previous_geometry, previous_cell = previous
            if (
                previous_geometry is None
                or geometry is None
                or previous_cell != support[1]
                or not from_wkb(previous_geometry).equals(from_wkb(geometry))
            ):
                raise ValueError(
                    "one support_id cannot identify different source supports"
                )
        key = tuple(record[name] for name in _IDENTITY) + (record["time"],)
        if key in keys:
            raise ValueError(f"duplicate sample observation: {key!r}")
        keys.add(key)
        _number(record["value"], "sample value")
    if check_counts and metadata.report.complete:
        if metadata.report.counts.produced != table.num_rows:
            raise ValueError("complete evidence produced count disagrees with its rows")
    return ExecutionReport(
        complete=True,
        valid=metadata.report.valid is not False
        and not any(
            diagnostic.severity is DiagnosticSeverity.ERROR
            for diagnostic in metadata.report.diagnostics
        ),
        diagnostics=metadata.report.diagnostics,
        counts=ExecutionCounts(requested=table.num_rows, produced=table.num_rows),
        source_identities=metadata.report.source_identities,
    )


def fit_sample_table(
    table: pa.Table,
    metadata: EvidenceMetadata,
    specification: FitSpec,
    *,
    on_failure: FitFailure = FitFailure.RAISE,
) -> tuple[tuple[DatasetRow, ...], DatasetMetadata]:
    if not metadata.report.complete:
        raise ValueError(
            "an incomplete sample preview cannot be fitted; collect the full selection "
            "with .read() before calling .fit(...)"
        )
    if metadata.report.valid is False:
        raise ValueError("invalid sample evidence cannot be fitted")
    if not isinstance(on_failure, FitFailure):
        raise TypeError("on_failure must be FitFailure")
    if not isinstance(specification, (SampleFit, ModelSelection)):
        raise TypeError("sample evidence requires SampleFit or sample ModelSelection")
    if isinstance(specification, ModelSelection) and any(
        not isinstance(candidate, SampleFit) for candidate in specification.candidates
    ):
        raise TypeError("sample model selection requires SampleFit candidates")
    validate_evidence(table, metadata)
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for record in table.to_pylist():
        groups.setdefault(tuple(record[name] for name in _IDENTITY), []).append(record)
    rows: list[DatasetRow] = []
    diagnostics = list(metadata.report.diagnostics)
    skipped = no_data = 0
    for identity, records in groups.items():
        variable, scenario, period, support, cell = identity
        common: _RowIdentity = {
            "variable_id": cast(str, variable),
            "scenario_id": cast(str | None, scenario),
            "period_id": cast(str | None, period),
            "support_id": cast(str, support),
            "cell_id": cast(int, cell),
            "geometry": cast(bytes | None, records[0]["geometry"]),
        }
        try:
            result = fit_samples(
                [cast(float, record["value"]) for record in records], specification
            )
            rows.append(
                DatasetRow(**common, curve=result.distribution, fit=result.record)
            )
            diagnostics.extend(
                ExecutionDiagnostic(
                    code=d.code,
                    severity=d.severity,
                    stage=Stage.FITTING,
                    message=d.message,
                    source=None if metadata.source is None else metadata.source.product,
                    support_id=cast(str, support),
                )
                for d in result.diagnostics
            )
        except FitError as error:
            if on_failure is FitFailure.RAISE:
                raise _SampleFitError(
                    error,
                    variable_id=cast(str, variable),
                    support_id=cast(str, support),
                    source=None if metadata.source is None else metadata.source.product,
                ) from error
            diagnostics.append(
                FittingDiagnostic(
                    code=error.code,
                    severity=DiagnosticSeverity.WARNING,
                    stage=Stage.FITTING,
                    message=f"{variable}: {error.message}",
                    source=None if metadata.source is None else metadata.source.product,
                    support_id=cast(str, support),
                    remedy="Inspect the retained fitting record and choose an explicit "
                    "SampleFit specification or supply more observations.",
                    record=error.record,
                )
            )
            if on_failure is FitFailure.KEEP_NO_DATA:
                no_data += 1
                rows.append(
                    DatasetRow(**common, failure=RetainedFailure.from_error(error))
                )
            else:
                skipped += 1
    report = ExecutionReport(
        complete=True,
        valid=True,
        diagnostics=tuple(diagnostics),
        counts=ExecutionCounts(
            requested=len(groups),
            produced=len(rows),
            skipped=skipped,
            no_data=no_data,
            usable_years=metadata.report.counts.usable_years,
            incomplete_years=metadata.report.counts.incomplete_years,
        ),
        source_identities=metadata.report.source_identities,
    )
    output = DatasetMetadata(
        spec=metadata.spec,
        spatial=metadata.spatial,
        report=report,
        source=metadata.source,
        provenance=metadata.provenance,
    )
    validate_rows(rows, output)
    return tuple(rows), output
