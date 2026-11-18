"""Owned long-form evidence and explicit sample modelling."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from enum import Enum
from typing import Generic, TypedDict, cast

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
from typing_extensions import TypeVar

from ._aggregation import (
    ReductionKind,
    Sampling,
    TemporalReduction,
    TimestampPosition,
    _annual_sampling,
    _date,
    _position,
    _sampling_slots,
    _year_slots,
    sampling_from_document,
    sampling_schema,
)
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
    UnavailableStationYear,
    _dump,
    _identifier,
    _load,
    _number,
    _object,
    _report,
    _source,
    _spatial,
    _transformation,
    _wire_integer,
)
from ._declarations import Calendar, DataSpec, ProbabilityKind, _data_spec
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
    for component, maximum in (
        ("hour", 23),
        ("minute", 59),
        ("second", 59),
        ("offset_hour", 23),
        ("offset_minute", 59),
    ):
        if match[component] is not None and int(match[component]) > maximum:
            raise ValueError("sample time has an invalid clock or UTC offset")
    return value.replace("t", "T").replace("z", "Z")


class EvidenceKind(Enum):
    SAMPLES = "samples"
    OBSERVATIONS = "observations"
    PROJECTIONS = "projections"
    LAYER = "layer"


KindT = TypeVar("KindT", bound=EvidenceKind, covariant=True, default=EvidenceKind)


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
        support_id: str | None,
        source: str | None,
        requested: int,
        diagnostics: tuple[ExecutionDiagnostic, ...],
    ) -> None:
        super().__init__(error.message, record=error.record)
        self.code = error.code
        self.variable_id = _identifier(variable_id, "variable_id")
        self.support_id = (
            None if support_id is None else _identifier(support_id, "support_id")
        )
        self.source = None if source is None else _identifier(source, "source")
        self.requested = requested
        self.diagnostics = diagnostics


def evidence_metadata_schema() -> dict[str, object]:
    """Generate the persisted header schema from the shared scientific metadata."""
    from ._schema.contract import _record, dataset_json_schema

    dataset = dataset_json_schema()
    definitions = cast(dict[str, dict[str, object]], dataset["$defs"])
    metadata = deepcopy(definitions["metadata"])
    properties = cast(dict[str, object], metadata["properties"])
    properties["schema_version"] = {"const": "1.0"}
    properties["kind"] = {"enum": [kind.value for kind in EvidenceKind]}
    cast(list[str], metadata["required"]).append("kind")
    properties["sampling"] = sampling_schema()
    row = cast(dict[str, object], definitions["row"]["properties"])
    support_fields = ("support_id", "cell_id", "geometry")
    properties["empty_supports"] = {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": _record(
            {name: row[name] for name in support_fields}, list(support_fields)
        ),
    }
    metadata["dependentSchemas"] = {
        "empty_supports": {
            "properties": {
                "kind": {"const": EvidenceKind.OBSERVATIONS.value},
                "report": {
                    "properties": {"counts": {"properties": {"produced": {"const": 0}}}}
                },
            }
        }
    }
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


@dataclass(frozen=True, slots=True)
class ObservationSupport:
    id: str
    cell_id: int
    geometry: bytes
    kind: SupportKind = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.id, "support_id")
        _cell(self.cell_id)
        object.__setattr__(self, "kind", _geometry(self.geometry))

    def _document(self) -> dict[str, object]:
        return {
            "support_id": self.id,
            "cell_id": self.cell_id,
            "geometry": self.geometry.hex(),
        }

    @classmethod
    def _from_document(cls, value: object) -> ObservationSupport:
        record = _object(value, {"support_id", "cell_id", "geometry"})
        geometry = record["geometry"]
        if not isinstance(geometry, str) or not re.fullmatch(
            r"(?:[0-9a-f]{2}){21,}", geometry
        ):
            raise ValueError("support geometry must be lowercase hexadecimal WKB")
        return cls(
            _identifier(record["support_id"], "support_id"),
            _wire_integer(record["cell_id"], "cell_id"),
            bytes.fromhex(geometry),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceMetadata(Generic[KindT]):
    kind: KindT
    spec: DataSpec
    spatial: SpatialContract
    report: ExecutionReport
    source: SourceIdentity | None = None
    provenance: tuple[Transformation, ...] = ()
    sampling: Sampling | None = None
    _empty_supports: tuple[ObservationSupport, ...] = field(default=(), repr=False)

    @property
    def schema_version(self) -> str:
        return "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceKind):
            raise TypeError("kind must be EvidenceKind")
        if self._empty_supports:
            if (
                self.kind is not EvidenceKind.OBSERVATIONS
                or self.report.counts.produced
            ):
                raise ValueError("empty supports require observations with zero rows")
            support_ids: set[str] = set()
            for support in self._empty_supports:
                if support.id in support_ids:
                    raise ValueError(f"duplicate empty support_id: {support.id!r}")
                support_ids.add(support.id)
                if (
                    self.spatial.index_resolution is not None
                    and (support.cell_id >> 52) & 15 != self.spatial.index_resolution
                ):
                    raise ValueError(
                        "empty support cell_id contradicts the spatial resolution"
                    )
                if support.kind not in self.spatial.supports:
                    raise ValueError(
                        "empty support geometry contradicts the spatial contract"
                    )
        if self.sampling is not None:
            if not isinstance(self.sampling, Sampling):
                raise TypeError("sampling must be Sampling")
            for series in self.spec.series:
                if (
                    series.period is not None
                    and series.period.calendar is not None
                    and series.period.calendar is not self.sampling.calendar
                ):
                    raise ValueError(
                        "sampling calendar contradicts declared source calendar"
                    )
        if self.kind is EvidenceKind.PROJECTIONS and any(
            s.probability is not None and s.probability.kind is ProbabilityKind.ANNUAL
            for s in self.spec.series
        ):
            raise ValueError(
                "projection evidence cannot claim annual exceedance probability"
            )
        if (
            self.report.complete
            and self.kind is EvidenceKind.SAMPLES
            and _annual_sampling(self.sampling)
        ):
            counts = self.report.counts
            unavailable = sum(
                isinstance(d, UnavailableStationYear) for d in self.report.diagnostics
            )
            if (
                counts.requested != counts.produced + unavailable
                or counts.usable_years != counts.produced - counts.no_data
                or counts.incomplete_years != counts.no_data + unavailable
            ):
                raise ValueError(
                    "complete annual evidence counts disagree with its report"
                )
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
        if self.sampling is not None:
            value["sampling"] = self.sampling._document()
        if self._empty_supports:
            value["empty_supports"] = [
                support._document() for support in self._empty_supports
            ]
        return _dump(value)

    @classmethod
    def from_json(cls, text: str) -> EvidenceMetadata:
        value = _object(
            _load(text),
            {"schema_version", "kind", "spec", "spatial", "report", "provenance"},
            {"source", "sampling", "empty_supports"},
        )
        if value["schema_version"] != "1.0":
            raise ValueError("expected evidence schema 1.0")
        provenance = value["provenance"]
        if not isinstance(provenance, list):
            raise ValueError("provenance must be an array")
        supports = value.get("empty_supports", [])
        if not isinstance(supports, list) or (
            "empty_supports" in value and not supports
        ):
            raise ValueError("empty_supports must be a nonempty array when supplied")
        return cls(
            kind=cast(KindT, EvidenceKind(value["kind"])),
            spec=_data_spec(value["spec"]),
            spatial=_spatial(value["spatial"]),
            report=_report(value["report"]),
            source=None if "source" not in value else _source(value["source"]),
            provenance=tuple(_transformation(p) for p in provenance),
            sampling=None
            if "sampling" not in value
            else sampling_from_document(value["sampling"]),
            _empty_supports=tuple(
                ObservationSupport._from_document(s) for s in supports
            ),
        )


_FAILURE_TYPE = pa.struct(
    [
        pa.field("code", pa.string(), nullable=False),
        pa.field("message", pa.string(), nullable=False),
    ]
)
_COVERAGE_TYPE = pa.struct(
    [
        pa.field("valid", pa.int64(), nullable=False),
        pa.field("expected", pa.int64(), nullable=False),
        pa.field("minimum_fraction", pa.float64(), nullable=False),
    ]
)
_COORDINATE_TYPE = pa.struct(
    [
        pa.field("kind", pa.string(), nullable=False),
        pa.field("time", pa.string()),
        pa.field("years", pa.float64()),
    ]
)
_INTERVAL_TYPE = pa.struct(
    [
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("lower", pa.float64(), nullable=False),
        pa.field("upper", pa.float64(), nullable=False),
    ]
)


def evidence_arrow_schema(metadata: EvidenceMetadata | None = None) -> pa.Schema:
    kind = EvidenceKind.SAMPLES if metadata is None else metadata.kind
    fields = [
        pa.field("variable_id", pa.string(), nullable=False),
        pa.field("scenario_id", pa.string()),
        pa.field("period_id", pa.string()),
        pa.field("support_id", pa.string(), nullable=False),
        pa.field("cell_id", pa.uint64(), nullable=False),
        pa.field("geometry", pa.binary()),
    ]
    if kind in (EvidenceKind.SAMPLES, EvidenceKind.OBSERVATIONS):
        fields += [
            pa.field("time", pa.string(), nullable=False),
            pa.field("value", pa.float64()),
            pa.field("station_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("failure", _FAILURE_TYPE),
            pa.field("coverage", _COVERAGE_TYPE),
        ]
    elif kind is EvidenceKind.PROJECTIONS:
        fields += [
            pa.field("coordinate", _COORDINATE_TYPE, nullable=False),
            pa.field("probability", pa.float64()),
            pa.field("value", pa.float64()),
            pa.field("interval", _INTERVAL_TYPE),
            pa.field("station_id", pa.string()),
            pa.field("failure", _FAILURE_TYPE),
        ]
    else:
        fields += [
            pa.field("value", pa.float64()),
            pa.field("label", pa.string()),
            pa.field("failure", _FAILURE_TYPE),
        ]
    schema = pa.schema(fields)
    return (
        schema
        if metadata is None
        else schema.with_metadata({_METADATA_KEY: metadata.to_json().encode()})
    )


def metadata_from_evidence_table(table: pa.Table) -> EvidenceMetadata:
    value = (table.schema.metadata or {}).get(_METADATA_KEY)
    if value is None:
        raise ValueError("canonical evidence metadata is missing")
    return EvidenceMetadata.from_json(value.decode("utf-8"))


def evidence_table(
    records: Sequence[dict[str, object]], metadata: EvidenceMetadata
) -> pa.Table:
    schema = evidence_arrow_schema(metadata)
    fields = set(schema.names)
    required = {field.name for field in schema if not field.nullable}
    if any(
        not required <= record.keys() or record.keys() - fields for record in records
    ):
        raise ValueError(
            "evidence records must contain exactly the applicable canonical fields"
        )
    table = pa.Table.from_pylist(list(records), schema=schema)
    validate_evidence(table, metadata)
    return table


def _observation_year(time: str, sampling: Sampling | None) -> int:
    if sampling is not None and sampling.timestamp is TimestampPosition.END:
        ordinal, _ = _position(time, sampling)
        return _date(ordinal, sampling.calendar)[0]
    return int(time[:4])


def evidence_row_key(
    record: Mapping[str, object], metadata: EvidenceMetadata
) -> tuple[object, ...]:
    kind = metadata.kind
    common = tuple(record[name] for name in _IDENTITY)
    if kind in (EvidenceKind.SAMPLES, EvidenceKind.OBSERVATIONS):
        if _annual_sampling(metadata.sampling):
            return (
                *common,
                _observation_year(cast(str, record["time"]), metadata.sampling),
            )
        return (*common, record["time"])
    if kind is EvidenceKind.PROJECTIONS:
        coordinate = cast(Mapping[str, object], record["coordinate"])
        return (
            *common,
            coordinate["kind"],
            coordinate.get("time"),
            coordinate.get("years"),
            record["probability"],
        )
    return common


def validate_evidence(
    table: pa.Table,
    metadata: EvidenceMetadata | None = None,
    *,
    check_counts: bool = True,
) -> ExecutionReport:
    if not isinstance(table, pa.Table):
        raise TypeError("evidence must be an Arrow Table")
    stored = metadata_from_evidence_table(table)
    if metadata is not None and stored.to_json() != metadata.to_json():
        raise ValueError("supplied and stored evidence metadata disagree")
    metadata = stored
    if not table.schema.equals(evidence_arrow_schema(metadata), check_metadata=False):
        raise ValueError(
            "Arrow fields do not match the declared canonical evidence kind"
        )
    if table.num_rows and metadata._empty_supports:
        raise ValueError("empty supports require observations with zero rows")
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
    missing = 0
    reductions = tuple(
        p for p in metadata.provenance if isinstance(p, TemporalReduction)
    )
    reduction = reductions[-1] if reductions else None
    annual = reduction is not None and _annual_sampling(reduction.output_sampling)
    rolling_loss = 0
    if (
        annual
        and len(reductions) > 1
        and reductions[-2].operation is ReductionKind.ROLLING_TOTAL
    ):
        window = reductions[-2].window
        assert window is not None
        rolling_loss = int(window.value) - 1
    slots = None
    if reduction is not None:
        if metadata.sampling != reduction.output_sampling:
            raise ValueError("sampling contradicts the last temporal reduction")
        if reduction.operation is ReductionKind.ROLLING_TOTAL:
            assert reduction.window is not None
            slots = int(reduction.window.value)
        else:
            slots = _sampling_slots(reduction.input_sampling, daily=not annual)
    for record in table.to_pylist():
        for name in ("variable_id", "support_id"):
            _identifier(record[name], name)
        for name in ("scenario_id", "period_id", "station_id", "group_id"):
            if record.get(name) is not None:
                _identifier(record[name], name)
        identity = tuple(record[name] for name in _IDENTITY[:3])
        if identity not in series:
            raise ValueError(f"evidence refers to an undeclared series: {identity!r}")
        declared_period = series[identity].period
        calendar = None if metadata.sampling is None else metadata.sampling.calendar
        if declared_period is not None and declared_period.calendar is not None:
            calendar = declared_period.calendar
        if metadata.kind in (EvidenceKind.SAMPLES, EvidenceKind.OBSERVATIONS):
            if _sample_time(record["time"], calendar) != record["time"]:
                raise ValueError("evidence time must use canonical uppercase T and Z")
            if (
                declared_period is not None
                and declared_period.start is not None
                and declared_period.end is not None
            ):
                year = _observation_year(record["time"], metadata.sampling)
                if not declared_period.start <= year <= declared_period.end:
                    raise ValueError(
                        f"observation year {year} lies outside its declared period "
                        f"[{declared_period.start}, {declared_period.end}]"
                    )
        elif metadata.kind is EvidenceKind.PROJECTIONS:
            coordinate = record["coordinate"]
            if coordinate is None or coordinate["kind"] not in (
                "time",
                "return_period",
            ):
                raise ValueError(
                    "projection requires a typed time or return-period coordinate"
                )
            if coordinate["kind"] == "time":
                if coordinate["years"] is not None or coordinate["time"] is None:
                    raise ValueError(
                        "time projection coordinate has inactive return-period fields"
                    )
                _sample_time(coordinate["time"], calendar)
            elif (
                coordinate["time"] is not None
                or _number(coordinate["years"], "return period") <= 0
            ):
                raise ValueError(
                    "return-period projection coordinate requires positive years only"
                )
            probability = record["probability"]
            if (
                probability is not None
                and not 0 <= _number(probability, "probability") <= 1
            ):
                raise ValueError("projection probability must lie in [0, 1]")
            interval = record["interval"]
            if interval is not None:
                if not 0 < _number(interval["confidence"], "confidence") < 1:
                    raise ValueError("interval confidence must lie in (0, 1)")
                if _number(interval["lower"], "lower") > _number(
                    interval["upper"], "upper"
                ):
                    raise ValueError("interval bounds must be ordered")
        cell = _cell(record["cell_id"])
        if (
            metadata.spatial.index_resolution is not None
            and (cell >> 52) & 15 != metadata.spatial.index_resolution
        ):
            raise ValueError("cell_id resolution contradicts the spatial contract")
        geometry = record["geometry"]
        support_kind = SupportKind.CELL if geometry is None else _geometry(geometry)
        if support_kind not in metadata.spatial.supports:
            raise ValueError("geometry contradicts the declared spatial support")
        support = (geometry, cell if geometry is None else None)
        previous = supports.setdefault(record["support_id"], support)
        if previous != support:
            if (
                previous[0] is None
                or geometry is None
                or previous[1] != support[1]
                or not from_wkb(previous[0]).equals(from_wkb(geometry))
            ):
                raise ValueError(
                    "one support_id cannot identify different source supports"
                )
        key = evidence_row_key(record, metadata)
        if key in keys:
            raise ValueError(f"duplicate evidence record: {key!r}")
        keys.add(key)
        value, failure = record["value"], record["failure"]
        if value is not None:
            _number(value, "evidence value")
        if failure is not None:
            if (
                value is not None
                or metadata.kind is EvidenceKind.PROJECTIONS
                and record["interval"] is not None
            ):
                raise ValueError("failed evidence cannot contain a value or interval")
            _identifier(failure["code"], "failure.code")
            if not isinstance(failure["message"], str) or not failure["message"]:
                raise ValueError("failure.message must be nonempty text")
            missing += 1
        elif value is None and metadata.kind not in (
            EvidenceKind.LAYER,
            EvidenceKind.PROJECTIONS,
        ):
            raise ValueError("missing numeric evidence requires an explicit failure")
        elif (
            value is None
            and metadata.kind is EvidenceKind.PROJECTIONS
            and record["interval"] is None
        ):
            raise ValueError(
                "projection requires a value, interval, or retained failure"
            )
        quality = record.get("coverage")
        if reduction is not None and quality is None:
            raise ValueError("reduced temporal evidence requires row coverage")
        if quality is not None:
            if (
                type(quality["valid"]) is not int
                or type(quality["expected"]) is not int
                or not 0 <= quality["valid"] <= quality["expected"]
                or quality["expected"] == 0
            ):
                raise ValueError(
                    "coverage counts must satisfy 0 <= valid <= expected "
                    "and expected > 0"
                )
            fraction = _number(quality["minimum_fraction"], "minimum_fraction")
            if not 0 < fraction <= 1:
                raise ValueError("minimum coverage fraction must lie in (0, 1]")
            if reduction is not None:
                assert slots is not None
                stamp = record["time"]
                if len(stamp) != (4 if annual else 10):
                    raise ValueError(
                        "reduced evidence must label its calendar year or day"
                    )
                year = int(stamp[:4])
                expected = (
                    _year_slots(reduction.input_sampling, slots, year) - rolling_loss
                    if annual
                    else slots
                )
                if (
                    fraction != reduction.coverage.fraction
                    or quality["expected"] != expected
                ):
                    raise ValueError(
                        "row coverage contradicts the declared temporal reduction"
                    )
            if value is not None and (
                quality["valid"] != quality["expected"]
                if fraction == 1.0
                else quality["valid"] / quality["expected"] < fraction
            ):
                raise ValueError(
                    "numeric evidence does not meet its recorded completeness rule"
                )
    if check_counts and metadata.report.complete:
        if (
            metadata.report.counts.produced != table.num_rows
            or metadata.report.counts.no_data != missing
        ):
            raise ValueError("complete evidence counts disagree with its rows")
    return ExecutionReport(
        True,
        metadata.report.valid is not False
        and not any(
            d.severity is DiagnosticSeverity.ERROR for d in metadata.report.diagnostics
        ),
        metadata.report.diagnostics,
        replace(metadata.report.counts, produced=table.num_rows, no_data=missing),
        metadata.report.source_identities,
    )


def fit_sample_table(
    table: pa.Table,
    metadata: EvidenceMetadata,
    specification: FitSpec,
    *,
    on_failure: FitFailure = FitFailure.RAISE,
) -> tuple[tuple[DatasetRow, ...], DatasetMetadata]:
    if metadata.kind is not EvidenceKind.SAMPLES:
        raise TypeError(
            "fitting requires sample evidence; aggregate raw observations first"
        )
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
    observed_stations = set()
    for record in table.to_pylist():
        groups.setdefault(tuple(record[name] for name in _IDENTITY), []).append(record)
        if record["station_id"] is not None:
            observed_stations.add(record["station_id"])
    unlocated = dict.fromkeys(
        diagnostic.station_id
        for diagnostic in metadata.report.diagnostics
        if metadata.spec.series
        and isinstance(diagnostic, UnavailableStationYear)
        and diagnostic.station_id not in observed_stations
    )
    requested = len(groups) + len(unlocated)
    source = None if metadata.source is None else metadata.source.product
    rows: list[DatasetRow] = []
    diagnostics = list(metadata.report.diagnostics)
    skipped = no_data = 0

    def record_failure(
        original: FitError,
        *,
        variable_id: str,
        support_id: str | None,
        station_id: str | None = None,
    ) -> FitError:
        message = original.message
        if station_id is not None:
            message = f"Station {station_id!r} has no usable annual evidence. {message}"
        error = FitError(message, record=original.record)
        error.code = original.code
        if on_failure is FitFailure.RAISE:
            raise _SampleFitError(
                error,
                variable_id=variable_id,
                support_id=support_id,
                source=source,
                requested=requested,
                diagnostics=tuple(diagnostics),
            ) from original
        diagnostics.append(
            FittingDiagnostic(
                code=error.code,
                severity=DiagnosticSeverity.WARNING,
                stage=Stage.FITTING,
                message=f"{variable_id}: {error.message}",
                source=source,
                support_id=support_id,
                remedy="Inspect the retained fitting record and choose an explicit "
                "SampleFit specification or supply more observations.",
                record=error.record,
            )
        )
        return error

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
        values = [
            cast(float, record["value"])
            for record in records
            if record["value"] is not None and record["failure"] is None
        ]
        unavailable = len(records) - len(values)
        if unavailable and not _annual_sampling(metadata.sampling):
            diagnostics.append(
                ExecutionDiagnostic(
                    code="fit.unavailable_observations",
                    severity=DiagnosticSeverity.WARNING,
                    stage=Stage.FITTING,
                    message=f"{unavailable} of {len(records)} observations for "
                    f"variable {variable!r} were unavailable for fitting",
                    source=source,
                    support_id=cast(str, support),
                )
            )
        try:
            result = fit_samples(values, specification)
            rows.append(
                DatasetRow(**common, curve=result.distribution, fit=result.record)
            )
            diagnostics.append(
                ExecutionDiagnostic(
                    code="fit.samples",
                    severity=DiagnosticSeverity.INFO,
                    stage=Stage.FITTING,
                    message="Requested sample estimator converged",
                    source=source,
                    support_id=cast(str, support),
                )
            )
        except FitError as original_error:
            error = record_failure(
                original_error,
                variable_id=cast(str, variable),
                support_id=cast(str, support),
            )
            if on_failure is FitFailure.KEEP_NO_DATA:
                no_data += 1
                rows.append(
                    DatasetRow(**common, failure=RetainedFailure.from_error(error))
                )
            else:
                skipped += 1
    if unlocated:
        # The selected station has no observed support anywhere in this
        # window. Retain the failed fit in the report without inventing a row.
        # These observations belong to the provider's single annual series.
        variable_id = metadata.spec.series[0].variable.id
        try:
            fit_samples([], specification)
        except FitError as error:
            for station_id in unlocated:
                record_failure(
                    error,
                    variable_id=variable_id,
                    support_id=None,
                    station_id=station_id,
                )
                skipped += int(on_failure is FitFailure.SKIP)
    report = ExecutionReport(
        complete=True,
        valid=True,
        diagnostics=tuple(diagnostics),
        counts=ExecutionCounts(
            requested=requested,
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
