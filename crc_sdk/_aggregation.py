"""Explicit temporal sampling, completeness policies, and reductions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias, cast

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]

from ._declarations import (
    Calendar,
    DataSpec,
    Probability,
    SeriesSpec,
    Variable,
    _fields,
)

if TYPE_CHECKING:
    from ._evidence import EvidenceMetadata


class TimestampPosition(Enum):
    START = "start"
    END = "end"


@dataclass(frozen=True, slots=True, kw_only=True)
class Sampling:
    calendar: Calendar
    step: cf.Duration
    timestamp: TimestampPosition = TimestampPosition.START

    def __post_init__(self) -> None:
        if not isinstance(self.calendar, Calendar):
            raise TypeError("calendar must be Calendar")
        if not isinstance(self.step, cf.Duration):
            raise TypeError("step must be framework Duration")
        if not isinstance(self.timestamp, TimestampPosition):
            raise TypeError("timestamp must be TimestampPosition")

    def _document(self) -> dict[str, object]:
        return {
            "calendar": self.calendar.value,
            "step": {"value": self.step.value, "unit": self.step.unit.value},
            "timestamp": self.timestamp.value,
        }


def sampling_from_document(value: object) -> Sampling:
    if not isinstance(value, dict) or set(value) != {"calendar", "step", "timestamp"}:
        raise ValueError("sampling requires calendar, step, and timestamp")
    step = value["step"]
    if not isinstance(step, dict) or set(step) != {"value", "unit"}:
        raise ValueError("sampling step requires value and unit")
    return Sampling(
        calendar=Calendar(value["calendar"]),
        step=cf.Duration(step["value"], unit=cf.TimeUnit(step["unit"])),
        timestamp=TimestampPosition(value["timestamp"]),
    )


def sampling_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["calendar", "step", "timestamp"],
        "properties": {
            "calendar": {"enum": [v.value for v in Calendar]},
            "timestamp": {"enum": [v.value for v in TimestampPosition]},
            "step": {
                "type": "object",
                "additionalProperties": False,
                "required": ["value", "unit"],
                "properties": {
                    "value": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1.7976931348623157e308,
                    },
                    "unit": {"enum": [v.value for v in cf.TimeUnit]},
                },
            },
        },
    }


@dataclass(frozen=True, slots=True, init=False)
class Coverage:
    fraction: float

    def __init__(self) -> None:
        raise TypeError("use Coverage.complete() or Coverage.minimum_fraction(...)")

    @classmethod
    def complete(cls) -> Coverage:
        return cls.minimum_fraction(1.0)

    @classmethod
    def minimum_fraction(cls, fraction: float) -> Coverage:
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (float, int))
            or not math.isfinite(fraction)
            or not 0 < fraction <= 1
        ):
            raise ValueError("coverage fraction must be finite and in (0, 1]")
        result = object.__new__(cls)
        object.__setattr__(result, "fraction", float(fraction))
        return result


@dataclass(frozen=True, slots=True, init=False)
class DailyTotal:
    coverage: Coverage

    def __init__(self, *, coverage: Coverage | None = None) -> None:
        if coverage is not None and not isinstance(coverage, Coverage):
            raise TypeError("coverage must be Coverage")
        object.__setattr__(
            self, "coverage", Coverage.complete() if coverage is None else coverage
        )


@dataclass(frozen=True, slots=True, init=False)
class AnnualMaximum:
    coverage: Coverage

    def __init__(self, *, coverage: Coverage | None = None) -> None:
        if coverage is not None and not isinstance(coverage, Coverage):
            raise TypeError("coverage must be Coverage")
        object.__setattr__(
            self, "coverage", Coverage.complete() if coverage is None else coverage
        )


@dataclass(frozen=True, slots=True, init=False)
class AnnualMinimum:
    coverage: Coverage

    def __init__(self, *, coverage: Coverage | None = None) -> None:
        if coverage is not None and not isinstance(coverage, Coverage):
            raise TypeError("coverage must be Coverage")
        object.__setattr__(
            self, "coverage", Coverage.complete() if coverage is None else coverage
        )


Aggregation: TypeAlias = DailyTotal | AnnualMaximum | AnnualMinimum


class ReductionKind(Enum):
    DAILY_TOTAL = "daily_total"
    DAILY_MAXIMUM = "daily_maximum"
    DAILY_MINIMUM = "daily_minimum"
    ROLLING_TOTAL = "rolling_total"
    ANNUAL_MAXIMUM = "annual_maximum"
    ANNUAL_MINIMUM = "annual_minimum"


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementChange:
    variable_id: str
    input: cf.Measurement
    output: cf.Measurement

    def __post_init__(self) -> None:
        if (
            not isinstance(self.variable_id, str)
            or not self.variable_id
            or self.variable_id.strip() != self.variable_id
        ):
            raise ValueError("variable_id must be a nonempty exact identifier")
        if not isinstance(self.input, cf.Measurement) or not isinstance(
            self.output, cf.Measurement
        ):
            raise TypeError("measurement changes require framework Measurement records")
        if self.input == self.output:
            raise ValueError("measurement change must change scientific meaning")

    def _document(self) -> dict[str, object]:
        return {
            "variable_id": self.variable_id,
            "input": json.loads(self.input.to_json()),
            "output": json.loads(self.output.to_json()),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalReduction:
    operation: ReductionKind
    coverage: Coverage
    input_sampling: Sampling
    output_sampling: Sampling
    measurements: tuple[MeasurementChange, ...] = ()
    window: cf.Duration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, ReductionKind) or not isinstance(
            self.coverage, Coverage
        ):
            raise TypeError("temporal reduction requires ReductionKind and Coverage")
        if not isinstance(self.input_sampling, Sampling) or not isinstance(
            self.output_sampling, Sampling
        ):
            raise TypeError(
                "temporal reduction requires explicit input/output Sampling"
            )
        if self.input_sampling.calendar is not self.output_sampling.calendar:
            raise ValueError("temporal reduction cannot change the source calendar")
        if not isinstance(self.measurements, tuple) or not all(
            isinstance(m, MeasurementChange) for m in self.measurements
        ):
            raise TypeError(
                "measurements must be an immutable tuple of MeasurementChange"
            )
        if len({m.variable_id for m in self.measurements}) != len(self.measurements):
            raise ValueError("measurement changes must refer to distinct variables")
        expected = (
            cf.TimeUnit.YEAR
            if self.operation
            in (ReductionKind.ANNUAL_MAXIMUM, ReductionKind.ANNUAL_MINIMUM)
            else cf.TimeUnit.DAY
        )
        if (self.operation is ReductionKind.ROLLING_TOTAL) != (self.window is not None):
            raise ValueError("only rolling totals require an explicit window")
        if self.window is not None and (
            not isinstance(self.window, cf.Duration)
            or self.window.unit is not cf.TimeUnit.DAY
            or self.window.value < 1
            or not self.window.value.is_integer()
        ):
            raise ValueError("rolling totals require a positive whole-day window")
        if self.operation is ReductionKind.ROLLING_TOTAL:
            if self.input_sampling.step != cf.Duration(1, unit=cf.TimeUnit.DAY):
                raise ValueError("rolling totals require daily input sampling")
        else:
            _sampling_slots(self.input_sampling, daily=expected is cf.TimeUnit.DAY)
        if (
            self.output_sampling.step != cf.Duration(1, unit=expected)
            or self.output_sampling.timestamp is not TimestampPosition.START
        ):
            raise ValueError(
                "temporal reduction output sampling contradicts its operation"
            )

    def _document(self) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": "temporal_reduction",
            "operation": self.operation.value,
            "coverage": self.coverage.fraction,
            "input_sampling": self.input_sampling._document(),
            "output_sampling": self.output_sampling._document(),
        }
        if self.measurements:
            value["measurements"] = [m._document() for m in self.measurements]
        if self.window is not None:
            value["window"] = {
                "value": self.window.value,
                "unit": self.window.unit.value,
            }
        return value


def temporal_reduction_from_document(value: object) -> TemporalReduction:
    fields = {"kind", "operation", "coverage", "input_sampling", "output_sampling"}
    if (
        not isinstance(value, dict)
        or not fields <= set(value)
        or set(value) - fields - {"measurements", "window"}
        or value["kind"] != "temporal_reduction"
    ):
        raise ValueError("invalid temporal reduction record")
    measurements = value.get("measurements", [])
    if not isinstance(measurements, list):
        raise TypeError("temporal reduction measurements must be an array")
    changes = []
    for measurement in measurements:
        record = _fields(measurement, {"variable_id", "input", "output"})
        changes.append(
            MeasurementChange(
                variable_id=cast(str, record["variable_id"]),
                input=cf.Measurement.from_json(json.dumps(record["input"])),
                output=cf.Measurement.from_json(json.dumps(record["output"])),
            )
        )
    window = None
    if "window" in value:
        record = _fields(value["window"], {"value", "unit"})
        window = cf.Duration(
            cast(float, record["value"]), unit=cf.TimeUnit(record["unit"])
        )
    return TemporalReduction(
        operation=ReductionKind(value["operation"]),
        coverage=Coverage.minimum_fraction(value["coverage"]),
        input_sampling=sampling_from_document(value["input_sampling"]),
        output_sampling=sampling_from_document(value["output_sampling"]),
        measurements=tuple(changes),
        window=window,
    )


def temporal_reduction_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "kind",
            "operation",
            "coverage",
            "input_sampling",
            "output_sampling",
        ],
        "properties": {
            "kind": {"const": "temporal_reduction"},
            "operation": {"enum": [v.value for v in ReductionKind]},
            "coverage": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "input_sampling": sampling_schema(),
            "output_sampling": sampling_schema(),
            "window": {
                "type": "object",
                "additionalProperties": False,
                "required": ["value", "unit"],
                "properties": {
                    "value": {"type": "integer", "minimum": 1},
                    "unit": {"const": "d"},
                },
            },
            "measurements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["variable_id", "input", "output"],
                    "properties": {
                        "variable_id": {
                            "type": "string",
                            "minLength": 1,
                            "pattern": r"^\S(?:[\s\S]*\S)?$",
                        },
                        "input": {
                            "$ref": "https://climateriskcommons.org/schemas/measurement-1.0.schema.json"
                        },
                        "output": {
                            "$ref": "https://climateriskcommons.org/schemas/measurement-1.0.schema.json"
                        },
                    },
                },
            },
        },
    }


def days_in_month(year: int, month: int, calendar: Calendar) -> int:
    if calendar is Calendar.DAYS_360:
        return 30
    leap = (
        calendar is Calendar.ALL_LEAP
        or calendar is Calendar.JULIAN
        and year % 4 == 0
        or calendar in (Calendar.GREGORIAN, Calendar.PROLEPTIC_GREGORIAN)
        and year % 4 == 0
        and (year % 100 != 0 or year % 400 == 0)
    )
    return (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]


def _ordinal(year: int, month: int, day: int, calendar: Calendar) -> int:
    previous = year - 1
    if calendar is Calendar.DAYS_360:
        years = previous * 360
    elif calendar is Calendar.NO_LEAP:
        years = previous * 365
    elif calendar is Calendar.ALL_LEAP:
        years = previous * 366
    elif calendar is Calendar.JULIAN:
        years = previous * 365 + previous // 4
    else:
        years = previous * 365 + previous // 4 - previous // 100 + previous // 400
    return (
        years + sum(days_in_month(year, m, calendar) for m in range(1, month)) + day - 1
    )


def _date(ordinal: int, calendar: Calendar) -> tuple[int, int, int]:
    year = max(1, min(9999, ordinal // 365 + 1))
    while _ordinal(year, 1, 1, calendar) > ordinal:
        year -= 1
    while year < 9999 and _ordinal(year + 1, 1, 1, calendar) <= ordinal:
        year += 1
    day = ordinal - _ordinal(year, 1, 1, calendar)
    month = 1
    while day >= days_in_month(year, month, calendar):
        day -= days_in_month(year, month, calendar)
        month += 1
    return year, month, day + 1


def step_seconds(step: cf.Duration) -> float:
    factors = {
        cf.TimeUnit.SECOND: 1.0,
        cf.TimeUnit.HOUR: 3600.0,
        cf.TimeUnit.DAY: 86400.0,
    }
    if step.unit not in factors:
        raise ValueError(
            "this operation requires a fixed sampling step in seconds, hours, or days"
        )
    return step.value * factors[step.unit]


def _position(text: str, sampling: Sampling) -> tuple[int, float]:
    match = re.fullmatch(
        r"(\d{4})(?:-(\d{2})-(\d{2})(?:[Tt](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(?:[Zz]|[+-]00:00)?)?)?",
        text,
    )
    if match is None:
        raise ValueError(
            "temporal aggregation requires native calendar dates "
            "with no nonzero UTC offset"
        )
    year = int(match[1])
    if sampling.calendar is Calendar.GREGORIAN and year < 1583:
        raise ValueError(
            "mixed Gregorian calendar reduction before 1583 is unsupported; "
            "a proleptic or Julian calendar may be declared "
            "only when it is the source calendar"
        )
    month = int(match[2] or 1)
    day = int(match[3] or 1)
    seconds: float = (
        int(match[4] or 0) * 3600 + int(match[5] or 0) * 60 + int(match[6] or 0)
    )
    seconds += float("0." + match[7]) if match[7] else 0
    ordinal = _ordinal(year, month, day, sampling.calendar)
    if sampling.timestamp is TimestampPosition.END:
        if sampling.step.unit in (cf.TimeUnit.MONTH, cf.TimeUnit.YEAR):
            months = sampling.step.value * (
                12 if sampling.step.unit is cf.TimeUnit.YEAR else 1
            )
            if not months.is_integer() or day != 1 or seconds != 0:
                raise ValueError(
                    "calendar interval-end timestamps require a month boundary"
                )
            absolute = (year - 1) * 12 + month - 1 - int(months)
            if absolute < 0:
                raise ValueError("sampling interval precedes the supported calendar")
            year, month = absolute // 12 + 1, absolute % 12 + 1
            ordinal = _ordinal(year, month, 1, sampling.calendar)
        else:
            seconds -= step_seconds(sampling.step)
            while seconds < 0:
                seconds += 86400
                ordinal -= 1
    return ordinal, seconds


def _sampling_slots(sampling: Sampling, *, daily: bool) -> int:
    step = sampling.step
    if step.unit is cf.TimeUnit.MONTH:
        if daily or not step.value.is_integer() or 12 % int(step.value) != 0:
            raise ValueError(
                "annual monthly sampling must divide twelve calendar months"
            )
        return 12 // int(step.value)
    if step.unit is cf.TimeUnit.YEAR:
        if daily or step.value != 1:
            raise ValueError("annual sampling requires one calendar-year observations")
        return 1
    per_day = 86400 / step_seconds(step)
    if (
        not math.isfinite(per_day)
        or per_day < 1
        or not math.isclose(per_day, round(per_day), rel_tol=1e-12)
    ):
        raise ValueError("sampling step must divide a native calendar day")
    return round(per_day)


def _annual_sampling(sampling: Sampling | None) -> bool:
    return (
        sampling is not None
        and sampling.step.unit is cf.TimeUnit.YEAR
        and sampling.step.value == 1
    )


def _year_slots(sampling: Sampling, slots: int, year: int) -> int:
    if sampling.step.unit in (cf.TimeUnit.MONTH, cf.TimeUnit.YEAR):
        return slots
    days = sum(days_in_month(year, month, sampling.calendar) for month in range(1, 13))
    return days * slots


def _measurement_daily(
    variable: Variable, sampling: Sampling
) -> tuple[Variable, float]:
    measurement = variable.measurement
    step = step_seconds(sampling.step)
    if not isinstance(measurement, (cf.PrecipitationRate, cf.PrecipitationAmount)):
        raise ValueError(
            "DailyTotal requires a typed PrecipitationRate "
            "or PrecipitationAmount measurement"
        )
    if not math.isclose(
        step_seconds(measurement.accumulation_duration), step, rel_tol=1e-12
    ):
        raise ValueError(
            "DailyTotal requires independent accumulations matching Sampling.step; "
            "deaccumulate cumulative source counters first"
        )
    if isinstance(measurement, cf.PrecipitationRate):
        divisor = (
            3600
            if measurement.unit is cf.PrecipitationRateUnit.MILLIMETRES_PER_HOUR
            else 86400
        )
        scale = step / divisor
    else:
        scale = {
            cf.LengthUnit.METRE: 1000.0,
            cf.LengthUnit.CENTIMETRE: 10.0,
            cf.LengthUnit.MILLIMETRE: 1.0,
            cf.LengthUnit.FOOT: 304.8,
        }[measurement.unit]
    return Variable(
        variable.id,
        measurement=cf.PrecipitationAmount(
            unit=cf.LengthUnit.MILLIMETRE,
            accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.DAY),
        ),
        label=variable.label,
    ), scale


def aggregation_metadata(
    metadata: EvidenceMetadata, specification: Aggregation
) -> EvidenceMetadata:
    from ._dataset_schema import ExecutionCounts, ExecutionReport
    from ._evidence import EvidenceKind, EvidenceMetadata

    if metadata.kind is not EvidenceKind.OBSERVATIONS:
        raise ValueError("temporal aggregation requires observation evidence")
    if not isinstance(specification, (DailyTotal, AnnualMaximum, AnnualMinimum)):
        raise TypeError("spec must be DailyTotal, AnnualMaximum, or AnnualMinimum")
    if metadata.sampling is None:
        raise ValueError(
            "temporal aggregation requires Sampling(calendar=..., step=...); "
            "supply sampling= when constructing observations"
        )
    sampling = metadata.sampling
    daily = isinstance(specification, DailyTotal)
    _sampling_slots(sampling, daily=daily)
    variables: list[Variable] = []
    changes: list[MeasurementChange] = []
    for variable in metadata.spec.variables:
        converted = _measurement_daily(variable, sampling)[0] if daily else variable
        variables.append(converted)
        if converted.measurement != variable.measurement:
            assert (
                converted.measurement is not None and variable.measurement is not None
            )
            changes.append(
                MeasurementChange(
                    variable_id=variable.id,
                    input=variable.measurement,
                    output=converted.measurement,
                )
            )
    by_id = {v.id: v for v in variables}
    output_sampling = Sampling(
        calendar=sampling.calendar,
        step=cf.Duration(1, unit=cf.TimeUnit.DAY if daily else cf.TimeUnit.YEAR),
    )
    operation = (
        ReductionKind.DAILY_TOTAL
        if daily
        else ReductionKind.ANNUAL_MAXIMUM
        if isinstance(specification, AnnualMaximum)
        else ReductionKind.ANNUAL_MINIMUM
    )
    return EvidenceMetadata(
        kind=EvidenceKind.OBSERVATIONS if daily else EvidenceKind.SAMPLES,
        spec=DataSpec(
            variables,
            series=[
                SeriesSpec(
                    by_id[s.variable.id],
                    scenario=s.scenario,
                    period=s.period,
                    probability=s.probability
                    if daily
                    else Probability.annual(
                        tail=cf.Tail.UPPER
                        if isinstance(specification, AnnualMaximum)
                        else cf.Tail.LOWER
                    ),
                )
                for s in metadata.spec.series
            ],
        ),
        spatial=metadata.spatial,
        report=ExecutionReport(
            False,
            False if metadata.report.valid is False else None,
            metadata.report.diagnostics,
            ExecutionCounts(),
            metadata.report.source_identities,
        ),
        source=metadata.source,
        sampling=output_sampling,
        _empty_supports=metadata._empty_supports if daily else (),
        provenance=(
            *metadata.provenance,
            TemporalReduction(
                operation=operation,
                coverage=specification.coverage,
                input_sampling=sampling,
                output_sampling=output_sampling,
                measurements=tuple(changes),
            ),
        ),
    )


def aggregate_table(
    table: pa.Table,
    metadata: EvidenceMetadata,
    specification: Aggregation,
) -> tuple[pa.Table, EvidenceMetadata]:
    from ._dataset_schema import (
        ExecutionCounts,
        ExecutionDiagnostic,
        ExecutionReport,
        ObservationTimeSelection,
        Stage,
    )
    from ._evidence import evidence_table, validate_evidence

    planned = aggregation_metadata(metadata, specification)
    if not metadata.report.complete:
        raise ValueError(
            "an incomplete observation preview cannot be aggregated; "
            "read the full selection first"
        )
    if metadata.report.valid is False:
        raise ValueError("invalid observation evidence cannot be aggregated")
    validate_evidence(table, metadata)
    assert metadata.sampling is not None
    sampling = metadata.sampling
    daily = isinstance(specification, DailyTotal)
    slots = _sampling_slots(sampling, daily=daily)
    calendar_step = sampling.step.unit in (cf.TimeUnit.MONTH, cf.TimeUnit.YEAR)
    interval = 1.0 if calendar_step else step_seconds(sampling.step)
    scales = {
        v.id: _measurement_daily(v, sampling)[1] if daily else 1.0
        for v in metadata.spec.variables
    }
    groups: dict[tuple[object, ...], dict[int, dict[int, dict[str, object]]]] = {}
    templates: dict[tuple[object, ...], dict[str, object]] = {}
    for record in table.to_pylist():
        identity = tuple(
            record[n]
            for n in (
                "variable_id",
                "scenario_id",
                "period_id",
                "support_id",
                "cell_id",
            )
        )
        ordinal, second = _position(record["time"], sampling)
        year, month, day = _date(ordinal, sampling.calendar)
        bucket = ordinal if daily else year
        if calendar_step:
            months = 12 // slots
            slot = (month - 1) // months
            if (month - 1) % months or day != 1 or second != 0:
                raise ValueError(
                    "Calendar sampling intervals must start at their native "
                    "month or year boundary"
                )
        else:
            position = second / interval
            slot = round(position)
            if slot >= slots or not math.isclose(
                position, slot, rel_tol=0, abs_tol=1e-9
            ):
                boundary = "midnight" if daily else "the native year boundary"
                raise ValueError(
                    "Sampling intervals must align with the native calendar grid; "
                    f"an interval crossing {boundary} cannot be allocated"
                )
            if not daily:
                slot += (ordinal - _ordinal(year, 1, 1, sampling.calendar)) * slots
        occupied = groups.setdefault(identity, {}).setdefault(bucket, {})
        if slot in occupied:
            raise ValueError(
                "more than one observation occupies the declared sampling interval"
            )
        occupied[slot] = record
        templates.setdefault(identity, record)
    records: list[dict[str, object]] = []
    incomplete = usable = 0
    requested_years = tuple(
        step.time
        for step in metadata.provenance
        if isinstance(step, ObservationTimeSelection)
    )
    requested_bounds = None
    if not daily and requested_years:
        first = max(cast(int, period.start) for period in requested_years)
        last = min(cast(int, period.end) for period in requested_years)
        earliest = 1583 if sampling.calendar is Calendar.GREGORIAN else 1
        if first < earliest or last > 9999:
            raise ValueError(
                f"Requested reduction years for {sampling.calendar.value} "
                f"must lie in [{earliest}, 9999]"
            )
        requested_bounds = first, last
    if not groups and requested_bounds is not None:
        for series in metadata.spec.series:
            for support in metadata._empty_supports:
                identity = (
                    series.variable.id,
                    None if series.scenario is None else series.scenario.id,
                    None if series.period is None else series.period.id,
                    support.id,
                    support.cell_id,
                )
                groups[identity] = {}
                templates[identity] = dict(
                    zip(
                        (
                            "variable_id",
                            "scenario_id",
                            "period_id",
                            "support_id",
                            "cell_id",
                        ),
                        identity,
                        strict=True,
                    ),
                    geometry=support.geometry,
                )
    for identity, buckets in groups.items():
        first, last = requested_bounds or (min(buckets), max(buckets))
        for bucket in range(first, last + 1):
            observations = list(buckets.get(bucket, {}).values())
            expected = slots if daily else _year_slots(sampling, slots, bucket)
            values = [
                float(cast(float, r["value"])) * scales[cast(str, identity[0])]
                for r in observations
                if r["value"] is not None and r["failure"] is None
            ]
            if daily and any(value < 0 for value in values):
                raise ValueError(
                    "DailyTotal requires nonnegative precipitation; decode source "
                    "counter resets and missing values before reduction"
                )
            accepted = bool(values) and (
                len(values) == expected
                if specification.coverage.fraction == 1.0
                else len(values) / expected >= specification.coverage.fraction
            )
            value = (
                math.fsum(values)
                if daily and accepted
                else max(values)
                if isinstance(specification, AnnualMaximum) and accepted
                else min(values)
                if accepted
                else None
            )
            template = templates[identity]
            year, month, day = (
                _date(bucket, sampling.calendar) if daily else (bucket, 1, 1)
            )
            time = f"{year:04d}-{month:02d}-{day:02d}" if daily else f"{year:04d}"
            record = {
                n: template[n]
                for n in (
                    "variable_id",
                    "scenario_id",
                    "period_id",
                    "support_id",
                    "cell_id",
                    "geometry",
                )
            }
            record.update(
                time=time,
                value=value,
                station_id=template.get("station_id"),
                group_id=template.get("group_id"),
                coverage={
                    "valid": len(values),
                    "expected": expected,
                    "minimum_fraction": specification.coverage.fraction,
                },
                failure=None
                if accepted
                else {
                    "code": "incomplete_day" if daily else "incomplete_year",
                    "message": (
                        f"{len(values)} of {expected} expected valid observations; "
                        f"required fraction {specification.coverage.fraction:g}"
                    ),
                },
            )
            records.append(record)
            incomplete += int(not accepted)
            usable += int(accepted)
    diagnostics = metadata.report.diagnostics
    if incomplete:
        diagnostics = (
            *diagnostics,
            ExecutionDiagnostic(
                "incomplete_days" if daily else "incomplete_years",
                cf.DiagnosticSeverity.WARNING,
                Stage.REDUCTION,
                f"{incomplete} temporal groups did not meet "
                "the explicit completeness rule",
                source=None if metadata.source is None else metadata.source.product,
            ),
        )
    report = ExecutionReport(
        True,
        True,
        diagnostics,
        ExecutionCounts(
            requested=len(records),
            produced=len(records),
            no_data=incomplete,
            usable_years=0 if daily else usable,
            incomplete_years=0 if daily else incomplete,
        ),
        metadata.report.source_identities,
    )
    result = replace(planned, report=report)
    return evidence_table(records, result), result
