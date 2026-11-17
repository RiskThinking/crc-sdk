"""Station-specific complete-day extrema from irregular native reports."""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import cast

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]

from ._aggregation import Coverage, ReductionKind, Sampling, TemporalReduction
from ._dataset_schema import (
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    Stage,
)
from ._declarations import Calendar
from ._evidence import EvidenceMetadata, evidence_table


def complete_daily_extrema(
    records: Sequence[dict[str, object]],
    metadata: EvidenceMetadata,
    *,
    year: int,
    maximum: bool,
) -> tuple[pa.Table, EvidenceMetadata]:
    """Require an accepted report in every hour before retaining a daily extreme.

    NOAA station files can contain several reports in an hour. Their exact native
    timestamps remain available in Observations; hourly coverage is distinct from
    the number of reports, and several reports never substitute for a missing hour.
    """
    groups: dict[tuple[object, ...], dict[date, list[dict[str, object]]]] = {}
    templates: dict[tuple[object, ...], dict[str, object]] = {}
    for row in records:
        identity = tuple(
            row[name]
            for name in (
                "variable_id",
                "scenario_id",
                "period_id",
                "support_id",
                "cell_id",
            )
        )
        when = datetime.fromisoformat(cast(str, row["time"]))
        if when.year != year:
            raise ValueError("Station daily reduction contains a different year")
        groups.setdefault(identity, {}).setdefault(when.date(), []).append(row)
        templates.setdefault(identity, row)
    result: list[dict[str, object]] = []
    incomplete = 0
    first = date(year, 1, 1)
    day_count = 366 if calendar.isleap(year) else 365
    for identity, days in groups.items():
        day = first
        for offset in range(day_count):
            day = first + timedelta(days=offset)
            reports = days.get(day, ())
            valid = [
                row
                for row in reports
                if row["value"] is not None and row["failure"] is None
            ]
            hours = {
                datetime.fromisoformat(cast(str, row["time"])).hour for row in valid
            }
            accepted = len(hours) == 24
            values = [cast(float, row["value"]) for row in valid]
            value = (max(values) if maximum else min(values)) if accepted else None
            row = {
                key: templates[identity][key]
                for key in (
                    "variable_id",
                    "scenario_id",
                    "period_id",
                    "support_id",
                    "cell_id",
                    "geometry",
                    "station_id",
                    "group_id",
                )
            }
            row.update(
                time=day.isoformat(),
                value=value,
                coverage={"valid": len(hours), "expected": 24, "minimum_fraction": 1.0},
                failure=None
                if accepted
                else {
                    "code": "incomplete_day",
                    "message": (
                        f"{len(hours)} of 24 hours have "
                        "a source-quality-accepted report."
                    ),
                },
            )
            incomplete += int(not accepted)
            result.append(row)
    daily = Sampling(
        calendar=Calendar.GREGORIAN, step=cf.Duration(1, unit=cf.TimeUnit.DAY)
    )
    reduction = TemporalReduction(
        operation=ReductionKind.DAILY_MAXIMUM
        if maximum
        else ReductionKind.DAILY_MINIMUM,
        coverage=Coverage.complete(),
        input_sampling=Sampling(
            calendar=Calendar.GREGORIAN, step=cf.Duration(1, unit=cf.TimeUnit.HOUR)
        ),
        output_sampling=daily,
    )
    diagnostics = metadata.report.diagnostics
    if incomplete:
        diagnostics = (
            *diagnostics,
            ExecutionDiagnostic(
                "incomplete_days",
                cf.DiagnosticSeverity.WARNING,
                Stage.REDUCTION,
                f"{incomplete} station days lack accepted reports "
                "in all 24 distinct hours.",
            ),
        )
    output = replace(
        metadata,
        sampling=daily,
        provenance=(*metadata.provenance, reduction),
        report=ExecutionReport(
            True,
            True,
            diagnostics,
            ExecutionCounts(
                requested=len(result), produced=len(result), no_data=incomplete
            ),
            metadata.report.source_identities,
        ),
    )
    return evidence_table(result, output), output
