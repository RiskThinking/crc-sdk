from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from crc_sdk._aggregation import (
    AnnualMaximum,
    AnnualMinimum,
    Coverage,
    DailyTotal,
    Sampling,
    TemporalReduction,
    TimestampPosition,
    _date,
    _ordinal,
    days_in_month,
)
from crc_sdk._dataset_schema import FittingDiagnostic
from crc_sdk._declarations import Calendar, Period, Probability, SeriesSpec, Variable
from crc_sdk._errors import CapabilityError, ValidationError
from crc_sdk._evidence import (
    EvidenceKind,
    EvidenceMetadata,
    FitFailure,
    evidence_arrow_schema,
)
from crc_sdk._evidence_recipes import Layer, Observations, Projections
from crc_sdk._recipes import Samples
from crc_sdk._schema import table_to_rows
from crc_sdk._table_input import LayerColumns, ProjectionColumns


def observations(
    times: list[str],
    values: Sequence[float | None],
    *,
    calendar: Calendar = Calendar.GREGORIAN,
    step: cf.Duration | None = None,
    timestamp: TimestampPosition = TimestampPosition.START,
    measurement: cf.Measurement | None = None,
) -> Observations:
    return Observations.from_table(
        {
            "support_id": ["a"] * len(times),
            "longitude": [0] * len(times),
            "latitude": [0] * len(times),
            "time": times,
            "value": values,
        },
        variable=Variable(
            "user:value",
            measurement=measurement or cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
        ),
        sampling=Sampling(
            calendar=calendar,
            step=step or cf.Duration(1, unit=cf.TimeUnit.DAY),
            timestamp=timestamp,
        ),
    )


def amount() -> cf.PrecipitationAmount:
    return cf.PrecipitationAmount(
        unit=cf.LengthUnit.MILLIMETRE,
        accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.HOUR),
    )


def test_interval_end_accumulation_belongs_to_the_day_it_observed() -> None:
    times = [f"2000-01-01T{hour:02d}:00:00Z" for hour in range(1, 24)] + [
        "2000-01-02T00:00:00Z"
    ]
    result = (
        observations(
            times,
            [1.0] * 24,
            step=cf.Duration(1, unit=cf.TimeUnit.HOUR),
            timestamp=TimestampPosition.END,
            measurement=amount(),
        )
        .aggregate(DailyTotal())
        .read()
    )
    row = result.table.to_pylist()[0]
    assert row["time"] == "2000-01-01" and row["value"] == 24
    assert row["coverage"] == {"valid": 24, "expected": 24, "minimum_fraction": 1.0}
    measurement = result.metadata.spec.variables[0].measurement
    assert measurement == cf.PrecipitationAmount(
        unit=cf.LengthUnit.MILLIMETRE,
        accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.DAY),
    )
    assert isinstance(result.metadata.provenance[-1], TemporalReduction)
    assert result.metadata.provenance[-1].measurements[0].input == amount()


def test_daily_total_rejects_intervals_crossing_midnight() -> None:
    times = [f"2000-01-01T{hour:02d}:30:00Z" for hour in range(24)]
    source = observations(
        times,
        [0.0] * 23 + [20.0],
        step=cf.Duration(1, unit=cf.TimeUnit.HOUR),
        measurement=amount(),
    )
    with pytest.raises(ValidationError, match="crossing midnight"):
        source.aggregate(DailyTotal()).read()


def test_annual_reduction_rejects_an_interval_crossing_year_end() -> None:
    source = observations(
        ["2000-12-31T23:30:00Z"],
        [20.0],
        step=cf.Duration(1, unit=cf.TimeUnit.HOUR),
        measurement=amount(),
    )
    with pytest.raises(ValidationError, match="crossing the native year boundary"):
        source.aggregate(AnnualMaximum()).read()


def test_annual_maximum_hourly_amount_is_not_the_wettest_day() -> None:
    times = [
        f"2000-01-{day:02d}T{hour:02d}:00:00Z" for day in (1, 2) for hour in range(24)
    ]
    source = observations(
        times,
        [1.0] * 24 + [2.0] + [0.0] * 23,
        step=cf.Duration(1, unit=cf.TimeUnit.HOUR),
        measurement=amount(),
    )
    policy = AnnualMaximum(coverage=Coverage.minimum_fraction(0.001))
    hourly = source.aggregate(policy).read()
    daily = source.aggregate(DailyTotal()).aggregate(policy).read()
    assert hourly.table["value"].to_pylist() == [2.0]
    assert daily.table["value"].to_pylist() == [24.0]
    assert len(daily.metadata.provenance) == 2


@pytest.mark.parametrize("calendar", list(Calendar))
def test_native_calendar_full_year_completeness(calendar: Calendar) -> None:
    count = sum(days_in_month(2000, month, calendar) for month in range(1, 13))
    first = _ordinal(2000, 1, 1, calendar)
    times = [
        f"{year:04d}-{month:02d}-{day:02d}"
        for offset in range(count)
        for year, month, day in [_date(first + offset, calendar)]
    ]
    result = (
        observations(times, [float(i) for i in range(count)], calendar=calendar)
        .aggregate(AnnualMaximum())
        .read()
    )
    assert result.table["value"].to_pylist() == [count - 1]
    assert result.report.counts.usable_years == 1
    assert result.report.counts.incomplete_years == 0
    assert result.table["coverage"].to_pylist()[0]["expected"] == count


def test_incomplete_year_is_retained_and_not_fitted_as_a_valid_observation() -> None:
    annual = observations(["2000-01-01"], [4.0]).aggregate(AnnualMaximum())
    read = annual.read()
    assert read.table["value"].to_pylist() == [None]
    assert read.table["failure"].to_pylist()[0]["code"] == "incomplete_year"
    assert read.report.counts.no_data == read.report.counts.incomplete_years == 1
    specification = cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT)
    report = annual.fit(specification).validate()
    assert report.valid is False
    failure = next(d for d in report.diagnostics if isinstance(d, FittingDiagnostic))
    assert failure.record.evidence_count == failure.record.excluded_count == 1
    assert failure.record.used_count == 0
    kept = annual.fit(specification, on_failure=FitFailure.KEEP_NO_DATA).read()
    row = table_to_rows(kept.table)[0]
    assert row.curve is None and row.failure is not None
    assert row.failure.record.excluded_count == 1


def test_missing_year_between_observed_years_remains_visible() -> None:
    result = (
        observations(["2000-01-01", "2002-12-31"], [1.0, 5.0])
        .aggregate(AnnualMaximum(coverage=Coverage.minimum_fraction(0.001)))
        .read()
    )
    assert result.table["time"].to_pylist() == ["2000", "2001", "2002"]
    assert result.table["value"].to_pylist() == [1.0, None, 5.0]
    assert result.report.counts.usable_years == 2
    assert result.report.counts.incomplete_years == 1


def test_requested_years_outside_observed_coverage_remain_visible() -> None:
    source = observations(["2001"], [5.0], step=cf.Duration(1, unit=cf.TimeUnit.YEAR))
    requested = Period.window(2000, 2002)
    selected = source.select(time=requested)
    for annual in (
        selected.aggregate(AnnualMaximum()),
        selected.read().aggregate(AnnualMaximum()),
        source.aggregate(AnnualMaximum()).select(time=requested),
    ):
        result = annual.read()
        assert result.table["time"].to_pylist() == ["2000", "2001", "2002"]
        assert result.table["value"].to_pylist() == [None, 5.0, None]
        assert result.report.counts.requested == 3
        assert result.report.counts.usable_years == 1
        assert result.report.counts.incomplete_years == 2
        assert result.table["coverage"].to_pylist()[0] == {
            "valid": 0,
            "expected": 1,
            "minimum_fraction": 1.0,
        }


def test_reopened_annual_samples_recount_selected_years(tmp_path: Path) -> None:
    annual = observations(
        ["2001", "2002", "2003"],
        [5.0, 6.0, None],
        step=cf.Duration(1, unit=cf.TimeUnit.YEAR),
    ).aggregate(AnnualMaximum())
    path = tmp_path / "annual.parquet"
    annual.write(path)
    for year, usable in ((2001, 1), (2003, 0)):
        requested = Period.window(year, year)
        live = annual.select(time=requested).read()
        reopened = Samples.open(path).select(time=requested).read()
        assert reopened.table["time"].to_pylist() == [str(year)]
        assert reopened.report.counts == live.report.counts
        assert reopened.report.counts.usable_years == usable
        assert reopened.report.counts.incomplete_years == 1 - usable
        assert bool(reopened.report.diagnostics) == (usable == 0)


@pytest.mark.parametrize("mutation", ["fraction", "expected", "missing", "counts"])
def test_saved_annual_samples_reject_contradictory_coverage(
    tmp_path: Path, mutation: str
) -> None:
    annual = observations(["2001-01-01"], [3.0]).aggregate(AnnualMaximum()).read()
    record = annual.table.to_pylist()[0]
    metadata = json.loads(annual.metadata.to_json())
    if mutation == "fraction":
        record["value"] = 3.0
        record["failure"] = None
        record["coverage"]["minimum_fraction"] = 0.001
        metadata["report"]["counts"].update(
            no_data=0, incomplete_years=0, usable_years=1
        )
    elif mutation == "expected":
        record["coverage"]["expected"] *= 2
    elif mutation == "missing":
        record["coverage"] = None
    else:
        metadata["report"]["counts"].update(usable_years=17, incomplete_years=88)
    table = pa.Table.from_pylist([record], schema=annual.table.schema)
    path = tmp_path / "contradictory.parquet"
    pq.write_table(
        table.replace_schema_metadata({b"crc.evidence": json.dumps(metadata).encode()}),
        path,
    )
    recipe = Samples.open(path)
    assert recipe.validate().valid is False
    with pytest.raises(ValidationError, match="coverage|annual evidence counts"):
        recipe.read()


def test_monthly_native_observations_use_twelve_months_not_invented_days() -> None:
    source = observations(
        [f"2000-{month:02d}-01" for month in range(1, 13)],
        [float(month) for month in range(1, 13)],
        calendar=Calendar.DAYS_360,
        step=cf.Duration(1, unit=cf.TimeUnit.MONTH),
    )
    result = source.aggregate(AnnualMinimum()).read()
    assert result.table["value"].to_pylist() == [1.0]
    assert result.table["coverage"].to_pylist()[0]["expected"] == 12
    assert result.metadata.spec.series[0].probability == Probability.annual(
        tail=cf.Tail.LOWER
    )
    with pytest.raises(CapabilityError):
        source.aggregate(DailyTotal()).read()


def test_calendar_sampling_rejects_a_midmonth_interval_start() -> None:
    source = observations(
        ["2000-12-15"],
        [5.0],
        calendar=Calendar.DAYS_360,
        step=cf.Duration(1, unit=cf.TimeUnit.MONTH),
    )
    with pytest.raises(ValidationError, match="month or year boundary"):
        source.aggregate(AnnualMaximum()).read()


def test_pre_transition_gregorian_reduction_is_rejected_instead_of_relabelled() -> None:
    source = observations(["1500-01-01"], [1.0])
    report = source.aggregate(AnnualMaximum()).validate()
    assert report.valid is False
    assert "1583" in report.diagnostics[-1].message


def test_explicit_relaxed_rule_is_retained_without_rescaling() -> None:
    times = [f"2000-01-01T{hour:02d}:00:00Z" for hour in range(23)]
    source = observations(
        times,
        [1.0] * 23,
        step=cf.Duration(1, unit=cf.TimeUnit.HOUR),
        measurement=amount(),
    )
    complete = source.aggregate(DailyTotal()).read()
    relaxed = source.aggregate(
        DailyTotal(coverage=Coverage.minimum_fraction(0.95))
    ).read()
    assert complete.table["value"].to_pylist() == [None]
    assert relaxed.table["value"].to_pylist() == [23.0]
    assert relaxed.table["coverage"].to_pylist()[0] == {
        "valid": 23,
        "expected": 24,
        "minimum_fraction": 0.95,
    }


def test_daily_total_integrates_rates_and_records_changed_measurement() -> None:
    rate = cf.PrecipitationRate(
        unit=cf.PrecipitationRateUnit.MILLIMETRES_PER_DAY,
        accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.HOUR),
        aggregation=cf.Aggregation.MEAN,
    )
    times = [f"2000-01-01T{hour:02d}:00:00Z" for hour in range(24)]
    result = (
        observations(
            times,
            [24.0] * 24,
            step=cf.Duration(1, unit=cf.TimeUnit.HOUR),
            measurement=rate,
        )
        .aggregate(DailyTotal())
        .read()
    )
    assert result.table["value"].to_pylist() == [24.0]
    reduction = result.metadata.provenance[-1]
    assert isinstance(reduction, TemporalReduction)
    change = reduction.measurements[0]
    assert change.input == rate and isinstance(change.output, cf.PrecipitationAmount)


def test_unknown_sampling_is_readable_but_cannot_be_aggregated() -> None:
    source = Observations.from_table(
        {
            "support_id": ["a"],
            "longitude": [0],
            "latitude": [0],
            "time": [2000],
            "value": [1],
        },
        variable=Variable("a"),
    )
    assert source.read().table.num_rows == 1
    with pytest.raises(CapabilityError, match="Sampling"):
        source.aggregate(AnnualMaximum()).read()


def test_interval_and_projection_coordinates_are_not_annual_cdf_knots(
    tmp_path: Path,
) -> None:
    projection = Projections.from_table(
        {
            "support_id": ["station"],
            "longitude": [0],
            "latitude": [0],
            "T": [100],
            "estimate": [20],
            "lo": [17],
            "hi": [25],
            "confidence": [0.95],
        },
        variable=Variable("idf"),
        probability=Probability.confidence(),
        columns=ProjectionColumns(
            return_period="T",
            value="estimate",
            lower="lo",
            upper="hi",
            confidence="confidence",
        ),
    )
    assert not hasattr(projection, "fit")
    row = projection.read().table.to_pylist()[0]
    assert row["coordinate"] == {"kind": "return_period", "time": None, "years": 100.0}
    assert row["probability"] is None and row["interval"]["confidence"] == 0.95
    saved = projection.write(tmp_path / "projection.parquet")
    assert type(saved) is Projections
    assert saved.path is not None
    assert Projections.open(saved.path).read().table.to_pylist() == [row]


def test_geometry_only_layer_keeps_its_actual_footprint(tmp_path: Path) -> None:
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    layer = Layer.from_table(
        {"support_id": ["field"], "wkb": [polygon.wkb]},
        variable=Variable("field_boundary"),
        columns=LayerColumns(geometry="wkb", value=None),
    )
    assert not hasattr(layer, "fit") and not hasattr(layer, "aggregate")
    read = layer.read()
    assert read.table["geometry"].to_pylist() == [polygon.wkb]
    assert read.report.counts.no_data == 0
    saved = layer.write(tmp_path / "layer.parquet")
    assert type(saved) is Layer
    assert saved.path is not None
    assert Layer.open(saved.path).read().metadata.kind is EvidenceKind.LAYER


def test_genuine_sample_missing_values_are_visible_and_excluded_from_the_fit() -> None:
    samples = Samples.from_table(
        {
            "support_id": ["a"] * 5,
            "longitude": [0] * 5,
            "latitude": [0] * 5,
            "time": [2000, 2001, 2002, 2003, 2004],
            "value": [1, None, 3, 7, 9],
        },
        variable=Variable("a"),
    )
    assert samples.read().report.counts.no_data == 1
    result = samples.fit(cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT)).read()
    record = table_to_rows(result.table)[0].fit
    assert (
        record is not None
        and record.evidence_count == 5
        and record.used_count == 4
        and record.excluded_count == 1
    )


def test_schema_kind_and_calendar_contradictions_are_rejected() -> None:
    source = observations(["2000-01-01"], [1.0])
    metadata: EvidenceMetadata = source.read().metadata
    with pytest.raises(ValueError, match="projection"):
        replace(
            metadata,
            kind=EvidenceKind.PROJECTIONS,
            spec=metadata.spec.__class__(
                [Variable("a")],
                series=[
                    SeriesSpec(
                        Variable("a"),
                        probability=Probability.annual(tail=cf.Tail.UPPER),
                    )
                ],
            ),
        )
    assert evidence_arrow_schema(metadata).get_field_index("time") >= 0
    assert (
        evidence_arrow_schema(
            replace(metadata, kind=EvidenceKind.LAYER)
        ).get_field_index("time")
        == -1
    )


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan"), True])
def test_invalid_completeness_policies_fail_before_execution(fraction: float) -> None:
    with pytest.raises(ValueError):
        Coverage.minimum_fraction(fraction)
