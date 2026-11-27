"""Owned long evidence, explicit fitting, and canonical input layouts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from importlib.resources import files

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    DistributionFamily,
    Duration,
    FitError,
    ModelSelection,
    QuantileCurve,
    QuantileFit,
    SampleEstimator,
    SampleFit,
    SelectionCriterion,
    Tail,
    TimeUnit,
)
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from crc_sdk import Observations, ObservationTimeSelection, Sampling
from crc_sdk._dataset_schema import ExecutionReport, RetainedFailure
from crc_sdk._declarations import (
    Calendar,
    DataSpec,
    Period,
    Probability,
    Scenario,
    SeriesSpec,
    Variable,
)
from crc_sdk._evidence import (
    EvidenceKind,
    EvidenceMetadata,
    FitFailure,
    evidence_arrow_schema,
    evidence_metadata_schema,
    evidence_table,
    fit_sample_table,
    metadata_from_evidence_table,
    validate_evidence,
)
from crc_sdk._schema.dataset import rows_to_table, table_to_rows
from crc_sdk._table_input import (
    QuantileColumns,
    TimeSeriesColumns,
    normalize_quantiles,
    normalize_samples,
    snapshot_table,
)


def samples(count: int = 5) -> dict[str, list[object]]:
    return {
        "support_id": ["site"] * count,
        "longitude": [-79.38] * count,
        "latitude": [43.65] * count,
        "time": list(range(2000, 2000 + count)),
        "value": list(range(1, count + 1)),
    }


def quantiles() -> dict[str, list[object]]:
    return {
        "support_id": ["site"] * 3,
        "longitude": [-79.38] * 3,
        "latitude": [43.65] * 3,
        "probability": [0.9, 0.5, 0.99],
        "value": [0.8, 0.2, 1.6],
    }


def specification() -> SampleFit:
    return SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        estimator=SampleEstimator.L_MOMENTS,
    )


def test_long_samples_remain_readable_and_optional_dimensions_stay_absent() -> None:
    table, metadata = normalize_samples(samples(), variable=Variable("my:rainfall"))
    assert table.schema.equals(evidence_arrow_schema(metadata), check_metadata=True)
    assert table["time"].to_pylist() == ["2000", "2001", "2002", "2003", "2004"]
    assert table["scenario_id"].null_count == table.num_rows
    assert table["period_id"].null_count == table.num_rows
    assert metadata.kind is EvidenceKind.SAMPLES
    assert metadata.report.complete and metadata.report.valid
    assert metadata_from_evidence_table(table) == metadata
    assert EvidenceMetadata.from_json(metadata.to_json()) == metadata
    assert validate_evidence(table).counts.produced == 5
    with pytest.raises(FrozenInstanceError):
        metadata.kind = EvidenceKind.LAYER  # type: ignore[misc,assignment]


def test_explicit_fit_uses_original_observations_and_authoritative_record() -> None:
    table, metadata = normalize_samples(samples(), variable=Variable("my:rainfall"))
    rows, fitted = fit_sample_table(table, metadata, specification())
    assert len(rows) == 1
    assert rows[0].fit is not None and rows[0].fit.converged
    assert rows[0].fit.evidence_count == 5
    assert rows[0].fit.used_count == 5
    assert rows[0].fit.estimator is SampleEstimator.L_MOMENTS
    assert fitted.report.counts.requested == fitted.report.counts.produced == 1
    assert fitted.report.diagnostics[0].stage.value == "fitting"
    assert table.num_rows == 5
    restored = table_to_rows(rows_to_table(rows, fitted))
    assert [row._document() for row in restored] == [row._document() for row in rows]


@pytest.mark.parametrize("policy", [FitFailure.KEEP_NO_DATA, FitFailure.SKIP])
def test_failed_fits_follow_explicit_retention_policy(policy: FitFailure) -> None:
    table, metadata = normalize_samples(samples(2), variable=Variable("my:rainfall"))
    rows, fitted = fit_sample_table(table, metadata, specification(), on_failure=policy)
    assert fitted.report.complete and fitted.report.valid
    assert fitted.report.counts.requested == 1
    assert fitted.report.diagnostics[0].severity.value == "warning"
    if policy is FitFailure.KEEP_NO_DATA:
        assert len(rows) == 1 and rows[0].curve is None and rows[0].fit is None
        assert isinstance(rows[0].failure, RetainedFailure)
        assert rows[0].failure.record.evidence_count == 2
        assert not rows[0].failure.record.converged
        assert fitted.report.counts.no_data == fitted.report.counts.produced == 1
    else:
        assert rows == ()
        assert fitted.report.counts.skipped == 1
        assert fitted.report.counts.no_data == fitted.report.counts.produced == 0


def test_raise_policy_retains_framework_fit_error() -> None:
    table, metadata = normalize_samples(samples(2), variable=Variable("my:rainfall"))
    with pytest.raises(FitError) as raised:
        fit_sample_table(table, metadata, specification())
    assert raised.value.record.evidence_count == 2


def test_incomplete_preview_cannot_become_a_different_fitting_sample() -> None:
    table, metadata = normalize_samples(samples(), variable=Variable("my:rainfall"))
    incomplete = replace(metadata, report=ExecutionReport(complete=False, valid=None))
    with pytest.raises(ValueError, match="incomplete sample preview"):
        fit_sample_table(table.slice(0, 2), incomplete, specification())


def test_owned_snapshots_do_not_borrow_mutable_input_buffers() -> None:
    values = np.array([1.0, 2.0, 3.0])
    table = pa.table({"value": values})
    snapshot = snapshot_table(table)
    values[:] = 99
    assert snapshot["value"].to_pylist() == [1.0, 2.0, 3.0]
    assert snapshot_table(table.to_batches()[0]).num_rows == 3


def test_column_mappings_accept_owned_numpy_columns_and_python_dates() -> None:
    table, _ = normalize_samples(
        {
            "support_id": ["station-a", "station-a"],
            "longitude": np.array([-79.38, -79.38]),
            "latitude": np.array([43.65, 43.65]),
            "time": [date(2000, 2, 29), date(2001, 2, 28)],
            "value": np.array([1.0, 2.0]),
        },
        variable=Variable("my:rainfall"),
    )
    assert table["time"].to_pylist() == ["2000-02-29", "2001-02-28"]
    timed = samples(1)
    timed["time"] = [datetime(2000, 1, 1, tzinfo=UTC)]
    snapshot, _ = normalize_samples(timed, variable=Variable("my:rainfall"))
    assert snapshot["time"].to_pylist() == ["2000-01-01T00:00:00+00:00"]


def test_arrow_c_stream_input_uses_the_same_owned_boundary() -> None:
    source = pa.table({"x": [1, 2]})

    class ArrowInput:
        def __arrow_c_stream__(self, requested_schema: object = None) -> object:
            return source.__arrow_c_stream__(requested_schema)

    assert snapshot_table(ArrowInput()).equals(source)


def test_quantile_inputs_use_supplied_knots_without_fitting() -> None:
    table, metadata = normalize_quantiles(
        quantiles(),
        variable=Variable("my:depth"),
        probability=Probability.annual(tail=Tail.UPPER),
    )
    rows = table_to_rows(table)
    assert len(rows) == 1 and rows[0].fit is None
    assert isinstance(rows[0].curve, QuantileCurve)
    assert rows[0].curve.quantile(0.9) == 0.8
    assert rows[0].curve.probability_interval == (0.5, 0.99)
    assert metadata.spec.series[0].scenario is None
    assert metadata.spec.series[0].period is None


def test_custom_quantile_and_sample_columns_share_identity_mapping_rules() -> None:
    variable = Variable("my:temperature")
    second = Variable("my:rainfall")
    declarations = DataSpec(
        [variable, second], series=[SeriesSpec(variable), SeriesSpec(second)]
    )
    table = samples(2)
    table["variable"] = [variable.id, second.id]
    evidence, metadata = normalize_samples(
        table, spec=declarations, columns=TimeSeriesColumns(variable_id="variable")
    )
    assert evidence["variable_id"].to_pylist() == [variable.id, second.id]
    assert len(metadata.spec.series) == 2
    qtable = quantiles()
    qtable["variable"] = [second.id] * 3
    values, _ = normalize_quantiles(
        qtable, spec=declarations, columns=QuantileColumns(variable_id="variable")
    )
    assert values["variable_id"].to_pylist() == [second.id]
    with pytest.raises(ValueError, match="explicit column mapping"):
        normalize_samples(samples(), spec=declarations)


def test_multiple_scenarios_are_explicit_and_never_cartesian() -> None:
    variable = Variable("my:quantity")
    first, second = Scenario("first"), Scenario("second")
    declarations = DataSpec(
        [variable],
        series=[
            SeriesSpec(variable, scenario=first),
            SeriesSpec(variable, scenario=second),
        ],
    )
    table = samples(2)
    table["scenario"] = [first.id, second.id]
    evidence, _ = normalize_samples(
        table, spec=declarations, columns=TimeSeriesColumns(scenario_id="scenario")
    )
    assert evidence["scenario_id"].to_pylist() == [first.id, second.id]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_missing_or_nonfinite_samples_are_not_silently_dropped(bad: object) -> None:
    table = samples()
    table["value"][0] = bad
    with pytest.raises((TypeError, ValueError)):
        normalize_samples(table, variable=Variable("my:quantity"))


def test_duplicate_observation_identity_is_rejected() -> None:
    table = samples(2)
    table["time"] = [2000, 2000]
    with pytest.raises(ValueError, match="duplicate evidence record"):
        normalize_samples(table, variable=Variable("my:quantity"))


@pytest.mark.parametrize("longitude,latitude", [(181, 0), (0, 91), (float("nan"), 0)])
def test_point_coordinates_are_validated(longitude: float, latitude: float) -> None:
    table = quantiles()
    table["longitude"] = [longitude] * 3
    table["latitude"] = [latitude] * 3
    with pytest.raises(ValueError):
        normalize_quantiles(table, variable=Variable("my:quantity"))


def test_native_calendar_dates_are_preserved_and_checked_against_declaration() -> None:
    table = samples(1)
    table["time"] = ["2000-02-30"]
    unknown, _ = normalize_samples(table, variable=Variable("my:quantity"))
    assert unknown["time"].to_pylist() == ["2000-02-30"]
    evidence, _ = normalize_samples(
        table,
        variable=Variable("my:quantity"),
        period=Period.window(2000, 2000, calendar=Calendar.DAYS_360),
    )
    assert evidence["time"].to_pylist() == ["2000-02-30"]
    with pytest.raises(ValueError, match="invalid day"):
        normalize_samples(
            table,
            variable=Variable("my:quantity"),
            period=Period.window(2000, 2000, calendar=Calendar.GREGORIAN),
        )


def test_observation_time_selection_inherits_the_sampling_calendar() -> None:
    table = samples(1)
    table["time"] = ["2000-02-30"]
    observations = Observations.from_table(
        table,
        variable=Variable("my:quantity"),
        sampling=Sampling(
            calendar=Calendar.DAYS_360, step=Duration(1, unit=TimeUnit.DAY)
        ),
    )
    result = observations.select(time=Period.window(2000, 2000)).read()
    assert result.table["time"].to_pylist() == ["2000-02-30"]
    assert result.metadata.provenance[-1] == ObservationTimeSelection(
        time=Period.window(2000, 2000, calendar=Calendar.DAYS_360)
    )
    contradicting = observations.select(
        time=Period.window(2000, 2000, calendar=Calendar.GREGORIAN)
    )
    with pytest.raises(ValueError, match="calendar contradicts"):
        contradicting.read()


@pytest.mark.parametrize("time", ["2000-02-31", "2000-04-31"])
def test_unknown_calendar_rejects_dates_impossible_in_all_supported_calendars(
    time: str,
) -> None:
    table = samples(1)
    table["time"] = [time]
    with pytest.raises(ValueError, match="invalid day"):
        normalize_samples(table, variable=Variable("my:quantity"))


@pytest.mark.parametrize(
    "time", ["0000", "2000-99-01", "2000-01-99", "2000-01-01Tgarbage"]
)
def test_malformed_observation_times_are_rejected(time: str) -> None:
    table = samples(1)
    table["time"] = [time]
    with pytest.raises(ValueError, match="sample time"):
        normalize_samples(table, variable=Variable("my:quantity"))


def test_empty_long_tables_have_the_same_typed_schema() -> None:
    table, metadata = normalize_samples(samples(0), variable=Variable("my:quantity"))
    assert table.num_rows == 0
    rows, output = fit_sample_table(table, metadata, specification())
    assert rows == ()
    assert output.report.complete and output.report.counts.produced == 0


def test_empty_evidence_still_rejects_a_quantile_fitting_specification() -> None:
    table, metadata = normalize_samples(samples(0), variable=Variable("my:quantity"))
    wrong = ModelSelection(
        candidates=(QuantileFit(family=DistributionFamily.GUMBEL_RIGHT),),
        criterion=SelectionCriterion.NORMALIZED_ROOT_MEAN_SQUARE_ERROR,
    )
    with pytest.raises(TypeError, match="SampleFit candidates"):
        fit_sample_table(table, metadata, wrong)


def test_known_invalid_evidence_cannot_be_relabelled_as_valid_by_fitting() -> None:
    table, metadata = normalize_samples(samples(), variable=Variable("my:quantity"))
    invalid = replace(metadata, report=ExecutionReport(complete=True, valid=False))
    with pytest.raises(ValueError, match="invalid sample evidence"):
        fit_sample_table(table, invalid, specification())


def test_persisted_evidence_rejects_dead_fields_and_untrusted_counts() -> None:
    table, metadata = normalize_samples(samples(), variable=Variable("my:quantity"))
    with pytest.raises(ValueError, match="unknown"):
        EvidenceMetadata.from_json(
            metadata.to_json().replace('"kind":', '"extra":1,"kind":')
        )
    records = table.to_pylist()[:-1]
    with pytest.raises(ValueError, match="counts"):
        evidence_table(records, metadata)
    partial = pa.Table.from_pylist(records, schema=evidence_arrow_schema(metadata))
    assert validate_evidence(partial, check_counts=False).counts.produced == 4


def test_evidence_metadata_schema_is_current_minimal_and_packaged() -> None:
    schema = evidence_metadata_schema()
    Draft202012Validator.check_schema(schema)
    resource = files("crc_sdk").joinpath("contracts/schemas/evidence-1.0.schema.json")
    assert resource.read_text() == json.dumps(schema, indent=2, allow_nan=False) + "\n"
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    assert not {"row", "failure", "metadata"} & definitions.keys()
    # Fitting records are reachable only through fitting-specific diagnostics.
    assert "https://climateriskcommons.org/schemas/fitting/1.0" in resource.read_text()
    assert "numerical/1.0" not in resource.read_text()
    _, metadata = normalize_samples(samples(), variable=Variable("my:quantity"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(
        json.loads(metadata.to_json())
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "0.4"),
        ("kind", "legacy_samples"),
        ("unit", "m"),
        ("rows", []),
    ],
)
def test_header_schema_and_runtime_reject_unknown_fields_and_formats(
    field: str, value: object
) -> None:
    _, metadata = normalize_samples(samples(), variable=Variable("my:quantity"))
    document = json.loads(metadata.to_json())
    document[field] = value
    assert not Draft202012Validator(evidence_metadata_schema()).is_valid(document)
    with pytest.raises((ValueError, TypeError)):
        EvidenceMetadata.from_json(json.dumps(document))


def test_header_scientific_measurements_resolve_from_packaged_schema() -> None:
    from crc_framework import DepthAboveGround, LengthUnit

    measurement_schema = json.loads(
        files("crc_framework")
        .joinpath("contracts/scientific/measurement-1.0.schema.json")
        .read_text()
    )
    registry = Registry().with_resource(
        measurement_schema["$id"], Resource.from_contents(measurement_schema)
    )
    _, metadata = normalize_samples(
        samples(),
        variable=Variable(
            "my:depth", measurement=DepthAboveGround(unit=LengthUnit.METRE)
        ),
    )
    document = json.loads(metadata.to_json())
    validator = Draft202012Validator(evidence_metadata_schema(), registry=registry)
    validator.validate(document)
    invalid = deepcopy(document)
    invalid["spec"]["variables"][0]["measurement"]["unit"] = "unknown-unit"
    assert not validator.is_valid(invalid)
    with pytest.raises((ValueError, TypeError)):
        EvidenceMetadata.from_json(json.dumps(invalid))


def test_header_schema_and_runtime_reject_an_incomplete_valid_claim() -> None:
    _, metadata = normalize_samples(samples(), variable=Variable("my:quantity"))
    document = json.loads(metadata.to_json())
    document["report"]["complete"] = False
    assert not Draft202012Validator(evidence_metadata_schema()).is_valid(document)
    with pytest.raises(ValueError, match="complete"):
        EvidenceMetadata.from_json(json.dumps(document))
