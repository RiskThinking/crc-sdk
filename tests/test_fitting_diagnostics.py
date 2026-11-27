"""Failed modelling retains its authoritative record through every report."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from importlib.resources import files
from pathlib import Path

import pytest
from crc_framework import (
    DiagnosticSeverity,
    DistributionFamily,
    FitError,
    FitFailureDiagnostic,
    SampleFit,
    fit_samples,
)
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from crc_sdk._dataset_schema import FittingDiagnostic, Stage, _diagnostic
from crc_sdk._declarations import Variable
from crc_sdk._errors import ValidationError
from crc_sdk._evidence import FitFailure, evidence_metadata_schema
from crc_sdk._recipes import Dataset, Samples


def failed_fit() -> FitError:
    try:
        fit_samples([1.0, 2.0], SampleFit(family=DistributionFamily.GUMBEL_RIGHT))
    except FitError as error:
        return error
    raise AssertionError("two observations must fail the declared minimum count")


def sample_recipe() -> Samples:
    return Samples.from_table(
        {
            "support_id": ["station-a", "station-a"],
            "longitude": [-79.38, -79.38],
            "latitude": [43.65, 43.65],
            "time": [2000, 2001],
            "value": [1.0, 2.0],
        },
        variable=Variable("my:rainfall"),
    )


def test_fitting_diagnostics_are_typed_immutable_and_roundtrip_the_actual_trace() -> (
    None
):
    error = failed_fit()
    diagnostic = FittingDiagnostic(
        code=error.code,
        severity=DiagnosticSeverity.ERROR,
        stage=Stage.FITTING,
        message=error.message,
        support_id="station-a",
        record=error.record,
    )
    decoded = _diagnostic(json.loads(json.dumps(diagnostic._document())))
    assert isinstance(decoded, FittingDiagnostic)
    assert decoded == diagnostic
    assert decoded.record.specification == SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT
    )
    assert decoded.record.evidence_count == 2 and not decoded.record.converged
    with pytest.raises(FrozenInstanceError):
        decoded.record = error.record  # type: ignore[misc]
    with pytest.raises(ValueError, match="Stage.FITTING"):
        replace(diagnostic, stage=Stage.DECODING)


def test_report_schema_accepts_fitting_records_only_for_fitting_diagnostics() -> None:
    error = failed_fit()
    report = sample_recipe().read().metadata
    diagnostic = FittingDiagnostic(
        error.code,
        DiagnosticSeverity.WARNING,
        Stage.FITTING,
        error.message,
        record=error.record,
    )
    metadata = replace(
        report,
        report=replace(report.report, diagnostics=(diagnostic,)),
    )
    schema = json.loads(
        files("crc_framework")
        .joinpath("contracts/numerical/fitting-1.1.schema.json")
        .read_text()
    )
    registry = Registry().with_resource(schema["$id"], Resource.from_contents(schema))
    validator = Draft202012Validator(evidence_metadata_schema(), registry=registry)
    document = json.loads(metadata.to_json())
    validator.validate(document)
    document["report"]["diagnostics"][0]["stage"] = "acquisition"
    assert not validator.is_valid(document)


def test_default_fit_failure_validation_retains_the_record_after_scope_exit() -> None:
    recipe = sample_recipe().fit(SampleFit(family=DistributionFamily.GUMBEL_RIGHT))
    with recipe.execute() as execution:
        report = execution.validate()
        assert report.valid is False
        diagnostics = [
            d for d in report.diagnostics if isinstance(d, FittingDiagnostic)
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0].record.evidence_count == 2
        assert diagnostics[0].record.attempts
        assert diagnostics[0].support_id == "station-a"
        assert "my:rainfall" in diagnostics[0].message
        with pytest.raises(ValidationError) as raised:
            execution.read()
        assert raised.value.report.diagnostics == report.diagnostics
    outcome = diagnostics[0].record.attempts[0].diagnostic
    assert isinstance(outcome, FitFailureDiagnostic)
    assert outcome.message
    assert report.valid is False


@pytest.mark.parametrize("policy", [FitFailure.KEEP_NO_DATA, FitFailure.SKIP])
def test_retained_fit_failures_keep_records_in_valid_reports_and_saved_artifacts(
    tmp_path: Path,
    policy: FitFailure,
) -> None:
    recipe = sample_recipe().fit(
        SampleFit(family=DistributionFamily.GUMBEL_RIGHT), on_failure=policy
    )
    report = recipe.validate()
    assert report.complete and report.valid
    diagnostics = [d for d in report.diagnostics if isinstance(d, FittingDiagnostic)]
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is DiagnosticSeverity.WARNING
    assert diagnostics[0].record.evidence_count == 2
    assert diagnostics[0].support_id == "station-a"
    assert "my:rainfall" in diagnostics[0].message
    saved = recipe.write(tmp_path / f"{policy.value}.parquet")
    assert saved.path is not None
    reopened = Dataset.open(saved.path).read()
    matching = [
        d for d in reopened.report.diagnostics if isinstance(d, FittingDiagnostic)
    ]
    assert len(matching) == 1 and matching[0].record == diagnostics[0].record
    assert reopened.report.complete and reopened.report.valid
    assert reopened.report.counts == report.counts
