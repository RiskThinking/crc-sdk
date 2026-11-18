"""Named responses preserve public calibration and asset context semantics."""

from __future__ import annotations

import json
from pathlib import Path

import crc_framework as cf
import pytest

from crc_sdk import (
    Assets,
    Dataset,
    EvaluationArtifact,
    Probability,
    Runtime,
    Stage,
    ValidationError,
    Variable,
)

_DEPTH = "company:flood-depth"


def _depth(measurement: cf.Measurement) -> Dataset:
    return Dataset.from_quantiles(
        {
            "support_id": ["station", "station"],
            "longitude": [24.0, 24.0],
            "latitude": [60.0, 60.0],
            "probability": [0.2, 0.8],
            "value": [0.5, 1.5],
        },
        variable=Variable(_DEPTH, measurement=measurement),
        probability=Probability.annual(tail=cf.Tail.UPPER),
    )


def _assets(region: str | None = cf.FloodRegion.EUROPE.value) -> Assets:
    return Assets(
        {
            "asset_id": ["A01", "A02"],
            "longitude": [24.0, 24.0],
            "latitude": [60.0, 60.0],
            "region": [region, region],
        },
        columns=[],
    )


def _context() -> cf.InundationContext:
    return cf.InundationContext(
        region=cf.FloodRegion.EUROPE, asset_type=cf.AssetType.RESIDENTIAL
    )


def test_bound_column_resolves_once_and_artifact_keeps_model_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    depth = _depth(cf.DepthAboveGround(unit=cf.LengthUnit.METRE))
    assets = _assets()
    reference = cf.Impact.builtin(cf.BuiltinImpact.INUNDATION)
    resolved = reference.resolve(_context())
    evaluation = (
        depth.evaluate(assets)
        .probabilities([0.5])
        .impact(
            "flood_damage",
            reference,
            variable=_DEPTH,
            context=cf.InundationContext(
                region=assets.column("region"), asset_type=cf.AssetType.RESIDENTIAL
            ),
        )
    )
    original = cf.ImpactReference.resolve
    calls: list[str] = []

    def counted(
        model: cf.ImpactReference, context: cf.ImpactContext | None = None
    ) -> cf.Impact:
        assert context is not None
        calls.append(context.to_json())
        return original(model, context)

    path = tmp_path / "responses.parquet"
    with monkeypatch.context() as patch:
        patch.setattr(cf.ImpactReference, "resolve", counted)
        with evaluation.execute(runtime=Runtime(batch_rows=1)) as execution:
            result = execution.read()
            assert execution.validate().valid
            execution.write(path)
            assert calls == [_context().to_json()]

    assert result.report.counts.matched_assets == 2
    assert result.report.counts.no_data == 0
    assert "region" not in result.table.column_names
    rows = result.table.to_pylist()
    assert [(r["asset_id"], r["response_id"]) for r in rows] == [
        ("A01", None),
        ("A01", "flood_damage"),
        ("A02", None),
        ("A02", "flood_damage"),
    ]
    expected = resolved.evaluate(1.0)
    assert [r["value"] for r in rows] == pytest.approx([1.0, expected, 1.0, expected])
    assert {r["resolution_id"] for r in rows} == {None, resolved.sha256}
    (response,) = result.metadata.responses
    assert response.context is not None
    assert json.loads(response.context.to_json())["region"] == {"column": "region"}
    (resolution,) = result.metadata.resolutions
    assert resolution.impact.to_json() == resolved.to_json()

    reopened = EvaluationArtifact.open(path).read()
    assert reopened.table.equals(result.table, check_metadata=True)
    assert reopened.metadata.resolutions[0].impact.sha256 == resolved.sha256
    assert reopened.metadata.responses[0].output == resolved.output


def test_resolved_responses_require_only_their_existing_calibration() -> None:
    reference = cf.Impact.builtin(cf.BuiltinImpact.INUNDATION)
    resolved = reference.resolve(_context())
    evaluation = _depth(resolved.input).evaluate(_assets()).probabilities([0.5])
    with pytest.raises(cf.ImpactError, match="unexpected_context"):
        evaluation.impact("damage", resolved, variable=_DEPTH, context=_context())
    with pytest.raises(cf.ImpactError, match="missing_context.*InundationContext"):
        evaluation.impact("damage", reference, variable=_DEPTH)
    with pytest.raises(cf.ImpactError, match="invalid_context.*InundationContext"):
        evaluation.impact(
            "damage",
            reference,
            variable=_DEPTH,
            context=cf.CountryContext(country="CA"),
        )
    result = evaluation.impact("damage", resolved, variable=_DEPTH).read()
    assert result.metadata.resolutions == ()
    assert result.metadata.responses[0].model.to_json() == resolved.to_json()
    assert all(r["resolution_id"] is None for r in result.table.to_pylist())


@pytest.mark.parametrize(
    ("measurement", "code"),
    [
        (cf.DepthAboveGround(unit=cf.LengthUnit.CENTIMETRE), "unit_mismatch"),
        (
            cf.WaterSurfaceElevation(unit=cf.LengthUnit.METRE, datum="company:datum"),
            "measurement_type_mismatch",
        ),
    ],
)
def test_builtin_response_reports_incompatible_input_science(
    measurement: cf.Measurement, code: str
) -> None:
    evaluation = (
        _depth(measurement)
        .evaluate(_assets())
        .probabilities([0.5])
        .impact(
            "damage",
            cf.Impact.builtin(cf.BuiltinImpact.INUNDATION),
            variable=_DEPTH,
            context=_context(),
        )
    )
    report = evaluation.validate()
    assert report.valid is False
    (diagnostic,) = report.diagnostics
    assert diagnostic.code == code
    assert diagnostic.stage is Stage.EVALUATION
    assert _DEPTH in diagnostic.message
    with pytest.raises(ValidationError) as raised:
        evaluation.read()
    assert raised.value.report.diagnostics == report.diagnostics


def test_unsupported_query_keeps_no_data_without_resolving_asset_context() -> None:
    assets = _assets(region=None)
    depth = _depth(cf.DepthAboveGround(unit=cf.LengthUnit.METRE))
    result = (
        depth.evaluate(assets)
        .probabilities([0.99])
        .impact(
            "damage",
            cf.Impact.builtin(cf.BuiltinImpact.INUNDATION),
            variable=_DEPTH,
            context=cf.InundationContext(
                region=assets.column("region"), asset_type=cf.AssetType.RESIDENTIAL
            ),
        )
        .read()
    )
    assert result.report.valid and result.report.counts.no_data == 4
    assert result.metadata.resolutions == ()
    assert {row["failure"]["code"] for row in result.table.to_pylist()} == {
        "query_outside_support"
    }
