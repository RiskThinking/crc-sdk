"""Named responses preserve public calibration and asset context semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from jsonschema import Draft202012Validator

from crc_sdk import (
    Assets,
    ColumnBinding,
    ContextualImpact,
    Dataset,
    EvaluationArtifact,
    EvaluationResolution,
    Probability,
    ResponseContext,
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


def test_contextual_impact_persists_verified_user_cases(tmp_path: Path) -> None:
    class Use(Enum):
        DEPOT = "depot"
        OFFICE = "office"

    @dataclass(frozen=True)
    class Facility(ResponseContext):
        use: Use | ColumnBinding

    measurement = cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
    days = cf.TimeDuration(unit=cf.TimeUnit.DAY)
    depot = cf.Impact.linear(slope=2, input=measurement, output=days)
    office = cf.Impact.tabulated(
        [(0, 0), (1, 3), (2, 5)], input=measurement, output=days
    )
    cases = [(Facility(Use.DEPOT), depot), (Facility(Use.OFFICE), office)]
    model = ContextualImpact(cases)
    assert model.requirements.context_type is Facility
    assert model.requirements.context_fields[0].choices == ("depot", "office")
    assert model.sha256 == ContextualImpact(list(reversed(cases))).sha256
    assert ContextualImpact.from_json(model.to_json()).to_json() == model.to_json()
    assets = Assets(
        {
            "asset_id": ["A01", "A02"],
            "longitude": [24.0, 24.0],
            "latitude": [60.0, 60.0],
            "use": [Use.DEPOT.value, Use.OFFICE.value],
        },
        columns=[],
    )
    recipe = (
        _depth(measurement)
        .evaluate(assets)
        .probabilities([0.5])
        .impact(
            "closure", model, variable=_DEPTH, context=Facility(assets.column("use"))
        )
    )
    saved = recipe.write(tmp_path / "cases.parquet")
    result = EvaluationArtifact.open(saved.path).read(runtime=Runtime(batch_rows=1))
    assert result.table["value"].to_pylist() == pytest.approx([1.0, 2.0, 1.0, 3.0])
    assert "asset" not in result.table.column_names
    first, second = result.metadata.resolutions
    assert first.id != second.id
    assert first.asset_ids == ("A01",) and second.asset_ids == ("A02",)
    assert (
        first.context is not None and first.context.to_json() == cases[0][0].to_json()
    )
    reopened_model = result.metadata.responses[0].model
    assert isinstance(reopened_model, ContextualImpact)
    assert isinstance(first.context, reopened_model.requirements.context_type)
    assert getattr(first.context, "use") == Use.DEPOT.value
    Draft202012Validator.check_schema(first.context.schema)
    forged = replace(first, impact=office)
    with pytest.raises(ValueError, match="contradicts the calibrated case"):
        replace(result.metadata, resolutions=(forged, second))
    missing = recipe.impact(
        "closure",
        model,
        variable=_DEPTH,
        context=Facility(assets.column("unknown")),
        replace=True,
    )
    assert [r.code for r in missing.inspect().requirements] == [
        "asset_context_columns_required"
    ]


def test_reopened_artifact_rejects_inconsistent_retained_attributes(
    tmp_path: Path,
) -> None:
    @dataclass(frozen=True)
    class Facility(ResponseContext):
        use: str | ColumnBinding

    measurement = cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
    office = cf.Impact.linear(slope=2, input=measurement, output=cf.Ratio())
    model = ContextualImpact([(Facility("office"), office)])
    assets = Assets(
        {
            "asset_id": ["x"],
            "longitude": [24.0],
            "latitude": [60.0],
            "use": ["office"],
        },
        columns=["use"],
    )
    saved = (
        _depth(measurement)
        .evaluate(assets)
        .probabilities([0.5])
        .impact(
            "closure",
            model,
            variable=_DEPTH,
            context=Facility(assets.column("use")),
        )
        .write(tmp_path / "original.parquet")
    )
    table = pq.read_table(saved.path)
    rows = table.to_pylist()
    rows[0]["asset"]["use"] = "depot"
    altered = tmp_path / "altered.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), altered)
    artifact = EvaluationArtifact.open(altered)
    assert artifact.validate(runtime=Runtime(batch_rows=1)).valid is False
    with pytest.raises(
        ValidationError, match="retained attributes disagree across rows"
    ):
        artifact.read()


def test_contextual_impact_checks_measurements_and_unknown_cases() -> None:
    @dataclass(frozen=True)
    class Facility(ResponseContext):
        use: str
        district: str

    input = cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
    slope = cf.Impact.linear(slope=1, input=input, output=cf.Ratio())
    day = cf.Impact.linear(
        slope=1, input=input, output=cf.TimeDuration(unit=cf.TimeUnit.DAY)
    )
    first, second = Facility("depot", "north"), Facility("office", "south")
    with pytest.raises(ValueError, match="identical input and output"):
        ContextualImpact([(first, slope), (second, day)])
    with pytest.raises(ValueError, match="duplicate response context"):
        ContextualImpact([(first, slope), (first, slope)])
    model = ContextualImpact([(first, slope), (second, slope)])
    assert (
        EvaluationResolution("repair", slope, asset_ids=("a",), context=first).id
        != EvaluationResolution("repair", slope, asset_ids=("b",), context=second).id
    )
    with pytest.raises(cf.ImpactError, match="unknown_context_case"):
        model.resolve(Facility("depot", "south"))
    with pytest.raises(cf.ImpactError, match="invalid_context_value"):
        model.resolve(Facility("Depot", "north"))


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
    assert resolution.context == _context()
    assert resolution.asset_ids == ("A01", "A02")

    reopened = EvaluationArtifact.open(path).read()
    assert reopened.table.equals(result.table, check_metadata=True)
    assert reopened.metadata.resolutions[0].impact.sha256 == resolved.sha256
    assert reopened.metadata.responses[0].output == resolved.output


def test_reopening_rejects_swapped_resolutions_for_hidden_context_columns(
    tmp_path: Path,
) -> None:
    reference = cf.Impact.builtin(cf.BuiltinImpact.INUNDATION)
    assets = Assets(
        {
            "asset_id": ["A01", "A02"],
            "longitude": [24.0, 24.0],
            "latitude": [60.0, 60.0],
            "region": [cf.FloodRegion.EUROPE.value, cf.FloodRegion.NORTH_AMERICA.value],
        },
        columns=[],
    )
    path = tmp_path / "responses.parquet"
    (
        _depth(reference.requirements.input)
        .evaluate(assets)
        .probabilities([0.5])
        .impact(
            "damage",
            reference,
            variable=_DEPTH,
            context=cf.InundationContext(
                region=assets.column("region"), asset_type=cf.AssetType.RESIDENTIAL
            ),
        )
        .select(responses="damage")
        .write(path)
    )
    original = EvaluationArtifact.open(path).read()
    assert "asset" not in original.table.column_names
    assert [r.asset_ids for r in original.metadata.resolutions] == [("A01",), ("A02",)]
    left, right = original.table.to_pylist()
    assert left["resolution_id"] != right["resolution_id"]
    for column in ("resolution_id", "value"):
        left[column], right[column] = right[column], left[column]
    invalid = tmp_path / "swapped.parquet"
    pq.write_table(
        pa.Table.from_pylist([left, right], schema=original.table.schema), invalid
    )

    with pytest.raises(
        ValidationError, match="did not use the recorded response resolution"
    ):
        EvaluationArtifact.open(invalid).read()
    assert EvaluationArtifact.open(invalid).validate().valid is False


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
