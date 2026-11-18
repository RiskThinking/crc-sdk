"""Agreement between the evaluation JSON contract and typed Arrow admission."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from crc_sdk._assets import Assets
from crc_sdk._dataset_schema import (
    ExecutionReport,
    _dump,
)
from crc_sdk._declarations import DataSpec, Probability, SeriesSpec
from crc_sdk._errors import CapabilityError
from crc_sdk._evaluation_schema import (
    EvaluationFailure,
    EvaluationInput,
    EvaluationMetadata,
    EvaluationQuery,
    EvaluationResponse,
    EvaluationRow,
    EvaluationValidator,
    QueryKind,
    evaluation_arrow_schema,
    evaluation_metadata_from_schema,
    evaluation_table,
    validate_evaluation,
)
from crc_sdk._evaluation_spatial import Distance, SpatialMatch
from crc_sdk._schema.evaluation_contract import evaluation_json_schema


def _metadata() -> EvaluationMetadata:
    accepted = _fixtures()["accepted"]
    return EvaluationMetadata.from_json(_dump(accepted[0]["document"]["metadata"]))


def _fixtures() -> dict[str, Any]:
    contract = files("crc_sdk").joinpath("contracts/fixtures/evaluation-2.0.json")
    return cast(dict[str, Any], json.loads(contract.read_text(encoding="utf-8")))


def _rows(values: list[dict[str, Any]]) -> list[EvaluationRow]:
    rows = []
    for raw in values:
        value, failure = raw["value"], raw["failure"]
        row = {
            **raw,
            "value": float(value) if isinstance(value, str) else value,
            "failure": None
            if failure is None
            else EvaluationFailure(
                failure["code"],
                failure["message"],
                None
                if "record" not in failure
                else cf.FitRecord.from_json(_dump(failure["record"])),
            ),
        }
        rows.append(EvaluationRow(**row))
    return rows


def _fixture_table(document: dict[str, Any]) -> pa.Table:
    metadata = EvaluationMetadata.from_json(_dump(document["metadata"]))
    return evaluation_table(_rows(document["rows"]), metadata)


def test_evaluation_fixtures_agree_across_json_schema_and_typed_arrow() -> None:
    def schemas(package: str, paths: tuple[str, ...]) -> dict[str, Resource[Any]]:
        result = {}
        for path in paths:
            directory = files(package).joinpath("contracts", path)
            for entry in directory.iterdir():
                if entry.name.endswith(".schema.json"):
                    schema = json.loads(entry.read_text(encoding="utf-8"))
                    result[schema["$id"]] = Resource.from_contents(schema)
        return result

    resources = {
        **schemas("crc_framework", ("scientific", "numerical")),
        **schemas("crc_sdk", ("schemas",)),
    }
    schema = evaluation_json_schema()
    stored = files("crc_sdk").joinpath("contracts/schemas/evaluation-2.0.schema.json")
    assert json.loads(stored.read_text(encoding="utf-8")) == schema
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(
        schema,
        registry=Registry().with_resources(resources.items()),
        format_checker=FormatChecker(),
    )
    cases = _fixtures()
    for case in cases["accepted"]:
        document = case["document"]
        errors = list(validator.iter_errors(document))
        assert not errors, (case["name"], errors)
        table = _fixture_table(document)
        metadata = validate_evaluation(table)
        assert EvaluationMetadata.from_json(metadata.to_json()) == metadata
    for case in cases["rejected"]:
        document = case["document"]
        assert validator.is_valid(document) is not case["structural"], case["name"]
        with pytest.raises((ValueError, TypeError)):
            validate_evaluation(_fixture_table(document))


def test_actual_context_resolutions_are_bound_and_checkable_across_batches() -> None:
    initial = _metadata()
    original_case = _fixtures()["accepted"][1]["document"]
    actual = EvaluationMetadata.from_json(_dump(original_case["metadata"]))
    reference = actual.responses[1].model
    assert isinstance(reference, cf.ImpactReference)
    response = EvaluationResponse(
        "damage",
        actual.responses[1].variable_id,
        reference,
        cf.InundationContext(
            region=cf.FloodRegion.EUROPE,
            asset_type=Assets({"building": ["residential"]}).column("building"),
        ),
    )
    planned = replace(
        initial,
        include_exposure=False,
        responses=(response,),
        asset_fields=pa.schema(
            [
                pa.field("asset_id", pa.string(), nullable=False),
                pa.field("building", pa.string()),
            ]
        ),
        report=ExecutionReport(False, None),
    )
    original_rows = _rows(original_case["rows"][-2:])
    validator = EvaluationValidator(planned)
    try:
        validator.register_resolution(actual.resolutions[0])
        bad = replace(original_rows[0], asset={"building": "industrial"})
        with pytest.raises(ValueError, match="recorded response context"):
            validator.check(evaluation_table([bad], planned).to_batches()[0])
        for row in original_rows:
            good = replace(row, asset={"building": "residential"})
            validator.check(evaluation_table([good], planned).to_batches()[0])
        final = replace(planned, resolutions=actual.resolutions, report=initial.report)
        validator.finish(final)
    finally:
        validator.close()


def test_query_keeps_endpoints_and_validates_exact_annual_meaning() -> None:
    initial = _metadata()
    assert initial.query.probability(100, initial.spec.series[0]) == 0.99
    variable = initial.spec.variables[0]
    lower = SeriesSpec(variable, probability=Probability.annual(tail=cf.Tail.LOWER))
    assert initial.query.probability(100, lower) == 0.01
    assert initial.query.probability(1, lower) == 1
    with pytest.raises(CapabilityError, match="declared annual"):
        initial.query.probability(100, SeriesSpec(variable))
    dataset = replace(
        initial.inputs[0].metadata,
        spec=DataSpec([variable], series=[SeriesSpec(variable)]),
    )
    metadata = replace(
        initial,
        query=EvaluationQuery(QueryKind.PROBABILITY, (0.0, 1.0)),
        inputs=(EvaluationInput(None, None, dataset),),
    )
    rows = [
        EvaluationRow(
            asset_id="a",
            variable_id=variable.id,
            support_id="s",
            query_value=q,
            value=v,
        )
        for q, v in zip(metadata.query.values, (-math.inf, math.inf), strict=True)
    ]
    table = evaluation_table(rows, metadata)
    assert validate_evaluation(table) == metadata
    assert table["value"].to_pylist() == [-math.inf, math.inf]
    bad = evaluation_table([replace(rows[0], value=math.nan), rows[1]], metadata)
    with pytest.raises(ValueError, match="NaN"):
        validate_evaluation(bad)


def test_parquet_preserves_nested_asset_columns_and_unsigned_asset_ids(
    tmp_path: Path,
) -> None:
    initial = _metadata()
    identity = 2**64 - 1
    asset_fields = pa.schema(
        [
            pa.field("asset_id", pa.uint64(), nullable=False),
            pa.field("labels", pa.list_(pa.string()), nullable=False),
            pa.field("scaled", pa.decimal128(6, 2)),
            pa.field("binary", pa.binary(), metadata={b"meaning": b"data"}),
        ]
    )
    metadata = replace(initial, asset_fields=asset_fields, asset_ids=(identity,))
    values = {"labels": ["a", "b"], "scaled": None, "binary": b"bytes"}
    rows = [
        EvaluationRow(
            asset_id=identity,
            support_id="site",
            variable_id=initial.spec.variables[0].id,
            query_value=q,
            value=1.0,
            asset=values,
        )
        for q in initial.query.values
    ]
    table = evaluation_table(rows, metadata)
    path = tmp_path / "events.parquet"
    pq.write_table(table, path)
    reopened = pq.read_table(path)
    assert reopened["asset_id"].to_pylist() == [identity, identity]
    assert reopened["asset"].to_pylist() == [values, values]
    assert validate_evaluation(reopened) == metadata
    assert reopened.schema.field("query_value").metadata[b"crc.unit"] == b"year"
    bad_schema = reopened.schema.set(
        11, reopened.schema.field("value").with_metadata({b"crc.unit": b"mm"})
    )
    with pytest.raises(ValueError, match="canonical evaluation schema"):
        evaluation_metadata_from_schema(bad_schema)


def test_nearest_bound_and_duplicate_rows_across_batches(tmp_path: Path) -> None:
    initial = _metadata()
    metadata = replace(
        initial, spatial=SpatialMatch.nearest(max_distance=Distance.metres(10))
    )
    first = EvaluationRow(
        asset_id="a",
        variable_id=initial.spec.variables[0].id,
        support_id="s",
        distance_metres=4,
        query_value=100,
        value=1.6,
    )
    second = replace(first, query_value=10, value=0.8)
    validator = EvaluationValidator(metadata, scope_directory=tmp_path)
    try:
        with pytest.raises(ValueError, match="selected bound"):
            record = replace(first, distance_metres=15)
            validator.check(evaluation_table([record], metadata).to_batches()[0])
        validator.check(evaluation_table([first], metadata).to_batches()[0])
        with pytest.raises(ValueError, match="duplicate evaluation row"):
            validator.check(evaluation_table([first], metadata).to_batches()[0])
        validator.check(evaluation_table([second], metadata).to_batches()[0])
        validator.finish()
    finally:
        validator.close()
    assert not list(tmp_path.iterdir())


def test_unknown_query_and_old_header_are_rejected() -> None:
    info = _metadata()
    document = json.loads(info.to_json())
    document["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="schema 2.0"):
        EvaluationMetadata.from_json(_dump(document))
    row = EvaluationRow(
        asset_id="a",
        variable_id=info.spec.variables[0].id,
        query_value=25,
        support_id="s",
        value=1.0,
    )
    with pytest.raises(ValueError, match="unrequested query"):
        validate_evaluation(evaluation_table([row], info))
    with pytest.raises(ValueError, match="metadata is missing"):
        evaluation_metadata_from_schema(
            evaluation_arrow_schema(info, include_metadata=False)
        )
