"""Generate evaluation 2.0 JSON Schema and its conformance cases."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import crc_framework as cf
import pyarrow as pa

from crc_sdk._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionReport,
    SpatialContract,
    SupportKind,
)
from crc_sdk._declarations import DataSpec, Probability, SeriesSpec, Variable
from crc_sdk._evaluation_schema import (
    EvaluationInput,
    EvaluationMetadata,
    EvaluationQuery,
    EvaluationResolution,
    EvaluationResponse,
    QueryKind,
)
from crc_sdk._schema.evaluation_contract import evaluation_json_schema

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "crc_sdk/contracts"


def _row(**changes: object) -> dict[str, object]:
    return {
        "asset_id": "a",
        "member_id": None,
        "resource_id": None,
        "variable_id": "project:depth",
        "scenario_id": None,
        "period_id": None,
        "support_id": "site",
        "distance_metres": None,
        "response_id": None,
        "resolution_id": None,
        "query_value": 100.0,
        "value": 1.6,
        "failure": None,
        **changes,
    }


def fixture_cases() -> dict[str, object]:
    depth = cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
    variable = Variable("project:depth", measurement=depth)
    dataset = DatasetMetadata(
        spec=DataSpec(
            [variable],
            series=[
                SeriesSpec(variable, probability=Probability.annual(tail=cf.Tail.UPPER))
            ],
        ),
        spatial=SpatialContract(supports=(SupportKind.POINT,), index_resolution=9),
        report=ExecutionReport(
            True, True, counts=ExecutionCounts(requested=1, produced=1)
        ),
    )
    artifact = EvaluationMetadata(
        query=EvaluationQuery(QueryKind.RETURN_PERIOD, (100.0, 10.0)),
        inputs=(EvaluationInput(None, None, dataset),),
        asset_fields=pa.schema([pa.field("asset_id", pa.string(), nullable=False)]),
        asset_ids=("a",),
        report=ExecutionReport(
            True,
            True,
            counts=ExecutionCounts(requested=2, produced=2, matched_assets=1),
        ),
    )
    direct: dict[str, Any] = {
        "metadata": json.loads(artifact.to_json()),
        "rows": [_row(), _row(query_value=10.0, value=0.8)],
    }

    response = EvaluationResponse(
        "downtime",
        variable.id,
        cf.Impact.linear(
            slope=2,
            input=depth,
            output=cf.TimeDuration(unit=cf.TimeUnit.DAY),
        ),
    )
    builtin = cf.Impact.builtin(cf.BuiltinImpact.INUNDATION)
    context = cf.InundationContext(
        region=cf.FloodRegion.EUROPE, asset_type=cf.AssetType.RESIDENTIAL
    )
    impact = builtin.resolve(context)
    composed_metadata = replace(
        artifact,
        responses=(
            response,
            EvaluationResponse("damage", variable.id, builtin, context),
        ),
        resolutions=(EvaluationResolution("damage", impact),),
        report=ExecutionReport(
            True,
            True,
            counts=ExecutionCounts(requested=6, produced=6, matched_assets=1),
        ),
    )
    composed: dict[str, Any] = {
        "metadata": json.loads(composed_metadata.to_json()),
        "rows": [
            *direct["rows"],
            _row(response_id="downtime", value=3.2),
            _row(response_id="downtime", query_value=10.0, value=1.6),
            _row(
                response_id="damage",
                resolution_id=impact.sha256,
                value=impact.evaluate(1.6),
            ),
            _row(
                response_id="damage",
                resolution_id=impact.sha256,
                query_value=10.0,
                value=impact.evaluate(0.8),
            ),
        ],
    }

    unmatched = deepcopy(direct)
    unmatched["metadata"]["report"]["counts"].update(
        matched_assets=0, unmatched_assets=1, no_data=2
    )
    for row in unmatched["rows"]:
        row.update(
            support_id=None,
            value=None,
            failure={"code": "no_covering_support", "message": "No covering support."},
        )
    empty = {
        "metadata": json.loads(
            EvaluationMetadata(
                query=artifact.query,
                asset_ids=(),
                include_exposure=False,
                report=ExecutionReport(True, True),
            ).to_json()
        ),
        "rows": [],
    }
    accepted = [
        {"name": "direct-ordered-return-periods", "document": direct},
        {"name": "named-and-contextual-responses", "document": composed},
        {"name": "unmatched-assets-retained", "document": unmatched},
        {"name": "explicit-empty", "document": empty},
    ]

    rejected = []

    def reject(
        name: str, document: dict[str, Any], *, structural: bool = False
    ) -> None:
        rejected.append({"name": name, "document": document, "structural": structural})

    old = deepcopy(direct)
    old["metadata"]["schema_version"] = "1.0"
    reject("old-version", old, structural=True)
    dead = deepcopy(direct)
    dead["rows"][0]["probability"] = 0.99
    reject("redundant-probability-field", dead, structural=True)
    duplicate_axis = deepcopy(direct)
    duplicate_axis["metadata"]["query"]["values"] = [100.0, 100.0]
    reject("duplicate-query-coordinates", duplicate_axis, structural=True)
    failure_with_value = deepcopy(direct)
    failure_with_value["rows"][0]["failure"] = {"code": "x", "message": "Contradiction"}
    reject("failure-with-value", failure_with_value, structural=True)
    missing = deepcopy(direct)
    missing["rows"].pop()
    reject("missing-query-row", missing)
    duplicate_row = deepcopy(direct)
    duplicate_row["rows"].insert(1, deepcopy(duplicate_row["rows"][0]))
    reject("duplicate-scientific-row", duplicate_row)
    wrong_order = deepcopy(direct)
    wrong_order["rows"].reverse()
    reject("out-of-order-query", wrong_order)
    wrong_member = deepcopy(direct)
    wrong_member["rows"][0]["member_id"] = "undeclared"
    reject("wrong-scientific-member", wrong_member)
    wrong_asset = deepcopy(direct)
    wrong_asset["rows"][0]["asset_id"] = "unknown"
    reject("wrong-asset-reference", wrong_asset)
    wrong_resolution = deepcopy(composed)
    wrong_resolution["rows"][-1]["resolution_id"] = "sha256:" + "f" * 64
    reject("wrong-context-resolution", wrong_resolution)
    forged_resolution = deepcopy(composed)
    forged_resolution["metadata"]["resolutions"][0]["id"] = "sha256:" + "f" * 64
    reject("forged-resolution-hash", forged_resolution)
    wrong_proof = deepcopy(composed)
    wrong_proof["metadata"]["responses"][-1]["context"]["region"] = "asia"
    reject("wrong-resolved-context", wrong_proof)
    wrong_science = deepcopy(composed)
    wrong_science["metadata"]["inputs"][0]["metadata"]["spec"]["variables"][0][
        "measurement"
    ]["unit"] = "cm"
    reject("response-calibration-mismatch", wrong_science)
    wrong_counts = deepcopy(unmatched)
    wrong_counts["metadata"]["report"]["counts"]["matched_assets"] = 1
    reject("unmatched-count-mismatch", wrong_counts)
    return {"schema_version": "2.0", "accepted": accepted, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = {
        OUTPUT / "schemas/evaluation-2.0.schema.json": evaluation_json_schema(),
        OUTPUT / "fixtures/evaluation-2.0.json": fixture_cases(),
    }
    for path, value in generated.items():
        content = json.dumps(value, indent=2, allow_nan=False) + "\n"
        if args.check:
            if path.read_text() != content:
                raise SystemExit(f"stale generated evaluation contract: {path}")
        else:
            path.write_text(content)
    print(
        "Evaluation schema and fixtures verified"
        if args.check
        else "Evaluation schema and fixtures generated"
    )


if __name__ == "__main__":
    main()
