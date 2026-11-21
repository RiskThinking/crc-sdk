"""JSON Schema for insured annual totals and optional aligned components."""

from __future__ import annotations

from .._loss_artifact import _STREAM_LAYOUT, _U64_MAX, ANNUAL_LOSS_VERSION
from .contract import _record

_DATASET = "https://climateriskcommons.org/schemas/dataset/2.0#/$defs/"


def annual_loss_json_schema() -> dict[str, object]:
    identifier = {"type": "string", "pattern": r"^\S(?:[\s\S]*\S)?$"}
    digest = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
    null = {"type": "null"}
    amount = {"type": "number", "minimum": 0}
    fraction = {"type": "number", "minimum": 0, "maximum": 1}
    positive = {"type": "integer", "minimum": 1}
    u64 = {"type": "integer", "minimum": 0, "maximum": _U64_MAX}

    def ref(name: str) -> dict[str, str]:
        return {"$ref": f"#/$defs/{name}"}

    def dataset(name: str) -> dict[str, str]:
        return {"$ref": _DATASET + name}

    def nullable(value: object) -> dict[str, object]:
        return {"anyOf": [value, null]}

    def record(**fields: object) -> dict[str, object]:
        return _record(fields, list(fields))

    definitions = {
        "spec": record(
            trials={**positive, "maximum": 2**63 - 1},
            seed=u64,
            peak_correlation=fraction,
            decay_length_km={"type": "number", "exclusiveMinimum": 0},
            damage_cv=amount,
            repair_tolerance=nullable(amount),
            calibration_source=nullable(identifier),
        ),
        "correlation_repair": record(
            minimum_eigenvalue={"type": "number"}, maximum_change=amount
        ),
        "selected_series": record(
            variable=dataset("variable"),
            scenario=nullable(dataset("scenario")),
            period=nullable(dataset("period")),
            probability={
                "allOf": [
                    dataset("probability"),
                    {"properties": {"kind": {"const": "annual"}}},
                ]
            },
        ),
        "tail_completion": record(
            name={"const": "finite_endpoints"}, lower=fraction, upper=fraction
        ),
        "resolved_impact": record(
            resolution_id=digest, impact_sha256=digest, asset_count=positive
        ),
        "input_resource": record(
            member_id=nullable(identifier),
            resource_id=nullable(identifier),
            source=nullable(dataset("source")),
            transformations={"type": "array", "items": dataset("transformation")},
        ),
        "evidence": record(
            damage_response=identifier,
            response_model_sha256=digest,
            vulnerability_sha256=digest,
            member_id=nullable(identifier),
            variable_id=identifier,
            scenario_id=nullable(identifier),
            period_id=nullable({"type": "string", "pattern": r"^period:[0-9a-f]{64}$"}),
            series=ref("selected_series"),
            probability_grid={
                "type": "array",
                "minItems": 2,
                "uniqueItems": True,
                "items": fraction,
            },
            tail_completion=nullable(ref("tail_completion")),
            matched_support_count=positive,
            matched_supports_sha256=digest,
            resolved_impacts={
                "type": "array",
                "minItems": 1,
                "items": ref("resolved_impact"),
            },
            input_resources={
                "type": "array",
                "minItems": 1,
                "items": ref("input_resource"),
            },
        ),
        "metadata": {
            **record(
                version={"const": ANNUAL_LOSS_VERSION},
                loss_basis={"const": "insured"},
                run_id=digest,
                peril=identifier,
                currency=identifier,
                valuation_basis=identifier,
                portfolio_sha256=digest,
                input_sha256=digest,
                evidence=ref("evidence"),
                report={
                    "allOf": [
                        dataset("report"),
                        {"properties": {"valid": {"const": True}}},
                    ]
                },
                spec=ref("spec"),
                secondary_stream=u64,
                stream_layout={"const": _STREAM_LAYOUT},
                rng_algorithm=identifier,
                framework_version=identifier,
                sdk_version=identifier,
                total_insured_value=amount,
                correlation_repair=nullable(ref("correlation_repair")),
                component_count={"type": "integer", "minimum": 0},
                totals_sha256=digest,
                components_sha256=nullable(digest),
            ),
            "if": {"properties": {"component_count": {"const": 0}}},
            "then": {"properties": {"components_sha256": null}},
            "else": {"properties": {"components_sha256": digest}},
        },
        "component_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": identifier,
        },
        "component_losses": {"type": "array", "minItems": 1, "items": amount},
        "total_row": record(
            trial_id={"type": "integer", "minimum": 0}, insured_loss=amount
        ),
        "first_component_row": record(
            trial_id={"const": 0},
            insured_loss=amount,
            component_ids=ref("component_ids"),
            component_losses=ref("component_losses"),
        ),
        "subsequent_component_row": record(
            trial_id=positive,
            insured_loss=amount,
            component_ids=null,
            component_losses=ref("component_losses"),
        ),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://climateriskcommons.org/schemas/annual-loss/{ANNUAL_LOSS_VERSION}",
        "title": "Insured annual loss sample",
        "$comment": (
            "The run ID and peril in metadata scope all trial rows. Trial IDs are "
            "zero-based and consecutive. Component IDs are stored on trial 0, and "
            "every trial's loss vector follows that ID order. The typed reader "
            "checks footer and payload digests, and component and trial alignment."
        ),
        **record(metadata=ref("metadata"), rows={"type": "array", "minItems": 1}),
        "$defs": definitions,
        "if": {
            "properties": {
                "metadata": {"properties": {"component_count": {"const": 0}}}
            }
        },
        "then": {"properties": {"rows": {"items": ref("total_row")}}},
        "else": {
            "properties": {
                "rows": {
                    "prefixItems": [ref("first_component_row")],
                    "items": ref("subsequent_component_row"),
                }
            }
        },
    }
