"""JSON declaration of the canonical evaluation envelope and logical rows."""

from __future__ import annotations

from .contract import FITTING_SCHEMA, _record

_DATASET = "https://climateriskcommons.org/schemas/dataset/2.0#/$defs/"
_SCIENCE = "https://climateriskcommons.org/schemas/"


def evaluation_json_schema() -> dict[str, object]:
    """Describe evaluation 2.0 and the references shared with its Arrow form."""
    identifier = {"type": "string", "pattern": r"^\S(?:[\s\S]*\S)?$"}
    null = {"type": "null"}
    nullable_id = {"anyOf": [identifier, null]}
    digest = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
    asset_id = {
        "anyOf": [
            identifier,
            {"type": "integer", "minimum": -(2**63), "maximum": 2**64 - 1},
        ]
    }

    def query(kind: str, coordinates: dict[str, object]) -> dict[str, object]:
        return _record(
            {
                "kind": {"const": kind},
                "values": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": coordinates,
                },
            },
            ["kind", "values"],
        )

    def science_ref(name: str) -> dict[str, object]:
        return {"$ref": _SCIENCE + name + ".schema.json"}

    def response_context(value: object) -> dict[str, object]:
        return _record(
            {
                "kind": {"const": "response_context"},
                "values": {
                    "type": "object",
                    "minProperties": 1,
                    "propertyNames": identifier,
                    "additionalProperties": value,
                },
            },
            ["kind", "values"],
        )

    definitions: dict[str, object] = {
        "query": {
            "oneOf": [
                query("probability", {"type": "number", "minimum": 0, "maximum": 1}),
                query("return_period", {"type": "number", "minimum": 1}),
            ]
        },
        "input": _record(
            {
                "member_id": nullable_id,
                "resource_id": nullable_id,
                "metadata": {"$ref": _DATASET + "metadata"},
            },
            ["member_id", "resource_id", "metadata"],
        ),
        "case_context": response_context(identifier),
        "response_context": response_context(
            {"anyOf": [identifier, _record({"column": identifier}, ["column"])]}
        ),
        "contextual_impact": _record(
            {
                "kind": {"const": "contextual"},
                "cases": {
                    "type": "array",
                    "minItems": 1,
                    "items": _record(
                        {
                            "context": {"$ref": "#/$defs/case_context"},
                            "impact": science_ref("impact-2.0"),
                        },
                        ["context", "impact"],
                    ),
                },
            },
            ["kind", "cases"],
        ),
        "response": {
            **_record(
                {
                    "name": identifier,
                    "variable_id": identifier,
                    "model": {
                        "oneOf": [
                            science_ref("impact-2.0"),
                            science_ref("impact-reference-2.0"),
                            {"$ref": "#/$defs/contextual_impact"},
                        ]
                    },
                    "context": {"type": "object"},
                },
                ["name", "variable_id", "model"],
            ),
            "if": {"properties": {"model": {"$ref": "#/$defs/contextual_impact"}}},
            "then": {
                "required": ["context"],
                "properties": {"context": {"$ref": "#/$defs/response_context"}},
            },
            "else": {"properties": {"context": science_ref("impact-context-1.0")}},
        },
        "resolution": _record(
            {
                "id": digest,
                "response_id": identifier,
                "impact": science_ref("impact-2.0"),
                "context": {"$ref": "#/$defs/case_context"},
                "asset_ids": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": asset_id,
                },
            },
            ["id", "response_id", "impact", "asset_ids"],
        ),
        "spatial": {
            "oneOf": [
                _record(
                    {
                        "kind": {"const": "covering"},
                        "preferred_supports": {
                            "type": "array",
                            "items": identifier,
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                    },
                    ["kind"],
                ),
                _record(
                    {
                        "kind": {"const": "nearest"},
                        "max_distance_metres": {
                            "type": "number",
                            "minimum": 0,
                        },
                    },
                    ["kind", "max_distance_metres"],
                ),
            ]
        },
        "failure": _record(
            {
                "code": identifier,
                "message": {"type": "string", "minLength": 1},
                "record": {
                    "allOf": [
                        {"$ref": FITTING_SCHEMA},
                        {"properties": {"converged": {"const": False}}},
                    ]
                },
            },
            ["code", "message"],
        ),
    }
    metadata = _record(
        {
            "schema_version": {"const": "2.0"},
            "query": {"$ref": "#/$defs/query"},
            "inputs": {"type": "array", "items": {"$ref": "#/$defs/input"}},
            "responses": {"type": "array", "items": {"$ref": "#/$defs/response"}},
            "resolutions": {"type": "array", "items": {"$ref": "#/$defs/resolution"}},
            "spatial": {"$ref": "#/$defs/spatial"},
            "include_exposure": {"type": "boolean"},
            "report": {"$ref": _DATASET + "report"},
            "source": {"$ref": _DATASET + "source"},
            "asset_fields": {
                "type": "string",
                "minLength": 1,
                "contentEncoding": "base64",
                "contentMediaType": "application/vnd.apache.arrow.stream",
            },
            "asset_ids": {"type": "array", "uniqueItems": True, "items": asset_id},
        },
        [
            "schema_version",
            "query",
            "inputs",
            "responses",
            "resolutions",
            "spatial",
            "include_exposure",
            "report",
        ],
    )
    metadata["allOf"] = [
        {
            "if": {
                "properties": {"report": {"properties": {"complete": {"const": True}}}}
            },
            "then": {"required": ["asset_ids"]},
        },
        {
            "if": {
                "properties": {"asset_ids": {"minItems": 1}},
                "required": ["asset_ids"],
            },
            "then": {"required": ["asset_fields"]},
        },
    ]
    definitions["metadata"] = metadata
    row_properties: dict[str, object] = {
        "asset_id": asset_id,
        "member_id": nullable_id,
        "resource_id": nullable_id,
        "variable_id": identifier,
        "scenario_id": nullable_id,
        "period_id": nullable_id,
        "support_id": nullable_id,
        "distance_metres": {"anyOf": [null, {"type": "number", "minimum": 0}]},
        "response_id": nullable_id,
        "resolution_id": {"anyOf": [null, digest]},
        "query_value": {"type": "number"},
        "value": {
            "anyOf": [null, {"type": "number"}, {"enum": ["-Infinity", "Infinity"]}]
        },
        "failure": {"anyOf": [null, {"$ref": "#/$defs/failure"}]},
    }
    row = _record(
        {**row_properties, "asset": {"type": "object"}},
        list(row_properties),
    )
    row["allOf"] = [
        {
            "if": {"properties": {"failure": null}},
            "then": {"properties": {"value": {"not": null}}},
            "else": {"properties": {"value": null}},
        },
        {
            "if": {"properties": {"support_id": null}},
            "then": {
                "properties": {
                    "distance_metres": null,
                    "resource_id": null,
                    "resolution_id": null,
                    "failure": {"not": null},
                }
            },
        },
        {
            "if": {"properties": {"response_id": null}},
            "then": {"properties": {"resolution_id": null}},
            "else": {"properties": {"value": {"type": ["number", "null"]}}},
        },
    ]
    definitions["row"] = row
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/evaluation/2.0",
        "title": "Canonical scientific evaluation",
        **_record(
            {
                "metadata": {"$ref": "#/$defs/metadata"},
                "rows": {"type": "array", "items": {"$ref": "#/$defs/row"}},
            },
            ["metadata", "rows"],
        ),
        "$defs": definitions,
    }
