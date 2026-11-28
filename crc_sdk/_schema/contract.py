"""JSON Schema for the canonical dataset envelope."""

from __future__ import annotations

from .._aggregation import temporal_reduction_schema
from .._declarations import _schema_definitions

NUMERICAL_SCHEMA = "https://climateriskcommons.org/schemas/numerical/1.0"
FITTING_SCHEMA = "https://climateriskcommons.org/schemas/fitting/1.1"


def _record(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def dataset_json_schema() -> dict[str, object]:
    identifier: dict[str, object] = {
        "type": "string",
        "minLength": 1,
        "pattern": r"^\S(?:[\s\S]*\S)?$",
    }
    text: dict[str, object] = {"type": "string", "minLength": 1}
    digest: dict[str, object] = {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}
    timestamp: dict[str, object] = {"type": "string", "format": "date-time"}
    nullable_id: dict[str, object] = {"anyOf": [identifier, {"type": "null"}]}
    definitions = _schema_definitions()
    definitions.update(
        {
            "resource": _record(
                {
                    "uri": identifier,
                    "sha256": digest,
                    "license": text,
                    "attribution": text,
                    "retrieved_at": timestamp,
                },
                ["uri", "sha256"],
            ),
            "release": _record(
                {"id": identifier, "sha256": digest, "retrieved_at": timestamp},
                ["id", "sha256"],
            ),
            "source": _record(
                {
                    "provider": identifier,
                    "product": identifier,
                    "member": identifier,
                    "model": identifier,
                    "release": {"$ref": "#/$defs/release"},
                    "resources": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/resource"},
                    },
                },
                ["provider", "product", "resources"],
            ),
            "diagnostic": _record(
                {
                    "code": identifier,
                    "severity": {"enum": ["error", "warning", "info"]},
                    "stage": {
                        "enum": [
                            "acquisition",
                            "decoding",
                            "selection",
                            "reduction",
                            "fitting",
                            "joining",
                            "evaluation",
                            "validation",
                            "writing",
                        ]
                    },
                    "message": text,
                    "source": identifier,
                    "support_id": identifier,
                    "query_value": {"type": "number"},
                    "remedy": text,
                },
                ["code", "severity", "stage", "message"],
            ),
            "counts": _record(
                {
                    name: {"type": "integer", "minimum": 0}
                    for name in (
                        "requested",
                        "produced",
                        "skipped",
                        "no_data",
                        "matched_assets",
                        "unmatched_assets",
                        "usable_years",
                        "incomplete_years",
                    )
                },
                [
                    "requested",
                    "produced",
                    "skipped",
                    "no_data",
                    "matched_assets",
                    "unmatched_assets",
                    "usable_years",
                    "incomplete_years",
                ],
            ),
            "report": _record(
                {
                    "complete": {"type": "boolean"},
                    "valid": {"type": ["boolean", "null"]},
                    "counts": {"$ref": "#/$defs/counts"},
                    "diagnostics": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/diagnostic"},
                    },
                    "source_identities": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/source"},
                    },
                },
                ["complete", "valid", "counts", "diagnostics", "source_identities"],
            ),
            "spatial": _record(
                {
                    "supports": {
                        "type": "array",
                        "items": {"enum": ["point", "polygon", "cell"]},
                        "uniqueItems": True,
                        "minItems": 1,
                    },
                    "index_resolution": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 15,
                    },
                },
                ["supports"],
            ),
            "transformation": {
                "oneOf": [
                    _record(
                        {
                            "kind": {"const": "unit_conversion"},
                            "variable_id": identifier,
                            "source_unit": identifier,
                            "target_unit": identifier,
                            "scale": {"type": "number", "not": {"const": 0}},
                            "offset": {"type": "number"},
                        },
                        [
                            "kind",
                            "variable_id",
                            "source_unit",
                            "target_unit",
                            "scale",
                            "offset",
                        ],
                    ),
                    _record(
                        {
                            "kind": {"const": "rename"},
                            "dimension": {"enum": ["variable", "scenario"]},
                            "mapping": {
                                "type": "object",
                                "propertyNames": identifier,
                                "additionalProperties": identifier,
                            },
                        },
                        ["kind", "dimension", "mapping"],
                    ),
                    _record(
                        {
                            "kind": {"const": "observation_time_selection"},
                            "time": {
                                "allOf": [
                                    {"$ref": "#/$defs/period"},
                                    {"properties": {"kind": {"const": "window"}}},
                                ]
                            },
                        },
                        ["kind", "time"],
                    ),
                ]
            },
            "failure": {
                "oneOf": [
                    _record(
                        {
                            "code": identifier,
                            "message": text,
                            "record": {
                                "allOf": [
                                    {"$ref": FITTING_SCHEMA},
                                    {"properties": {"converged": {"const": False}}},
                                ]
                            },
                        },
                        ["code", "message", "record"],
                    ),
                    _record(
                        {
                            "kind": {"const": "source_unavailable"},
                            "code": identifier,
                            "message": text,
                        },
                        ["kind", "code", "message"],
                    ),
                ]
            },
            "metadata": _record(
                {
                    "schema_version": {"const": "2.0"},
                    "spec": {"$ref": "#/$defs/data_spec"},
                    "spatial": {"$ref": "#/$defs/spatial"},
                    "source": {"$ref": "#/$defs/source"},
                    "report": {"$ref": "#/$defs/report"},
                    "provenance": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/transformation"},
                    },
                },
                ["schema_version", "spec", "spatial", "report", "provenance"],
            ),
        }
    )
    diagnostic = definitions["diagnostic"]
    assert isinstance(diagnostic, dict)
    diagnostic_properties = diagnostic["properties"]
    assert isinstance(diagnostic_properties, dict)
    fitting_diagnostic = _record(
        {
            **diagnostic_properties,
            "stage": {"const": "fitting"},
            "record": {"$ref": FITTING_SCHEMA},
        },
        ["code", "severity", "stage", "message", "record"],
    )
    station_year_properties = dict(diagnostic_properties)
    del station_year_properties["support_id"]
    unavailable_station_year = _record(
        {
            **station_year_properties,
            "code": {"const": "station_year_unavailable"},
            "severity": {"const": "warning"},
            "stage": {"enum": ["acquisition", "decoding"]},
            "station_id": identifier,
            "year": {"type": "integer", "minimum": 1, "maximum": 9999},
            "minimum_fraction": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        },
        [
            "code",
            "severity",
            "stage",
            "message",
            "station_id",
            "year",
            "minimum_fraction",
        ],
    )
    definitions["diagnostic"] = {
        "oneOf": [diagnostic, fitting_diagnostic, unavailable_station_year]
    }
    transformation = definitions["transformation"]
    assert isinstance(transformation, dict)
    variants = transformation["oneOf"]
    assert isinstance(variants, list)
    variants.append(temporal_reduction_schema())
    row = _record(
        {
            "variable_id": identifier,
            "scenario_id": nullable_id,
            "period_id": nullable_id,
            "support_id": identifier,
            "cell_id": {"type": "integer", "minimum": 0, "maximum": 2**64 - 1},
            "geometry": {
                "type": "string",
                "minLength": 42,
                "pattern": "^(?:[0-9a-f]{2})+$",
            },
            "curve": {"$ref": NUMERICAL_SCHEMA},
            "fit": {
                "allOf": [
                    {"$ref": FITTING_SCHEMA},
                    {"properties": {"converged": {"const": True}}},
                ]
            },
            "failure": {"$ref": "#/$defs/failure"},
        },
        ["variable_id", "scenario_id", "period_id", "support_id", "cell_id"],
    )
    row["allOf"] = [
        {
            "oneOf": [
                {"required": ["curve"], "not": {"required": ["failure"]}},
                {
                    "required": ["failure"],
                    "not": {"anyOf": [{"required": ["curve"]}, {"required": ["fit"]}]},
                },
            ]
        },
        {
            "if": {"required": ["fit"]},
            "then": {
                "properties": {
                    "curve": {"properties": {"object": {"const": "distribution"}}}
                }
            },
        },
    ]
    definitions["row"] = row
    report = definitions["report"]
    assert isinstance(report, dict)
    report["allOf"] = [
        {
            "if": {"properties": {"valid": {"const": True}}},
            "then": {
                "properties": {
                    "complete": {"const": True},
                    "diagnostics": {
                        "not": {
                            "contains": {"properties": {"severity": {"const": "error"}}}
                        }
                    },
                }
            },
        }
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/dataset/2.0",
        "title": "Canonical scientific dataset",
        "type": "object",
        "additionalProperties": False,
        "required": ["metadata", "rows"],
        "properties": {
            "metadata": {"$ref": "#/$defs/metadata"},
            "rows": {"type": "array", "items": {"$ref": "#/$defs/row"}},
        },
        "$defs": definitions,
    }
