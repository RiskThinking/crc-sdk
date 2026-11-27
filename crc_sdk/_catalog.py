"""Optional immutable exact catalogs; user-owned identifiers need no registration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import cast

import crc_framework

from ._declarations import (
    Scenario,
    Variable,
    _descriptors,
    _fields,
    _identifier,
    _json,
    _load,
    _merge_descriptor,
    _scenario,
    _variable,
)

# These are explicit local conveniences. The provider's external release
# catalog may expose additional exact IDs without inventing Python enum members.
_CRC_VARIABLES = (
    ("crc:txx", "Annual maximum daily maximum temperature"),
    ("crc:txn", "Annual minimum daily maximum temperature"),
    ("crc:tnx", "Annual maximum daily minimum temperature"),
    ("crc:tnn", "Annual minimum daily minimum temperature"),
    ("crc:rx1day", "Annual maximum daily precipitation"),
    ("crc:rx5day", "Annual maximum five-day precipitation"),
    ("crc:sdii", "Annual mean wet-day precipitation"),
    ("crc:cdd", "Annual maximum consecutive dry days"),
    ("crc:cwd", "Annual maximum consecutive wet days"),
    ("crc:fd", "Annual frost-day count"),
    ("crc:su", "Annual summer-day count"),
    ("crc:tr", "Annual tropical-night count"),
    ("crc:dtr", "Annual mean daily temperature range"),
    ("crc:prcptot", "Annual total wet-day precipitation"),
    ("crc:r10mm", "Annual precipitation days at least ten millimetres"),
    ("crc:r20mm", "Annual precipitation days at least twenty millimetres"),
    ("crc:r95p", "Annual precipitation above baseline wet-day ninety-fifth percentile"),
)


class Catalog:
    """Exact immutable scientific vocabulary with explicit optional fit presets.

    Lookup never folds case, trims IDs, consults global state, or selects a
    preset or impact model implicitly. Composition rejects conflicting science
    for shared IDs; a display-label difference does not create such a conflict.
    """

    _namespace: str
    _revision: str
    _variables: Mapping[str, Variable]
    _scenarios: Mapping[str, Scenario]
    _presets: Mapping[str, crc_framework.FitSpec]
    _json: str
    _sha256: str

    __slots__ = (
        "_namespace",
        "_revision",
        "_variables",
        "_scenarios",
        "_presets",
        "_json",
        "_sha256",
    )

    def __init__(
        self,
        *,
        namespace: str,
        revision: str,
        variables: Sequence[Variable] = (),
        scenarios: Sequence[Scenario] = (),
        presets: Mapping[str, crc_framework.FitSpec] | None = None,
    ) -> None:
        _identifier(namespace, "namespace")
        _identifier(revision, "revision")
        variable_map = _descriptors(variables, Variable)
        scenario_map = _descriptors(scenarios, Scenario)
        if presets is not None and not isinstance(presets, Mapping):
            raise TypeError("presets must be a mapping or None")
        preset_map = {} if presets is None else dict(presets)
        for id, specification in preset_map.items():
            _identifier(id, "preset id")
            if not isinstance(specification, crc_framework.FitSpec):
                raise TypeError(
                    "catalog presets must be framework fitting specifications"
                )
        object.__setattr__(self, "_namespace", namespace)
        object.__setattr__(self, "_revision", revision)
        object.__setattr__(self, "_variables", MappingProxyType(variable_map))
        object.__setattr__(self, "_scenarios", MappingProxyType(scenario_map))
        object.__setattr__(self, "_presets", MappingProxyType(preset_map))
        document = {
            "schema_version": "1.0",
            "namespace": namespace,
            "revision": revision,
            "variables": [variable_map[id]._document() for id in sorted(variable_map)],
            "scenarios": [scenario_map[id]._document() for id in sorted(scenario_map)],
            "presets": {
                id: _load(preset_map[id].to_json()) for id in sorted(preset_map)
            },
        }
        canonical = _json(document)
        object.__setattr__(self, "_json", canonical)
        object.__setattr__(
            self, "_sha256", "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("catalogs are immutable")

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def variables(self) -> tuple[Variable, ...]:
        return tuple(self._variables.values())

    @property
    def scenarios(self) -> tuple[Scenario, ...]:
        return tuple(self._scenarios.values())

    @property
    def presets(self) -> Mapping[str, crc_framework.FitSpec]:
        return self._presets

    @property
    def sha256(self) -> str:
        return self._sha256

    @classmethod
    def crc(cls) -> Catalog:
        return cls(
            namespace="crc",
            revision="1.0",
            variables=tuple(Variable(id, label=label) for id, label in _CRC_VARIABLES),
            scenarios=(Scenario("ssp585", label="SSP5-8.5"),),
        )

    def variable(self, id: str) -> Variable:
        return self._variables[_identifier(id)]

    def scenario(self, id: str) -> Scenario:
        return self._scenarios[_identifier(id)]

    def preset(self, id: str) -> crc_framework.FitSpec:
        return self._presets[_identifier(id)]

    def merged(self, other: Catalog) -> Catalog:
        if not isinstance(other, Catalog):
            raise TypeError("other must be Catalog")
        variables = dict(self._variables)
        scenarios = dict(self._scenarios)
        presets = dict(self._presets)
        for variable in other.variables:
            _merge_descriptor(variable, variables)
        for scenario in other.scenarios:
            _merge_descriptor(scenario, scenarios)
        for id, specification in other._presets.items():
            previous = presets.get(id)
            if previous is not None and _load(previous.to_json()) != _load(
                specification.to_json()
            ):
                raise ValueError(f"conflicting fitting preset {id!r}")
            presets.setdefault(id, specification)
        namespace = "+".join(sorted({self.namespace, other.namespace}))
        revision = "+".join(sorted({self.revision, other.revision}))
        return Catalog(
            namespace=namespace,
            revision=revision,
            variables=tuple(variables.values()),
            scenarios=tuple(scenarios.values()),
            presets=presets,
        )

    def to_json(self) -> str:
        return self._json

    @classmethod
    def from_json(cls, text: str) -> Catalog:
        value = _fields(
            _load(text),
            {
                "schema_version",
                "namespace",
                "revision",
                "variables",
                "scenarios",
                "presets",
            },
        )
        if value["schema_version"] != "1.0":
            raise ValueError("expected taxonomy schema 1.0")
        if (
            not isinstance(value["variables"], list)
            or not isinstance(value["scenarios"], list)
            or not isinstance(value["presets"], dict)
        ):
            raise TypeError(
                "catalog variables/scenarios must be arrays; presets must be an object"
            )
        return cls(
            namespace=cast(str, value["namespace"]),
            revision=cast(str, value["revision"]),
            variables=tuple(_variable(v) for v in value["variables"]),
            scenarios=tuple(_scenario(v) for v in value["scenarios"]),
            presets={
                id: crc_framework.FitSpec.from_json(_json(specification))
                for id, specification in value["presets"].items()
            },
        )


def _schema_document() -> dict[str, object]:
    from ._declarations import _schema_definitions

    definitions = _schema_definitions()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/taxonomy/1.0",
        "type": "object",
        "required": [
            "schema_version",
            "namespace",
            "revision",
            "variables",
            "scenarios",
            "presets",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": "1.0"},
            "namespace": {
                "type": "string",
                "minLength": 1,
                "pattern": r"^\S(?:[\s\S]*\S)?$",
            },
            "revision": {
                "type": "string",
                "minLength": 1,
                "pattern": r"^\S(?:[\s\S]*\S)?$",
            },
            "variables": {
                "type": "array",
                "items": {"$ref": "#/$defs/variable"},
                "uniqueItems": True,
            },
            "scenarios": {
                "type": "array",
                "items": {"$ref": "#/$defs/scenario"},
                "uniqueItems": True,
            },
            "presets": {
                "type": "object",
                "propertyNames": {"minLength": 1, "pattern": r"^\S(?:[\s\S]*\S)?$"},
                "additionalProperties": {
                    "$ref": "https://climateriskcommons.org/schemas/fitting/1.1#/$defs/specification"
                },
            },
        },
        "$defs": definitions,
    }
