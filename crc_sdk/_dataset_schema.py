"""Immutable artifact metadata for canonical scientific datasets."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import cast

import crc_framework
from crc_framework import FitError, FitRecord
from rfc3339_validator import validate_rfc3339  # type: ignore[import-untyped]

from ._declarations import DataSpec, Period, PeriodKind, _data_spec, _period

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(
            f"{name} must be a nonempty exact string without boundary whitespace"
        )
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _object(
    value: object, required: set[str], optional: set[str] | frozenset[str] = frozenset()
) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("expected an object")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        raise ValueError(
            f"invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _wire_integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    return _integer(value, name, minimum)


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("sha256 must be a lowercase sha256: digest")
    return value


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not validate_rfc3339(value.upper()):
            raise ValueError("retrieved_at must be an RFC3339 date-time")
        value = datetime.fromisoformat(value.upper().replace("Z", "+00:00"))
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("retrieved_at must be a timezone-aware timestamp")
    return value


def _dump(document: object) -> str:
    return json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _load(text: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate field: {key}")
            result[key] = value
        return result

    def invalid(token: str) -> None:
        raise ValueError(f"non-finite JSON number: {token}")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class ResourceIdentity:
    uri: str
    sha256: str
    license: str | None = None
    attribution: str | None = None
    retrieved_at: datetime | None = None

    def __post_init__(self) -> None:
        _identifier(self.uri, "uri")
        _digest(self.sha256)
        for name in ("license", "attribution"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        if self.retrieved_at is not None and not isinstance(
            self.retrieved_at, datetime
        ):
            raise TypeError("retrieved_at must be datetime")
        _timestamp(self.retrieved_at)

    def _document(self) -> dict[str, object]:
        value: dict[str, object] = {"uri": self.uri, "sha256": self.sha256}
        for name in ("license", "attribution"):
            if getattr(self, name) is not None:
                value[name] = getattr(self, name)
        if self.retrieved_at is not None:
            value["retrieved_at"] = self.retrieved_at.astimezone(UTC).isoformat()
        return value


@dataclass(frozen=True, slots=True)
class ReleaseReference:
    id: str
    sha256: str
    retrieved_at: datetime | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "release.id")
        _digest(self.sha256)
        if self.retrieved_at is not None and not isinstance(
            self.retrieved_at, datetime
        ):
            raise TypeError("retrieved_at must be datetime")
        _timestamp(self.retrieved_at)

    def _document(self) -> dict[str, object]:
        value: dict[str, object] = {"id": self.id, "sha256": self.sha256}
        if self.retrieved_at is not None:
            value["retrieved_at"] = self.retrieved_at.astimezone(UTC).isoformat()
        return value


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    provider: str
    product: str
    release: ReleaseReference | None = None
    resources: tuple[ResourceIdentity, ...] = ()
    member: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.provider, "provider")
        _identifier(self.product, "product")
        if self.member is not None:
            _identifier(self.member, "member")
        if self.release is not None and not isinstance(self.release, ReleaseReference):
            raise TypeError("release must be ReleaseReference")
        if not isinstance(self.resources, tuple) or not all(
            isinstance(r, ResourceIdentity) for r in self.resources
        ):
            raise TypeError("resources must be a tuple of ResourceIdentity")
        if len({r.uri for r in self.resources}) != len(self.resources):
            raise ValueError("resource URIs must be unique within a source identity")

    def _document(self) -> dict[str, object]:
        value: dict[str, object] = {
            "provider": self.provider,
            "product": self.product,
            "resources": [r._document() for r in self.resources],
        }
        if self.release is not None:
            value["release"] = self.release._document()
        if self.member is not None:
            value["member"] = self.member
        return value


def _resource(value: object) -> ResourceIdentity:
    v = _object(value, {"uri", "sha256"}, {"license", "attribution", "retrieved_at"})
    if "retrieved_at" in v and not isinstance(v["retrieved_at"], str):
        raise ValueError("retrieved_at must be a timestamp string")
    return ResourceIdentity(
        _identifier(v["uri"], "uri"),
        _digest(v["sha256"]),
        _text(v["license"], "license") if "license" in v else None,
        _text(v["attribution"], "attribution") if "attribution" in v else None,
        _timestamp(v["retrieved_at"]) if "retrieved_at" in v else None,
    )


def _release(value: object) -> ReleaseReference:
    v = _object(value, {"id", "sha256"}, {"retrieved_at"})
    if "retrieved_at" in v and not isinstance(v["retrieved_at"], str):
        raise ValueError("retrieved_at must be a timestamp string")
    return ReleaseReference(
        _identifier(v["id"], "id"),
        _digest(v["sha256"]),
        _timestamp(v["retrieved_at"]) if "retrieved_at" in v else None,
    )


def _source(value: object) -> SourceIdentity:
    v = _object(value, {"provider", "product", "resources"}, {"release", "member"})
    return SourceIdentity(
        _identifier(v["provider"], "provider"),
        _identifier(v["product"], "product"),
        _release(v["release"]) if "release" in v else None,
        tuple(_resource(r) for r in _array(v["resources"], "resources")),
        _identifier(v["member"], "member") if "member" in v else None,
    )


class Stage(Enum):
    ACQUISITION = "acquisition"
    DECODING = "decoding"
    SELECTION = "selection"
    REDUCTION = "reduction"
    FITTING = "fitting"
    JOINING = "joining"
    EVALUATION = "evaluation"
    VALIDATION = "validation"
    WRITING = "writing"


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostic:
    code: str
    severity: crc_framework.DiagnosticSeverity
    stage: Stage
    message: str
    source: str | None = None
    support_id: str | None = None
    query_value: float | None = None
    remedy: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.code, "diagnostic.code")
        _text(self.message, "diagnostic.message")
        if not isinstance(
            self.severity, crc_framework.DiagnosticSeverity
        ) or not isinstance(self.stage, Stage):
            raise TypeError("diagnostics require DiagnosticSeverity and Stage enums")
        for name in ("source", "support_id"):
            if getattr(self, name) is not None:
                _identifier(getattr(self, name), name)
        if self.remedy is not None:
            _text(self.remedy, "remedy")
        if self.query_value is not None:
            _number(self.query_value, "query_value")

    def _document(self) -> dict[str, object]:
        value: dict[str, object] = {
            "code": self.code,
            "severity": self.severity.value,
            "stage": self.stage.value,
            "message": self.message,
        }
        for name in ("source", "support_id", "query_value", "remedy"):
            if getattr(self, name) is not None:
                value[name] = getattr(self, name)
        return value


@dataclass(frozen=True, slots=True)
class FittingDiagnostic(ExecutionDiagnostic):
    record: FitRecord = field(kw_only=True)

    def __post_init__(self) -> None:
        ExecutionDiagnostic.__post_init__(self)
        if self.stage is not Stage.FITTING:
            raise ValueError("fitting diagnostics require Stage.FITTING")
        if not isinstance(self.record, FitRecord):
            raise TypeError("record must be a framework FitRecord")
        # FitRecord's wire decoder validates its complete trace and freezes all
        # nested diagnostics, including records manually constructed by callers.
        object.__setattr__(self, "record", FitRecord.from_json(self.record.to_json()))

    def _document(self) -> dict[str, object]:
        value = ExecutionDiagnostic._document(self)
        value["record"] = _load(self.record.to_json())
        return value


@dataclass(frozen=True, slots=True)
class ExecutionCounts:
    requested: int = 0
    produced: int = 0
    skipped: int = 0
    no_data: int = 0
    matched_assets: int = 0
    unmatched_assets: int = 0
    usable_years: int = 0
    incomplete_years: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), name)

    def _document(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    complete: bool
    valid: bool | None
    diagnostics: tuple[ExecutionDiagnostic, ...] = ()
    counts: ExecutionCounts = field(default_factory=ExecutionCounts)
    source_identities: tuple[SourceIdentity, ...] = ()

    def __post_init__(self) -> None:
        _boolean(self.complete, "complete")
        if self.valid is not None:
            _boolean(self.valid, "valid")
        if self.valid is True and not self.complete:
            raise ValueError("valid can be true only after complete validation")
        if not isinstance(self.counts, ExecutionCounts):
            raise TypeError("counts must be ExecutionCounts")
        for name, cls in (
            ("diagnostics", ExecutionDiagnostic),
            ("source_identities", SourceIdentity),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(
                isinstance(v, cls) for v in values
            ):
                raise TypeError(f"{name} must be an immutable typed tuple")
        if self.valid is True and any(
            d.severity is crc_framework.DiagnosticSeverity.ERROR
            for d in self.diagnostics
        ):
            raise ValueError("a valid report cannot contain error diagnostics")

    def _document(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "valid": self.valid,
            "counts": self.counts._document(),
            "diagnostics": [d._document() for d in self.diagnostics],
            "source_identities": [s._document() for s in self.source_identities],
        }


def _diagnostic(value: object) -> ExecutionDiagnostic:
    v = _object(
        value,
        {"code", "severity", "stage", "message"},
        {"source", "support_id", "query_value", "remedy", "record"},
    )
    arguments = (
        _identifier(v["code"], "code"),
        crc_framework.DiagnosticSeverity(v["severity"]),
        Stage(v["stage"]),
        _text(v["message"], "message"),
        _identifier(v["source"], "source") if "source" in v else None,
        _identifier(v["support_id"], "support_id") if "support_id" in v else None,
        None if "query_value" not in v else _number(v["query_value"], "query_value"),
        _text(v["remedy"], "remedy") if "remedy" in v else None,
    )
    if "record" in v:
        return FittingDiagnostic(
            *arguments, record=FitRecord.from_json(_dump(v["record"]))
        )
    return ExecutionDiagnostic(*arguments)


def _report(value: object) -> ExecutionReport:
    v = _object(
        value, {"complete", "valid", "diagnostics", "counts", "source_identities"}
    )
    counts = _object(v["counts"], set(ExecutionCounts.__dataclass_fields__))
    return ExecutionReport(
        _boolean(v["complete"], "complete"),
        None if v["valid"] is None else _boolean(v["valid"], "valid"),
        tuple(_diagnostic(d) for d in _array(v["diagnostics"], "diagnostics")),
        ExecutionCounts(**{n: _wire_integer(x, n) for n, x in counts.items()}),
        tuple(_source(s) for s in _array(v["source_identities"], "source_identities")),
    )


class SupportKind(Enum):
    POINT = "point"
    POLYGON = "polygon"
    CELL = "cell"


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialContract:
    supports: tuple[SupportKind, ...]
    index_resolution: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.supports, tuple)
            or not self.supports
            or not all(isinstance(k, SupportKind) for k in self.supports)
        ):
            raise TypeError("supports must be a nonempty tuple of SupportKind")
        if len(set(self.supports)) != len(self.supports):
            raise ValueError("support kinds must be unique")
        if (
            self.index_resolution is not None
            and _integer(self.index_resolution, "index_resolution") > 15
        ):
            raise ValueError("index_resolution must be between zero and fifteen")

    def _document(self) -> dict[str, object]:
        value: dict[str, object] = {"supports": [kind.value for kind in self.supports]}
        if self.index_resolution is not None:
            value["index_resolution"] = self.index_resolution
        return value


def _spatial(value: object) -> SpatialContract:
    v = _object(value, {"supports"}, {"index_resolution"})
    return SpatialContract(
        supports=tuple(SupportKind(k) for k in _array(v["supports"], "supports")),
        index_resolution=None
        if "index_resolution" not in v
        else _wire_integer(v["index_resolution"], "index_resolution"),
    )


class IdentityKind(Enum):
    VARIABLE = "variable"
    SCENARIO = "scenario"


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityRename:
    dimension: IdentityKind
    mapping: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, IdentityKind) or not isinstance(
            self.mapping, Mapping
        ):
            raise TypeError("rename provenance requires IdentityKind and a mapping")
        mapping = {
            _identifier(k, "old id"): _identifier(v, "new id")
            for k, v in self.mapping.items()
        }
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("rename mappings must be injective")
        object.__setattr__(self, "mapping", MappingProxyType(mapping))

    def _document(self) -> dict[str, object]:
        return {
            "kind": "rename",
            "dimension": self.dimension.value,
            "mapping": dict(self.mapping),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitConversion:
    variable_id: str
    source_unit: str
    target_unit: str
    scale: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        _identifier(self.variable_id, "variable_id")
        _identifier(self.source_unit, "source_unit")
        _identifier(self.target_unit, "target_unit")
        if self.source_unit == self.target_unit:
            raise ValueError("a conversion must change units")
        if _number(self.scale, "scale") == 0:
            raise ValueError("conversion scale must be nonzero")
        _number(self.offset, "offset")

    def _document(self) -> dict[str, object]:
        return {
            "kind": "unit_conversion",
            "variable_id": self.variable_id,
            "source_unit": self.source_unit,
            "target_unit": self.target_unit,
            "scale": self.scale,
            "offset": self.offset,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationTimeSelection:
    time: Period

    def __post_init__(self) -> None:
        if not isinstance(self.time, Period) or self.time.kind is not PeriodKind.WINDOW:
            raise ValueError("observation time selection requires a Period.window")

    def _document(self) -> dict[str, object]:
        return {"kind": "observation_time_selection", "time": self.time._document()}


Transformation = UnitConversion | IdentityRename | ObservationTimeSelection


def _transformation(value: object) -> Transformation:
    v = _object(
        value,
        {"kind"},
        {
            "dimension",
            "mapping",
            "variable_id",
            "source_unit",
            "target_unit",
            "scale",
            "offset",
            "time",
        },
    )
    if v["kind"] == "rename":
        _object(v, {"kind", "dimension", "mapping"})
        mapping = v["mapping"]
        if not isinstance(mapping, Mapping):
            raise ValueError("rename mapping must be an object")
        return IdentityRename(
            dimension=IdentityKind(v["dimension"]),
            mapping=cast(Mapping[str, str], mapping),
        )
    if v["kind"] == "unit_conversion":
        _object(
            v, {"kind", "variable_id", "source_unit", "target_unit", "scale", "offset"}
        )
        return UnitConversion(
            variable_id=_identifier(v["variable_id"], "variable_id"),
            source_unit=_identifier(v["source_unit"], "source_unit"),
            target_unit=_identifier(v["target_unit"], "target_unit"),
            scale=_number(v["scale"], "scale"),
            offset=_number(v["offset"], "offset"),
        )
    if v["kind"] == "observation_time_selection":
        _object(v, {"kind", "time"})
        return ObservationTimeSelection(time=_period(v["time"]))
    raise ValueError("unknown transformation kind")


@dataclass(frozen=True, slots=True)
class RetainedFailure:
    code: str
    message: str
    record: FitRecord

    def __post_init__(self) -> None:
        _identifier(self.code, "failure.code")
        _text(self.message, "failure.message")
        if not isinstance(self.record, FitRecord) or self.record.converged:
            raise ValueError(
                "retained failures require an authoritative failed fit record"
            )

    @classmethod
    def from_error(cls, error: FitError) -> RetainedFailure:
        if not isinstance(error, FitError):
            raise TypeError("error must be a framework FitError")
        return cls(error.code, error.message, error.record)

    def _document(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "record": _load(self.record.to_json()),
        }


def _failure(value: object) -> RetainedFailure:
    v = _object(value, {"code", "message", "record"})
    return RetainedFailure(
        _identifier(v["code"], "code"),
        _text(v["message"], "message"),
        FitRecord.from_json(_dump(v["record"])),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetMetadata:
    spec: DataSpec
    spatial: SpatialContract
    report: ExecutionReport
    source: SourceIdentity | None = None
    provenance: tuple[Transformation, ...] = ()

    @property
    def schema_version(self) -> str:
        return "2.0"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.spec, DataSpec)
            or not isinstance(self.spatial, SpatialContract)
            or not isinstance(self.report, ExecutionReport)
        ):
            raise TypeError(
                "metadata requires DataSpec, SpatialContract, and ExecutionReport"
            )
        if self.source is not None and not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        if not isinstance(self.provenance, tuple) or not all(
            isinstance(p, (UnitConversion, IdentityRename, ObservationTimeSelection))
            for p in self.provenance
        ):
            raise TypeError("provenance must contain typed immutable transformations")
        units: dict[str, str | None] = {}
        for variable in self.spec.variables:
            unit = None if variable.measurement is None else variable.measurement.unit
            units[variable.id] = unit.value if isinstance(unit, Enum) else unit
        for step in reversed(self.provenance):
            if isinstance(step, UnitConversion):
                if step.variable_id not in units:
                    raise ValueError(
                        "conversion provenance refers to an unknown variable"
                    )
                if units[step.variable_id] != step.target_unit:
                    raise ValueError(
                        "conversion target unit contradicts the variable measurement"
                    )
                units[step.variable_id] = step.source_unit
            elif (
                isinstance(step, IdentityRename)
                and step.dimension is IdentityKind.VARIABLE
            ):
                inverse = {new: old for old, new in step.mapping.items()}
                before = {inverse.get(name, name): unit for name, unit in units.items()}
                if len(before) != len(units):
                    raise ValueError(
                        "rename provenance conflicts with participating identities"
                    )
                units = before

    def _document(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "spec": self.spec._document(),
            "spatial": self.spatial._document(),
            "report": self.report._document(),
            "provenance": [p._document() for p in self.provenance],
        }
        if self.source is not None:
            result["source"] = self.source._document()
        return result

    def to_json(self) -> str:
        return _dump(self._document())

    @classmethod
    def from_json(cls, text: str) -> DatasetMetadata:
        v = _object(
            _load(text),
            {"schema_version", "spec", "spatial", "report", "provenance"},
            {"source"},
        )
        if v["schema_version"] != "2.0":
            raise ValueError("expected dataset schema 2.0")
        return cls(
            spec=_data_spec(v["spec"]),
            spatial=_spatial(v["spatial"]),
            report=_report(v["report"]),
            source=None if "source" not in v else _source(v["source"]),
            provenance=tuple(
                _transformation(p) for p in _array(v["provenance"], "provenance")
            ),
        )
