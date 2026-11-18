"""Typed collection inventory, selection, and canonical Arrow layout."""

from __future__ import annotations

import re
import types
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]

from ._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    ReleaseReference,
    SourceIdentity,
    _array,
    _digest,
    _dump,
    _identifier,
    _load,
    _number,
    _object,
    _release,
    _report,
    _source,
)
from ._declarations import (
    DataSpec,
    Period,
    Scenario,
    SeriesSpec,
    Variable,
    _merge_descriptor,
    _period,
    _probability,
    _scenario,
    _variable,
    scientific_identity,
)
from ._inspection import Bounds, SeriesReference
from ._schema import dataset_arrow_schema
from ._schema.contract import _record, dataset_json_schema
from ._selection import (
    IdentifierSelector,
    PeriodSelector,
    ScenarioSelector,
    SelectionState,
    SeriesKey,
    SeriesSelector,
    VariableSelector,
    _series_key,
)

_METADATA_KEY = b"crc.collection"
_RESOURCE_PATH = re.compile(r"resources/[0-9a-f]{64}\.parquet")


def _collection_report(
    reports: Iterable[ExecutionReport], *, complete: bool = False
) -> ExecutionReport:
    diagnostics: dict[str, ExecutionDiagnostic] = {}
    sources: dict[str, SourceIdentity] = {}
    counts = dict.fromkeys(ExecutionCounts.__dataclass_fields__, 0)
    for report in reports:
        for diagnostic in report.diagnostics:
            diagnostics.setdefault(_dump(diagnostic._document()), diagnostic)
        for source in report.source_identities:
            sources.setdefault(_dump(source._document()), source)
        for name in counts:
            counts[name] += getattr(report.counts, name)
    return ExecutionReport(
        complete,
        True if complete else None,
        tuple(diagnostics.values()),
        ExecutionCounts(**counts),
        tuple(sources.values()),
    )


def _member_ids(value: IdentifierSelector) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (_identifier(value, "member id"),)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError("members must be one id or a sequence of ids")
    return tuple(dict.fromkeys(_identifier(v, "member id") for v in value))


def _merge_specs(specs: Sequence[DataSpec]) -> DataSpec:
    variables: dict[str, Variable] = {}
    scenarios: dict[str, Scenario] = {}
    periods: dict[str, Period] = {}
    series: dict[SeriesKey, SeriesSpec] = {}
    for spec in specs:
        for item in spec.series:
            _merge_descriptor(item.variable, variables)
            _merge_descriptor(item.scenario, scenarios)
            _merge_descriptor(item.period, periods)
            key = _series_key(item)
            previous = series.get(key)
            if previous is not None and scientific_identity(
                previous
            ) != scientific_identity(item):
                raise ValueError(
                    f"conflicting scientific definition for series {key!r}"
                )
            series.setdefault(key, item)
    return DataSpec(
        tuple(variables.values()),
        series=tuple(series.values()),
        scenarios=tuple(scenarios.values()),
        periods=tuple(periods.values()),
    )


def _release_key(release: ReleaseReference | None) -> tuple[str, str] | None:
    return None if release is None else (release.id, release.sha256)


def _requested_ids(
    values: Sequence[str | Variable | Scenario | Period] | None,
) -> set[str] | None:
    return (
        None
        if values is None
        else {value if isinstance(value, str) else value.id for value in values}
    )


def _typed_match(
    values: Sequence[str | Variable | Scenario | Period] | None,
    actual: Variable | Scenario | Period | None,
) -> None:
    if actual is None:
        return
    for value in values or ():
        if (
            not isinstance(value, str)
            and value.id == actual.id
            and scientific_identity(value) != scientific_identity(actual)
        ):
            raise ValueError(f"conflicting scientific definition for {actual.id!r}")


def _reference_matches(
    item: SeriesSpec,
    reference: SeriesReference,
    source: SourceIdentity | None,
    *,
    check_science: bool = True,
) -> bool:
    if _series_key(item) != _series_key(reference):
        return False
    release = None if source is None else source.release
    if reference.release is not None and _release_key(
        reference.release
    ) != _release_key(release):
        return False
    if any(
        getattr(reference, name) is not None
        and (source is None or getattr(reference, name) != getattr(source, name))
        for name in ("model", "member")
    ):
        return False
    if not check_science:
        return True
    for actual, requested in (
        (item.variable, reference.variable),
        (item.scenario, reference.scenario),
        (item.period, reference.period),
        (item.probability, reference.probability),
    ):
        if (
            actual is not None
            and requested is not None
            and scientific_identity(actual) != scientific_identity(requested)
        ):
            raise ValueError(
                f"conflicting scientific series definition {_series_key(item)!r}"
            )
    return True


def _series_document(reference: SeriesReference) -> dict[str, object]:
    result: dict[str, object] = {
        "variable": reference.variable._document(),
        "scenario": None
        if reference.scenario is None
        else reference.scenario._document(),
        "period": None if reference.period is None else reference.period._document(),
    }
    if reference.release is not None:
        result["release"] = reference.release._document()
    if reference.probability is not None:
        result["probability"] = reference.probability._document()
    if reference.model is not None:
        result["model"] = reference.model
    if reference.member is not None:
        result["member"] = reference.member
    return result


def _series_reference(value: object) -> SeriesReference:
    record = _object(
        value,
        {"variable", "scenario", "period"},
        {"release", "probability", "model", "member"},
    )
    return SeriesReference(
        variable=_variable(record["variable"]),
        scenario=None if record["scenario"] is None else _scenario(record["scenario"]),
        period=None if record["period"] is None else _period(record["period"]),
        probability=None
        if "probability" not in record
        else _probability(record["probability"]),
        release=None if "release" not in record else _release(record["release"]),
        model=None
        if "model" not in record
        else _identifier(record["model"], "series model"),
        member=None
        if "member" not in record
        else _identifier(record["member"], "series member"),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectionSelection:
    members: tuple[str, ...] | None = None
    variables: tuple[Variable | str, ...] | None = None
    scenarios: tuple[Scenario | str, ...] | None = None
    periods: tuple[Period, ...] | None = None
    series: tuple[SeriesReference, ...] | None = None
    area: Bounds | None = None

    def __post_init__(self) -> None:
        for name in ("members", "variables", "scenarios", "periods", "series"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                raise TypeError(f"collection selection {name} must be a tuple or None")
        _member_ids(self.members)
        SelectionState().updated(
            variables=self.variables,
            scenarios=self.scenarios,
            periods=self.periods,
            series=self.series,
            area=self.area,
        )

    def _updated(
        self,
        *,
        members: IdentifierSelector | types.EllipsisType = ...,
        variables: VariableSelector | types.EllipsisType = ...,
        scenarios: ScenarioSelector | types.EllipsisType = ...,
        periods: PeriodSelector | types.EllipsisType = ...,
        series: SeriesSelector | types.EllipsisType = ...,
        area: Bounds | None | types.EllipsisType = ...,
    ) -> CollectionSelection:
        state = self._state().updated(
            variables=variables,
            scenarios=scenarios,
            periods=periods,
            series=series,
            area=area,
        )
        return CollectionSelection(
            members=self.members if members is ... else _member_ids(members),
            variables=state.variables,
            scenarios=state.scenarios,
            periods=state.periods,
            series=state.series,
            area=state.area,
        )

    def _state(self) -> SelectionState:
        return SelectionState(
            variables=self.variables,
            scenarios=self.scenarios,
            periods=self.periods,
            series=self.series,
            area=self.area,
        )

    def _empty(self) -> bool:
        return any(
            value == ()
            for value in (
                self.members,
                self.variables,
                self.scenarios,
                self.periods,
                self.series,
            )
        )

    def _has_scientific_filters(self) -> bool:
        return any((self.variables, self.scenarios, self.periods, self.series))

    def _document(self) -> dict[str, object]:
        return {
            "members": None if self.members is None else list(self.members),
            "variables": None
            if self.variables is None
            else [
                value if isinstance(value, str) else value._document()
                for value in self.variables
            ],
            "scenarios": None
            if self.scenarios is None
            else [
                value if isinstance(value, str) else value._document()
                for value in self.scenarios
            ],
            "periods": None
            if self.periods is None
            else [p._document() for p in self.periods],
            "series": None
            if self.series is None
            else [_series_document(s) for s in self.series],
            "area": self._state().document()["area"],
        }


def _collection_selection(value: object) -> CollectionSelection:
    record = _object(
        value, {"members", "variables", "scenarios", "periods", "series", "area"}
    )
    members = (
        None
        if record["members"] is None
        else tuple(
            _identifier(member, "member id")
            for member in _array(record["members"], "members")
        )
    )
    variables = (
        None
        if record["variables"] is None
        else tuple(
            _identifier(v, "variable id") if isinstance(v, str) else _variable(v)
            for v in _array(record["variables"], "variables")
        )
    )
    scenarios = (
        None
        if record["scenarios"] is None
        else tuple(
            _identifier(s, "scenario id") if isinstance(s, str) else _scenario(s)
            for s in _array(record["scenarios"], "scenarios")
        )
    )
    periods = (
        None
        if record["periods"] is None
        else tuple(_period(p) for p in _array(record["periods"], "periods"))
    )
    series = (
        None
        if record["series"] is None
        else tuple(_series_reference(s) for s in _array(record["series"], "series"))
    )
    bounds = None
    if record["area"] is not None:
        area = _object(record["area"], {"west", "south", "east", "north"})
        bounds = Bounds(
            west=_number(area["west"], "west"),
            south=_number(area["south"], "south"),
            east=_number(area["east"], "east"),
            north=_number(area["north"], "north"),
        )
    return CollectionSelection(
        members=members,
        variables=variables,
        scenarios=scenarios,
        periods=periods,
        series=series,
        area=bounds,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectionMember:
    id: str
    resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "member id")
        if not isinstance(self.resource_ids, tuple):
            raise TypeError("resource_ids must be an immutable tuple")
        for resource_id in self.resource_ids:
            _identifier(resource_id, "resource id")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise ValueError(f"duplicate resource id in member {self.id!r}")

    def _document(self) -> dict[str, object]:
        return {"id": self.id, "resource_ids": list(self.resource_ids)}


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectionResource:
    id: str
    metadata: DatasetMetadata
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "resource id")
        if not isinstance(self.metadata, DatasetMetadata):
            raise TypeError("resource metadata must be DatasetMetadata")
        if (self.path is None) != (self.sha256 is None):
            raise ValueError("persisted resource path and sha256 must appear together")
        if self.path is not None:
            if not isinstance(self.path, str) or not _RESOURCE_PATH.fullmatch(
                self.path
            ):
                raise ValueError("invalid collection resource path")
            _digest(self.sha256)

    def _document(self) -> dict[str, object]:
        value: dict[str, object] = {
            "id": self.id,
            "metadata": self.metadata._document(),
        }
        if self.path is not None:
            value.update(path=self.path, sha256=self.sha256)
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectionMetadata:
    members: tuple[CollectionMember, ...]
    resources: tuple[CollectionResource, ...]
    selection: CollectionSelection
    report: ExecutionReport
    source: SourceIdentity | None = None
    _spec: DataSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.selection, CollectionSelection):
            raise TypeError("selection must be CollectionSelection")
        if not isinstance(self.report, ExecutionReport):
            raise TypeError("report must be ExecutionReport")
        if self.source is not None and not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        for name, values, expected in (
            ("members", self.members, CollectionMember),
            ("resources", self.resources, CollectionResource),
        ):
            if not isinstance(values, tuple) or not all(
                isinstance(v, expected) for v in values
            ):
                raise TypeError(f"{name} must be an immutable typed tuple")
            if len({item.id for item in values}) != len(values):
                raise ValueError(f"duplicate collection {name} identity")
        declared = {resource.id for resource in self.resources}
        referenced = {
            resource_id
            for member in self.members
            for resource_id in member.resource_ids
        }
        if declared != referenced:
            raise ValueError(
                "collection member resource references must match its inventory"
            )
        paths = [
            resource.path for resource in self.resources if resource.path is not None
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("collection resource paths must be unique")
        if self.report.complete and self.report.valid is True:
            self._validate_selection()
        object.__setattr__(
            self, "_spec", _merge_specs([r.metadata.spec for r in self.resources])
        )

    def _validate_selection(self) -> None:
        selection = self.selection
        if selection.members is not None and {m.id for m in self.members} != set(
            selection.members
        ):
            raise ValueError(
                "saved member selection does not match the bundle inventory"
            )
        if selection._empty() and self.resources:
            raise ValueError("an empty collection selection cannot contain resources")
        filters = (selection.variables, selection.scenarios, selection.periods)
        requested = tuple(map(_requested_ids, filters))
        for resource in self.resources:
            metadata = resource.metadata
            for item in metadata.spec.series:
                for actual, allowed, descriptors in zip(
                    (item.variable, item.scenario, item.period),
                    requested,
                    filters,
                    strict=True,
                ):
                    if allowed is not None and (
                        actual is None or actual.id not in allowed
                    ):
                        raise ValueError(
                            "resource series contradicts the saved selection"
                        )
                    _typed_match(descriptors, actual)
                if selection.series is not None and not any(
                    _reference_matches(item, reference, metadata.source)
                    for reference in selection.series
                ):
                    raise ValueError(
                        "resource series contradicts the saved series selection"
                    )

    @property
    def schema_version(self) -> str:
        return "1.0"

    @property
    def spec(self) -> DataSpec:
        """Participating scientific series reconciled across members."""
        return self._spec

    def to_json(self) -> str:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "members": [member._document() for member in self.members],
            "resources": [resource._document() for resource in self.resources],
            "selection": self.selection._document(),
            "report": self.report._document(),
        }
        if self.source is not None:
            value["source"] = self.source._document()
        return _dump(value)

    @classmethod
    def from_json(cls, text: str) -> CollectionMetadata:
        value = _object(
            _load(text),
            {"schema_version", "members", "resources", "selection", "report"},
            {"source"},
        )
        if value["schema_version"] != "1.0":
            raise ValueError("expected collection schema 1.0")
        members = []
        for item in _array(value["members"], "members"):
            member = _object(item, {"id", "resource_ids"})
            members.append(
                CollectionMember(
                    id=_identifier(member["id"], "member id"),
                    resource_ids=tuple(
                        _identifier(resource_id, "resource id")
                        for resource_id in _array(
                            member["resource_ids"], "resource_ids"
                        )
                    ),
                )
            )
        resources = []
        for item in _array(value["resources"], "resources"):
            resource = _object(item, {"id", "metadata"}, {"path", "sha256"})
            resources.append(
                CollectionResource(
                    id=_identifier(resource["id"], "resource id"),
                    metadata=DatasetMetadata.from_json(_dump(resource["metadata"])),
                    path=_identifier(resource["path"], "resource path")
                    if "path" in resource
                    else None,
                    sha256=_digest(resource["sha256"])
                    if "sha256" in resource
                    else None,
                )
            )
        return cls(
            members=tuple(members),
            resources=tuple(resources),
            selection=_collection_selection(value["selection"]),
            report=_report(value["report"]),
            source=_source(value["source"]) if "source" in value else None,
        )


def collection_arrow_schema(metadata: CollectionMetadata | None = None) -> pa.Schema:
    fields = [
        pa.field("member_id", pa.string(), nullable=False),
        pa.field("resource_id", pa.string(), nullable=False),
        *dataset_arrow_schema(),
    ]
    return pa.schema(
        fields,
        metadata=None
        if metadata is None
        else {_METADATA_KEY: metadata.to_json().encode()},
    )


def collection_metadata_schema() -> dict[str, object]:
    dataset = dataset_json_schema()
    definitions = dict(cast(dict[str, object], dataset["$defs"]))
    identifier = {"type": "string", "minLength": 1, "pattern": r"^\S(?:[\s\S]*\S)?$"}
    digest = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}

    def reference(name: str) -> dict[str, object]:
        return {"$ref": "#/$defs/" + name}

    def array(items: dict[str, object], *, nullable: bool = False) -> dict[str, object]:
        return {"type": ["array", "null"] if nullable else "array", "items": items}

    metadata = cast(dict[str, object], dataset["properties"])["metadata"]
    definitions["dataset_metadata"] = metadata
    definitions["collection_selection"] = _record(
        {
            "members": array(identifier, nullable=True),
            "variables": array(
                {"anyOf": [identifier, reference("variable")]}, nullable=True
            ),
            "scenarios": array(
                {"anyOf": [identifier, reference("scenario")]}, nullable=True
            ),
            "periods": array(reference("period"), nullable=True),
            "series": array(
                _record(
                    {
                        "variable": reference("variable"),
                        "scenario": {
                            "anyOf": [reference("scenario"), {"type": "null"}]
                        },
                        "period": {"anyOf": [reference("period"), {"type": "null"}]},
                        "release": reference("release"),
                        "probability": reference("probability"),
                        "model": identifier,
                        "member": identifier,
                    },
                    ["variable", "scenario", "period"],
                ),
                nullable=True,
            ),
            "area": {
                "anyOf": [
                    {"type": "null"},
                    _record(
                        {
                            name: {"type": "number"}
                            for name in ("west", "south", "east", "north")
                        },
                        ["west", "south", "east", "north"],
                    ),
                ]
            },
        },
        ["members", "variables", "scenarios", "periods", "series", "area"],
    )
    definitions["collection_member"] = _record(
        {"id": identifier, "resource_ids": array(identifier)}, ["id", "resource_ids"]
    )
    definitions["collection_resource"] = {
        **_record(
            {
                "id": identifier,
                "metadata": reference("dataset_metadata"),
                "path": {
                    "type": "string",
                    "pattern": r"^resources/[0-9a-f]{64}\.parquet$",
                },
                "sha256": digest,
            },
            ["id", "metadata"],
        ),
        "dependentRequired": {"path": ["sha256"], "sha256": ["path"]},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://climateriskcommons.org/schemas/collection-1.0.schema.json",
        "title": "CRC collection metadata",
        **_record(
            {
                "schema_version": {"const": "1.0"},
                "members": array(reference("collection_member")),
                "resources": array(reference("collection_resource")),
                "selection": reference("collection_selection"),
                "report": reference("report"),
                "source": reference("source"),
            },
            ["schema_version", "members", "resources", "selection", "report"],
        ),
        "$defs": definitions,
    }
