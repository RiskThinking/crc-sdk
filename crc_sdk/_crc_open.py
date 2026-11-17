"""Current CRC Open release indexes and exact canonical dataset inputs."""

from __future__ import annotations

import json
import os
import urllib.parse
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ._dataset_schema import (
    DatasetMetadata,
    ExecutionReport,
    ReleaseReference,
    SourceIdentity,
    Stage,
    SupportKind,
    _digest,
    _identifier,
    _object,
)
from ._declarations import (
    DataSpec,
    Period,
    Scenario,
    SeriesSpec,
    Variable,
    _merge_descriptor,
    scientific_identity,
)
from ._graph import ExecutionScope, NodeStream
from ._inspection import Capability, Inspection, Operation, SeriesReference
from ._recipes import Dataset, _checked_stream, _key
from ._runtime import LocalPath
from ._selection import SelectionState, selected_metadata
from ._source_io import SourceNode, check_release, fetch, fetch_json, native_data
from ._terminals import _schema

CRC_OPEN_ROOT = "https://data.riskthinking.ai/crc-open"


def _relative(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or "?" in value
        or "#" in value
    ):
        raise ValueError("release resources must use relative paths within the release")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise ValueError("release resource escapes its release directory")
    return value


def _location(value: LocalPath | None, requested: ReleaseReference | None) -> str:
    if value is None:
        return CRC_OPEN_ROOT + (
            "/_CURRENT.json"
            if requested is None
            else "/" + urllib.parse.quote(requested.id, safe="") + "/_INDEX.json"
        )
    text = os.fspath(value)
    if urllib.parse.urlsplit(text).scheme in {"http", "https", "file"}:
        return text
    return Path(text).absolute().as_uri()


@dataclass(frozen=True, slots=True)
class ReleaseDataset:
    id: str
    uri: str
    sha256: str
    metadata: DatasetMetadata

    @classmethod
    def parse(cls, value: object) -> ReleaseDataset:
        obj = _object(value, {"id", "uri", "sha256", "metadata"})
        return cls(
            _identifier(obj["id"], "release dataset id"),
            _relative(obj["uri"]),
            _digest(obj["sha256"]),
            DatasetMetadata.from_json(json.dumps(obj["metadata"], allow_nan=False)),
        )


def release_index(value: object) -> tuple[str, tuple[ReleaseDataset, ...]]:
    obj = _object(value, {"schema_version", "id", "datasets"})
    if obj["schema_version"] != "1.0":
        raise ValueError(
            "CRC Open requires a current release index and dataset format2.0"
        )
    if not isinstance(obj["datasets"], list):
        raise ValueError("release datasets must be an array")
    entries = tuple(ReleaseDataset.parse(item) for item in obj["datasets"])
    if len({item.id for item in entries}) != len(entries) or len(
        {item.uri for item in entries}
    ) != len(entries):
        raise ValueError("release dataset IDs and resource paths must be unique")
    return _identifier(obj["id"], "release id"), entries


def _series_key(item: SeriesSpec) -> tuple[str, str | None, str | None]:
    return (
        item.variable.id,
        None if item.scenario is None else item.scenario.id,
        None if item.period is None else item.period.id,
    )


def _selected(
    entries: tuple[ReleaseDataset, ...], state: SelectionState, source: SourceIdentity
) -> tuple[tuple[ReleaseDataset, DatasetMetadata], ...]:
    metadata = _merge(tuple((entry, entry.metadata) for entry in entries), source)
    resolved = state.resolve(metadata.spec, source=source)
    result: list[tuple[ReleaseDataset, DatasetMetadata]] = []
    for entry in entries:
        selected = selected_metadata(replace(entry.metadata, source=source), resolved)
        if selected.spec.series:
            result.append((entry, selected))
    return tuple(result)


def _merge(
    parts: tuple[tuple[ReleaseDataset, DatasetMetadata], ...], source: SourceIdentity
) -> DatasetMetadata:
    variables: dict[str, Variable] = {}
    scenarios: dict[str, Scenario] = {}
    periods: dict[str, Period] = {}
    series: dict[tuple[str, str | None, str | None], SeriesSpec] = {}
    supports: list[SupportKind] = []
    resolutions: set[int | None] = set()
    for _, metadata in parts:
        for value in metadata.spec.variables:
            _merge_descriptor(value, variables)
        for scenario in metadata.spec.scenarios:
            _merge_descriptor(scenario, scenarios)
        for period in metadata.spec.periods:
            _merge_descriptor(period, periods)
        for item in metadata.spec.series:
            key = _series_key(item)
            if key in series and scientific_identity(
                series[key]
            ) != scientific_identity(item):
                raise ValueError(
                    "participating release series have conflicting "
                    "scientific definitions"
                )
            series[key] = item
        supports.extend(metadata.spatial.supports)
        resolutions.add(metadata.spatial.index_resolution)
    from ._dataset_schema import SpatialContract, SupportKind

    spec = DataSpec(
        tuple(variables.values()),
        series=tuple(series.values()),
        scenarios=tuple(scenarios.values()),
        periods=tuple(periods.values()),
    )
    return DatasetMetadata(
        spec=spec,
        spatial=SpatialContract(
            supports=tuple(dict.fromkeys(supports)) or (SupportKind.CELL,),
            index_resolution=next(iter(resolutions)) if len(resolutions) == 1 else None,
        ),
        source=source,
        report=ExecutionReport(False, None, source_identities=(source,)),
    )


@dataclass(frozen=True)
class _CRCNode(SourceNode[DatasetMetadata]):
    catalog: str
    release: ReleaseReference | None
    state: SelectionState = SelectionState()
    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _key(
                "crc-open",
                self.catalog,
                None
                if self.release is None
                else (self.release.id, self.release.sha256),
                self.state.document(),
            ),
        )

    def selected(self, state: SelectionState) -> _CRCNode:
        return replace(self, state=state)

    def _resolve(
        self, scope: ExecutionScope
    ) -> tuple[str, tuple[ReleaseDataset, ...], SourceIdentity]:
        with native_data(scope=scope, source="crc-open"):
            pointer_id: str | None = None
            document, resource = fetch_json(
                scope, self.catalog, license="CC-BY-NC-4.0", attribution="RiskThinking"
            )
            if isinstance(document, Mapping) and set(document) == {
                "schema_version",
                "release",
            }:
                if document["schema_version"] != "1.0":
                    raise ValueError("unsupported CRC Open release pointer")
                pointer = _object(document["release"], {"id", "uri", "sha256"})
                pointer_id = _identifier(pointer["id"], "release pointer id")
                uri = urllib.parse.urljoin(self.catalog, _relative(pointer["uri"]))
                document, resource = fetch_json(
                    scope,
                    uri,
                    expected_sha256=_digest(pointer["sha256"]),
                    license="CC-BY-NC-4.0",
                    attribution="RiskThinking",
                )
            else:
                uri = self.catalog
            id, entries = release_index(document)
            if pointer_id is not None and pointer_id != id:
                raise ValueError("release pointer identity differs from its index")
            actual = ReleaseReference(
                id, resource.sha256, retrieved_at=resource.retrieved_at
            )
            check_release(actual, self.release)
            source = SourceIdentity(
                provider="crc",
                product="crc-open",
                release=actual,
                resources=(resource,),
            )
            return uri, entries, source

    def _empty(self) -> bool:
        return any(
            value == ()
            for value in (
                self.state.variables,
                self.state.scenarios,
                self.state.periods,
                self.state.series,
            )
        )

    def inspection(self, scope: ExecutionScope) -> Inspection:
        if self._empty():
            source = SourceIdentity(provider="crc", product="crc-open")
            selected: tuple[tuple[ReleaseDataset, DatasetMetadata], ...] = ()
        else:
            _, entries, source = self._resolve(scope)
            selected = _selected(entries, self.state, source)
        metadata = _merge(selected, source)
        return Inspection(
            schema=_schema(metadata),
            series=tuple(
                SeriesReference(
                    s.variable,
                    scenario=s.scenario,
                    period=s.period,
                    probability=s.probability,
                    release=source.release,
                )
                for s in metadata.spec.series
            ),
            releases=() if source.release is None else (source.release,),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
                Capability.FIT,
            ),
            source=source,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        if self._empty():
            return _checked_stream(
                scope,
                _merge((), SourceIdentity(provider="crc", product="crc-open")),
                iter(()),
            )
        catalog, entries, source = self._resolve(scope)
        selected = _selected(entries, self.state, source)
        metadata = _merge(selected, source)
        identities = list(source.resources)

        def batches() -> Iterator[pa.RecordBatch]:
            for entry, entrymeta in selected:
                uri = urllib.parse.urljoin(catalog, entry.uri)
                path, resource = fetch(
                    scope,
                    uri,
                    expected_sha256=entry.sha256,
                    license="CC-BY-NC-4.0",
                    attribution="RiskThinking",
                )
                identities.append(resource)
                with pq.ParquetFile(path) as reader:
                    stored = (reader.schema_arrow.metadata or {}).get(b"crc.dataset")
                    if (
                        stored is None
                        or DatasetMetadata.from_json(stored.decode()).to_json()
                        != entry.metadata.to_json()
                    ):
                        raise ValueError(
                            "release dataset header differs from "
                            "its indexed declaration"
                        )
                    from ._schema import rows_to_table, table_to_rows

                    if reader.num_row_groups == 0:
                        continue
                    for batch in reader.iter_batches(
                        batch_size=scope.runtime.batch_rows
                    ):
                        raw = pa.Table.from_batches([batch]).replace_schema_metadata(
                            _schema(
                                replace(
                                    entry.metadata,
                                    report=replace(
                                        entry.metadata.report,
                                        complete=False,
                                        valid=None,
                                    ),
                                )
                            ).metadata
                        )
                        rows = table_to_rows(raw)
                        resolved = SelectionState(area=self.state.area).resolve(
                            entrymeta.spec, source=source
                        )
                        keys = {_series_key(item) for item in entrymeta.spec.series}
                        rows = tuple(
                            row
                            for row in rows
                            if (row.variable_id, row.scenario_id, row.period_id) in keys
                            and resolved.matches(row)
                        )
                        if rows:
                            yield from rows_to_table(rows, metadata).to_batches()

        stream = _checked_stream(scope, metadata, batches())
        original = stream.report

        def report() -> ExecutionReport:
            assert original is not None
            final = replace(source, resources=tuple(identities))
            return replace(original(), source_identities=(final,))

        stream.report = report
        stream.final_metadata = lambda: replace(
            metadata, source=replace(source, resources=tuple(identities))
        )
        return stream


class CRCOpen(Dataset):
    class Variable(str, Enum):
        ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE = "crc:txx"
        ANNUAL_MINIMUM_DAILY_MAXIMUM_TEMPERATURE = "crc:txn"
        ANNUAL_MAXIMUM_DAILY_MINIMUM_TEMPERATURE = "crc:tnx"
        ANNUAL_MINIMUM_DAILY_MINIMUM_TEMPERATURE = "crc:tnn"
        ANNUAL_MAXIMUM_DAILY_PRECIPITATION = "crc:rx1day"
        ANNUAL_MAXIMUM_FIVE_DAY_PRECIPITATION = "crc:rx5day"
        ANNUAL_MEAN_WET_DAY_PRECIPITATION = "crc:sdii"
        ANNUAL_MAXIMUM_CONSECUTIVE_DRY_DAYS = "crc:cdd"
        ANNUAL_MAXIMUM_CONSECUTIVE_WET_DAYS = "crc:cwd"
        ANNUAL_FROST_DAY_COUNT = "crc:fd"
        ANNUAL_SUMMER_DAY_COUNT = "crc:su"
        ANNUAL_TROPICAL_NIGHT_COUNT = "crc:tr"
        ANNUAL_MEAN_DAILY_TEMPERATURE_RANGE = "crc:dtr"
        ANNUAL_TOTAL_WET_DAY_PRECIPITATION = "crc:prcptot"
        ANNUAL_PRECIPITATION_DAYS_AT_LEAST_TEN_MILLIMETRES = "crc:r10mm"
        ANNUAL_PRECIPITATION_DAYS_AT_LEAST_TWENTY_MILLIMETRES = "crc:r20mm"
        ANNUAL_PRECIPITATION_ABOVE_BASELINE_WET_DAY_NINETY_FIFTH_PERCENTILE = "crc:r95p"

    class Scenario(str, Enum):
        SSP585 = "ssp585"

    def __init__(
        self,
        *,
        release: ReleaseReference | None = None,
        catalog: LocalPath | None = None,
    ) -> None:
        if release is not None and not isinstance(release, ReleaseReference):
            raise TypeError("release must be an inspected ReleaseReference")
        node = _CRCNode(_location(catalog, release), release)
        for name, value in (
            ("_base", node),
            ("_selection", SelectionState()),
            (
                "_operations",
                (Operation(Stage.DECODING, "Read current CRC Open release datasets"),),
            ),
            ("_path", None),
            ("_metadata", None),
        ):
            object.__setattr__(self, name, value)
