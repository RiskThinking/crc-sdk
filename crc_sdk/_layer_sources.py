"""Native precipitation rasters and agricultural field boundaries."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar, cast
from urllib.parse import urljoin, urlsplit

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely import from_wkb  # type: ignore[import-untyped]

from ._aggregation import Sampling
from ._climate_io import raster_views, raster_window_snapshot
from ._dataset_schema import (
    ExecutionCounts,
    ExecutionReport,
    ReleaseReference,
    ResourceIdentity,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
)
from ._declarations import Calendar, DataSpec, Period, SeriesSpec, Variable
from ._errors import CapabilityError
from ._evidence import (
    EvidenceKind,
    EvidenceMetadata,
    evidence_arrow_schema,
    evidence_table,
)
from ._evidence_recipes import Layer, Observations
from ._graph import ExecutionScope, Node, NodeStream
from ._inspection import (
    Bounds,
    Capability,
    Inspection,
    Operation,
    Requirement,
    SeriesReference,
)
from ._observed_decoders import array, number, obj, text
from ._observed_rasters import raster_records
from ._recipes import _key
from ._selection import (
    ResolvedSelection,
    SelectionState,
    _validate_static_selection,
    selected_metadata,
    support_intersects,
)
from ._source_io import (
    SourceNode,
    check_release,
    fetch,
    fetch_json,
    inventory_payload_pin,
    inventory_release,
    native_data,
    payload_selection,
    release_snapshot,
    selected_series_period,
    selected_source_time,
)

_CHIRPS = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/cogs/"
_FTW = "https://data.source.coop/ftw/global-data-2e/vector/catalog.json"
M = TypeVar("M")


def _initialize(value: object, node: Node[M], description: str) -> None:
    for name, argument in (
        ("_base", node),
        ("_selection", SelectionState()),
        ("_operations", (Operation(Stage.ACQUISITION, description),)),
        ("_path", None),
        ("_metadata", None),
    ):
        object.__setattr__(value, name, argument)


def _release(value: ReleaseReference | None) -> None:
    if value is not None and not isinstance(value, ReleaseReference):
        raise TypeError("release must be ReleaseReference or None")


def _terms(provider: str) -> tuple[str, str]:
    if provider == "chirps":
        return (
            "CC-BY-4.0",
            (
                "Climate Hazards Center, UC Santa Barbara; CHIRPS v3 "
                "monthly final precipitation"
            ),
        )
    return (
        "CC-BY-4.0",
        (
            "Fields of the World, second edition; predicted "
            "agricultural fields from Sentinel-2 imagery"
        ),
    )


def _variables(provider: str) -> tuple[Variable, ...]:
    if provider == "chirps":
        return (
            Variable(
                "chirps:monthly_precipitation",
                measurement=cf.PrecipitationAmount(
                    unit=cf.LengthUnit.MILLIMETRE,
                    accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.MONTH),
                ),
                label="Monthly precipitation amount",
            ),
        )
    return (
        Variable(
            "ftw:field_boundary",
            measurement=cf.Measurement(
                quantity="predicted_agricultural_field_boundary", unit="1"
            ),
            label="Predicted field boundary",
        ),
        Variable(
            "ftw:field_area",
            measurement=cf.Measurement(quantity="area", unit="m2"),
            label="Published field area",
        ),
        Variable(
            "ftw:field_perimeter",
            measurement=cf.Measurement(quantity="length", unit="m"),
            label="Published field perimeter",
        ),
        Variable(
            "ftw:model_score",
            measurement=cf.Measurement(quantity="model_field_probability", unit="%"),
            label="Rounded mean field probability inside the predicted boundary",
        ),
    )


def _empty(state: SelectionState) -> bool:
    return any(
        value == ()
        for value in (
            state.variables,
            state.periods,
            state.series,
            state.stations,
            state.scenarios,
        )
    )


def _source(
    provider: str,
    resources: list[ResourceIdentity],
    requested: ReleaseReference | None,
    *,
    complete: bool = True,
) -> SourceIdentity | None:
    if not resources:
        return None
    release = None
    if complete:
        release = release_snapshot(resources, id=f"{provider}:selected-source-snapshot")
        check_release(release, requested)
    return SourceIdentity(
        provider=provider,
        product="monthly_final_precipitation"
        if provider == "chirps"
        else "global_fields_second_edition",
        release=release,
        resources=tuple(resources),
    )


def _metadata(
    provider: str,
    state: SelectionState,
    periods: tuple[Period, ...],
    source: SourceIdentity | None,
) -> EvidenceMetadata:
    variables = _variables(provider)
    spec = DataSpec(
        variables, series=[SeriesSpec(v, period=p) for v in variables for p in periods]
    )
    return EvidenceMetadata(
        kind=EvidenceKind.OBSERVATIONS if provider == "chirps" else EvidenceKind.LAYER,
        spec=spec,
        spatial=SpatialContract(supports=(SupportKind.POLYGON,), index_resolution=9),
        source=source,
        report=ExecutionReport(
            False, None, source_identities=() if source is None else (source,)
        ),
        sampling=Sampling(
            calendar=Calendar.GREGORIAN, step=cf.Duration(1, unit=cf.TimeUnit.MONTH)
        )
        if provider == "chirps"
        else None,
    )


def _inspection(
    metadata: EvidenceMetadata, requirements: tuple[Requirement, ...] = ()
) -> Inspection:
    return Inspection(
        schema=evidence_arrow_schema(metadata),
        series=tuple(
            SeriesReference(
                s.variable,
                period=s.period,
                release=None if metadata.source is None else metadata.source.release,
            )
            for s in metadata.spec.series
        ),
        releases=()
        if metadata.source is None or metadata.source.release is None
        else (metadata.source.release,),
        capabilities=(
            Capability.INSPECT,
            Capability.READ,
            Capability.BATCHES,
            Capability.WRITE,
            *(
                (Capability.AGGREGATE,)
                if metadata.kind is EvidenceKind.OBSERVATIONS
                and metadata.sampling is not None
                else ()
            ),
        ),
        requirements=requirements,
        source=metadata.source,
    )


class _CHIRPSNode(SourceNode[EvidenceMetadata]):
    def __init__(
        self,
        release: ReleaseReference | None = None,
        state: SelectionState = SelectionState(),
    ) -> None:
        self.release, self.state = release, state
        self.key = _key(
            "chirps_v3_monthly",
            None if release is None else release._document(),
            state.document(),
        )

    def selected(self, state: SelectionState) -> _CHIRPSNode:
        return _CHIRPSNode(self.release, state)

    def _variable_ids(self) -> frozenset[str]:
        return frozenset(v.id for v in _variables("chirps"))

    def requirements(self) -> tuple[Requirement, ...]:
        if _empty(self.state):
            return ()
        result = []
        if self.state.area is None:
            result.append(
                Requirement(
                    "area_required",
                    "Select the area whose actual source pixels should be read.",
                    "area",
                )
            )
        if selected_source_time(self.state) is None:
            result.append(
                Requirement(
                    "observation_time_required",
                    (
                        "Select published observation years with "
                        "time=Period.window(start, end)."
                    ),
                    "time",
                )
            )
        return tuple(result)

    def _science(
        self,
    ) -> tuple[EvidenceMetadata, ResolvedSelection, ReleaseReference | None]:
        state, release = payload_selection(self.state, self.release)
        window = selected_source_time(state)
        period = selected_series_period(state) or window

        def gregorian(value: Period) -> Period:
            if value.calendar is not None and value.calendar is not Calendar.GREGORIAN:
                raise ValueError("CHIRPS observation years use the Gregorian calendar")
            return Period.window(
                cast(int, value.start),
                cast(int, value.end),
                calendar=Calendar.GREGORIAN,
            )

        if not _empty(state) and period is not None and cast(int, period.start) < 1981:
            raise ValueError("CHIRPS v3 observations begin in 1981")
        if period is not None:
            period = gregorian(period)
        state = replace(
            state,
            time=None if window is None else gregorian(window),
            periods=None
            if state.periods is None
            else tuple(
                gregorian(p) if p.start is not None else p for p in state.periods
            ),
        )
        metadata = _metadata("chirps", state, () if period is None else (period,), None)
        selection = state.resolve(metadata.spec)
        return selected_metadata(metadata, selection), selection, release.exact_only()

    def inspection(self, scope: ExecutionScope) -> Inspection:
        metadata, _, _ = self._science()
        requirements = self.requirements()
        if not _empty(self.state):
            requirements = (
                *requirements,
                Requirement(
                    "snapshot_resolved_on_read",
                    "Native COG bytes are pinned and any requested release "
                    "is verified when read.",
                    "release",
                ),
            )
        return _inspection(metadata, requirements)

    def open(self, scope: ExecutionScope) -> NodeStream[EvidenceMetadata]:
        metadata, resolved, release = self._science()
        if _empty(self.state):
            return NodeStream(evidence_arrow_schema(metadata), metadata, iter(()))
        if requirements := self.requirements():
            raise CapabilityError(" ".join(v.message for v in requirements))
        window = resolved.time
        assert (
            window is not None and window.start is not None and window.end is not None
        )
        area = self.state.area
        assert area is not None
        if resolved.empty or area.north < -60 or area.south > 60:
            return NodeStream(evidence_arrow_schema(metadata), metadata, iter(()))
        period = metadata.spec.series[0].period
        assert period is not None
        years = range(window.start, window.end + 1)
        variable = metadata.spec.series[0].variable.id
        license, attribution = _terms("chirps")
        resources: list[ResourceIdentity] = []
        actual: SourceIdentity | None = None
        count = missing = 0

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal count, missing, actual
            for year in years:
                for month in range(1, 13):
                    uri = _CHIRPS + f"chirps-v3.0.{year:04d}.{month:02d}.cog"
                    bundle, resource = raster_window_snapshot(
                        scope, uri, area=area, license=license, attribution=attribution
                    )
                    resources.append(resource)
                    pending = []
                    for path in raster_views(scope, bundle):
                        for row in raster_records(
                            path,
                            provider="chirps:v3:monthly",
                            variable_id=variable,
                            period_id=period.id,
                            area=area,
                            time=f"{year:04d}-{month:02d}-01",
                            block_pixels=scope.runtime.batch_rows,
                        ):
                            if not resolved.matches_record(row):
                                continue
                            count += 1
                            missing += int(row["value"] is None)
                            pending.append(row)
                            if len(pending) >= scope.runtime.batch_rows:
                                yield from evidence_table(
                                    pending, metadata
                                ).to_batches()
                                pending.clear()
                    if pending:
                        yield from evidence_table(pending, metadata).to_batches()
            actual = _source("chirps", resources, release)

        def report() -> ExecutionReport:
            source = actual or _source("chirps", resources, None, complete=False)
            return ExecutionReport(
                False,
                None,
                counts=ExecutionCounts(
                    requested=count, produced=count, no_data=missing
                ),
                source_identities=() if source is None else (source,),
            )

        def final_metadata() -> EvidenceMetadata:
            outcome = report()
            source = outcome.source_identities[0] if outcome.source_identities else None
            return replace(metadata, source=source, report=outcome)

        return NodeStream(
            evidence_arrow_schema(metadata),
            metadata,
            batches(),
            report=report,
            final_metadata=final_metadata,
            verify_before_yield=release is not None,
        )


class CHIRPS(Observations):
    def __init__(self, *, release: ReleaseReference | None = None) -> None:
        _release(release)
        _initialize(
            self,
            _CHIRPSNode(release),
            "Read verified CHIRPS v3 monthly final precipitation pixels",
        )


@dataclass(frozen=True, slots=True)
class _FieldPart:
    path: Path
    period: Period
    collection: str


def _href(parent: str, value: object) -> str:
    result = urljoin(parent, text(value, "STAC resource URI"))
    base, target = urlsplit(_FTW), urlsplit(result)
    if (
        target.scheme != "https"
        or target.netloc != base.netloc
        or not target.path.startswith("/ftw/global-data-2e/")
    ):
        raise ValueError("FTW STAC resource escaped the published product")
    return result


def _bounds(value: object) -> Bounds:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("FTW item requires a four-coordinate WGS84 bbox")
    return Bounds(
        **{
            key: cast(float, number(v, "FTW bbox"))
            for key, v in zip(("west", "south", "east", "north"), value, strict=True)
        }
    )


def _overlaps(left: Bounds, right: Bounds) -> bool:
    if left.west > left.east:
        return _overlaps(
            Bounds(west=left.west, south=left.south, east=180, north=left.north), right
        ) or _overlaps(
            Bounds(west=-180, south=left.south, east=left.east, north=left.north), right
        )
    return (
        left.west <= right.east
        and left.east >= right.west
        and left.south <= right.north
        and left.north >= right.south
    )


class _FieldsNode(SourceNode[EvidenceMetadata]):
    def __init__(
        self,
        release: ReleaseReference | None = None,
        state: SelectionState = SelectionState(),
    ) -> None:
        self.release, self.state = release, state
        self.key = _key(
            "ftw_second_edition",
            None if release is None else release._document(),
            state.document(),
        )

    def selected(self, state: SelectionState) -> _FieldsNode:
        return _FieldsNode(self.release, state)

    def _variable_ids(self) -> frozenset[str]:
        return frozenset(v.id for v in _variables("ftw"))

    def requirements(self) -> tuple[Requirement, ...]:
        if self.state.area is not None or _empty(self.state):
            return ()
        return (
            Requirement(
                "area_required",
                "Select an area before reading predicted field boundaries.",
                "area",
            ),
        )

    def _catalogue(
        self, scope: ExecutionScope
    ) -> tuple[dict[int, str], list[ResourceIdentity]]:
        with native_data(scope=scope, source="ftw"):
            license, attribution = _terms("ftw")
            payload, resource = fetch_json(
                scope, _FTW, license=license, attribution=attribution
            )
            result = {}
            for value in array(
                obj(payload, "FTW vector catalogue").get("links"), "FTW catalogue links"
            ):
                link = obj(value, "FTW catalogue link")
                if link.get("rel") != "child":
                    continue
                href = _href(_FTW, link.get("href"))
                match = re.search(r"/(\d{4})/collection\.json$", href)
                if match is None:
                    raise ValueError(
                        "FTW vector catalogue child has no publication year"
                    )
                year = int(match[1])
                if year in result:
                    raise ValueError("FTW vector catalogue repeats a year")
                result[year] = href
            if not result:
                raise ValueError("FTW vector catalogue has no published years")
            return result, [resource]

    def _prepare(
        self, scope: ExecutionScope, *, inventory: bool = False, acquire: bool = True
    ) -> tuple[EvidenceMetadata, ResolvedSelection, tuple[_FieldPart, ...]]:
        state, requested = payload_selection(self.state, self.release)
        if _empty(state):
            _validate_static_selection(state, _variables("ftw"))
            metadata = _metadata("ftw", state, (), None)
            resolved = SelectionState().resolve(metadata.spec)
            return selected_metadata(metadata, resolved), resolved, ()
        catalog, resources = self._catalogue(scope)
        published = inventory_release(resources, id="ftw:selected-source-snapshot")
        pin = inventory_payload_pin(published, requested)
        periods = tuple(Period.window(y, y) for y in sorted(catalog))
        if not inventory and state.periods is None and state.series is None:
            periods = periods[-1:]
        metadata = _metadata("ftw", state, periods, None)
        resolved = state.resolve(metadata.spec)
        metadata = selected_metadata(metadata, resolved)
        selected = {
            cast(int, s.period.start)
            for s in resolved.spec.series
            if s.period is not None
        }
        if inventory:
            source = SourceIdentity(
                provider="ftw",
                product="global_fields_second_edition",
                release=published,
                resources=tuple(resources),
            )
            report = replace(metadata.report, source_identities=(source,))
            return replace(metadata, source=source, report=report), resolved, ()
        if resolved.area is None:
            raise CapabilityError(
                "Select an area before reading predicted field boundaries"
            )
        with native_data(scope=scope, source="ftw"):
            parts = []
            license, attribution = _terms("ftw")
            for year in sorted(selected):
                collection_uri = catalog[year]
                payload, resource = fetch_json(
                    scope, collection_uri, license=license, attribution=attribution
                )
                resources.append(resource)
                collection = obj(payload, "FTW yearly collection")
                collection_id = text(collection.get("id"), "FTW collection ID")
                for value in array(collection.get("links"), "FTW yearly links"):
                    link = obj(value, "FTW item link")
                    if link.get("rel") != "item":
                        continue
                    item_uri = _href(collection_uri, link.get("href"))
                    item_payload, item_resource = fetch_json(
                        scope, item_uri, license=license, attribution=attribution
                    )
                    resources.append(item_resource)
                    item = obj(item_payload, "FTW item")
                    if item.get("collection") != collection_id:
                        raise ValueError("FTW item collection differs from its parent")
                    if not _overlaps(resolved.area, _bounds(item.get("bbox"))):
                        continue
                    properties = obj(item.get("properties"), "FTW item properties")
                    if properties.get("proj:code") != "EPSG:4326":
                        raise ValueError("FTW item does not declare WGS84 geometry")
                    asset = obj(
                        obj(item.get("assets"), "FTW item assets").get("data"),
                        "FTW data asset",
                    )
                    checksum = text(asset.get("file:checksum"), "FTW asset multihash")
                    if re.fullmatch(r"1220[0-9a-f]{64}", checksum) is None:
                        raise ValueError(
                            "FTW data asset must publish a SHA256 multihash"
                        )
                    data_uri = _href(item_uri, asset.get("href"))
                    expected = "sha256:" + checksum[4:]
                    if not acquire:
                        resources.append(
                            ResourceIdentity(
                                data_uri,
                                expected,
                                license=license,
                                attribution=attribution,
                            )
                        )
                        continue
                    path, resource = fetch(
                        scope,
                        data_uri,
                        expected_sha256=expected,
                        license=license,
                        attribution=attribution,
                    )
                    resources.append(resource)
                    parts.append(
                        _FieldPart(
                            path,
                            Period.window(year, year),
                            collection_id,
                        )
                    )
            snapshot = _source("ftw", resources, pin)
            assert snapshot is not None
            report = replace(metadata.report, source_identities=(snapshot,))
            return (
                replace(metadata, source=snapshot, report=report),
                resolved,
                tuple(parts),
            )

    def inspection(self, scope: ExecutionScope) -> Inspection:
        metadata, _, _ = self._prepare(
            scope, inventory=self.state.area is None, acquire=False
        )
        requirements = (
            ()
            if self.state.area is not None or _empty(self.state)
            else (
                Requirement(
                    "area_required",
                    "Select the area whose predicted field boundaries should be read.",
                    "area",
                ),
            )
        )
        return _inspection(metadata, requirements)

    def open(self, scope: ExecutionScope) -> NodeStream[EvidenceMetadata]:
        metadata, resolved, parts = self._prepare(scope)
        selected_variables = {s.variable.id for s in resolved.spec.series}
        count = 0

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal count
            pending = []
            assert self.state.area is not None or not parts
            for part in parts:
                parquet = pq.ParquetFile(part.path)
                required = {
                    "id",
                    "collection",
                    "geometry",
                    "metrics:area",
                    "metrics:perimeter",
                    "score",
                    "determination:datetime",
                }
                if not required <= set(parquet.schema_arrow.names):
                    raise ValueError("FTW data asset is missing declared native fields")
                for batch in parquet.iter_batches(
                    batch_size=scope.runtime.batch_rows, columns=sorted(required)
                ):
                    for row in batch.to_pylist():
                        field_id = text(row["id"], "FTW field ID")
                        if row["collection"] != part.collection:
                            raise ValueError(
                                "FTW record collection differs from the STAC asset"
                            )
                        when = str(row["determination:datetime"])
                        if not when.startswith(str(part.period.start) + "-"):
                            raise ValueError(
                                "FTW record publication year differs "
                                "from its collection"
                            )
                        geometry = from_wkb(row["geometry"])
                        if (
                            geometry.geom_type not in ("Polygon", "MultiPolygon")
                            or geometry.is_empty
                            or not geometry.is_valid
                        ):
                            raise ValueError(
                                "FTW field boundary must be a valid nonempty polygon"
                            )
                        if not (
                            -180 <= geometry.bounds[0] <= geometry.bounds[2] <= 180
                            and -90 <= geometry.bounds[1] <= geometry.bounds[3] <= 90
                        ):
                            raise ValueError("FTW field boundary is outside WGS84")
                        wkb = cast(bytes, geometry.wkb)
                        centre = geometry.representative_point()
                        cell = h3.latlng_to_cell(centre.y, centre.x, 9)
                        assert self.state.area is not None
                        if not support_intersects(
                            self.state.area, geometry=wkb, cell_id=cell
                        ):
                            continue
                        values = {
                            "ftw:field_boundary": None,
                            "ftw:field_area": number(row["metrics:area"], "field area"),
                            "ftw:field_perimeter": number(
                                row["metrics:perimeter"], "field perimeter"
                            ),
                            "ftw:model_score": number(
                                row["score"], "field model score"
                            ),
                        }
                        if (
                            cast(float, values["ftw:field_area"]) < 0
                            or cast(float, values["ftw:field_perimeter"]) < 0
                            or not 0 <= cast(float, values["ftw:model_score"]) <= 100
                        ):
                            raise ValueError(
                                "FTW field metrics lie outside their published domains"
                            )
                        for variable, value in values.items():
                            if variable not in selected_variables:
                                continue
                            record = {
                                "variable_id": variable,
                                "scenario_id": None,
                                "period_id": part.period.id,
                                "support_id": (
                                    f"ftw:{part.collection}:{field_id}@"
                                    f"{hashlib.sha256(wkb).hexdigest()}"
                                ),
                                "cell_id": cell,
                                "geometry": wkb,
                                "value": value,
                                "label": field_id,
                                "failure": None,
                            }
                            if not resolved.matches_record(record):
                                continue
                            pending.append(record)
                            count += 1
                            if len(pending) >= scope.runtime.batch_rows:
                                yield from evidence_table(
                                    pending, metadata
                                ).to_batches()
                                pending = []
            if pending:
                yield from evidence_table(pending, metadata).to_batches()

        def report() -> ExecutionReport:
            return ExecutionReport(
                False,
                None,
                counts=ExecutionCounts(requested=count, produced=count),
                source_identities=() if metadata.source is None else (metadata.source,),
            )

        return NodeStream(
            evidence_arrow_schema(metadata), metadata, batches(), report=report
        )


class FieldsOfTheWorld(Layer):
    def __init__(self, *, release: ReleaseReference | None = None) -> None:
        _release(release)
        _initialize(
            self,
            _FieldsNode(release),
            (
                "Read verified Fields of the World second-edition predicted"
                " field boundaries and native metrics"
            ),
        )
