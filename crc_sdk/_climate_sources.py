"""Typed, source-native climate recipes using shared execution and acquisition."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, cast

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely.geometry import Point, shape  # type: ignore[import-untyped]

import crc_sdk

from ._climate_io import raster_quantile_rows, raster_views, raster_window_snapshot
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    ReleaseReference,
    ResourceIdentity,
    SourceIdentity,
    SourceUnavailable,
    SpatialContract,
    Stage,
    SupportKind,
)
from ._declarations import Calendar, DataSpec, Period, Probability, SeriesSpec, Variable
from ._errors import CapabilityError
from ._evidence import EvidenceKind, EvidenceMetadata, evidence_arrow_schema
from ._evidence_recipes import Observations
from ._graph import ExecutionScope, NodeStream
from ._inspection import (
    Capability,
    IndicatorReference,
    Inspection,
    Operation,
    Requirement,
    SeriesReference,
    StationReference,
)
from ._providers import eccc_idf as idf
from ._providers import jrc as flood
from ._recipes import Dataset, _checked_stream, _key
from ._runtime import _memory_bytes
from ._schema import DatasetRow, dataset_arrow_schema, rows_to_table
from ._selection import (
    ResolvedSelection,
    SelectionState,
    _validate_static_selection,
    participating_spec,
    selected_metadata,
)
from ._source_io import (
    ReleasePins,
    SourceNode,
    fetch,
    inventory_payload_pin,
    inventory_release,
    payload_selection,
    selected_series_period,
    selected_source_time,
    verify_payload,
)

MetadataT = TypeVar("MetadataT")
_IDFMetadata: TypeAlias = (
    DatasetMetadata | EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]]
)
_IDF_VERSION = "3.40"


def _initialize(recipe: object, node: SourceNode[MetadataT], description: str) -> None:
    for name, value in (
        ("_base", node),
        ("_selection", SelectionState()),
        ("_operations", (Operation(Stage.ACQUISITION, description),)),
        ("_path", None),
        ("_metadata", None),
    ):
        object.__setattr__(recipe, name, value)


def _release_key(value: ReleaseReference | None) -> object:
    return None if value is None else value._document()


def _requirements(*values: tuple[str, str]) -> tuple[Requirement, ...]:
    return tuple(
        Requirement("selection_required", message, parameter=name)
        for name, message in values
    )


def _report(
    count: int,
    source: SourceIdentity,
    *,
    skipped: int = 0,
    no_data: int = 0,
    diagnostics: tuple[ExecutionDiagnostic, ...] = (),
) -> ExecutionReport:
    return ExecutionReport(
        False,
        None,
        diagnostics,
        ExecutionCounts(
            requested=count + skipped, produced=count, skipped=skipped, no_data=no_data
        ),
        (source,),
    )


def _empty_requested(state: SelectionState) -> bool:
    value = state.document()
    return any(
        value.get(name) == []
        for name in ("variables", "scenarios", "periods", "series", "stations")
    )


def _stream_rows(
    scope: ExecutionScope,
    metadata: DatasetMetadata,
    produce: Any,
) -> NodeStream[DatasetMetadata]:
    current = metadata

    def batches() -> Iterator[pa.RecordBatch]:
        nonlocal current
        rows, source, skipped = produce()
        diagnostics = (
            ()
            if not skipped
            else (
                ExecutionDiagnostic(
                    "source_no_data",
                    cf.DiagnosticSeverity.WARNING,
                    Stage.DECODING,
                    f"Skipped {skipped} source supports without "
                    "at least two valid probability values.",
                    source=source.product,
                ),
            )
        )
        report = _report(len(rows), source, skipped=skipped, diagnostics=diagnostics)
        publication = metadata.source
        if (
            publication is not None
            and publication.release is not None
            and publication.release.id.startswith("inventory:")
            and publication.release != source.release
        ):
            report = replace(report, source_identities=(publication, source))
        current = replace(
            metadata,
            source=source,
            report=report,
        )
        table = rows_to_table(rows, current)
        yield from table.to_batches(max_chunksize=scope.runtime.batch_rows)

    checked = _checked_stream(scope, metadata, batches())
    checked.report = lambda: current.report
    checked.final_metadata = lambda: current
    return checked


class JRC(Dataset):
    """Published JRC flood depths, preserving their supplied probability knots."""

    __slots__ = ()

    class Product(Enum):
        EFAS = "efas"
        GLOFAS = "glofas"

    def __init__(
        self,
        *,
        product: JRC.Product | None = None,
        release: ReleaseReference | None = None,
    ) -> None:
        if product is not None and not isinstance(product, JRC.Product):
            raise TypeError("product must be JRC.Product")
        if release is not None and not isinstance(release, ReleaseReference):
            raise TypeError("release must be ReleaseReference")
        _initialize(
            self, _FloodNode(product, release), "Read published JRC flood quantiles"
        )


@dataclass(frozen=True, slots=True)
class _FloodNode(SourceNode[DatasetMetadata]):
    product: JRC.Product | None
    release: ReleaseReference | None
    state: SelectionState = SelectionState()

    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _key(
                "jrc-flood",
                None if self.product is None else self.product.value,
                _release_key(self.release),
                self.state.document(),
            ),
        )

    def requirements(self) -> tuple[Requirement, ...]:
        _, products = self._plan()
        return self._selection_requirements(products)

    def _selection_requirements(
        self, products: tuple[JRC.Product, ...]
    ) -> tuple[Requirement, ...]:
        if not products:
            return ()
        values = (
            ()
            if len(products) == 1
            else _requirements(
                ("product", "Choose JRC.Product.EFAS or JRC.Product.GLOFAS.")
            )
        )
        return (
            values
            if self.state.area is not None
            else values
            + _requirements(
                (
                    "area",
                    "Select bounded flood coverage with .select(area=Bounds(...)).",
                )
            )
        )

    def selected(self, state: SelectionState) -> _FloodNode:
        return replace(self, state=state)

    def _variable_ids(self) -> frozenset[str]:
        _, products = self._plan()
        return frozenset(_flood_variable(p).id for p in products)

    _publishes_payload_release = True

    def _configured_payload_pin(self) -> ReleaseReference | None:
        _, products = self._plan()
        return (
            self._requested_release(products[0]).payload if len(products) == 1 else None
        )

    def _available_products(self) -> tuple[JRC.Product, ...]:
        if self.product is not None:
            return (self.product,)
        if self.release is not None:
            version = self.release.id.removeprefix("inventory:")
            matches = tuple(
                p for p in JRC.Product if flood.JRC_DATASETS[p.value].version == version
            )
            if len(matches) == 1:
                return matches
        return tuple(JRC.Product)

    def _plan(self) -> tuple[ResolvedSelection, tuple[JRC.Product, ...]]:
        products = self._available_products()
        # Resolve independent selectors against the declared inventory before
        # using the intersection to decide which native products are needed.
        state = self.state
        if state.series is not None:
            state = replace(
                state, series=tuple(replace(ref, release=None) for ref in state.series)
            )
        selection = state.resolve(_flood_spec(products))
        if selection.empty:
            return selection, ()
        selected = {v.id for v in selection.spec.variables}
        return selection, tuple(
            p for p in products if _flood_variable(p).id in selected
        )

    def _requested_release(self, product: JRC.Product) -> ReleasePins:
        variable_id = _flood_variable(product).id
        state = replace(
            self.state,
            series=tuple(
                ref for ref in self.state.series or () if ref.variable.id == variable_id
            ),
        )
        _, requested = payload_selection(state, self.release)
        return requested

    def _discovery(
        self, scope: ExecutionScope, product: JRC.Product
    ) -> tuple[flood.JRCRasterDataset, SourceIdentity]:
        dataset = flood.JRC_DATASETS[product.value]
        path, resource = fetch(
            scope,
            dataset.readme_url,
            license="ODbL-1.0",
            attribution=(
                "Copernicus Emergency Management Service; "
                "European Commission Joint Research Centre"
            ),
        )
        text = path.read_text(encoding="utf-8")
        match = re.search(r"Dataset version\s+([0-9]+(?:\.[0-9]+)+)", text)
        if match is None or match[1] != dataset.version:
            raise CapabilityError(
                f"The JRC {product.value} publication does not match "
                f"the supported source layout {dataset.version}."
            )
        return dataset, SourceIdentity(
            "jrc",
            product.value,
            release=inventory_release((resource,), id=dataset.version),
            resources=(resource,),
        )

    def inspection(self, scope: ExecutionScope) -> Inspection:
        selection, products = self._plan()
        spec = DataSpec([], series=[]) if selection.empty else selection.spec
        discovered = []
        for product in products:
            requested = self._requested_release(product)
            _, identity = self._discovery(scope, product)
            assert identity.release is not None
            inventory_payload_pin(identity.release, requested)
            discovered.append(identity)
        sources = tuple(discovered)
        releases = tuple(
            release for source in sources if (release := source.release) is not None
        )
        source = sources[0] if len(sources) == 1 else None
        metadata = DatasetMetadata(
            spec=spec,
            spatial=SpatialContract(
                supports=(SupportKind.POLYGON,), index_resolution=9
            ),
            report=ExecutionReport(False, None, source_identities=sources),
            source=source,
        )
        return Inspection(
            schema=dataset_arrow_schema(metadata),
            series=tuple(
                SeriesReference(
                    s.variable, probability=s.probability, release=identity.release
                )
                for s, identity in zip(spec.series, sources, strict=True)
            ),
            releases=releases,
            indicators=tuple(
                IndicatorReference(
                    s.variable.id,
                    s.variable.label or s.variable.id,
                    s.variable.measurement,
                    s.probability,
                )
                for s in spec.series
            ),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
                Capability.FIT,
            ),
            requirements=self._selection_requirements(products),
            source=source,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        selection, products = self._plan()
        if selection.empty:
            empty = DatasetMetadata(
                spec=DataSpec([], series=[]),
                spatial=SpatialContract(
                    supports=(SupportKind.POLYGON,), index_resolution=9
                ),
                report=ExecutionReport(False, None),
            )
            return _checked_stream(scope, empty, iter(()))
        if len(products) != 1:
            raise CapabilityError(
                "Choose JRC(product=JRC.Product.EFAS) or "
                "JRC.Product.GLOFAS before reading."
            )
        (product,) = products
        release = self._requested_release(product)
        dataset, source = self._discovery(scope, product)
        assert source.release is not None
        variable = _flood_variable(product)
        metadata = DatasetMetadata(
            spec=selection.spec,
            spatial=SpatialContract(
                supports=(SupportKind.POLYGON,), index_resolution=9
            ),
            report=_report(0, source),
            source=source,
        )
        pin = inventory_payload_pin(source.release, release)
        metadata = selected_metadata(metadata, selection)
        if selection.area is None:
            raise CapabilityError(
                "JRC raster acquisition requires .select(area=Bounds(...))."
            )

        def produce() -> tuple[tuple[DatasetRow, ...], SourceIdentity, int]:
            assert selection.area is not None
            resources = list(source.resources)
            if dataset.layout == "continental":
                tiles: tuple[str | None, ...] = (None,)
            else:
                assert dataset.tile_index_url is not None
                index, identity = fetch(
                    scope,
                    dataset.tile_index_url,
                    license="ODbL-1.0",
                    attribution=source.resources[0].attribution,
                )
                resources.append(identity)
                features = json.loads(index.read_text())["features"]
                tiles = tuple(
                    f"ID{f['properties']['id']}_{f['properties']['name']}"
                    for f in features
                    if _area_intersects(selection.area, shape(f["geometry"]))
                )
            rows: list[DatasetRow] = []
            skipped = 0
            for tile in tiles:
                paths: dict[float, tuple[Path, ...]] = {}
                for return_period in dataset.available_return_periods:
                    bundle, identity = raster_window_snapshot(
                        scope,
                        dataset.raster_url(return_period, tile_id=tile),
                        area=selection.area,
                        license="ODbL-1.0",
                        attribution=source.resources[0].attribution,
                    )
                    paths[1 - 1 / return_period] = raster_views(scope, bundle)
                    resources.append(identity)
                decoded_rows: list[DatasetRow] = []
                missing = 0
                for view_index in range(len(next(iter(paths.values())))):
                    decoded, omitted = raster_quantile_rows(
                        {q: views[view_index] for q, views in paths.items()},
                        variable=variable,
                        area=selection.area,
                        support_prefix=f"jrc:{product.value}:{tile or 'continental'}",
                    )
                    decoded_rows.extend(decoded)
                    missing += omitted
                decoded = tuple(decoded_rows)
                rows.extend(decoded)
                skipped += missing
            actual = verify_payload(
                replace(source, resources=tuple(resources)),
                id=dataset.version,
                expected=pin,
            )
            return tuple(rows), actual, skipped

        return _stream_rows(scope, metadata, produce)


def _flood_variable(product: JRC.Product) -> Variable:
    return Variable(
        f"jrc:{product.value}:flood_depth",
        measurement=cf.DepthAboveGround(unit=cf.LengthUnit.METRE),
        label=f"{product.name} flood water depth",
    )


def _flood_spec(products: tuple[JRC.Product, ...]) -> DataSpec:
    variables = tuple(_flood_variable(product) for product in products)
    probability = Probability.annual(tail=cf.Tail.UPPER)
    return DataSpec(
        variables, series=[SeriesSpec(v, probability=probability) for v in variables]
    )


def _area_intersects(area: Any, geometry: Any) -> bool:
    from ._climate_io import _parts

    return any(piece.intersects(geometry) for piece in _parts(area))


class ECCCIDF(Dataset):
    """Published IDF rainfall rates; confidence bands remain separate evidence."""

    __slots__ = ()

    class Province(Enum):
        ALBERTA = "AB"
        BRITISH_COLUMBIA = "BC"
        MANITOBA = "MB"
        NEW_BRUNSWICK = "NB"
        NEWFOUNDLAND_AND_LABRADOR = "NL"
        NOVA_SCOTIA = "NS"
        NORTHWEST_TERRITORIES = "NT"
        NUNAVUT = "NU"
        ONTARIO = "ON"
        PRINCE_EDWARD_ISLAND = "PE"
        QUEBEC = "QC"
        SASKATCHEWAN = "SK"
        YUKON = "YT"

    def __init__(
        self,
        *,
        duration: cf.Duration | None = None,
        province: ECCCIDF.Province | None = None,
        release: ReleaseReference | None = None,
    ) -> None:
        if duration is not None and not isinstance(duration, cf.Duration):
            raise TypeError("duration must be framework Duration")
        if (
            duration is not None
            and _idf_minutes(duration) not in idf.DURATIONS.values()
        ):
            raise ValueError(
                "duration must be one of the published ECCC IDF durations "
                "(5 minutes through 24 hours)."
            )
        if province is not None and not isinstance(province, ECCCIDF.Province):
            raise TypeError("province must be ECCCIDF.Province")
        if release is not None and not isinstance(release, ReleaseReference):
            raise TypeError("release must be ReleaseReference")
        _initialize(
            self,
            _IDFNode(duration, None if province is None else province.value, release),
            "Read published ECCC rainfall IDF evidence",
        )

    def confidence_intervals(self) -> crc_sdk.Projections:
        from ._evidence_recipes import Projections

        node = cast(_IDFNode, self._base)
        return Projections._create(
            cast(
                SourceNode[EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]]],
                replace(node, confidence=True),
            ),
            selection=self._selection,
            operations=(
                *self._operations,
                Operation(
                    Stage.DECODING,
                    "Read the source's published 95% confidence intervals",
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _IDFNode(SourceNode[_IDFMetadata]):
    duration: cf.Duration | None
    province: str | None
    release: ReleaseReference | None
    state: SelectionState = SelectionState()
    confidence: bool = False

    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _key(
                "eccc-idf",
                None if self.duration is None else _idf_minutes(self.duration),
                self.province,
                _release_key(self.release),
                self.confidence,
                self.state.document(),
            ),
        )

    def requirements(self) -> tuple[Requirement, ...]:
        return (
            ()
            if self.province is not None or _empty_requested(self.state)
            else _requirements(("province", "Choose an exact ECCC province code."))
        )

    def selected(self, state: SelectionState) -> _IDFNode:
        return replace(self, state=state)

    def _variable_ids(self) -> frozenset[str]:
        return frozenset(v.id for v in self._variables())

    def _discovery(self, scope: ExecutionScope) -> SourceIdentity:
        path, resource = fetch(
            scope, idf.ROOT + "/", license=idf.LICENCE, attribution=idf.ATTRIBUTION
        )
        versions = re.findall(
            r'href=["\'](idf_v([0-9]+)-([0-9]+)_[^"\']+\.zip)["\']', path.read_text()
        )
        if not versions:
            raise ValueError(
                "ECCC IDF directory contains no supported current rainfall archive"
            )
        name, major, minor = max(versions, key=lambda v: (int(v[1]), int(v[2])))
        current = f"{major}.{minor}"
        if current != _IDF_VERSION:
            raise CapabilityError(
                f"ECCC IDF release {current} has not established "
                "the supported archive layout."
            )
        return SourceIdentity("eccc", "idf", resources=(resource,), member=name)

    def _variables(self) -> tuple[Variable, ...]:
        durations = (
            tuple(idf.DURATIONS.values())
            if self.duration is None
            else (int(_idf_minutes(self.duration)),)
        )
        return tuple(_idf_variable(value) for value in durations)

    def _spec(self, periods: tuple[Period | None, ...]) -> DataSpec:
        variables = self._variables()
        probability = (
            None if self.confidence else Probability.annual(tail=cf.Tail.UPPER)
        )
        return DataSpec(
            variables,
            series=[
                SeriesSpec(v, period=p, probability=probability)
                for v in variables
                for p in periods
            ],
        )

    def _metadata(
        self, spec: DataSpec, source: SourceIdentity | None = None
    ) -> _IDFMetadata:
        spatial = SpatialContract(supports=(SupportKind.POINT,), index_resolution=9)
        report = ExecutionReport(False, None) if source is None else _report(0, source)
        if self.confidence:
            return EvidenceMetadata(
                kind=EvidenceKind.PROJECTIONS,
                spec=spec,
                spatial=spatial,
                report=report,
                source=source,
            )
        return DatasetMetadata(spec=spec, spatial=spatial, report=report, source=source)

    @staticmethod
    def _schema(metadata: _IDFMetadata) -> pa.Schema:
        if isinstance(metadata, DatasetMetadata):
            return dataset_arrow_schema(metadata)
        return evidence_arrow_schema(metadata)

    def _station_records(
        self,
        scope: ExecutionScope,
        source: SourceIdentity,
        *,
        release: ReleaseReference | None,
    ) -> tuple[SourceIdentity, list[tuple[str, bytes, dict[str, Any]]]]:
        assert self.province is not None and source.member is not None
        path, resource = _idf_province_snapshot(
            scope,
            idf.ROOT + "/" + source.member,
            self.province,
            license=idf.LICENCE,
            attribution=idf.ATTRIBUTION,
        )
        source = verify_payload(
            replace(source, resources=(*source.resources, resource)),
            id=_IDF_VERSION,
            expected=release,
        )
        records = []
        seen = set()
        for member, raw in _idf_station_files(path, self.province):
            match = re.search(
                rf"_{self.province}_([A-Z0-9]{{7}})_.+\.txt$", member, re.I
            )
            if match is None:
                continue
            station = match[1]
            if station in seen:
                raise ValueError(f"ambiguous ECCC IDF station {station}")
            seen.add(station)
            decoded = idf.parse_station(raw, station, self.province)
            records.append((member, raw, decoded))
        return source, records

    def _prepare(
        self, scope: ExecutionScope
    ) -> tuple[
        _IDFMetadata,
        SourceIdentity,
        list[tuple[str, bytes, dict[str, Any]]],
    ]:
        state, pins = payload_selection(self.state, self.release)
        release = pins.exact_only()
        source, records = self._station_records(
            scope, self._discovery(scope), release=release
        )
        periods = tuple(dict.fromkeys(_idf_period(d) for _, _, d in records))
        spec = self._spec(periods)
        selection = state.resolve(spec, source=source)
        stations = selection.stations
        if stations is not None and (
            missing := set(stations) - {d["station"] for _, _, d in records}
        ):
            raise ValueError(
                f"ECCC {self.province} has no requested stations {sorted(missing)}"
            )
        selected_periods = set(selection.spec.periods)
        records = [
            (member, raw, d)
            for member, raw, d in records
            if _idf_period(d) in selected_periods
            and (stations is None or d["station"] in stations)
            and (
                state.area is None
                or _area_intersects(state.area, Point(d["longitude"], d["latitude"]))
            )
        ]
        in_scope = {_idf_period(d) for _, _, d in records}
        selected = selected_metadata(self._metadata(spec, source), selection)
        spec = participating_spec(
            spec, [s for s in selection.spec.series if s.period in in_scope]
        )
        return replace(selected, spec=spec), source, records

    def inspection(self, scope: ExecutionScope) -> Inspection:
        stations: tuple[StationReference, ...] = ()
        if _empty_requested(self.state):
            metadata = self._metadata(DataSpec([], series=[]))
        elif self.province is None:
            # Station records supply the exact period of each series.
            indicators = replace(self.state, periods=None, series=None)
            declared = indicators.resolve(self._spec((None,))).spec.variables
            metadata = self._metadata(DataSpec(declared, series=[]))
        else:
            metadata, _, records = self._prepare(scope)
            stations = tuple(
                StationReference(
                    d["station"],
                    d["name"],
                    d["longitude"],
                    d["latitude"],
                    _idf_period(d),
                )
                for _, _, d in records
            )
        source = metadata.source
        release = None if source is None else source.release
        probability = (
            None if self.confidence else Probability.annual(tail=cf.Tail.UPPER)
        )
        return Inspection(
            schema=self._schema(metadata),
            series=tuple(
                SeriesReference(
                    s.variable,
                    period=s.period,
                    release=release,
                    probability=s.probability,
                )
                for s in metadata.spec.series
            ),
            indicators=tuple(
                IndicatorReference(v.id, v.label or v.id, v.measurement, probability)
                for v in metadata.spec.variables
            ),
            releases=() if release is None else (release,),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
                *((Capability.FIT,) if not self.confidence else ()),
            ),
            requirements=self.requirements(),
            source=source,
            stations=stations,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[_IDFMetadata]:
        from ._evidence import evidence_table

        if _empty_requested(self.state):
            metadata = self._metadata(DataSpec([], series=[]))
            return NodeStream(self._schema(metadata), metadata, iter(()))
        if self.province is None:
            raise CapabilityError("Reading ECCC IDF requires ECCCIDF(province=...).")
        metadata, source, station_records = self._prepare(scope)
        schema = self._schema(metadata)
        if not station_records:
            return NodeStream(schema, metadata, iter(()))
        current = metadata
        series = {
            (s.variable.id, None if s.period is None else s.period.id)
            for s in metadata.spec.series
        }

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal current
            assert source.member is not None and self.province is not None
            archive_resource = source.resources[-1]
            resources = list(source.resources)
            rows: list[DatasetRow] = []
            records: list[dict[str, object]] = []
            no_data = 0
            for member, raw, decoded in station_records:
                station = decoded["station"]
                geometry = Point(decoded["longitude"], decoded["latitude"]).wkb
                cell = int(
                    h3.latlng_to_cell(decoded["latitude"], decoded["longitude"], 9)
                )
                resources.append(
                    ResourceIdentity(
                        archive_resource.uri + "!" + member,
                        "sha256:" + hashlib.sha256(raw).hexdigest(),
                        idf.LICENCE,
                        idf.ATTRIBUTION,
                        archive_resource.retrieved_at,
                    )
                )
                for curve in decoded["curves"]:
                    variable = _idf_variable(curve["duration_minutes"])
                    period = _idf_period(decoded)
                    if (variable.id, period.id) not in series:
                        continue
                    if curve["intensity"] is None:
                        no_data += 1
                        unavailable = SourceUnavailable(
                            "source_missing",
                            f"ECCC publishes no {curve['duration_minutes']}-minute "
                            f"estimate for station {station}.",
                        )
                        if self.confidence:
                            records.extend(
                                {
                                    "variable_id": variable.id,
                                    "scenario_id": None,
                                    "period_id": period.id,
                                    "support_id": station,
                                    "cell_id": cell,
                                    "geometry": geometry,
                                    "coordinate": {
                                        "kind": "return_period",
                                        "years": return_period,
                                    },
                                    "probability": None,
                                    "value": None,
                                    "interval": None,
                                    "failure": {
                                        "code": unavailable.code,
                                        "message": unavailable.message,
                                    },
                                }
                                for return_period in decoded["return_periods"]
                            )
                            no_data += len(decoded["return_periods"]) - 1
                        else:
                            rows.append(
                                DatasetRow(
                                    variable_id=variable.id,
                                    period_id=period.id,
                                    support_id=station,
                                    cell_id=cell,
                                    geometry=geometry,
                                    failure=unavailable,
                                )
                            )
                        continue
                    if self.confidence:
                        for return_period, value, width in zip(
                            decoded["return_periods"],
                            curve["intensity"],
                            curve["confidence_half_width"],
                            strict=True,
                        ):
                            records.append(
                                {
                                    "variable_id": variable.id,
                                    "scenario_id": None,
                                    "period_id": period.id,
                                    "support_id": station,
                                    "cell_id": cell,
                                    "geometry": geometry,
                                    "coordinate": {
                                        "kind": "return_period",
                                        "years": return_period,
                                    },
                                    "probability": None,
                                    "value": value,
                                    "interval": {
                                        "confidence": 0.95,
                                        "lower": value - width,
                                        "upper": value + width,
                                    },
                                    "failure": None,
                                }
                            )
                    else:
                        rows.append(
                            DatasetRow(
                                variable_id=variable.id,
                                period_id=period.id,
                                support_id=station,
                                cell_id=cell,
                                geometry=geometry,
                                curve=cf.QuantileCurve(
                                    [1 - 1 / t for t in decoded["return_periods"]],
                                    curve["intensity"],
                                ),
                            )
                        )
            actual_source = replace(source, resources=tuple(resources))
            current = replace(
                metadata,
                source=actual_source,
                report=_report(
                    len(records) if self.confidence else len(rows),
                    actual_source,
                    no_data=no_data,
                ),
            )
            table = (
                evidence_table(
                    records,
                    cast(EvidenceMetadata[Literal[EvidenceKind.PROJECTIONS]], current),
                )
                if self.confidence
                else rows_to_table(rows, cast(DatasetMetadata, current))
            )
            yield from table.to_batches(max_chunksize=scope.runtime.batch_rows)

        return NodeStream(
            schema,
            metadata,
            batches(),
            report=lambda: current.report,
            final_metadata=lambda: current,
        )


def _idf_variable(minutes: int) -> Variable:
    return Variable(
        f"eccc_idf:rainfall_rate:{minutes}min",
        measurement=cf.PrecipitationRate(
            unit=cf.PrecipitationRateUnit.MILLIMETRES_PER_HOUR,
            accumulation_duration=cf.Duration(minutes * 60, unit=cf.TimeUnit.SECOND),
            aggregation=cf.Aggregation.MAXIMUM,
        ),
        label=f"Annual maximum {minutes}-minute rainfall rate",
    )


def _idf_station_files(path: Path, province: str) -> Iterator[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as outer:
        candidates = [n for n in outer.namelist() if Path(n).name == province + ".zip"]
        if len(candidates) != 1:
            raise ValueError(f"ECCC current archive has no unique {province} archive")
        with outer.open(candidates[0]) as source:
            raw = source.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name in sorted(archive.namelist()):
                if re.search(rf"_{province}_[A-Z0-9]{{7}}_.+\.txt$", name, re.I):
                    yield candidates[0] + "!" + name, archive.read(name)


class EDO(Observations):
    """Native dekadal European Drought Observatory observations."""

    __slots__ = ()

    class Product(Enum):
        SOIL_MOISTURE_INDEX = "soil_moisture_index"

    def __init__(
        self,
        *,
        product: EDO.Product | None = None,
        release: ReleaseReference | None = None,
    ) -> None:
        if product is not None and not isinstance(product, EDO.Product):
            raise TypeError("product must be EDO.Product")
        if release is not None and not isinstance(release, ReleaseReference):
            raise TypeError("release must be ReleaseReference")
        _initialize(
            self,
            _EDONode(product, release),
            "Read native EDO soil moisture observations",
        )


@dataclass(frozen=True, slots=True)
class _EDONode(SourceNode[EvidenceMetadata]):
    product: EDO.Product | None
    release: ReleaseReference | None
    state: SelectionState = SelectionState()
    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _key(
                "edo-observations",
                None if self.product is None else self.product.value,
                _release_key(self.release),
                self.state.document(),
            ),
        )

    def requirements(self) -> tuple[Requirement, ...]:
        return (
            ()
            if self.state.area is not None or _empty_requested(self.state)
            else _requirements(
                ("area", "Select bounded EDO coverage with .select(area=Bounds(...)).")
            )
        )

    def selected(self, state: SelectionState) -> _EDONode:
        return replace(self, state=state)

    def _variable_ids(self) -> frozenset[str]:
        return frozenset((_edo_variable().id,))

    def _requested_selection(self) -> tuple[SelectionState, ReleasePins]:
        if self.state.stations:
            raise CapabilityError(
                "EDO uses a grid. Select coverage with .select(area=Bounds(...))."
            )
        return payload_selection(self.state, self.release)

    def _discovery(
        self, scope: ExecutionScope
    ) -> tuple[SourceIdentity, dict[int, str]]:
        from ._providers.jrc_edo import SMI

        path, resource = fetch(
            scope,
            SMI.base_url + "/",
            license="CC-BY-4.0",
            attribution="Copernicus European Drought Observatory; "
            "European Commission Joint Research Centre",
        )
        files = re.findall(
            r"href=[\"\'](sminx_m_eul_(\d{4})\d{4}_\d{8}_t\.nc)[\"\']", path.read_text()
        )
        years: dict[int, str] = {}
        for filename, year in files:
            if int(year) in years and years[int(year)] != filename:
                raise ValueError(
                    "EDO source publishes ambiguous resources for one year"
                )
            years[int(year)] = filename
        if not years:
            raise CapabilityError(
                "EDO directory contains no supported native soil moisture resources"
            )
        return SourceIdentity(
            "edo",
            "soil_moisture_index",
            release=inventory_release((resource,), id=SMI.version),
            resources=(resource,),
        ), years

    def _metadata(
        self, source: SourceIdentity, years: dict[int, str]
    ) -> EvidenceMetadata:
        from ._climate_sampling import _period
        from ._declarations import Calendar
        from ._evidence import EvidenceKind

        period = _period(
            selected_series_period(self.state) or selected_source_time(self.state),
            min(years),
            max(years),
            Calendar.PROLEPTIC_GREGORIAN,
        )
        variable = _edo_variable()
        return EvidenceMetadata(
            kind=EvidenceKind.OBSERVATIONS,
            spec=DataSpec([variable], series=[SeriesSpec(variable, period=period)]),
            spatial=SpatialContract(
                supports=(SupportKind.POLYGON,), index_resolution=9
            ),
            report=_report(0, source),
            source=source,
        )

    def inspection(self, scope: ExecutionScope) -> Inspection:
        capabilities = (
            Capability.INSPECT,
            Capability.READ,
            Capability.BATCHES,
            Capability.WRITE,
        )
        state, release = self._requested_selection()
        if _empty_requested(state):
            _validate_static_selection(state, (_edo_variable(),))
            return Inspection(
                schema=evidence_arrow_schema(self._empty_metadata()),
                capabilities=capabilities,
            )

        source, years = self._discovery(scope)
        assert source.release is not None
        inventory_payload_pin(source.release, release)
        metadata = self._metadata(source, years)
        selection = replace(state, time=selected_source_time(state)).resolve(
            metadata.spec, source=source
        )
        metadata = selected_metadata(metadata, selection)
        return Inspection(
            schema=evidence_arrow_schema(metadata),
            series=tuple(
                SeriesReference(s.variable, period=s.period, release=source.release)
                for s in metadata.spec.series
            ),
            releases=(source.release,),
            indicators=(
                IndicatorReference(
                    _edo_variable().id,
                    "Soil moisture index",
                    _edo_variable().measurement,
                ),
            ),
            requirements=self.requirements(),
            capabilities=capabilities,
            source=source,
        )

    @staticmethod
    def _empty_metadata() -> EvidenceMetadata:
        return EvidenceMetadata(
            kind=EvidenceKind.OBSERVATIONS,
            spec=DataSpec([], series=[]),
            spatial=SpatialContract(
                supports=(SupportKind.POLYGON,), index_resolution=9
            ),
            report=ExecutionReport(False, None),
        )

    def open(self, scope: ExecutionScope) -> NodeStream[EvidenceMetadata]:
        import numpy as np

        from ._climate_grid import (
            GridReadPlan,
            calendar_name,
            cf_calendar,
            grid_cells,
            grid_values,
        )
        from ._climate_io import netcdf_window_snapshot
        from ._declarations import Calendar
        from ._evidence import evidence_table
        from ._providers.cmip6 import decode_time_ordinals
        from ._providers.jrc_edo import SMI

        state, release = self._requested_selection()
        if _empty_requested(state):
            _validate_static_selection(state, (_edo_variable(),))
            empty = self._empty_metadata()
            return NodeStream(evidence_arrow_schema(empty), empty, iter(()))
        if self.state.area is None:
            raise CapabilityError("EDO acquisition requires .select(area=Bounds(...)).")
        source, years = self._discovery(scope)
        assert source.release is not None
        metadata = self._metadata(source, years)
        pin = inventory_payload_pin(source.release, release)
        selection = replace(state, time=selected_source_time(state)).resolve(
            metadata.spec, source=source
        )
        metadata = selected_metadata(metadata, selection)
        current = metadata
        period = metadata.spec.series[0].period if metadata.spec.series else None
        variable = _edo_variable().id

        def with_report(
            actual: SourceIdentity, count: int, missing: int
        ) -> EvidenceMetadata:
            return replace(
                metadata,
                source=actual,
                report=ExecutionReport(
                    False,
                    None,
                    counts=ExecutionCounts(
                        requested=count, produced=count, no_data=missing
                    ),
                    source_identities=(actual,)
                    if actual.release == source.release
                    else (source, actual),
                ),
            )

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal current
            resources = list(source.resources)
            count = missing = 0
            if not selection.empty:
                assert selection.area is not None
                assert period is not None
                window = selection.time or period
                assert window.start is not None and window.end is not None
                for year, filename in sorted(years.items()):
                    if not window.start <= year <= window.end:
                        continue
                    group, resource = netcdf_window_snapshot(
                        scope,
                        SMI.base_url + "/" + filename,
                        area=selection.area,
                        variable=SMI.variable,
                        license="CC-BY-4.0",
                        attribution=source.resources[0].attribution,
                    )
                    resources.append(resource)
                    calendar = calendar_name(group["time"].attrs.get("calendar"))
                    if calendar not in {
                        Calendar.GREGORIAN,
                        Calendar.PROLEPTIC_GREGORIAN,
                    }:
                        raise ValueError(
                            "EDO native calendar contradicts its Gregorian source"
                        )
                    cal = cf_calendar(calendar)
                    times = decode_time_ordinals(
                        np.asarray(group["time"][:]),
                        str(group["time"].attrs.get("units", "")),
                        cal,
                    )
                    if np.any(np.diff(times) <= 0):
                        raise ValueError(
                            "EDO source dates must be unique and increasing"
                        )
                    start, stop = np.searchsorted(
                        times,
                        [
                            cal.to_ordinal(window.start, 1, 1),
                            cal.to_ordinal(window.end + 1, 1, 1),
                        ],
                    )
                    cells = grid_cells(
                        group, lat_name="lat", lon_name="lon", area=selection.area
                    )
                    values = grid_values(
                        group[SMI.variable],
                        time_name="time",
                        lat_name="lat",
                        lon_name="lon",
                        start=int(start),
                        stop=int(stop),
                        plan=GridReadPlan.from_cells(cells),
                        memory_limit=_memory_bytes(scope.runtime.memory_limit),
                    )
                    records: list[dict[str, object]] = []
                    actual = replace(source, resources=tuple(resources))
                    for ti, ordinal in enumerate(times[start:stop]):
                        # Native dates remain observations. No annual law, fixed
                        # ten-day interval, or probability convention is invented.
                        date = np.datetime64(int(ordinal), "D")
                        for ci, cell in enumerate(cells):
                            value = float(values[ti, ci])
                            valid = np.isfinite(value)
                            if valid and not 0 <= value <= 1:
                                raise ValueError(
                                    "EDO soil moisture index must be within [0, 1]"
                                )
                            records.append(
                                {
                                    "variable_id": variable,
                                    "scenario_id": None,
                                    "period_id": period.id,
                                    "support_id": (
                                        f"edo:{cell.source_row}:{cell.source_column}"
                                    ),
                                    "cell_id": cell.cell_id,
                                    "geometry": cell.geometry,
                                    "time": str(date),
                                    "value": value if valid else None,
                                    "failure": None
                                    if valid
                                    else {
                                        "code": "source_missing",
                                        "message": "EDO masked this observation.",
                                    },
                                    "coverage": None,
                                    "group_id": None,
                                }
                            )
                            count += 1
                            missing += not valid
                            if len(records) >= scope.runtime.batch_rows:
                                current = with_report(actual, count, missing)
                                yield from evidence_table(records, current).to_batches()
                                records.clear()
                    if records:
                        current = with_report(actual, count, missing)
                        yield from evidence_table(records, current).to_batches()
            actual = verify_payload(
                replace(source, resources=tuple(resources)),
                id=SMI.version,
                expected=pin,
            )
            current = with_report(actual, count, missing)

        return NodeStream(
            evidence_arrow_schema(metadata),
            metadata,
            batches(),
            report=lambda: current.report,
            final_metadata=lambda: current,
            verify_before_yield=pin is not None,
        )


def _edo_variable() -> Variable:
    return Variable(
        "edo:soil_moisture_index",
        measurement=cf.Measurement(
            quantity="soil_moisture_index", unit=cf.RatioUnit.FRACTION
        ),
        label="Soil moisture index",
    )


def _idf_province_snapshot(
    scope: ExecutionScope, uri: str, province: str, *, license: str, attribution: str
) -> tuple[Path, ResourceIdentity]:
    import urllib.parse
    import urllib.request

    from ._climate_io import _bundle, _json_bytes, _RangeFile, _validator
    from ._source_io import fetch_custom

    def acquire(destination: Path) -> None:
        before = _validator(uri)
        parsed = urllib.parse.urlsplit(uri)
        raw: Any = (
            Path(urllib.request.url2pathname(parsed.path))
            if parsed.scheme in {"", "file"}
            else _RangeFile(uri, before)
        )
        with zipfile.ZipFile(raw) as archive:
            members = [
                name
                for name in archive.namelist()
                if Path(name).name == province + ".zip"
            ]
            if len(members) != 1:
                raise ValueError(
                    "ECCC archive does not contain exactly one requested province"
                )
            payload = archive.read(members[0])
        if before != _validator(uri):
            raise OSError("ECCC archive changed while acquiring the province")
        _bundle(
            destination,
            {
                province + ".zip": payload,
                "metadata.json": _json_bytes(
                    {"source_uri": uri, "member": members[0], "validator": before}
                ),
            },
        )

    return fetch_custom(
        scope,
        uri + "#province=" + province,
        acquire,
        license=license,
        attribution=attribution,
    )


def _idf_period(record: dict[str, Any]) -> Period:
    return Period.window(
        record["start_year"], record["end_year"], calendar=Calendar.GREGORIAN
    )


def _idf_minutes(duration: cf.Duration) -> float:
    factors = {cf.TimeUnit.SECOND: 1 / 60, cf.TimeUnit.HOUR: 60, cf.TimeUnit.DAY: 1440}
    if duration.unit not in factors:
        raise ValueError("ECCC IDF durations require seconds, hours, or days")
    return duration.value * factors[duration.unit]
