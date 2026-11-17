"""USDA CroplandCROS categorical pixels from its published image service."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlencode

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]

from ._dataset_schema import (
    ExecutionCounts,
    ExecutionReport,
    ReleaseReference,
    ResourceIdentity,
    SourceIdentity,
    SpatialContract,
    SupportKind,
)
from ._declarations import DataSpec, Period, SeriesSpec, Variable
from ._errors import CapabilityError
from ._evidence import (
    EvidenceKind,
    EvidenceMetadata,
    evidence_arrow_schema,
    evidence_table,
)
from ._evidence_recipes import Layer
from ._graph import ExecutionScope, NodeStream
from ._inspection import Capability, Inspection, Requirement, SeriesReference
from ._layer_sources import _initialize
from ._observed_decoders import array, number, obj
from ._observed_rasters import raster_records
from ._recipes import _key
from ._selection import SelectionState, selected_metadata
from ._source_io import (
    SourceNode,
    check_release,
    fetch,
    fetch_json,
    native_data,
    release_snapshot,
)

_SERVICE = "https://pdi.scinet.usda.gov/image/rest/services/CDL_WM/ImageServer"
_LICENSE = "CC0-1.0"
_ATTRIBUTION = "USDA National Agricultural Statistics Service; Cropland Data Layer"
_VARIABLE = Variable(
    "usda:cropland_class",
    measurement=cf.Measurement(quantity="cropland_class", unit="1"),
    label="Cropland Data Layer class code",
)
_LayerMetadata = EvidenceMetadata[Literal[EvidenceKind.LAYER]]


def _uri(operation: str = "", **parameters: object) -> str:
    return _SERVICE + operation + "?" + urlencode(parameters)


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    result = number(value, name)
    if result is None or result != int(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be an {'positive ' if positive else ''}integer")
    return int(result)


def _empty(state: SelectionState) -> bool:
    return state.variables == () or state.periods == () or state.series == ()


def _metadata(
    periods: tuple[Period, ...], source: SourceIdentity | None = None
) -> _LayerMetadata:
    return EvidenceMetadata(
        kind=EvidenceKind.LAYER,
        spec=DataSpec(
            [_VARIABLE], series=[SeriesSpec(_VARIABLE, period=p) for p in periods]
        ),
        spatial=SpatialContract(supports=(SupportKind.POLYGON,), index_resolution=9),
        report=ExecutionReport(
            False, None, source_identities=() if source is None else (source,)
        ),
        source=source,
    )


@dataclass(frozen=True, slots=True)
class _Image:
    id: int
    year: int
    pixel_size: float


@dataclass(frozen=True, slots=True)
class _Part:
    path: Path
    period: Period
    width: int
    height: int
    bounds: tuple[float, float, float, float]


class _CroplandNode(SourceNode[_LayerMetadata]):
    def __init__(
        self,
        release: ReleaseReference | None = None,
        state: SelectionState = SelectionState(),
    ) -> None:
        self.release, self.state = release, state
        self.key = _key(
            "usda_croplandcros",
            None if release is None else release._document(),
            state.document(),
        )

    def selected(self, state: SelectionState) -> _CroplandNode:
        return _CroplandNode(self.release, state)

    def requirements(self) -> tuple[Requirement, ...]:
        if self.state.area is not None or _empty(self.state):
            return ()
        return (
            Requirement(
                "area_required",
                "Select the area whose published cropland pixels should be read.",
                "area",
            ),
        )

    @staticmethod
    def _json(
        scope: ExecutionScope, uri: str
    ) -> tuple[dict[str, object], ResourceIdentity]:
        with native_data(source="usda", scope=scope):
            payload, resource = fetch_json(
                scope, uri, license=_LICENSE, attribution=_ATTRIBUTION
            )
            result = obj(payload, "USDA image service response")
            if "error" in result:
                raise ValueError(
                    f"USDA image service rejected the request: {result['error']}"
                )
            return result, resource

    def _catalogue(
        self, scope: ExecutionScope
    ) -> tuple[dict[str, object], tuple[_Image, ...], list[ResourceIdentity]]:
        with native_data(source="usda", scope=scope):
            return self._native_catalogue(scope)

    def _native_catalogue(
        self, scope: ExecutionScope
    ) -> tuple[dict[str, object], tuple[_Image, ...], list[ResourceIdentity]]:
        service, first = self._json(scope, _uri(f="json"))
        if (
            service.get("serviceDataType") != "esriImageServiceDataTypeThematic"
            or service.get("bandCount") != 1
            or service.get("pixelType") != "U8"
        ):
            raise ValueError("USDA service must publish one categorical U8 band")
        reference = obj(service.get("spatialReference"), "USDA service CRS")
        if reference.get("latestWkid", reference.get("wkid")) != 3857:
            raise ValueError("USDA categorical service must declare EPSG:3857")
        records, second = self._json(
            scope,
            _uri(
                "/query",
                where="Category=1",
                outFields="OBJECTID,Name,Year,LowPS",
                returnGeometry="false",
                f="json",
            ),
        )
        if records.get("exceededTransferLimit") is True:
            raise ValueError(
                "USDA image inventory exceeded its published response limit"
            )
        images = []
        for feature in array(records.get("features"), "USDA raster inventory"):
            values = obj(
                obj(feature, "USDA raster item").get("attributes"), "USDA attributes"
            )
            pixel = number(values.get("LowPS"), "USDA source pixel size")
            if pixel is None or pixel <= 0:
                raise ValueError("USDA source pixel size must be positive")
            images.append(
                _Image(
                    _integer(values.get("OBJECTID"), "USDA raster ID", positive=True),
                    _integer(values.get("Year"), "USDA crop year", positive=True),
                    pixel,
                )
            )
        if not images or len({item.id for item in images}) != len(images):
            raise ValueError(
                "USDA raster inventory must contain unique published images"
            )
        return service, tuple(images), [first, second]

    def _prepare(
        self, scope: ExecutionScope, *, inventory: bool = False
    ) -> tuple[_LayerMetadata, tuple[_Part, ...], dict[int, str]]:
        if _empty(self.state):
            metadata = _metadata(())
            return (
                selected_metadata(metadata, self.state.resolve(metadata.spec)),
                (),
                {},
            )
        service, images, resources = self._catalogue(scope)
        periods = tuple(
            Period.window(year, year) for year in sorted({item.year for item in images})
        )
        metadata = _metadata(periods)
        resolved = self.state.resolve(metadata.spec)
        if inventory:
            source = SourceIdentity(
                "usda",
                "croplandcros_published_image_inventory",
                resources=tuple(resources),
            )
            return replace(metadata, source=source), (), {}
        if self.state.area is None:
            raise CapabilityError("Select an area before reading cropland pixels")
        if self.state.periods is None and self.state.series is None:
            latest = max(periods, key=lambda period: cast(int, period.start))
            metadata = _metadata((latest,))
            resolved = self.state.resolve(metadata.spec)
        metadata = selected_metadata(metadata, resolved)
        selected_years = {
            cast(int, series.period.start)
            for series in resolved.spec.series
            if series.period is not None
        }
        legend, resource = self._json(scope, _uri("/rasterAttributeTable", f="json"))
        resources.append(resource)
        with native_data(source="usda", scope=scope):
            names = {}
            codes = set()
            for feature in array(legend.get("features"), "USDA raster attribute table"):
                attributes = obj(
                    obj(feature, "USDA class").get("attributes"),
                    "USDA class attributes",
                )
                code = _integer(attributes.get("Value"), "USDA class code")
                if not 0 <= code <= 255 or code in codes:
                    raise ValueError(
                        "USDA raster class codes must be unique U8 integers"
                    )
                codes.add(code)
                label = attributes.get("Class_Names")
                if label is not None:
                    if not isinstance(label, str):
                        raise ValueError("USDA class name must be text or null")
                    # The actual RAT includes unnamed categories and padded labels.
                    # Retain published names verbatim, without making names into IDs.
                    if label.strip():
                        names[code] = label
            if not names:
                raise ValueError("USDA raster attribute table has no named classes")
            extent = obj(service.get("extent"), "USDA service extent")
            x_origin = cast(float, number(extent.get("xmin"), "USDA grid origin"))
            y_origin = cast(float, number(extent.get("ymax"), "USDA grid origin"))
            maximum_width = min(
                1024,
                _integer(
                    service.get("maxImageWidth"), "USDA image width", positive=True
                ),
            )
            maximum_height = min(
                1024,
                _integer(
                    service.get("maxImageHeight"), "USDA image height", positive=True
                ),
            )
        try:
            from rasterio.warp import transform_bounds  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "USDA Cropland requires `pip install crc-sdk[raster]`"
            ) from error
        area = self.state.area
        pieces = (
            (
                (area.west, area.south, 180.0, area.north),
                (-180.0, area.south, area.east, area.north),
            )
            if area.west > area.east
            else ((area.west, area.south, area.east, area.north),)
        )
        parts = []
        # Lock every primary image for a year: older publications can contain
        # several native-resolution regions. The service handles their mosaic.
        for year in sorted(selected_years):
            yearly = [image for image in images if image.year == year]
            pixel_size = min(image.pixel_size for image in yearly)
            period = Period.window(year, year)
            mosaic = json.dumps(
                {
                    "mosaicMethod": "esriMosaicLockRaster",
                    "lockRasterIds": [image.id for image in yearly],
                    "mosaicOperation": "MT_FIRST",
                },
                separators=(",", ":"),
            )
            seen: set[tuple[int, int]] = set()
            for piece in pieces:
                bounds = transform_bounds(
                    "EPSG:4326", "EPSG:3857", *piece, densify_pts=21
                )
                west = max(
                    bounds[0], cast(float, number(extent.get("xmin"), "USDA extent"))
                )
                south = max(
                    bounds[1], cast(float, number(extent.get("ymin"), "USDA extent"))
                )
                east = min(
                    bounds[2], cast(float, number(extent.get("xmax"), "USDA extent"))
                )
                north = min(
                    bounds[3], cast(float, number(extent.get("ymax"), "USDA extent"))
                )
                if west > east or south > north:
                    continue
                first_column = math.floor((west - x_origin) / pixel_size)
                last_column = max(
                    first_column + 1, math.ceil((east - x_origin) / pixel_size)
                )
                first_row = math.floor((y_origin - north) / pixel_size)
                last_row = max(
                    first_row + 1, math.ceil((y_origin - south) / pixel_size)
                )
                for row in range(first_row, last_row, maximum_height):
                    height = min(maximum_height, last_row - row)
                    for column in range(first_column, last_column, maximum_width):
                        if (row, column) in seen:
                            continue
                        seen.add((row, column))
                        width = min(maximum_width, last_column - column)
                        left, top = (
                            x_origin + column * pixel_size,
                            y_origin - row * pixel_size,
                        )
                        requested_bounds = (
                            left,
                            top - height * pixel_size,
                            left + width * pixel_size,
                            top,
                        )
                        uri = _uri(
                            "/exportImage",
                            f="image",
                            bbox=",".join(format(v, ".15g") for v in requested_bounds),
                            bboxSR=3857,
                            imageSR=3857,
                            size=f"{width},{height}",
                            format="tiff",
                            pixelType="U8",
                            noData=0,
                            interpolation="RSP_NearestNeighbor",
                            renderingRule='{"rasterFunction":"None"}',
                            mosaicRule=mosaic,
                        )
                        path, resource = fetch(
                            scope, uri, license=_LICENSE, attribution=_ATTRIBUTION
                        )
                        resources.append(resource)
                        parts.append(
                            _Part(path, period, width, height, requested_bounds)
                        )
        release = release_snapshot(
            resources, id="usda:croplandcros:selected-source-snapshot"
        )
        check_release(release, self.release)
        source = SourceIdentity(
            "usda",
            "croplandcros_categorical_pixels",
            release=release,
            resources=tuple(resources),
        )
        return replace(metadata, source=source), tuple(parts), names

    def inspection(self, scope: ExecutionScope) -> Inspection:
        metadata, _, _ = self._prepare(scope, inventory=self.state.area is None)
        return Inspection(
            schema=evidence_arrow_schema(metadata),
            series=tuple(
                SeriesReference(
                    item.variable,
                    period=item.period,
                    release=None
                    if metadata.source is None
                    else metadata.source.release,
                )
                for item in metadata.spec.series
            ),
            releases=()
            if metadata.source is None or metadata.source.release is None
            else (metadata.source.release,),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
            ),
            requirements=self.requirements(),
            source=metadata.source,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[_LayerMetadata]:
        metadata, parts, names = self._prepare(scope)
        resolved = self.state.resolve(metadata.spec, source=metadata.source)
        count = missing = 0

        def native_batches() -> Iterator[pa.RecordBatch]:
            nonlocal count, missing
            pending = []
            seen: set[tuple[str | None, object]] = set()
            for part in parts:
                assert self.state.area is not None
                import rasterio  # type: ignore[import-untyped]

                with rasterio.open(part.path) as raster:
                    if (
                        raster.count != 1
                        or raster.dtypes != ("uint8",)
                        or raster.crs is None
                        or raster.crs.to_epsg() != 3857
                        or raster.width != part.width
                        or raster.height != part.height
                        or not all(
                            math.isclose(actual, requested, rel_tol=0, abs_tol=1e-5)
                            for actual, requested in zip(
                                raster.bounds, part.bounds, strict=True
                            )
                        )
                    ):
                        raise ValueError(
                            "USDA categorical export differs from its requested "
                            "pixel grid"
                        )
                for row in raster_records(
                    part.path,
                    provider="usda:croplandcros",
                    variable_id=_VARIABLE.id,
                    period_id=part.period.id,
                    area=self.state.area,
                    categorical=True,
                    block_pixels=scope.runtime.batch_rows,
                ):
                    key = (part.period.id, row["support_id"])
                    if key in seen or not resolved.matches_record(row):
                        continue
                    seen.add(key)
                    value = row["value"]
                    if value is not None:
                        code = _integer(value, "USDA pixel class code")
                        if not 0 <= code <= 255:
                            raise ValueError(
                                "USDA pixel is outside its U8 class domain"
                            )
                        row["label"] = names.get(code)
                    count += 1
                    missing += int(value is None)
                    pending.append(row)
                    if len(pending) >= scope.runtime.batch_rows:
                        yield from evidence_table(pending, metadata).to_batches()
                        pending = []
            if pending:
                yield from evidence_table(pending, metadata).to_batches()

        def batches() -> Iterator[pa.RecordBatch]:
            with native_data(source="usda", scope=scope):
                yield from native_batches()

        def report() -> ExecutionReport:
            return ExecutionReport(
                False,
                None,
                counts=ExecutionCounts(
                    requested=count, produced=count, no_data=missing
                ),
                source_identities=() if metadata.source is None else (metadata.source,),
            )

        return NodeStream(
            evidence_arrow_schema(metadata), metadata, batches(), report=report
        )


class USDACropland(Layer):
    """Published crop class codes and labels, with actual service pixel footprints.

    Select an area before reading. Omitting a period selects the latest published
    crop year; inspection lists all published years. Pixel values remain numeric
    categories from the service's unrendered thematic band.
    """

    def __init__(self, *, release: ReleaseReference | None = None) -> None:
        if release is not None and not isinstance(release, ReleaseReference):
            raise TypeError("release must be ReleaseReference or None")
        _initialize(
            self,
            _CroplandNode(release),
            "Read verified USDA CroplandCROS crop classes and their published labels",
        )
