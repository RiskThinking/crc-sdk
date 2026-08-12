"""Versioned JRC flood-raster discovery and access."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

from crc_sdk.connectors.adapters import (
    CanonicalHazardBatch,
    CanonicalHazardStream,
    CurveFitIngestPolicy,
)
from crc_sdk.connectors.duckdb import DuckDBConnection, default_work_dir
from crc_sdk.connectors.duckdb.geotiff import JRCReturnPeriodRaster
from crc_sdk.connectors.duckdb.zarr import Bounds, RasterMetadata


@dataclass(frozen=True)
class JRCRasterResource:
    """One same-grid stack of return-period rasters within a JRC release."""

    source_id: str
    urls: Mapping[int, str]


@dataclass(frozen=True)
class JRCRasterDataset:
    """URL, release, and layout conventions for one JRC flood dataset."""

    name: str
    base_url: str
    tile_index_url: str | None
    filename_template: str
    available_return_periods: tuple[int, ...]
    version: str = "unknown"
    layout: Literal["tiled", "continental"] = "tiled"
    hazard_name: str = "RiverineInundation"
    indicator_id: str = "flood_depth"
    units: str = "m"
    pathway: str = "historical"
    horizon: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.base_url or not self.filename_template:
            raise ValueError("name, base_url, and filename_template must be non-empty")
        if self.layout == "tiled" and not self.tile_index_url:
            raise ValueError("tiled datasets require a tile_index_url")
        if not self.available_return_periods:
            raise ValueError("available_return_periods must be non-empty")

    @property
    def readme_url(self) -> str:
        return f"{self.base_url}/README.txt"

    def tile_url(self, tile_id: str, return_period: int) -> str:
        if self.layout != "tiled":
            raise ValueError(f"{self.name} uses a continental raster layout")
        return self.raster_url(return_period, tile_id=tile_id)

    def raster_url(self, return_period: int, *, tile_id: str | None = None) -> str:
        if return_period not in self.available_return_periods:
            raise ValueError(
                f"return period {return_period} is not available for "
                f"{self.name}; choose from {self.available_return_periods}"
            )
        if self.layout == "tiled" and not tile_id:
            raise ValueError("tile_id is required for a tiled JRC dataset")
        filename = self.filename_template.format(
            tile_id=tile_id, return_period=return_period
        )
        if self.layout == "tiled":
            return f"{self.base_url}/RP{return_period}/{filename}"
        return f"{self.base_url}/{filename}"


_GLOFAS_BASE = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
)
_EFAS_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/flood_hazard"

GLOFAS = JRCRasterDataset(
    name="cems-glofas-river-flood",
    base_url=_GLOFAS_BASE,
    tile_index_url=f"{_GLOFAS_BASE}/tile_extents.geojson",
    filename_template="{tile_id}_RP{return_period}_depth.tif",
    available_return_periods=(10, 20, 50, 75, 100, 200, 500),
    version="2.1.2",
)

EFAS = JRCRasterDataset(
    name="cems-efas-river-flood",
    base_url=_EFAS_BASE,
    tile_index_url=None,
    filename_template="Europe_RP{return_period}_filled_depth.tif",
    available_return_periods=(10, 20, 30, 40, 50, 75, 100, 200, 500),
    version="3.1.1",
    layout="continental",
)

JRC_DATASETS: Mapping[str, JRCRasterDataset] = {
    "glofas": GLOFAS,
    "efas": EFAS,
}


def jrc_dataset(name: str) -> JRCRasterDataset:
    """Return a configured JRC dataset without creating execution resources."""
    try:
        return JRC_DATASETS[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown JRC dataset {name!r}; choose from {tuple(JRC_DATASETS)}"
        ) from error


class JRCProvider:
    """Resolve JRC releases and AOIs into return-period raster stacks."""

    def __init__(
        self,
        dataset: JRCRasterDataset = GLOFAS,
        *,
        connection: DuckDBConnection | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.dataset = dataset
        self.connection = connection or DuckDBConnection.for_analytics(
            work_dir or default_work_dir(), extensions=()
        )
        self.work_dir = work_dir
        self._tile_index: tuple[Mapping[str, Any], ...] | None = None

    @classmethod
    def for_dataset(
        cls,
        name: str,
        *,
        connection: DuckDBConnection | None = None,
        work_dir: str | Path | None = None,
    ) -> JRCProvider:
        dataset = jrc_dataset(name)
        return cls(dataset, connection=connection, work_dir=work_dir)

    def resolve_version(self, requested: str) -> str:
        """Resolve and validate a source version without downloading rasters."""
        if requested != "latest":
            if requested != self.dataset.version:
                raise ValueError(
                    f"{self.dataset.name} release {requested!r} is not available at "
                    f"the current JRC URL; configured release: {self.dataset.version}"
                )
            return requested
        with urlopen(self.dataset.readme_url) as response:
            readme = response.read().decode("utf-8", errors="replace")
        match = re.search(r"Dataset version\s+([0-9]+(?:\.[0-9]+)+)", readme)
        if match is None:
            raise RuntimeError(
                f"could not resolve the release in {self.dataset.readme_url}"
            )
        resolved = match.group(1)
        if resolved != self.dataset.version:
            raise RuntimeError(
                f"{self.dataset.name} latest release is {resolved}, but crc-sdk "
                f"is configured for {self.dataset.version}; its source layout may "
                "have changed"
            )
        return resolved

    def _index(self) -> tuple[Mapping[str, Any], ...]:
        if self.dataset.layout != "tiled" or self.dataset.tile_index_url is None:
            raise ValueError(f"{self.dataset.name} does not publish a tile index")
        if self._tile_index is None:
            with urlopen(self.dataset.tile_index_url) as response:
                document = json.load(response)
            self._tile_index = tuple(document["features"])
        return self._tile_index

    def tiles_for(self, bounds: Bounds) -> tuple[str, ...]:
        """Return tiled resource ids intersecting WGS84 bounds."""
        if self.dataset.layout != "tiled":
            return ("continental",)
        try:
            from shapely.geometry import box, shape  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "JRC tile resolution requires `pip install crc-sdk[geometry]`"
            ) from error

        aoi = box(*bounds)
        tiles = []
        for feature in self._index():
            properties = feature["properties"]
            if box(*shape(feature["geometry"]).bounds).intersects(aoi):
                tiles.append(f"ID{properties['id']}_{properties['name']}")
        return tuple(tiles)

    def tile_url(self, tile_id: str, return_period: int) -> str:
        return self.dataset.tile_url(tile_id, return_period)

    def resources_for(
        self,
        bounds: Bounds,
        *,
        return_periods: Sequence[int] | None = None,
    ) -> tuple[JRCRasterResource, ...]:
        periods = tuple(return_periods or self.dataset.available_return_periods)
        if not periods:
            raise ValueError("at least one source return period is required")
        for period in periods:
            self.dataset.raster_url(
                period,
                tile_id="validation" if self.dataset.layout == "tiled" else None,
            )
        resource_ids = self.tiles_for(bounds)
        return tuple(
            JRCRasterResource(
                source_id=resource_id,
                urls={
                    period: self.dataset.raster_url(
                        period,
                        tile_id=(
                            resource_id if self.dataset.layout == "tiled" else None
                        ),
                    )
                    for period in periods
                },
            )
            for resource_id in resource_ids
        )

    def open_resource(
        self,
        resource: JRCRasterResource,
        *,
        cache_dir: str | Path | None = None,
    ) -> JRCReturnPeriodRaster:
        metadata = RasterMetadata(
            hazard_type=self.dataset.hazard_name,
            indicator_id=self.dataset.indicator_id,
            scenario=self.dataset.pathway,
            year=self.dataset.horizon,
            units=self.dataset.units,
            path=f"{self.dataset.name}/{resource.source_id}",
        )
        return JRCReturnPeriodRaster.open(
            resource.urls,
            metadata,
            cache_dir=cache_dir,
            connection=self.connection,
            work_dir=self.work_dir,
        )

    def open_aoi(
        self,
        bounds: Bounds,
        *,
        return_periods: Sequence[int] | None = None,
        cache_dir: str | Path | None = None,
    ) -> JRCReturnPeriodRaster:
        """Open a single-resource AOI as a return-period raster stack."""
        resources = self.resources_for(bounds, return_periods=return_periods)
        if len(resources) != 1:
            raise ValueError(
                f"AOI intersects {len(resources)} JRC tiles; materialize the AOI "
                "workflow or open each resource separately"
            )
        return self.open_resource(resources[0], cache_dir=cache_dir)

    def open_tile(
        self,
        tile_id: str,
        *,
        return_periods: Sequence[int] | None = None,
        cache_dir: str | Path | None = None,
    ) -> JRCReturnPeriodRaster:
        periods = tuple(return_periods or self.dataset.available_return_periods)
        resource = JRCRasterResource(
            source_id=tile_id,
            urls={period: self.dataset.tile_url(tile_id, period) for period in periods},
        )
        return self.open_resource(resource, cache_dir=cache_dir)

    def canonicalize_tile(
        self,
        tile_id: str,
        policy: CurveFitIngestPolicy,
        *,
        return_periods: Sequence[int] | None = None,
        bounds: Bounds | None = None,
        cache_dir: str | Path | None = None,
    ) -> CanonicalHazardStream:
        from crc_sdk.connectors.jrc import canonicalize_jrc_flood

        with self.open_tile(
            tile_id, return_periods=return_periods, cache_dir=cache_dir
        ) as source:
            stream = canonicalize_jrc_flood(source, policy, bounds=bounds)
            table = stream.read_all()
        return CanonicalHazardStream(
            metadata=stream.metadata,
            batches=iter([CanonicalHazardBatch(hazard_rows=table)]),
        )
