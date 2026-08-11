"""JRC tile-index resolution and return-period raster access.

Generalizes the URL/tile-index conventions JRC's return-period GeoTIFF
collections share (a `tile_extents.geojson` index plus a `{base_url}/RP{rp}/
{tile_id}_RP{rp}_depth.tif`-shaped filename template) into one
`JRCRasterDataset`, so CEMS-GLOFAS (global) and CEMS-EFAS (Europe-regional)
are two instances of the same dataset description rather than two providers
-- the same generalization `OSClimateInventory`/`OSClimateProvider` already
apply to the many OS-Climate hazard indicators sharing one inventory format.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
class JRCRasterDataset:
    """One JRC return-period GeoTIFF collection's URL/tile conventions."""

    name: str
    base_url: str
    tile_index_url: str
    filename_template: str  # e.g. "{tile_id}_RP{return_period}_depth.tif"
    available_return_periods: tuple[int, ...]
    hazard_name: str = "RiverineInundation"
    indicator_id: str = "flood_depth"
    units: str = "m"
    # JRC's baseline return-period maps carry no climate-scenario/year
    # dimension of their own; `pathway`/`horizon` are fixed sentinels so
    # canonical rows still have a well-defined (non-null) scenario/horizon.
    pathway: str = "historical"
    horizon: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.base_url or not self.tile_index_url:
            raise ValueError("name, base_url, and tile_index_url must be non-empty")
        if not self.filename_template:
            raise ValueError("filename_template must be non-empty")
        if not self.available_return_periods:
            raise ValueError("available_return_periods must be non-empty")

    def tile_url(self, tile_id: str, return_period: int) -> str:
        if return_period not in self.available_return_periods:
            raise ValueError(
                f"return period {return_period} is not available for "
                f"{self.name}; choose from {self.available_return_periods}"
            )
        filename = self.filename_template.format(
            tile_id=tile_id, return_period=return_period
        )
        return f"{self.base_url}/RP{return_period}/{filename}"


_GLOFAS_BASE = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
)
_EFAS_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/flood_hazard"

#: JRC Global River Flood Hazard Map (CEMS-GLOFAS v2.1.2, CC BY 4.0).
GLOFAS = JRCRasterDataset(
    name="cems-glofas-river-flood",
    base_url=_GLOFAS_BASE,
    tile_index_url=f"{_GLOFAS_BASE}/tile_extents.geojson",
    filename_template="{tile_id}_RP{return_period}_depth.tif",
    available_return_periods=(10, 20, 50, 75, 100, 200, 500),
)

#: JRC River Flood Hazard Map for Europe (CEMS-EFAS, higher-resolution
#: regional sibling of GLOFAS, CC BY 4.0). Same layout/filename convention.
EFAS = JRCRasterDataset(
    name="cems-efas-river-flood",
    base_url=_EFAS_BASE,
    tile_index_url=f"{_EFAS_BASE}/tile_extents.geojson",
    filename_template="{tile_id}_RP{return_period}_depth.tif",
    available_return_periods=(10, 20, 50, 75, 100, 200, 500),
)


class JRCProvider:
    """Resolve an AOI to JRC tiles and canonicalize them via `JRCIngestPolicy`."""

    def __init__(
        self,
        dataset: JRCRasterDataset = GLOFAS,
        *,
        connection: DuckDBConnection | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.dataset = dataset
        # An explicit connection means the caller is already in control; only
        # build (and resource-tune) one when they didn't supply their own.
        # No extensions requested -- same reasoning as GeoTiffRaster itself.
        self.connection = connection or DuckDBConnection.for_analytics(
            work_dir or default_work_dir(), extensions=()
        )
        self.work_dir = work_dir
        self._tile_index: tuple[Mapping[str, Any], ...] | None = None

    def _index(self) -> tuple[Mapping[str, Any], ...]:
        if self._tile_index is None:
            with urlopen(self.dataset.tile_index_url) as response:
                document = json.load(response)
            self._tile_index = tuple(document["features"])
        return self._tile_index

    def tiles_for(self, bounds: Bounds) -> tuple[str, ...]:
        """Tile ids (e.g. `"ID54_N50_W80"`) whose extent intersects `bounds`."""
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
            tile_bounds = box(*shape(feature["geometry"]).bounds)
            if tile_bounds.intersects(aoi):
                tiles.append(f"ID{properties['id']}_{properties['name']}")
        return tuple(tiles)

    def tile_url(self, tile_id: str, return_period: int) -> str:
        return self.dataset.tile_url(tile_id, return_period)

    def open_tile(
        self,
        tile_id: str,
        *,
        return_periods: Sequence[int] | None = None,
        cache_dir: str | Path | None = None,
    ) -> JRCReturnPeriodRaster:
        """Open one tile's return-period GeoTIFF stack as one curve source.

        `return_periods=None` (default) opens every return period the
        dataset advertises. Returns a context manager (`JRCReturnPeriodRaster`);
        callers own its lifecycle, same as `GeoTiffRaster.open()`.
        """
        periods = tuple(return_periods or self.dataset.available_return_periods)
        urls = {period: self.dataset.tile_url(tile_id, period) for period in periods}
        metadata = RasterMetadata(
            hazard_type=self.dataset.hazard_name,
            indicator_id=self.dataset.indicator_id,
            scenario=self.dataset.pathway,
            year=self.dataset.horizon,
            units=self.dataset.units,
            path=f"{self.dataset.name}/{tile_id}",
        )
        return JRCReturnPeriodRaster.open(
            urls,
            metadata,
            cache_dir=cache_dir,
            connection=self.connection,
            work_dir=self.work_dir,
        )

    def canonicalize_tile(
        self,
        tile_id: str,
        policy: CurveFitIngestPolicy,
        *,
        return_periods: Sequence[int] | None = None,
        bounds: Bounds | None = None,
        cache_dir: str | Path | None = None,
    ) -> CanonicalHazardStream:
        """Open one tile, fit its return-period curves, then close it.

        Unlike calling `canonicalize_jrc_flood` on an already-open
        `JRCReturnPeriodRaster`, this owns the tile's GeoTIFF file handles
        for exactly one call -- pixel strips still stream in bounded memory
        (nothing about the tile's raw size is ever fully materialized), but
        the resulting canonical rows (already reduced to fitted H3 cells,
        far smaller than the source pixel volume) are read into memory
        before the files close, so a caller iterating many tiles never has
        to manage a `JRCReturnPeriodRaster`'s lifecycle itself.
        """
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
