"""Source-native JRC flood release layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


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
    indicator_id: str = "flood_depth"
    units: str = "m"

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
