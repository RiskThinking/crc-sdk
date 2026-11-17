"""Source-native EDO soil moisture file layout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EDODataset:
    """One EDO drought indicator's URL/filename conventions."""

    name: str
    base_url: str
    filename_template: str  # e.g. "sminx_m_eul_{year}0101_{year}1221_t.nc"
    variable: str
    version: str = "unknown"
    indicator_id: str = "soil_moisture_index"
    units: str = "index"

    def __post_init__(self) -> None:
        if not self.name or not self.base_url or not self.filename_template:
            raise ValueError("name, base_url, and filename_template must be non-empty")
        if not self.variable:
            raise ValueError("variable must be non-empty")

    def year_url(self, year: int) -> str:
        return f"{self.base_url}/{self.filename_template.format(year=year)}"


#: JRC/Copernicus EDO Soil Moisture Index, LISFLOOD-derived (CC BY 4.0).
SMI = EDODataset(
    name="jrc-edo-soil-moisture-index",
    base_url=(
        "https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets"
        "/EDO_Soil_Moisture_Index/ver3-0-1"
    ),
    filename_template="sminx_m_eul_{year}0101_{year}1221_t.nc",
    variable="sminx",
    version="3.0.1",
)
