"""EDO (European Drought Observatory) yearly SMI file access.

EDO's Soil Moisture Index ships one NetCDF file per *complete* calendar
year (each holding all ~36 dekads for that year); the current, still-in-
progress year is a separate file whose end-date-stamped filename isn't
predictable without a live directory listing, so `EDODataset.year_url`
only targets complete years -- pass a `years` range that excludes the
current year.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from crc_sdk.connectors.adapters import (
    CanonicalHazardBatch,
    CanonicalHazardStream,
    CurveFitIngestPolicy,
)
from crc_sdk.connectors.duckdb import DuckDBConnection, NetCDFRaster, default_work_dir
from crc_sdk.connectors.duckdb.netcdf import EDOAnnualMinimaCurveSource
from crc_sdk.connectors.duckdb.zarr import Bounds, RasterMetadata


@dataclass(frozen=True)
class EDODataset:
    """One EDO drought indicator's URL/filename conventions."""

    name: str
    base_url: str
    filename_template: str  # e.g. "sminx_m_eul_{year}0101_{year}1221_t.nc"
    variable: str
    hazard_name: str = "Drought"
    indicator_id: str = "soil_moisture_index"
    units: str = "index"
    # EDO's SMI carries no climate-scenario dimension of its own; `pathway`
    # is a fixed sentinel so canonical rows still have a well-defined
    # (non-null) scenario, matching the JRC flood convention.
    pathway: str = "historical"

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
)


class EDOProvider:
    """Open EDO yearly SMI files and canonicalize them via `EDOIngestPolicy`."""

    def __init__(
        self,
        dataset: EDODataset = SMI,
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

    def year_url(self, year: int) -> str:
        return self.dataset.year_url(year)

    def open_years(
        self,
        years: Sequence[int],
        *,
        cache_dir: str | Path | None = None,
    ) -> EDOAnnualMinimaCurveSource:
        """Open one `NetCDFRaster` per year, as one annual-block-minima curve source.

        Returns a context manager (`EDOAnnualMinimaCurveSource`); callers
        own its lifecycle, same as `GeoTiffRaster.open()`/`JRCProvider.open_tile()`.
        """
        if not years:
            raise ValueError("at least one year is required")
        rasters: dict[int, NetCDFRaster] = {}
        try:
            for year in years:
                rasters[year] = NetCDFRaster.open(
                    self.dataset.year_url(year),
                    variable=self.dataset.variable,
                    cache_dir=cache_dir,
                    connection=self.connection,
                    work_dir=self.work_dir,
                )
        except Exception:
            for raster in rasters.values():
                raster.close()
            raise

        metadata = RasterMetadata(
            hazard_type=self.dataset.hazard_name,
            indicator_id=self.dataset.indicator_id,
            scenario=self.dataset.pathway,
            year=max(years),
            units=self.dataset.units,
            path=f"{self.dataset.name}/{min(years)}-{max(years)}",
        )
        return EDOAnnualMinimaCurveSource(rasters=rasters, metadata=metadata)

    def canonicalize_years(
        self,
        years: Sequence[int],
        policy: CurveFitIngestPolicy,
        *,
        bounds: Bounds | None = None,
        cache_dir: str | Path | None = None,
    ) -> CanonicalHazardStream:
        """Open every requested year, fit its drought curves, then close them.

        Owns every year's NetCDF file handle for exactly one call -- reads
        still stream in bounded memory (nothing about the underlying files
        is ever fully materialized), but the resulting canonical rows
        (already reduced to fitted H3 cells) are read into memory before
        the files close, so a caller never has to manage an
        `EDOAnnualMinimaCurveSource`'s lifecycle itself.
        """
        from crc_sdk.connectors.jrc_edo import canonicalize_edo_drought

        with self.open_years(years, cache_dir=cache_dir) as source:
            stream = canonicalize_edo_drought(source, policy, bounds=bounds)
            table = stream.read_all()
        return CanonicalHazardStream(
            metadata=stream.metadata,
            batches=iter([CanonicalHazardBatch(hazard_rows=table)]),
        )
