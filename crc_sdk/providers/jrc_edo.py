"""EDO (European Drought Observatory) yearly SMI file access.

EDO's Soil Moisture Index ships one NetCDF file per *complete* calendar
year (each holding all ~36 dekads for that year); the current, still-in-
progress year is a separate file whose end-date-stamped filename isn't
predictable without a live directory listing, so `EDODataset.year_url`
only targets complete years -- pass a `years` range that excludes the
current year.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from crc_sdk.connectors.adapters import (
    CanonicalHazardBatch,
    CanonicalHazardStream,
    CurveFitIngestPolicy,
)
from crc_sdk.connectors.duckdb import DuckDBConnection, NetCDFRaster, default_work_dir
from crc_sdk.connectors.duckdb.netcdf import EDOAnnualMinimaCurveSource
from crc_sdk.connectors.duckdb.zarr import Bounds, RasterMetadata


def _at_least_two(start: int, stop: int, size: int) -> tuple[int, int]:
    if stop - start >= 2:
        return start, stop
    if stop < size:
        return start, stop + 1
    if start > 0:
        return start - 1, stop
    return start, stop


@dataclass(frozen=True)
class EDODataset:
    """One EDO drought indicator's URL/filename conventions."""

    name: str
    base_url: str
    filename_template: str  # e.g. "sminx_m_eul_{year}0101_{year}1221_t.nc"
    variable: str
    version: str = "unknown"
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
    version="3.0.1",
)

EDO_DATASETS = {"smi": SMI}


def edo_dataset(name: str) -> EDODataset:
    try:
        return EDO_DATASETS[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown EDO dataset {name!r}; choose from {tuple(EDO_DATASETS)}"
        ) from error


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

    def resolve_version(self, requested: str) -> str:
        if requested != "latest":
            if requested != self.dataset.version:
                raise ValueError(
                    f"{self.dataset.name} release {requested!r} is not available; "
                    f"configured release: {self.dataset.version}"
                )
            return requested
        collection_url = self.dataset.base_url.rsplit("/", 1)[0] + "/"
        with urlopen(collection_url) as response:
            listing = response.read().decode("utf-8", errors="replace")
        versions: set[str] = {
            match.replace("-", ".")
            for match in re.findall(r'href="ver([0-9]+-[0-9]+-[0-9]+)/?"', listing)
        }
        if not versions:
            raise RuntimeError(f"no EDO releases found at {collection_url}")
        resolved = max(versions, key=lambda value: tuple(map(int, value.split("."))))
        if resolved != self.dataset.version:
            raise RuntimeError(
                f"{self.dataset.name} latest release is {resolved}, but crc-sdk "
                f"is configured for {self.dataset.version}"
            )
        return resolved

    def complete_years(self) -> tuple[int, ...]:
        """Discover complete calendar-year resources in the configured release."""
        with urlopen(self.dataset.base_url + "/") as response:
            listing = response.read().decode("utf-8", errors="replace")
        years = {
            int(year)
            for year in re.findall(
                r"sminx_m_eul_([0-9]{4})0101_\1(?:1221|1231)_t\.nc", listing
            )
        }
        if not years:
            raise RuntimeError(
                f"no complete EDO years found at {self.dataset.base_url}"
            )
        return tuple(sorted(years))

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
        return self.open_resources(
            {year: self.dataset.year_url(year) for year in years},
            cache_dir=cache_dir,
        )

    def open_resources(
        self,
        resources: dict[int, str | Path],
        *,
        cache_dir: str | Path | None = None,
    ) -> EDOAnnualMinimaCurveSource:
        """Open explicit yearly resources as an annual-minima curve source."""
        if not resources:
            raise ValueError("at least one year is required")
        rasters: dict[int, NetCDFRaster] = {}
        try:
            for year, source in resources.items():
                rasters[year] = NetCDFRaster.open(
                    source,
                    variable=self.dataset.variable,
                    cache_dir=cache_dir,
                    connection=self.connection,
                    work_dir=self.work_dir,
                )
        except Exception:
            for raster in rasters.values():
                raster.close()
            raise

        years = tuple(resources)
        metadata = RasterMetadata(
            hazard_type=self.dataset.hazard_name,
            indicator_id=self.dataset.indicator_id,
            scenario=self.dataset.pathway,
            year=max(years),
            units=self.dataset.units,
            path=f"{self.dataset.name}/{min(years)}-{max(years)}",
        )
        return EDOAnnualMinimaCurveSource(rasters=rasters, metadata=metadata)

    def cache_annual_minimum(
        self,
        year: int,
        bounds: Bounds,
        destination: str | Path,
    ) -> Path:
        """Persist one year's AOI annual minimum as a compact NetCDF source."""
        try:
            import h5netcdf  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "EDO caching requires `pip install crc-sdk[netcdf]`"
            ) from error

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.nc")
        with NetCDFRaster.open(
            self.dataset.year_url(year),
            variable=self.dataset.variable,
            connection=self.connection,
            work_dir=self.work_dir,
        ) as raster:
            row_start, row_stop, col_start, col_stop = raster._row_col_window(bounds)
            row_start, row_stop = _at_least_two(row_start, row_stop, len(raster._lat))
            col_start, col_stop = _at_least_two(col_start, col_stop, len(raster._lon))
            block = np.asarray(
                raster._variable[:, row_start:row_stop, col_start:col_stop],
                dtype=np.float64,
            )
            finite = np.isfinite(block)
            if raster.nodata is not None:
                finite &= block != raster.nodata
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                minimum = np.nanmin(np.where(finite, block, np.nan), axis=0)
            latitudes = raster._lat[row_start:row_stop]
            longitudes = raster._lon[col_start:col_stop]

        with h5netcdf.File(temporary, "w") as output:
            output.dimensions = {
                "time": 1,
                "lat": len(latitudes),
                "lon": len(longitudes),
            }
            output.create_variable("lat", ("lat",), dtype=np.float64, data=latitudes)
            output.create_variable("lon", ("lon",), dtype=np.float64, data=longitudes)
            output.create_variable(
                self.dataset.variable,
                ("time", "lat", "lon"),
                dtype=np.float32,
                data=minimum[np.newaxis].astype(np.float32),
            )
        temporary.replace(target)
        return target

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
