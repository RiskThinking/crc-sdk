"""DuckDB-backed connector helpers."""

from .connection import (
    DuckDBConnection,
    DuckDBStreamEngine,
    RuntimeResources,
    default_work_dir,
    detected_cpu_count,
    ensure_extensions,
    partitioned_write_open_files_hint,
    sql_identifier,
    sql_quote,
)
from .geotiff import (
    GeoTiffH3Scan,
    GeoTiffRaster,
    GeoTiffScan,
    JRCReturnPeriodRaster,
    trim_cache_dir,
)
from .netcdf import EDOAnnualMinimaCurveSource, NetCDFH3Scan, NetCDFRaster, NetCDFScan
from .zarr import Bounds, Point, RasterCurve, RasterMetadata, ZarrRaster, ZarrScan

__all__ = [
    "Bounds",
    "DuckDBConnection",
    "EDOAnnualMinimaCurveSource",
    "GeoTiffH3Scan",
    "GeoTiffRaster",
    "GeoTiffScan",
    "JRCReturnPeriodRaster",
    "NetCDFH3Scan",
    "NetCDFRaster",
    "NetCDFScan",
    "Point",
    "RasterCurve",
    "RasterMetadata",
    "DuckDBStreamEngine",
    "RuntimeResources",
    "ZarrRaster",
    "ZarrScan",
    "default_work_dir",
    "detected_cpu_count",
    "ensure_extensions",
    "partitioned_write_open_files_hint",
    "sql_identifier",
    "sql_quote",
    "trim_cache_dir",
]
