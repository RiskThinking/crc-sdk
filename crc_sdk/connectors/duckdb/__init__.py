"""DuckDB-backed connector helpers."""

from .connection import (
    DuckDBConnection,
    DuckDBSecret,
    DuckDBStreamEngine,
    RuntimeResources,
    apply_secret,
    default_work_dir,
    detected_cpu_count,
    ensure_extensions,
    gcs_hmac_secret_from_env,
    secret_sql,
    sql_identifier,
    sql_quote,
)
from .geotiff import GeoTiffH3Scan, GeoTiffRaster, GeoTiffScan, trim_cache_dir
from .zarr import Bounds, Point, RasterCurve, RasterMetadata, ZarrRaster, ZarrScan

__all__ = [
    "Bounds",
    "DuckDBConnection",
    "DuckDBSecret",
    "GeoTiffH3Scan",
    "GeoTiffRaster",
    "GeoTiffScan",
    "Point",
    "RasterCurve",
    "RasterMetadata",
    "DuckDBStreamEngine",
    "RuntimeResources",
    "ZarrRaster",
    "ZarrScan",
    "apply_secret",
    "default_work_dir",
    "detected_cpu_count",
    "ensure_extensions",
    "gcs_hmac_secret_from_env",
    "secret_sql",
    "sql_identifier",
    "sql_quote",
    "trim_cache_dir",
]
