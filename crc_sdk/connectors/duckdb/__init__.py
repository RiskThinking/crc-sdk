"""DuckDB-backed connector helpers."""

from .connection import (
    DuckDBConnection,
    DuckDBStreamEngine,
    RuntimeResources,
    default_work_dir,
    detected_cpu_count,
    ensure_extensions,
    sql_quote,
)
from .zarr import Bounds, Point, RasterCurve, RasterMetadata, ZarrRaster, ZarrScan

__all__ = [
    "Bounds",
    "DuckDBConnection",
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
    "sql_quote",
]
