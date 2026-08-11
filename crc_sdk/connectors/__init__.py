"""External format and query-engine connectors."""

from .adapters import (
    CanonicalHazardBatch,
    CanonicalHazardStream,
    CurveFitIngestPolicy,
    CurveSource,
    HurdleFitPolicy,
    OSClimateIngestPolicy,
    canonicalize_curve_source,
    canonicalize_os_climate,
)
from .jrc import JRCIngestPolicy, canonicalize_jrc_flood
from .parquet import (
    hazard_arrow_schema,
    read_hazard_dataset,
    read_hazard_metadata,
    sort_hazard_table,
    validate_hazard_table,
    write_hazard_dataset,
    write_hazard_stream,
)
from .protocols import HazardReader

__all__ = [
    "HazardReader",
    "CanonicalHazardBatch",
    "CanonicalHazardStream",
    "CurveFitIngestPolicy",
    "CurveSource",
    "HurdleFitPolicy",
    "JRCIngestPolicy",
    "OSClimateIngestPolicy",
    "canonicalize_curve_source",
    "canonicalize_jrc_flood",
    "canonicalize_os_climate",
    "hazard_arrow_schema",
    "read_hazard_dataset",
    "read_hazard_metadata",
    "sort_hazard_table",
    "validate_hazard_table",
    "write_hazard_dataset",
    "write_hazard_stream",
]
