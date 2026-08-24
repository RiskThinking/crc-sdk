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
from .agriculture import FTWFields, FTWFieldScan, USDACropland, USDACroplandScan
from .jrc import JRCIngestPolicy, canonicalize_jrc_flood
from .jrc_edo import EDOIngestPolicy, canonicalize_edo_drought
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
    "FTWFieldScan",
    "FTWFields",
    "HazardReader",
    "CanonicalHazardBatch",
    "CanonicalHazardStream",
    "CurveFitIngestPolicy",
    "CurveSource",
    "EDOIngestPolicy",
    "HurdleFitPolicy",
    "JRCIngestPolicy",
    "OSClimateIngestPolicy",
    "USDACropland",
    "USDACroplandScan",
    "canonicalize_curve_source",
    "canonicalize_edo_drought",
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
