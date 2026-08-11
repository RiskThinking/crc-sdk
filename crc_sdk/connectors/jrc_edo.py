"""EDO drought (SMI) curves, canonicalized via the shared curve-fit core.

EDO ships a continuous dekadal time series, not a return-period raster.
`crc_sdk.connectors.duckdb.netcdf.EDOAnnualMinimaCurveSource` derives one
annual-block-minima curve per H3 cell from it first (empirical Gringorten
plotting positions, not literal return periods); this module fits that
derived curve through the same core `canonicalize_curve_source` JRC flood
and OS-Climate both use. Use `tail="lower"` -- drought severity is worse at
*lower* SMI, the opposite convention from flood depth.
"""

from __future__ import annotations

from crc_sdk.connectors.adapters import (
    CanonicalHazardStream,
    CurveFitIngestPolicy,
    canonicalize_curve_source,
)
from crc_sdk.connectors.duckdb.netcdf import EDOAnnualMinimaCurveSource
from crc_sdk.connectors.duckdb.zarr import Bounds

# Same fields as `CurveFitIngestPolicy` -- EDO needs no source-specific
# ingest knobs today. Kept as a distinct name for discoverability alongside
# `EDOAnnualMinimaCurveSource`/`EDOProvider`, not as a second, diverging class.
EDOIngestPolicy = CurveFitIngestPolicy


def canonicalize_edo_drought(
    source: EDOAnnualMinimaCurveSource,
    policy: EDOIngestPolicy,
    *,
    bounds: Bounds | None = None,
) -> CanonicalHazardStream:
    """Fit an EDO annual-block-minima drought curve into canonical hazard rows."""
    return canonicalize_curve_source(source, policy, provider="jrc-edo", bounds=bounds)
