"""JRC return-period rasters, canonicalized via the shared curve-fit core.

See the module docstring in `crc_sdk.connectors.adapters` for why JRC's
exact per-return-period depths are fitted rather than persisted verbatim.
"""

from __future__ import annotations

from crc_sdk.connectors.adapters import (
    CanonicalHazardStream,
    CurveFitIngestPolicy,
    canonicalize_curve_source,
)
from crc_sdk.connectors.duckdb.geotiff import JRCReturnPeriodRaster
from crc_sdk.connectors.duckdb.zarr import Bounds

# Same fields as `CurveFitIngestPolicy` -- JRC needs no source-specific
# ingest knobs today. Kept as a distinct name for discoverability alongside
# `JRCReturnPeriodRaster`/`JRCProvider`, not as a second, diverging class.
JRCIngestPolicy = CurveFitIngestPolicy


def canonicalize_jrc_flood(
    source: JRCReturnPeriodRaster,
    policy: JRCIngestPolicy,
    *,
    bounds: Bounds | None = None,
) -> CanonicalHazardStream:
    """Fit a JRC per-tile return-period depth stack into canonical hazard rows."""
    return canonicalize_curve_source(source, policy, provider="jrc", bounds=bounds)
