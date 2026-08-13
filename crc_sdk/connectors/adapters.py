"""Adapters from external connector results to canonical hazard rows.

Every source below is fitted into the same `curve_kind in {"fitted",
"hurdle"}` canonical rows -- including JRC, whose raw rasters already carry
exact per-return-period depths. This is deliberate, not a loss of fidelity
by accident: crc-sdk's own contract treats "source knots and fit
diagnostics" as transient ingest inputs, never a second persisted data
contract (see README.md, "Canonical hazard datasets"), and JRC's per-pixel
depth-by-return-period sequence -- typically zero at low return periods,
positive and increasing from some higher return period onward -- is exactly
the zero-inflated shape `HurdleFitPolicy`/`fit_hurdle_quantiles` already
exists to fit. Reading a fitted/hurdle curve back at a given return period
therefore evaluates the curve rather than reproducing the source pixel's
value bit-for-bit; `maximum_normalized_rmse`/`maximum_absolute_residual`
bound how far that evaluation may drift, and `on_fit_failure="skip"` drops
pixels that can't be usefully fitted rather than aborting a whole ingest.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Protocol, get_args, runtime_checkable

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import (
    FittedDistribution,
    HurdleDistribution,
    QuantileFitDiagnostics,
    TabulatedDistribution,
    fit_hurdle_quantiles,
    fit_quantiles,
)
from crc_framework.distributions import DistributionFamily

from crc_sdk.connectors.duckdb.zarr import (
    Bounds,
    RasterCurve,
    RasterMetadata,
    ZarrRaster,
)
from crc_sdk.connectors.parquet import (
    hazard_arrow_schema,
    validate_hazard_table,
)
from crc_sdk.geometry import intersecting_cells
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance


@runtime_checkable
class CurveSource(Protocol):
    """Anything presenting a per-pixel leading-axis curve, ready to fit.

    `ZarrRaster` (OS-Climate) and `JRCReturnPeriodRaster`
    (`crc_sdk.connectors.duckdb.geotiff`, JRC) both satisfy this structurally
    -- `canonicalize_curve_source` fits curves against whichever one is
    passed, with no source-specific code of its own.
    """

    @property
    def axis_name(self) -> str: ...

    @property
    def metadata(self) -> RasterMetadata: ...

    def iter_curves(self, bounds: Bounds | None = None) -> Iterator[RasterCurve]: ...


@dataclass(frozen=True)
class HurdleFitPolicy:
    """Explicit point-mass policy for one external quantile dataset."""

    atom_probability: float
    atom_location: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.atom_probability < 1.0:
            raise ValueError("atom_probability must be strictly between zero and one")
        if not np.isfinite(self.atom_location):
            raise ValueError("atom_location must be finite")


@dataclass(frozen=True)
class CurveFitIngestPolicy:
    """Explicit policy controlling curve-source-to-canonical conversion.

    Source-agnostic by design: nothing here is specific to OS-Climate or
    JRC (or any other return-period raster source) -- it only controls how
    a per-pixel curve gets fitted. `OSClimateIngestPolicy` is kept as a
    backward-compatible alias for this exact class; `JRCIngestPolicy`
    (`crc_sdk.connectors.jrc`) is another.
    """

    h3_resolution: int
    family: DistributionFamily
    producer: str
    creation_version: str
    tail: Literal["upper", "lower"] = "upper"
    batch_rows: int = 65_536
    value_semantics: str | None = None
    source_version: str | None = None
    hurdle: HurdleFitPolicy | None = None
    maximum_normalized_rmse: float | None = None
    maximum_absolute_residual: float | None = None
    # Most pixels in an area (as opposed to a single known-exposed point) never
    # exceed the hazard threshold and carry a constant, unfittable curve;
    # "skip" drops those rather than aborting the whole area ingest.
    on_fit_failure: Literal["raise", "skip"] = "raise"

    def __post_init__(self) -> None:
        if not 0 <= self.h3_resolution <= 15:
            raise ValueError("H3 resolution must be between 0 and 15")
        if self.family not in get_args(DistributionFamily):
            raise ValueError(f"unknown distribution family {self.family!r}")
        if not self.producer or not self.creation_version:
            raise ValueError("producer and creation_version must be non-empty")
        if self.batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        if self.on_fit_failure not in ("raise", "skip"):
            raise ValueError("on_fit_failure must be 'raise' or 'skip'")
        for name, value in (
            ("maximum_normalized_rmse", self.maximum_normalized_rmse),
            ("maximum_absolute_residual", self.maximum_absolute_residual),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


# Backward-compatible alias: every field above was already source-agnostic,
# so this is the same class under its original name, not a copy.
OSClimateIngestPolicy = CurveFitIngestPolicy


@dataclass(frozen=True)
class CanonicalHazardBatch:
    """One batch of canonical H3-expanded hazard rows."""

    hazard_rows: Any


@dataclass
class CanonicalHazardStream:
    """Canonical metadata and one-shot Arrow batches."""

    metadata: HazardDatasetMetadata
    batches: Iterator[CanonicalHazardBatch]

    def read_all(self) -> Any:
        """Consume this stream into one canonical Arrow table."""
        batches = list(self.batches)
        if batches:
            return pa.concat_tables(
                [batch.hazard_rows for batch in batches],
                promote_options="none",
            )
        return pa.Table.from_batches(
            [],
            schema=hazard_arrow_schema(self.metadata),
        )


def _source_id(provider: str, path: str, curve: RasterCurve) -> str:
    identity = f"{provider}\0{path}\0{curve.row}\0{curve.column}".encode()
    return sha256(identity).hexdigest()


def _metadata(
    source: CurveSource, policy: CurveFitIngestPolicy, provider: str
) -> HazardDatasetMetadata:
    values = source.metadata
    support = getattr(source, "return_period_support", None)
    return HazardDatasetMetadata(
        h3_resolution=policy.h3_resolution,
        return_period_tail=policy.tail,
        return_period_support=support,
        value_unit=values.units,
        value_semantics=policy.value_semantics or values.indicator_id,
        producer=policy.producer,
        creation_version=policy.creation_version,
        source=SourceProvenance(
            provider=provider,
            dataset=f"{values.hazard_type}:{values.indicator_id}",
            uri=values.path,
            version=policy.source_version,
        ),
    )


def _fit_curve(
    tabulated: TabulatedDistribution,
    policy: CurveFitIngestPolicy,
) -> tuple[Any, Any]:
    distribution: FittedDistribution | HurdleDistribution
    diagnostics: QuantileFitDiagnostics
    if policy.hurdle is None:
        quantile_result = fit_quantiles(tabulated, family=policy.family)
        distribution = quantile_result.distribution
        diagnostics = quantile_result.diagnostics
    else:
        hurdle_result = fit_hurdle_quantiles(
            tabulated,
            family=policy.family,
            atom_probability=policy.hurdle.atom_probability,
            atom_location=policy.hurdle.atom_location,
        )
        distribution = hurdle_result.distribution
        diagnostics = hurdle_result.diagnostics.tail
    if not diagnostics.converged:
        raise ValueError("quantile optimizer did not converge")
    if (
        policy.maximum_normalized_rmse is not None
        and diagnostics.normalized_rmse > policy.maximum_normalized_rmse
    ):
        raise ValueError(
            f"normalized RMSE {diagnostics.normalized_rmse} exceeds policy "
            f"{policy.maximum_normalized_rmse}"
        )
    if (
        policy.maximum_absolute_residual is not None
        and diagnostics.maximum_absolute_residual > policy.maximum_absolute_residual
    ):
        raise ValueError(
            "maximum absolute residual "
            f"{diagnostics.maximum_absolute_residual} exceeds policy "
            f"{policy.maximum_absolute_residual}"
        )
    base = (
        distribution.base
        if isinstance(distribution, HurdleDistribution)
        else distribution
    )
    return distribution, base


def _canonical_batches(
    source: CurveSource,
    policy: CurveFitIngestPolicy,
    metadata: HazardDatasetMetadata,
    provider: str,
    bounds: Bounds | None,
) -> Iterator[CanonicalHazardBatch]:
    try:
        from shapely.geometry import Polygon  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "Curve-fit ingest requires `pip install crc-sdk[geometry]`"
        ) from error

    hazard_schema = hazard_arrow_schema(metadata)
    hazard_rows: list[dict[str, Any]] = []
    if "return period" not in source.axis_name.lower():
        raise ValueError(
            f"{source.metadata.path} has axis {source.axis_name!r}, not return periods"
        )
    for curve in source.iter_curves(bounds):
        valid = np.isfinite(curve.axis_values) & np.isfinite(curve.values)
        periods = curve.axis_values[valid]
        values = curve.values[valid]
        if len(values) < 4:
            continue
        try:
            tabulated = TabulatedDistribution.from_return_periods(
                periods,
                values,
                tail=policy.tail,
            )
            distribution, base = _fit_curve(tabulated, policy)
        except ValueError as error:
            # Non-monotonic quantiles (e.g. small per-return-period modeling
            # noise near a DEM sink or tile edge) fail the same way an
            # unconverged/out-of-tolerance fit does -- both mean "this pixel
            # can't be usefully fitted," so on_fit_failure governs both.
            if policy.on_fit_failure == "skip":
                continue
            raise ValueError(
                f"failed to fit source pixel row={curve.row}, "
                f"column={curve.column}: {error}"
            ) from error
        geometry = Polygon(curve.boundary)
        source_id = _source_id(provider, source.metadata.path, curve)
        cells = intersecting_cells(geometry, policy.h3_resolution)
        if not cells:
            continue
        curve_kind = (
            "hurdle" if isinstance(distribution, HurdleDistribution) else "fitted"
        )
        for cell_index in cells:
            hazard_rows.append(
                {
                    "cell_index": cell_index,
                    "source_id": source_id,
                    "source_geometry": geometry.wkb,
                    "hazard_name": source.metadata.hazard_type,
                    "horizon": source.metadata.year,
                    "pathway": source.metadata.scenario,
                    "curve_kind": curve_kind,
                    "curve_type": base.family,
                    "curve_shape": base.shape,
                    "curve_location": base.location,
                    "curve_scale": base.scale,
                    "curve_atom_probability": (
                        distribution.atom_probability
                        if isinstance(distribution, HurdleDistribution)
                        else None
                    ),
                    "curve_atom_location": (
                        distribution.atom_location
                        if isinstance(distribution, HurdleDistribution)
                        else None
                    ),
                }
            )
        if len(hazard_rows) >= policy.batch_rows:
            hazards = validate_hazard_table(
                pa.Table.from_pylist(hazard_rows, schema=hazard_schema),
                metadata=metadata,
            )
            yield CanonicalHazardBatch(hazard_rows=hazards)
            hazard_rows.clear()
    if hazard_rows:
        hazards = validate_hazard_table(
            pa.Table.from_pylist(hazard_rows, schema=hazard_schema),
            metadata=metadata,
        )
        yield CanonicalHazardBatch(hazard_rows=hazards)


def canonicalize_curve_source(
    source: CurveSource,
    policy: CurveFitIngestPolicy,
    *,
    provider: str,
    bounds: Bounds | None = None,
) -> CanonicalHazardStream:
    """Return a lazy canonical stream for one return-period curve source.

    Source-agnostic core: `canonicalize_os_climate` and
    `crc_sdk.connectors.jrc.canonicalize_jrc_flood` are both thin wrappers
    around this, differing only in `provider` and the `CurveSource`
    implementation they pass in.
    """
    metadata = _metadata(source, policy, provider)
    return CanonicalHazardStream(
        metadata=metadata,
        batches=_canonical_batches(source, policy, metadata, provider, bounds),
    )


def canonicalize_os_climate(
    raster: ZarrRaster,
    policy: OSClimateIngestPolicy,
    *,
    bounds: Bounds | None = None,
) -> CanonicalHazardStream:
    """Return a lazy canonical stream for one selected OS-Climate raster."""
    return canonicalize_curve_source(
        raster, policy, provider="os-climate", bounds=bounds
    )
