"""Adapters from external connector results to canonical hazard rows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, get_args

import numpy as np
from crc_framework import (
    FittedDistribution,
    HurdleDistribution,
    QuantileFitDiagnostics,
    TabulatedDistribution,
    fit_hurdle_quantiles,
    fit_quantiles,
)
from crc_framework.distributions import DistributionFamily

from crc_sdk.connectors.duckdb.zarr import Bounds, RasterCurve, ZarrRaster
from crc_sdk.connectors.parquet import (
    hazard_arrow_schema,
    validate_hazard_table,
)
from crc_sdk.geometry import intersecting_cells
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance


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
class OSClimateIngestPolicy:
    """Explicit policy controlling raster-to-canonical conversion."""

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

    def __post_init__(self) -> None:
        if not 0 <= self.h3_resolution <= 15:
            raise ValueError("H3 resolution must be between 0 and 15")
        if self.family not in get_args(DistributionFamily):
            raise ValueError(f"unknown distribution family {self.family!r}")
        if not self.producer or not self.creation_version:
            raise ValueError("producer and creation_version must be non-empty")
        if self.batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        for name, value in (
            ("maximum_normalized_rmse", self.maximum_normalized_rmse),
            ("maximum_absolute_residual", self.maximum_absolute_residual),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


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
        try:
            import pyarrow as pa  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "OS-Climate ingest requires `crc-sdk[connectors,geometry]`"
            ) from error
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


def _source_id(raster: ZarrRaster, curve: RasterCurve) -> str:
    identity = (
        f"os-climate\0{raster.metadata.path}\0{curve.row}\0{curve.column}"
    ).encode()
    return sha256(identity).hexdigest()


def _metadata(
    raster: ZarrRaster, policy: OSClimateIngestPolicy
) -> HazardDatasetMetadata:
    values = raster.metadata
    return HazardDatasetMetadata(
        h3_resolution=policy.h3_resolution,
        value_unit=values.units,
        value_semantics=policy.value_semantics or values.indicator_id,
        producer=policy.producer,
        creation_version=policy.creation_version,
        source=SourceProvenance(
            provider="os-climate",
            dataset=f"{values.hazard_type}:{values.indicator_id}",
            uri=values.path,
            version=policy.source_version,
        ),
    )


def _fit_curve(
    tabulated: TabulatedDistribution,
    policy: OSClimateIngestPolicy,
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
        and diagnostics.maximum_absolute_residual
        > policy.maximum_absolute_residual
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
    raster: ZarrRaster,
    policy: OSClimateIngestPolicy,
    metadata: HazardDatasetMetadata,
    bounds: Bounds | None,
) -> Iterator[CanonicalHazardBatch]:
    try:
        import pyarrow as pa
        from shapely.geometry import Polygon  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "OS-Climate ingest requires `crc-sdk[connectors,geometry]`"
        ) from error

    hazard_schema = hazard_arrow_schema(metadata)
    hazard_rows: list[dict[str, Any]] = []
    if "return period" not in raster.axis_name.lower():
        raise ValueError(
            f"{raster.metadata.path} has axis {raster.axis_name!r}, "
            "not return periods"
        )
    for curve in raster.iter_curves(bounds):
        valid = np.isfinite(curve.axis_values) & np.isfinite(curve.values)
        periods = curve.axis_values[valid]
        values = curve.values[valid]
        if len(values) < 4:
            continue
        tabulated = TabulatedDistribution.from_return_periods(
            periods,
            values,
            tail=policy.tail,
        )
        try:
            distribution, base = _fit_curve(tabulated, policy)
        except ValueError as error:
            raise ValueError(
                f"failed to fit source pixel row={curve.row}, "
                f"column={curve.column}: {error}"
            ) from error
        geometry = Polygon(curve.boundary)
        source_id = _source_id(raster, curve)
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
                    "hazard_name": raster.metadata.hazard_type,
                    "horizon": raster.metadata.year,
                    "pathway": raster.metadata.scenario,
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


def canonicalize_os_climate(
    raster: ZarrRaster,
    policy: OSClimateIngestPolicy,
    *,
    bounds: Bounds | None = None,
) -> CanonicalHazardStream:
    """Return a lazy canonical stream for one selected raster."""
    metadata = _metadata(raster, policy)
    return CanonicalHazardStream(
        metadata=metadata,
        batches=_canonical_batches(raster, policy, metadata, bounds),
    )
