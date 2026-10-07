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
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, get_args, runtime_checkable

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from crc_framework import (
    FittedDistribution,
    HurdleDistribution,
    QuantileFitDiagnostics,
    TabulatedDistribution,
    fit_distribution,
    fit_hurdle_quantiles,
    fit_quantiles,
)
from crc_framework.distributions import DistributionFamily

from crc_sdk._version import framework_version, platform_tag, sdk_version
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
from crc_sdk.types import (
    CurveFitProvenance,
    EnsembleDescriptor,
    HazardDatasetMetadata,
    ProbabilitySemantics,
    SourceProvenance,
    TemporalWindow,
)


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
class CurveSourceInfo:
    """Optional schema-1.3 metadata a curve source can offer its canonicalizer.

    A source that carries one (`BlockExtremaCurveSource.info`, set by the ERA5
    plan) gets licence/attribution, its temporal window, ensemble and
    probability semantics -- plus dataset-wide fit provenance -- recorded in
    the canonical file. Sources without one keep their existing metadata.
    """

    uri: str | None = None
    licence: str | None = None
    attribution: str | None = None
    retrieved_at: str | None = None
    checksum: str | None = None
    temporal_window: TemporalWindow | None = None
    ensemble: EnsembleDescriptor | None = None
    probability_semantics: ProbabilitySemantics | None = None


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
    creation_version: str = field(default_factory=sdk_version)
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
    # "sample_mle" fits the per-block samples themselves with crc-framework's
    # maximum-likelihood `fit_distribution` instead of least-squares on the
    # plotting-position knots. "sample_lmoments" uses the standalone L-moment
    # estimator for GEV/Gumbel. Both are only for sources whose values are genuine
    # samples (`values_are_samples`), never for probability-labelled inputs.
    fit_method: Literal["quantile_least_squares", "sample_mle", "sample_lmoments"] = (
        "quantile_least_squares"
    )
    # Persist one row per source pixel (fitted, or skipped with its reason) to
    # this Parquet path (ADR-0007).
    diagnostics: str | Path | None = None

    def __post_init__(self) -> None:
        if self.fit_method not in (
            "quantile_least_squares",
            "sample_mle",
            "sample_lmoments",
        ):
            raise ValueError(
                "fit_method must be quantile_least_squares, sample_mle "
                "or sample_lmoments"
            )
        if self.fit_method != "quantile_least_squares" and (
            self.hurdle is not None
            or self.maximum_normalized_rmse is not None
            or self.maximum_absolute_residual is not None
        ):
            raise ValueError(
                f"{self.fit_method} does not support hurdle fits "
                "or quantile quality gates"
            )
        if self.fit_method == "sample_lmoments" and self.family not in (
            "genextreme",
            "gumbel_r",
            "gumbel_l",
        ):
            raise ValueError(
                "sample_lmoments supports genextreme, gumbel_r and gumbel_l only"
            )
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


def _source_id(provider: str, path: str, row: int, column: int) -> str:
    identity = f"{provider}\0{path}\0{row}\0{column}".encode()
    return sha256(identity).hexdigest()


def _metadata(
    source: CurveSource, policy: CurveFitIngestPolicy, provider: str
) -> HazardDatasetMetadata:
    values = source.metadata
    support = getattr(source, "return_period_support", None)
    info: CurveSourceInfo | None = getattr(source, "info", None)
    if info is None:
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
            uri=info.uri or values.path,
            version=policy.source_version,
            licence=info.licence,
            attribution=info.attribution,
            retrieved_at=info.retrieved_at,
            checksum=info.checksum,
        ),
        fitting=_fit_provenance(policy),
        probability_semantics=info.probability_semantics,
        temporal_window=info.temporal_window,
        ensemble=info.ensemble,
    )


def _fit_provenance(policy: CurveFitIngestPolicy) -> CurveFitProvenance:
    sample_fit = policy.fit_method == "sample_mle"
    return CurveFitProvenance(
        method=policy.fit_method,
        initialization="lmoments" if sample_fit else None,
        input_kind="samples",
        platform=platform_tag(),
        crc_framework_version=framework_version(),
        families=(policy.family,),
        atom_policy="none",
        maximum_normalized_rmse=policy.maximum_normalized_rmse,
        maximum_absolute_residual=policy.maximum_absolute_residual,
        on_fit_failure=policy.on_fit_failure,
    )


def _fit_curve(
    tabulated: TabulatedDistribution,
    policy: CurveFitIngestPolicy,
) -> tuple[Any, Any, Any]:
    distribution: FittedDistribution | HurdleDistribution
    diagnostics: QuantileFitDiagnostics
    if policy.fit_method != "quantile_least_squares":
        raise AssertionError("sample fits go through _fit_samples")
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
    return distribution, base, diagnostics


def _fit_samples(values: Any, policy: CurveFitIngestPolicy) -> tuple[Any, Any]:
    """Fit one pixel's genuine block samples using the selected estimator."""
    if policy.fit_method == "sample_lmoments":
        try:
            result = fit_distribution(values, family=policy.family, method="lmoments")
        except TypeError as error:
            raise RuntimeError(
                "sample_lmoments requires crc-framework with L-moments support; "
                "install crc-framework 0.3.0 or newer in this environment"
            ) from error
    else:
        result = fit_distribution(values, family=policy.family)
    return result.distribution, result.distribution


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
    if policy.fit_method != "quantile_least_squares" and not getattr(
        source, "values_are_samples", False
    ):
        raise ValueError(
            f"fit_method={policy.fit_method!r} needs a source whose curve values are "
            "genuine samples; probability-labelled inputs use quantile fits"
        )
    writer = None
    records: list[dict[str, Any]] = []
    if policy.diagnostics is not None:
        # Imported here: `crc_sdk.fitting` imports this module.
        from crc_sdk.fitting.diagnostics import DiagnosticsWriter

        writer = DiagnosticsWriter(policy.diagnostics)

    def record_cell(
        row: int,
        column: int,
        blocks: int,
        outcome: str,
        *,
        cell: int | None = None,
        source_id: str | None = None,
        reason: str | None = None,
        message: str | None = None,
        base: Any = None,
        quality: Any = None,
    ) -> None:
        if writer is None:
            return
        records.append(
            {
                "cell_index": cell,
                "source_id": source_id
                or _source_id(provider, source.metadata.path, row, column),
                "hazard_name": source.metadata.hazard_type,
                "horizon": source.metadata.year,
                "pathway": source.metadata.scenario,
                "outcome": outcome,
                "curve_type": base.family if base is not None else None,
                "normalized_rmse": getattr(quality, "normalized_rmse", None),
                "maximum_absolute_residual": getattr(
                    quality, "maximum_absolute_residual", None
                ),
                "attempted_families": [policy.family],
                "failed_families": [policy.family] if outcome != "fitted" else [],
                "fallback": False,
                "reason": reason,
                "message": message or f"{blocks} blocks",
                "treatment": policy.fit_method,
                "minimum_informative_value": None,
            }
        )

    def record(curve: RasterCurve, outcome: str, **details: Any) -> None:
        record_cell(curve.row, curve.column, len(curve.values), outcome, **details)

    if writer is not None and hasattr(source, "skip_sink"):
        # Cells a source drops before yielding a curve (no valid blocks, too
        # few of them) are traced too, not only the ones that fail to fit.
        source.skip_sink = lambda row, column, reason, message: record_cell(
            row, column, 0, "skipped", reason=reason, message=message
        )

    try:
        for curve in source.iter_curves(bounds):
            valid = np.isfinite(curve.axis_values) & np.isfinite(curve.values)
            periods = curve.axis_values[valid]
            values = curve.values[valid]
            if len(values) < 4:
                record(curve, "skipped", reason="too_few_knots")
                continue
            try:
                quality: Any = None
                if policy.fit_method != "quantile_least_squares":
                    distribution, base = _fit_samples(values, policy)
                else:
                    tabulated = TabulatedDistribution.from_return_periods(
                        periods,
                        values,
                        tail=policy.tail,
                    )
                    distribution, base, quality = _fit_curve(tabulated, policy)
            except ValueError as error:
                # Non-monotonic quantiles (e.g. small per-return-period modeling
                # noise near a DEM sink or tile edge) fail the same way an
                # unconverged/out-of-tolerance fit does -- both mean "this pixel
                # can't be usefully fitted," so on_fit_failure governs both.
                if policy.on_fit_failure == "skip":
                    record(curve, "skipped", reason="fit_failed", message=str(error))
                    continue
                raise ValueError(
                    f"failed to fit source pixel row={curve.row}, "
                    f"column={curve.column}: {error}"
                ) from error
            geometry = Polygon(curve.boundary)
            source_id = _source_id(
                provider, source.metadata.path, curve.row, curve.column
            )
            cells = intersecting_cells(geometry, policy.h3_resolution)
            if not cells:
                record(curve, "skipped", reason="no_cells", source_id=source_id)
                continue
            record(
                curve,
                "fitted",
                cell=int(cells[0]),
                source_id=source_id,
                base=base,
                quality=quality,
            )
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
            if writer is not None and len(records) >= 65_536:
                writer.write(records)
                records.clear()
        if hazard_rows:
            hazards = validate_hazard_table(
                pa.Table.from_pylist(hazard_rows, schema=hazard_schema),
                metadata=metadata,
            )
            yield CanonicalHazardBatch(hazard_rows=hazards)
        if writer is not None:
            writer.write(records)
            writer.finish()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise


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
