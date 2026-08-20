"""Curve-fitting API backed by ``crc_framework``."""

from crc_framework import (
    FitConstraints,
    FitDiagnostics,
    FitResult,
    HurdleQuantileFitDiagnostics,
    HurdleQuantileFitResult,
    QuantileFitDiagnostics,
    QuantileFitResult,
    fit_all,
    fit_distribution,
    fit_hurdle_quantiles,
    fit_quantiles,
    quality_metrics,
)

from .workflows import (
    CDFColumnSchema,
    CDFCurveFitPolicy,
    CDFFitResult,
    CDFFitSummary,
    fit_cdf_quantile_batches,
)

__all__ = [
    "CDFColumnSchema",
    "CDFCurveFitPolicy",
    "CDFFitResult",
    "CDFFitSummary",
    "FitConstraints",
    "FitDiagnostics",
    "FitResult",
    "HurdleQuantileFitDiagnostics",
    "HurdleQuantileFitResult",
    "QuantileFitDiagnostics",
    "QuantileFitResult",
    "fit_all",
    "fit_distribution",
    "fit_hurdle_quantiles",
    "fit_quantiles",
    "fit_cdf_quantile_batches",
    "quality_metrics",
]
