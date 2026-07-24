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

__all__ = [
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
    "quality_metrics",
]
