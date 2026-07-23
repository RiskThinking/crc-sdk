"""Curve-fitting API backed by ``crc_framework``."""

from crc_framework import (
    FitConstraints,
    FitDiagnostics,
    FitResult,
    fit_all,
    fit_distribution,
    quality_metrics,
)

__all__ = [
    "FitConstraints",
    "FitDiagnostics",
    "FitResult",
    "fit_all",
    "fit_distribution",
    "quality_metrics",
]
