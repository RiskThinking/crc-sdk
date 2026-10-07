"""Fit policy shared by block-extrema recipes (ERA5 today, ISD and CMIP6 next)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from crc_framework.distributions import DistributionFamily

from crc_sdk._version import sdk_version
from crc_sdk.connectors import CurveFitIngestPolicy


@dataclass(frozen=True)
class BlockExtremaPolicy:
    """Safe defaults for fitting per-block (annual) extremes.

    Modelled on `EDODroughtPolicy.curated()`: a minimum number of complete
    blocks before anything is fitted, and ``on_fit_failure="skip"`` so cells
    that cannot be usefully fitted (a constant record, a masked-out sea cell)
    are dropped rather than aborting an area ingest -- with ``diagnostics``
    set, every dropped cell is traced to a sidecar row and a reason.
    """

    h3_resolution: int = 5
    family: DistributionFamily = "gumbel_r"
    producer: str = "crc-sdk"
    creation_version: str = field(default_factory=sdk_version)
    minimum_years: int = 20
    on_fit_failure: Literal["raise", "skip"] = "skip"
    maximum_normalized_rmse: float | None = None
    maximum_absolute_residual: float | None = None
    fit_method: Literal["quantile_least_squares", "sample_mle", "sample_lmoments"] = (
        "quantile_least_squares"
    )
    diagnostics: str | Path | None = None

    def __post_init__(self) -> None:
        if self.minimum_years < 4:
            raise ValueError("minimum_years must be at least four")

    @classmethod
    def curated(
        cls,
        *,
        h3_resolution: int = 5,
        minimum_years: int = 20,
    ) -> BlockExtremaPolicy:
        return cls(h3_resolution=h3_resolution, minimum_years=minimum_years)

    def ingest_policy(
        self,
        *,
        tail: Literal["upper", "lower"],
        value_semantics: str,
        source_version: str,
    ) -> CurveFitIngestPolicy:
        return CurveFitIngestPolicy(
            h3_resolution=self.h3_resolution,
            family=self.family,
            producer=self.producer,
            creation_version=self.creation_version,
            tail=tail,
            value_semantics=value_semantics,
            source_version=source_version,
            maximum_normalized_rmse=self.maximum_normalized_rmse,
            maximum_absolute_residual=self.maximum_absolute_residual,
            on_fit_failure=self.on_fit_failure,
            fit_method=self.fit_method,
            diagnostics=self.diagnostics,
        )

    def validate_years(self, years: tuple[int, ...]) -> None:
        if len(years) < self.minimum_years:
            raise ValueError(
                f"curated block-extrema fitting requires at least "
                f"{self.minimum_years} complete years; received {len(years)}"
            )
