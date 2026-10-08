"""Fit policy shared by block-extrema recipes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

    Unspecified ``family`` and ``fit_method`` inherit the recipe's defaults.
    Without a recipe, fitting uses GEV with sample L-moments.
    """

    h3_resolution: int = 5
    family: DistributionFamily | None = None
    producer: str = "crc-sdk"
    creation_version: str = field(default_factory=sdk_version)
    minimum_years: int = 20
    on_fit_failure: Literal["raise", "skip"] = "skip"
    maximum_normalized_rmse: float | None = None
    maximum_absolute_residual: float | None = None
    fit_method: (
        Literal["quantile_least_squares", "sample_mle", "sample_lmoments"] | None
    ) = None
    diagnostics: str | Path | None = None
    require_sample_support: bool = True
    validation_return_periods: tuple[float, ...] = (2, 5, 10, 20, 50, 100)
    minimum_return_value: float | None = None
    maximum_return_value: float | None = None
    fallback_family: Literal["gumbel_r", "gumbel_l"] | None = None

    def __post_init__(self) -> None:
        if self.minimum_years < 4:
            raise ValueError("minimum_years must be at least four")

    @classmethod
    def curated(
        cls,
        *,
        h3_resolution: int = 5,
        minimum_years: int = 20,
        family: DistributionFamily | None = None,
        fit_method: Literal["quantile_least_squares", "sample_mle", "sample_lmoments"]
        | None = None,
        diagnostics: str | Path | None = None,
        require_sample_support: bool = True,
        validation_return_periods: tuple[float, ...] = (2, 5, 10, 20, 50, 100),
        minimum_return_value: float | None = None,
        maximum_return_value: float | None = None,
        fallback_family: Literal["gumbel_r", "gumbel_l"] | None = None,
    ) -> BlockExtremaPolicy:
        return cls(
            h3_resolution=h3_resolution,
            minimum_years=minimum_years,
            family=family,
            fit_method=fit_method,
            diagnostics=diagnostics,
            require_sample_support=require_sample_support,
            validation_return_periods=validation_return_periods,
            minimum_return_value=minimum_return_value,
            maximum_return_value=maximum_return_value,
            fallback_family=fallback_family,
        )

    def resolve(
        self,
        *,
        family: DistributionFamily = "genextreme",
        fit_method: Literal[
            "quantile_least_squares", "sample_mle", "sample_lmoments"
        ] = "sample_lmoments",
    ) -> BlockExtremaPolicy:
        """Fill unspecified fitting choices without changing explicit overrides."""
        return replace(
            self,
            family=family if self.family is None else self.family,
            fit_method=fit_method if self.fit_method is None else self.fit_method,
        )

    def ingest_policy(
        self,
        *,
        tail: Literal["upper", "lower"],
        value_semantics: str,
        source_version: str,
    ) -> CurveFitIngestPolicy:
        method = "sample_lmoments" if self.fit_method is None else self.fit_method
        sample_fit = method != "quantile_least_squares"
        return CurveFitIngestPolicy(
            h3_resolution=self.h3_resolution,
            family="genextreme" if self.family is None else self.family,
            producer=self.producer,
            creation_version=self.creation_version,
            tail=tail,
            value_semantics=value_semantics,
            source_version=source_version,
            maximum_normalized_rmse=self.maximum_normalized_rmse,
            maximum_absolute_residual=self.maximum_absolute_residual,
            on_fit_failure=self.on_fit_failure,
            fit_method=method,
            diagnostics=self.diagnostics,
            require_sample_support=self.require_sample_support if sample_fit else False,
            validation_return_periods=self.validation_return_periods,
            minimum_return_value=self.minimum_return_value,
            maximum_return_value=self.maximum_return_value,
            fallback_family=self.fallback_family,
        )

    def validate_years(self, years: tuple[int, ...]) -> None:
        if len(years) < self.minimum_years:
            raise ValueError(
                f"curated block-extrema fitting requires at least "
                f"{self.minimum_years} complete years; received {len(years)}"
            )
