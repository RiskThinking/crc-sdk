"""Hazard query and fitted-curve parameter models."""

from collections.abc import Mapping
from typing import Any, Literal, Optional, Union, cast

from crc_framework.distributions import (
    DistributionFamily,
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
    TabulatedDistribution,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator


class NoDataCurveError(ValueError):
    """Raised when distribution reconstruction is requested for a no-data row."""


class CurveParameters(BaseModel):
    """Parameters for a distribution family supported by the core."""

    model_config = ConfigDict(frozen=True)

    curve_kind: Literal["fitted", "hurdle", "point_mass", "tabulated", "no_data"] = (
        "fitted"
    )
    curve_type: Union[
        DistributionFamily,
        Literal[
            "point_mass",
            "linear_probability",
            "below_effective_resolution",
            "insufficient_informative_support",
            "degenerate_effective_range",
            "scientific_exclusion",
        ],
    ]
    curve_shape: Optional[float] = None
    curve_location: Optional[float] = None
    curve_scale: Optional[float] = None
    curve_atom_probability: Optional[float] = None
    curve_atom_location: Optional[float] = None
    curve_probabilities: Optional[tuple[float, ...]] = None
    curve_values: Optional[tuple[float, ...]] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_shape(cls, value: Any) -> Any:
        """Gumbel families have no shape parameter in the canonical form."""
        if isinstance(value, Mapping) and value.get("curve_type") in {
            "gumbel_r",
            "gumbel_l",
        }:
            return {**value, "curve_shape": None}
        return value

    @model_validator(mode="after")
    def validate_distribution(self) -> "CurveParameters":
        """Delegate parameter validation to the framework's public API."""
        scalar_optional = (
            self.curve_shape,
            self.curve_location,
            self.curve_scale,
            self.curve_atom_probability,
            self.curve_atom_location,
        )
        if self.curve_kind == "no_data":
            if self.curve_type not in {
                "below_effective_resolution",
                "insufficient_informative_support",
                "degenerate_effective_range",
                "scientific_exclusion",
            }:
                raise ValueError(
                    "no-data curves require a no-data reason as curve_type"
                )
            if any(value is not None for value in scalar_optional) or any(
                value is not None
                for value in (self.curve_probabilities, self.curve_values)
            ):
                raise ValueError("no-data curves must not define curve parameters")
            return self
        if self.curve_kind == "tabulated":
            if self.curve_type != "linear_probability":
                raise ValueError(
                    "tabulated curves require curve_type='linear_probability'"
                )
            if any(value is not None for value in scalar_optional):
                raise ValueError("tabulated curves must not define scalar parameters")
            if self.curve_probabilities is None or self.curve_values is None:
                raise ValueError("tabulated curves require probability and value knots")
            self.to_distribution()
            return self
        if self.curve_probabilities is not None or self.curve_values is not None:
            raise ValueError("non-tabulated curves must not define tabulated knots")
        if self.curve_kind == "point_mass":
            if self.curve_type != "point_mass":
                raise ValueError("point-mass curves require curve_type='point_mass'")
            if (
                self.curve_shape is not None
                or self.curve_location is None
                or self.curve_scale != 0.0
            ):
                raise ValueError("point-mass curves require no shape and zero scale")
            if (
                self.curve_atom_probability != 1.0
                or self.curve_atom_location != self.curve_location
            ):
                raise ValueError(
                    "point-mass curves require probability one at curve_location"
                )
            self.to_distribution()
            return self
        if self.curve_type == "point_mass":
            raise ValueError("curve_type='point_mass' requires curve_kind='point_mass'")
        if self.curve_type not in {
            "genextreme",
            "weibull_min",
            "weibull_max",
            "skewnorm",
            "gumbel_r",
            "gumbel_l",
            "genpareto",
        }:
            raise ValueError("fitted and hurdle curves require a fitted family")
        if self.curve_location is None or self.curve_scale is None:
            raise ValueError("fitted and hurdle curves require location and scale")
        if self.curve_kind == "fitted" and (
            self.curve_atom_probability is not None
            or self.curve_atom_location is not None
        ):
            raise ValueError("fitted curves must not define hurdle atom fields")
        if self.curve_kind == "hurdle" and (
            self.curve_atom_probability is None or self.curve_atom_location is None
        ):
            raise ValueError("hurdle curves require atom probability and atom location")
        self.to_distribution()
        return self

    def to_distribution(
        self,
    ) -> Union[
        FittedDistribution,
        HurdleDistribution,
        PointMassDistribution,
        TabulatedDistribution,
    ]:
        """Reconstruct the framework distribution represented by these fields."""
        if self.curve_kind == "no_data":
            raise NoDataCurveError(f"curve has no data: {self.curve_type}")
        if self.curve_kind == "tabulated":
            if self.curve_probabilities is None or self.curve_values is None:
                raise AssertionError("tabulated knots were validated")
            return TabulatedDistribution(
                self.curve_probabilities,
                self.curve_values,
                interpolation="linear_probability",
            )
        if self.curve_kind == "point_mass":
            if self.curve_location is None:
                raise AssertionError("point-mass location was validated")
            return PointMassDistribution(self.curve_location)
        if self.curve_type == "point_mass":
            raise AssertionError("curve kind and type were validated")
        if self.curve_location is None or self.curve_scale is None:
            raise AssertionError("fitted scalar parameters were validated")
        base = FittedDistribution.from_parameters(
            cast(DistributionFamily, self.curve_type),
            location=self.curve_location,
            scale=self.curve_scale,
            shape=self.curve_shape,
        )
        if self.curve_kind == "fitted":
            return base
        if self.curve_atom_probability is None or self.curve_atom_location is None:
            raise AssertionError("hurdle fields were validated")
        return HurdleDistribution(
            base,
            atom_probability=self.curve_atom_probability,
            atom_location=self.curve_atom_location,
        )


class HazardQuery(BaseModel):
    """Filters used to select hazard curves from a provider."""

    model_config = ConfigDict(frozen=True)

    hazard_name: str
    horizon: Optional[int] = None
    pathway: Optional[str] = None
    cell_index: Optional[int] = Field(
        default=None,
        ge=0,
        le=(1 << 64) - 1,
    )
