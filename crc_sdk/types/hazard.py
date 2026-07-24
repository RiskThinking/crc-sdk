"""Hazard query and fitted-curve parameter models."""

from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

from crc_framework.distributions import (
    DistributionFamily,
    FittedDistribution,
    HurdleDistribution,
)
from pydantic import BaseModel, ConfigDict, model_validator


class CurveParameters(BaseModel):
    """Parameters for a distribution family supported by the core."""

    model_config = ConfigDict(frozen=True)

    curve_kind: Literal["fitted", "hurdle"] = "fitted"
    curve_type: DistributionFamily
    curve_shape: Optional[float] = None
    curve_location: float
    curve_scale: float
    curve_atom_probability: Optional[float] = None
    curve_atom_location: Optional[float] = None

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
        if self.curve_kind == "fitted" and (
            self.curve_atom_probability is not None
            or self.curve_atom_location is not None
        ):
            raise ValueError("fitted curves must not define hurdle atom fields")
        if self.curve_kind == "hurdle" and (
            self.curve_atom_probability is None
            or self.curve_atom_location is None
        ):
            raise ValueError(
                "hurdle curves require atom probability and atom location"
            )
        self.to_distribution()
        return self

    def to_distribution(self) -> Union[FittedDistribution, HurdleDistribution]:
        """Reconstruct the framework distribution represented by these fields."""
        base = FittedDistribution.from_parameters(
            self.curve_type,
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
