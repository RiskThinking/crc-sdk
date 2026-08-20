"""Hazard query and fitted-curve parameter models."""

from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

from crc_framework.distributions import (
    DistributionFamily,
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CurveParameters(BaseModel):
    """Parameters for a distribution family supported by the core."""

    model_config = ConfigDict(frozen=True)

    curve_kind: Literal["fitted", "hurdle", "point_mass"] = "fitted"
    curve_type: Union[DistributionFamily, Literal["point_mass"]]
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
        if self.curve_kind == "point_mass":
            if self.curve_type != "point_mass":
                raise ValueError("point-mass curves require curve_type='point_mass'")
            if self.curve_shape is not None or self.curve_scale != 0.0:
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
    ) -> Union[FittedDistribution, HurdleDistribution, PointMassDistribution]:
        """Reconstruct the framework distribution represented by these fields."""
        if self.curve_kind == "point_mass":
            return PointMassDistribution(self.curve_location)
        if self.curve_type == "point_mass":
            raise AssertionError("curve kind and type were validated")
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
    cell_index: Optional[int] = Field(
        default=None,
        ge=0,
        le=(1 << 64) - 1,
    )
