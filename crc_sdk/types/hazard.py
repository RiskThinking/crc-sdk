"""Hazard query and curve parameter models."""

from typing import Optional

from pydantic import BaseModel, ConfigDict


class CurveParameters(BaseModel):
    """Parameters for a distribution family supported by the core."""

    model_config = ConfigDict(frozen=True)

    curve_type: str
    curve_shape: Optional[float] = None
    curve_location: float
    curve_scale: float


class HazardQuery(BaseModel):
    """Filters used to select hazard curves from a provider."""

    model_config = ConfigDict(frozen=True)

    hazard_name: str
    horizon: Optional[int] = None
    pathway: Optional[str] = None

