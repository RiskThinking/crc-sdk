"""Dataset-level metadata models."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HazardDatasetMetadata(BaseModel):
    """Metadata shared by every row in one canonical hazard dataset."""

    model_config = ConfigDict(frozen=True)

    schema_version: str
    h3_resolution: int = Field(ge=0, le=15)
    probability_convention: str = "non_exceedance"
    value_unit: str
    geometry_crs: str = "EPSG:4326"
    source: Optional[str] = None
