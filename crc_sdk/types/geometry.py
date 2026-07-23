"""Serialized geometry metadata models."""

from pydantic import BaseModel, ConfigDict


class GeometryMetadata(BaseModel):
    """Geometry encoding information stored with a dataset."""

    model_config = ConfigDict(frozen=True)

    encoding: str = "WKB"
    crs: str = "EPSG:4326"
