"""Storage configuration models."""

from pydantic import BaseModel, ConfigDict


class StorageLocation(BaseModel):
    """Logical location resolved by a provider."""

    model_config = ConfigDict(frozen=True)

    uri: str

