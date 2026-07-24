"""Dataset-level metadata models."""

from collections.abc import Mapping
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crc_sdk.schema import (
    HAZARD_ROW_KEY,
    HAZARD_SORT_ORDER,
)

PARQUET_METADATA_KEY = "crc.hazard.metadata"


class SourceProvenance(BaseModel):
    """Stable identification of the external dataset used during ingest."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    uri: Optional[str] = None
    version: Optional[str] = None


class HazardDatasetMetadata(BaseModel):
    """Metadata shared by every row in one canonical hazard dataset."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    h3_resolution: int = Field(ge=0, le=15)
    probability_convention: Literal["non_exceedance"] = "non_exceedance"
    value_unit: str = Field(min_length=1)
    value_semantics: str = Field(min_length=1)
    geometry_encoding: Literal["WKB"] = "WKB"
    geometry_crs: str = "EPSG:4326"
    producer: str = Field(min_length=1)
    source: SourceProvenance
    creation_version: str = Field(min_length=1)
    row_key: tuple[str, ...] = HAZARD_ROW_KEY
    sort_order: tuple[str, ...] = HAZARD_SORT_ORDER

    @model_validator(mode="after")
    def validate_contract_constants(self) -> "HazardDatasetMetadata":
        if self.row_key != HAZARD_ROW_KEY:
            raise ValueError(f"row_key must be {HAZARD_ROW_KEY!r}")
        if self.sort_order != HAZARD_SORT_ORDER:
            raise ValueError(f"sort_order must be {HAZARD_SORT_ORDER!r}")
        return self

    def to_json_bytes(self) -> bytes:
        """Serialize the complete Parquet metadata payload."""
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_json_bytes(cls, value: bytes) -> "HazardDatasetMetadata":
        return cls.model_validate_json(value)

    def to_parquet_metadata(
        self, existing: Optional[Mapping[bytes, bytes]] = None
    ) -> dict[bytes, bytes]:
        result = dict(existing or {})
        result[PARQUET_METADATA_KEY.encode("utf-8")] = self.to_json_bytes()
        return result

    @classmethod
    def from_parquet_metadata(
        cls, metadata: Optional[Mapping[bytes, bytes]]
    ) -> "HazardDatasetMetadata":
        key = PARQUET_METADATA_KEY.encode("utf-8")
        if metadata is None or key not in metadata:
            raise ValueError("Parquet schema is missing canonical hazard metadata")
        return cls.from_json_bytes(metadata[key])
