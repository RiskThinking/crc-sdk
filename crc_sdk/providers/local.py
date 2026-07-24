"""Local single-user provider placeholder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crc_sdk.types import HazardDatasetMetadata, HazardQuery


@dataclass(frozen=True)
class LocalProvider:
    """Read one canonical Parquet file from local storage."""

    source: str | Path

    def list_hazards(self) -> tuple[str, ...]:
        from crc_sdk.connectors.parquet import read_hazard_dataset

        table = read_hazard_dataset(self.source, columns=["hazard_name"])
        return tuple(sorted(set(table["hazard_name"].to_pylist())))

    def metadata(self, hazard_name: str) -> HazardDatasetMetadata:
        from crc_sdk.connectors.parquet import (
            read_hazard_dataset,
            read_hazard_metadata,
        )

        table = read_hazard_dataset(self.source, columns=["hazard_name"])
        if hazard_name not in set(table["hazard_name"].to_pylist()):
            raise LookupError(f"unknown hazard {hazard_name!r}")
        return read_hazard_metadata(self.source)

    def read(self, query: HazardQuery) -> Any:
        from crc_sdk.connectors.parquet import read_hazard_dataset

        return read_hazard_dataset(self.source, query)
