"""Domain-facing provider protocol."""

from collections.abc import Sequence
from typing import Any, Protocol

from crc_sdk.types import HazardDatasetMetadata, HazardQuery


class Provider(Protocol):
    """Discover and read hazard datasets from a storage location."""

    def list_hazards(self) -> Sequence[str]:
        """Return the available hazard names."""
        ...

    def metadata(self, hazard_name: str) -> HazardDatasetMetadata:
        """Return metadata for one hazard dataset."""
        ...

    def read(self, query: HazardQuery) -> Any:
        """Return an Arrow-compatible batch or relation."""
        ...

