"""Connector protocols."""

from typing import Any, Protocol

from crc_sdk.types import HazardQuery


class HazardReader(Protocol):
    """Read a columnar hazard result from one external source."""

    def read(self, query: HazardQuery) -> Any:
        """Return an Arrow-compatible batch or relation."""
        ...
