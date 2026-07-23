"""DuckDB connection and extension lifecycle."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class DuckDBConnection:
    """Configuration for a lazily created DuckDB connection."""

    database: Optional[str] = None
    read_only: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)

    def connect(self) -> Any:
        """Create a connection without imposing SDK-level resource limits."""
        try:
            import duckdb
        except ImportError as error:
            raise ImportError(
                "DuckDB support requires `pip install crc-sdk[connectors]`"
            ) from error

        return duckdb.connect(
            self.database or ":memory:",
            read_only=self.read_only,
            config=dict(self.config),
        )
