"""DuckDB connection and extension lifecycle."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DuckDBConnection:
    """Configuration for a lazily created DuckDB connection."""

    database: Optional[str] = None
    read_only: bool = False

