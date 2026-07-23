"""Local single-user provider placeholder."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalProvider:
    """Configuration for datasets rooted in a local directory."""

    root: Path
