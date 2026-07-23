"""Storage provider interfaces and implementations."""

from .local import LocalProvider
from .protocol import Provider

__all__ = ["LocalProvider", "Provider"]
