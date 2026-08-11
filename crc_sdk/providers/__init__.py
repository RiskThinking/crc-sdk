"""Storage provider interfaces and implementations."""

from .jrc import EFAS, GLOFAS, JRCProvider, JRCRasterDataset
from .local import LocalProvider
from .os_climate import (
    DEFAULT_INVENTORY_URL,
    OSClimateInventory,
    OSClimateProvider,
    OSClimateResource,
    OSClimateSelection,
)
from .protocol import Provider

__all__ = [
    "DEFAULT_INVENTORY_URL",
    "EFAS",
    "GLOFAS",
    "JRCProvider",
    "JRCRasterDataset",
    "LocalProvider",
    "OSClimateInventory",
    "OSClimateProvider",
    "OSClimateResource",
    "OSClimateSelection",
    "Provider",
]
