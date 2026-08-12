"""Storage provider interfaces and implementations."""

from .jrc import (
    EFAS,
    GLOFAS,
    JRC_DATASETS,
    JRCProvider,
    JRCRasterDataset,
    JRCRasterResource,
    jrc_dataset,
)
from .jrc_edo import SMI, EDODataset, EDOProvider
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
    "EDODataset",
    "EDOProvider",
    "EFAS",
    "GLOFAS",
    "JRCProvider",
    "JRC_DATASETS",
    "JRCRasterDataset",
    "JRCRasterResource",
    "jrc_dataset",
    "LocalProvider",
    "OSClimateInventory",
    "OSClimateProvider",
    "OSClimateResource",
    "OSClimateSelection",
    "Provider",
    "SMI",
]
