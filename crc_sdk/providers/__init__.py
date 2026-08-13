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
from .jrc_edo import EDO_DATASETS, SMI, EDODataset, EDOProvider, edo_dataset
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
    "EDO_DATASETS",
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
    "edo_dataset",
]
