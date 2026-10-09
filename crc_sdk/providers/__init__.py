"""Storage provider interfaces and implementations."""

from .crc_open import CRCOpenFixtureWarning, CRCOpenHazards
from .era5 import (
    ARCO_0P25,
    ERA5_RECIPES,
    ERA5_STORES,
    WB2_1P5,
    ERA5Provider,
    ERA5Recipe,
    ERA5Store,
    era5_recipe,
    era5_store,
)
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
    "CRCOpenFixtureWarning",
    "CRCOpenHazards",
    "ARCO_0P25",
    "ERA5_RECIPES",
    "ERA5_STORES",
    "ERA5Provider",
    "ERA5Recipe",
    "ERA5Store",
    "WB2_1P5",
    "era5_recipe",
    "era5_store",
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
