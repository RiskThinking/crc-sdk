"""Discoverable entry points for lazy agricultural data workflows."""

from crc_sdk.connectors.agriculture import FTWFields, USDACropland


class AgriculturalLayer:
    """Factories for agricultural layers that compose with DuckDB pipelines."""

    @staticmethod
    def usda_cdl(*, version: str = "v0.1.0") -> USDACropland:
        if version != "v0.1.0":
            raise ValueError("the Source Cooperative USDA CDL version is v0.1.0")
        return USDACropland(repository=f"usda-cropland-data-layer/{version}.icechunk")

    @staticmethod
    def ftw_fields(*, version: str = "alpha") -> FTWFields:
        if version != "alpha":
            raise ValueError("the Source Cooperative FTW vector version is alpha")
        return FTWFields()
