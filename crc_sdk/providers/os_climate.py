"""OS-Climate inventory discovery and public S3 Zarr access."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType
from typing import Any, Optional
from urllib.request import urlopen

from crc_sdk.connectors.duckdb import DuckDBConnection
from crc_sdk.connectors.duckdb.zarr import RasterMetadata, ZarrRaster

DEFAULT_INVENTORY_URL = (
    "https://raw.githubusercontent.com/os-climate/hazard/main/"
    "src/inventories/hazard/inventory.json"
)


@dataclass(frozen=True)
class OSClimateResource:
    """One resource template from the OS-Climate hazard inventory."""

    inventory_index: int
    hazard_type: str
    indicator_id: str
    path: str
    model_gcm: str
    units: str
    source: str
    display_name: str
    store_netcdf_coords: bool
    parameters: Mapping[str, tuple[str, ...]]
    scenarios: Mapping[str, tuple[int, ...]]

    @property
    def key(self) -> str:
        return f"{self.inventory_index}:{self.path}"

    def resolve(
        self,
        *,
        scenario: str,
        year: int,
        parameters: Optional[Mapping[str, str]] = None,
    ) -> "OSClimateSelection":
        """Validate dimensions and resolve the concrete Zarr array path."""
        years = self.scenarios.get(scenario)
        if years is None or year not in years:
            available = ", ".join(
                f"{name}={list(values)}" for name, values in self.scenarios.items()
            )
            raise ValueError(
                f"{scenario}/{year} is unavailable for {self.key}; "
                f"available: {available}"
            )

        supplied = dict(parameters or {})
        expected = set(self.parameters)
        if set(supplied) != expected:
            raise ValueError(
                f"parameters for {self.key} must be exactly {sorted(expected)}"
            )
        for name, value in supplied.items():
            if value not in self.parameters[name]:
                raise ValueError(
                    f"unsupported {name}={value!r}; "
                    f"choose from {list(self.parameters[name])}"
                )

        values = {"scenario": scenario, "year": str(year), **supplied}
        missing = {
            name
            for _, name, _, _ in Formatter().parse(self.path)
            if name and name not in values
        }
        if missing:
            raise ValueError(f"unresolved path fields: {sorted(missing)}")
        return OSClimateSelection(
            resource=self,
            scenario=scenario,
            year=year,
            parameters=MappingProxyType(supplied),
            path=self.path.format_map(values),
        )


@dataclass(frozen=True)
class OSClimateSelection:
    """A resource with all path dimensions resolved."""

    resource: OSClimateResource
    scenario: str
    year: int
    parameters: Mapping[str, str]
    path: str


class OSClimateInventory:
    """Parsed OS-Climate resource inventory."""

    def __init__(self, resources: Sequence[OSClimateResource]) -> None:
        self.resources = tuple(resources)

    @classmethod
    def from_url(cls, url: str = DEFAULT_INVENTORY_URL) -> "OSClimateInventory":
        with urlopen(url) as response:
            return cls.from_dict(json.load(response))

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "OSClimateInventory":
        resources = []
        for index, item in enumerate(document.get("resources", ())):
            resources.append(
                OSClimateResource(
                    inventory_index=index,
                    hazard_type=str(item["hazard_type"]),
                    indicator_id=str(item["indicator_id"]),
                    path=str(item["path"]),
                    model_gcm=str(item.get("indicator_model_gcm") or ""),
                    units=str(item.get("units") or ""),
                    source=str(item.get("source") or ""),
                    display_name=str(item.get("display_name") or ""),
                    store_netcdf_coords=bool(item.get("store_netcdf_coords", False)),
                    parameters=MappingProxyType(
                        {
                            str(name): tuple(str(value) for value in values)
                            for name, values in item.get("params", {}).items()
                        }
                    ),
                    scenarios=MappingProxyType(
                        {
                            str(scenario["id"]): tuple(
                                int(year) for year in scenario["years"]
                            )
                            for scenario in item.get("scenarios", ())
                        }
                    ),
                )
            )
        return cls(resources)

    def find(
        self,
        *,
        hazard_type: Optional[str] = None,
        indicator_id: Optional[str] = None,
        model_gcm: Optional[str] = None,
        source: Optional[str] = None,
        path: Optional[str] = None,
    ) -> tuple[OSClimateResource, ...]:
        """Return resources matching exact inventory fields."""
        return tuple(
            resource
            for resource in self.resources
            if (hazard_type is None or resource.hazard_type == hazard_type)
            and (indicator_id is None or resource.indicator_id == indicator_id)
            and (model_gcm is None or resource.model_gcm == model_gcm)
            and (source is None or resource.source == source)
            and (path is None or resource.path == path)
        )

    def select(self, **filters: Optional[str]) -> OSClimateResource:
        """Return one unambiguous resource."""
        matches = self.find(**filters)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise LookupError(f"no OS-Climate resource matches {filters}")
        choices = "\n".join(
            f"- {item.key} | {item.model_gcm or 'unspecified GCM'} | {item.path}"
            for item in matches
        )
        raise LookupError(
            f"{len(matches)} OS-Climate resources match {filters}; "
            f"add model_gcm/source or choose a key:\n{choices}"
        )


class OSClimateProvider:
    """Open OS-Climate's anonymous S3-hosted Zarr resources."""

    def __init__(
        self,
        inventory: Optional[OSClimateInventory] = None,
        *,
        inventory_url: str = DEFAULT_INVENTORY_URL,
        bucket: str = "os-climate-physical-risk",
        root: str = "hazard-indicators/hazard.zarr",
        storage_options: Optional[Mapping[str, Any]] = None,
        connection: Optional[DuckDBConnection] = None,
    ) -> None:
        self._inventory = inventory
        self.inventory_url = inventory_url
        self.bucket = bucket
        self.root = root.strip("/")
        self.storage_options = dict(storage_options or {})
        self.connection = connection or DuckDBConnection()
        self._filesystem: Any = None

    @property
    def inventory(self) -> OSClimateInventory:
        if self._inventory is None:
            self._inventory = OSClimateInventory.from_url(self.inventory_url)
        return self._inventory

    def find(self, **filters: Optional[str]) -> tuple[OSClimateResource, ...]:
        return self.inventory.find(**filters)

    def select(self, **filters: Optional[str]) -> OSClimateResource:
        return self.inventory.select(**filters)

    def open(self, selection: OSClimateSelection) -> ZarrRaster:
        """Open metadata only; chunks remain remote until a scan executes."""
        try:
            import s3fs  # type: ignore[import-untyped]
            import zarr  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "OS-Climate access requires `pip install crc-sdk[connectors]`"
            ) from error

        if self._filesystem is None:
            options = {"anon": True, **self.storage_options}
            self._filesystem = s3fs.S3FileSystem(**options)
        path = selection.path
        if selection.resource.store_netcdf_coords:
            path = f"{path}/indicator"
        store = s3fs.S3Map(
            root=f"{self.bucket}/{self.root}/{path}",
            s3=self._filesystem,
            check=False,
        )
        array = zarr.open_array(store=store, mode="r")
        resource = selection.resource
        return ZarrRaster(
            array,
            RasterMetadata(
                hazard_type=resource.hazard_type,
                indicator_id=resource.indicator_id,
                scenario=selection.scenario,
                year=selection.year,
                units=resource.units,
                path=selection.path,
            ),
            connection=self.connection,
        )
