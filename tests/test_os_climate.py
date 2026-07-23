from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from crc_sdk.connectors.duckdb import RasterMetadata, ZarrRaster
from crc_sdk.providers import OSClimateInventory


class FakeZarrArray:
    def __init__(self) -> None:
        self.data = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        self.shape = self.data.shape
        self.chunks = (2, 2, 2)
        self.attrs = {
            "index_name": "return period (years)",
            "index_values": [10, 100],
            "transform_mat3x3": [1, 0, 0, 0, -1, 2, 0, 0, 1],
        }
        self.reads: list[Any] = []

    def __getitem__(self, key: Any) -> Any:
        self.reads.append(key)
        return self.data[key]

    def get_coordinate_selection(self, coordinates: Any) -> Any:
        self.reads.append(coordinates)
        return self.data[coordinates]


def test_inventory_validates_parameters_and_resolves_paths() -> None:
    inventory = OSClimateInventory.from_dict(
        {
            "resources": [
                {
                    "hazard_type": "ChronicHeat",
                    "indicator_id": "days_tas/above/{temp_c}c",
                    "path": "heat/{temp_c}/{gcm}/{scenario}/{year}",
                    "indicator_model_gcm": "{gcm}",
                    "params": {"temp_c": ["35"], "gcm": ["Model-A"]},
                    "scenarios": [{"id": "ssp585", "years": [2050]}],
                    "units": "days/year",
                }
            ]
        }
    )

    resource = inventory.select(
        hazard_type="ChronicHeat",
        indicator_id="days_tas/above/{temp_c}c",
    )
    selection = resource.resolve(
        scenario="ssp585",
        year=2050,
        parameters={"temp_c": "35", "gcm": "Model-A"},
    )

    assert selection.path == "heat/35/Model-A/ssp585/2050"
    with pytest.raises(ValueError, match="must be exactly"):
        resource.resolve(scenario="ssp585", year=2050)
    with pytest.raises(ValueError, match="unsupported temp_c"):
        resource.resolve(
            scenario="ssp585",
            year=2050,
            parameters={"temp_c": "40", "gcm": "Model-A"},
        )


def test_scan_is_lazy_and_streams_into_duckdb() -> None:
    array = FakeZarrArray()
    raster = ZarrRaster(
        array,
        RasterMetadata(
            hazard_type="RiverineInundation",
            indicator_id="flood_depth",
            scenario="historical",
            year=1980,
            units="metres",
            path="test/flood",
        ),
    )

    scan = raster.scan(batch_rows=4)
    relation = scan.relation()
    assert array.reads == []

    count, total = relation.aggregate(
        "count(*) AS row_count, sum(value) AS total"
    ).fetchone()
    assert count == 12
    assert total == pytest.approx(66.0)
    assert array.reads


def test_point_read_fetches_only_requested_curve() -> None:
    array = FakeZarrArray()
    raster = ZarrRaster(
        array,
        RasterMetadata(
            hazard_type="Wind",
            indicator_id="max_speed",
            scenario="historical",
            year=2010,
            units="m/s",
            path="test/wind",
        ),
    )

    periods, values = raster.point_values(0.2, 1.8)
    assert periods.tolist() == [10.0, 100.0]
    assert values.tolist() == [0.0, 6.0]
    assert len(array.reads) == 1


def test_inventory_requires_unambiguous_resource_selection() -> None:
    inventory = OSClimateInventory.from_dict(
        {
            "resources": [
                {
                    "hazard_type": "Wind",
                    "indicator_id": "max_speed",
                    "path": f"wind/model-{index}/{{scenario}}/{{year}}",
                    "indicator_model_gcm": f"model-{index}",
                    "params": {},
                    "scenarios": [{"id": "historical", "years": [2010]}],
                    "units": "m/s",
                }
                for index in range(2)
            ]
        }
    )

    with pytest.raises(LookupError, match="2 OS-Climate resources"):
        inventory.select(hazard_type="Wind", indicator_id="max_speed")
    selected = inventory.select(
        hazard_type="Wind",
        indicator_id="max_speed",
        model_gcm="model-1",
    )
    assert selected.model_gcm == "model-1"
