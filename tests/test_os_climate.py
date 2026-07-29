from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crc_sdk.connectors import (
    HurdleFitPolicy,
    OSClimateIngestPolicy,
    canonicalize_os_climate,
    write_hazard_stream,
)
from crc_sdk.connectors.duckdb import DuckDBConnection, RasterMetadata, ZarrRaster
from crc_sdk.providers import LocalProvider, OSClimateInventory
from crc_sdk.providers.os_climate import OSClimateProvider
from crc_sdk.types import HazardQuery


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


class FakeReturnPeriodArray:
    def __init__(self) -> None:
        self.data = np.asarray(
            [0.0, 0.2, 0.5, 1.0, 2.0], dtype=np.float64
        ).reshape(
            5, 1, 1
        )
        self.shape = self.data.shape
        self.chunks = self.data.shape
        self.attrs = {
            "index_name": "return period (years)",
            "index_values": [2, 5, 10, 100, 1000],
            "transform_mat3x3": [1, 0, 0, 0, -1, 1, 0, 0, 1],
        }
        self.reads: list[Any] = []

    def __getitem__(self, key: Any) -> Any:
        self.reads.append(key)
        return self.data[key]


class FakeMixedReturnPeriodArray:
    """Two pixels: one constant/unfittable, one with a real return-level curve."""

    def __init__(self) -> None:
        flat = [0.0, 0.0, 0.0, 0.0, 0.0]
        varying = [0.0, 0.2, 0.5, 1.0, 2.0]
        self.data = np.asarray(
            [[flat[i], varying[i]] for i in range(5)], dtype=np.float64
        ).reshape(5, 1, 2)
        self.shape = self.data.shape
        self.chunks = self.data.shape
        self.attrs = {
            "index_name": "return period (years)",
            "index_values": [2, 5, 10, 100, 1000],
            "transform_mat3x3": [1, 0, 0, 0, -1, 1, 0, 0, 1],
        }
        self.reads: list[Any] = []

    def __getitem__(self, key: Any) -> Any:
        self.reads.append(key)
        return self.data[key]


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


def test_os_climate_canonical_stream_persists_and_queries(
    tmp_path: Path,
) -> None:
    array = FakeReturnPeriodArray()
    raster = ZarrRaster(
        array,
        RasterMetadata(
            hazard_type="Wind",
            indicator_id="max_speed",
            scenario="ssp585",
            year=2050,
            units="m/s",
            path="test/wind/ssp585/2050",
        ),
    )
    policy = OSClimateIngestPolicy(
        h3_resolution=3,
        family="gumbel_r",
        producer="tests",
        creation_version="1",
        batch_rows=2,
        hurdle=HurdleFitPolicy(
            atom_probability=0.5,
            atom_location=0.0,
        ),
        maximum_normalized_rmse=0.2,
    )
    stream = canonicalize_os_climate(raster, policy)
    destination = tmp_path / "wind-ssp585-2050.parquet"
    write_hazard_stream(stream, destination)

    table = LocalProvider(destination).read(
        HazardQuery(hazard_name="Wind", horizon=2050, pathway="ssp585")
    )
    assert table.num_rows > 0
    assert set(table["curve_type"].to_pylist()) == {"gumbel_r"}
    assert set(table["curve_kind"].to_pylist()) == {"hurdle"}
    assert set(table["curve_atom_probability"].to_pylist()) == {0.5}
    assert len(set(table["source_id"].to_pylist())) == 1
    assert len(array.reads) == 1


def test_os_climate_plain_quantile_fit_remains_available() -> None:
    raster = ZarrRaster(
        FakeReturnPeriodArray(),
        RasterMetadata(
            hazard_type="Wind",
            indicator_id="max_speed",
            scenario="ssp585",
            year=2050,
            units="m/s",
            path="test/wind/ssp585/2050",
        ),
    )
    stream = canonicalize_os_climate(
        raster,
        OSClimateIngestPolicy(
            h3_resolution=3,
            family="gumbel_r",
            producer="tests",
            creation_version="1",
        ),
    )
    table = stream.read_all()

    assert set(table["curve_kind"].to_pylist()) == {"fitted"}


def test_on_fit_failure_raise_aborts_on_constant_pixel() -> None:
    raster = ZarrRaster(
        FakeMixedReturnPeriodArray(),
        RasterMetadata(
            hazard_type="RiverineInundation",
            indicator_id="flood_depth",
            scenario="historical",
            year=1980,
            units="metres",
            path="test/flood/mixed",
        ),
    )
    policy = OSClimateIngestPolicy(
        h3_resolution=3,
        family="gumbel_r",
        producer="tests",
        creation_version="1",
    )
    with pytest.raises(ValueError, match="failed to fit source pixel"):
        canonicalize_os_climate(raster, policy).read_all()


def test_on_fit_failure_skip_drops_unfittable_pixels() -> None:
    raster = ZarrRaster(
        FakeMixedReturnPeriodArray(),
        RasterMetadata(
            hazard_type="RiverineInundation",
            indicator_id="flood_depth",
            scenario="historical",
            year=1980,
            units="metres",
            path="test/flood/mixed",
        ),
    )
    policy = OSClimateIngestPolicy(
        h3_resolution=3,
        family="gumbel_r",
        producer="tests",
        creation_version="1",
        on_fit_failure="skip",
    )
    table = canonicalize_os_climate(raster, policy).read_all()

    assert table.num_rows > 0
    assert len(set(table["source_id"].to_pylist())) == 1


def test_on_fit_failure_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="on_fit_failure"):
        OSClimateIngestPolicy(
            h3_resolution=3,
            family="gumbel_r",
            producer="tests",
            creation_version="1",
            on_fit_failure="ignore",  # type: ignore[arg-type]
        )


def test_os_climate_provider_default_connection_is_resource_tuned(
    tmp_path: Path,
) -> None:
    provider = OSClimateProvider(work_dir=tmp_path)
    assert provider.connection.config.get("threads")
    assert provider.connection.config.get("memory_limit")
    assert (tmp_path / "duckdb-temp").is_dir()


def test_os_climate_provider_explicit_connection_is_not_overridden() -> None:
    explicit = DuckDBConnection()
    provider = OSClimateProvider(connection=explicit)
    assert provider.connection is explicit


def test_zarr_raster_default_connection_is_resource_tuned(tmp_path: Path) -> None:
    raster = ZarrRaster(
        FakeReturnPeriodArray(),
        RasterMetadata(
            hazard_type="Wind",
            indicator_id="max_speed",
            scenario="historical",
            year=2010,
            units="m/s",
            path="test/wind",
        ),
        work_dir=tmp_path,
    )
    assert raster.connection.config.get("threads")
    assert (tmp_path / "duckdb-temp").is_dir()


def test_zarr_raster_explicit_connection_is_not_overridden() -> None:
    explicit = DuckDBConnection()
    raster = ZarrRaster(
        FakeReturnPeriodArray(),
        RasterMetadata(
            hazard_type="Wind",
            indicator_id="max_speed",
            scenario="historical",
            year=2010,
            units="m/s",
            path="test/wind",
        ),
        connection=explicit,
    )
    assert raster.connection is explicit
