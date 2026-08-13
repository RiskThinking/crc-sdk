from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
import rasterio  # type: ignore[import-untyped]
from rasterio.transform import from_bounds, from_origin  # type: ignore[import-untyped]
from rasterio.warp import transform_bounds  # type: ignore[import-untyped]

from crc_sdk.connectors import read_hazard_metadata
from crc_sdk.geometry import point_to_cell
from crc_sdk.providers.jrc import (
    JRC_DATASETS,
    JRCProvider,
    JRCRasterDataset,
    JRCRasterResource,
)
from crc_sdk.workflows import HazardDataset, JRCCanonicalizationPlan, JRCFloodPolicy
from crc_sdk.workflows.jrc import _crop_raster


def _dataset(directory: Path) -> JRCRasterDataset:
    directory.mkdir(parents=True, exist_ok=True)
    periods = (2, 5, 10, 100, 1000)
    values = (0.0, 0.2, 0.5, 1.0, 2.0)
    transform = from_origin(0.0, 0.01, 0.01, 0.01)
    for period, value in zip(periods, values):
        path = directory / f"Europe_RP{period}_depth.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
        ) as target:
            target.write(np.asarray([[[value]]], dtype=np.float32))
    return JRCRasterDataset(
        name="test-jrc-flood",
        base_url=str(directory),
        tile_index_url=None,
        filename_template="Europe_RP{return_period}_depth.tif",
        available_return_periods=periods,
        version="1.2.3",
        layout="continental",
    )


def _projected_dataset(directory: Path) -> JRCRasterDataset:
    directory.mkdir(parents=True, exist_ok=True)
    periods = (2, 5, 10, 100, 1000)
    native_bounds = transform_bounds("EPSG:4326", "EPSG:3857", 7.0, 49.0, 7.1, 49.1)
    transform = from_bounds(*native_bounds, width=2, height=2)
    for period, value in zip(periods, (0.0, 0.2, 0.5, 1.0, 2.0)):
        path = directory / f"Europe_RP{period}_depth.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:3857",
            transform=transform,
            nodata=-9999.0,
        ) as target:
            target.write(np.full((1, 2, 2), value, dtype=np.float32))
    return JRCRasterDataset(
        name="test-projected-jrc-flood",
        base_url=str(directory),
        tile_index_url=None,
        filename_template="Europe_RP{return_period}_depth.tif",
        available_return_periods=periods,
        version="1.2.3",
        layout="continental",
    )


def _tile_resource(directory: Path, source_id: str, west: float) -> JRCRasterResource:
    directory.mkdir(parents=True, exist_ok=True)
    urls = {}
    for period, value in zip((2, 5, 10, 100, 1000), (0.0, 0.2, 0.5, 1.0, 2.0)):
        path = directory / f"{source_id}_RP{period}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(west, 0.01, 0.01, 0.01),
            nodata=-9999.0,
        ) as target:
            target.write(np.asarray([[[value]]], dtype=np.float32))
        urls[period] = str(path)
    return JRCRasterResource(source_id=source_id, urls=urls)


def _plan(tmp_path: Path) -> JRCCanonicalizationPlan:
    return (
        HazardDataset.jrc("test", version="1.2.3")
        .for_area((0.0, 0.0, 0.01, 0.01))
        .cache(tmp_path / "cache", mode="reuse")
        .canonicalize(policy=JRCFloodPolicy.curated(h3_resolution=9))
    )


def test_builder_is_lazy_and_explain_does_not_resolve_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(JRC_DATASETS, "test", _dataset(tmp_path / "source"))
    plan = _plan(tmp_path)

    details = plan.explain(format="json")

    assert details["resolved_version"] is None
    assert details["source_periods"] == (2, 5, 10, 100, 1000)
    assert not (tmp_path / "cache" / "manifest.json").exists()


@pytest.mark.parametrize("periods", [[], [10]])
def test_source_periods_requires_four_fit_knots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, periods: list[int]
) -> None:
    monkeypatch.setitem(JRC_DATASETS, "test", _dataset(tmp_path / "source"))

    with pytest.raises(ValueError, match="at least four source return periods"):
        HazardDataset.jrc("test", version="1.2.3").for_area(
            (0.0, 0.0, 0.01, 0.01)
        ).source_periods(periods)


def test_crop_uses_shared_geotiff_remote_open_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _dataset(tmp_path / "source").raster_url(2)
    remote = "https://example.test/jrc-rp2.tif"
    opened = []
    original = rasterio.open

    def tracking_open(path: object, *args: object, **kwargs: object) -> object:
        opened.append(str(path))
        if str(path) == f"/vsicurl/{remote}":
            path = source
        return original(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", tracking_open)
    _crop_raster(remote, (0.0, 0.0, 0.01, 0.01), tmp_path / "crop.tif")

    assert opened[0] == f"/vsicurl/{remote}"


def test_prefetch_pins_manifest_and_offline_materializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(JRC_DATASETS, "test", _dataset(tmp_path / "source"))
    plan = _plan(tmp_path)

    prefetched = plan.prefetch()
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    offline = plan.cache(tmp_path / "cache", mode="offline")
    hazard = offline.materialize(tmp_path / "hazard.parquet")

    assert prefetched.source_version == "1.2.3"
    assert prefetched.cache_misses == 5
    assert manifest["requested_version"] == "1.2.3"
    assert manifest["resolved_version"] == "1.2.3"
    assert hazard.provider.source == tmp_path / "hazard.parquet"
    assert hazard.materialization is not None
    assert hazard.materialization.source_cache_hits == 5
    assert hazard.materialization.source_cache_misses == 0
    assert hazard.materialization.canonical_rows > 0
    assert hazard.metadata() == read_hazard_metadata(hazard.provider.source)
    assert hazard.provenance().version == "1.2.3"


def test_materialize_transforms_wgs84_aoi_to_projected_raster_crs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        JRC_DATASETS, "projected", _projected_dataset(tmp_path / "source")
    )
    hazard = (
        HazardDataset.jrc("projected", version="1.2.3")
        .for_area((7.0, 49.0, 7.1, 49.1))
        .cache(tmp_path / "cache", mode="reuse")
        .canonicalize(policy=JRCFloodPolicy.curated(h3_resolution=7))
        .materialize(tmp_path / "hazard.parquet")
    )

    assert hazard.materialization is not None
    assert hazard.materialization.canonical_rows > 0


def test_multitile_provenance_records_all_contributing_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    periods = (2, 5, 10, 100, 1000)
    dataset = JRCRasterDataset(
        name="test-tiled-jrc-flood",
        base_url=str(tmp_path / "source"),
        tile_index_url="unused",
        filename_template="{tile_id}_RP{return_period}.tif",
        available_return_periods=periods,
        version="1.2.3",
    )
    resources = (
        _tile_resource(tmp_path / "source", "tile-west", 0.0),
        _tile_resource(tmp_path / "source", "tile-east", 0.01),
    )
    monkeypatch.setitem(JRC_DATASETS, "tiled", dataset)
    monkeypatch.setattr(
        JRCProvider,
        "resources_for",
        lambda self, bounds, return_periods=None: resources,
    )

    hazard = (
        HazardDataset.jrc("tiled", version="1.2.3")
        .for_area((0.0, 0.0, 0.02, 0.01))
        .cache(None, mode="stream")
        .canonicalize(policy=JRCFloodPolicy.curated(h3_resolution=9))
        .materialize(tmp_path / "hazard.parquet")
    )

    uri = hazard.provenance().uri
    assert uri is not None
    query = parse_qs(urlparse(uri).query)
    assert query["sources"] == ["tile-west,tile-east"]
    assert query["source_periods"] == ["2,5,10,100,1000"]

    result = (
        hazard.for_assets(
            pa.table(
                {
                    "asset_id": ["seam"],
                    "longitude": [0.01],
                    "latitude": [0.005],
                }
            )
        )
        .return_periods([100])
        .write_parquet(tmp_path / "seam.parquet")
    )
    assert result.row_count == 1

    cell_result = (
        hazard.for_assets(
            pa.table(
                {
                    "asset_id": ["seam-cell"],
                    "cell_index": pa.array(
                        [point_to_cell(0.01, 0.005, 9)], type=pa.uint64()
                    ),
                }
            )
        )
        .return_periods([100])
        .write_parquet(tmp_path / "seam-cell.parquet")
    )
    assert cell_result.row_count == 1


def test_cache_skips_envelope_tile_without_intersecting_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    periods = (2, 5, 10, 100, 1000)
    dataset = JRCRasterDataset(
        name="test-tiled-jrc-flood",
        base_url=str(tmp_path / "source"),
        tile_index_url="unused",
        filename_template="{tile_id}_RP{return_period}.tif",
        available_return_periods=periods,
        version="1.2.3",
    )
    resources = (
        _tile_resource(tmp_path / "source", "intersecting", 0.0),
        _tile_resource(tmp_path / "source", "empty", 10.0),
    )
    monkeypatch.setitem(JRC_DATASETS, "tiled-empty", dataset)
    monkeypatch.setattr(
        JRCProvider,
        "resources_for",
        lambda self, bounds, return_periods=None: resources,
    )
    plan = (
        HazardDataset.jrc("tiled-empty", version="1.2.3")
        .for_area((0.0, 0.0, 0.01, 0.01))
        .cache(tmp_path / "cache", mode="reuse")
        .canonicalize(policy=JRCFloodPolicy.curated(h3_resolution=9))
    )

    prefetched = plan.prefetch()
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    hazard = plan.materialize(tmp_path / "hazard.parquet")

    assert prefetched.resources == 1
    assert [entry["source_id"] for entry in manifest["resources"]] == ["intersecting"]
    assert hazard.materialization is not None
    assert hazard.materialization.canonical_rows > 0


def test_one_chain_forwards_to_local_portfolio_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(JRC_DATASETS, "test", _dataset(tmp_path / "source"))
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "longitude": [0.005],
            "latitude": [0.005],
        }
    )

    result = (
        _plan(tmp_path)
        .for_assets(assets)
        .select(hazard_names=["RiverineInundation"])
        .return_periods([10, 100])
        .write_parquet(tmp_path / "portfolio.parquet")
    )

    assert result.output == tmp_path / "portfolio.parquet"
    assert result.row_count == 1
    assert result.value_columns == ("value_rp10", "value_rp100")


def test_offline_reports_all_invalid_cached_periods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(JRC_DATASETS, "test", _dataset(tmp_path / "source"))
    plan = _plan(tmp_path)
    plan.prefetch()
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    for entry in manifest["resources"]:
        for path in entry["local"].values():
            Path(path).write_bytes(b"corrupt")

    with pytest.raises(FileNotFoundError) as error:
        plan.cache(tmp_path / "cache", mode="offline").materialize(
            tmp_path / "hazard.parquet"
        )
    for period in (2, 5, 10, 100, 1000):
        assert f"continental:RP{period}" in str(error.value)
