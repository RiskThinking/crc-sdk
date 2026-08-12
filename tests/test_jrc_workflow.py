from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
import rasterio  # type: ignore[import-untyped]
from rasterio.transform import from_origin  # type: ignore[import-untyped]

from crc_sdk.connectors import read_hazard_metadata
from crc_sdk.providers.jrc import JRC_DATASETS, JRCRasterDataset
from crc_sdk.workflows import HazardDataset, JRCCanonicalizationPlan, JRCFloodPolicy


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
