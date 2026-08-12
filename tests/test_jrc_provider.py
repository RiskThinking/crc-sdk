from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from crc_sdk.providers.jrc import EFAS, GLOFAS, JRCProvider, JRCRasterDataset

_FAKE_TILE_INDEX = {
    "features": [
        {
            "type": "Feature",
            "properties": {"id": "54", "name": "N50_W80"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-80.0, 40.0],
                        [-70.0, 40.0],
                        [-70.0, 50.0],
                        [-80.0, 50.0],
                        [-80.0, 40.0],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"id": "99", "name": "N0_E0"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
                ],
            },
        },
    ]
}


def test_jrc_raster_dataset_rejects_unavailable_return_period() -> None:
    with pytest.raises(ValueError, match="10000 is not available"):
        GLOFAS.tile_url("ID54_N50_W80", 10_000)


def test_glofas_and_efas_encode_their_current_release_layouts() -> None:
    assert isinstance(GLOFAS, JRCRasterDataset)
    assert isinstance(EFAS, JRCRasterDataset)
    assert GLOFAS.name != EFAS.name
    assert GLOFAS.base_url != EFAS.base_url
    assert GLOFAS.layout == "tiled"
    assert GLOFAS.version == "2.1.2"
    assert EFAS.layout == "continental"
    assert EFAS.version == "3.1.1"
    assert EFAS.available_return_periods == (10, 20, 30, 40, 50, 75, 100, 200, 500)
    assert "CEMS-GLOFAS" in GLOFAS.tile_url("ID54_N50_W80", 100)
    assert EFAS.raster_url(100).endswith("Europe_RP100_filled_depth.tif")
    with pytest.raises(ValueError, match="continental raster layout"):
        EFAS.tile_url("unused", 100)


def test_dataset_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        JRCRasterDataset(
            name="",
            base_url="https://example.test",
            tile_index_url="https://example.test/tiles.geojson",
            filename_template="{tile_id}_RP{return_period}.tif",
            available_return_periods=(100,),
        )
    with pytest.raises(ValueError, match="non-empty"):
        JRCRasterDataset(
            name="custom",
            base_url="https://example.test",
            tile_index_url="https://example.test/tiles.geojson",
            filename_template="{tile_id}_RP{return_period}.tif",
            available_return_periods=(),
        )


def test_tiles_for_resolves_intersecting_tiles_only(tmp_path: Path) -> None:
    provider = JRCProvider(GLOFAS, work_dir=tmp_path)
    with (
        patch("crc_sdk.providers.jrc.urlopen") as mock_urlopen,
        patch("json.load", return_value=_FAKE_TILE_INDEX),
    ):
        mock_urlopen.return_value.__enter__.return_value = object()
        tiles = provider.tiles_for((-79.65, 43.58, -79.15, 43.85))

    assert tiles == ("ID54_N50_W80",)
    mock_urlopen.assert_called_once_with(GLOFAS.tile_index_url)


def test_tile_index_is_fetched_once_and_cached(tmp_path: Path) -> None:
    provider = JRCProvider(GLOFAS, work_dir=tmp_path)
    with (
        patch("crc_sdk.providers.jrc.urlopen") as mock_urlopen,
        patch("json.load", return_value=_FAKE_TILE_INDEX),
    ):
        mock_urlopen.return_value.__enter__.return_value = object()
        provider.tiles_for((-79.65, 43.58, -79.15, 43.85))
        provider.tiles_for((0.0, 0.0, 5.0, 5.0))

    assert mock_urlopen.call_count == 1


def test_continental_aoi_is_one_resource_without_tile_index(tmp_path: Path) -> None:
    provider = JRCProvider(EFAS, work_dir=tmp_path)

    resources = provider.resources_for((7.0, 49.0, 8.0, 50.0), return_periods=[30, 100])

    assert len(resources) == 1
    assert resources[0].source_id == "continental"
    assert resources[0].urls == {
        30: f"{EFAS.base_url}/Europe_RP30_filled_depth.tif",
        100: f"{EFAS.base_url}/Europe_RP100_filled_depth.tif",
    }


def test_latest_version_is_read_from_release_readme(tmp_path: Path) -> None:
    provider = JRCProvider(EFAS, work_dir=tmp_path)
    response = patch("crc_sdk.providers.jrc.urlopen")
    with response as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            b"G01. Dataset version 3.1.1\n"
        )
        assert provider.resolve_version("latest") == "3.1.1"

    mock_urlopen.assert_called_once_with(EFAS.readme_url)


def test_provider_default_connection_is_resource_tuned(tmp_path: Path) -> None:
    provider = JRCProvider(GLOFAS, work_dir=tmp_path)
    assert provider.connection.config.get("threads")
    assert (tmp_path / "duckdb-temp").is_dir()
