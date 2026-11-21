from __future__ import annotations

import pytest

from crc_sdk._providers.jrc import EFAS, GLOFAS, JRCRasterDataset


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
