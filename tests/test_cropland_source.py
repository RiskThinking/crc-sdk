"""USDA's real categorical raster protocol with an offline native transport."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from crc_sdk import _source_io
from crc_sdk._cropland_source import USDACropland
from crc_sdk._declarations import Period
from crc_sdk._evidence import EvidenceKind, validate_evidence
from crc_sdk._evidence_recipes import Layer
from crc_sdk._inspection import Bounds, Capability
from crc_sdk._runtime import Cache, Runtime


class _Transport:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.value = 1
        self.bad_legend = False

    def __call__(self, uri: str, path: Path) -> int:
        self.calls.append(uri)
        parsed = urlsplit(uri)
        query = parse_qs(parsed.query)
        data: object
        if parsed.path.endswith("/ImageServer"):
            data = {
                "serviceDataType": "esriImageServiceDataTypeThematic",
                "bandCount": 1,
                "pixelType": "U8",
                "spatialReference": {"wkid": 102100, "latestWkid": 3857},
                "maxImageWidth": 1024,
                "maxImageHeight": 1024,
                "extent": {
                    "xmin": -12000000.0,
                    "ymin": 4000000.0,
                    "xmax": -10000000.0,
                    "ymax": 6000000.0,
                },
            }
        elif parsed.path.endswith("/query"):
            assert query["where"] == ["Category=1"]
            data = {
                "features": [
                    {"attributes": {"OBJECTID": 1, "Year": 2023, "LowPS": 30}},
                    {"attributes": {"OBJECTID": 2, "Year": 2024, "LowPS": 10}},
                ]
            }
        elif parsed.path.endswith("/rasterAttributeTable"):
            data = {
                "features": [
                    {"attributes": {"Value": 1, "Class_Names": "Corn"}},
                    {"attributes": {"Value": 5, "Class_Names": "Soybeans"}},
                ]
            }
            if self.bad_legend:
                data = {
                    "features": [{"attributes": {"Value": 1.5, "Class_Names": "Corn"}}]
                }
        elif parsed.path.endswith("/exportImage"):
            rasterio = pytest.importorskip("rasterio")
            from rasterio.io import MemoryFile  # type: ignore[import-untyped]

            assert query["interpolation"] == ["RSP_NearestNeighbor"]
            assert json.loads(query["renderingRule"][0]) == {"rasterFunction": "None"}
            assert json.loads(query["mosaicRule"][0])["lockRasterIds"] in ([1], [2])
            width, height = (int(v) for v in query["size"][0].split(","))
            west, south, east, north = (float(v) for v in query["bbox"][0].split(","))
            with MemoryFile() as memory:
                with memory.open(
                    driver="GTiff",
                    width=width,
                    height=height,
                    count=1,
                    dtype="uint8",
                    crs="EPSG:3857",
                    nodata=0,
                    transform=rasterio.transform.from_bounds(
                        west, south, east, north, width, height
                    ),
                ) as raster:
                    values = np.full((height, width), self.value, dtype="uint8")
                    values[0, 0] = 0
                    raster.write(values, 1)
                path.write_bytes(memory.read())
            return 200
        else:
            raise AssertionError(f"Unexpected USDA resource: {uri}")
        path.write_text(json.dumps(data))
        return 200


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _Transport:
    value = _Transport()
    monkeypatch.setattr(_source_io, "_download", value)
    return value


_AREA = Bounds(west=-95.0, south=39.0, east=-94.9995, north=39.0005)


def test_builder_is_pure_and_inventory_discovers_native_publications(
    transport: _Transport,
) -> None:
    source = USDACropland()
    assert not transport.calls
    explanation = source.explain()
    assert explanation.requirements[0].parameter == "area"
    assert not transport.calls
    inspection = source.inspect()
    assert {s.period.start for s in inspection.series if s.period is not None} == {
        2023,
        2024,
    }
    assert inspection.requirements[0].parameter == "area"
    assert Capability.FIT not in inspection.capabilities
    assert not hasattr(source, "fit")
    assert len(transport.calls) == 2


def test_native_classes_labels_footprints_and_missing_counts(
    transport: _Transport,
) -> None:
    source = USDACropland().select(area=_AREA)
    result = source.read()
    assert result.metadata.kind is EvidenceKind.LAYER
    assert {
        s.period.start for s in result.metadata.spec.series if s.period is not None
    } == {2024}
    rows = result.table.to_pylist()
    assert rows and any(row["value"] == 1 and row["label"] == "Corn" for row in rows)
    assert all(row["geometry"] for row in rows)
    assert all(row["label"] is None for row in rows if row["value"] is None)
    assert result.report.counts.no_data == 1
    assert validate_evidence(result.table, result.metadata).valid
    assert result.metadata.source is not None
    assert all(
        r.sha256 and r.license == "CC0-1.0" and r.retrieved_at
        for r in result.metadata.source.resources
    )
    assert len(transport.calls) == 4


def test_inspected_publication_can_select_one_cropland_year(
    transport: _Transport,
) -> None:
    inventory = USDACropland().inspect()
    reference, _ = inventory.series
    assert reference.release is not None and inventory.releases == (reference.release,)
    assert reference.period == Period.window(2023, 2023)
    configured = USDACropland().select(area=_AREA, series=reference)
    assert configured.inspect().series == (reference,)
    assert all("/exportImage" not in url for url in transport.calls)
    result = configured.read()
    assert result.table.num_rows > 0
    assert result.metadata.spec.series[0].period == reference.period
    assert result.metadata.source is not None
    assert result.metadata.source.release != reference.release


def test_explicit_crop_year_and_canonical_layer_persistence(
    tmp_path: Path, transport: _Transport
) -> None:
    source = USDACropland().select(area=_AREA, periods=Period.window(2023, 2023))
    saved = source.write(tmp_path / "cropland.parquet")
    assert type(saved) is Layer
    result = saved.read()
    assert {
        s.period.start for s in result.metadata.spec.series if s.period is not None
    } == {2023}
    assert all("confidence" not in row for row in result.table.to_pylist())
    assert len(transport.calls) == 4


def test_shared_verified_cache_offline_release_pin_and_refresh(
    tmp_path: Path, transport: _Transport
) -> None:
    source = USDACropland().select(area=_AREA)
    cache = tmp_path / "cache"
    first = source.read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert first.metadata.source is not None
    release = first.metadata.source.release
    assert release is not None
    calls = len(transport.calls)
    pinned = USDACropland(release=release).select(area=_AREA)
    offline = pinned.read(runtime=Runtime(cache=Cache.offline(cache)))
    assert offline.table.equals(first.table)
    assert len(transport.calls) == calls
    transport.value = 5
    with pytest.raises(OSError, match="source release changed"):
        pinned.read(runtime=Runtime(cache=Cache.refresh(cache)))
    assert len(transport.calls) == calls * 2


def test_empty_selection_has_stable_layer_schema_without_network(
    tmp_path: Path, transport: _Transport
) -> None:
    empty = (
        USDACropland()
        .select(variables=[])
        .read(runtime=Runtime(cache=Cache.offline(tmp_path / "empty-cache")))
    )
    assert empty.table.num_rows == 0
    assert empty.metadata.kind is EvidenceKind.LAYER
    assert "time" not in empty.table.schema.names
    assert not transport.calls


def test_missing_area_and_bad_native_legend_fail_explicitly(
    transport: _Transport,
) -> None:
    with pytest.raises(Exception, match="area"):
        USDACropland().read()
    transport.bad_legend = True
    report = USDACropland().select(area=_AREA).validate()
    assert report.valid is False
    assert any("integer" in item.message for item in report.diagnostics)
