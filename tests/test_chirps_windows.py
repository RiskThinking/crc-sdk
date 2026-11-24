"""Monthly CHIRPS acquisition owns bounded views of actual source pixels."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import crc_framework as cf
import numpy as np
import pytest
from shapely import from_wkb  # type: ignore[import-untyped]

from crc_sdk import (
    CHIRPS,
    AnnualMaximum,
    Bounds,
    Cache,
    Period,
    Runtime,
    _climate_io,
    _layer_sources,
)
from crc_sdk._graph import ExecutionScope
from crc_sdk._inspection import Capability


def _write_month(path: Path, month: int, *, changed: bool = False) -> None:
    rasterio = pytest.importorskip("rasterio")
    data = np.arange(256, dtype=np.float32).reshape(16, 16) + month
    if changed:
        data[7, 7] += 100
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(-80, 45, 0.1, 0.1),
        nodata=-9999,
    ) as output:
        output.write(data, 1)


@pytest.fixture
def publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "publication"
    path.mkdir()
    for month in range(1, 13):
        _write_month(path / f"chirps-v3.0.2001.{month:02d}.cog", month)
    monkeypatch.setattr(_layer_sources, "_CHIRPS", path.as_uri() + "/")
    return path


_AREA = Bounds(west=-79.251, south=44.249, east=-79.249, north=44.251)


def _recipe() -> CHIRPS:
    return CHIRPS().select(area=_AREA, time=Period.window(2001, 2001))


def test_monthly_chirps_reads_only_selected_native_windows_and_replays_offline(
    publication: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rasterio = pytest.importorskip("rasterio")
    original = rasterio.open
    reads: list[tuple[int, int, int, int]] = []

    class Reader:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __getattr__(self, name: str) -> Any:
            return getattr(self.value, name)

        def __enter__(self) -> Reader:
            self.value.__enter__()
            return self

        def __exit__(self, *args: Any) -> None:
            self.value.__exit__(*args)

        def read(self, *args: Any, **kwargs: Any) -> Any:
            window = kwargs["window"]
            reads.append(
                (
                    int(window.row_off),
                    int(window.col_off),
                    int(window.height),
                    int(window.width),
                )
            )
            return self.value.read(*args, **kwargs)

    def open_source(path: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(path, *args, **kwargs)
        source = str(path).startswith((publication.as_uri(), str(publication)))
        return Reader(result) if source else result

    monkeypatch.setattr(rasterio, "open", open_source)
    inventory = _recipe().inspect()
    assert not reads
    assert Capability.AGGREGATE in inventory.capabilities
    cache = tmp_path / "cache"
    result = _recipe().read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert reads == [(7, 7, 1, 1)] * 12
    assert result.table["value"].to_pylist() == list(range(120, 132))
    assert result.table["time"].to_pylist()[0] == "2001-01-01"
    assert result.metadata.sampling is not None
    assert result.metadata.sampling.step == cf.Duration(1, unit=cf.TimeUnit.MONTH)
    assert result.metadata.source is not None
    assert len(result.metadata.source.resources) == 12
    assert all(
        "#crc-window=" in r.uri and r.license == "CC-BY-4.0" and r.retrieved_at
        for r in result.metadata.source.resources
    )
    assert from_wkb(result.table["geometry"][0].as_py()).bounds == pytest.approx(
        (-79.3, 44.2, -79.2, 44.3)
    )
    monkeypatch.setattr(
        _climate_io,
        "_validator",
        lambda *a: pytest.fail("offline acquired source headers"),
    )
    replay = _recipe().read(runtime=Runtime(cache=Cache.offline(cache)))
    assert reads == [(7, 7, 1, 1)] * 12
    assert replay.table.equals(result.table, check_metadata=False)
    annual = (
        _recipe()
        .aggregate(AnnualMaximum())
        .read(runtime=Runtime(cache=Cache.offline(cache)))
    )
    assert annual.table["value"].to_pylist() == [131.0]


def test_chirps_refresh_verifies_new_window_content_without_rewriting_old_snapshot(
    publication: Path,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    first = _recipe().read(runtime=Runtime(cache=Cache.reuse(cache)))
    assert first.metadata.source is not None
    _write_month(publication / "chirps-v3.0.2001.01.cog", 1, changed=True)
    second = _recipe().read(runtime=Runtime(cache=Cache.refresh(cache)))
    assert second.metadata.source is not None
    assert first.table["value"].to_pylist()[0] == 120
    assert second.table["value"].to_pylist()[0] == 220
    assert (
        first.metadata.source.resources[0].sha256
        != second.metadata.source.resources[0].sha256
    )
    assert first.metadata.source.release != second.metadata.source.release
    assert (
        _recipe()
        .read(runtime=Runtime(cache=Cache.offline(cache)))
        .table.equals(second.table, check_metadata=False)
    )
    with pytest.raises(OSError, match="release changed"):
        CHIRPS(release=first.metadata.source.release).select(
            area=_AREA, time=Period.window(2001, 2001)
        ).read(runtime=Runtime(cache=Cache.offline(cache)))


def test_window_keeps_original_offsets_and_overlapping_pixel_identity(
    publication: Path,
) -> None:
    uri = (publication / "chirps-v3.0.2001.01.cog").as_uri()
    wider = Bounds(west=-79.65, south=44.15, east=-79.24, north=44.65)
    with ExecutionScope() as scope:
        directory = scope.directory
        bundle, _ = _climate_io.raster_window_snapshot(scope, uri, area=_AREA)
        with zipfile.ZipFile(bundle) as archive:
            assert json.loads(archive.read("metadata.json"))["source_uri"] == uri
        rasterio = pytest.importorskip("rasterio")
        path = _climate_io.raster_views(scope, bundle)[0]
        with rasterio.open(path) as view:
            assert view.shape == (1, 1)
            assert view.tags()["crc_source_row"] == "7"
            assert view.tags()["crc_source_column"] == "7"
            assert json.loads(view.tags()["crc_source_transform"]) == [
                0.1,
                0.0,
                -80.0,
                0.0,
                -0.1,
                45.0,
            ]
    assert not directory.exists()
    narrow = _recipe().read()
    broad = CHIRPS().select(area=wider, time=Period.window(2001, 2001)).read()
    narrow_record = narrow.table.to_pylist()[0]
    matches = [
        r
        for r in broad.table.to_pylist()
        if r["support_id"] == narrow_record["support_id"]
        and r["time"] == narrow_record["time"]
    ]
    assert len(matches) == 1
    assert matches[0]["geometry"] == narrow_record["geometry"]


def test_remote_raster_reads_send_the_etag_with_its_quotes() -> None:
    from crc_sdk._climate_io import _gdal_options

    with _gdal_options(None) as options:
        assert "GDAL_HTTP_HEADER_FILE" not in options
    with _gdal_options('"67be0fed-1c41a45"') as options:
        headers = Path(options["GDAL_HTTP_HEADER_FILE"])
        # GDAL_HTTP_HEADERS would strip these quotes and servers answer 412.
        assert headers.read_text() == 'If-Match: "67be0fed-1c41a45"\n'
        assert "GDAL_HTTP_HEADERS" not in options
    assert not headers.exists()
