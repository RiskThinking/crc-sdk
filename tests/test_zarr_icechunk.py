from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crc_sdk.connectors.duckdb.zarr import RasterMetadata, ZarrRaster

icechunk = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")

METADATA = RasterMetadata(
    hazard_type="Wind",
    indicator_id="max_speed",
    scenario="historical",
    year=2010,
    units="m/s",
    path="test/wind",
)


def _repository(tmp_path: Path) -> tuple[Any, str]:
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo"))
    )
    session = repo.writable_session("main")
    root = zarr.open_group(store=session.store, mode="w")
    group = root.require_group("hazard")
    array = group.create_array("wind", shape=(2, 2, 3), chunks=(2, 2, 2), dtype="f4")
    array[:] = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    array.attrs.update(
        {
            "index_name": "return period (years)",
            "index_values": [10, 100],
            "transform_mat3x3": [1, 0, 0, 0, -1, 2, 0, 0, 1],
        }
    )
    first = session.commit("first")
    # A later commit changes the data, to prove snapshot pinning.
    session = repo.writable_session("main")
    zarr.open_group(store=session.store, mode="r+")["hazard/wind"][0, 0, 0] = 99.0
    session.commit("second")
    return repo, first


def test_open_icechunk_from_repository_reads_curves(tmp_path: Path) -> None:
    repo, _ = _repository(tmp_path)
    raster = ZarrRaster.open_icechunk(
        repo, array_path="hazard/wind", metadata=METADATA, work_dir=tmp_path
    )
    assert raster.shape == (2, 2, 3)
    periods, values = raster.point_values(0.5, 1.5)
    assert periods.tolist() == [10.0, 100.0]
    assert values.tolist() == [99.0, 6.0]
    assert len(list(raster.iter_curves())) == 6


def test_open_icechunk_pins_snapshot_and_accepts_session(tmp_path: Path) -> None:
    repo, first = _repository(tmp_path)
    pinned = ZarrRaster.open_icechunk(
        repo,
        array_path="/hazard/wind",
        metadata=METADATA,
        snapshot_id=first,
        work_dir=tmp_path,
    )
    assert pinned.point_values(0.5, 1.5)[1].tolist() == [0.0, 6.0]
    session = repo.readonly_session("main")
    from_session = ZarrRaster.open_icechunk(
        session, array_path="hazard/wind", metadata=METADATA, work_dir=tmp_path
    )
    assert from_session.point_values(0.5, 1.5)[1].tolist() == [99.0, 6.0]


def test_open_icechunk_rejects_unrelated_objects() -> None:
    with pytest.raises(TypeError, match="Repository"):
        ZarrRaster.open_icechunk(object(), array_path="a", metadata=METADATA)


def test_open_icechunk_names_extra_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(ImportError, match=r"crc-sdk\[icechunk\]"):
        ZarrRaster.open_icechunk(object(), array_path="a", metadata=METADATA)
