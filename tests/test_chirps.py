from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from crc_sdk.workflows import CHIRPSMonthlyPlan

rasterio = pytest.importorskip("rasterio")


class Response(io.BytesIO):
    headers: dict[str, str] = {}


def raster_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "fixture.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(36.0, 1.0, 0.05, 0.05),
    ) as output:
        output.write(np.array([[0.0, -9999.0], [2.0, 3.0]], dtype="float32"), 1)
    return path.read_bytes()


def test_snapshot_and_source_nodata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = raster_bytes(tmp_path)
    monkeypatch.setattr(
        "crc_sdk.workflows.chirps.urlopen", lambda *a, **kw: Response(data)
    )
    base = CHIRPSMonthlyPlan.monthly(2020, 1)
    request = base.for_area((36.0, 0.9, 36.1, 1.0)).cache(tmp_path)
    assert base.bounds is None
    provenance = request.prefetch()
    replay = request.cache(tmp_path, mode="offline", sha256=provenance["sha256"])
    rows = [row for batch in replay.batches(batch_size=2) for row in batch.to_pylist()]
    assert sorted(row["precipitation_mm"] for row in rows) == [0, 2, 3]
    assert all(row["month"].isoformat() == "2020-01-01" for row in rows)
    assert provenance["licence"] == "CC-BY-4.0"
    (replay._directory() / "source.cog").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="checksum"):
        list(replay.batches())


def test_snapshot_mismatch_and_invalid_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = raster_bytes(tmp_path)
    monkeypatch.setattr(
        "crc_sdk.workflows.chirps.urlopen", lambda *a, **kw: Response(data)
    )
    plan = CHIRPSMonthlyPlan.monthly(2020, 1).cache(tmp_path, sha256="0" * 64)
    with pytest.raises(RuntimeError, match="expected SHA"):
        plan.prefetch()
    assert not (plan._directory() / "source.cog").exists()
    with pytest.raises(ValueError):
        CHIRPSMonthlyPlan.monthly(1980, 1)
    with pytest.raises(ValueError):
        CHIRPSMonthlyPlan.monthly(2020, 13)
    with pytest.raises(ValueError):
        CHIRPSMonthlyPlan.monthly(2020, 1).for_area((0, 59, 1, 61))
    with pytest.raises(ValueError, match="for_area"):
        list(CHIRPSMonthlyPlan.monthly(2020, 1).batches())


def test_failed_refresh_keeps_existing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = raster_bytes(tmp_path)
    monkeypatch.setattr(
        "crc_sdk.workflows.chirps.urlopen", lambda *a, **kw: Response(payload)
    )
    request = CHIRPSMonthlyPlan.monthly(2020, 1).cache(tmp_path)
    original = request.prefetch()
    payload = b"changed source"
    with pytest.raises(RuntimeError, match="expected SHA"):
        request.cache(tmp_path, mode="refresh", sha256=original["sha256"]).prefetch()
    assert (
        request.cache(tmp_path, mode="offline").prefetch()["sha256"]
        == original["sha256"]
    )
