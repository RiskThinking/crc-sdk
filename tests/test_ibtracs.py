from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import pytest

from crc_sdk.workflows import IBTrACSPlan

CSV = (
    b"SID,SEASON,ISO_TIME,LAT,USA_WIND\n,, ,degrees_north,kts\n"
    b"A,2020,2020-01-01 00:00:00,10, \nA,2020,2020-01-01 03:00:00,11,40\n"
    b"B,2021,2021-01-01 00:00:00,12,50\n"
)


class Response(io.BytesIO):
    headers = {"ETag": '"sample"'}


def test_snapshot_and_filtered_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "crc_sdk.workflows.ibtracs.urlopen", lambda *a, **kw: Response(CSV)
    )
    plan = IBTrACSPlan("NA", tmp_path)
    selected = plan.for_storms(["A"]).years(2020, 2020)
    assert plan.storm_ids is None
    provenance = selected.prefetch()
    assert cast(dict[str, str], provenance["units"])["USA_WIND"] == "kts"
    batches = list(selected.batches(batch_size=1))
    assert len(batches) == 2
    assert batches[0].to_pylist()[0]["USA_WIND"] is None
    offline = IBTrACSPlan(
        "NA", tmp_path, mode="offline", expected_sha256=str(provenance["sha256"])
    )
    assert sum(batch.num_rows for batch in offline.batches()) == 3
    (tmp_path / "ibtracs/v04r01/NA/source.csv").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        offline.prefetch()


def test_missing_and_mismatched_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError):
        IBTrACSPlan("NA", tmp_path, mode="offline").prefetch()
    monkeypatch.setattr(
        "crc_sdk.workflows.ibtracs.urlopen", lambda *a, **kw: Response(CSV)
    )
    with pytest.raises(RuntimeError, match="expected SHA"):
        IBTrACSPlan("NA", tmp_path, expected_sha256="0" * 64).prefetch()
    assert not (tmp_path / "ibtracs/v04r01/NA/source.csv").exists()


def test_invalid_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        IBTrACSPlan("unknown", tmp_path)
    with pytest.raises(ValueError):
        IBTrACSPlan("NA", tmp_path).years(2021, 2020)
