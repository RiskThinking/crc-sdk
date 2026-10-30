from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from crc_sdk.workflows import ECCCDailyPlan


def feature(day: int, *, climate_id: str = "6158731") -> dict[str, object]:
    return {
        "type": "Feature",
        "id": f"{climate_id}.2020.1.{day}",
        "geometry": {"type": "Point", "coordinates": [-79.63, 43.68]},
        "properties": {
            "CLIMATE_IDENTIFIER": climate_id,
            "LOCAL_DATE": f"2020-01-{day:02d} 00:00:00",
            "LOCAL_YEAR": 2020,
            "LOCAL_MONTH": 1,
            "LOCAL_DAY": day,
            "STATION_NAME": "TEST",
            "STN_ID": 51459,
            "MAX_TEMPERATURE": 1.5,
            "TOTAL_PRECIPITATION": 0,
            "TOTAL_PRECIPITATION_FLAG": "T",
        },
    }


def plan(tmp_path: Path) -> ECCCDailyPlan:
    return ECCCDailyPlan.for_station("6158731").years(2020, 2020).cache(tmp_path)


def test_pagination_trace_flags_and_offline_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [
        {
            "type": "FeatureCollection",
            "features": [feature(1)],
            "links": [{"rel": "next", "href": "?offset=1"}],
        },
        {"type": "FeatureCollection", "features": [feature(2)], "links": []},
    ]
    calls: list[str] = []

    def fetch(url: str, **kwargs: object) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(json.dumps(pages[len(calls) - 1]).encode())

    monkeypatch.setattr("crc_sdk.workflows.eccc_daily.urlopen", fetch)
    base = ECCCDailyPlan.for_station("6158731")
    request = base.years(2020, 2020).cache(tmp_path)
    assert base.selected_years == ()
    (manifest,) = request.prefetch()
    assert len(manifest["page_sha256"]) == 2
    assert manifest["observations"] == 2
    replay = request.cache(tmp_path, mode="offline", sha256={2020: manifest["sha256"]})
    batches = list(replay.batches(batch_size=1))
    assert len(calls) == len(batches) == 2
    row = batches[0].to_pylist()[0]
    assert row["TOTAL_PRECIPITATION"] == 0.0
    assert row["TOTAL_PRECIPITATION_FLAG"] == "T"
    assert row["STN_ID"] == 51459
    assert row["MIN_TEMPERATURE"] is None
    (replay._directory(2020) / "observations.jsonl").write_text("corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        replay.prefetch()


def test_selection_and_next_url_are_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = {
        "type": "FeatureCollection",
        "features": [feature(1, climate_id="1234567")],
        "links": [],
    }
    monkeypatch.setattr(
        "crc_sdk.workflows.eccc_daily.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(page).encode()),
    )
    with pytest.raises(ValueError, match="selection"):
        plan(tmp_path).prefetch()
    page["features"] = []
    page["links"] = [{"rel": "next", "href": "https://example.com/other"}]
    with pytest.raises(RuntimeError, match="pagination"):
        plan(tmp_path).prefetch()


def test_refresh_failure_preserves_pinned_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = {"type": "FeatureCollection", "features": [feature(1)], "links": []}
    monkeypatch.setattr(
        "crc_sdk.workflows.eccc_daily.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(page).encode()),
    )
    request = plan(tmp_path)
    (original,) = request.prefetch()
    page["features"] = [feature(2)]
    with pytest.raises(RuntimeError, match="expected SHA"):
        request.cache(
            tmp_path, mode="refresh", sha256={2020: original["sha256"]}
        ).prefetch()
    replay = request.cache(tmp_path, mode="offline", sha256={2020: original["sha256"]})
    assert next(replay.batches()).to_pylist()[0]["date"].isoformat() == "2020-01-01"


def test_invalid_and_missing_requests(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ECCCDailyPlan.for_station("../BAD")
    with pytest.raises(ValueError):
        plan(tmp_path).years(2021, 2020)
    with pytest.raises(FileNotFoundError):
        plan(tmp_path).cache(tmp_path, mode="offline").prefetch()
    with pytest.raises(ValueError, match="checksum years"):
        plan(tmp_path).cache(tmp_path, sha256={2021: "0" * 64}).prefetch()
