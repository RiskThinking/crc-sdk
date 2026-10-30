from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from crc_sdk.workflows import HYDATDailyPlan


def feature(day: int, station: str = "02HA014") -> dict[str, object]:
    return {
        "type": "Feature",
        "id": f"{station}.2020-01-{day:02d}",
        "geometry": {"type": "Point", "coordinates": [-79.7, 43.2]},
        "properties": {
            "STATION_NUMBER": station,
            "STATION_NAME": "TEST",
            "DATE": f"2020-01-{day:02d}",
            "DISCHARGE": 1.2 if day == 1 else None,
            "LEVEL": 11.3,
            "DISCHARGE_SYMBOL_EN": "Ice Conditions",
            "DISCHARGE_SYMBOL_FR": "Conditions de glace",
        },
    }


def plan(tmp_path: Path) -> HYDATDailyPlan:
    return HYDATDailyPlan.for_station("02HA014").years(2020, 2020).cache(tmp_path)


def test_snapshot_flags_missing_and_replay(
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

    monkeypatch.setattr("crc_sdk.workflows.hydat.urlopen", fetch)
    base = HYDATDailyPlan.for_station("02HA014")
    assert base.selected_years == ()
    request = base.years(2020, 2020).cache(tmp_path)
    (source,) = request.prefetch()
    assert source["level_datum"] == "unspecified"
    assert len(source["page_sha256"]) == 2
    replay = request.cache(tmp_path, mode="offline", sha256={2020: source["sha256"]})
    batches = list(replay.batches(batch_size=1))
    assert len(calls) == len(batches) == 2
    row = batches[0].to_pylist()[0]
    assert row["discharge_m3_s"] == 1.2
    assert row["discharge_symbol_en"] == "Ice Conditions"
    assert row["level_m"] == 11.3
    assert batches[1].to_pylist()[0]["discharge_m3_s"] is None
    (replay._directory(2020) / "observations.jsonl").write_text("broken")
    with pytest.raises(RuntimeError, match="checksum"):
        replay.prefetch()


def test_wrong_station_and_untrusted_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = {
        "type": "FeatureCollection",
        "features": [feature(1, "02HA015")],
        "links": [],
    }
    monkeypatch.setattr(
        "crc_sdk.workflows.hydat.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(page).encode()),
    )
    with pytest.raises(ValueError, match="selection"):
        plan(tmp_path).prefetch()
    page["features"] = []
    page["links"] = [{"rel": "next", "href": "https://example.com/other"}]
    with pytest.raises(RuntimeError, match="pagination"):
        plan(tmp_path).prefetch()


def test_failed_refresh_and_invalid_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = {"type": "FeatureCollection", "features": [feature(1)], "links": []}
    monkeypatch.setattr(
        "crc_sdk.workflows.hydat.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(page).encode()),
    )
    request = plan(tmp_path)
    (source,) = request.prefetch()
    page["features"] = [feature(2)]
    with pytest.raises(RuntimeError, match="expected SHA"):
        request.cache(
            tmp_path, mode="refresh", sha256={2020: source["sha256"]}
        ).prefetch()
    assert (
        request.cache(tmp_path, mode="offline").prefetch()[0]["sha256"]
        == source["sha256"]
    )
    with pytest.raises(ValueError):
        HYDATDailyPlan.for_station("../NOPE")
    with pytest.raises(ValueError):
        request.years(2021, 2020)
    with pytest.raises(ValueError, match="checksum years"):
        request.cache(tmp_path, sha256={2021: "0" * 64}).prefetch()
