from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from crc_sdk.workflows import NOAASeaLevelPlan


class Response(io.BytesIO):
    headers: dict[str, str] = {}


def payload() -> dict[str, object]:
    return {
        "count": 2,
        "currPage": 1,
        "totalPages": 1,
        "SlrProjections": [
            {
                "stationId": "8638610",
                "stationName": "TEST",
                "longitude": -76.3,
                "latitude": 36.9,
                "reportYear": 2022,
                "projectionYear": 2005,
                "scenario": "Low",
                "units": "cm",
                "projectionRsl": 0,
                "projectionCiLow": 0,
                "projectionCiHigh": 0,
            },
            {
                "stationId": "8638610",
                "stationName": "TEST",
                "longitude": -76.3,
                "latitude": 36.9,
                "reportYear": 2022,
                "projectionYear": 2050,
                "scenario": "Low",
                "units": "cm",
                "projectionRsl": 30,
                "projectionCiLow": 20,
                "projectionCiHigh": 45,
            },
        ],
    }


def test_scenario_baseline_units_and_pinned_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = payload()
    calls: list[str] = []

    def fetch(url: str, **kwargs: object) -> Response:
        calls.append(url)
        return Response(json.dumps(source).encode())

    monkeypatch.setattr("crc_sdk.workflows.noaa_sea_level.urlopen", fetch)
    base = NOAASeaLevelPlan.for_station("8638610")
    request = base.scenario("low").cache(tmp_path)
    assert base.selected_scenario == "all"
    manifest = request.prefetch()
    replay = request.cache(tmp_path, mode="offline", sha256=manifest["sha256"])
    table = replay.projections()
    assert len(calls) == 1
    assert table.num_rows == 2
    row = table.to_pylist()[1]
    assert row["report_year"] == 2022 and row["baseline_year"] == 2005
    assert row["projection_year"] == 2050 and row["scenario"] == "Low"
    assert row["relative_change_cm"] == 30 and row["interval_low_cm"] == 20
    assert "annual_exceedance_probability" not in table.column_names
    (replay._directory() / "source.json").write_text("corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        replay.prefetch()


def test_wrong_report_units_scenario_and_incomplete_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = payload()
    monkeypatch.setattr(
        "crc_sdk.workflows.noaa_sea_level.urlopen",
        lambda *a, **kw: Response(json.dumps(source).encode()),
    )
    request = NOAASeaLevelPlan.for_station("8638610").cache(tmp_path)
    source["totalPages"] = 2
    with pytest.raises(ValueError, match="pagination"):
        request.prefetch()
    source["totalPages"] = 1
    source["count"] = 3
    with pytest.raises(ValueError, match="count mismatch"):
        request.prefetch()
    source["count"] = 2
    with pytest.raises(ValueError, match="selection"):
        request.scenario("high").prefetch()
    source["SlrProjections"] = [
        {
            "stationId": "8638610",
            "reportYear": 2017,
            "projectionYear": 2050,
            "scenario": "Low",
            "units": "cm",
        }
    ]
    source["count"] = 1
    with pytest.raises(ValueError, match="report mismatch"):
        request.prefetch()


def test_failed_refresh_preserves_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = payload()
    monkeypatch.setattr(
        "crc_sdk.workflows.noaa_sea_level.urlopen",
        lambda *a, **kw: Response(json.dumps(source).encode()),
    )
    request = NOAASeaLevelPlan.for_station("8638610").cache(tmp_path)
    original = request.prefetch()
    source["count"] = 0
    source["SlrProjections"] = []
    with pytest.raises(ValueError, match="unavailable"):
        request.cache(tmp_path, mode="refresh").prefetch()
    assert (
        request.cache(tmp_path, mode="offline").prefetch()["sha256"]
        == original["sha256"]
    )
    with pytest.raises(ValueError):
        NOAASeaLevelPlan.for_station("../bad")
    with pytest.raises(ValueError):
        request.scenario("ssp585")
