from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast
from urllib.request import Request

import pytest

from crc_sdk.workflows import USGSStreamflowPlan


def feature(day: str, value: str | None) -> dict[str, object]:
    return {
        "type": "Feature",
        "id": day,
        "geometry": {"type": "Point", "coordinates": [-75.0, 39.0]},
        "properties": {
            "monitoring_location_id": "USGS-01491000",
            "parameter_code": "00060",
            "statistic_id": "00003",
            "time": day,
            "value": value,
            "unit_of_measure": "ft^3/s",
            "approval_status": "Provisional",
            "qualifier": ["Ice"],
            "time_series_id": "a",
        },
    }


def plan(tmp_path: Path) -> USGSStreamflowPlan:
    return (
        USGSStreamflowPlan.for_station("USGS-01491000")
        .between("2020-01-01", "2020-01-02")
        .cache(tmp_path)
    )


def test_paginated_snapshot_and_offline_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [
        {
            "type": "FeatureCollection",
            "features": [feature("2020-01-01", "4.2")],
            "links": [{"rel": "next", "href": "?page=2"}],
        },
        {
            "type": "FeatureCollection",
            "features": [feature("2020-01-02", None)],
            "links": [],
        },
    ]
    calls: list[str] = []

    def fetch(url: Request, **kwargs: object) -> io.BytesIO:
        assert url.get_header("User-agent") == "crc-sdk"
        calls.append(url.full_url)
        return io.BytesIO(json.dumps(pages[len(calls) - 1]).encode())

    monkeypatch.setattr("crc_sdk.workflows.usgs.urlopen", fetch)
    request = plan(tmp_path)
    assert request.start is not None
    provenance = request.prefetch()
    assert provenance["pages"] == 2
    assert provenance["observations"] == 2
    offline = request.cache(tmp_path, mode="offline", sha256=provenance["sha256"])
    batches = list(offline.batches(batch_size=1))
    assert len(batches) == 2 and len(calls) == 2
    assert batches[0].to_pylist()[0]["flow"] == 4.2
    assert batches[0].to_pylist()[0]["qualifier_json"] == '["Ice"]'
    assert batches[1].to_pylist()[0]["flow"] is None
    (offline._directory() / "observations.jsonl").write_text("corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        offline.prefetch()


def test_rejects_wrong_selection_and_untrusted_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = feature("2020-01-01", "4")
    observation["properties"] = {
        **cast(dict[str, object], observation["properties"]),
        "parameter_code": "00065",
    }
    page = {"type": "FeatureCollection", "features": [observation], "links": []}
    monkeypatch.setattr(
        "crc_sdk.workflows.usgs.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(page).encode()),
    )
    with pytest.raises(ValueError, match="outside"):
        plan(tmp_path).prefetch()
    page["features"] = []
    page["links"] = [{"rel": "next", "href": "https://example.com/evil"}]
    with pytest.raises(RuntimeError, match="pagination"):
        plan(tmp_path).prefetch()


def test_invalid_arguments_and_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        USGSStreamflowPlan.for_station("../invalid")
    with pytest.raises(ValueError):
        plan(tmp_path).between("2020-02-01", "2020-01-01")
    with pytest.raises(FileNotFoundError):
        plan(tmp_path).cache(tmp_path, mode="offline").prefetch()
    with pytest.raises(ValueError):
        list(plan(tmp_path).batches(batch_size=0))


def test_failed_refresh_preserves_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = {
        "type": "FeatureCollection",
        "features": [feature("2020-01-01", "4")],
        "links": [],
    }
    monkeypatch.setattr(
        "crc_sdk.workflows.usgs.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(page).encode()),
    )
    request = plan(tmp_path)
    original = request.prefetch()
    page["features"] = [feature("2020-01-01", "5")]
    with pytest.raises(RuntimeError, match="expected SHA"):
        request.cache(tmp_path, mode="refresh", sha256=original["sha256"]).prefetch()
    replay = request.cache(tmp_path, mode="offline", sha256=original["sha256"])
    assert next(replay.batches()).to_pylist()[0]["flow"] == 4.0


@pytest.mark.parametrize(
    ("pages", "error", "message"),
    [
        ([{"numberMatched": 2, "numberReturned": 1}], RuntimeError, "incomplete"),
        ([{"numberReturned": 2}], ValueError, "page count"),
        (
            [{"numberMatched": 2}, {"numberMatched": 3}],
            RuntimeError,
            "match count changed",
        ),
    ],
)
def test_rejects_inconsistent_pagination_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pages: list[dict[str, int]],
    error: type[Exception],
    message: str,
) -> None:
    request = plan(tmp_path)
    replies = []
    for index, counts in enumerate(pages):
        replies.append(
            {
                "type": "FeatureCollection",
                "features": [feature(f"2020-01-0{index + 1}", "4")],
                "links": (
                    [{"rel": "next", "href": "?page=2"}]
                    if index + 1 < len(pages)
                    else []
                ),
                **counts,
            }
        )
    responses = iter(replies)
    monkeypatch.setattr(
        "crc_sdk.workflows.usgs.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(next(responses)).encode()),
    )
    with pytest.raises(error, match=message):
        request.prefetch()
    assert not (request._directory() / "observations.jsonl").exists()
    with pytest.raises(FileNotFoundError):
        request.cache(tmp_path, mode="offline").prefetch()


def test_complete_pagination_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = iter(
        [
            {
                "type": "FeatureCollection",
                "features": [feature("2020-01-01", "4")],
                "numberMatched": 2,
                "numberReturned": 1,
                "links": [{"rel": "next", "href": "?page=2"}],
            },
            {
                "type": "FeatureCollection",
                "features": [feature("2020-01-02", "5")],
                "numberMatched": 2,
                "numberReturned": 1,
                "links": [],
            },
        ]
    )
    monkeypatch.setattr(
        "crc_sdk.workflows.usgs.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(next(pages)).encode()),
    )
    assert plan(tmp_path).prefetch()["observations"] == 2


@pytest.mark.parametrize(
    "timestamp", ["2020-01-01", "2020-01-01T00:00:00Z", "2020-01-01T23:00:00-05:00"]
)
def test_observation_date_and_datetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timestamp: str
) -> None:
    observation = feature("2020-01-01", "4")
    properties = cast(dict[str, object], observation["properties"])
    properties["time"] = timestamp
    page = {"type": "FeatureCollection", "features": [observation], "links": []}
    monkeypatch.setattr(
        "crc_sdk.workflows.usgs.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(page).encode()),
    )
    assert (
        next(plan(tmp_path).batches()).to_pylist()[0]["date"].isoformat()
        == "2020-01-01"
    )
