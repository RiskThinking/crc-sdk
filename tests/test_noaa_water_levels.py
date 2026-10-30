from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from crc_sdk.workflows import NOAAExtremeWaterLevelPlan


class Response(io.BytesIO):
    headers: dict[str, str] = {}


def payloads() -> dict[str, dict[str, object]]:
    return {
        "station": {
            "ExtremeWaterLevels": [
                {
                    "stationId": "8638610",
                    "stationName": "TEST",
                    "longitude": -76.3,
                    "latitude": 36.9,
                    "epoch": "1983-2001",
                }
            ]
        },
        "levels": {
            "stationId": "8638610",
            "units": "meters",
            "datum": "MHHW",
            "GEV_HI": {
                "exceedanceLevels": [
                    {
                        "probability": 0.313,
                        "returnPeriod": 3.2,
                        "level": 0.87,
                        "lower90CI": 0.82,
                        "upper90CI": 0.92,
                    }
                ]
            },
        },
        "annuals": {
            "stationId": "8638610",
            "units": "meters",
            "datum": "mhhw",
            "trend": 4.84,
            "trendUnits": "mm/yr",
            "notice": "Detrended; do not shift datum.",
            "extremes": [{"year": 2020, "value": 1.25}],
        },
    }


def install_fake(
    monkeypatch: pytest.MonkeyPatch, sources: dict[str, dict[str, object]]
) -> list[str]:
    requests: list[str] = []

    def fetch(url: str, **kwargs: object) -> Response:
        query = parse_qs(urlsplit(url).query)
        resource = (
            "levels"
            if "details" in query
            else "annuals"
            if "extremeType" in query
            else "station"
        )
        requests.append(resource)
        return Response(json.dumps(sources[resource]).encode())

    monkeypatch.setattr("crc_sdk.workflows.noaa_water_levels.urlopen", fetch)
    return requests


def test_snapshot_preserves_rounded_labels_confidence_and_detrending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = install_fake(monkeypatch, payloads())
    base = NOAAExtremeWaterLevelPlan.for_station("8638610")
    request = base.cache(tmp_path)
    assert base.cache_dir is None
    provenance = request.prefetch()
    hashes = {key: item["sha256"] for key, item in provenance.items()}
    offline = request.cache(tmp_path, mode="offline", sha256=hashes)
    row = offline.source_levels().to_pylist()[0]
    assert row["annual_exceedance_probability"] == 0.313
    assert row["published_return_period_years"] == 3.2
    assert row["lower_90ci_m"] == 0.82
    assert row["upper_90ci_m"] == 0.92
    assert row["datum"] == "MHHW" and row["datum_epoch"] == "1983-2001"
    annual = offline.annual_extremes().to_pylist()[0]
    assert annual["detrended_extreme_m"] == 1.25
    assert "do not shift" in annual["reference_notice"]
    assert annual["datum_epoch"] == "1983-2001"
    assert len(requests) == 3
    (offline._directory("levels") / "source.json").write_text("corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        offline.source_levels()


def test_bad_confidence_and_datum_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = payloads()
    sources["levels"] = {
        "stationId": "8638610",
        "units": "meters",
        "datum": "MHHW",
        "GEV_HI": {
            "exceedanceLevels": [
                {
                    "probability": 0.1,
                    "returnPeriod": 10,
                    "level": 1,
                    "lower90CI": 2,
                    "upper90CI": 3,
                }
            ]
        },
    }
    install_fake(monkeypatch, sources)
    with pytest.raises(ValueError, match="confidence"):
        NOAAExtremeWaterLevelPlan.for_station("8638610").cache(tmp_path).prefetch()
    sources = payloads()
    sources["annuals"]["datum"] = "NAVD88"
    install_fake(monkeypatch, sources)
    with pytest.raises(ValueError, match="datum labels differ"):
        NOAAExtremeWaterLevelPlan.for_station("8638610").cache(
            tmp_path, mode="refresh"
        ).source_info()


def test_failed_refresh_keeps_cached_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = payloads()
    install_fake(monkeypatch, sources)
    request = NOAAExtremeWaterLevelPlan.for_station("8638610").cache(tmp_path)
    source = request.prefetch()
    sources["levels"]["GEV_HI"] = {
        "exceedanceLevels": [{"probability": 0.5, "returnPeriod": 2, "level": 1}]
    }
    with pytest.raises(RuntimeError, match="expected SHA"):
        request.cache(
            tmp_path, mode="refresh", sha256={"levels": source["levels"]["sha256"]}
        ).source_levels()
    assert (
        request.cache(tmp_path, mode="offline")
        .source_levels()
        .to_pylist()[0]["water_level_m"]
        == 0.87
    )
    with pytest.raises(ValueError):
        NOAAExtremeWaterLevelPlan.for_station("../bad")


def test_missing_trend_context_is_not_silently_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = payloads()
    sources["annuals"]["notice"] = None
    install_fake(monkeypatch, sources)
    with pytest.raises(ValueError, match="detrending notice"):
        NOAAExtremeWaterLevelPlan.for_station("8638610").cache(tmp_path).prefetch()


def test_unavailable_and_malformed_axis_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError):
        NOAAExtremeWaterLevelPlan.for_station("8638610").cache(
            tmp_path, mode="offline"
        ).source_levels()
    sources = payloads()
    sources["levels"] = {
        "stationId": "8638610",
        "units": "meters",
        "datum": "MHHW",
        "GEV_HI": {
            "exceedanceLevels": [
                {"probability": 0.5, "returnPeriod": 2, "level": 1},
                {"probability": 0.5, "returnPeriod": 2, "level": 2},
            ]
        },
    }
    install_fake(monkeypatch, sources)
    with pytest.raises(ValueError, match="exceedance"):
        NOAAExtremeWaterLevelPlan.for_station("8638610").cache(tmp_path).prefetch()
