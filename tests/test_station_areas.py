from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.providers.station_catalogues import fetch_catalogue
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

BOUNDS = (-87.94, 41.95, -87.92, 41.97)
STATION = "USW00094846"


def catalogue() -> Any:
    return pa.Table.from_pylist(
        [
            {
                "station": station,
                "name": station,
                "latitude": lat,
                "longitude": lon,
                "elevation": 200.0,
                "start_year": start,
                "end_year": end,
            }
            for station, lat, lon, start, end in [
                (STATION, 41.96, -87.93, 1990, 2025),
                ("USW00000001", 41.95, -87.94, 1990, 2025),
                ("USW00000002", 42.0, -87.0, 1990, 2025),
                ("USW00000003", 41.96, -87.93, 1980, 1990),
            ]
        ]
    )


def fake_catalogue(*args: Any) -> Any:
    return catalogue(), {
        "url": "https://example.test/catalogue.csv",
        "raw_sha256": "abc",
        "etag": "test",
        "last_modified": "test",
    }


@pytest.mark.parametrize("provider", ["isd", "ghcnh"])
def test_catalogue_decode_preserves_ids_and_missing_elevation(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if provider == "isd":
        raw = (
            b"USAF,WBAN,STATION NAME,LAT,LON,ELEV(M),BEGIN,END\n"
            b"007018,00001,Example,41.96,-87.93,9999,19900101,20250824\n"
            b"007019,00002,Missing,,,9999,19900101,20250824\n"
        )
        pattern, station = r"[0-9]{11}", "00701800001"
    else:
        raw = (
            b"GHCN_ID,NAME,LATITUDE,LONGITUDE,ELEVATION\n"
            b"CAU60-00150,Example,41.96,-87.93,-999.9\n"
        )
        pattern, station = r"[A-Z]{2}[A-Z0-9-]{9}", "CAU60-00150"

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(raw)), "ETag": "test"}

    monkeypatch.setattr(
        "crc_sdk.providers.station_catalogues.urlopen",
        lambda *args, **kwargs: Response(raw),
    )
    table, meta = fetch_catalogue(provider, "https://example.test", pattern)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["station"] == station
    assert row["elevation"] is None
    assert row["start_year"] == (1990 if provider == "isd" else None)
    assert meta["raw_sha256"] == hashlib.sha256(raw).hexdigest()


def test_area_builders_are_lazy_and_switch_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("builder must not access the network")

    monkeypatch.setattr("crc_sdk.workflows.stations.fetch_catalogue", fail)
    plan = HazardDataset.ghcnh().for_area(BOUNDS).years(2000, 2024)
    assert plan.selected_stations == ()
    details = plan.explain(format="json")
    assert isinstance(details, dict)
    assert details["areas"] == (BOUNDS,)
    pinned = plan.for_stations([STATION])
    assert pinned.requested_areas == ()
    assert pinned.for_area(BOUNDS).selected_stations == ()
    with pytest.raises(ValueError, match="at least one"):
        plan.for_areas([])
    with pytest.raises(ValueError, match="ordered"):
        plan.for_area((10, 10, 0, 0))


def test_area_union_year_filter_metadata_cache_and_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crc_sdk.workflows.stations.fetch_catalogue", fake_catalogue)
    plan = (
        HazardDataset.ghcnh()
        .for_areas([BOUNDS, BOUNDS, (-88.0, 41.9, -87.9, 42.0)])
        .years(2000, 2024)
        .cache(tmp_path)
    )
    table = plan.stations()
    assert table.column("station").to_pylist() == ["USW00000001", STATION]
    resolved = plan.resolve_stations()
    assert resolved.selected_stations == ("USW00000001", STATION)
    assert resolved.selection_catalogue_sha256 == "abc"
    assert plan.selected_stations == ()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline catalogue lookup must not use network")

    monkeypatch.setattr("crc_sdk.workflows.stations.fetch_catalogue", fail)
    assert plan.cache(tmp_path, mode="offline").stations().equals(table)
    path = tmp_path / "catalogues" / "ghcnh.parquet"
    path.write_bytes(b"corrupt")
    with pytest.raises(FileNotFoundError, match="catalogue"):
        plan.cache(tmp_path, mode="offline").stations()
    with pytest.raises(FileNotFoundError, match="catalogue"):
        plan.cache(tmp_path / "empty", mode="offline").stations()


def test_empty_selection_stops_before_observation_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crc_sdk.workflows.stations.fetch_catalogue", fake_catalogue)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no observation requests expected")

    monkeypatch.setattr("crc_sdk.providers.ghcnh.fetch_daily", fail)
    plan = (
        HazardDataset.ghcnh().for_area((0, 0, 1, 1)).years(2000, 2020).cache(tmp_path)
    )
    with pytest.raises(ValueError, match="no catalogue stations"):
        plan.prefetch()


def test_area_and_explicit_station_observations_share_cache_and_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_ghcnh import fake_daily

    from crc_sdk.connectors.parquet import read_hazard_metadata

    monkeypatch.setattr("crc_sdk.workflows.stations.fetch_catalogue", fake_catalogue)
    calls: list[str] = []

    def fetch(station: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(station)
        return fake_daily(station, *args, **kwargs)

    monkeypatch.setattr("crc_sdk.providers.ghcnh.fetch_daily", fetch)
    # A narrow box contains O'Hare, excluding the station on BOUNDS' west edge.
    plan = (
        HazardDataset.ghcnh()
        .for_area((-87.935, 41.955, -87.925, 41.965))
        .years(2000, 2005)
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy.curated(minimum_years=4))
    )
    assert plan.prefetch().cache_misses == 6
    assert set(calls) == {STATION}
    assert plan.for_stations([STATION]).prefetch().cache_hits == 6
    output = tmp_path / "area.parquet"
    plan.cache(tmp_path / "cache", mode="offline").materialize(output)
    explicit = tmp_path / "explicit.parquet"
    plan.for_stations([STATION]).materialize(explicit)
    assert pq.ParquetFile(output).read().equals(pq.ParquetFile(explicit).read())
    meta = read_hazard_metadata(output)
    assert meta.source is not None
    request = json.loads(unquote(str(meta.source.uri).split("#crc-request=")[1]))
    assert request["stations"] == [STATION]
    assert request["areas"] == [[-87.935, 41.955, -87.925, 41.965]]
    assert request["catalogue"]["raw_sha256"] == "abc"
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert "ghcnh" in manifest["catalogues"]
    assert len(manifest["entries"]) == 6


def test_refresh_updates_discovery_without_changing_pinned_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crc_sdk.workflows.stations.fetch_catalogue", fake_catalogue)
    plan = HazardDataset.ghcnh().for_area(BOUNDS).years(2000, 2024).cache(tmp_path)
    pinned = plan.resolve_stations()

    def updated(*args: Any) -> Any:
        return catalogue().slice(0, 1), {
            "url": "https://example.test/catalogue.csv",
            "raw_sha256": "def",
            "etag": "new",
        }

    monkeypatch.setattr("crc_sdk.workflows.stations.fetch_catalogue", updated)
    refreshed = plan.cache(tmp_path, mode="refresh").resolve_stations()
    assert refreshed.selected_stations == (STATION,)
    assert refreshed.selection_catalogue_sha256 == "def"
    assert pinned.selected_stations == ("USW00000001", STATION)
    assert pinned.selection_catalogue_sha256 == "abc"
