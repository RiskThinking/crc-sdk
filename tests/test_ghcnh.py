from __future__ import annotations

import calendar
import io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.providers.ghcnh import GHCNhCompleteness, fetch_daily, reduce_daily
from crc_sdk.providers.isd import ISDCompleteness
from crc_sdk.providers.isd import reduce_daily as reduce_isd
from crc_sdk.workflows import BlockExtremaPolicy, GHCNhPlan, HazardDataset

STATION = "USW00094846"


def parquet_bytes() -> bytes:
    flags = [
        ("1", "343", -2.2),
        ("5", "343", 3.4),
        ("5", "220", 50.0),
        ("1", "220", 4.1),
        ("", "999", -3.0),
        (None, "343", 2.0),
        ("s", "343", 55.0),
        ("1s", "343", 55.0),
        ("1", "999", 55.0),
        ("9", "343", 55.0),
        ("", "343", -999.9),
    ]
    rows = [
        {
            "STATION": STATION,
            "DATE": f"2020-01-01T{hour:02d}:00:00",
            "LATITUDE": 41.9603,
            "LONGITUDE": -87.9317,
            "ELEVATION": -999.9,
            "temperature": value,
            "temperature_Quality_Code": flag,
            "temperature_Source_Code": source,
            "unused": "not part of the projected observation contract",
        }
        for hour, (flag, source, value) in enumerate(flags)
    ]
    output = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), output, row_group_size=3)
    return output.getvalue()


def test_source_aware_qc_celsius_and_nulls(tmp_path: Path) -> None:
    table, checksum = reduce_daily(
        io.BytesIO(parquet_bytes()),
        STATION,
        2020,
        GHCNhCompleteness(),
        temporary_directory=str(tmp_path),
    )
    row = table.to_pylist()[0]
    assert row["reports"] == 11
    assert row["accepted_reports"] == row["valid_hours"] == 5
    assert row["temperature_min"] == -3.0 and row["temperature_max"] == 4.1
    assert row["elevation"] is None
    assert len(checksum) == 64
    strict, _ = reduce_daily(
        io.BytesIO(parquet_bytes()),
        STATION,
        2020,
        GHCNhCompleteness(quality_codes=("",)),
        temporary_directory=str(tmp_path),
    )
    assert strict.to_pylist()[0]["accepted_reports"] == 2
    with pytest.raises(ValueError, match="quality_codes"):
        GHCNhCompleteness(quality_codes=("9",))


def test_common_daily_contract_with_isd(tmp_path: Path) -> None:
    raw = (
        b"STATION,DATE,LATITUDE,LONGITUDE,ELEVATION,TMP\n"
        b'72530094846,2020-01-01T00:00:00,41.9603,-87.9317,204.8,"-0022,1"\n'
    )
    isd, _ = reduce_isd(
        io.BytesIO(raw),
        "72530094846",
        2020,
        ISDCompleteness(),
        temporary_directory=str(tmp_path),
    )
    output = io.BytesIO()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "STATION": "72530094846",
                    "DATE": "2020-01-01T00:00:00",
                    "LATITUDE": 41.9603,
                    "LONGITUDE": -87.9317,
                    "ELEVATION": 204.8,
                    "temperature": -2.2,
                    "temperature_Quality_Code": "1",
                    "temperature_Source_Code": "343",
                }
            ]
        ),
        output,
    )
    ghcnh, _ = reduce_daily(
        io.BytesIO(output.getvalue()),
        "72530094846",
        2020,
        GHCNhCompleteness(),
        temporary_directory=str(tmp_path),
    )
    assert isd.equals(ghcnh)


def test_fluent_compatibility_and_identifiers() -> None:
    plan = (
        HazardDataset.ghcnh()
        .for_stations([STATION])
        .years(2000, 2024)
        .quality()
        .execution(workers=2)
        .canonicalize()
    )
    assert isinstance(plan, GHCNhPlan)
    details = plan.explain(format="json")
    assert isinstance(details, dict)
    assert details["hazard_name"] == "ghcnh_temperature_max"
    assert details["completeness"]["quality_codes"] == ("", "1", "5")
    assert plan.for_stations(["CAU60-00150"]).selected_stations == ("CAU60-00150",)
    with pytest.raises(ValueError, match="GHCN"):
        plan.for_stations(["72530094846"])
    with pytest.raises(ValueError, match="USAF"):
        HazardDataset.isd().for_stations([STATION])
    with pytest.raises(ValueError, match="complete-year"):
        plan.years(2020, datetime.now(timezone.utc).year)
    assert plan.years(2025, 2025).selected_years == (2025,)


def fake_daily(station: str, year: int, *args: Any, **kwargs: Any) -> Any:
    rows = [
        {
            "station": station,
            "day": date(year, 1, 1) + timedelta(days=day),
            "latitude": 41.9603,
            "latitude_max": 41.9603,
            "longitude": -87.9317,
            "longitude_max": -87.9317,
            "elevation": 204.8,
            "elevation_max": 204.8,
            "reports": 24,
            "accepted_reports": 24,
            "valid_hours": 24,
            "temperature_min": -20.0 - (year - 2000) ** 0.7,
            "temperature_max": 30.0 + (year - 2000) ** 0.7,
        }
        for day in range(366 if calendar.isleap(year) else 365)
    ]
    return pa.Table.from_pylist(rows), {"url": "https://example.test", "etag": "test"}


def test_separate_cache_identity_both_tails_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crc_sdk.connectors.parquet import read_hazard_metadata

    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fake_daily)
    monkeypatch.setattr("crc_sdk.providers.ghcnh.fetch_daily", fake_daily)
    cache = tmp_path / "cache"
    legacy = (
        HazardDataset.isd().for_stations(["72530094846"]).years(2000, 2005).cache(cache)
    )
    assert legacy.prefetch().cache_misses == 6
    for recipe in ("temperature_max", "temperature_min"):
        plan = (
            HazardDataset.ghcnh(recipe)
            .for_stations([STATION])
            .years(2000, 2005)
            .cache(cache)
            .canonicalize(policy=BlockExtremaPolicy.curated(minimum_years=4))
        )
        path = tmp_path / f"{recipe}.parquet"
        plan.materialize(path)
        rows = pq.ParquetFile(path).read().to_pylist()
        assert rows[0]["source_id"] == f"noaa-ghcnh:{STATION}"
        assert rows[0]["hazard_name"] == f"ghcnh_{recipe}"
        assert len(list((cache / "daily").glob("*.parquet"))) == 12
        metadata = read_hazard_metadata(path)
        assert metadata.source is not None
        assert metadata.source.provider == "noaa-ghcnh"
        assert metadata.source.licence == "CC0-1.0"
        assert metadata.source.version == "ghcnh-v1"
        assert plan.cache(cache, mode="offline").prefetch().cache_hits == 6


def test_conditional_reads_and_projection_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = parquet_bytes()

    class Head(io.BytesIO):
        headers = {
            "ETag": '"object-etag"',
            "Content-Length": str(len(raw)),
            "Last-Modified": "test",
        }

    monkeypatch.setattr("crc_sdk.providers.ghcnh.urlopen", lambda *a, **kw: Head())
    requests: list[dict[str, Any]] = []

    def open_remote(url: str, *args: Any, **kwargs: Any) -> Any:
        requests.append(kwargs)
        return io.BytesIO(raw)

    monkeypatch.setattr("crc_sdk.providers.ghcnh.fsspec.open", open_remote)
    table, info = fetch_daily(
        STATION,
        2020,
        GHCNhCompleteness(),
        temporary_directory=str(tmp_path),
        memory_limit="256MB",
    )
    assert table.num_rows == 1
    assert requests[0]["headers"] == {"If-Match": '"object-etag"'}
    assert "raw_sha256" not in info
    assert len(info["projected_columns"]) == 8
    assert len(info["projection_sha256"]) == 64
