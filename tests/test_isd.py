from __future__ import annotations

import calendar
import csv
import hashlib
import io
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.providers.isd import ISDCompleteness, reduce_daily
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset
from crc_sdk.workflows.isd import ISDPlan

STATION = "71624099999"


def test_streaming_decode_qc_units_and_duplicate_hours(tmp_path: Path) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "TMP"])
    for time, value in [
        ("00:00:00", "-0123,1"),
        ("00:00:00", "+0450,3"),
        ("00:30:00", "+0010,5"),
        ("01:00:00", "+9999,1"),
        ("02:00:00", "+0200,9"),
        ("03:00:00", "+0040,5"),
    ]:
        writer.writerow([STATION, f"2020-01-01T{time}", 43.67, -79.63, 173, value])
    raw = buffer.getvalue().encode()
    daily, digest = reduce_daily(
        io.BytesIO(raw),
        STATION,
        2020,
        ISDCompleteness(),
        temporary_directory=str(tmp_path),
    )
    row = daily.to_pylist()[0]
    assert row["valid_hours"] == 2
    assert row["accepted_reports"] == 3
    assert row["reports"] == 6
    assert row["temperature_min"] == -12.3
    assert row["temperature_max"] == 4
    assert digest == hashlib.sha256(raw).hexdigest()


def fake_fetch(
    station: str, year: int, completeness: ISDCompleteness, **kwargs: Any
) -> tuple[Any, dict[str, Any]]:
    days = 366 if calendar.isleap(year) else 365
    # One rejected day is retained in QA; 2002 has only half a calendar year.
    observed = days // 2 if year == 2002 else days
    rows = [
        {
            "station": station,
            "day": date(year, 1, 1) + timedelta(days=day),
            "latitude": 43.67,
            "latitude_max": 43.67,
            "longitude": -79.63,
            "longitude_max": -79.63,
            "elevation": 173.0,
            "elevation_max": 173.0,
            "reports": 30,
            "accepted_reports": 24,
            "valid_hours": 17 if day == 0 else 24,
            "temperature_min": -20.0 - (year - 2000) ** 0.7,
            "temperature_max": 30.0 + (year - 2000) ** 0.7,
        }
        for day in range(observed)
    ]
    return pa.Table.from_pylist(rows), {
        "url": f"https://example.test/{year}/{station}.csv",
        "raw_sha256": str(year),
        "etag": f"etag-{year}",
        "last_modified": "test",
    }


def cached_plan(tmp_path: Path) -> ISDPlan:
    return (
        HazardDataset.isd()
        .for_stations([STATION])
        .years(2000, 2005)
        .cache(tmp_path / "cache")
        .canonicalize(policy=BlockExtremaPolicy.curated(minimum_years=4))
    )


def test_lazy_planning_validation() -> None:
    original = HazardDataset.isd()
    plan = original.for_stations([STATION]).years(1991, 2020)
    assert original.selected_stations == ()
    assert plan.explain(format="json")["spatial_support"].startswith("station point")  # type: ignore[index]
    with pytest.raises(ValueError, match="1901"):
        original.years(2020, 2025)
    with pytest.raises(ValueError, match="eleven-digit"):
        original.for_stations(["71624"])
    with pytest.raises(ValueError, match="recipe"):
        HazardDataset.isd("rx1day")
    with pytest.raises(ValueError, match="fractions"):
        original.quality(minimum_day_fraction=0)
    with pytest.raises(ValueError, match="quality_codes"):
        original.quality(quality_codes=("3",))
    with pytest.raises(ValueError, match="prefetch"):
        original.prefetch()


def test_cache_completeness_leap_years_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fake_fetch)
    plan = cached_plan(tmp_path)
    assert plan.prefetch().cache_misses == 6
    assert plan.prefetch().cache_hits == 6

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline must not access network")

    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fail)
    offline = plan.cache(tmp_path / "cache", mode="offline")
    rows = offline.annual_extremes().to_pylist()
    assert rows[0]["expected_days"] == 366
    assert rows[0]["valid_days"] == 365
    assert rows[2]["value"] is None and not rows[2]["complete"]
    assert rows[1]["value"] == 31.0
    # Threshold changes cannot silently reuse reductions with another policy.
    with pytest.raises(FileNotFoundError):
        offline.quality(minimum_hour_fraction=1).prefetch()
    path = next((tmp_path / "cache" / "daily").glob("*.parquet"))
    path.write_bytes(b"corrupt")
    with pytest.raises(FileNotFoundError, match="corrupt"):
        offline.prefetch()


def test_station_points_fit_both_tails_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import h3  # type: ignore[import-untyped]
    from shapely import wkb  # type: ignore[import-untyped]

    from crc_sdk.connectors.parquet import read_hazard_metadata

    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fake_fetch)
    for recipe in ("temperature_max", "temperature_min"):
        plan = cached_plan(tmp_path)
        plan = ISDPlan(**{**plan.__dict__, "recipe": recipe})
        output = tmp_path / f"{recipe}.parquet"
        plan.materialize(output)
        rows = pq.ParquetFile(output).read().to_pylist()
        assert len(rows) == 1
        assert rows[0]["source_id"] == f"noaa-isd:{STATION}"
        assert wkb.loads(rows[0]["source_geometry"]).geom_type == "Point"
        assert rows[0]["cell_index"] == h3.str_to_int(
            h3.latlng_to_cell(43.67, -79.63, 5)
        )
        assert rows[0]["hazard_name"] == f"isd_{recipe}"
        metadata = read_hazard_metadata(output)
        assert metadata.source is not None
        assert metadata.source.version == "isd-2025-08-24"
        assert "minimum_hour_fraction" in str(metadata.source.uri)
        assert metadata.temporal_window is not None
        assert metadata.temporal_window.minimum_complete_years == 4


def test_too_few_complete_years_and_coordinate_jumps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fake_fetch)
    plan = cached_plan(tmp_path).canonicalize(
        policy=BlockExtremaPolicy.curated(minimum_years=6)
    )
    output = tmp_path / "empty.parquet"
    plan.materialize(output)
    assert pq.ParquetFile(output).metadata.num_rows == 0
    with pytest.raises(ValueError, match="complete years"):
        plan.years(2000, 2003).materialize(output)


def test_missing_object_is_an_excluded_year_and_other_http_errors_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from urllib.error import HTTPError

    from crc_sdk.providers.isd import fetch_daily

    def missing(url: str, **kwargs: Any) -> Any:
        raise HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("crc_sdk.providers.isd.urlopen", missing)
    table, remote = fetch_daily(
        STATION,
        2020,
        ISDCompleteness(),
        temporary_directory=str(tmp_path),
        memory_limit="256MB",
    )
    assert table.num_rows == 0 and remote["missing"]
    plan = cached_plan(tmp_path)
    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fetch_daily)
    annual = plan.annual_extremes().to_pylist()
    assert len(annual) == 6
    assert all(row["value"] is None for row in annual)
    path = tmp_path / "empty.parquet"
    plan.materialize(path)
    assert pq.ParquetFile(path).metadata.num_rows == 0

    def forbidden(url: str, **kwargs: Any) -> Any:
        raise HTTPError(url, 403, "Forbidden", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("crc_sdk.providers.isd.urlopen", forbidden)
    with pytest.raises(HTTPError):
        plan.cache(tmp_path / "cache", mode="refresh").prefetch()


def test_cache_refresh_and_stream_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []

    def fetch(station: str, year: int, *args: Any, **kwargs: Any) -> Any:
        calls.append((station, year))
        return fake_fetch(station, year, *args, **kwargs)

    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fetch)
    plan = cached_plan(tmp_path)
    plan.prefetch()
    plan.cache(tmp_path / "cache", mode="refresh").prefetch()
    assert len(calls) == 12
    path = tmp_path / "annual.parquet"
    plan.write_annual_extremes(path)
    assert pq.ParquetFile(path).read().equals(plan.annual_extremes())
    plan.cache(None, mode="stream").materialize(tmp_path / "stream.parquet")
    assert len(calls) == 18
    assert not list(tmp_path.glob("crc-isd-*"))


def test_coordinate_jump_and_same_cell_station_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second_station = "71624099998"
    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fake_fetch)
    plan = cached_plan(tmp_path).for_stations([STATION, second_station])
    output = tmp_path / "stations.parquet"
    plan.materialize(output)
    rows = pq.ParquetFile(output).read().to_pylist()
    assert len(rows) == 2
    assert rows[0]["cell_index"] == rows[1]["cell_index"]
    assert rows[0]["source_id"] != rows[1]["source_id"]

    def moved(station: str, year: int, *args: Any, **kwargs: Any) -> Any:
        table, remote = fake_fetch(station, year, *args, **kwargs)
        if year == 2001:
            for column in ("latitude", "latitude_max"):
                table = table.set_column(
                    table.schema.get_field_index(column),
                    column,
                    pa.array([45.0] * table.num_rows),
                )
        return table, remote

    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", moved)
    with pytest.raises(ValueError, match="moved"):
        plan.cache(tmp_path / "moved").materialize(output)


def test_large_csv_stream_and_truncation_detection(tmp_path: Path) -> None:
    from http.client import IncompleteRead

    header = b"STATION,DATE,LATITUDE,LONGITUDE,ELEVATION,TMP\n"
    row = b'71624099999,2020-01-01T00:00:00,43.67,-79.63,9999,"+0100,1"\n'
    raw = header + row * 20000
    assert len(raw) > 1024 * 1024
    daily, digest = reduce_daily(
        io.BytesIO(raw),
        STATION,
        2020,
        ISDCompleteness(),
        temporary_directory=str(tmp_path),
        expected_bytes=len(raw),
    )
    assert daily.to_pylist()[0]["reports"] == 20000
    assert daily.to_pylist()[0]["valid_hours"] == 1
    assert daily.to_pylist()[0]["elevation"] is None
    assert digest == hashlib.sha256(raw).hexdigest()
    with pytest.raises(IncompleteRead):
        reduce_daily(
            io.BytesIO(header + row),
            STATION,
            2020,
            ISDCompleteness(),
            temporary_directory=str(tmp_path),
            expected_bytes=len(raw),
        )


def test_transient_retry_and_partial_cache_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from urllib.error import URLError

    from crc_sdk.providers.isd import fetch_daily
    from crc_sdk.providers.stations import empty_daily

    attempts = 0

    def transient(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("connection reset")
        return empty_daily(), {"missing": True}

    monkeypatch.setattr("crc_sdk.providers.isd._fetch_daily", transient)
    monkeypatch.setattr("crc_sdk.providers.isd.time.sleep", lambda _: None)
    fetch_daily(
        STATION,
        2020,
        ISDCompleteness(),
        temporary_directory=str(tmp_path),
        memory_limit="256MB",
    )
    assert attempts == 2

    def interrupted(station: str, year: int, *args: Any, **kwargs: Any) -> Any:
        if year == 2002:
            raise URLError("offline")
        return fake_fetch(station, year, *args, **kwargs)

    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", interrupted)
    plan = cached_plan(tmp_path).execution(workers=1)
    with pytest.raises(URLError):
        plan.prefetch()
    monkeypatch.setattr("crc_sdk.providers.isd.fetch_daily", fake_fetch)
    result = plan.prefetch()
    assert result.cache_hits == 2 and result.cache_misses == 4
