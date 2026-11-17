from __future__ import annotations

import calendar
import csv
import hashlib
import io
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk._providers.isd import ISDCompleteness, reduce_daily

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
