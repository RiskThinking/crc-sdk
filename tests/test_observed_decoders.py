"""Conformance of the source-native station quality and unit decoding."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk._observed_decoders import ghcnh_rows, isd_rows
from crc_sdk._observed_sources import _catalogue_csv


def test_ghcnh_quality_is_source_aware_and_missing_reports_remain_visible(
    tmp_path: Path,
) -> None:
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
    path = tmp_path / "station.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "STATION": "TEST0000001",
                    "DATE": f"2020-01-01T{hour:02d}:00:00",
                    "LATITUDE": 41.96,
                    "LONGITUDE": -87.93,
                    "temperature": value,
                    "temperature_Quality_Code": flag,
                    "temperature_Source_Code": source,
                }
                for hour, (flag, source, value) in enumerate(flags)
            ]
        ),
        path,
    )
    rows = list(ghcnh_rows(path, "TEST0000001", 2020))
    assert len(rows) == len(flags)
    valid = [row["value"] for row in rows if row["value"] is not None]
    assert valid == [-2.2, 3.4, 4.1, -3.0, 2.0]
    assert all(row["failure"] for row in rows if row["value"] is None)


def test_isd_tenths_celsius_quality_and_missing_sentinels(tmp_path: Path) -> None:
    path = tmp_path / "station.csv"
    path.write_text(
        'STATION,DATE,LATITUDE,LONGITUDE,TMP\n00701800001,2020-01-01T00:00:00,41.96,-87.93,"-0123,1"\n00701800001,2020-01-01T01:00:00,41.96,-87.93,"+0010,5"\n00701800001,2020-01-01T02:00:00,41.96,-87.93,"+9999,1"\n00701800001,2020-01-01T03:00:00,41.96,-87.93,"+0200,9"\n'
    )
    rows = list(isd_rows(path, "00701800001", 2020))
    assert [row["value"] for row in rows] == [-12.3, 1.0, None, None]
    assert all(row["station_id"] == "00701800001" for row in rows)


def test_station_catalogue_keeps_native_zero_padded_ids_and_unknown_locations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.csv"
    path.write_text(
        "USAF,WBAN,STATION NAME,LAT,LON,BEGIN,END\n"
        "007018,00001,Example,41.96,-87.93,19900101,20250824\n"
        "007019,00002,Missing,,,19900101,20250824\n"
    )
    rows = _catalogue_csv("isd", path)
    assert [row.id for row in rows] == ["00701800001", "00701900002"]
    assert rows[0].period is not None and rows[0].period.start == 1990
    assert rows[1].longitude is None and rows[1].latitude is None


def test_missing_native_columns_and_unrequested_station_fail_clearly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "station.parquet"
    pq.write_table(pa.table({"STATION": ["TEST0000001"]}), path)
    with pytest.raises(ValueError, match="required source columns"):
        list(ghcnh_rows(path, "TEST0000001", 2020))
    csv = tmp_path / "station.csv"
    csv.write_text(
        'STATION,DATE,LATITUDE,LONGITUDE,TMP\nOTHER,2020-01-01T00:00:00,41.96,-87.93,"+0100,1"\n'
    )
    with pytest.raises(ValueError, match="unrequested station"):
        list(isd_rows(csv, "00701800001", 2020))
