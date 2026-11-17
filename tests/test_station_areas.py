from __future__ import annotations

import hashlib
import io
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk._providers.station_catalogues import fetch_catalogue

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
        "crc_sdk._providers.station_catalogues.urlopen",
        lambda *args, **kwargs: Response(raw),
    )
    table, meta = fetch_catalogue(provider, "https://example.test", pattern)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["station"] == station
    assert row["elevation"] is None
    assert row["start_year"] == (1990 if provider == "isd" else None)
    assert meta["raw_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("sentinel", ["-999.0", "-0999.9", "+9999.0"])
def test_isd_history_elevation_sentinels_are_missing(
    sentinel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # isd-history.csv marks unknown elevation -999.0/-999.9 (hundreds of
    # stations), not only 9999.
    raw = (
        b"USAF,WBAN,STATION NAME,LAT,LON,ELEV(M),BEGIN,END\n"
        b"007018,00001,Unknown,41.96,-87.93," + sentinel.encode() + b",1990,2025\n"
        b"007018,00002,Dead Sea,31.5,35.5,-0415.0,19900101,20250824\n"
    )

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(raw))}

    monkeypatch.setattr(
        "crc_sdk._providers.station_catalogues.urlopen",
        lambda *args, **kwargs: Response(raw),
    )
    table, _ = fetch_catalogue("isd", "https://example.test", r"[0-9]{11}")
    assert table.column("elevation").to_pylist() == [None, -415.0]
