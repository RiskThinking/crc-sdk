from __future__ import annotations

import hashlib
import io
import math
import zipfile
from typing import Any

import pytest

from crc_sdk._providers import eccc_idf as provider

PERIODS = [2, 5, 10, 25, 50, 100]
STATION = "6158731"


def station_text(station: str = STATION, start: int = 1950) -> bytes:
    lines = [
        f"SYNTHETIC A                       ON        {station}",
        "Latitude:  43 41'N    Longitude: 79 38'W    Elevation/Altitude: 173 m",
        f"Years/Années : {start} - 2017   # Years/Années : 40",
        "Table 1 : Annual Maximum (mm)",
        "1950 -99.9 -99.9 -99.9",
        "Table 2a : Return Period Rainfall Amounts (mm)",
        "1 h 9999 9999 9999 9999 9999 9999 40",
        "Table 2b :",
        "Return Period Rainfall Rates (mm/h) - 95% Confidence limits",
        "Duration/Durée        2        5       10       25       50      100   #Years",
        "yr/ans yr/ans yr/ans yr/ans yr/ans yr/ans Années",
    ]
    for duration, minutes in provider.DURATIONS.items():
        rates = [
            round((20 - 5 * math.log(-math.log(1 - 1 / t))) * 60 / minutes, 1)
            for t in PERIODS
        ]
        label = duration.replace("min", " min").replace("h", " h")
        lines.append(f"{label} " + " ".join(map(str, rates)) + " 40")
        lines.append(" ".join(["+/- 0.5"] * len(PERIODS)) + " 40")
    lines.append("Table 3 : Interpolation Equation")
    return "\r\n".join(lines).encode("cp1252")


def fake_fetch(version: str, province: str, stations: Any) -> Any:
    for station in stations or [STATION, "6158355"]:
        raw = station_text(station, 1950 if station == STATION else 1960)
        yield (
            station,
            raw,
            {
                "url": provider.archive_url(version, province),
                "member": f"_{province}_{station}_SYNTHETIC.txt",
                "checksum": hashlib.sha256(raw).hexdigest(),
            },
        )


def test_parser_labels_rates_confidence_and_record() -> None:
    record = provider.parse_station(station_text(), STATION, "ON")
    assert record["return_periods"] == PERIODS
    assert record["latitude"] == pytest.approx(43 + 41 / 60)
    assert record["longitude"] == pytest.approx(-79 - 38 / 60)
    assert record["curves"][4]["intensity"][0] < 30
    assert record["curves"][4]["confidence_half_width"] == [0.5] * 6
    assert record["curves"][4]["record_years"] == 40
    assert record["curves"][-1]["duration_minutes"] == 1440


@pytest.mark.parametrize(
    "old,new",
    [
        ("      100   #Years", "      50   #Years"),
        ("Table 2b :", "Missing table"),
        ("1 h ", "3 h "),
        ("+/- 0.5", "+/- -0.5"),
        ("43 41'N", "43 99'N"),
        (" 40\r\n", " 400\r\n"),
    ],
)
def test_malformed_scientific_input_rejected(old: str, new: str) -> None:
    raw = station_text().decode("cp1252").replace(old, new).encode("cp1252")
    with pytest.raises(ValueError):
        provider.parse_station(raw, STATION, "ON")


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return buffer.getvalue()


@pytest.mark.parametrize("version", ["3.20", "3.40"])
def test_real_zip_range_access_without_network(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    raw = station_text()
    inner = zip_bytes(
        {f"province/_ON_{STATION}_SYNTHETIC.txt": raw, "ignored.png": b"not text"}
    )
    data = (
        zip_bytes({"idf_v3-40_2025-12-5/IDF_Files_Fichiers/ON.zip": inner})
        if version == "3.40"
        else inner
    )
    ranges: list[tuple[int, int]] = []

    class Response(io.BytesIO):
        def __init__(
            self, payload: bytes, headers: dict[str, str], status: int
        ) -> None:
            super().__init__(payload)
            self.headers = headers
            self.status = status

    def urlopen(request: Any, **kwargs: Any) -> Any:
        if request.get_method() == "HEAD":
            return Response(
                b"", {"Content-Length": str(len(data)), "ETag": '"test"'}, 200
            )
        assert request.get_header("If-match") == '"test"'
        start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
        ranges.append((start, end))
        return Response(
            data[start : end + 1],
            {"Content-Range": f"bytes {start}-{end}/{len(data)}"},
            206,
        )

    monkeypatch.setattr(provider, "urlopen", urlopen)
    fetched = list(provider.fetch_stations(version, "ON", [STATION]))
    assert len(fetched) == 1 and fetched[0][1] == raw
    assert fetched[0][2]["checksum"] == hashlib.sha256(raw).hexdigest()
    assert len(ranges) > 1
    with pytest.raises(ValueError, match="no stations"):
        list(provider.fetch_stations(version, "ON", ["9999999"]))


def test_range_reader_reads_ahead_instead_of_one_request_per_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = {
        f"station_{i:03d}.txt": (f"{i},1.0,2.0\n" * 100).encode() for i in range(100)
    }
    data = zip_bytes(members)
    requests: list[str] = []

    class Response(io.BytesIO):
        def __init__(
            self, payload: bytes, headers: dict[str, str], status: int
        ) -> None:
            super().__init__(payload)
            self.headers = headers
            self.status = status

    def urlopen(request: Any, **kwargs: Any) -> Any:
        requests.append(request.get_method())
        if request.get_method() == "HEAD":
            return Response(b"", {"Content-Length": str(len(data))}, 200)
        start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
        return Response(
            data[start : end + 1],
            {"Content-Range": f"bytes {start}-{end}/{len(data)}"},
            206,
        )

    monkeypatch.setattr(provider, "urlopen", urlopen)
    reader = provider._RangeReader("https://example.test/archive.zip")
    with zipfile.ZipFile(reader) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == members
    # Previously ~3 requests per member (header, name, data).
    assert len(requests) < 10
