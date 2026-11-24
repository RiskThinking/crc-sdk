from __future__ import annotations

import math

import pytest

from crc_sdk._providers import eccc_idf as provider

PERIODS = [2, 5, 10, 25, 50, 100]
STATION = "6158731"


def station_text(
    station: str = STATION, start: int = 1950, missing: tuple[str, ...] = ()
) -> bytes:
    """A station file; durations in `missing` carry ECCC's -99.9 "no estimate" rows."""
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
        if duration in missing:
            lines.append(f"{label} " + " ".join(["-99.9"] * len(PERIODS)) + " 9")
            lines.append(" ".join(["+/--99.9"] * len(PERIODS)) + " 9")
            continue
        lines.append(f"{label} " + " ".join(map(str, rates)) + " 40")
        lines.append(" ".join(["+/- 0.5"] * len(PERIODS)) + " 40")
    lines.append("Table 3 : Interpolation Equation")
    return "\r\n".join(lines).encode("cp1252")


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


def test_durations_eccc_did_not_estimate_are_kept_as_missing() -> None:
    record = provider.parse_station(station_text(missing=("5min",)), STATION, "ON")
    first, *rest = record["curves"]
    assert (first["duration_minutes"], first["intensity"]) == (5, None)
    assert first["confidence_half_width"] is None
    assert all(curve["intensity"] is not None for curve in rest)
    # Only a complete row of sentinels means "not estimated".
    raw = station_text().decode("cp1252")
    label = "5 min "
    start = raw.index(label) + len(label)
    partial = raw[:start] + "-99.9 " + raw[start:].split(" ", 1)[1]
    with pytest.raises(ValueError, match="rainfall rates"):
        provider.parse_station(partial.encode("cp1252"), STATION, "ON")
