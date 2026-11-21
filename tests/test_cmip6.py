"""Native CF calendar and daily coordinate conformance."""

from __future__ import annotations

import numpy as np
import pytest

from crc_sdk._climate_grid import calendar_name, cf_calendar
from crc_sdk._providers.cmip6 import decode_time_ordinals


@pytest.mark.parametrize(
    ("label", "name", "year_days", "leap_days"),
    [
        ("standard", "proleptic_gregorian", 365, 366),
        ("gregorian", "proleptic_gregorian", 365, 366),
        ("noleap", "noleap", 365, 365),
        ("365_day", "noleap", 365, 365),
        ("360_day", "360_day", 360, 360),
        ("all_leap", "all_leap", 366, 366),
    ],
)
def test_calendars_count_their_own_years(
    label: str, name: str, year_days: int, leap_days: int
) -> None:
    calendar = cf_calendar(calendar_name(label))
    assert calendar.name == name
    start, stop = calendar.to_ordinal(2015, 1, 1), calendar.to_ordinal(2016, 1, 1)
    leap_start, leap_stop = (
        calendar.to_ordinal(2016, 1, 1),
        calendar.to_ordinal(2017, 1, 1),
    )
    assert (stop - start, leap_stop - leap_start) == (year_days, leap_days)


@pytest.mark.parametrize("label", ["standard", "noleap", "360_day", "all_leap"])
def test_calendar_ordinals_round_trip(label: str) -> None:
    calendar = cf_calendar(calendar_name(label))
    dates = [(2015, 1, 1), (2015, 2, 28), (2049, 7, 15), (2100, 12, 30)]
    ordinals = np.array([calendar.to_ordinal(*date) for date in dates])
    years, months, days = calendar.from_ordinal(ordinals)
    assert [tuple(int(v) for v in row) for row in zip(years, months, days)] == dates


def test_calendar_rejects_what_it_cannot_count() -> None:
    with pytest.raises(ValueError, match="Julian calendar"):
        cf_calendar(calendar_name("julian"))


def test_decoding_floors_day_centred_stamps_in_the_model_calendar() -> None:
    # A 360-day model's day 360 is 1 January, five days before the real one.
    calendar = cf_calendar(calendar_name("360_day"))
    ordinals = decode_time_ordinals(
        np.arange(0.5, 720.0, 1.0), "days since 2015-01-01", calendar
    )
    years, months, days = calendar.from_ordinal(ordinals)
    assert (int(years[0]), int(months[0]), int(days[0])) == (2015, 1, 1)
    assert (int(years[360]), int(months[360]), int(days[360])) == (2016, 1, 1)
    assert int((months[:360] == 2).sum()) == 30  # February has thirty days


def test_decoding_accepts_sub_daily_units() -> None:
    calendar = cf_calendar(calendar_name("noleap"))
    ordinals = decode_time_ordinals(
        np.array([12.0, 36.0]), "hours since 2015-01-01 00:00:00", calendar
    )
    _, _, days = calendar.from_ordinal(ordinals)
    assert days.tolist() == [1, 2]
    with pytest.raises(ValueError, match="unsupported CMIP6 time units"):
        decode_time_ordinals(np.array([0.0]), "fortnights since 2015", calendar)
