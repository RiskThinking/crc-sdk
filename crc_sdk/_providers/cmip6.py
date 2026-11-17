"""Source identities, daily variable definitions, and native CF calendar math."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

CATALOGUE_URL = "https://cmip6-pds.s3.us-west-2.amazonaws.com/pangeo-cmip6.csv"
CATALOGUE_COLUMNS = (
    "activity_id",
    "institution_id",
    "source_id",
    "experiment_id",
    "member_id",
    "table_id",
    "variable_id",
    "grid_label",
    "zstore",
    "version",
)
GRID_PREFERENCE = ("gn", "gr", "gr1", "gr2")
_FIXED_MONTH_LENGTHS: dict[str, tuple[int, ...]] = {
    "noleap": (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    "all_leap": (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    "360_day": (30,) * 12,
}


@dataclass(frozen=True)
class CFCalendar:
    """One CF calendar, as a bijection between dates and day ordinals.

    Fixed-length model calendars do not pass through Gregorian datetime values.
    """

    name: str

    @property
    def fixed_months(self) -> tuple[int, ...] | None:
        return _FIXED_MONTH_LENGTHS.get(self.name)

    def to_ordinal(self, year: int, month: int, day: int) -> int:
        """Day number of one date; the zero point is arbitrary but fixed."""
        months = self.fixed_months
        if months is None:
            stamp = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "D")
            return int(stamp.astype("int64"))
        starts = np.cumsum((0,) + months[:-1])
        return int(year * sum(months) + starts[month - 1] + day - 1)

    def from_ordinal(
        self, ordinal: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Vectorised inverse of :meth:`to_ordinal`, as ``(year, month, day)``."""
        ordinal = np.asarray(ordinal, dtype=np.int64)
        months = self.fixed_months
        if months is None:
            stamps = ordinal.astype("datetime64[D]")
            month_index = stamps.astype("datetime64[M]")
            return (
                month_index.astype("datetime64[Y]").astype(np.int64) + 1970,
                month_index.astype(np.int64) % 12 + 1,
                (stamps - month_index).astype(np.int64) + 1,
            )
        starts = np.cumsum((0,) + months[:-1])
        year, remainder = np.divmod(ordinal, sum(months))
        month = np.searchsorted(starts, remainder, side="right")
        return year, month, remainder - starts[month - 1] + 1


def decode_time_ordinals(
    values: np.ndarray[Any, Any], units: str, calendar: CFCalendar
) -> np.ndarray[Any, Any]:
    """CF ``<unit> since <date>`` time values -> whole-day ordinals.

    CMIP6 daily stamps are usually the *centre* of their day (``0.5``, ``1.5``,
    ...), so the elapsed time is floored: a stamp anywhere inside a day belongs
    to that day.
    """
    match = re.match(
        r"\s*(days|hours|minutes|seconds)\s+since\s+"
        r"(\d{1,4})-(\d{1,2})-(\d{1,2})",
        str(units),
    )
    if match is None:
        raise ValueError(f"unsupported CMIP6 time units {units!r}")
    unit, year, month, day = match.groups()
    per_day = {"days": 1.0, "hours": 24.0, "minutes": 1440.0, "seconds": 86400.0}[unit]
    elapsed = np.floor(np.asarray(values, dtype=np.float64) / per_day + 1e-9)
    origin = calendar.to_ordinal(int(year), int(month), int(day))
    return origin + elapsed.astype(np.int64)


@dataclass(frozen=True)
class CMIP6Dataset:
    """One published (model, experiment, member, variable, grid, version)."""

    activity_id: str
    institution_id: str
    source_id: str
    experiment_id: str
    member_id: str
    table_id: str
    variable_id: str
    grid_label: str
    zstore: str
    version: str

    @property
    def dataset_id(self) -> str:
        """The CMIP6 DRS identity of this dataset, as one dotted string."""
        return ".".join(
            (
                "CMIP6",
                self.activity_id,
                self.institution_id,
                self.source_id,
                self.experiment_id,
                self.member_id,
                self.table_id,
                self.variable_id,
                self.grid_label,
                f"v{self.version}",
            )
        )


@dataclass(frozen=True, slots=True)
class NativeVariable:
    variable: str
    source_unit: str


INDICATOR_VARIABLES = {
    "txx": NativeVariable("tasmax", "K"),
    "tnn": NativeVariable("tasmin", "K"),
    "rx1day": NativeVariable("pr", "kg m-2 s-1"),
    "rx5day": NativeVariable("pr", "kg m-2 s-1"),
}
