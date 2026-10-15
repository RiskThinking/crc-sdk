"""CMIP6 daily model output access for annual-extreme recipes.

Reads the public Zarr copies of CMIP6 published to the AWS Open Data
registry (``s3://cmip6-pds``, anonymous) -- the Pangeo mirror, the same one
the catalogue CSV indexes -- reduces one model year at a time to an *annual
extreme per grid cell* for the requested area, and caches that small result,
never the daily fields.

This is the projection counterpart to `crc_sdk.providers.era5`: the same four
recipes (TXx, TNn, Rx1day, Rx5day), the same block-extrema engine, the same
cached per-year NetCDF, so an ERA5 historical baseline and a CMIP6 scenario
run produce canonical files that join on ``(hazard_name, horizon, pathway,
cell_index)`` and can be differenced. Three things differ, and all three are
why a CMIP6 recipe cannot simply be an ERA5 store:

* **The dataset must be found.** CMIP6 is a catalogue, not a store: one Zarr
  per (model, experiment, member, variable, grid, version). `CMIP6Catalogue`
  resolves a selection to exactly one of them and records which.
* **Calendars are model calendars.** CMIP6 models run ``noleap``, ``360_day``
  or ``all_leap`` years as readily as proleptic Gregorian ones, so time is
  decoded into the model's own calendar (`CFCalendar`) and blocks are cut on
  calendar day ordinals. Treating a ``360_day`` axis as real dates drifts by
  five days a year and silently mislabels every block.
* **A pathway is not "historical".** Each dataset carries its ``experiment_id``
  (``historical``, ``ssp245``, ``ssp585``, ...) into the canonical rows as the
  pathway label, resolved through `crc_sdk.pathways` (ADR-0002).

Caveats worth carrying into any use of the results, beyond ERA5's: a climate
model is a model world, so these are *change* signals rather than design
values; one member of one model is one realisation, not an ensemble
(ADR-0001 -- the spread between models is itself the signal); and a ~1 degree
grid cell smooths terrain, so orographic and convective extremes are damped.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from crc_framework.distributions import DistributionFamily

from crc_sdk.connectors.blocks import (
    STRIP_BYTES,
    BlockDefinition,
    BlockExtremaCurveSource,
    BlockSpec,
    BlockStatistic,
    DailyAggregation,
    UnitConversion,
    _strip_row_count,
    contiguous_runs,
    reduce_days,
    select_grid_window,
)
from crc_sdk.connectors.duckdb.netcdf import NetCDFBlockReader, NetCDFRaster
from crc_sdk.connectors.duckdb.zarr import Bounds, RasterMetadata

CMIP6_ATTRIBUTION = (
    "CMIP6 model output prepared for the World Climate Research Programme's "
    "Coupled Model Intercomparison Project Phase 6"
)
#: What `source.licence` records when a store's licence text is not one of the
#: labels below. "CC-BY-4.0" is the common case but it is *not* universal --
#: NOAA-GFDL, for one, publishes share-alike -- so an unrecognised licence is
#: recorded as unknown rather than assumed open (see docs/licence-policy.md).
CMIP6_LICENCE_UNKNOWN = "unknown"
#: Licence texts CMIP6 stores actually carry, longest-obligation first: the
#: share-alike and non-commercial labels contain the plain-attribution wording
#: as a substring, so they have to be tested before it.
_LICENCE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"attribution[-\s]*share[-\s]*alike|\bcc[-\s]?by[-\s]?sa\b", "CC-BY-SA-4.0"),
    (r"attribution[-\s]*non[-\s]*commercial|\bcc[-\s]?by[-\s]?nc\b", "CC-BY-NC-4.0"),
    (r"creative\s+commons\s+zero|public\s+domain\s+dedication|\bcc0\b", "CC0-1.0"),
    (r"creative\s+commons\s+attribution|\bcc[-\s]?by\b", "CC-BY-4.0"),
)
#: Licences whose terms go beyond crediting the source, so a derived canonical
#: file inherits an obligation the licence policy wants surfaced, not buried.
RESTRICTIVE_LICENCES = frozenset({"CC-BY-SA-4.0", "CC-BY-NC-4.0"})


class RestrictiveLicenceWarning(UserWarning):
    """A source licence carries terms beyond attribution."""


def enforce_licence(identifier: str, *, dataset: str, strict: bool = False) -> str:
    """Surface a licence whose terms go beyond attribution.

    The obligation belongs to the *output*, not to the act of reading the
    store, so this is applied wherever a canonical dataset is produced or
    handed back -- including from a warm cache or an already-materialized
    file, which never reopen the store. ``strict=True`` refuses instead of
    warning (docs/licence-policy.md).
    """
    if identifier in RESTRICTIVE_LICENCES:
        message = (
            f"{dataset} is published under {identifier}, whose terms go beyond "
            "attribution; canonical files derived from it inherit them "
            "(see docs/licence-policy.md)"
        )
    elif identifier == CMIP6_LICENCE_UNKNOWN:
        message = (
            f"{dataset} declares a licence this adapter does not recognise; it "
            f"is recorded as {CMIP6_LICENCE_UNKNOWN!r}, and the terms should be "
            "read before anything derived from it is published"
        )
    else:
        return identifier
    if strict:
        raise ValueError(message)
    warnings.warn(message, RestrictiveLicenceWarning, stacklevel=3)
    return identifier


def normalize_licence(text: str) -> str:
    """One licence identifier for a store's free-text licence attribute."""
    for pattern, identifier in _LICENCE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return identifier
    return CMIP6_LICENCE_UNKNOWN


EXTREME_VARIABLE = "extreme"
#: Longest block any supported `BlockDefinition` can span, in days.
MAX_BLOCK_DAYS = 366


# ---------------------------------------------------------------------------
# CF calendars
# ---------------------------------------------------------------------------

_FIXED_MONTH_LENGTHS: dict[str, tuple[int, ...]] = {
    "noleap": (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    "all_leap": (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    "360_day": (30,) * 12,
}
_CALENDAR_ALIASES = {
    "standard": "proleptic_gregorian",
    "gregorian": "proleptic_gregorian",
    "proleptic_gregorian": "proleptic_gregorian",
    "noleap": "noleap",
    "365_day": "noleap",
    "all_leap": "all_leap",
    "366_day": "all_leap",
    "360_day": "360_day",
}
#: `TemporalWindow.calendar` has no name for `all_leap`; it is recorded as
#: unspecified rather than quietly reported as one of the others.
_WINDOW_CALENDARS: dict[str, str] = {
    "proleptic_gregorian": "gregorian",
    "noleap": "noleap",
    "360_day": "360_day",
    "all_leap": "unspecified",
}


@dataclass(frozen=True)
class CFCalendar:
    """One CF calendar, as a bijection between dates and day ordinals.

    Every calendar question a block needs -- which days a year contains, where
    a day sits inside its block, how long a block is -- is answered by
    counting ordinals, so the fixed-length calendars never touch
    ``datetime64`` and never drift against it.
    """

    name: str

    @classmethod
    def parse(cls, label: str | None) -> CFCalendar:
        key = (label or "standard").strip().lower()
        try:
            return cls(_CALENDAR_ALIASES[key])
        except KeyError as error:
            raise ValueError(
                f"unsupported CMIP6 calendar {label!r}; supported: "
                f"{sorted(set(_CALENDAR_ALIASES))}"
            ) from error

    @property
    def fixed_months(self) -> tuple[int, ...] | None:
        return _FIXED_MONTH_LENGTHS.get(self.name)

    @property
    def window_calendar(
        self,
    ) -> Literal["gregorian", "noleap", "360_day", "unspecified"]:
        """How `TemporalWindow` should record this calendar."""
        value = _WINDOW_CALENDARS[self.name]
        return value  # type: ignore[return-value]

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

    def block_ordinals(self, block: BlockDefinition, label: int) -> tuple[int, int]:
        """Half-open ``[start, stop)`` ordinal range of one block.

        `BlockDefinition.span` already settles which months a labelled block
        covers; it returns first-of-month dates, which exist in every calendar,
        so only the year and month are read back out of them.
        """

        def year_month(stamp: np.datetime64) -> tuple[int, int]:
            month_index = stamp.astype("datetime64[M]")
            year = int(month_index.astype("datetime64[Y]").astype(np.int64)) + 1970
            return year, int(month_index.astype(np.int64)) % 12 + 1

        start, stop = block.span(label)
        return (
            self.to_ordinal(*year_month(start), 1),
            self.to_ordinal(*year_month(stop), 1),
        )


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


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

#: Columns of the Pangeo CMIP6 catalogue, in its own order.
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
#: Preference order when a selection leaves several grids: the model's native
#: grid first, then the regridded copies.
GRID_PREFERENCE = ("gn", "gr", "gr1", "gr2")


@dataclass(frozen=True)
class CMIP6Archive:
    """One published copy of CMIP6 and the catalogue that indexes it."""

    name: str
    catalogue_url: str
    storage_options: Mapping[str, Any] = field(default_factory=lambda: {"anon": True})
    zarr_format: int = 2
    lat_name: str = "lat"
    lon_name: str = "lon"
    time_name: str = "time"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.catalogue_url:
            raise ValueError("name and catalogue_url must be non-empty")


AWS_OPEN_DATA = CMIP6Archive(
    name="aws",
    catalogue_url="https://cmip6-pds.s3.us-west-2.amazonaws.com/pangeo-cmip6.csv",
    description=(
        "CMIP6 on the AWS Open Data registry (s3://cmip6-pds), Pangeo Zarr "
        "mirror, anonymous"
    ),
)

CMIP6_ARCHIVES = {archive.name: archive for archive in (AWS_OPEN_DATA,)}


def cmip6_archive(name: str) -> CMIP6Archive:
    try:
        return CMIP6_ARCHIVES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown CMIP6 archive {name!r}; choose from {tuple(CMIP6_ARCHIVES)}"
        ) from error


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

    def describe(self) -> str:
        return (
            f"{self.source_id} {self.experiment_id} {self.member_id} "
            f"{self.table_id} {self.variable_id} ({self.grid_label}, "
            f"v{self.version})"
        )


class CMIP6Catalogue:
    """Query the Pangeo CMIP6 catalogue for the one dataset a plan needs.

    The published catalogue is an 80 MB CSV, so it is read with DuckDB (a core
    dependency) rather than pulled into memory: a search streams it and keeps
    only matching rows. :meth:`ensure_local` caches the *daily* subset as a
    few-megabyte Parquet file, after which searches are local and offline.
    """

    def __init__(
        self,
        archive: CMIP6Archive = AWS_OPEN_DATA,
        *,
        local: str | Path | None = None,
    ) -> None:
        self.archive = archive
        self.local = Path(local) if local is not None else None

    @property
    def source(self) -> str:
        """Whatever a query should read: the local subset, else the CSV."""
        if self.local is not None and self.local.is_file():
            return str(self.local)
        return self.archive.catalogue_url

    def _scan_sql(self) -> str:
        from crc_sdk.connectors.duckdb import sql_quote

        source = self.source
        if source.endswith(".parquet"):
            return f"read_parquet({sql_quote(source)})"
        columns = ", ".join(f"'{name}': 'VARCHAR'" for name in CATALOGUE_COLUMNS)
        # dcpp_init_year is the one numeric column and is never selected here;
        # typing the rest as VARCHAR keeps DRS version strings ("20180701")
        # from being read as integers and losing any leading zero.
        return (
            f"read_csv({sql_quote(source)}, header=true, "
            f"types={{{columns}}}, union_by_name=true)"
        )

    def _connect(self) -> Any:
        from crc_sdk.connectors.duckdb import DuckDBConnection

        remote = not self.source.startswith(("/", ".")) and "://" in self.source
        return DuckDBConnection(extensions=("httpfs",) if remote else ()).connect()

    def ensure_local(self, destination: str | Path, *, table_id: str = "day") -> Path:
        """Cache the catalogue's ``table_id`` rows locally, once.

        The daily subset is a small fraction of the published rows, so the
        whole CSV is read exactly once per cache directory and every later
        search is a local Parquet scan.
        """
        from crc_sdk.connectors.duckdb import sql_quote

        target = Path(destination)
        if target.is_file():
            self.local = target
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.parquet")
        columns = ", ".join(CATALOGUE_COLUMNS)
        con = self._connect()
        try:
            con.execute(
                f"COPY (SELECT {columns} FROM {self._scan_sql()} "
                f"WHERE table_id = {sql_quote(table_id)}) "
                f"TO {sql_quote(str(temporary))} (FORMAT PARQUET)"
            )
        finally:
            con.close()
        temporary.replace(target)
        self.local = target
        return target

    def search(
        self,
        *,
        model: str | None = None,
        scenario: str | None = None,
        member: str | None = None,
        variable: str | None = None,
        grid_label: str | None = None,
        version: str | None = None,
        table_id: str = "day",
        limit: int = 1000,
    ) -> tuple[CMIP6Dataset, ...]:
        """Every published dataset matching the given fields."""
        from crc_sdk.connectors.duckdb import sql_quote

        filters = {
            "source_id": model,
            "experiment_id": scenario,
            "member_id": member,
            "variable_id": variable,
            "grid_label": grid_label,
            "table_id": table_id,
        }
        if version not in (None, "latest"):
            filters["version"] = version
        where = " AND ".join(
            f"{column} = {sql_quote(value)}"
            for column, value in filters.items()
            if value is not None
        )
        columns = ", ".join(CATALOGUE_COLUMNS)
        statement = (
            f"SELECT {columns} FROM {self._scan_sql()}"
            f"{' WHERE ' + where if where else ''} "
            f"ORDER BY source_id, experiment_id, member_id, grid_label, version "
            f"LIMIT {int(limit)}"
        )
        con = self._connect()
        try:
            rows = con.execute(statement).fetchall()
        finally:
            con.close()
        return tuple(CMIP6Dataset(*(str(value) for value in row)) for row in rows)

    def resolve(
        self,
        *,
        model: str,
        scenario: str,
        member: str,
        variable: str,
        grid_label: str | None = None,
        version: str = "latest",
        table_id: str = "day",
    ) -> CMIP6Dataset:
        """The single dataset a plan names, or a message saying why not.

        With several grids published, the model's native grid wins; with
        several versions, ``latest`` takes the newest. Anything still
        ambiguous is an error naming the candidates rather than a silent pick.
        """
        matches = self.search(
            model=model,
            scenario=scenario,
            member=member,
            variable=variable,
            grid_label=grid_label,
            version=version,
            table_id=table_id,
        )
        if not matches:
            raise LookupError(
                f"no CMIP6 {table_id} {variable} published for "
                f"{model} / {scenario} / {member}"
                f"{f' / {grid_label}' if grid_label else ''}"
                f"{'' if version == 'latest' else f' / v{version}'}; "
                f"catalogue: {self.source}"
            )
        if grid_label is None:
            grids = {dataset.grid_label for dataset in matches}
            preferred = next(
                (grid for grid in GRID_PREFERENCE if grid in grids),
                sorted(grids)[0],
            )
            matches = tuple(
                dataset for dataset in matches if dataset.grid_label == preferred
            )
        if version == "latest":
            newest = max(dataset.version for dataset in matches)
            matches = tuple(dataset for dataset in matches if dataset.version == newest)
        if len(matches) > 1:
            listed = ", ".join(sorted(dataset.describe() for dataset in matches))
            raise LookupError(
                f"CMIP6 selection is ambiguous between {len(matches)} datasets: "
                f"{listed}; name a grid_label or version"
            )
        return matches[0]


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CMIP6Recipe:
    """One supported dataset -> hazard recipe (ADR-0004 names).

    The ERA5 recipe of the same name computes the same statistic, so the two
    are comparable -- with the caveat that a model's day boundary is its own
    calendar's, not UTC clock time.
    """

    name: str
    variable: str  # CMIP6 variable_id
    unit: str
    source_unit: str  # the CF unit the store must declare
    value_semantics: str
    spec: BlockSpec
    default_family: DistributionFamily = "genextreme"
    default_fit_method: Literal[
        "quantile_least_squares", "sample_mle", "sample_lmoments"
    ] = "sample_lmoments"

    @property
    def tail(self) -> Literal["upper", "lower"]:
        return self.spec.tail


# CMIP6 `day` output is already one sample per day, so the "daily aggregation"
# is the identity: it records the source cadence (and lets a rolling k-day
# statistic validate) rather than reducing anything.
_ALREADY_DAILY = DailyAggregation("max", step_hours=24.0, min_valid_fraction=1.0)
_KELVIN_TO_CELSIUS = UnitConversion(offset=-273.15)
# `pr` is a flux in kg m-2 s-1, numerically mm/s over water: a day's depth in
# millimetres is the flux times the seconds in a day.
_FLUX_TO_MM_PER_DAY = UnitConversion(scale=86400.0)

CMIP6_RECIPES: dict[str, CMIP6Recipe] = {
    recipe.name: recipe
    for recipe in (
        CMIP6Recipe(
            "txx",
            "tasmax",
            "degC",
            "K",
            "annual maximum of daily maximum near-surface air temperature",
            BlockSpec(
                statistic=BlockStatistic.maximum(),
                daily=_ALREADY_DAILY,
                conversion=_KELVIN_TO_CELSIUS,
                min_valid_fraction=0.9,
            ),
        ),
        CMIP6Recipe(
            "tnn",
            "tasmin",
            "degC",
            "K",
            "annual minimum of daily minimum near-surface air temperature",
            BlockSpec(
                statistic=BlockStatistic.minimum(),
                daily=_ALREADY_DAILY,
                conversion=_KELVIN_TO_CELSIUS,
                min_valid_fraction=0.9,
            ),
        ),
        CMIP6Recipe(
            "rx1day",
            "pr",
            "mm/day",
            "kg m-2 s-1",
            "annual maximum one-day precipitation total",
            BlockSpec(
                statistic=BlockStatistic.maximum(),
                daily=_ALREADY_DAILY,
                conversion=_FLUX_TO_MM_PER_DAY,
                min_valid_fraction=0.9,
            ),
        ),
        CMIP6Recipe(
            "rx5day",
            "pr",
            "mm",
            "kg m-2 s-1",
            "annual maximum five-day precipitation total",
            BlockSpec(
                statistic=BlockStatistic.k_day_sum(5),
                daily=_ALREADY_DAILY,
                conversion=_FLUX_TO_MM_PER_DAY,
                min_valid_fraction=0.9,
            ),
        ),
    )
}


def cmip6_recipe(name: str) -> CMIP6Recipe:
    try:
        return CMIP6_RECIPES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown CMIP6 recipe {name!r}; choose from {tuple(CMIP6_RECIPES)}"
        ) from error


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _open_group(dataset: CMIP6Dataset, archive: CMIP6Archive) -> Any:
    try:
        import zarr
    except ImportError as error:
        raise ImportError(
            "CMIP6 access requires `pip install crc-sdk[zarr]`"
        ) from error
    options = dict(archive.storage_options)
    try:
        return zarr.open_group(
            dataset.zstore,
            mode="r",
            storage_options=options or None,
            zarr_format=archive.zarr_format,  # type: ignore[arg-type]
        )
    except TypeError:  # zarr 2 has no zarr_format argument
        import fsspec  # type: ignore[import-untyped]

        return zarr.open_group(fsspec.get_mapper(dataset.zstore, **options), mode="r")


@dataclass(frozen=True)
class CMIP6Coverage:
    """The complete calendar years one published dataset actually contains."""

    first_year: int
    last_year: int
    calendar: str
    first_ordinal: int
    last_ordinal: int

    def describe(self) -> str:
        return f"{self.first_year}-{self.last_year} ({self.calendar} calendar)"


class CMIP6BlockReader:
    """`BlockReader` that reduces one CMIP6 model year straight from the store.

    One block is read in a single slab (a year of daily values for the strip,
    a few megabytes), placed on a dense per-day array by calendar ordinal so
    that gaps stay missing rather than shifting later days, and reduced by the
    recipe's statistic. Note that CMIP6 Zarr chunks usually span the full
    globe, so a small area still pays for whole chunks; the per-year cache
    exists precisely so that cost is paid once.
    """

    def __init__(
        self,
        provider: CMIP6Provider,
        recipe: CMIP6Recipe,
        bounds: Bounds,
        *,
        years: Sequence[int],
    ) -> None:
        self._provider = provider
        self.recipe = recipe
        self.spec = recipe.spec
        self._years = tuple(years)
        archive = provider.archive
        group = provider.group
        self._array = group[recipe.variable]
        dims = tuple(self._array.attrs.get("_ARRAY_DIMENSIONS", ()))
        wanted = (archive.time_name, archive.lat_name, archive.lon_name)
        if sorted(dims) != sorted(wanted):
            raise ValueError(
                f"{recipe.variable!r} has dimensions {dims!r}; expected "
                f"{wanted!r} in any order (a dataset with extra levels or a "
                "curvilinear grid is out of scope)"
            )
        self._ordinals = provider.ordinals
        self._calendar = provider.calendar
        window = select_grid_window(
            np.asarray(group[archive.lat_name][:], dtype=np.float64),
            np.asarray(group[archive.lon_name][:], dtype=np.float64),
            bounds,
            source="CMIP6",
        )
        self._rows = window.rows
        self._cols = window.columns
        self._lat: np.ndarray[Any, Any] = window.lat
        self._lon: np.ndarray[Any, Any] = window.lon
        self._fill = _fill_value(self._array)

    @property
    def lat(self) -> np.ndarray[Any, Any]:
        return self._lat

    @property
    def lon(self) -> np.ndarray[Any, Any]:
        return self._lon

    @property
    def labels(self) -> tuple[int, ...]:
        return self._years

    @property
    def leading_axis_samples(self) -> int:
        # Resident at once: the raw slab, the dense per-day array it is placed
        # into, and -- for a rolling statistic -- the materialised windows.
        copies = 3 if self.spec.statistic.window_days > 1 else 2
        return MAX_BLOCK_DAYS * copies

    @property
    def block_height(self) -> int:
        return 1

    def close(self) -> None:
        return None

    def _read(
        self, t0: int, t1: int, rows: np.ndarray[Any, Any], cols: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """``(time, row, column)`` float64 for a row run and column indices."""
        archive = self._provider.archive
        dims = list(self._array.attrs["_ARRAY_DIMENSIONS"])
        time_axis = dims.index(archive.time_name)
        lat_axis, lon_axis = dims.index(archive.lat_name), dims.index(archive.lon_name)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        order = [time_axis, lat_axis, lon_axis]
        pieces = []
        for c0, c1 in contiguous_runs(cols):
            key: list[Any] = [slice(None)] * len(dims)
            key[time_axis] = slice(t0, t1)
            key[lat_axis] = slice(r0, r1)
            key[lon_axis] = slice(c0, c1)
            block = np.asarray(self._array[tuple(key)], dtype=np.float64)
            pieces.append(np.transpose(block, order))
        return np.concatenate(pieces, axis=2) if len(pieces) > 1 else pieces[0]

    def reduce_block(
        self, label: int, rows: slice, columns: slice
    ) -> np.ndarray[Any, Any]:
        spec = self.spec
        start, stop = self._calendar.block_ordinals(spec.block, label)
        day_count = stop - start
        window_rows = self._rows[rows]
        window_cols = self._cols[columns]
        series = np.full(
            (day_count, window_rows.size, window_cols.size), np.nan, dtype=np.float64
        )
        t0, t1 = (
            int(index) for index in np.searchsorted(self._ordinals, [start, stop])
        )
        if t1 > t0:
            samples = self._read(t0, t1, window_rows, window_cols)
            if self._fill is not None:
                samples = np.where(samples == self._fill, np.nan, samples)
            # Place by ordinal, not by position: a gap in the record must stay
            # a gap rather than pulling every later day one slot earlier.
            series[self._ordinals[t0:t1] - start] = samples
        series = spec.conversion.apply(series)
        return reduce_days(series, spec.statistic, spec.min_valid_fraction)


def _fill_value(array: Any) -> float | None:
    for key in ("_FillValue", "missing_value"):
        value = array.attrs.get(key)
        if value is not None:
            return float(value)
    return None


class CMIP6Provider:
    """Open one published CMIP6 Zarr dataset and build per-year extreme files."""

    def __init__(
        self,
        dataset: CMIP6Dataset,
        *,
        archive: CMIP6Archive = AWS_OPEN_DATA,
        group: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.archive = archive
        self._group = group
        self._ordinals: np.ndarray[Any, Any] | None = None
        self._calendar: CFCalendar | None = None

    @property
    def group(self) -> Any:
        if self._group is None:
            self._group = _open_group(self.dataset, self.archive)
        return self._group

    @property
    def calendar(self) -> CFCalendar:
        if self._calendar is None:
            time = self.group[self.archive.time_name]
            self._calendar = CFCalendar.parse(time.attrs.get("calendar"))
        return self._calendar

    @property
    def ordinals(self) -> np.ndarray[Any, Any]:
        """The time axis as whole-day ordinals in the model's own calendar."""
        if self._ordinals is None:
            time = self.group[self.archive.time_name]
            self._ordinals = decode_time_ordinals(
                np.asarray(time[:]), str(time.attrs.get("units", "")), self.calendar
            )
        return self._ordinals

    @property
    def licence_text(self) -> str:
        """The store's licence attribute, verbatim (often a long paragraph)."""
        attributes = dict(self.group.attrs)
        for key in ("license", "licence"):
            value = attributes.get(key)
            if value:
                return str(value)
        return ""

    @property
    def licence(self) -> str:
        """The store's own licence, as one identifier."""
        text = self.licence_text
        return normalize_licence(text) if text else CMIP6_LICENCE_UNKNOWN

    def check_licence(self, *, strict: bool = False) -> str:
        """Read this store's licence and apply the policy to it.

        CMIP6 licences are per modelling centre, and a share-alike or
        non-commercial one attaches an obligation to every canonical file
        derived from it -- so it is reported rather than recorded silently.
        """
        return enforce_licence(
            self.licence, dataset=self.dataset.describe(), strict=strict
        )

    def attribution(self) -> str:
        attributes = dict(self.group.attrs)
        institution = attributes.get("institution") or self.dataset.institution_id
        return f"{institution}; {CMIP6_ATTRIBUTION}"

    def check_variable(self, recipe: CMIP6Recipe) -> None:
        """Fail before any data is read if the store is not what the recipe assumes."""
        if recipe.variable not in self.group:
            raise ValueError(
                f"dataset {self.dataset.describe()} has no variable "
                f"{recipe.variable!r} for recipe {recipe.name!r}"
            )
        declared = str(self.group[recipe.variable].attrs.get("units", "")).strip()
        if declared and declared != recipe.source_unit:
            raise ValueError(
                f"{recipe.variable!r} is published in {declared!r}, but recipe "
                f"{recipe.name!r} converts from {recipe.source_unit!r}; the "
                "conversion would be wrong"
            )

    def coverage(self) -> CMIP6Coverage:
        """The first and last *complete* calendar years in the record."""
        ordinals = self.ordinals
        if ordinals.size == 0:
            raise RuntimeError(f"{self.dataset.describe()} has an empty time axis")
        calendar = self.calendar
        years, _, _ = calendar.from_ordinal(ordinals[[0, -1]])
        first, last = int(years[0]), int(years[-1])
        if int(ordinals[0]) > calendar.to_ordinal(first, 1, 1):
            first += 1
        if int(ordinals[-1]) < calendar.to_ordinal(last + 1, 1, 1) - 1:
            last -= 1
        if first > last:
            raise RuntimeError(
                f"{self.dataset.describe()} contains no complete calendar year"
            )
        return CMIP6Coverage(
            first_year=first,
            last_year=last,
            calendar=calendar.name,
            first_ordinal=int(ordinals[0]),
            last_ordinal=int(ordinals[-1]),
        )

    def resolve_version(self, requested: str) -> str:
        """The DRS version this provider is reading.

        Unlike ERA5's growing store, a CMIP6 dataset version is immutable: the
        catalogue already chose it, and this only checks the plan agrees.
        """
        current = self.dataset.version
        if requested not in ("latest", current):
            raise ValueError(
                f"{self.dataset.describe()} is version {current!r}, "
                f"not the requested {requested!r}"
            )
        return current

    def check_years(self, years: Sequence[int]) -> None:
        coverage = self.coverage()
        outside = [
            year
            for year in years
            if not coverage.first_year <= year <= coverage.last_year
        ]
        if outside:
            raise ValueError(
                f"years {outside} are outside the complete record "
                f"{coverage.describe()} of {self.dataset.describe()}"
            )

    def reader(
        self, recipe: CMIP6Recipe, bounds: Bounds, years: Sequence[int]
    ) -> CMIP6BlockReader:
        self.check_variable(recipe)
        return CMIP6BlockReader(self, recipe, bounds, years=years)

    def cache_annual_extreme(
        self,
        recipe: CMIP6Recipe,
        year: int,
        bounds: Bounds,
        destination: str | Path,
        *,
        strip_bytes: int = STRIP_BYTES,
        attributes: Mapping[str, str] | None = None,
    ) -> Path:
        """Persist one year's area annual extreme as a compact NetCDF file."""
        try:
            import h5netcdf  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "CMIP6 caching requires `pip install crc-sdk[netcdf]`"
            ) from error
        self.check_years([year])
        reader = self.reader(recipe, bounds, [year])
        width = reader.lon.size
        strip_rows = _strip_row_count(
            strip_bytes, width, 8, 1, leading_axis=reader.leading_axis_samples
        )
        parts = [
            reader.reduce_block(
                year,
                slice(start, min(start + strip_rows, reader.lat.size)),
                slice(0, width),
            )
            for start in range(0, reader.lat.size, strip_rows)
        ]
        extreme = np.concatenate(parts, axis=0)

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.nc")
        with h5netcdf.File(temporary, "w") as output:
            output.dimensions = {"time": 1, "lat": reader.lat.size, "lon": width}
            output.create_variable("lat", ("lat",), dtype=np.float64, data=reader.lat)
            output.create_variable("lon", ("lon",), dtype=np.float64, data=reader.lon)
            variable = output.create_variable(
                EXTREME_VARIABLE,
                ("time", "lat", "lon"),
                dtype=np.float32,
                data=extreme[np.newaxis].astype(np.float32),
            )
            variable.attrs["units"] = recipe.unit
            variable.attrs["recipe"] = recipe.name
            variable.attrs["year"] = int(year)
            variable.attrs["dataset_id"] = self.dataset.dataset_id
            variable.attrs["calendar"] = self.calendar.name
            for key, value in (attributes or {}).items():
                variable.attrs[key] = value
        temporary.replace(target)
        return target

    def open_resources(
        self,
        recipe: CMIP6Recipe,
        resources: Mapping[int, str | Path],
        metadata: RasterMetadata,
        *,
        strip_bytes: int = STRIP_BYTES,
        minimum_blocks: int = 4,
        info: Any = None,
    ) -> BlockExtremaCurveSource:
        """Open cached annual-extreme files as one block-extrema curve source."""
        if not resources:
            raise ValueError("at least one year is required")
        rasters: dict[int, NetCDFRaster] = {}
        try:
            for year, path in resources.items():
                rasters[int(year)] = NetCDFRaster.open(path, variable=EXTREME_VARIABLE)
        except Exception:
            for raster in rasters.values():
                raster.close()
            raise
        # A cached one-step file reduces to itself under any statistic.
        # (Not `recipe.spec.statistic`: a rolling k-day statistic needs a daily
        # series, which a cached annual-extreme file no longer has.)
        identity = (
            BlockStatistic.minimum()
            if recipe.spec.tail == "lower"
            else BlockStatistic.maximum()
        )
        return BlockExtremaCurveSource(
            reader=NetCDFBlockReader(rasters, spec=BlockSpec(statistic=identity)),
            metadata=metadata,
            tail=recipe.spec.tail,
            strip_bytes=strip_bytes,
            minimum_blocks=minimum_blocks,
            info=info,
        )
