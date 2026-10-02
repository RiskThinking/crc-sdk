"""ERA5 hourly reanalysis access for annual-extreme recipes.

Reads public ARCO-style Zarr copies of ERA5 anonymously (no CDS account, no
GRIB tooling; see ``docs/spikes/era5.md``), reduces one calendar year at a time
to an *annual extreme per grid cell* for the requested area, and caches that
small result -- never the hourly data.

Two stores are built in. They hold the same reanalysis at different grids:

* ``arco-0p25`` -- Google's ARCO-ERA5, native 0.25 degrees (~28 km). Chunked one
  hour per global field, so a year costs ~8,800 chunk reads whatever the area
  size: minutes per variable-year.
* ``wb2-1p5`` -- WeatherBench2's conservatively regridded 1.5 degree copy,
  chunked 8 hours per field: ~15 seconds per variable-year, the laptop route.

The recipe set is deliberately small (TXx, TNn, Rx1day, Rx5day); see
`ERA5_RECIPES`. Caveats worth carrying into any use of the results: reanalysis
blends observations with a model and its biases move with the observing
system; convective rainfall extremes are underestimated; and a grid cell is an
area average, not a point.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from crc_sdk.connectors.blocks import (
    STRIP_BYTES,
    BlockExtremaCurveSource,
    BlockSpec,
    BlockStatistic,
    DailyAggregation,
    UnitConversion,
    _strip_row_count,
    aggregate_daily,
    reduce_days,
)
from crc_sdk.connectors.duckdb.netcdf import NetCDFBlockReader, NetCDFRaster
from crc_sdk.connectors.duckdb.zarr import Bounds, RasterMetadata

ERA5_LICENCE = "CC-BY-4.0"
ERA5_ATTRIBUTION = "Contains modified Copernicus Climate Change Service information"
# Days read per request while building one year. Bounds the hourly slab
# (segment_days * 24 * cells) independently of the block length.
SEGMENT_DAYS = 30
LAND_THRESHOLD = 0.5
EXTREME_VARIABLE = "extreme"


@dataclass(frozen=True)
class ERA5Store:
    """One public Zarr copy of ERA5 and where to find things in it."""

    name: str
    url: str
    resolution_degrees: float
    variables: Mapping[str, str]  # recipe variable key -> array name in the store
    zarr_format: int = 2
    storage_options: Mapping[str, Any] = field(
        default_factory=lambda: {"token": "anon"}
    )
    land_sea_mask: str | None = "land_sea_mask"
    lat_name: str = "latitude"
    lon_name: str = "longitude"
    time_name: str = "time"
    step_hours: float = 1.0
    # Fallback coverage when the store carries no valid_time_* attributes.
    first_year: int | None = None
    last_final_year: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.url:
            raise ValueError("name and url must be non-empty")


ARCO_0P25 = ERA5Store(
    name="arco-0p25",
    url="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    resolution_degrees=0.25,
    variables={"t2m": "2m_temperature", "tp": "total_precipitation"},
    description="Google ARCO-ERA5, native 0.25 degree, hourly, one hour per chunk",
)

WB2_1P5 = ERA5Store(
    name="wb2-1p5",
    url=(
        "gs://weatherbench2/datasets/era5/"
        "1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr"
    ),
    resolution_degrees=1.5,
    variables={"t2m": "2m_temperature", "tp": "total_precipitation"},
    first_year=1959,
    last_final_year=2022,
    description="WeatherBench2 ERA5, conservatively regridded to 1.5 degrees, hourly",
)

ERA5_STORES = {store.name: store for store in (ARCO_0P25, WB2_1P5)}


def era5_store(name: str) -> ERA5Store:
    try:
        return ERA5_STORES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown ERA5 store {name!r}; choose from {tuple(ERA5_STORES)}"
        ) from error


@dataclass(frozen=True)
class ERA5Recipe:
    """One supported dataset -> hazard recipe (ADR-0004 names).

    ``hazard_name`` matches the open registry / internal name only where the
    definition is the same; the daily boundary here is **UTC**.
    """

    name: str
    variable: str  # key into ERA5Store.variables
    unit: str
    value_semantics: str
    spec: BlockSpec

    @property
    def tail(self) -> Literal["upper", "lower"]:
        return self.spec.tail


# 2 m temperature is instantaneous at hourly stamps: the "daily maximum" is the
# largest of 24 hourly values, a slight underestimate of the true extreme.
_KELVIN_TO_CELSIUS = UnitConversion(offset=-273.15)
# Hourly total precipitation is an accumulation (metres) over the hour *ending*
# at the stamp, so the hour stamped 00:00 belongs to the day that just ended.
_METRES_TO_MM = UnitConversion(scale=1000.0)
_PRECIP_DAY = DailyAggregation(
    "sum", step_hours=1.0, min_valid_fraction=1.0, interval_end_stamps=True
)

ERA5_RECIPES: dict[str, ERA5Recipe] = {
    recipe.name: recipe
    for recipe in (
        ERA5Recipe(
            "txx",
            "t2m",
            "degC",
            "annual maximum of daily maximum 2 m temperature (UTC days)",
            BlockSpec(
                statistic=BlockStatistic.maximum(),
                daily=DailyAggregation("max", min_valid_fraction=0.75),
                conversion=_KELVIN_TO_CELSIUS,
                min_valid_fraction=0.9,
            ),
        ),
        ERA5Recipe(
            "tnn",
            "t2m",
            "degC",
            "annual minimum of daily minimum 2 m temperature (UTC days)",
            BlockSpec(
                statistic=BlockStatistic.minimum(),
                daily=DailyAggregation("min", min_valid_fraction=0.75),
                conversion=_KELVIN_TO_CELSIUS,
                min_valid_fraction=0.9,
            ),
        ),
        ERA5Recipe(
            "rx1day",
            "tp",
            "mm/day",
            "annual maximum one-day precipitation total (UTC days)",
            BlockSpec(
                statistic=BlockStatistic.maximum(),
                daily=_PRECIP_DAY,
                conversion=_METRES_TO_MM,
                min_valid_fraction=0.9,
            ),
        ),
        ERA5Recipe(
            "rx5day",
            "tp",
            "mm",
            "annual maximum five-day precipitation total (UTC days)",
            BlockSpec(
                statistic=BlockStatistic.k_day_sum(5),
                daily=_PRECIP_DAY,
                conversion=_METRES_TO_MM,
                min_valid_fraction=0.9,
            ),
        ),
    )
}


def era5_recipe(name: str) -> ERA5Recipe:
    try:
        return ERA5_RECIPES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown ERA5 recipe {name!r}; choose from {tuple(ERA5_RECIPES)}"
        ) from error


def _open_group(store: ERA5Store) -> Any:
    try:
        import zarr
    except ImportError as error:
        raise ImportError("ERA5 access requires `pip install crc-sdk[zarr]`") from error
    options = dict(store.storage_options)
    config = getattr(zarr, "config", None)
    if config is not None:
        # Hourly chunks are small and latency-bound; zarr's default of 10
        # concurrent requests leaves most of the bandwidth idle.
        config.set({"async.concurrency": 64})
    try:
        return zarr.open_group(
            store.url,
            mode="r",
            storage_options=options or None,
            zarr_format=store.zarr_format,  # type: ignore[arg-type]
        )
    except TypeError:  # zarr 2 has no zarr_format argument
        import fsspec  # type: ignore[import-untyped]

        return zarr.open_group(fsspec.get_mapper(store.url, **options), mode="r")


def _decode_time(array: Any) -> np.ndarray[Any, Any]:
    """CF ``<unit> since <date>`` integer time -> ``datetime64[s]``."""
    units = str(array.attrs.get("units", ""))
    match = re.match(r"\s*(hours|days|seconds)\s+since\s+(\S+)(?:\s+(\S+))?", units)
    if match is None:
        raise ValueError(f"unsupported time units {units!r}")
    unit, date, clock = match.groups()
    origin = np.datetime64(f"{date}T{clock}" if clock else date, "s")
    seconds = {"hours": 3600, "days": 86400, "seconds": 1}[unit]
    raw = np.asarray(array[:], dtype=np.int64)
    return origin + raw * np.timedelta64(seconds, "s")


def _runs(indices: np.ndarray[Any, Any]) -> list[tuple[int, int]]:
    """Maximal ``[start, stop)`` runs of consecutive ascending integers."""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, indices.size]
    return [(int(indices[a]), int(indices[b - 1]) + 1) for a, b in zip(starts, stops)]


def _at_least_two(indices: np.ndarray[Any, Any], size: int) -> np.ndarray[Any, Any]:
    if indices.size >= 2 or size < 2:
        return indices
    only = int(indices[0])
    return np.array([only, only + 1] if only + 1 < size else [only - 1, only])


@dataclass(frozen=True)
class ERA5Coverage:
    first_year: int
    last_final_date: np.datetime64
    last_era5t_date: np.datetime64 | None
    last_updated: str | None

    def last_complete_year(self, *, interval_end: bool = False) -> int:
        """Last year whose every hour (and, for accumulations, every interval) exists.

        An accumulation stamped at the *end* of its hour needs the 00:00 sample
        of 1 January to close 31 December, so it needs one more day of record.
        """
        stop = self.last_final_date.astype("datetime64[D]")
        extra = np.timedelta64(1 if interval_end else 0, "D")
        year = int(str(stop)[:4])
        closes = np.datetime64(f"{year}-12-31", "D") + extra
        return year if stop >= closes else year - 1


class ERA5BlockReader:
    """`BlockReader` that reduces one ERA5 year at a time straight from the store."""

    def __init__(
        self,
        provider: ERA5Provider,
        recipe: ERA5Recipe,
        bounds: Bounds,
        *,
        land_only: bool = False,
        years: Sequence[int],
    ) -> None:
        self._provider = provider
        self.recipe = recipe
        self.spec = recipe.spec
        self._years = tuple(years)
        group = provider.group
        store = provider.store
        self._array = group[store.variables[recipe.variable]]
        dims = tuple(self._array.attrs.get("_ARRAY_DIMENSIONS", ()))
        wanted = (store.time_name, store.lat_name, store.lon_name)
        if sorted(dims) != sorted(wanted):
            raise ValueError(
                f"{store.variables[recipe.variable]!r} has dimensions {dims!r}; "
                f"expected {wanted!r} in any order"
            )
        self._axis = {role: dims.index(role) for role in wanted}
        self._times = provider.times
        lat = np.asarray(group[store.lat_name][:], dtype=np.float64)
        lon = np.asarray(group[store.lon_name][:], dtype=np.float64)
        lon180 = ((lon + 180.0) % 360.0) - 180.0

        min_lon, min_lat, max_lon, max_lat = bounds
        half_lat = abs(float(lat[1] - lat[0])) / 2
        half_lon = abs(float(lon[1] - lon[0])) / 2
        # The 180 meridian is one column that normalises to -180; a window that
        # ends at +180 (and does not start at -180) must see it at +180.
        if min_lon > -180.0 + half_lon:
            lon180[np.isclose(lon180, -180.0)] = 180.0
        rows = np.flatnonzero((lat >= min_lat - half_lat) & (lat <= max_lat + half_lat))
        columns = np.flatnonzero(
            (lon180 >= min_lon - half_lon) & (lon180 <= max_lon + half_lon)
        )
        if rows.size == 0 or columns.size == 0:
            raise ValueError("bounds do not intersect the ERA5 grid")
        columns = columns[np.argsort(lon180[columns], kind="stable")]
        self._rows = _at_least_two(rows, lat.size)
        self._cols = columns if columns.size >= 2 else _at_least_two(columns, lon.size)
        self._lat: np.ndarray[Any, Any] = lat[self._rows]
        self._lon: np.ndarray[Any, Any] = lon180[self._cols]
        self._mask: np.ndarray[Any, Any] | None = None
        if land_only:
            self._mask = self._read_land_mask()

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
        # The block's daily array plus one segment of sub-daily samples.
        spec = self.spec.daily
        per_day = spec.samples_per_day if spec is not None else 1
        return 366 + SEGMENT_DAYS * per_day

    @property
    def block_height(self) -> int:
        return 1

    def close(self) -> None:
        return None

    def _read(
        self,
        name: str,
        t0: int,
        t1: int,
        rows: np.ndarray[Any, Any],
        cols: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """``(time, row, column)`` float64 for a row run and column indices."""
        store = self._provider.store
        array = self._array if name == "data" else self._provider.group[name]
        dims = list(array.attrs["_ARRAY_DIMENSIONS"])
        has_time = store.time_name in dims
        lat_axis, lon_axis = dims.index(store.lat_name), dims.index(store.lon_name)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        # Output axes, as positions in the stored array: (time?, lat, lon).
        order = [lat_axis, lon_axis]
        if has_time:
            order.insert(0, dims.index(store.time_name))
        pieces = []
        for c0, c1 in _runs(cols):
            key: list[Any] = [slice(None)] * len(dims)
            key[lat_axis] = slice(r0, r1)
            key[lon_axis] = slice(c0, c1)
            if has_time:
                key[order[0]] = slice(t0, t1)
            block = np.asarray(array[tuple(key)], dtype=np.float64)
            block = np.transpose(block, order)
            pieces.append(block if has_time else block[np.newaxis])
        return np.concatenate(pieces, axis=2) if len(pieces) > 1 else pieces[0]

    def _read_land_mask(self) -> np.ndarray[Any, Any]:
        store = self._provider.store
        if (
            store.land_sea_mask is None
            or store.land_sea_mask not in self._provider.group
        ):
            raise ValueError(f"store {store.name!r} has no land-sea mask")
        mask = self._read(store.land_sea_mask, 0, 1, self._rows, self._cols)[0]
        return np.asarray(mask >= LAND_THRESHOLD)

    def reduce_block(
        self, label: int, rows: slice, columns: slice
    ) -> np.ndarray[Any, Any]:
        spec = self.spec
        daily = spec.daily
        assert daily is not None
        first_day, stop_day = spec.block.span(label)
        day_count = int((stop_day - first_day) / np.timedelta64(1, "D"))
        window_rows = self._rows[rows]
        window_cols = self._cols[columns]
        series = np.full(
            (day_count, window_rows.size, window_cols.size), np.nan, dtype=np.float64
        )
        shift = daily.stamp_shift
        for offset in range(0, day_count, SEGMENT_DAYS):
            stop = min(offset + SEGMENT_DAYS, day_count)
            lower = (first_day + np.timedelta64(offset, "D")).astype(
                "datetime64[s]"
            ) - shift
            upper = (first_day + np.timedelta64(stop, "D")).astype(
                "datetime64[s]"
            ) - shift
            t0, t1 = (int(i) for i in np.searchsorted(self._times, [lower, upper]))
            if t1 <= t0:
                continue
            samples = self._read("data", t0, t1, window_rows, window_cols)
            samples = spec.conversion.apply(samples)
            series[offset:stop] = aggregate_daily(
                self._times[t0:t1],
                samples,
                daily,
                first_day + np.timedelta64(offset, "D"),
                stop - offset,
            )
        result = reduce_days(series, spec.statistic, spec.min_valid_fraction)
        if self._mask is not None:
            result = np.where(self._mask[rows, :][:, columns], result, np.nan)
        return result


class ERA5Provider:
    """Open an ERA5 Zarr store and build per-year annual-extreme files."""

    def __init__(self, store: ERA5Store = WB2_1P5, *, group: Any | None = None) -> None:
        self.store = store
        self._group = group
        self._times: np.ndarray[Any, Any] | None = None

    @property
    def group(self) -> Any:
        if self._group is None:
            self._group = _open_group(self.store)
        return self._group

    @property
    def times(self) -> np.ndarray[Any, Any]:
        if self._times is None:
            self._times = _decode_time(self.group[self.store.time_name])
        return self._times

    def coverage(self) -> ERA5Coverage:
        attrs = dict(self.group.attrs)
        start = attrs.get("valid_time_start")
        stop = attrs.get("valid_time_stop")
        if stop is not None and start is not None:
            era5t = attrs.get("valid_time_stop_era5t")
            return ERA5Coverage(
                first_year=int(str(start)[:4]),
                last_final_date=np.datetime64(str(stop)[:10], "D"),
                last_era5t_date=(
                    np.datetime64(str(era5t)[:10], "D") if era5t is not None else None
                ),
                last_updated=attrs.get("last_updated"),
            )
        if self.store.first_year is None or self.store.last_final_year is None:
            raise RuntimeError(
                f"store {self.store.name!r} does not declare its coverage"
            )
        return ERA5Coverage(
            first_year=self.store.first_year,
            last_final_date=np.datetime64(f"{self.store.last_final_year}-12-31", "D"),
            last_era5t_date=None,
            last_updated=None,
        )

    def resolve_version(self, requested: str) -> str:
        """The data release this store currently serves.

        ARCO appends new months in place, so the version is the end of the
        *final* (non-ERA5T) record: data up to it is stable, and a cache keyed
        on it stays valid when the store grows. ERA5T months are never used.
        """
        coverage = self.coverage()
        current = f"final-through-{coverage.last_final_date}"
        if requested not in ("latest", current):
            raise ValueError(
                f"{self.store.name} release {requested!r} is not served; "
                f"current release: {current}"
            )
        return current

    def check_years(
        self, years: Sequence[int], recipe: ERA5Recipe | None = None
    ) -> None:
        coverage = self.coverage()
        daily = recipe.spec.daily if recipe is not None else None
        interval_end = daily is not None and daily.interval_end_stamps
        low = coverage.first_year
        high = coverage.last_complete_year(interval_end=interval_end)
        outside = [year for year in years if not low <= year <= high]
        if outside:
            raise ValueError(
                f"years {outside} are outside the final, complete ERA5 record "
                f"{low}-{high} in {self.store.name} (ERA5T months are not used)"
            )

    def reader(
        self,
        recipe: ERA5Recipe,
        bounds: Bounds,
        years: Sequence[int],
        *,
        land_only: bool = False,
    ) -> ERA5BlockReader:
        if recipe.variable not in self.store.variables:
            raise ValueError(
                f"store {self.store.name!r} has no variable for recipe {recipe.name!r}"
            )
        return ERA5BlockReader(self, recipe, bounds, land_only=land_only, years=years)

    def cache_annual_extreme(
        self,
        recipe: ERA5Recipe,
        year: int,
        bounds: Bounds,
        destination: str | Path,
        *,
        land_only: bool = False,
        strip_bytes: int = STRIP_BYTES,
        attributes: Mapping[str, str] | None = None,
    ) -> Path:
        """Persist one year's area annual extreme as a compact NetCDF file."""
        try:
            import h5netcdf  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "ERA5 caching requires `pip install crc-sdk[netcdf]`"
            ) from error
        self.check_years([year], recipe)
        reader = self.reader(recipe, bounds, [year], land_only=land_only)
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
            variable.attrs["store"] = self.store.name
            for key, value in (attributes or {}).items():
                variable.attrs[key] = value
        temporary.replace(target)
        return target

    def open_resources(
        self,
        recipe: ERA5Recipe,
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
        # aggregation, which a cached annual-extreme file no longer has.)
        identity = (
            BlockStatistic.minimum()
            if recipe.spec.tail == "lower"
            else BlockStatistic.maximum()
        )
        spec = BlockSpec(statistic=identity)
        return BlockExtremaCurveSource(
            reader=NetCDFBlockReader(rasters, spec=spec),
            metadata=metadata,
            tail=recipe.spec.tail,
            strip_bytes=strip_bytes,
            minimum_blocks=minimum_blocks,
            info=info,
        )
