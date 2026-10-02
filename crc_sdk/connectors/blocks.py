"""Generic block-extrema curve source.

A block-extrema curve is built from a time series, not read from a published
return-period map: for each grid cell, one value per *block* (a calendar year,
a season or a water year) -- the annual maximum of daily maximum temperature,
the annual minimum of a soil-moisture index, the largest five-day rainfall
total -- and the resulting handful of per-year values is turned into an
empirical return-period curve with a plotting position (Gringorten by
default).

Reading and reduction are deliberately separate. A `BlockReader` turns one
block label (a year) into one 2-D array of that block's statistic for a window
of rows and columns; everything remote and expensive (an hourly reanalysis
store, say) happens inside `reduce_block`, and its small per-year result is
what callers cache -- annual extremes, never the daily or hourly data. The
curve source then streams those per-year arrays strip by strip, so peak memory
is bounded by `strip_bytes` regardless of the grid size. `EDOAnnualMinimaCurveSource`
(`crc_sdk.connectors.duckdb.netcdf`) is this class with a NetCDF-backed reader.

Statistical conventions (recorded for audit, never silently changed):

* a *block* is labelled by the calendar year in which it **ends** (a water year
  from October is labelled by the following year, DJF by the year of its
  January);
* a sample is valid when finite and not the nodata value; a *day* needs
  ``min_valid_fraction`` of its samples, and a *block* needs
  ``min_valid_fraction`` of its days (``0.0`` means "at least one"); a pixel
  that fails the block test gets no value for that year rather than a biased one;
* rolling k-day statistics never cross a block boundary and need all k days
  valid.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    # Imported lazily at runtime: the duckdb package imports this module (its
    # NetCDF reader is built on it), so a top-level import would be circular.
    from crc_sdk.connectors.duckdb.zarr import (
        Bounds,
        Point,
        RasterCurve,
        RasterMetadata,
    )

# Decompressed bytes per strip read; bounds worker RAM independent of raster
# size. Same default as the GeoTIFF and NetCDF readers.
STRIP_BYTES = 256 * 1024**2

#: Gringorten plotting-position constant, a standard, mildly conservative
#: choice for annual extremes.
GRINGORTEN_A = 0.44
MINIMUM_BLOCKS = 4


def _index_range(
    coords: np.ndarray[Any, Any], low: float, high: float, step: float
) -> tuple[int, int]:
    """Half-open `[start, stop)` index range covering `[low, high]` along `coords`.

    Works regardless of whether `coords` is ascending or descending (EDO's
    own `lat` axis runs north-to-south, i.e. descending).
    """
    half_step = abs(step) / 2
    mask = (coords >= low - half_step) & (coords <= high + half_step)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("bounds do not intersect the raster")
    return int(indices.min()), int(indices.max()) + 1


def _strip_row_count(
    strip_bytes: int,
    width: int,
    itemsize: int,
    block_height: int,
    *,
    leading_axis: int = 1,
) -> int:
    """Row count per strip, bounding `leading_axis * rows * width * itemsize`
    by `strip_bytes` and aligned to `block_height`.

    `leading_axis` is 1 for a single-time-slice read (`NetCDFRaster`); a
    caller reading `leading_axis` steps per row at once (e.g. a whole year's
    dekads before reducing them) must pass that count, or the strip is
    undersized by roughly that factor -- the actual resident array before any
    reduction is `(leading_axis, rows, width)`, not `(rows, width)`.
    """
    rows_per_strip = max(
        block_height, strip_bytes // max(1, width * itemsize * leading_axis)
    )
    return max(block_height, (rows_per_strip // block_height) * block_height)


def pixel_boundary(
    lat: np.ndarray[Any, Any],
    lon: np.ndarray[Any, Any],
    lat_step: float,
    lon_step: float,
    row: int,
    column: int,
) -> tuple[Point, Point, Point, Point]:
    """Pixel corner boundary, counter-clockwise from the north-west corner.

    Uses `abs(step)` rather than the signed step: a coordinate array's
    direction is not guaranteed by the format (EDO's own `lat` runs
    north-to-south, ERA5's weatherbench copies run south-to-north). Corners
    are built in a fixed NW/SW/SE/NE geographic order, so the ring winds the
    same way whichever way `lat`/`lon` run.
    """
    half_lat = abs(lat_step) / 2
    half_lon = abs(lon_step) / 2
    centre_lat = float(lat[row])
    centre_lon = float(lon[column])
    return (
        (centre_lon - half_lon, centre_lat + half_lat),
        (centre_lon - half_lon, centre_lat - half_lat),
        (centre_lon + half_lon, centre_lat - half_lat),
        (centre_lon + half_lon, centre_lat + half_lat),
    )


# ---------------------------------------------------------------------------
# Block definitions and statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockDefinition:
    """How a calendar is cut into blocks, one block per label year."""

    kind: Literal["annual", "season", "water_year"] = "annual"
    months: tuple[int, ...] | None = None
    start_month: int = 10

    def __post_init__(self) -> None:
        if self.kind == "season":
            months = self.months
            if not months or len(set(months)) != len(months):
                raise ValueError("a season needs distinct, non-empty months")
            if any(not 1 <= month <= 12 for month in months):
                raise ValueError("season months must be between 1 and 12")
            if len(months) == 12:
                raise ValueError("use BlockDefinition.annual() for a full year")
            if any(
                (months[i] % 12) + 1 != months[i + 1] for i in range(len(months) - 1)
            ):
                raise ValueError("season months must be consecutive and in order")
        elif self.months is not None:
            raise ValueError("months only apply to season blocks")
        if self.kind == "water_year" and not 2 <= self.start_month <= 12:
            raise ValueError("water-year start_month must be between 2 and 12")

    @classmethod
    def annual(cls) -> BlockDefinition:
        return cls("annual")

    @classmethod
    def season(cls, months: Sequence[int]) -> BlockDefinition:
        return cls("season", months=tuple(months))

    @classmethod
    def water_year(cls, start_month: int = 10) -> BlockDefinition:
        return cls("water_year", start_month=start_month)

    def span(self, label: int) -> tuple[np.datetime64, np.datetime64]:
        """Half-open ``[start, stop)`` day range of block ``label``."""
        if self.kind == "annual":
            start, stop = (label, 1), (label + 1, 1)
        elif self.kind == "water_year":
            start, stop = (label - 1, self.start_month), (label, self.start_month)
        else:
            assert self.months is not None
            first, last = self.months[0], self.months[-1]
            # A season labelled `label` ends in `label`; one that wraps the new
            # year (DJF) therefore starts the year before.
            start_year = label if last >= first else label - 1
            stop_year = label + 1 if last == 12 else label
            start, stop = (start_year, first), (stop_year, last % 12 + 1)
        return (
            np.datetime64(f"{start[0]:04d}-{start[1]:02d}-01", "D"),
            np.datetime64(f"{stop[0]:04d}-{stop[1]:02d}-01", "D"),
        )

    def describe(self) -> str:
        if self.kind == "season":
            return f"season months {self.months}"
        if self.kind == "water_year":
            return f"water year starting month {self.start_month}"
        return "calendar year"


@dataclass(frozen=True)
class BlockStatistic:
    """Per-block statistic over the (optionally daily) series.

    ``kind`` picks the reduction over the block; ``window_days > 1`` first
    rolls a ``rolling`` (``sum`` or ``mean``) window of that many days, so
    ``k_day_sum(5)`` is "the largest five-day total in the block" (Rx5day).
    """

    kind: Literal["max", "min", "count_above", "count_below"] = "max"
    window_days: int = 1
    rolling: Literal["sum", "mean"] | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.window_days < 1:
            raise ValueError("window_days must be at least one")
        if self.window_days > 1 and self.rolling is None:
            raise ValueError("a multi-day window needs rolling='sum' or 'mean'")
        if self.window_days == 1 and self.rolling is not None:
            raise ValueError("rolling needs window_days greater than one")
        if self.kind.startswith("count") and (
            self.threshold is None or self.window_days != 1
        ):
            raise ValueError("count statistics need a threshold and no rolling window")
        if self.kind in ("max", "min") and self.threshold is not None:
            raise ValueError("threshold only applies to count statistics")

    @classmethod
    def maximum(cls) -> BlockStatistic:
        return cls("max")

    @classmethod
    def minimum(cls) -> BlockStatistic:
        return cls("min")

    @classmethod
    def k_day_sum(
        cls, days: int, *, reduce: Literal["max", "min"] = "max"
    ) -> BlockStatistic:
        return cls(reduce, window_days=days, rolling="sum")

    @classmethod
    def k_day_mean(
        cls, days: int, *, reduce: Literal["max", "min"] = "max"
    ) -> BlockStatistic:
        return cls(reduce, window_days=days, rolling="mean")

    @classmethod
    def count_above(cls, threshold: float) -> BlockStatistic:
        return cls("count_above", threshold=threshold)

    @classmethod
    def count_below(cls, threshold: float) -> BlockStatistic:
        return cls("count_below", threshold=threshold)

    @property
    def tail(self) -> Literal["upper", "lower"]:
        """The severe direction: low for minima and below-threshold counts."""
        return "lower" if self.kind in ("min", "count_below") else "upper"


@dataclass(frozen=True)
class DailyAggregation:
    """Hourly (or sub-daily) samples -> one value per day.

    ``how`` is the daily reduction. ``utc_offset_hours`` shifts the day
    boundary from UTC to local standard time (one offset for the whole area;
    default UTC). ``interval_end_stamps`` is for accumulated variables whose
    stamp is the *end* of the accumulation interval (ERA5 ``tp``): a sample
    stamped 00:00 belongs to the day that just ended, not the one starting.
    """

    how: Literal["max", "min", "mean", "sum"]
    step_hours: float = 1.0
    min_valid_fraction: float = 0.75
    utc_offset_hours: float = 0.0
    interval_end_stamps: bool = False

    def __post_init__(self) -> None:
        if self.step_hours <= 0 or 24 % self.step_hours != 0:
            raise ValueError("step_hours must divide 24")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be within [0, 1]")

    @property
    def samples_per_day(self) -> int:
        return int(round(24 / self.step_hours))

    @property
    def stamp_shift(self) -> np.timedelta64:
        """Added to a stamp to get the local time of its *interval start*."""
        seconds = self.utc_offset_hours * 3600.0
        if self.interval_end_stamps:
            seconds -= self.step_hours * 3600.0
        return np.timedelta64(int(round(seconds)), "s")


@dataclass(frozen=True)
class UnitConversion:
    """``value * scale + offset``, applied to raw samples before any reduction."""

    scale: float = 1.0
    offset: float = 0.0

    def apply(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.scale == 1.0 and self.offset == 0.0:
            return values
        return values * self.scale + self.offset


@dataclass(frozen=True)
class BlockSpec:
    """Everything that defines one block-extrema recipe, apart from the data."""

    statistic: BlockStatistic = BlockStatistic()
    block: BlockDefinition = BlockDefinition()
    daily: DailyAggregation | None = None
    conversion: UnitConversion = UnitConversion()
    min_valid_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be within [0, 1]")
        if self.statistic.window_days > 1 and self.daily is None:
            raise ValueError("rolling statistics need a daily aggregation")

    @property
    def tail(self) -> Literal["upper", "lower"]:
        return self.statistic.tail


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def reduce_days(
    daily: np.ndarray[Any, Any],
    statistic: BlockStatistic,
    min_valid_fraction: float,
) -> np.ndarray[Any, Any]:
    """Reduce a ``(days, rows, cols)`` series (NaN = invalid) to ``(rows, cols)``."""
    expected = daily.shape[0]
    valid_count = np.isfinite(daily).sum(axis=0)
    complete = valid_count >= max(1.0, math.ceil(min_valid_fraction * expected - 1e-9))
    with warnings.catch_warnings():
        # An all-NaN pixel (outside the data's coverage, a masked-out sea cell)
        # is an expected, not exceptional, slice.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if statistic.kind == "count_above":
            assert statistic.threshold is not None
            result = (daily > statistic.threshold).sum(axis=0).astype(np.float64)
        elif statistic.kind == "count_below":
            assert statistic.threshold is not None
            result = (daily < statistic.threshold).sum(axis=0).astype(np.float64)
        else:
            series = daily
            if statistic.window_days > 1:
                if expected < statistic.window_days:
                    return np.full(daily.shape[1:], np.nan)
                windows = np.lib.stride_tricks.sliding_window_view(
                    daily, statistic.window_days, axis=0
                )
                series = (
                    windows.sum(axis=-1)
                    if statistic.rolling == "sum"
                    else windows.mean(axis=-1)
                )
            result = (
                np.nanmax(series, axis=0)
                if statistic.kind == "max"
                else np.nanmin(series, axis=0)
            )
    return np.where(complete, result, np.nan)


def aggregate_daily(
    times: np.ndarray[Any, Any],
    values: np.ndarray[Any, Any],
    daily: DailyAggregation,
    first_day: np.datetime64,
    day_count: int,
) -> np.ndarray[Any, Any]:
    """Sub-daily ``(time, rows, cols)`` samples -> ``(day_count, rows, cols)``.

    ``times`` must be ascending. Days with no samples, or fewer than
    ``daily.min_valid_fraction`` of the expected ones, are NaN.
    """
    out = np.full((day_count,) + values.shape[1:], np.nan, dtype=np.float64)
    if times.size == 0:
        return out
    local = times.astype("datetime64[s]") + daily.stamp_shift
    day_index = (local.astype("datetime64[D]") - first_day).astype(np.int64)
    inside = (day_index >= 0) & (day_index < day_count)
    if not inside.all():
        day_index = day_index[inside]
        values = values[inside]
        if day_index.size == 0:
            return out
    starts = np.flatnonzero(np.r_[True, day_index[1:] != day_index[:-1]])
    groups = day_index[starts]
    finite = np.isfinite(values)
    counts = np.add.reduceat(finite, starts, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if daily.how == "max":
            reduced = np.fmax.reduceat(values, starts, axis=0)
        elif daily.how == "min":
            reduced = np.fmin.reduceat(values, starts, axis=0)
        else:
            total = np.add.reduceat(np.where(finite, values, 0.0), starts, axis=0)
            reduced = total if daily.how == "sum" else total / np.maximum(counts, 1)
    needed = max(1, math.ceil(daily.min_valid_fraction * daily.samples_per_day - 1e-9))
    reduced = np.where(counts >= needed, reduced, np.nan)
    out[groups] = reduced
    return out


def reduce_block_samples(
    values: np.ndarray[Any, Any],
    spec: BlockSpec,
    *,
    times: np.ndarray[Any, Any] | None = None,
    span: tuple[np.datetime64, np.datetime64] | None = None,
) -> np.ndarray[Any, Any]:
    """Reduce one block's ``(time, rows, cols)`` samples to ``(rows, cols)``.

    Without a daily aggregation the samples themselves are the series (EDO's
    dekads); with one, ``times`` and the block ``span`` are required.
    """
    values = spec.conversion.apply(np.asarray(values, dtype=np.float64))
    if spec.daily is None:
        return reduce_days(values, spec.statistic, spec.min_valid_fraction)
    if times is None or span is None:
        raise ValueError("a daily aggregation needs sample times and the block span")
    first_day, stop_day = span
    day_count = int((stop_day - first_day) / np.timedelta64(1, "D"))
    series = aggregate_daily(times, values, spec.daily, first_day, day_count)
    return reduce_days(series, spec.statistic, spec.min_valid_fraction)


# ---------------------------------------------------------------------------
# Plotting-position curves
# ---------------------------------------------------------------------------


def plotting_position_curve(
    block_values: np.ndarray[Any, Any],
    tail: Literal["upper", "lower"],
    *,
    minimum_blocks: int = MINIMUM_BLOCKS,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """One pixel's per-block values -> ``(periods, values)`` ready to fit.

    Treats each block's value as one extreme-value sample (missing blocks are
    dropped, never zero-filled) and assigns each an empirical return period
    via the Gringorten plotting position (``a = 0.44``): the estimate that
    this value's severity is reached, on average, once every that-many blocks.
    The ranks run from the *severe* end -- the largest value for the upper
    tail, the smallest for the lower -- so periods increase with severity and
    the pair can be fitted like any other tabulated curve via
    ``TabulatedDistribution.from_return_periods(periods, values, tail=tail)``.
    No distribution family is assumed: the position is purely rank-based.

    Returns two empty arrays if fewer than ``minimum_blocks`` blocks are valid
    -- the same per-pixel floor `canonicalize_curve_source` enforces for any
    curve source, checked here too so a caller iterating this source directly
    sees the same "too few knots" signal.
    """
    valid = block_values[np.isfinite(block_values)]
    count = valid.size
    if count < max(minimum_blocks, MINIMUM_BLOCKS):
        return np.array([]), np.array([])
    values = np.sort(valid)
    ranks = np.arange(1, count + 1, dtype=np.float64)
    if tail == "upper":
        ranks = ranks[::-1]
    probabilities = (ranks - GRINGORTEN_A) / (count + 1 - 2 * GRINGORTEN_A)
    return 1.0 / probabilities, values


def gringorten_support(count: int) -> tuple[float, float]:
    """Return-period range ``count`` blocks support, shortest to longest."""
    low = (count - GRINGORTEN_A) / (count + 1 - 2 * GRINGORTEN_A)
    high = (1 - GRINGORTEN_A) / (count + 1 - 2 * GRINGORTEN_A)
    return 1.0 / low, 1.0 / high


# ---------------------------------------------------------------------------
# Readers and the curve source
# ---------------------------------------------------------------------------


@runtime_checkable
class BlockReader(Protocol):
    """A grid of per-block statistics, one 2-D array per block label.

    Implementations own their own I/O and keep it bounded: ``reduce_block``
    may be handed any row/column window, so it must itself limit the samples
    it holds at once (``leading_axis_samples`` states how many samples per
    pixel it makes resident, so the curve source can size strips).
    """

    @property
    def lat(self) -> np.ndarray[Any, Any]: ...

    @property
    def lon(self) -> np.ndarray[Any, Any]: ...

    @property
    def labels(self) -> tuple[int, ...]: ...

    @property
    def leading_axis_samples(self) -> int: ...

    @property
    def block_height(self) -> int: ...

    def reduce_block(
        self, label: int, rows: slice, columns: slice
    ) -> np.ndarray[Any, Any]: ...

    def close(self) -> None: ...


@dataclass
class BlockExtremaCurveSource:
    """Presents N blocks of a time series as one empirical return-period curve source.

    Satisfies `crc_sdk.connectors.adapters.CurveSource`: one curve per grid
    cell with at least ``minimum_blocks`` valid blocks, fed through the same
    `canonicalize_curve_source` JRC flood and OS-Climate use -- just with
    empirical (Gringorten) plotting-position return periods instead of
    literal return-period rasters or quantile samples.
    """

    reader: BlockReader
    metadata: RasterMetadata
    tail: Literal["upper", "lower"] = "upper"
    strip_bytes: int = STRIP_BYTES
    minimum_blocks: int = MINIMUM_BLOCKS
    #: Extra 1.3 metadata a canonicalizer should record; see
    #: `crc_sdk.connectors.adapters.CurveSourceInfo`.
    info: Any = None
    #: Optional ``(row, column, reason, message)`` callback told about every
    #: cell dropped before a curve is yielded (diagnostics sidecar).
    skip_sink: Any = None

    #: The curve values are genuine per-block samples, so a sample-based fit
    #: (``fit_method="sample_mle"``) is meaningful for this source.
    values_are_samples = True

    def __post_init__(self) -> None:
        if not self.reader.labels:
            raise ValueError("at least one block is required")
        if self.strip_bytes < 1:
            raise ValueError("strip_bytes must be positive")
        if self.minimum_blocks < MINIMUM_BLOCKS:
            raise ValueError(f"minimum_blocks must be at least {MINIMUM_BLOCKS}")
        self._labels = tuple(sorted(self.reader.labels))
        lat = np.asarray(self.reader.lat, dtype=np.float64)
        lon = np.asarray(self.reader.lon, dtype=np.float64)
        if lat.size < 2 or lon.size < 2:
            raise ValueError("lat/lon need at least two coordinate values")
        self._lat, self._lon = lat, lon
        self._lat_step = float(lat[1] - lat[0])
        self._lon_step = float(lon[1] - lon[0])

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> BlockExtremaCurveSource:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def axis_name(self) -> str:
        return "return period (empirical, Gringorten plotting position)"

    @property
    def return_period_support(self) -> tuple[float, float]:
        return gringorten_support(len(self._labels))

    @property
    def bounds(self) -> Bounds:
        half_lat = abs(self._lat_step) / 2
        half_lon = abs(self._lon_step) / 2
        return (
            float(self._lon.min()) - half_lon,
            float(self._lat.min()) - half_lat,
            float(self._lon.max()) + half_lon,
            float(self._lat.max()) + half_lat,
        )

    def _window(self, bounds: Bounds) -> tuple[int, int, int, int]:
        min_lon, min_lat, max_lon, max_lat = bounds
        if min_lon > max_lon or min_lat > max_lat:
            raise ValueError("bounds must be (min_lon, min_lat, max_lon, max_lat)")
        row_start, row_stop = _index_range(self._lat, min_lat, max_lat, self._lat_step)
        col_start, col_stop = _index_range(self._lon, min_lon, max_lon, self._lon_step)
        return row_start, row_stop, col_start, col_stop

    def block_array(
        self, bounds: Bounds | None = None
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """All blocks for a window as ``(values, lat, lon)``, a QA view.

        ``values`` is indexed ``[block, row, column]`` in ``labels`` order.
        Intended for small areas: it is *not* strip-bounded.
        """
        row_start, row_stop, col_start, col_stop = self._window(bounds or self.bounds)
        rows, columns = slice(row_start, row_stop), slice(col_start, col_stop)
        stack = np.stack(
            [self.reader.reduce_block(label, rows, columns) for label in self._labels]
        )
        return stack, self._lat[rows], self._lon[columns]

    @property
    def labels(self) -> tuple[int, ...]:
        return self._labels

    def iter_curves(self, bounds: Bounds | None = None) -> Iterator[RasterCurve]:
        from crc_sdk.connectors.duckdb.zarr import RasterCurve

        row_start, row_stop, col_start, col_stop = self._window(bounds or self.bounds)
        width = col_stop - col_start
        n_blocks = len(self._labels)
        # Budget for whichever is larger: the persistent per-strip block
        # array (n_blocks, strip_rows, width), or the bigger transient
        # per-block read before it is reduced away. Sizing off n_blocks alone
        # undercounts by roughly samples / n_blocks for short block ranges.
        leading_axis = max(n_blocks, int(self.reader.leading_axis_samples))
        strip_rows = _strip_row_count(
            self.strip_bytes,
            width,
            np.dtype(np.float64).itemsize,
            max(1, int(self.reader.block_height)),
            leading_axis=leading_axis,
        )
        columns = slice(col_start, col_stop)

        for row_off in range(row_start, row_stop, strip_rows):
            row_end = min(row_off + strip_rows, row_stop)
            rows = slice(row_off, row_end)
            stack = np.empty((n_blocks, row_end - row_off, width), dtype=np.float64)
            for index, label in enumerate(self._labels):
                stack[index] = self.reader.reduce_block(label, rows, columns)

            valid_count = np.isfinite(stack).sum(axis=0)
            if self.skip_sink is not None:
                for local_row, local_column in zip(
                    *(index.tolist() for index in np.where(valid_count == 0))
                ):
                    self.skip_sink(
                        row_off + local_row,
                        col_start + local_column,
                        "no_data",
                        "no valid block",
                    )
            local_rows, local_columns = np.where(valid_count > 0)
            for local_row, local_column in zip(
                local_rows.tolist(), local_columns.tolist()
            ):
                periods, values = plotting_position_curve(
                    stack[:, local_row, local_column],
                    self.tail,
                    minimum_blocks=self.minimum_blocks,
                )
                if periods.size == 0:
                    if self.skip_sink is not None:
                        self.skip_sink(
                            row_off + local_row,
                            col_start + local_column,
                            "too_few_blocks",
                            f"{int(valid_count[local_row, local_column])} valid "
                            f"blocks, need {self.minimum_blocks}",
                        )
                    continue
                source_row = row_off + local_row
                source_column = col_start + local_column
                yield RasterCurve(
                    row=source_row,
                    column=source_column,
                    boundary=pixel_boundary(
                        self._lat,
                        self._lon,
                        self._lat_step,
                        self._lon_step,
                        source_row,
                        source_column,
                    ),
                    axis_values=periods,
                    values=values,
                )
