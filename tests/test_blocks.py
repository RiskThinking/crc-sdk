from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest

from crc_sdk._connectors.blocks import (
    BlockDefinition,
    BlockSpec,
    BlockStatistic,
    DailyAggregation,
    UnitConversion,
    aggregate_daily,
    gringorten_support,
    plotting_position_curve,
    reduce_block_samples,
)
from crc_sdk._connectors.duckdb.zarr import RasterMetadata

D = np.datetime64


def test_block_spans() -> None:
    assert BlockDefinition.annual().span(2001) == (D("2001-01-01"), D("2002-01-01"))
    assert BlockDefinition.water_year().span(2001) == (
        D("2000-10-01"),
        D("2001-10-01"),
    )
    # DJF is labelled by the year it ends in, and wraps the new year.
    assert BlockDefinition.season((12, 1, 2)).span(2001) == (
        D("2000-12-01"),
        D("2001-03-01"),
    )
    assert BlockDefinition.season((6, 7, 8)).span(2001) == (
        D("2001-06-01"),
        D("2001-09-01"),
    )
    assert BlockDefinition.season((10, 11, 12)).span(2001) == (
        D("2001-10-01"),
        D("2002-01-01"),
    )


def test_block_definition_validation() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        BlockDefinition.season((1, 3))
    with pytest.raises(ValueError, match="full year"):
        BlockDefinition.season(tuple(range(1, 13)))


def test_statistic_validation() -> None:
    with pytest.raises(ValueError, match="rolling"):
        BlockStatistic("max", window_days=5)
    with pytest.raises(ValueError, match="threshold"):
        BlockStatistic("count_above")
    assert BlockStatistic.minimum().tail == "lower"
    assert BlockStatistic.count_above(30.0).tail == "upper"
    with pytest.raises(ValueError, match="daily aggregation"):
        BlockSpec(statistic=BlockStatistic.k_day_sum(3))


def _hourly(days: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    times = D("2001-01-01T00", "s") + np.arange(days * 24) * np.timedelta64(1, "h")
    return times, np.zeros((days * 24, 1, 2))


def test_daily_max_and_completeness() -> None:
    times, values = _hourly(3)
    values[5, 0, 0] = 7.0  # day 0
    values[24 + 3, 0, 0] = 9.0  # day 1
    values[48 + 20 :, 0, 1] = np.nan  # pixel 1 loses day 2's late hours
    out = aggregate_daily(
        times,
        values,
        DailyAggregation("max", min_valid_fraction=0.9),
        D("2001-01-01"),
        3,
    )
    assert out[:, 0, 0].tolist() == [7.0, 9.0, 0.0]
    # 20 of 24 hours valid on day 2 is below 0.9
    assert np.isnan(out[2, 0, 1]) and out[0, 0, 1] == 0.0


def test_precipitation_interval_end_stamps_belong_to_the_ended_day() -> None:
    times, values = _hourly(2)
    values[0, 0, 0] = 0.0
    values[12, 0, 0] = 1.0  # 12:00 stamp -> Jan 1
    values[24, 0, 0] = 2.0  # 00:00 stamp on Jan 2 -> the hour ending Jan 1
    daily = DailyAggregation("sum", min_valid_fraction=0.0, interval_end_stamps=True)
    out = aggregate_daily(times, values, daily, D("2001-01-01"), 2)
    assert out[0, 0, 0] == 3.0
    utc = aggregate_daily(
        times,
        values,
        DailyAggregation("sum", min_valid_fraction=0.0),
        D("2001-01-01"),
        2,
    )
    assert utc[0, 0, 0] == 1.0 and utc[1, 0, 0] == 2.0


def test_utc_offset_moves_the_day_boundary() -> None:
    times, values = _hourly(2)
    values[23, 0, 0] = 5.0  # 23:00 UTC
    local = DailyAggregation("max", min_valid_fraction=0.0, utc_offset_hours=2)
    out = aggregate_daily(times, values, local, D("2001-01-01"), 2)
    assert out[0, 0, 0] == 0.0 and out[1, 0, 0] == 5.0  # 01:00 local next day


def test_reduce_block_statistics() -> None:
    year = BlockDefinition.annual()
    days = 365
    times = D("2001-01-01", "s") + np.arange(days) * np.timedelta64(1, "D")
    values = np.zeros((days, 1, 1))
    values[10:13, 0, 0] = [4.0, 6.0, 5.0]
    span = year.span(2001)

    def run(statistic: BlockStatistic) -> float:
        spec = BlockSpec(
            statistic=statistic,
            daily=DailyAggregation("max", step_hours=24, min_valid_fraction=0.0),
        )
        return float(reduce_block_samples(values, spec, times=times, span=span)[0, 0])

    assert run(BlockStatistic.maximum()) == 6.0
    assert run(BlockStatistic.k_day_sum(3)) == 15.0
    assert run(BlockStatistic.k_day_mean(3)) == 5.0
    assert run(BlockStatistic.count_above(4.5)) == 2.0
    assert run(BlockStatistic.count_below(0.5)) == 362.0


def test_unit_conversion_and_block_completeness() -> None:
    spec = BlockSpec(
        statistic=BlockStatistic.maximum(),
        conversion=UnitConversion(offset=-273.15),
        min_valid_fraction=0.75,
    )
    values = np.full((4, 1, 2), 283.15)
    values[1:, 0, 1] = np.nan  # pixel 1 keeps 1 of 4 samples (< 75%)
    out = reduce_block_samples(values, spec)
    assert out[0, 0] == pytest.approx(10.0) and np.isnan(out[0, 1])


def test_plotting_positions_orient_by_tail() -> None:
    samples = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    periods, values = plotting_position_curve(samples, "upper")
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.all(np.diff(periods) > 0)  # more severe (larger) -> longer period
    assert periods[-1] == pytest.approx(gringorten_support(5)[1])
    periods, values = plotting_position_curve(samples, "lower")
    assert np.all(np.diff(periods) < 0)  # more severe (smaller) -> longer period
    assert plotting_position_curve(samples[:3], "upper")[0].size == 0


class _GridReader:
    """In-memory `BlockReader`: per-year arrays over a small lat/lon grid."""

    def __init__(self, stack: np.ndarray[Any, Any], years: tuple[int, ...]) -> None:
        self.stack = stack
        self._years = years
        self.lat = np.array([1.0, 0.0])
        self.lon = np.array([10.0, 11.0, 12.0])
        self.calls = 0

    labels = property(lambda self: self._years)
    leading_axis_samples = 1
    block_height = 1

    def reduce_block(self, label: int, rows: slice, columns: slice) -> Any:
        self.calls += 1
        return self.stack[self._years.index(label), rows, columns]

    def close(self) -> None:
        return None


def _stack(years: int = 12) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(3)
    base = rng.gumbel(30.0, 2.0, size=(years, 2, 3))
    base[:, 1, 2] = np.nan  # a cell with no data at all
    return base


class _TailedSource:
    """Minimal curve source: one tail, a skip sink, no curves."""

    axis_name = "return period (empirical)"
    values_are_samples = True

    def __init__(self, tail: Literal["upper", "lower"]) -> None:
        self.tail = tail
        self.metadata = RasterMetadata("Heat", "txx", "historical", 2000, "degC", "x")
        self.skip_sink: Any = None

    def iter_curves(self, bounds: Any = None) -> Any:
        if self.skip_sink is not None:
            self.skip_sink(0, 0, "no_data", "nothing here")
        return iter(())
