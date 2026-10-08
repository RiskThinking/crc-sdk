from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import fit_distribution

from crc_sdk.connectors import CurveFitIngestPolicy, read_hazard_metadata
from crc_sdk.connectors.adapters import CurveSourceInfo, canonicalize_curve_source
from crc_sdk.connectors.blocks import (
    BlockDefinition,
    BlockExtremaCurveSource,
    BlockSpec,
    BlockStatistic,
    DailyAggregation,
    UnitConversion,
    aggregate_daily,
    gringorten_support,
    plotting_position_curve,
    reduce_block_samples,
)
from crc_sdk.connectors.duckdb.zarr import RasterMetadata
from crc_sdk.connectors.parquet import write_hazard_stream
from crc_sdk.types import TemporalWindow

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


def _source(
    stack: np.ndarray[Any, Any], info: CurveSourceInfo | None = None
) -> BlockExtremaCurveSource:
    reader = _GridReader(stack, tuple(range(2000, 2000 + stack.shape[0])))
    return BlockExtremaCurveSource(
        reader=reader,
        metadata=RasterMetadata(
            hazard_type="txx",
            indicator_id="txx",
            scenario="historical",
            year=2005,
            units="degC",
            path="test/txx",
        ),
        tail="upper",
        info=info,
    )


def test_curve_source_yields_one_curve_per_valid_cell() -> None:
    source = _source(_stack())
    curves = list(source.iter_curves())
    assert len(curves) == 5  # the all-NaN cell is skipped
    assert curves[0].values.size == 12
    assert source.bounds == (9.5, -0.5, 12.5, 1.5)
    stack, lat, lon = source.block_array((9.6, -0.5, 10.4, 1.5))
    assert stack.shape == (12, 2, 1) and lon.tolist() == [10.0]
    assert source.return_period_support == gringorten_support(12)


def test_too_few_blocks_are_dropped() -> None:
    stack = _stack(8)
    stack[:5, 0, 0] = np.nan  # 3 valid years left
    curves = list(_source(stack).iter_curves())
    assert len(curves) == 4


def _policy(**changes: Any) -> CurveFitIngestPolicy:
    values: dict[str, Any] = {
        "h3_resolution": 4,
        "family": "gumbel_r",
        "producer": "test",
        "tail": "upper",
        "value_semantics": "annual maximum test",
        "source_version": "v1",
        "on_fit_failure": "skip",
    }
    values.update(changes)
    return CurveFitIngestPolicy(**values)


def test_canonicalization_records_schema_13_metadata(tmp_path: Path) -> None:
    info = CurveSourceInfo(
        uri="gs://example/era5",
        licence="CC-BY-4.0",
        attribution="Contains modified test information",
        retrieved_at="2026-10-02T00:00:00+00:00",
        temporal_window=TemporalWindow(start_year=2000, end_year=2011),
        probability_semantics="annual_exceedance",
    )
    source = _source(_stack(), info)
    output = tmp_path / "txx.parquet"
    stream = canonicalize_curve_source(source, _policy(), provider="era5")
    write_hazard_stream(stream, output)
    metadata = read_hazard_metadata(output)
    assert metadata.source.uri == "gs://example/era5"
    assert metadata.source.licence == "CC-BY-4.0"
    assert metadata.probability_semantics == "annual_exceedance"
    assert metadata.temporal_window is not None
    assert metadata.temporal_window.horizon == 2005
    assert metadata.fitting is not None
    assert metadata.fitting.method == "quantile_least_squares"
    assert metadata.fitting.input_kind == "samples"
    assert pq.read_table(output).num_rows > 0


@pytest.mark.parametrize("method", ["sample_mle", "sample_lmoments"])
def test_sample_fit_and_diagnostics_sidecar(tmp_path: Path, method: str) -> None:
    sidecar = tmp_path / "diag.parquet"
    source = _source(_stack(), CurveSourceInfo())
    output = tmp_path / "txx.parquet"
    stream = canonicalize_curve_source(
        source,
        _policy(fit_method=method, diagnostics=sidecar),
        provider="era5",
    )
    write_hazard_stream(stream, output)
    metadata = read_hazard_metadata(output)
    assert metadata.fitting is not None
    assert metadata.fitting.method == method
    assert metadata.fitting.input_kind == "samples"
    assert metadata.fitting.initialization == (
        "lmoments" if method == "sample_mle" else None
    )
    rows = [
        row for row in pq.read_table(sidecar).to_pylist() if row["outcome"] == "fitted"
    ]
    assert len(rows) == 5
    assert {row["treatment"] for row in rows} == {method}

    # The canonical parameters must come from the requested sample estimator,
    # not from the plotting-position quantiles or the other sample fitter.
    stack = _stack()
    expected = []
    for row, column in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]:
        fit = fit_distribution(
            stack[:, row, column].tolist(),
            "gumbel_r",
            method="lmoments" if method == "sample_lmoments" else "default",
        ).distribution
        expected.append([fit.location, fit.scale])
    parameters = pq.read_table(output).select(["curve_location", "curve_scale"])
    actual = sorted(
        set(
            zip(
                parameters["curve_location"].to_pylist(),
                parameters["curve_scale"].to_pylist(),
            )
        )
    )
    np.testing.assert_allclose(actual, sorted(expected), rtol=1e-14)


def test_quantile_fit_diagnostics_carry_fit_quality(tmp_path: Path) -> None:
    sidecar = tmp_path / "diag.parquet"
    stream = canonicalize_curve_source(
        _source(_stack()), _policy(diagnostics=sidecar), provider="era5"
    )
    stream.read_all()
    rows = [
        row for row in pq.read_table(sidecar).to_pylist() if row["outcome"] == "fitted"
    ]
    assert rows
    assert all(row["normalized_rmse"] is not None for row in rows)
    assert all(row["maximum_absolute_residual"] is not None for row in rows)


@pytest.mark.parametrize("method", ["sample_mle", "sample_lmoments"])
def test_sample_fit_rejected_for_probability_labelled_sources(method: str) -> None:
    source = _source(_stack())
    source.values_are_samples = False
    stream = canonicalize_curve_source(source, _policy(fit_method=method), provider="x")
    with pytest.raises(ValueError, match="genuine samples"):
        stream.read_all()


@pytest.mark.parametrize("method", ["sample_mle", "sample_lmoments"])
def test_policy_rejects_sample_fit_with_quality_gates(method: str) -> None:
    with pytest.raises(ValueError, match=method):
        _policy(fit_method=method, maximum_normalized_rmse=0.1)


def test_skipped_cells_leave_a_trace(tmp_path: Path) -> None:
    stack = _stack()
    stack[:, 0, 0] = 30.0  # a constant record cannot be fitted
    sidecar = tmp_path / "diag.parquet"
    stream = canonicalize_curve_source(
        _source(stack),
        _policy(diagnostics=sidecar),
        provider="era5",
    )
    stream.read_all()
    outcomes = [row["outcome"] for row in pq.read_table(sidecar).to_pylist()]
    assert outcomes.count("skipped") >= 1 and "fitted" in outcomes


def test_source_level_drops_are_traced(tmp_path: Path) -> None:
    stack = _stack(8)
    stack[:5, 0, 0] = np.nan  # 3 valid blocks: too few
    sidecar = tmp_path / "diag.parquet"
    stream = canonicalize_curve_source(
        _source(stack), _policy(diagnostics=sidecar), provider="era5"
    )
    stream.read_all()
    rows = pq.read_table(sidecar).to_pylist()
    reasons = sorted(row["reason"] for row in rows if row["outcome"] == "skipped")
    assert reasons == ["no_data", "too_few_blocks"]
    assert len(rows) == 6  # every cell of the 2x3 grid is accounted for


def test_lmoments_rejects_unsupported_family() -> None:
    with pytest.raises(ValueError, match="supports"):
        _policy(family="genpareto", fit_method="sample_lmoments")


def test_lmoments_reports_old_backend_instead_of_skipping_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def legacy_fit(values: Any, *, family: str) -> Any:
        raise AssertionError("the old backend cannot accept method=lmoments")

    monkeypatch.setattr("crc_sdk.connectors.adapters.fit_distribution", legacy_fit)
    stream = canonicalize_curve_source(
        _source(_stack()), _policy(fit_method="sample_lmoments"), provider="era5"
    )
    with pytest.raises(RuntimeError, match="requires crc-framework"):
        stream.read_all()


@pytest.mark.parametrize("shape, endpoint", [(0.5, 14.0), (-0.5, 6.0), (0.0, None)])
def test_gev_support_check(shape: float, endpoint: float | None) -> None:
    from crc_framework import FittedDistribution

    from crc_sdk.connectors.adapters import _validate_sample_fit

    fit = FittedDistribution("genextreme", location=10, scale=2, shape=shape)
    policy = _policy(
        family="genextreme", fit_method="sample_lmoments", require_sample_support=True
    )
    values = np.array([8.0, 10.0, 12.0] if endpoint is None else [10.0, endpoint])
    _validate_sample_fit(fit, values, policy)
    if endpoint is not None:
        # Permit endpoint roundoff, but reject a material support violation.
        _validate_sample_fit(
            fit,
            np.array([np.nextafter(endpoint, np.inf if shape > 0 else -np.inf)]),
            policy,
        )
        with pytest.raises(ValueError, match="outside fitted GEV support"):
            _validate_sample_fit(
                fit, np.array([endpoint + (0.01 if shape > 0 else -0.01)]), policy
            )
    else:
        _validate_sample_fit(fit, np.array([-1e100, 1e100]), policy)


@pytest.mark.parametrize("tail", ["upper", "lower"])
def test_sample_return_level_bounds_use_selected_tail(tail: str) -> None:
    from crc_framework import FittedDistribution

    from crc_sdk.connectors.adapters import _validate_sample_fit

    fit = FittedDistribution("gumbel_r", location=0, scale=1)
    policy = _policy(
        fit_method="sample_lmoments",
        tail=tail,
        validation_return_periods=(100,),
        minimum_return_value=0 if tail == "lower" else None,
        maximum_return_value=4 if tail == "upper" else None,
    )
    with pytest.raises(ValueError, match="return levels"):
        _validate_sample_fit(fit, np.array([-1, 0, 1, 2]), policy)


@pytest.mark.parametrize("period", [0, 1, float("nan"), float("inf")])
def test_sample_checks_reject_invalid_periods(period: float) -> None:
    with pytest.raises(ValueError, match="validation_return_periods"):
        _policy(fit_method="sample_lmoments", validation_return_periods=(period,))


def test_sample_checks_reject_incompatible_options() -> None:
    with pytest.raises(ValueError, match="sample support checks"):
        _policy(require_sample_support=True)
    with pytest.raises(ValueError, match="require validation_return_periods"):
        _policy(fit_method="sample_lmoments", minimum_return_value=0)
    with pytest.raises(ValueError, match="exceeds"):
        _policy(
            fit_method="sample_lmoments",
            validation_return_periods=(2,),
            minimum_return_value=10,
            maximum_return_value=0,
        )
    with pytest.raises(ValueError, match="requires diagnostics"):
        _policy(
            family="genextreme",
            fit_method="sample_lmoments",
            fallback_family="gumbel_r",
        )


@pytest.mark.parametrize("fallback_succeeds", [True, False])
def test_gev_fallback_is_audited_and_obeys_same_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback_succeeds: bool
) -> None:
    from types import SimpleNamespace

    from crc_framework import FittedDistribution

    calls: list[str] = []

    def fake_fit(values: Any, *, family: str, method: str) -> Any:
        calls.append(family)
        assert method == "lmoments"
        fit = (
            FittedDistribution("genextreme", location=0, scale=1, shape=1)
            if family == "genextreme"
            else FittedDistribution(
                "gumbel_r", location=30 if fallback_succeeds else -100, scale=2
            )
        )
        return SimpleNamespace(distribution=fit)

    monkeypatch.setattr("crc_sdk.connectors.adapters.fit_distribution", fake_fit)
    stack = _stack()
    stack[:, 0, 0] = 0  # A degenerate record must never trigger fallback.
    sidecar = tmp_path / "diag.parquet"
    stream = canonicalize_curve_source(
        _source(stack, CurveSourceInfo()),
        _policy(
            family="genextreme",
            fit_method="sample_lmoments",
            require_sample_support=True,
            validation_return_periods=(2, 100),
            minimum_return_value=0,
            fallback_family="gumbel_r",
            diagnostics=sidecar,
        ),
        provider="era5",
    )
    table = stream.read_all()
    assert calls == ["genextreme", "gumbel_r"] * 4
    metadata = stream.metadata.fitting
    assert metadata is not None
    assert metadata.families == ("genextreme", "gumbel_r")
    assert metadata.selection_metric == "first_acceptable"
    assert metadata.require_sample_support and metadata.minimum_return_value == 0
    rows = pq.read_table(sidecar).to_pylist()
    attempted = [row for row in rows if len(row["attempted_families"]) == 2]
    assert len(attempted) == 4
    assert all(
        row["failed_families"]
        == (["genextreme"] if fallback_succeeds else ["genextreme", "gumbel_r"])
        for row in attempted
    )
    if fallback_succeeds:
        assert table.num_rows > 0
        assert set(table["curve_type"].to_pylist()) == {"gumbel_r"}
        assert all(row["fallback"] and "support" in row["message"] for row in attempted)
    else:
        assert table.num_rows == 0
        assert all(
            row["outcome"] == "skipped" and "minimum_return_value" in row["message"]
            for row in attempted
        )


def test_return_level_check_rejects_nonfinite_levels() -> None:
    from crc_framework import FittedDistribution

    from crc_sdk.connectors.adapters import _validate_sample_fit

    fit = FittedDistribution("gumbel_r", location=0, scale=1e308)
    with pytest.raises(ValueError, match="nonfinite"):
        _validate_sample_fit(
            fit,
            np.array([0, 1, 2, 3]),
            _policy(fit_method="sample_lmoments", validation_return_periods=(100,)),
        )


@pytest.mark.parametrize("on_failure", ["skip", "raise"])
@pytest.mark.parametrize("tail", ["upper", "lower"])
def test_quantile_return_level_gates_are_enforced(
    tmp_path: Path, on_failure: str, tail: Literal["upper", "lower"]
) -> None:
    sidecar = tmp_path / "qls-diagnostics.parquet"
    source = _source(_stack())
    source.tail = tail
    stream = canonicalize_curve_source(
        source,
        _policy(
            validation_return_periods=(2, 10),
            maximum_return_value=0,
            tail=tail,
            diagnostics=sidecar,
            on_fit_failure=on_failure,
        ),
        provider="test",
    )
    if on_failure == "raise":
        with pytest.raises(ValueError, match="above maximum_return_value"):
            stream.read_all()
        assert not sidecar.exists()
        assert not Path(str(sidecar) + ".partial").exists()
    else:
        assert stream.read_all().num_rows == 0
        rows = pq.read_table(sidecar).to_pylist()
        failures = [row for row in rows if row["reason"] == "fit_failed"]
        assert len(failures) == 5
        assert all("above maximum_return_value" in row["message"] for row in failures)


def test_quantile_return_level_gates_accept_valid_fit() -> None:
    stream = canonicalize_curve_source(
        _source(_stack(), CurveSourceInfo()),
        _policy(
            validation_return_periods=(2, 10),
            minimum_return_value=0,
            maximum_return_value=100,
        ),
        provider="test",
    )
    assert stream.read_all().num_rows > 0
    assert stream.metadata.fitting is not None
    assert stream.metadata.fitting.validation_return_periods == (2, 10)
    assert stream.metadata.fitting.minimum_return_value == 0


def test_quantile_return_level_gate_checks_complete_hurdle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from crc_framework import (
        FittedDistribution,
        HurdleDistribution,
        TabulatedDistribution,
    )

    from crc_sdk.connectors.adapters import HurdleFitPolicy, _fit_curve

    base = FittedDistribution("gumbel_r", location=30, scale=2)
    hurdle = HurdleDistribution(base, atom_probability=0.75, atom_location=-1)
    monkeypatch.setattr(
        "crc_sdk.connectors.adapters.fit_hurdle_quantiles",
        lambda *args, **kwargs: SimpleNamespace(
            distribution=hurdle,
            diagnostics=SimpleNamespace(tail=SimpleNamespace(converged=True)),
        ),
    )
    knots = TabulatedDistribution.from_return_periods(
        [2, 5, 10, 20], [-1, 30, 35, 40], tail="upper"
    )
    with pytest.raises(ValueError, match="below minimum_return_value"):
        _fit_curve(
            knots,
            _policy(
                hurdle=HurdleFitPolicy(0.75, -1),
                validation_return_periods=(2,),
                minimum_return_value=0,
            ),
        )


@pytest.mark.parametrize("values_are_samples", [False, True])
def test_quantile_provenance_records_original_source_kind(
    tmp_path: Path, values_are_samples: bool
) -> None:
    source = _source(_stack(), CurveSourceInfo())
    source.values_are_samples = values_are_samples
    output = tmp_path / "quantile-fit.parquet"
    write_hazard_stream(
        canonicalize_curve_source(source, _policy(), provider="test"), output
    )
    metadata = read_hazard_metadata(output)
    assert metadata.fitting is not None
    assert metadata.fitting.method == "quantile_least_squares"
    assert metadata.fitting.input_kind == (
        "samples" if values_are_samples else "probability_labelled"
    )
    assert metadata.fitting.initialization is None
