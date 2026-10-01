"""Flexible probability axes, bounds, eligibility, guards and diagnostics."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution

from crc_sdk import registry as hazard_registry
from crc_sdk._version import sdk_version
from crc_sdk.connectors import read_hazard_metadata, write_hazard_stream
from crc_sdk.fitting import (
    CDFColumnSchema,
    CDFCurveFitPolicy,
    NoDataRateError,
    NoDataRateWarning,
    fit_cdf_quantile_batches,
)
from crc_sdk.types import CurveParameters, SourceProvenance, TemporalWindow
from crc_sdk.workflows import return_periods_to_probabilities


def _policy(**overrides: object) -> CDFCurveFitPolicy:
    values: dict[str, object] = {
        "h3_resolution": 5,
        "family": "gumbel_r",
        "value_unit": "mm",
        "value_semantics": "annual maximum",
        "producer": "tests",
        "source": SourceProvenance(provider="fixture", dataset="x"),
        "source_id": "fixture",
        "prefetch": False,
        "max_workers": 1,
    }
    values.update(overrides)
    return CDFCurveFitPolicy(**values)  # type: ignore[arg-type]


def _table(
    quantiles: list[list[float]],
    *,
    hazard: str = "rx1day",
    axis: list[list[float]] | None = None,
    axis_name: str = "axis",
    quantile_name: str = "cdf_quantiles",
    quantile_type: pa.DataType | None = None,
) -> pa.Table:
    n = len(quantiles)
    data: dict[str, object] = {
        "hex_id": pa.array(
            range(599024279241097215, 599024279241097215 + n), pa.uint64()
        ),
        "index_name": [hazard] * n,
        "year": pa.array([2050] * n, pa.int32()),
        "pathway": ["ssp245"] * n,
        quantile_name: pa.array(quantiles, quantile_type or pa.list_(pa.float64())),
    }
    if axis is not None:
        data[axis_name] = pa.array(axis, pa.list_(pa.float64()))
    return pa.table(data)


GUMBEL = FittedDistribution.from_parameters("gumbel_r", location=20.0, scale=5.0)
P11 = np.linspace(0.0, 1.0, 11)


def _gumbel_quantiles(probabilities: np.ndarray) -> list[float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return [float(GUMBEL.quantiles([p])[0]) for p in clipped]


def test_default_creation_version_is_installed_sdk_version() -> None:
    assert _policy().creation_version == sdk_version()
    result = fit_cdf_quantile_batches(
        _table([_gumbel_quantiles(P11)]), P11.tolist(), _policy()
    )
    assert result.stream.metadata.creation_version == sdk_version()
    assert result.stream.metadata.schema_version == "1.3"


def test_per_row_probability_axes_match_shared_axis() -> None:
    other = np.linspace(0.0, 1.0, 21)
    shared = list(
        fit_cdf_quantile_batches(
            _table([_gumbel_quantiles(P11)]), P11.tolist(), _policy()
        ).stream.batches
    )[0].hazard_rows.to_pylist()[0]
    # Rows given unsorted, on two different axes, in one batch.
    order = np.arange(len(P11))[::-1]
    table = _table(
        [
            [_gumbel_quantiles(P11)[i] for i in order],
            _gumbel_quantiles(other),
        ],
        axis=[[float(P11[i]) for i in order], other.tolist()],
    )
    columns = CDFColumnSchema(probabilities="axis")
    result = fit_cdf_quantile_batches(table, None, _policy(), columns=columns)
    rows = pa.concat_tables([b.hazard_rows for b in result.stream.batches]).to_pylist()
    assert rows[0]["curve_location"] == pytest.approx(
        shared["curve_location"], rel=1e-6
    )
    assert rows[0]["curve_scale"] == pytest.approx(shared["curve_scale"], rel=1e-6)
    assert rows[1]["curve_location"] == pytest.approx(20.0, rel=0.02)


def test_axis_requirements_are_validated() -> None:
    table = _table([_gumbel_quantiles(P11)])
    with pytest.raises(ValueError, match="probabilities are required"):
        fit_cdf_quantile_batches(table, None, _policy())
    with pytest.raises(ValueError, match="probabilities=None"):
        fit_cdf_quantile_batches(
            table, P11.tolist(), _policy(), columns=CDFColumnSchema(samples="s")
        )
    with pytest.raises(ValueError, match="at most one"):
        CDFColumnSchema(probabilities="a", return_periods="b")


def test_return_period_axis_uses_explicit_convention(tmp_path: Path) -> None:
    periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    values = [10.0, 14.0, 17.0, 21.0, 24.0, 27.0, 31.0, 34.0, 37.0]
    table = _table([values], axis=[periods])
    columns = CDFColumnSchema(return_periods="axis")
    fits = {}
    for convention in ("one_minus_inverse", "poisson"):
        result = fit_cdf_quantile_batches(
            table,
            None,
            _policy(return_period_convention=convention),
            columns=columns,
        )
        fits[convention] = list(result.stream.batches)[0].hazard_rows.to_pylist()[0]
        assert result.stream.metadata.source_return_period_convention == convention
    assert fits["one_minus_inverse"]["curve_location"] != pytest.approx(
        fits["poisson"]["curve_location"]
    )
    # The same input expressed as probabilities fits identically.
    probabilities = list(return_periods_to_probabilities(periods, convention="poisson"))
    direct = fit_cdf_quantile_batches(
        _table([values], axis=[probabilities]),
        None,
        _policy(),
        columns=CDFColumnSchema(probabilities="axis"),
    )
    row = list(direct.stream.batches)[0].hazard_rows.to_pylist()[0]
    assert row["curve_location"] == pytest.approx(fits["poisson"]["curve_location"])


def test_poisson_return_period_probabilities() -> None:
    upper = return_periods_to_probabilities([10.0], convention="poisson")
    assert upper == pytest.approx((np.exp(-0.1),))
    lower = return_periods_to_probabilities([10.0], tail="lower", convention="poisson")
    assert lower == pytest.approx((1.0 - np.exp(-0.1),))
    with pytest.raises(ValueError, match="convention"):
        return_periods_to_probabilities([10.0], convention="bogus")  # type: ignore[arg-type]


def test_unlabelled_or_mismatched_rows_are_rejected_with_a_diagnostic(
    tmp_path: Path,
) -> None:
    periods = [2.0, 5.0, 10.0, 25.0, 50.0]
    good = [10.0, 14.0, 17.0, 21.0, 24.0]
    table = _table(
        [good, good, good],
        axis=[periods, [], periods[:3]],
    )
    columns = CDFColumnSchema(return_periods="axis")
    sidecar = tmp_path / "diag.parquet"
    result = fit_cdf_quantile_batches(
        table,
        None,
        _policy(on_fit_failure="skip", diagnostics=sidecar),
        columns=columns,
    )
    rows = pa.concat_tables([b.hazard_rows for b in result.stream.batches])
    assert rows.num_rows == 1
    assert result.summary.rejected_rows == 2
    assert dict(result.summary.rejection_reasons) == {
        "missing_axis_labels": 1,
        "axis_length_mismatch": 1,
    }
    diagnostics = pq.read_table(sidecar).to_pylist()
    assert [d["outcome"] for d in diagnostics] == ["fitted", "rejected", "rejected"]
    assert [d["reason"] for d in diagnostics][1:] == [
        "missing_axis_labels",
        "axis_length_mismatch",
    ]
    assert not Path(f"{sidecar}.partial").exists()

    with pytest.raises(ValueError, match="failed to fit CDF row 1"):
        list(
            fit_cdf_quantile_batches(
                table, None, _policy(), columns=columns
            ).stream.batches
        )


def test_short_axis_rows_follow_policy() -> None:
    periods = [10.0, 100.0, 1000.0]
    table = _table([[5.0, 9.0, 14.0]], axis=[periods])
    columns = CDFColumnSchema(return_periods="axis")
    result = fit_cdf_quantile_batches(table, None, _policy(), columns=columns)
    row = list(result.stream.batches)[0].hazard_rows.to_pylist()[0]
    assert row["curve_kind"] == "tabulated"
    assert row["curve_probabilities"] == pytest.approx([0.9, 0.99, 0.999])
    assert result.summary.treatment_counts["tabulated_short_axis"] == 1

    with pytest.raises(ValueError, match="short_axis|at least 4"):
        list(
            fit_cdf_quantile_batches(
                table, None, _policy(short_axis_action="reject"), columns=columns
            ).stream.batches
        )
    # A shared axis shorter than four knots is handled the same way.
    shared = fit_cdf_quantile_batches(
        _table([[1.0, 2.0, 3.0, 4.0, 5.0]]), [0.0, 0.25, 0.5, 0.75, 1.0], _policy()
    )
    assert list(shared.stream.batches)[0].hazard_rows["curve_kind"].to_pylist() == [
        "tabulated"
    ]


def test_samples_are_resampled_and_recorded() -> None:
    rng = np.random.default_rng(3)
    samples = rng.gumbel(20.0, 5.0, size=400).tolist()
    table = _table([samples], quantile_name="samples")
    result = fit_cdf_quantile_batches(
        table,
        None,
        _policy(sample_quantile_count=201),
        columns=CDFColumnSchema(samples="samples"),
    )
    row = list(result.stream.batches)[0].hazard_rows.to_pylist()[0]
    assert row["curve_type"] == "gumbel_r"
    assert row["curve_location"] == pytest.approx(20.0, abs=1.5)
    fitting = result.stream.metadata.fitting
    assert fitting is not None
    assert fitting.input_kind == "samples"
    assert fitting.sample_resampling == 201
    assert result.stream.metadata.source_probability_support == (0.005, 0.995)


def test_lower_bound_censors_like_the_reference_zero_cap() -> None:
    # Gumbel with substantial mass below zero.
    probabilities = np.linspace(0.0, 1.0, 101)
    base = FittedDistribution.from_parameters("gumbel_r", location=1.0, scale=4.0)
    raw = [
        float(base.quantiles([p])[0]) for p in np.clip(probabilities, 1e-4, 1 - 1e-4)
    ]
    table = _table([raw], hazard="rflood")
    result = fit_cdf_quantile_batches(
        table, probabilities.tolist(), _policy(lower_bound=0.0)
    )
    row = list(result.stream.batches)[0].hazard_rows.to_pylist()[0]
    clipped = np.maximum(raw, 0.0)
    reference = fit_cdf_quantile_batches(
        _table([clipped.tolist()], hazard="rflood"),
        probabilities.tolist(),
        _policy(),
    )
    reference_row = list(reference.stream.batches)[0].hazard_rows.to_pylist()[0]
    # The input is censored before fitting, exactly as the flow did.
    for name in ("curve_kind", "curve_type", "curve_atom_probability"):
        assert row[name] == reference_row[name]
    assert row["curve_kind"] == "hurdle"
    assert row["curve_atom_location"] == 0.0
    assert result.summary.hurdle_rows == 1


def test_lower_bound_turns_an_all_negative_fit_into_a_point_mass() -> None:
    # Fit succeeds on negative values; with the bound censoring them to zero
    # the row is the constant zero.
    probabilities = np.linspace(0.0, 1.0, 11)
    result = fit_cdf_quantile_batches(
        _table([np.linspace(-9.0, -1.0, 11).tolist()], hazard="rflood"),
        probabilities.tolist(),
        _policy(lower_bound=0.0),
    )
    row = list(result.stream.batches)[0].hazard_rows.to_pylist()[0]
    assert (row["curve_kind"], row["curve_location"]) == ("point_mass", 0.0)


def test_fitted_mass_below_the_bound_becomes_an_atom() -> None:
    # Positive-only knots whose fitted Gumbel nonetheless puts mass below 0.
    probabilities = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    values = [0.2, 0.3, 0.9, 1.6, 2.8, 6.0, 9.0]
    plain = fit_cdf_quantile_batches(
        _table([values]), probabilities.tolist(), _policy(atom_policy="none")
    )
    plain_row = list(plain.stream.batches)[0].hazard_rows.to_pylist()[0]
    bounded = fit_cdf_quantile_batches(
        _table([values]),
        probabilities.tolist(),
        _policy(atom_policy="none", lower_bound=0.0),
    )
    bounded_row = list(bounded.stream.batches)[0].hazard_rows.to_pylist()[0]
    base = FittedDistribution.from_parameters(
        "gumbel_r",
        location=plain_row["curve_location"],
        scale=plain_row["curve_scale"],
    )
    mass = float(base.cdf(0.0))
    assert mass > 0.0
    assert bounded_row["curve_kind"] == "hurdle"
    assert bounded_row["curve_atom_probability"] == pytest.approx(mass)
    assert bounded.summary.treatment_counts["bound_censored"] == 1


def test_registry_and_overrides_set_eligibility() -> None:
    probabilities = np.linspace(0.0, 1.0, 11)
    negative = np.linspace(-8.0, -1.0, 11).tolist()
    negative[0] = -8.0
    table = _table([negative], hazard="TNn")

    def kind(policy: CDFCurveFitPolicy) -> str:
        result = fit_cdf_quantile_batches(table, probabilities.tolist(), policy)
        return str(list(result.stream.batches)[0].hazard_rows["curve_kind"][0])

    floor = _policy(minimum_informative_value=1.0)
    assert kind(floor) == "no_data"
    assert (
        kind(
            _policy(
                minimum_informative_value=1.0,
                registry=hazard_registry.public_registry(),
            )
        )
        != "no_data"
    )
    assert (
        kind(
            _policy(
                minimum_informative_value=1.0,
                minimum_informative_overrides={"TNn": None},
            )
        )
        != "no_data"
    )
    # An explicit override wins over the registry.
    assert (
        kind(
            _policy(
                registry=hazard_registry.public_registry(),
                minimum_informative_overrides={"TNn": 1.0},
            )
        )
        == "no_data"
    )


def test_no_data_guard_warns_then_raises_under_strict() -> None:
    probabilities = np.linspace(0.0, 1.0, 11)
    rows = [np.linspace(0.0, 0.5, 11).tolist()] * 20
    table = _table(rows, hazard="dtr")
    kwargs = dict(minimum_informative_value=1.0, no_data_min_rows=10)
    with pytest.warns(NoDataRateWarning, match="dtr"):
        result = fit_cdf_quantile_batches(
            table, probabilities.tolist(), _policy(**kwargs)
        )
        list(result.stream.batches)
    assert result.summary.no_data_by_hazard["dtr"] == 20
    with pytest.raises(NoDataRateError):
        list(
            fit_cdf_quantile_batches(
                table, probabilities.tolist(), _policy(strict=True, **kwargs)
            ).stream.batches
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(
            fit_cdf_quantile_batches(
                table,
                probabilities.tolist(),
                _policy(no_data_rate_threshold=None, **kwargs),
            ).stream.batches
        )


def test_diagnostics_sidecar_records_every_outcome(tmp_path: Path) -> None:
    probabilities = np.linspace(0.0, 1.0, 11)
    rows = [
        _gumbel_quantiles(probabilities),
        [3.0] * 11,
        np.linspace(0.0, 0.5, 11).tolist(),
    ]
    sidecar = tmp_path / "nested" / "diag.parquet"
    result = fit_cdf_quantile_batches(
        _table(rows),
        probabilities.tolist(),
        _policy(minimum_informative_value=1.0, diagnostics=sidecar),
    )
    output = tmp_path / "out.parquet"
    write_hazard_stream(result.stream, output, max_workers=1)
    diagnostics = pq.read_table(sidecar).to_pylist()
    assert [d["outcome"] for d in diagnostics] == ["fitted", "point_mass", "no_data"]
    assert diagnostics[0]["curve_type"] == "gumbel_r"
    assert diagnostics[0]["normalized_rmse"] is not None
    assert diagnostics[2]["reason"] == "below_effective_resolution"
    assert result.summary.diagnostics_rows == 3
    assert read_hazard_metadata(output).fitting is not None

    exceptions = tmp_path / "exceptions.parquet"
    list(
        fit_cdf_quantile_batches(
            _table(rows),
            probabilities.tolist(),
            _policy(
                minimum_informative_value=1.0,
                diagnostics=exceptions,
                diagnostics_rows="exceptions",
            ),
        ).stream.batches
    )
    assert [d["outcome"] for d in pq.read_table(exceptions).to_pylist()] == ["no_data"]


def test_failed_stream_leaves_no_published_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "diag.parquet"
    probabilities = np.linspace(0.0, 1.0, 11)
    result = fit_cdf_quantile_batches(
        _table([[float("nan")] * 11]),
        probabilities.tolist(),
        _policy(diagnostics=sidecar),
    )
    with pytest.raises(ValueError):
        list(result.stream.batches)
    assert not sidecar.exists()
    assert not Path(f"{sidecar}.partial").exists()


def test_temporal_window_is_carried_into_metadata(tmp_path: Path) -> None:
    window = TemporalWindow(start_year=2041, end_year=2060)
    assert window.horizon == 2050
    probabilities = np.linspace(0.0, 1.0, 11)
    result = fit_cdf_quantile_batches(
        _table([_gumbel_quantiles(probabilities)]),
        probabilities.tolist(),
        _policy(
            temporal_window=window, probability_semantics="annual_value_distribution"
        ),
    )
    output = tmp_path / "out.parquet"
    write_hazard_stream(result.stream, output, max_workers=1)
    metadata = read_hazard_metadata(output)
    assert metadata.temporal_window == window
    assert metadata.probability_semantics == "annual_value_distribution"


def test_schema_1_2_output_matches_1_3_curves_and_rejects_1_3_only_options(
    tmp_path: Path,
) -> None:
    probabilities = np.linspace(0.0, 1.0, 11)
    table = _table([_gumbel_quantiles(probabilities)])
    outputs = {}
    for version in ("1.2", "1.3"):
        result = fit_cdf_quantile_batches(
            table, probabilities.tolist(), _policy(schema_version=version)
        )
        path = tmp_path / f"{version}.parquet"
        write_hazard_stream(result.stream, path, max_workers=1)
        outputs[version] = path
    legacy = read_hazard_metadata(outputs["1.2"])
    assert legacy.schema_version == "1.2"
    assert legacy.fitting is not None
    assert legacy.fitting.platform is None and legacy.fitting.input_kind is None
    keys = ["curve_kind", "curve_type", "curve_location", "curve_scale"]
    assert (
        pq.read_table(outputs["1.2"], columns=keys).to_pylist()
        == pq.read_table(outputs["1.3"], columns=keys).to_pylist()
    )
    with pytest.raises(ValueError, match="require schema_version='1.3'"):
        _policy(schema_version="1.2", lower_bound=0.0)
    with pytest.raises(ValueError, match="require schema_version='1.3'"):
        _policy(schema_version="1.2", probability_semantics="annual_exceedance")


def test_override_keyed_by_canonical_name_applies_to_aliases() -> None:
    probabilities = np.linspace(0.0, 1.0, 11)
    negative = np.linspace(-8.0, -1.0, 11).tolist()
    table = _table([negative], hazard="TNn")  # alias of the registry's "tnn"

    def kind(**overrides: object) -> str:
        policy = _policy(registry=hazard_registry.public_registry(), **overrides)
        result = fit_cdf_quantile_batches(table, probabilities.tolist(), policy)
        return str(list(result.stream.batches)[0].hazard_rows["curve_kind"][0])

    # Registry says "no floor"; a canonical-name override must beat it.
    assert kind() != "no_data"
    assert kind(minimum_informative_overrides={"tnn": 1.0}) == "no_data"
    # The row's own label still wins over the canonical one.
    assert kind(minimum_informative_overrides={"tnn": 1.0, "TNn": None}) != "no_data"


def test_integer_sample_lists_are_accepted() -> None:
    samples = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    table = _table(samples, quantile_name="samples", quantile_type=pa.list_(pa.int64()))
    result = fit_cdf_quantile_batches(
        table,
        None,
        _policy(sample_quantile_count=11),
        columns=CDFColumnSchema(samples="samples"),
    )
    assert len(list(result.stream.batches)[0].hazard_rows) == 1


def test_bound_censoring_preserves_base_quantiles() -> None:
    """The censored hurdle is max(bound, base quantile) for every probability."""
    probabilities = np.linspace(0.0, 1.0, 101)
    base = FittedDistribution.from_parameters("gumbel_r", location=1.0, scale=4.0)
    raw = [
        float(base.quantiles([p])[0]) for p in np.clip(probabilities, 1e-4, 1 - 1e-4)
    ]
    result = fit_cdf_quantile_batches(
        _table([raw]),
        probabilities.tolist(),
        _policy(atom_policy="none", lower_bound=0.0),
    )
    row = list(result.stream.batches)[0].hazard_rows.to_pylist()[0]
    assert row["curve_kind"] == "hurdle"
    fitted = FittedDistribution.from_parameters(
        row["curve_type"],
        shape=row["curve_shape"],
        location=row["curve_location"],
        scale=row["curve_scale"],
    )
    hurdle = CurveParameters(**row).to_distribution()
    grid = np.linspace(0.001, 0.999, 999)
    np.testing.assert_allclose(
        hurdle.quantiles(grid), np.maximum(0.0, fitted.quantiles(grid)), atol=1e-9
    )
    assert float(np.min(hurdle.quantiles(grid))) >= 0.0


def test_null_integers_reject_their_row_not_the_batch() -> None:
    probabilities = np.linspace(0.0, 1.0, 11)
    values = [_gumbel_quantiles(probabilities)] * 2
    table = _table(values, quantile_name="cdf_quantiles")
    axis = pa.array(
        [list(range(2, 13)), [2, None, *range(4, 13)]], pa.list_(pa.int64())
    )
    table = table.append_column("axis", axis)
    result = fit_cdf_quantile_batches(
        table,
        None,
        _policy(on_fit_failure="skip"),
        columns=CDFColumnSchema(return_periods="axis"),
    )
    rows = pa.concat_tables([b.hazard_rows for b in result.stream.batches])
    assert rows.num_rows == 1
    assert dict(result.summary.rejection_reasons) == {"invalid_return_periods": 1}
