"""Sample sources: Monte Carlo from curves and per-year series, offline."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    FittedDistribution,
    HurdleDistribution,
    PiecewiseLinearImpact,
    TabulatedDistribution,
)
from crc_framework.distributions import PointMassDistribution

from crc_sdk.connectors import (
    hazard_arrow_schema,
    read_hazard_dataset,
    read_hazard_metadata,
    write_hazard_dataset,
)
from crc_sdk.geometry import point_to_cell
from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import (
    AnnualSeries,
    ComonotoneIndexWarning,
    ImpactNoise,
    LossModel,
    NoBasisRisk,
    OutcomeSample,
    ProbabilitySemanticsError,
    SampleError,
    holdout,
    sample_annual,
    sample_curve,
)
from crc_sdk.parametric.sample import _check_out_of_sample
from crc_sdk.types import (
    CurveParameters,
    EnsembleDescriptor,
    HazardDatasetMetadata,
    SourceProvenance,
)
from crc_sdk.workflows import ReturnPeriodExtrapolationWarning

UNTYPED_CURVE: Any = sample_curve  # for calls with deliberately invalid arguments
UNTYPED_ANNUAL: Any = sample_annual

GUMBEL = FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0)
RAMP = PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
NOISE = ImpactNoise(0.3, upper=1.0)
CELL = point_to_cell(6.96, 50.94, 7)


def _metadata(**fields: Any) -> HazardDatasetMetadata:
    base: dict[str, Any] = {
        "h3_resolution": 7,
        "value_unit": "mm/day",
        "value_semantics": "annual maximum daily rainfall",
        "producer": "tests",
        "source": SourceProvenance(
            provider="fixture",
            dataset="rx1day",
            version="2026.1",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
            checksum="sha256:" + "0" * 64,
        ),
        "creation_version": "1",
        "probability_semantics": "annual_exceedance",
        "return_period_support": (2.0, 200.0),
    }
    base.update(fields)
    return HazardDatasetMetadata(**base)


def _curve(
    curve: Any = GUMBEL,
    *,
    metadata: HazardDatasetMetadata | None = None,
    n_draws: int = 1000,
    seed: int = 7,
    basis_risk: Any = NOISE,
    **options: Any,
) -> OutcomeSample:
    sample: OutcomeSample = UNTYPED_CURVE(
        curve,
        metadata=metadata or _metadata(),
        n_draws=n_draws,
        rng=np.random.default_rng(seed),
        loss_model=options.pop("loss_model", LOSS),
        basis_risk=basis_risk,
        asset_id=options.pop("asset_id", "warehouse-a"),
        **options,
    )
    return sample


def _row(**fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cell_index": CELL,
        "source_id": "fixture-1",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": 2050,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 50.0,
        "curve_scale": 10.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }
    row.update(fields)
    return row


NO_DATA: dict[str, Any] = {
    "curve_kind": "no_data",
    "curve_type": "insufficient_informative_support",
    "curve_location": None,
    "curve_scale": None,
}


# -- Monte Carlo from a curve -----------------------------------------------


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_draws_follow_crc_frameworks_inverse_cdf_route() -> None:
    sample = _curve(basis_risk=NOISE)
    expected = GUMBEL.sample(1000, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(sample.index[:, 0], expected)
    assert sample.period.tolist() == list(range(1000))
    np.testing.assert_allclose(sample.probability, 1e-3)
    assert sample.index_unit == "mm/day" and sample.loss_unit == "EUR"
    assert sample.hazard is None  # impact noise leaves the index alone
    provenance = sample.provenance
    assert provenance.source_kind == "monte_carlo"
    assert provenance.probability_semantics == "annual_exceedance"
    assert provenance.semantics_source == "dataset"
    assert provenance.basis_risk == {"kind": "impact_noise", "sigma": 0.3, "upper": 1.0}
    assert provenance.loss_model is not None
    assert provenance.loss_model["value"]["unit"] == "EUR"
    assert [record["step"] for record in provenance.rng] == ["hazard", "basis_risk"]
    assert provenance.rng[0]["bit_generator"] == "PCG64"
    assert provenance.licences[0] == {
        "role": "hazard",
        "source": "rx1day",
        "licence": "CC-BY-4.0",
        "attribution": "Synthetic fixture",
    }
    source = provenance.sources[0]
    assert source["metadata"]["source"]["version"] == "2026.1"
    assert source["metadata"]["source"]["checksum"] == "sha256:" + "0" * 64
    assert source["curve"] == {
        "kind": "fitted",
        "family": "gumbel_r",
        "location": 50.0,
        "scale": 10.0,
        "shape": None,
    }


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_the_recorded_generator_state_rebuilds_the_sample() -> None:
    sample = _curve(basis_risk=NOISE)
    rebuilt = sample_curve(
        GUMBEL,
        metadata=_metadata(),
        n_draws=1000,
        rng=sample.provenance.generator(),
        loss_model=LOSS,
        basis_risk=NOISE,
        asset_id="warehouse-a",
    )
    assert rebuilt.content_hash == sample.content_hash
    assert rebuilt.lineage == sample.lineage
    other_seed = _curve(seed=8)
    assert other_seed.lineage != sample.lineage
    _check_out_of_sample(sample, other_seed)  # fresh draws are out of sample


@pytest.mark.parametrize(
    "declared",
    ["within_period_percentile", "projection_uncertainty", "estimate_confidence"],
)
def test_other_declared_semantics_raise(declared: str) -> None:
    metadata = _metadata(probability_semantics=declared)
    with pytest.raises(ProbabilitySemanticsError, match=declared):
        _curve(metadata=metadata)
    with pytest.raises(ProbabilitySemanticsError):
        _curve(metadata=metadata, assume_semantics="annual_exceedance")


def test_unspecified_semantics_raise_unless_the_caller_states_them() -> None:
    metadata = _metadata(probability_semantics=None)
    with pytest.raises(ProbabilitySemanticsError, match="assume_semantics"):
        _curve(metadata=metadata)
    with pytest.warns(ReturnPeriodExtrapolationWarning):
        sample = _curve(metadata=metadata, assume_semantics="annual_value_distribution")
    assert sample.provenance.semantics_source == "caller"
    assert sample.provenance.probability_semantics == "annual_value_distribution"
    assert any("caller stated" in note for note in sample.provenance.notes)
    with pytest.raises(ProbabilitySemanticsError, match="assume_semantics must be"):
        _curve(metadata=metadata, assume_semantics="within_period_percentile")


def test_a_stated_semantics_must_match_the_declared_one() -> None:
    with pytest.raises(ProbabilitySemanticsError, match="contradicts"):
        _curve(assume_semantics="annual_value_distribution")
    with pytest.warns(ReturnPeriodExtrapolationWarning):
        sample = _curve(assume_semantics="annual_exceedance")
    assert sample.provenance.semantics_source == "dataset"


def test_a_hurdle_curve_keeps_its_dry_year_atom() -> None:
    hurdle = HurdleDistribution(GUMBEL, atom_probability=0.4, atom_location=0.0)
    sample = _curve(
        hurdle, n_draws=5000, metadata=_metadata(return_period_support=None)
    )
    dry = sample.index[:, 0] == 0.0
    atom = sample.provenance.details["atom"]
    assert atom == {"probability": 0.4, "location": 0.0, "draws": int(dry.sum())}
    assert abs(dry.mean() - 0.4) < 0.02
    assert (sample.loss[dry] == 0.0).all()
    assert sample.provenance.sources[0]["curve"]["kind"] == "hurdle"


def test_a_point_mass_curve() -> None:
    sample = _curve(
        PointMassDistribution(70.0),
        metadata=_metadata(return_period_support=None),
        n_draws=10,
    )
    assert (sample.index == 70.0).all()
    assert sample.provenance.sources[0]["curve"] == {
        "kind": "point_mass",
        "location": 70.0,
    }


def test_no_data_curves_are_refused() -> None:
    with pytest.raises(SampleError, match="no data"):
        _curve(_row(**NO_DATA))
    with pytest.raises(SampleError, match="never sampled as zero"):
        _curve(CurveParameters(**NO_DATA))


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_a_canonical_row_records_its_key() -> None:
    sample = _curve(_row(), details={"file_sha256": "ab" * 32})
    source = sample.provenance.sources[0]
    assert source["row"] == {
        "hazard_name": "rx1day",
        "horizon": 2050,
        "pathway": "ssp245",
        "source_id": "fixture-1",
        "cell_index": format(CELL, "x"),
    }
    assert source["details"] == {"file_sha256": "ab" * 32}
    np.testing.assert_array_equal(sample.index, _curve(GUMBEL).index)


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_curves_from_a_canonical_file(tmp_path: Path) -> None:
    path = tmp_path / "rx1day.parquet"
    rows = [
        _row(),
        _row(
            source_id="fixture-2",
            curve_kind="hurdle",
            curve_atom_probability=0.3,
            curve_atom_location=0.0,
        ),
        _row(source_id="fixture-3", **NO_DATA),
    ]
    metadata = _metadata()
    write_hazard_dataset(
        pa.Table.from_pylist(rows, schema=hazard_arrow_schema()),
        path,
        metadata,
        max_workers=1,
    )
    table = read_hazard_dataset(path)
    read = read_hazard_metadata(path)
    by_source = {row["source_id"]: row for row in table.to_pylist()}
    fitted = _curve(by_source["fixture-1"], metadata=read)
    assert fitted.provenance.to_document()["sources"][0]["metadata"] == (
        metadata.model_dump(mode="json")
    )
    hurdle = _curve(by_source["fixture-2"], metadata=read)
    assert hurdle.provenance.details["atom"]["probability"] == 0.3
    with pytest.raises(SampleError, match="no data"):
        _curve(by_source["fixture-3"], metadata=read)


def test_support_counts_on_each_side() -> None:
    with pytest.warns(ReturnPeriodExtrapolationWarning, match="rarer"):
        sample = _curve(metadata=_metadata(return_period_support=(2.0, 50.0)))
    uniforms = np.random.default_rng(7).random(1000)
    counts = sample.provenance.details["outside_support"]
    assert counts == {
        "below": int((uniforms < 0.5).sum()),
        "above": int((uniforms > 0.98).sum()),
    }
    assert counts["below"] > 400 and counts["above"] > 0
    notes = " ".join(sample.provenance.notes)
    assert "more frequent than the shortest fitted return period (2 years)" in notes
    assert "rarer than the longest fitted return period (50 years)" in notes


def test_lower_tail_support_counts_the_other_side() -> None:
    metadata = _metadata(return_period_tail="lower", return_period_support=(2.0, 50.0))
    with pytest.warns(ReturnPeriodExtrapolationWarning):
        sample = _curve(metadata=metadata)
    uniforms = np.random.default_rng(7).random(1000)
    assert sample.provenance.details["outside_support"] == {
        "below": int((uniforms > 0.5).sum()),
        "above": int((uniforms < 0.02).sum()),
    }


def test_source_probability_support_and_no_support() -> None:
    metadata = _metadata(
        return_period_support=None, source_probability_support=(0.1, 0.9999999)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReturnPeriodExtrapolationWarning)
        sample = _curve(metadata=metadata)
    support = sample.provenance.details["support"]
    assert support["basis"] == "source_probability_support"
    assert sample.provenance.details["outside_support"]["above"] == 0
    bare = _curve(metadata=_metadata(return_period_support=None))
    assert bare.provenance.details["support"] is None
    assert any("no fitted support" in note for note in bare.provenance.notes)


def test_a_tabulated_curve_raises_outside_its_knots_by_default() -> None:
    curve = TabulatedDistribution([0.5, 0.9, 0.99], [40.0, 70.0, 100.0])
    metadata = _metadata(return_period_support=None)
    with pytest.raises(SampleError, match="outside the tabulated curve's knots"):
        _curve(curve, metadata=metadata)
    clamped = _curve(curve, metadata=metadata, outside_support="clamp")
    record = clamped.provenance.details["tabulated"]
    uniforms = np.random.default_rng(7).random(1000)
    assert record == {
        "knots": (0.5, 0.99),
        "outside_support": "clamp",
        "below_knots": int((uniforms < 0.5).sum()),
        "above_knots": int((uniforms > 0.99).sum()),
    }
    assert clamped.index.min() == 40.0 and clamped.index.max() == 100.0
    extrapolated = _curve(curve, metadata=metadata, outside_support="extrapolate")
    assert extrapolated.provenance.details["tabulated"]["outside_support"] == (
        "extrapolate"
    )
    assert any(
        "extrapolated by crc-framework" in n for n in extrapolated.provenance.notes
    )
    # crc-framework 0.6 extrapolates linear-probability knots flat, as the clamp.
    np.testing.assert_array_equal(extrapolated.index, clamped.index)


def test_a_pooled_ensemble_is_named() -> None:
    metadata = _metadata(
        return_period_support=None,
        ensemble=EnsembleDescriptor(pooling="pooled", models=5),
    )
    sample = _curve(metadata=metadata)
    assert sample.provenance.details["ensemble_pooling"] == "pooled"
    assert any("pools an ensemble" in note for note in sample.provenance.notes)


def test_no_basis_risk_warns_that_index_and_loss_are_comonotone() -> None:
    metadata = _metadata(return_period_support=None)
    with pytest.warns(ComonotoneIndexWarning, match="comonotone"):
        sample = _curve(metadata=metadata, basis_risk=NoBasisRisk())
    np.testing.assert_allclose(sample.loss, LOSS.evaluate(sample.index))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComonotoneIndexWarning)
        noisy = _curve(metadata=metadata, basis_risk=NOISE)
    assert not np.allclose(noisy.loss, sample.loss)
    np.testing.assert_array_equal(noisy.index, sample.index)
    summary = noisy.provenance.details["basis_risk"]
    assert summary["index_changed"] is False and summary["capped_draws"] >= 0


class _IndexShift:
    """A user basis-risk model that moves the index and keeps the impact."""

    name = "index_shift"

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.name, "shift": 5.0}

    def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
        return index + 5.0, impact


def test_a_user_model_that_moves_the_index_keeps_the_hazard() -> None:
    sample = _curve(
        metadata=_metadata(return_period_support=None), basis_risk=_IndexShift()
    )
    assert sample.hazard is not None
    np.testing.assert_allclose(sample.index - sample.hazard, 5.0)
    assert sample.provenance.basis_risk == {"kind": "index_shift", "shift": 5.0}


class _NaNCurve:
    def quantiles(self, probabilities: Any) -> Any:
        return np.full(np.shape(probabilities), np.nan)


class _FailingCurve:
    def quantiles(self, probabilities: Any) -> Any:
        raise ValueError("probability outside support")


def test_non_finite_or_failing_quantiles_raise() -> None:
    metadata = _metadata(return_period_support=None)
    with pytest.raises(SampleError, match="non-finite"):
        _curve(_NaNCurve(), metadata=metadata)
    with pytest.raises(SampleError, match="cannot be evaluated"):
        _curve(_FailingCurve(), metadata=metadata)
    assert (
        _curve(GUMBEL, metadata=metadata, n_draws=3).provenance.sources[0]["curve"][
            "kind"
        ]
        == "fitted"
    )


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"n_draws": 0}, SampleError, "n_draws"),
        ({"n_draws": True}, SampleError, "n_draws"),
        ({"basis_risk": None}, TypeError, "NoBasisRisk"),
        ({"loss_model": RAMP}, TypeError, "LossModel"),
        ({"outside_support": "ignore"}, SampleError, "outside_support"),
        ({"asset_id": ""}, SampleError, "asset_id"),
        ({"details": 3}, TypeError, "source"),
    ],
)
def test_invalid_arguments_raise(
    options: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _curve(metadata=_metadata(return_period_support=None), **options)


def test_the_generator_and_metadata_are_required_types() -> None:
    with pytest.raises(TypeError, match="Generator"):
        UNTYPED_CURVE(
            GUMBEL,
            metadata=_metadata(),
            n_draws=3,
            rng=7,
            loss_model=LOSS,
            basis_risk=NOISE,
            asset_id="a",
        )
    with pytest.raises(TypeError, match="HazardDatasetMetadata"):
        UNTYPED_CURVE(
            GUMBEL,
            metadata={"probability_semantics": "annual_exceedance"},
            n_draws=3,
            rng=np.random.default_rng(0),
            loss_model=LOSS,
            basis_risk=NOISE,
            asset_id="a",
        )


def test_a_loss_model_bound_to_another_asset_raises() -> None:
    assets = pa.table({"asset_id": ["b"], "value": [1.0]})
    bound = LossModel(RAMP, value_column="value", value_unit="EUR").for_asset(
        assets, "b"
    )
    with pytest.raises(SampleError, match="bound to asset 'b'"):
        _curve(metadata=_metadata(return_period_support=None), loss_model=bound)


# -- per-year series ----------------------------------------------------------

STATIONS = pa.table(
    {
        "station": ["S1"] * 5 + ["S2"] * 5,
        "year": pa.array([2000, 2001, 2002, 2003, 2004] * 2, pa.int32()),
        "value": [55.0, 72.0, None, 90.0, 61.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "complete": [True, True, False, True, True] + [True] * 5,
        "attribution": ["NOAA"] * 10,
    }
)

GRID = pa.table(
    {
        "year": pa.array(
            [y for y in range(2000, 2005) for _ in range(6) if True], pa.int32()
        ),
        "latitude": [
            lat for _ in range(5) for lat in (50.0, 50.0, 50.0, 50.25, 50.25, 50.25)
        ],
        "longitude": [lon for _ in range(5) for lon in (6.75, 7.0, 7.25) * 2],
        "value": [float(10 * y + c) for y in range(5) for c in range(6)],
    }
)


def test_from_table_selects_a_station_and_marks_null_years_missing() -> None:
    series = AnnualSeries.from_table(
        STATIONS, where={"station": "S1"}, unit="mm/day", source="ghcnh-rx1day"
    )
    assert series.years.tolist() == [2000, 2001, 2002, 2003, 2004]
    assert series.missing_years == (2002,)
    assert series.provenance["where"] == {"station": "S1"}
    assert series.provenance["table_columns"]["station"] == {
        "values": ("S1",),
        "distinct": 1,
    }
    assert series.provenance["table_columns"]["attribution"]["values"] == ("NOAA",)
    assert not series.values.flags.writeable


def test_from_table_fills_gaps_and_honours_a_window() -> None:
    table = pa.table({"year": [2000, 2003, 2001], "value": [1.0, 3.0, 2.0]})
    series = AnnualSeries.from_table(table, unit="m")
    assert series.years.tolist() == [2000, 2001, 2002, 2003]
    assert series.missing_years == (2002,)
    window = AnnualSeries.from_table(table, unit="m", years=[2001, 2003, 2004])
    assert window.years.tolist() == [2001, 2003, 2004]
    assert window.values[:2].tolist() == [2.0, 3.0]
    assert window.missing_years == (2004,)


@pytest.mark.parametrize(
    ("table", "options", "message"),
    [
        (STATIONS, {}, "several rows"),
        (STATIONS, {"where": {"station": "S9"}}, "no rows"),
        (STATIONS, {"where": {"station": 3}}, "cannot be compared"),
        (STATIONS, {"where": {"nowhere": "S1"}}, "no column 'nowhere'"),
        (STATIONS, {"where": {}}, "non-empty mapping"),
        (pa.table({"year": [2000], "value": [np.inf]}), {}, "infinite"),
        (pa.table({"year": [2000.0], "value": [1.0]}), {}, "integers"),
        (pa.table({"year": [2000], "value": ["1"]}), {}, "numeric"),
        (
            pa.table({"year": [2000], "value": [1.0], "unit": ["degC"]}),
            {},
            "unit column",
        ),
        (pa.table({"year": [2000], "value": [1.0]}), {"years": [2000, 2000]}, "repeat"),
    ],
)
def test_invalid_annual_tables_raise(
    table: Any, options: dict[str, Any], message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        AnnualSeries.from_table(table, unit="mm/day", **options)


def test_record_batches_with_licence_columns() -> None:
    batches = [
        pa.record_batch(
            {
                "year": pa.array([year], pa.int32()),
                "value": [float(year - 1990)],
                "unit": ["mm/day"],
                "archive_licence": ["CC0-1.0"],
                "attribution": ["NASA NEX-GDDP-CMIP6"],
                "retrieved_at": [f"2026-11-0{year - 1999}T00:00:00+00:00"],
            }
        )
        for year in (2000, 2001, 2002)
    ]
    series = AnnualSeries.from_table(iter(batches), unit="mm/day")
    recorded = series.provenance["table_columns"]
    assert recorded["archive_licence"]["values"] == ("CC0-1.0",)
    assert recorded["retrieved_at"]["distinct"] == 3
    sample = sample_annual(
        series,
        loss_model=LOSS,
        basis_risk=NOISE,
        asset_id="a",
        rng=np.random.default_rng(1),
    )
    assert sample.provenance.licences[0] == {
        "role": "hazard",
        "source": None,
        "licence": "CC0-1.0",
        "attribution": "NASA NEX-GDDP-CMIP6",
    }


def test_nearest_cell_accepts_a_centre_within_half_a_step() -> None:
    series = AnnualSeries.nearest_cell(
        GRID, longitude=6.9, latitude=50.1, unit="mm/day"
    )
    assert series.provenance["cell"] == {"latitude": 50.0, "longitude": 7.0}
    assert series.provenance["spacing_degrees"] == {"latitude": 0.25, "longitude": 0.25}
    assert 10.0 < series.provenance["distance_km"] < 15.0
    assert "6.9" not in str(series.provenance)  # the asset's point is not recorded
    assert series.values.tolist() == [1.0, 11.0, 21.0, 31.0, 41.0]


def test_nearest_cell_marks_years_the_cell_lacks() -> None:
    gappy = GRID.filter(
        pa.compute.invert(
            pa.compute.and_(
                pa.compute.equal(GRID.column("year"), 2002),
                pa.compute.equal(GRID.column("longitude"), 7.0),
            )
        )
    )
    series = AnnualSeries.nearest_cell(
        gappy, longitude=7.0, latitude=50.0, unit="mm/day"
    )
    assert series.missing_years == (2002,)


def test_nearest_cell_refusals() -> None:
    with pytest.raises(SampleError, match="within half a grid step"):
        AnnualSeries.nearest_cell(GRID, longitude=7.6, latitude=50.0, unit="mm/day")
    with pytest.raises(SampleError, match="equally near"):
        AnnualSeries.nearest_cell(GRID, longitude=6.875, latitude=50.0, unit="mm/day")
    one_row = GRID.filter(pa.compute.equal(GRID.column("latitude"), 50.0))
    with pytest.raises(SampleError, match="spacing"):
        AnnualSeries.nearest_cell(one_row, longitude=7.0, latitude=50.0, unit="mm/day")
    series = AnnualSeries.nearest_cell(
        one_row, longitude=7.0, latitude=50.1, unit="mm/day", spacing=(0.25, 0.25)
    )
    assert series.provenance["cell"] == {"latitude": 50.0, "longitude": 7.0}
    with pytest.raises(SampleError, match="latitude"):
        AnnualSeries.nearest_cell(GRID, longitude=7.0, latitude=95.0, unit="mm/day")


def test_nearest_cell_compares_longitudes_modulo_360() -> None:
    table = pa.table(
        {
            "year": [2000, 2000, 2001, 2001],
            "latitude": [10.0, 10.0, 10.0, 10.0],
            "longitude": [359.75, 0.0, 359.75, 0.0],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    series = AnnualSeries.nearest_cell(
        table, longitude=-0.2, latitude=10.0, unit="m", spacing=(0.25, 0.25)
    )
    assert series.provenance["cell"]["longitude"] == 359.75
    assert series.values.tolist() == [1.0, 3.0]


def test_an_annual_series_validates_its_fields() -> None:
    with pytest.raises(SampleError, match="strictly increasing"):
        AnnualSeries(np.array([2001, 2000]), np.array([1.0, 2.0]), "m", {})
    with pytest.raises(SampleError, match="one entry per year"):
        AnnualSeries(np.array([2000, 2001]), np.array([1.0]), "m", {})
    with pytest.raises(SampleError, match="finite"):
        AnnualSeries(np.array([2000]), np.array([np.inf]), "m", {})
    with pytest.raises(SampleError, match="plain JSON"):
        AnnualSeries(np.array([2000]), np.array([1.0]), "m", {"x": object()})


# -- per-year samples -----------------------------------------------------------


def _series(
    values: list[float | None], *, first: int = 2000, **provenance: Any
) -> AnnualSeries:
    data = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    return AnnualSeries(
        np.arange(first, first + len(values)), data, "mm/day", provenance
    )


def test_years_become_the_rows() -> None:
    with pytest.warns(ComonotoneIndexWarning):
        sample = sample_annual(
            _series([50.0, 70.0, 100.0, 40.0]),
            loss_model=LOSS,
            basis_risk=NoBasisRisk(),
            asset_id="warehouse-a",
        )
    assert sample.period.tolist() == [2000, 2001, 2002, 2003]
    np.testing.assert_allclose(sample.probability, 0.25)
    np.testing.assert_allclose(sample.loss[:, 0], [1e6 / 6, 0.5e6, 1e6, 0.0])
    provenance = sample.provenance
    assert provenance.source_kind == "annual"
    assert provenance.semantics_source == "data"
    assert provenance.probability_semantics is None
    assert provenance.rng == ()
    assert provenance.details["years"] == (2000, 2003)
    assert "no trend is removed" in provenance.notes[0]


def test_missing_years_raise_or_are_dropped_and_listed() -> None:
    series = _series([50.0, None, 100.0, None])
    with pytest.raises(SampleError, match="2 years have no value"):
        sample_annual(
            series,
            loss_model=LOSS,
            basis_risk=NOISE,
            asset_id="a",
            rng=np.random.default_rng(0),
        )
    sample = sample_annual(
        series,
        loss_model=LOSS,
        basis_risk=NOISE,
        asset_id="a",
        rng=np.random.default_rng(0),
        missing="drop_year",
    )
    assert sample.period.tolist() == [2000, 2002]
    assert sample.provenance.details["dropped_years"] == (2001, 2003)
    assert any("2001, 2003" in note for note in sample.provenance.notes)
    with pytest.raises(SampleError, match="missing must be"):
        UNTYPED_ANNUAL(
            series, loss_model=LOSS, basis_risk=NOISE, asset_id="a", missing="ignore"
        )


def test_an_index_series_carries_the_basis_risk() -> None:
    hazard = _series([50.0, 70.0, 100.0, 40.0], station="S1")
    index = AnnualSeries(
        np.arange(2000, 2004),
        np.array([45.0, 80.0, 90.0, 55.0]),
        "mm",
        {"station": "S2"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComonotoneIndexWarning)
        sample = sample_annual(
            hazard, index=index, loss_model=LOSS, basis_risk=NoBasisRisk(), asset_id="a"
        )
    assert sample.index[:, 0].tolist() == [45.0, 80.0, 90.0, 55.0]
    assert sample.hazard is not None
    assert sample.hazard[:, 0].tolist() == [50.0, 70.0, 100.0, 40.0]
    assert sample.index_unit == "mm"
    assert [source["role"] for source in sample.provenance.sources] == [
        "hazard",
        "index",
    ]
    assert sample.provenance.details["index_from"] == "series"


def test_the_index_series_must_cover_the_same_years() -> None:
    hazard = _series([50.0, 70.0, 100.0])
    with pytest.raises(SampleError, match="years differ"):
        sample_annual(
            hazard,
            index=_series([1.0, 2.0, 3.0], first=2001),
            loss_model=LOSS,
            basis_risk=NoBasisRisk(),
            asset_id="a",
        )
    with pytest.raises(SampleError, match=r"index \[2001\]"):
        sample_annual(
            hazard,
            index=_series([1.0, None, 3.0]),
            loss_model=LOSS,
            basis_risk=NoBasisRisk(),
            asset_id="a",
        )


def test_impact_noise_on_years_needs_and_records_a_generator() -> None:
    series = _series([50.0, 70.0, 100.0, 40.0])
    with pytest.raises(SampleError, match="rng"):
        sample_annual(series, loss_model=LOSS, basis_risk=NOISE, asset_id="a")
    sample = sample_annual(
        series,
        loss_model=LOSS,
        basis_risk=NOISE,
        asset_id="a",
        rng=np.random.default_rng(5),
    )
    assert [record["step"] for record in sample.provenance.rng] == ["basis_risk"]
    rebuilt = sample_annual(
        series,
        loss_model=LOSS,
        basis_risk=NOISE,
        asset_id="a",
        rng=sample.provenance.generator(),
    )
    assert rebuilt.content_hash == sample.content_hash
    with pytest.raises(TypeError, match="AnnualSeries"):
        UNTYPED_ANNUAL([1.0], loss_model=LOSS, basis_risk=NOISE, asset_id="a")


def test_windows_of_one_named_series_share_a_lineage() -> None:
    def window(first: int, last: int, source: str | None) -> OutcomeSample:
        series = AnnualSeries.from_table(
            STATIONS,
            where={"station": "S2"},
            unit="mm/day",
            years=range(first, last + 1),
            source=source,
        )
        return sample_annual(
            series,
            loss_model=LOSS,
            basis_risk=NOISE,
            asset_id="a",
            rng=np.random.default_rng(0),
        )

    design, evaluation = window(2000, 2002, "ghcnh"), window(2002, 2004, "ghcnh")
    with pytest.raises(Exception, match="shares 1 periods"):
        _check_out_of_sample(design, evaluation)
    # Without a source the identity is the data, so these are not linked...
    _check_out_of_sample(window(2000, 2002, None), window(2002, 2004, None))
    # ...but splits of one sample always are.
    full = window(2000, 2004, None)
    train, test = holdout(full, 0.6)
    _check_out_of_sample(train, test)
    with pytest.raises(Exception, match="shares"):
        _check_out_of_sample(train, full)


class _BadModel:
    """A user basis-risk model whose output is invalid in a stated way."""

    name = "bad"

    def __init__(self, output: Any) -> None:
        self.output = output

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.name}

    def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
        return self.output(index, impact)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (lambda index, impact: [index, impact], r"must return \(index, impact\)"),
        (lambda index, impact: (index, "x"), "non-numbers"),
        (lambda index, impact: (index[:1], impact), "shape"),
        (lambda index, impact: (index * np.nan, impact), "non-finite indices"),
        (lambda index, impact: (index, impact - 1.0), "negative or non-finite"),
    ],
)
def test_invalid_basis_risk_output_raises(output: Any, message: str) -> None:
    with pytest.raises(SampleError, match=message):
        _curve(
            metadata=_metadata(return_period_support=None),
            basis_risk=_BadModel(output),
            n_draws=5,
        )


# -- hazard units ------------------------------------------------------------

RX1DAY_LOSS = LossModel(
    ImpactExpression.linear(
        0.01,
        calibration=ImpactCalibration(
            exposure_variable="rx1day",
            exposure_unit="mm/day",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
        clamp={"minimum": 0.0, "maximum": 1.0},
    ),
    value=1e6,
    value_unit="EUR",
)


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_sample_curve_checks_the_curves_unit_before_drawing() -> None:
    sample = _curve(loss_model=RX1DAY_LOSS)
    assert (sample.index_unit, sample.loss_unit) == ("mm/day", "EUR")
    assert sample.provenance.loss_model is not None
    assert sample.provenance.loss_model["impact"]["exposure_unit"] == "mm/day"
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state
    with pytest.raises(SampleError, match="calibrated for hazard values in 'mm/day'"):
        sample_curve(
            GUMBEL,
            metadata=_metadata(value_unit="mm"),
            n_draws=100,
            rng=rng,
            loss_model=RX1DAY_LOSS,
            basis_risk=NOISE,
            asset_id="warehouse-a",
        )
    assert rng.bit_generator.state == before


def test_sample_annual_checks_the_hazard_series_unit_before_drawing() -> None:
    hazard = _series([50.0, 70.0, 100.0, 40.0])
    sample = sample_annual(
        hazard,
        loss_model=RX1DAY_LOSS,
        basis_risk=NOISE,
        asset_id="a",
        rng=np.random.default_rng(5),
    )
    assert sample.provenance.loss_model is not None
    assert sample.provenance.loss_model["impact"]["exposure_unit"] == "mm/day"
    # The index series is not what the loss reads, so its unit is not checked.
    index = AnnualSeries(
        hazard.years, np.array([45.0, 80.0, 90.0, 55.0]), "mm", {"station": "S2"}
    )
    paid_on = sample_annual(
        hazard,
        index=index,
        loss_model=RX1DAY_LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
    )
    assert paid_on.index_unit == "mm"
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    in_mm = AnnualSeries(hazard.years, hazard.values, "mm", {})
    with pytest.raises(SampleError, match="'mm', but impact 'linear'"):
        sample_annual(
            in_mm, loss_model=RX1DAY_LOSS, basis_risk=NOISE, asset_id="a", rng=rng
        )
    assert rng.bit_generator.state == before
    with pytest.raises(SampleError, match="calibrated"):
        sample_annual(
            in_mm,
            index=hazard,
            loss_model=RX1DAY_LOSS,
            basis_risk=NoBasisRisk(),
            asset_id="a",
        )


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_impacts_without_a_unit_are_not_checked_by_either_route() -> None:
    def ratio(values: Any) -> Any:
        return np.clip(np.asarray(values) / 100.0, 0.0, 1.0)

    for loss_model in (LOSS, LossModel(ratio, value=1e6, value_unit="EUR")):
        assert loss_model.exposure_unit is None
        curve = _curve(metadata=_metadata(value_unit="any unit"), loss_model=loss_model)
        annual = sample_annual(
            AnnualSeries(np.arange(2000, 2003), np.array([5.0, 50.0, 90.0]), "K", {}),
            loss_model=loss_model,
            basis_risk=NOISE,
            asset_id="a",
            rng=np.random.default_rng(1),
        )
        for sample in (curve, annual):
            assert sample.provenance.loss_model is not None
            assert sample.provenance.loss_model["impact"]["exposure_unit"] is None
