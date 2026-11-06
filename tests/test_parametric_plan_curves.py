"""ParametricPlan.for_evaluation: fitted curves, Monte Carlo draws and fresh draws."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact, TransformContext
from crc_framework import impacts as framework_impacts

from crc_sdk.connectors import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.geometry import point_to_cell
from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import (
    MSE,
    Budget,
    ComonotoneIndexWarning,
    Cover,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    FreshDraws,
    Holdout,
    ImpactNoise,
    IndexDefinition,
    KFold,
    LinearPayout,
    LossModel,
    MetricOptions,
    NoBasisRisk,
    ParametricPlan,
    ProbabilitySemanticsError,
    SampleError,
    StepPayout,
    WeightedCVaR,
    compare,
    design,
    sample_curve,
    setup_text,
)
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import (
    AssetPortfolio,
    ExecutionOptions,
    HazardDataset,
    ImpactContextColumns,
    PointColumns,
    ReturnPeriodExtrapolationWarning,
)

UNTYPED_PLAN: Any = ParametricPlan
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))
POINTS = {"warehouse-a": (6.96, 50.94), "depot-b": (7.50, 51.00)}
CELLS = {name: point_to_cell(lon, lat, 7) for name, (lon, lat) in POINTS.items()}
PROVENANCE = SourceProvenance(
    provider="example",
    dataset="synthetic rx1day",
    version="1",
    licence="CC0-1.0",
    attribution="Synthetic curves",
)


def _metadata(**update: Any) -> HazardDatasetMetadata:
    metadata = HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=PROVENANCE,
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 500.0),
    )
    return metadata.model_copy(update=update)


def _curve(asset: str, **fields: Any) -> dict[str, Any]:
    row = {
        "cell_index": CELLS[asset],
        "source_id": f"fixture-{asset}",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": 2050,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 50.0 if asset == "warehouse-a" else 55.0,
        "curve_scale": 12.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }
    row.update(fields)
    return row


def _dataset(
    path: Path,
    *rows: dict[str, Any],
    metadata: HazardDatasetMetadata | None = None,
) -> Path:
    stated = metadata or _metadata()
    table = pa.Table.from_pylist(
        list(rows) or [_curve("warehouse-a"), _curve("depot-b")],
        schema=hazard_arrow_schema(stated),
    )
    write_hazard_dataset(table, path, stated, max_workers=1)
    return path


def _assets(*ids: str) -> Any:
    ids = ids or ("warehouse-a", "depot-b")
    return pa.table(
        {
            "asset_id": list(ids),
            "longitude": [POINTS[i][0] for i in ids],
            "latitude": [POINTS[i][1] for i in ids],
            "value_eur": [2e6 + 1e5 * k for k in range(len(ids))],
            "building": ["warehouse"] * len(ids),
        }
    )


def _request(path: Path, assets: Any = None, impact: Any = RAMP, **select: Any) -> Any:
    selection: dict[str, Any] = {"horizons": [2050], "pathways": ["ssp245"]}
    selection.update(select)
    return (
        HazardDataset.local(path)
        .for_assets(_assets() if assets is None else assets)
        .select(**selection)
        .impact(impact, name="ramp", value_unit="1", value_semantics="damage ratio")
    )


def _plan(path: Path, **options: Any) -> ParametricPlan:
    plan: ParametricPlan = UNTYPED_PLAN.for_evaluation(
        _request(path),
        value_column="value_eur",
        value_unit="EUR",
        **options,
    )
    return plan


def _step() -> Cover:
    return Cover(
        INDEX,
        StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        label="step",
    )


def _linear() -> Cover:
    return Cover(
        INDEX, LinearPayout(60.0, 120.0, 5e5), payout_unit="EUR", label="linear"
    )


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        warnings.simplefilter("ignore", ReturnPeriodExtrapolationWarning)
        yield


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return _dataset(tmp_path / "rx1day.parquet")


# -- end to end ----------------------------------------------------------------


def test_for_evaluation_runs_end_to_end_offline(dataset: Path, tmp_path: Path) -> None:
    plan = (
        _plan(dataset)
        .monte_carlo(1_500, seed=42)
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear())
        .objective(
            MSE(), ExpectedUncovered(), WeightedCVaR(0.9, 0.9), budget=Budget(ratio=0.8)
        )
        .split(FreshDraws(1_000, seed=7))
        .metrics(OPTIONS)
    )
    output = tmp_path / "out" / "comparison.parquet"
    result = plan.write_parquet(output)

    read = pq.read_table(output)
    assert read.num_rows == 2 * 2 * 4  # assets, splits, three designs and a cover
    stored = EvaluationMetadata.from_table(read)
    assert stored == result.metadata
    methods = [(e["split"], e["method"]) for e in stored.setup.evaluations]
    assert sorted(set(methods)) == [
        ("in_sample", "in_sample"),
        ("out_of_sample", "fresh_draws"),
    ]
    for entry in stored.setup.evaluations:
        expected = {"plan_seed": 42}
        if entry["method"] == "fresh_draws":
            expected["fresh_draws_seed"] = 7
            assert entry["leakage_check"] == "passed"
        assert entry["details"] == expected
    draws = sorted(
        (tuple(entry["asset_ids"]), entry["draws"], tuple(entry["roles"]))
        for entry in stored.setup.samples
    )
    assert draws == [
        (("depot-b",), 1_000, ("evaluation",)),
        (("depot-b",), 1_500, ("design", "evaluation")),
        (("warehouse-a",), 1_000, ("evaluation",)),
        (("warehouse-a",), 1_500, ("design", "evaluation")),
    ]
    hazard = [entry for entry in stored.sources if entry["role"] == "hazard"]
    assert {entry["row"]["cell_index"] for entry in hazard} == {
        format(cell, "x") for cell in CELLS.values()
    }
    record = hazard[0]["canonical_file"]
    assert record["path"] == str(dataset)
    assert record["file_sha256"].startswith("sha256:")
    assert record["curves"] == 2
    assert hazard[0]["details"]["spatial_match"] == "h3_cell"
    assert hazard[0]["metadata"]["probability_semantics"] == "annual_exceedance"
    assert stored.licences[0]["licence"] == "CC0-1.0"
    assert stored.assets["count"] == 2 and stored.assets["values_sha256"]
    assert len(result.designs) == 2 * 3
    text = setup_text(stored)
    assert "on fresh Monte Carlo draws: evaluated on 1000 simulated years" in text
    assert "designed on 1500 simulated years" in text
    assert "with annual_exceedance semantics, as the dataset declares" in text
    assert "more frequent than the shortest fitted return period (2 years)" in text


def test_samples_follow_sample_curve_on_the_asset_curves_rows(dataset: Path) -> None:
    plan = _plan(dataset).monte_carlo(800, seed=3).basis_risk(ImpactNoise(0.25))
    samples = plan.samples()
    assert list(samples) == ["warehouse-a", "depot-b"]
    request = _request(dataset)
    curves = {row["asset_id"]: row for row in request.asset_curves().to_pylist()}
    metadata = request.dataset.metadata()
    assets = AssetPortfolio(_assets())
    children = np.random.SeedSequence(3).spawn(2)
    for (asset_id, sample), child in zip(samples.items(), children, strict=True):
        expected = sample_curve(
            curves[asset_id],
            metadata=metadata,
            n_draws=800,
            rng=np.random.default_rng(child),
            loss_model=LossModel(
                RAMP,
                value_column="value_eur",
                value_unit="EUR",
                context=TransformContext(cell=CELLS[asset_id]),
            ).for_asset(assets, asset_id),
            basis_risk=ImpactNoise(0.25),
            asset_id=asset_id,
        )
        assert sample.content_hash == expected.content_hash
        assert sample.provenance.source_kind == "monte_carlo"
        loss_document: Any = sample.provenance.loss_model
        assert loss_document["impact"]["name"] == "ramp"
        assert sample.provenance.sources[0]["row"]["source_id"] == f"fixture-{asset_id}"
    # Another asset after the first changes nothing for its draws.
    first = UNTYPED_PLAN.for_evaluation(
        _request(dataset, _assets("warehouse-a")),
        value_column="value_eur",
        value_unit="EUR",
    )
    alone = first.monte_carlo(800, seed=3).basis_risk(ImpactNoise(0.25)).samples()
    assert alone["warehouse-a"].content_hash == samples["warehouse-a"].content_hash


def test_fresh_draws_use_their_own_seed_and_designs_follow_the_function_route(
    dataset: Path,
) -> None:
    plan = (
        _plan(dataset)
        .monte_carlo(600, seed=11)
        .basis_risk(NoBasisRisk())
        .covers(_step(), _linear())
        .objective(ExpectedUncovered(), budget=Budget(ratio=0.8))
        .split(FreshDraws(400, seed=12))
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(dataset.parent / "comparison.parquet")
    samples = result.samples
    # designs() and samples() need no fresh draws, and give the same designs.
    assert [d.parameters for d in plan.designs()] == [
        d.parameters for d in result.designs
    ]
    assert {s.content_hash for s in plan.samples().values()} == {
        s.content_hash for s in samples.values()
    }
    request = _request(dataset)
    curves = {row["asset_id"]: row for row in request.asset_curves().to_pylist()}
    metadata = request.dataset.metadata()
    assets = AssetPortfolio(_assets())
    fresh_children = np.random.SeedSequence(12).spawn(2)
    for index, asset_id in enumerate(("warehouse-a", "depot-b")):
        loss_model = LossModel(
            RAMP,
            value_column="value_eur",
            value_unit="EUR",
            context=TransformContext(cell=CELLS[asset_id]),
        ).for_asset(assets, asset_id)
        fresh = sample_curve(
            curves[asset_id],
            metadata=metadata,
            n_draws=400,
            rng=np.random.default_rng(fresh_children[index]),
            loss_model=loss_model,
            basis_risk=NoBasisRisk(),
            asset_id=asset_id,
        )
        designed = design(
            _step(), samples[asset_id], ExpectedUncovered(), budget=Budget(ratio=0.8)
        )
        planned = next(
            d for d in result.designs if d.design_sample is samples[asset_id]
        )
        assert planned.parameters == designed.parameters
        by_hand = compare(
            [designed, _linear()],
            fresh,
            split="out_of_sample",
            design_sample=samples[asset_id],
            options=OPTIONS,
        )
        rows = [
            row
            for row in result.table.to_pylist()
            if row["asset_id"] == asset_id and row["split"] == "out_of_sample"
        ]
        assert [row["sample_hash"] for row in rows] == [fresh.content_hash] * 2
        for row, expected in zip(rows, by_hand.to_pylist(), strict=True):
            assert row["expected_uncovered"] == pytest.approx(
                expected["expected_uncovered"], rel=1e-12
            )


def test_holdout_and_kfold_split_the_draws(dataset: Path) -> None:
    base = (
        _plan(dataset)
        .monte_carlo(500, seed=5)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    held = base.split(Holdout(0.6)).write_parquet(dataset.parent / "holdout.parquet")
    out = [e for e in held.metadata.setup.evaluations if e["split"] == "out_of_sample"]
    assert {e["method"] for e in out} == {"holdout"}
    assert out[0]["details"] == {"train_fraction": 0.6, "plan_seed": 5}
    folded = base.split(KFold(4)).write_parquet(dataset.parent / "kfold.parquet")
    assert folded.row_count == 2 * 4 * 2


# -- the loss model -------------------------------------------------------------


def test_an_impact_expression_keeps_its_definition_hash(dataset: Path) -> None:
    expression = ImpactExpression.tabulated(
        [(40.0, 0.0), (120.0, 0.5)],
        calibration=ImpactCalibration(
            exposure_variable="rx1day",
            exposure_unit="mm/day",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
    )
    request = _request(dataset, impact=expression)
    plan = UNTYPED_PLAN.for_evaluation(request).monte_carlo(200, seed=1)
    samples = plan.basis_risk(NoBasisRisk()).samples()
    impact = samples["warehouse-a"].provenance.loss_model["impact"]
    assert impact["kind"] == "expression"
    assert impact["definition_hash"] == expression.definition_hash
    assert impact["exposure_unit"] == "mm/day"
    assert samples["warehouse-a"].loss_unit == "1"
    hazard = samples["warehouse-a"].index[:, 0]
    np.testing.assert_allclose(
        samples["warehouse-a"].loss[:, 0], expression.evaluate(hazard)
    )
    # A callable stays a callable.
    ramp = UNTYPED_PLAN.for_evaluation(_request(dataset, impact=_double_ramp))
    drawn = ramp.monte_carlo(50, seed=1).basis_risk(NoBasisRisk()).samples()
    assert drawn["warehouse-a"].provenance.loss_model["impact"]["kind"] == "callable"


def _double_ramp(values: Any) -> Any:
    return np.clip((np.asarray(values) - 40.0) / 160.0, 0.0, 1.0)


def test_impacts_get_the_matched_cell_and_context_columns(dataset: Path) -> None:
    # A registry impact that needs the H3 cell, as write_parquet gives it.
    rx1day = framework_impacts.for_factor("rx1day")
    plan = UNTYPED_PLAN.for_evaluation(_request(dataset, impact=rx1day))
    sample = (
        plan.monte_carlo(300, seed=2).basis_risk(NoBasisRisk()).samples()["depot-b"]
    )
    cell = CELLS["depot-b"]
    with_cell = framework_impacts.for_factor(
        "rx1day", context=TransformContext(cell=cell)
    )
    np.testing.assert_allclose(
        sample.loss[:, 0], with_cell.evaluate(sample.index[:, 0])
    )
    assert sample.provenance.loss_model["context"]["fixed"]["cell"] == format(cell, "x")

    # Context columns of the request fill the context, on top of the impact's own.
    class Tagged:
        context = TransformContext(continent="Europe", historic_mean=1.0)

        def evaluate(self, values: Any, *, context: Any = None) -> Any:
            assert context.cell == CELLS["warehouse-a"]
            assert context.building_type == "warehouse"
            assert context.continent == "Europe"
            return np.full(np.shape(values), 0.25)

    request = (
        HazardDataset.local(dataset)
        .for_assets(_assets("warehouse-a"))
        .select(horizons=[2050])
        .impact(
            Tagged(),
            name="tagged",
            value_unit="1",
            value_semantics="damage ratio",
            context=ImpactContextColumns(building_type="building"),
        )
    )
    tagged = UNTYPED_PLAN.for_evaluation(
        request, value_column="value_eur", value_unit="EUR"
    )
    drawn = tagged.monte_carlo(20, seed=1).basis_risk(NoBasisRisk()).samples()
    np.testing.assert_allclose(drawn["warehouse-a"].loss[:, 0], 0.25 * 2e6)
    resolved = drawn["warehouse-a"].provenance.loss_model["context"]["fixed"]
    assert resolved["building_type"] == "warehouse"
    assert resolved["continent"] == "Europe"
    # The request's own evaluation gives the same impact at a return period.
    written = request.return_periods([100]).write_parquet(
        dataset.parent / "portfolio.parquet",
        execution=ExecutionOptions(max_workers=1),
    )
    assert written.row_count == 1
    values = pq.read_table(dataset.parent / "portfolio.parquet")["value_rp100"]
    assert values.to_pylist() == [0.25]


# -- semantics and curves -------------------------------------------------------


def test_unspecified_semantics_need_assume_semantics(tmp_path: Path) -> None:
    path = _dataset(
        tmp_path / "unspecified.parquet",
        metadata=_metadata(probability_semantics=None),
    )
    plan = _plan(path).basis_risk(NoBasisRisk())
    with pytest.raises(ProbabilitySemanticsError, match="assume_semantics"):
        plan.monte_carlo(100, seed=1).samples()
    stated = plan.monte_carlo(100, seed=1, assume_semantics="annual_exceedance")
    sample = stated.samples()["warehouse-a"]
    assert sample.provenance.semantics_source == "caller"
    assert sample.provenance.probability_semantics == "annual_exceedance"
    assert any("the caller stated" in note for note in sample.provenance.notes)
    explained: Any = stated.explain(format="json")
    assert "as the caller states" in setup_text(explained["setup"])


def test_other_semantics_are_refused(tmp_path: Path) -> None:
    path = _dataset(
        tmp_path / "percentile.parquet",
        metadata=_metadata(probability_semantics="within_period_percentile"),
    )
    plan = _plan(path).basis_risk(NoBasisRisk()).monte_carlo(100, seed=1)
    with pytest.raises(ProbabilitySemanticsError, match="not a distribution"):
        plan.samples()
    contradicting = _plan(_dataset(tmp_path / "declared.parquet")).monte_carlo(
        100, seed=1, assume_semantics="annual_value_distribution"
    )
    with pytest.raises(ProbabilitySemanticsError, match="contradicts"):
        contradicting.basis_risk(NoBasisRisk()).samples()


def test_a_pooled_ensemble_and_the_support_are_noted(tmp_path: Path) -> None:
    from crc_sdk.types import EnsembleDescriptor

    path = _dataset(
        tmp_path / "pooled.parquet",
        metadata=_metadata(
            ensemble=EnsembleDescriptor(pooling="pooled", members=("a", "b"))
        ),
    )
    plan = _plan(path).basis_risk(NoBasisRisk()).monte_carlo(2_000, seed=1)
    with pytest.warns(ReturnPeriodExtrapolationWarning):
        sample = plan.samples()["warehouse-a"]
    notes = " ".join(sample.provenance.notes)
    assert "pools an ensemble" in notes
    assert "more frequent than the shortest fitted return period" in notes
    assert sample.provenance.details["outside_support"]["below"] > 0


def test_several_curves_no_curve_and_no_data_raise(tmp_path: Path) -> None:
    two_horizons = _dataset(
        tmp_path / "two.parquet",
        _curve("warehouse-a"),
        _curve("warehouse-a", horizon=2080, source_id="later"),
    )
    request = (
        HazardDataset.local(two_horizons)
        .for_assets(_assets("warehouse-a"))
        .impact(RAMP, name="ramp", value_unit="1", value_semantics="damage ratio")
    )
    plan = UNTYPED_PLAN.for_evaluation(request).monte_carlo(10, seed=1)
    with pytest.raises(SampleError, match="one curve per asset"):
        plan.basis_risk(NoBasisRisk()).samples()
    empty = UNTYPED_PLAN.for_evaluation(request.select(horizons=[2100]))
    with pytest.raises(SampleError, match="no curve"):
        empty.monte_carlo(10, seed=1).basis_risk(NoBasisRisk()).samples()
    no_data = _dataset(
        tmp_path / "no-data.parquet",
        _curve(
            "warehouse-a",
            curve_kind="no_data",
            curve_type="insufficient_informative_support",
            curve_location=None,
            curve_scale=None,
        ),
    )
    dry = UNTYPED_PLAN.for_evaluation(
        _request(no_data, _assets("warehouse-a"))
    ).monte_carlo(10, seed=1)
    with pytest.raises(SampleError, match="no data"):
        dry.basis_risk(NoBasisRisk()).samples()


def test_a_cover_on_another_unit_is_refused_at_execution(dataset: Path) -> None:
    other = Cover(
        IndexDefinition("rx1day", "mm", aggregation="annual_maximum"),
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
    )
    plan = _plan(dataset).monte_carlo(10, seed=1).basis_risk(NoBasisRisk())
    planned = plan.covers(other)  # the dataset's unit is not known before execution
    with pytest.raises(SampleError, match="dataset's values are in 'mm/day'"):
        planned.write_parquet(dataset.parent / "never.parquet")
    assert not (dataset.parent / "never.parquet").exists()


# -- builders and explain -------------------------------------------------------


def test_builders_and_explain_do_no_io(tmp_path: Path) -> None:
    absent = tmp_path / "absent.parquet"
    plan = (
        _plan(absent)
        .monte_carlo(1_000, seed=42, outside_support="clamp")
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(FreshDraws(500, seed=43))
        .metrics(OPTIONS)
    )
    document: Any = plan.explain(format="json")
    plan.explain()
    assert "for_evaluation(absent.parquet, assets=2, draws=1000" in repr(plan)
    assert not absent.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == []
    assert document["entry"] == "for_evaluation"
    assert document["licences"] is None
    assert document["monte_carlo"] == {
        "assume_semantics": None,
        "fresh_draws": {"kind": "fresh_draws", "n_draws": 500, "seed": 43},
        "n_draws": 1_000,
        "outside_support": "clamp",
        "seed": 42,
    }
    assert document["rng"]["seed"] == 42
    assert document["rng"]["fresh_draws_seed"] == 43
    source = document["sources"][0]
    assert source["kind"] == "canonical_dataset"
    assert source["path"] == str(absent)
    assert source["selection"] == {
        "hazard_names": None,
        "horizons": [2050],
        "pathways": ["ssp245"],
    }
    assert source["impact"] == {
        "name": "ramp",
        "value_semantics": "damage ratio",
        "value_unit": "1",
    }
    setup = document["setup"]
    assert [s["asset_ids"] for s in setup["samples"]] == [["warehouse-a"], ["depot-b"]]
    assert {s["source_kind"] for s in setup["samples"]} == {"monte_carlo"}
    assert {s["draws"] for s in setup["samples"]} == {1_000}
    assert [(e["split"], e["method"], e["details"]) for e in setup["evaluations"]] == [
        ("in_sample", "in_sample", {"plan_seed": 42}),
        ("out_of_sample", "fresh_draws", {"fresh_draws_seed": 43, "plan_seed": 42}),
    ]
    text = setup_text(setup)
    assert "Split: out of sample, fold 0, on fresh Monte Carlo draws" in text
    assert "with the probability semantics the dataset declares, not known yet" in text
    assert "impact noise, a mean-one lognormal factor with sigma 0.3" in text


def test_for_evaluation_checks_its_request(dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="PortfolioEvaluation"):
        UNTYPED_PLAN.for_evaluation(HazardDataset.local(dataset))
    remote = HazardDataset.crc_open(fixtures=tmp_path).for_assets(_assets())
    with pytest.raises(TypeError, match="materialize"):
        UNTYPED_PLAN.for_evaluation(remote)
    bare = HazardDataset.local(dataset).for_assets(_assets())
    with pytest.raises(ValueError, match="no impact"):
        UNTYPED_PLAN.for_evaluation(bare)
    with pytest.raises(ValueError, match="one curve per asset"):
        UNTYPED_PLAN.for_evaluation(_request(dataset, horizons=[2050, 2080]))
    sql = HazardDataset.local(dataset).for_assets(
        AssetPortfolio("SELECT 1", location=PointColumns())
    )
    with pytest.raises(TypeError, match="not SQL"):
        UNTYPED_PLAN.for_evaluation(
            sql.impact(RAMP, name="r", value_unit="1", value_semantics="d")
        )
    with pytest.raises(SampleError, match="no columns"):
        UNTYPED_PLAN.for_evaluation(
            _request(dataset), value_column="value_usd", value_unit="USD"
        )
    with pytest.raises(SampleError, match="damage ratio"):
        UNTYPED_PLAN.for_evaluation(_request(dataset), value_unit="EUR")


def test_monte_carlo_and_fresh_draws_are_checked(dataset: Path) -> None:
    plan = _plan(dataset).basis_risk(NoBasisRisk()).covers(_linear())
    with pytest.raises(ValueError, match="monte_carlo"):
        plan.samples()
    with pytest.raises(ValueError, match="n_draws"):
        plan.monte_carlo(0, seed=1)
    with pytest.raises(ValueError, match="seed"):
        plan.monte_carlo(10, seed=-1)
    with pytest.raises(ProbabilitySemanticsError, match="assume_semantics"):
        plan.monte_carlo(10, seed=1, assume_semantics="projection_uncertainty")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside_support"):
        plan.monte_carlo(10, seed=1, outside_support="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="another seed"):
        plan.monte_carlo(10, seed=4).split(FreshDraws(10, seed=4))
    with pytest.raises(ValueError, match="another seed"):
        plan.split(FreshDraws(10, seed=4)).monte_carlo(10, seed=4)
    with pytest.raises(ValueError, match="positive integer"):
        FreshDraws(0, seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        FreshDraws(10, seed=-2)
    with pytest.raises(TypeError, match="rule must be"):
        plan.split("fresh")  # type: ignore[arg-type]
    # Plans on per-year values take neither.
    annual = UNTYPED_PLAN.for_annual_extremes(
        pa.table(
            {"year": [2000], "latitude": [50.9], "longitude": [6.9], "value": [1.0]}
        ),
        _assets(),
        impact=RAMP,
        unit="mm/day",
        provenance=PROVENANCE,
    )
    with pytest.raises(ValueError, match="per-year values"):
        annual.monte_carlo(10, seed=1)
    with pytest.raises(TypeError, match="fresh draws need"):
        annual.split(FreshDraws(10, seed=1))
    # ImpactNoise needs no seed before monte_carlo on curves: monte_carlo states it.
    noisy = _plan(dataset).basis_risk(ImpactNoise(0.2)).monte_carlo(10, seed=9)
    assert noisy.seed == 9


def test_assets_from_a_parquet_file_are_read_at_execution(
    dataset: Path, tmp_path: Path
) -> None:
    assets_file = tmp_path / "assets.parquet"
    pq.write_table(_assets(), assets_file)
    request = _request(dataset, Path(assets_file))
    plan = UNTYPED_PLAN.for_evaluation(
        request, value_column="value_eur", value_unit="EUR"
    )
    document: Any = plan.monte_carlo(50, seed=1).explain(format="json")
    assert document["assets"]["path"] == str(assets_file)
    assert document["assets"]["count"] is None
    samples = plan.monte_carlo(50, seed=1).basis_risk(NoBasisRisk()).samples()
    assert list(samples) == ["warehouse-a", "depot-b"]
