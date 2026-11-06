"""ParametricPlan: builders, explain(), seeding, splits and the written output."""

from __future__ import annotations

import json
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

from crc_sdk.geometry import point_to_cell
from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    AtStation,
    Budget,
    ComonotoneIndexWarning,
    Cover,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    Holdout,
    ImpactNoise,
    IndexDefinition,
    KFold,
    LinearPayout,
    LossModel,
    MetricOptions,
    NearestCell,
    NoBasisRisk,
    ParametricPlan,
    SampleError,
    StationColumn,
    StepPayout,
    WeightedCVaR,
    compare,
    design,
    holdout,
    sample_annual,
    setup_text,
)
from crc_sdk.types import SourceProvenance
from crc_sdk.workflows import AssetPortfolio, CellColumn, ImpactContextColumns

UNTYPED_PLAN: Any = ParametricPlan
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
PROVENANCE = SourceProvenance(
    provider="example",
    dataset="synthetic rx1day",
    version="1",
    licence="CC0-1.0",
    attribution="Synthetic annual maxima",
)
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
CELLS = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
YEARS = np.arange(1981, 2021)
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))


def _annual(seed: int = 1, years: Any = YEARS) -> Any:
    """A per-cell annual-extremes table in the layout of ERA5's and CMIP6's."""
    rng = np.random.default_rng(seed)
    n = len(years) * len(CELLS)
    return pa.table(
        {
            "year": pa.array(np.repeat(years, len(CELLS)), pa.int32()),
            "latitude": np.tile([lat for lat, _ in CELLS], len(years)),
            "longitude": np.tile([lon for _, lon in CELLS], len(years)),
            "value": rng.gumbel(50.0, 12.0, n),
        }
    )


def _assets(*ids: str) -> Any:
    ids = ids or ("warehouse-a", "depot-b")
    points = {"warehouse-a": (6.96, 50.94), "depot-b": (7.10, 51.10)}
    return pa.table(
        {
            "asset_id": list(ids),
            "longitude": [points.get(i, (6.9, 50.9))[0] for i in ids],
            "latitude": [points.get(i, (6.9, 50.9))[1] for i in ids],
            "value_eur": [2e6 + 1e5 * k for k in range(len(ids))],
        }
    )


def _plan(source: Any = None, assets: Any = None, **options: Any) -> ParametricPlan:
    arguments: dict[str, Any] = {
        "impact": RAMP,
        "unit": "mm/day",
        "provenance": PROVENANCE,
        "value_column": "value_eur",
        "value_unit": "EUR",
        "seed": 42,
    }
    arguments.update(options)
    plan: ParametricPlan = UNTYPED_PLAN.for_annual_extremes(
        _annual() if source is None else source,
        _assets() if assets is None else assets,
        **arguments,
    )
    return plan


def _step(label: str | None = "step", asset_id: str | None = None) -> Cover:
    return Cover(
        INDEX,
        StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        asset_id=asset_id,
        label=label,
    )


def _label(cover: Cover) -> str:
    return cover.definition_hash.removeprefix("sha256:")[:12]


def _linear(asset_id: str | None = None) -> Cover:
    return Cover(
        INDEX,
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
        asset_id=asset_id,
        label="linear",
    )


class _SourcePlan:
    """A stand-in for a crc-sdk plan with an annual_extremes() execution point."""

    def __init__(self, table: Any, unit: str = "mm/day") -> None:
        self.table = table
        self.unit = unit
        self.calls: list[str] = []

    def annual_extremes(self) -> Iterator[Any]:
        self.calls.append("annual_extremes")
        return iter(self.table.to_batches(max_chunksize=50))

    def explain(self, *, format: str = "text") -> Any:
        self.calls.append("explain")
        return {"recipe": "rx1day", "unit": self.unit, "years": [1981, 2020]}


SERIES = (
    "the hazard series from provider example and dataset synthetic rx1day, with "
    "horizon and pathway not recorded"
)


def _lines(text: str) -> dict[str, list[str]]:
    """The lines of a setup text by their label (`Split`, `Source`, ...)."""
    lines: dict[str, list[str]] = {}
    for line in text.splitlines():
        lines.setdefault(line.split(":")[0], []).append(line)
    return lines


@pytest.fixture(autouse=True)
def _quiet_comonotone() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        yield


# -- builders and explain ------------------------------------------------------


def test_builders_return_new_plans_and_explain_does_no_io(tmp_path: Path) -> None:
    source = _SourcePlan(_annual())
    base = _plan(source)
    plan = (
        base.basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    assert base.basis_risk_model is None and base.cover_list == ()
    assert plan.split_rule == Holdout(0.7)
    plan.explain()
    plan.explain(format="json")
    repr(plan)
    assert source.calls == []
    # Files are not opened by builders or explain() either.
    missing = tmp_path / "absent.parquet"
    on_files = _plan(missing, tmp_path / "assets.parquet").basis_risk(NoBasisRisk())
    document: Any = on_files.covers(_linear()).explain(format="json")
    assert document["sources"][0]["kind"] == "file"
    assert document["sources"][0]["path"] == str(missing)
    assert _lines(setup_text(document))["Source"] == [
        f"Source: {SERIES}, read from a file {missing}."
    ]
    assert document["assets"]["count"] is None
    assert not missing.exists()


def test_explain_mirrors_the_written_metadata() -> None:
    plan = (
        _plan()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear())
        .objective(MSE(), WeightedCVaR(alpha=0.9, beta=0.9), budget=Budget(ratio=1.0))
        .split(KFold(4))
        .metrics(OPTIONS)
    )
    document: Any = plan.explain(format="json")
    assert set(document) == {
        "assets",
        "covers",
        "designs",
        "entry",
        "execution",
        "licences",
        "producer",
        "rng",
        "setup",
        "sources",
    }
    setup = document["setup"]
    assert setup["basis"] == "annual"
    assert setup["metric_options"] == OPTIONS.to_document()
    assert [sample["asset_ids"] for sample in setup["samples"]] == [
        ["warehouse-a"],
        ["depot-b"],
    ]
    sample = setup["samples"][0]
    assert sample["basis_risk"] == {"kind": "impact_noise", "sigma": 0.3, "upper": 1.0}
    assert sample["loss_model"]["value"] == {
        "asset_value_hash": None,
        "column": "value_eur",
        "source": "column",
        "unit": "EUR",
    }
    assert [(e["split"], e["fold"], e["method"]) for e in setup["evaluations"]] == [
        (split, fold, method)
        for fold in range(4)
        for split, method in (("in_sample", "in_sample"), ("out_of_sample", "kfold"))
    ]
    assert setup["evaluations"][1]["details"] == {"k": 4, "plan_seed": 42}
    assert setup["evaluations"][0]["details"] == {"plan_seed": 42}
    assert document["designs"]["objectives"] == [
        {"kind": "mse"},
        {"alpha": 0.9, "beta": 0.9, "kind": "weighted_cvar"},
    ]
    assert document["designs"]["budget"] == {"ratio": 1.0}
    assert document["designs"]["templates"] == [_step().definition_hash]
    assert [cover["free_parameters"] for cover in document["covers"]] == [
        ["levels[0]", "levels[1]", "levels[2]"],
        [],
    ]
    assert document["licences"] == [
        {
            "attribution": "Synthetic annual maxima",
            "licence": "CC0-1.0",
            "role": "hazard",
            "source": "synthetic rx1day",
        }
    ]
    assert document["rng"]["seed"] == 42
    assert document["sources"][0]["provenance"] == PROVENANCE.model_dump(mode="json")
    assert document["sources"][0]["rows"] == len(YEARS) * len(CELLS)
    assert json.loads(str(plan.explain())) == document
    with pytest.raises(ValueError, match="text"):
        UNTYPED_PLAN.explain(plan, format="yaml")


def test_setup_text_states_the_plan_before_execution() -> None:
    plan = (
        _plan()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    document: Any = plan.explain(format="json")
    text = setup_text(document["setup"])
    assert "Level: asset, one row per asset (warehouse-a and depot-b)" in text
    assert "Basis: annual, one draw is one year." in text
    assert "Split: out of sample, fold 0, a holdout with train fraction 0.7" in text
    assert "not known yet" in text
    assert "impact noise, a mean-one lognormal factor with sigma 0.3" in text
    assert "times the asset value in column value_eur, in EUR" in text
    unset: Any = _plan().explain(format="json")
    assert "Basis risk: not stated." in setup_text(unset["setup"])
    # The whole document names the source too: the provider and dataset the plan
    # states, and the table or source plan the values are read from.
    rows = len(YEARS) * len(CELLS)
    source = f"Source: {SERIES}, read from a table of {rows} rows."
    lines = text.splitlines()
    after = max(i for i, line in enumerate(lines) if line.startswith("Sample"))
    assert setup_text(document).splitlines() == [
        *lines[: after + 1],
        source,
        *lines[after + 1 :],
    ]
    planned: Any = _plan(_SourcePlan(_annual())).explain(format="json")
    assert _lines(setup_text(planned))["Source"] == [
        f"Source: {SERIES}, read from a source plan _SourcePlan."
    ]


def test_explain_without_a_split_lists_in_sample_rows_only() -> None:
    document: Any = _plan().basis_risk(NoBasisRisk()).explain(format="json")
    assert [(e["split"], e["fold"]) for e in document["setup"]["evaluations"]] == [
        ("in_sample", 0)
    ]
    assert document["designs"] is None
    assert document["setup"]["evaluations"][0]["pricing"] == {"kind": "pure"}


def test_the_entry_point_checks_its_arguments() -> None:
    with pytest.raises(TypeError, match="annual_extremes"):
        _plan(source=[1, 2, 3])
    with pytest.raises(TypeError, match="SourceProvenance"):
        _plan(provenance={"licence": "CC0-1.0"})
    calibration = ImpactCalibration(
        exposure_variable="flood_depth",
        exposure_unit="m",
        impact_metric="damage_ratio",
        impact_unit="1",
    )
    depth = ImpactExpression.tabulated(
        [(0.0, 0.0), (2.0, 1.0)], calibration=calibration
    )
    with pytest.raises(SampleError, match="calibrated"):
        _plan(impact=depth)
    with pytest.raises(SampleError, match="no columns"):
        _plan(value_column="replacement_value")
    with pytest.raises(SampleError, match="no columns"):
        _plan(location=StationColumn("station_id"))
    with pytest.raises(TypeError, match="NearestCell"):
        _plan(location="nearest")
    with pytest.raises(ValueError, match="years"):
        _plan(years=[2000, 2000])
    with pytest.raises(ValueError, match="missing"):
        _plan(missing="zero")
    with pytest.raises(ValueError, match="seed"):
        _plan(seed=-1)
    with pytest.raises(ValueError, match="spacing"):
        NearestCell(spacing=(0.25, 0.0))
    with pytest.raises(ValueError, match="column"):
        StationColumn("")


def test_nearest_cell_needs_point_assets() -> None:
    cell = point_to_cell(6.96, 50.94, 7)
    assets = AssetPortfolio(
        pa.table({"asset_id": ["a"], "cell_index": pa.array([cell], pa.uint64())}),
        location=CellColumn(),
    )
    with pytest.raises(SampleError, match="PointColumns"):
        _plan(assets=assets, value_column=None, value_unit="1")


def test_a_registry_impact_that_needs_a_cell_is_refused_without_one() -> None:
    rx1day = framework_impacts.for_factor("rx1day")
    with pytest.raises(SampleError, match="H3 cell"):
        _plan(impact=rx1day, value_column=None, value_unit="1")
    cell = point_to_cell(6.96, 50.94, 7)
    with_cell = framework_impacts.for_factor(
        "rx1day", context=TransformContext(cell=cell)
    )
    plan = _plan(impact=with_cell, value_column=None, value_unit="1")
    samples = plan.basis_risk(NoBasisRisk()).samples()
    assert all(float(sample.loss.max()) > 0.0 for sample in samples.values())
    fixed = _plan(
        impact=rx1day,
        value_column=None,
        value_unit="1",
        context=TransformContext(cell=cell),
    )
    assert fixed.loss_model.context == TransformContext(cell=cell)
    # Assets with an H3 cell column fill the cell when context columns are given.
    stations = pa.table(
        {
            "station": ["S1"] * 30,
            "year": pa.array(range(1991, 2021), pa.int32()),
            "value": np.linspace(30.0, 150.0, 30),
        }
    )
    located = AssetPortfolio(
        pa.table(
            {
                "asset_id": ["a"],
                "cell_index": pa.array([cell], pa.uint64()),
                "station_id": ["S1"],
                "country": ["DEU"],
            }
        ),
        location=CellColumn(),
    )
    by_cell = _plan(
        stations,
        located,
        impact=rx1day,
        value_column=None,
        value_unit="1",
        context=ImpactContextColumns(country="country"),
        location=StationColumn("station_id"),
    )
    sample = by_cell.basis_risk(NoBasisRisk()).samples()["a"]
    expected = with_cell.evaluate(np.linspace(30.0, 150.0, 30))
    np.testing.assert_allclose(sample.loss[:, 0], expected)


def test_basis_risk_is_required_and_noise_needs_a_seed() -> None:
    plan = _plan().covers(_linear())
    with pytest.raises(ValueError, match="basis-risk"):
        plan.samples()
    with pytest.raises(ValueError, match="basis-risk"):
        plan.write_parquet("never.parquet")
    with pytest.raises(ValueError, match="seed"):
        _plan(seed=None).basis_risk(ImpactNoise(0.3))
    with pytest.raises(TypeError, match="BasisRisk"):
        UNTYPED_PLAN.basis_risk(plan, "none")


def test_covers_are_checked_against_the_plans_units_and_location() -> None:
    plan = _plan()
    with pytest.raises(SampleError, match="pays in"):
        plan.covers(Cover(INDEX, LinearPayout(60.0, 120.0, 1.0), payout_unit="USD"))
    with pytest.raises(SampleError, match="index in"):
        plan.covers(
            Cover(
                IndexDefinition("rx1day", "mm", aggregation="annual_maximum"),
                LinearPayout(60.0, 120.0, 1.0),
                payout_unit="EUR",
            )
        )
    at_station = IndexDefinition(
        "rx1day", "mm/day", location=AtStation("X1"), aggregation="annual_maximum"
    )
    with pytest.raises(SampleError, match="AtAsset"):
        plan.covers(
            Cover(at_station, LinearPayout(60.0, 120.0, 1.0), payout_unit="EUR")
        )
    with pytest.raises(ValueError, match="twice"):
        plan.covers(_linear(), _linear())
    with pytest.raises(ValueError, match="at least one"):
        plan.covers()
    with pytest.raises(TypeError, match="Cover"):
        UNTYPED_PLAN.covers(plan, "cover")


def test_objectives_and_splits_are_checked() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="twice"):
        plan.objective(MSE(), MSE())
    with pytest.raises(ValueError, match="at least one"):
        plan.objective()
    with pytest.raises(TypeError, match="objectives"):
        UNTYPED_PLAN.objective(plan, "mse")
    with pytest.raises(TypeError, match="pricing"):
        UNTYPED_PLAN.objective(plan, MSE(), pricing="pure")
    with pytest.raises(TypeError, match="Holdout"):
        UNTYPED_PLAN.split(plan, 0.7)
    with pytest.raises(TypeError, match="MetricOptions"):
        UNTYPED_PLAN.metrics(plan, {"betas": (0.9,)})
    with pytest.raises(ValueError, match="train_fraction"):
        Holdout(1.0)
    with pytest.raises(ValueError, match="k must"):
        KFold(1)


# -- samples and seeding -------------------------------------------------------


def test_samples_match_the_function_route() -> None:
    table = _annual()
    plan = _plan(table).basis_risk(ImpactNoise(0.3, upper=1.0))
    samples = plan.samples()
    assert list(samples) == ["warehouse-a", "depot-b"]
    children = np.random.SeedSequence(42).spawn(2)
    assets = AssetPortfolio(_assets())
    for (asset_id, sample), child, (longitude, latitude) in zip(
        samples.items(), children, [(6.96, 50.94), (7.10, 51.10)], strict=True
    ):
        series = AnnualSeries.nearest_cell(
            table,
            longitude=longitude,
            latitude=latitude,
            unit="mm/day",
            source=PROVENANCE,
        )
        loss_model = LossModel(RAMP, value_column="value_eur", value_unit="EUR")
        expected = sample_annual(
            series,
            loss_model=loss_model.for_asset(assets, asset_id),
            basis_risk=ImpactNoise(0.3, upper=1.0),
            asset_id=asset_id,
            rng=np.random.default_rng(child),
        )
        assert sample.content_hash == expected.content_hash
        assert sample.period.tolist() == YEARS.tolist()
        assert sample.provenance.sources[0]["series"]["source"] == (
            PROVENANCE.model_dump(mode="json")
        )
        assert sample.provenance.licences[0]["licence"] == "CC0-1.0"
    again = plan.samples()
    assert [s.content_hash for s in again.values()] == [
        s.content_hash for s in samples.values()
    ]


def test_each_asset_draws_from_its_own_child_generator() -> None:
    plan = _plan().basis_risk(ImpactNoise(0.3, upper=1.0))
    both = plan.samples()
    first = _plan(assets=_assets("warehouse-a"))
    first_only = first.basis_risk(ImpactNoise(0.3, upper=1.0)).samples()
    # Another asset after it changes nothing for the first asset's draws.
    assert first_only["warehouse-a"].content_hash == both["warehouse-a"].content_hash
    other_seed = _plan(seed=7).basis_risk(ImpactNoise(0.3, upper=1.0)).samples()
    assert other_seed["warehouse-a"].content_hash != both["warehouse-a"].content_hash
    without_seed = _plan(seed=None).basis_risk(NoBasisRisk()).samples()
    assert without_seed["warehouse-a"].provenance.rng == ()


def test_stations_are_matched_by_an_asset_column() -> None:
    rng = np.random.default_rng(3)
    stations = pa.table(
        {
            "station": ["S1"] * 30 + ["S2"] * 30,
            "year": pa.array(list(range(1991, 2021)) * 2, pa.int32()),
            "value": rng.gumbel(50.0, 12.0, 60),
        }
    )
    assets = pa.table(
        {
            "asset_id": ["a", "b"],
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 1.0],
            "station_id": ["S2", "S1"],
        }
    )
    plan = _plan(
        stations,
        assets,
        value_column=None,
        value_unit="1",
        location=StationColumn("station_id"),
    ).basis_risk(NoBasisRisk())
    samples = plan.samples()
    expected = AnnualSeries.from_table(stations, where={"station": "S2"}, unit="mm/day")
    np.testing.assert_array_equal(samples["a"].index[:, 0], expected.values)
    series = samples["b"].provenance.sources[0]["series"]
    assert series["where"] == {"station": "S1"}


def test_missing_years_raise_unless_dropped() -> None:
    table = _annual()
    keep = np.ones(table.num_rows, dtype=bool)
    keep[np.flatnonzero(table.column("year").to_numpy() == 1990)[:1]] = False
    gappy = table.filter(pa.array(keep))
    plan = _plan(gappy, _assets("warehouse-a")).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="drop_year"):
        plan.samples()
    dropped = _plan(gappy, _assets("warehouse-a"), missing="drop_year")
    sample = dropped.basis_risk(NoBasisRisk()).samples()["warehouse-a"]
    assert 1990 not in sample.period.tolist() and len(sample) == len(YEARS) - 1
    assert list(sample.provenance.details["dropped_years"]) == [1990]


def test_an_asset_outside_the_grid_raises() -> None:
    far = pa.table(
        {
            "asset_id": ["far"],
            "longitude": [12.0],
            "latitude": [40.0],
            "value_eur": [1.0],
        }
    )
    with pytest.raises(SampleError, match="half a grid step"):
        _plan(assets=far).basis_risk(NoBasisRisk()).samples()


def test_repeated_assets_raise() -> None:
    assets = pa.concat_tables([_assets("warehouse-a"), _assets("warehouse-a")])
    with pytest.raises(SampleError, match="more than once"):
        _plan(assets=assets).basis_risk(NoBasisRisk()).samples()


# -- covers, designs and the comparison ---------------------------------------


def test_designs_follow_the_function_route() -> None:
    table = _annual()
    plan = (
        _plan(table)
        .basis_risk(NoBasisRisk())
        .covers(_step(), _linear())
        .objective(MSE(), ExpectedUncovered(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
    )
    designs = plan.designs()
    assert [(d.cover.asset_id, d.cover.label) for d in designs] == [
        ("warehouse-a", "step (mse)"),
        ("warehouse-a", "step (expected_uncovered)"),
        ("depot-b", "step (mse)"),
        ("depot-b", "step (expected_uncovered)"),
    ]
    sample = plan.samples()["warehouse-a"]
    train, _ = holdout(sample, 0.7)
    bound = Cover(
        INDEX,
        _step().payout,
        payout_unit="EUR",
        asset_id="warehouse-a",
        label="step",
    )
    expected = design(bound, train, MSE(), budget=Budget(ratio=1.0))
    assert designs[0].design_id == expected.design_id
    assert designs[0].cover.definition_hash == expected.cover.definition_hash
    assert designs[0].design_sample.content_hash == train.content_hash


def test_designs_need_an_objective_and_covers() -> None:
    plan = _plan().basis_risk(NoBasisRisk()).covers(_step())
    with pytest.raises(ValueError, match="objective"):
        plan.designs()
    with pytest.raises(ValueError, match="no covers"):
        _plan().basis_risk(NoBasisRisk()).objective(MSE()).designs()
    with pytest.raises(ValueError, match="free parameters"):
        plan.write_parquet("never.parquet")


def test_covers_are_bound_to_their_assets() -> None:
    twice = _plan().basis_risk(NoBasisRisk()).covers(_linear(), _linear("depot-b"))
    with pytest.raises(ValueError, match="applies to asset 'depot-b' twice"):
        # The cover written for no asset, bound to depot-b, is the second one.
        twice.write_parquet("never.parquet")
    unknown = _plan().basis_risk(NoBasisRisk()).covers(_linear(asset_id="elsewhere"))
    with pytest.raises(ValueError, match="do not hold"):
        unknown.write_parquet("never.parquet")
    only_b = _plan().basis_risk(NoBasisRisk()).covers(_linear("depot-b"))
    with pytest.raises(ValueError, match="has no cover"):
        only_b.write_parquet("never.parquet")


def test_write_parquet_writes_every_split_with_its_setup(tmp_path: Path) -> None:
    other = Cover(
        INDEX, LinearPayout(70.0, 130.0, 4e5), payout_unit="EUR", asset_id="depot-b"
    )
    plan = (
        _plan()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear(), other)
        .objective(MSE(), WeightedCVaR(alpha=0.9, beta=0.9), budget=Budget(ratio=1.0))
        .split(KFold(4))
        .metrics(OPTIONS)
    )
    output = tmp_path / "nested" / "comparison.parquet"
    result = plan.write_parquet(output)
    assert result.output == output and output.is_file()
    assert [path.name for path in output.parent.iterdir()] == ["comparison.parquet"]
    table = pq.read_table(output)
    assert table.num_rows == result.row_count
    # Per fold and split: warehouse-a has 2 designs and 1 cover, depot-b 2 and 2.
    assert result.row_count == 4 * 2 * (3 + 4)
    rows = table.select(["asset_id", "cover_label", "split", "fold"]).to_pylist()
    labels = {(row["asset_id"], row["cover_label"]) for row in rows}
    assert ("warehouse-a", "linear") in labels
    assert ("depot-b", "linear") in labels
    assert {r["asset_id"] for r in rows if r["cover_label"] == _label(other)} == {
        "depot-b"
    }
    out = table.filter(pa.compute.equal(table.column("split"), "out_of_sample"))
    designed = out.filter(pa.compute.is_valid(out.column("design_id")))
    assert set(designed.column("premium_basis").to_pylist()) == {"design"}
    fixed = out.filter(pa.compute.is_null(out.column("design_id")))
    assert set(fixed.column("premium_basis").to_pylist()) == {"evaluation"}
    assert len(result.designs) == 4 * 2 * 2
    assert set(result.samples) == {"warehouse-a", "depot-b"}

    metadata = EvaluationMetadata.from_table(table)
    assert metadata == result.metadata
    assert metadata.assets["count"] == 2
    assert metadata.assets["id_column"] == "asset_id"
    assert str(metadata.assets["values_sha256"]).startswith("sha256:")
    assert "2000000.0" not in metadata.to_json()  # asset values are not written
    methods = {(e["split"], e["method"]) for e in metadata.setup.evaluations}
    assert methods == {("in_sample", "in_sample"), ("out_of_sample", "kfold")}
    assert all(e["details"]["plan_seed"] == 42 for e in metadata.setup.evaluations)
    text = setup_text(metadata)
    assert "fold 3 of a 4-fold split over blocks of years" in text
    assert "the leakage check passed" in text
    assert "Design step (mse) (" in text
    hazard = [entry for entry in metadata.sources if entry["role"] == "hazard"]
    assert hazard and all(
        entry["annual_table"]["kind"] == "table"
        and entry["annual_table"]["table_sha256"].startswith("sha256:")
        for entry in hazard
    )
    assert metadata.licences[0]["licence"] == "CC0-1.0"
    assert len({entry["sample_hash"] for entry in metadata.rng}) == len(
        metadata.setup.samples
    )
    assert len(metadata.designs) == len(result.designs)


def test_without_a_split_every_row_is_in_sample(tmp_path: Path) -> None:
    result = (
        _plan(assets=_assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_linear(), _step())
        .objective(ExpectedUncovered(), budget=Budget(ratio=1.0))
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "in-sample.parquet")
    )
    assert set(result.table.column("split").to_pylist()) == {"in_sample"}
    details = [e["details"] for e in result.metadata.setup.evaluations]
    assert details == [{"plan_seed": 42}]
    assert result.row_count == 2
    sample = result.samples["warehouse-a"]
    assert result.designs[0].design_sample.content_hash == sample.content_hash


def test_a_written_file_is_replaced_atomically(tmp_path: Path) -> None:
    output = tmp_path / "comparison.parquet"
    output.write_bytes(b"previous")
    plan = (
        _plan(assets=_assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    plan.write_parquet(output)
    assert pq.read_table(output).num_rows == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["comparison.parquet"]
    with pytest.raises(ValueError, match="local"):
        plan.write_parquet("s3://bucket/comparison.parquet")
    unseeded = _plan(assets=_assets("warehouse-a"), seed=None)
    written = (
        unseeded.basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .write_parquet(output)
    )
    assert [e["details"] for e in written.metadata.setup.evaluations] == [{}]


def test_file_sources_and_assets_are_read_at_execution(tmp_path: Path) -> None:
    source, assets = tmp_path / "annual.parquet", tmp_path / "assets.parquet"
    plan = (
        _plan(source, assets)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    pq.write_table(_annual(), source)
    pq.write_table(_assets(), assets)
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    record = next(e for e in result.metadata.sources if e["role"] == "hazard")
    assert record["annual_table"]["path"] == str(source)
    assert record["annual_table"]["file_sha256"].startswith("sha256:")
    series_source = record["series"]["source"]
    assert series_source["file_sha256"] == record["annual_table"]["file_sha256"]
    assert series_source["licence"] == "CC0-1.0"
    assert result.row_count == 2


def test_a_source_plan_is_executed_once_and_explained(tmp_path: Path) -> None:
    source = _SourcePlan(_annual())
    plan = _plan(source).basis_risk(NoBasisRisk()).covers(_linear()).metrics(OPTIONS)
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    assert source.calls == ["annual_extremes", "explain"]
    record = next(e for e in result.metadata.sources if e["role"] == "hazard")
    assert record["annual_table"]["explain"]["recipe"] == "rx1day"
    assert record["series"]["source"]["source_type"] == "_SourcePlan"
    with pytest.raises(SampleError, match="unit"):
        _plan(_SourcePlan(_annual(), unit="mm")).basis_risk(NoBasisRisk()).samples()


def test_out_of_sample_rows_match_compare_on_the_held_out_years(
    tmp_path: Path,
) -> None:
    plan = (
        _plan(assets=_assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_step(), _linear())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    sample = result.samples["warehouse-a"]
    train, test = holdout(sample, 0.7)
    by_hand = compare(
        [result.designs[0], _linear("warehouse-a")],
        test,
        split="out_of_sample",
        design_sample=train,
        options=OPTIONS,
    )
    written = result.table.filter(
        pa.compute.equal(result.table.column("split"), "out_of_sample")
    )
    for name in ("cover_hash", "sample_hash", "expected_uncovered", "premium", "rmse"):
        assert written.column(name).to_pylist() == by_hand.column(name).to_pylist()


def test_the_repr_is_short() -> None:
    plan = _plan().covers(_step()).objective(MSE()).split(Holdout(0.7))
    text = repr(plan)
    assert text.startswith("ParametricPlan.for_annual_extremes(Table, assets=2")
    assert "objectives=1" in text and len(text) < 200
