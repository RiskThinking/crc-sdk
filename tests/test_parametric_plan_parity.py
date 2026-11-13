"""Parity of the plan on fitted curves with the golden k-fold and fresh-draw cases.

The cases in `tests/fixtures/parametric/cases/` give an outcome sample, a
template, its design terms, and the design and metrics the module must
reproduce, in and out of sample (`tests/test_parametric_design.py` checks
them on the function route). Here the curve route of `ParametricPlan`
reproduces the k-fold and fresh-draw cases end to end: the plan draws from a
synthetic canonical curve, and a basis-risk model of the test, written to the
`BasisRisk` protocol, replaces each draw's index and loss with the case's.
The plan then splits, designs, prices and evaluates the case's draws, so its
designs and its rows must be the case's: `KFold(5)` must hold out the case's
block, and `FreshDraws` must evaluate on the case's fresh draws.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact
from numpy.typing import NDArray

from crc_sdk.connectors import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.geometry import point_to_cell
from crc_sdk.parametric import (
    Cover,
    FreshDraws,
    KFold,
    ParametricPlan,
    design,
)
from crc_sdk.parametric.constraints import budget_from_document
from crc_sdk.parametric.objectives import objective_from_document
from crc_sdk.parametric.pricing import pricing_from_document
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import HazardDataset, ReturnPeriodExtrapolationWarning

CASES = Path(__file__).parent / "fixtures" / "parametric" / "cases"
POINT = (-80.2, 25.8)
UNITLESS = (
    "coverage_ratio",
    "payout_to_loss_ratio",
    "correlation",
    "tail_coverage_",
    "cvar_uncovered_to_loss_",
    "false_negative_rate_",
    "false_positive_rate_",
)
PLAN: Any = ParametricPlan


def _cases(kind: str) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(CASES.glob("*.json")):
        case: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if (case["split"] or {}).get("kind") == kind:
            cases.append(case)
    return cases


KFOLD_CASES = _cases("kfold")
FRESH_CASES = _cases("fresh_draws")


def _column(block: Mapping[str, Any], name: str) -> NDArray[np.float64]:
    """A block's one-asset column (`index` or `loss`), as a vector."""
    return np.asarray(block[name], dtype=np.float64)[:, 0]


def _full(case: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
    """A k-fold case's full sample: both blocks, in period order."""
    design_block, held_out = case["sample"], case["evaluation_sample"]
    period = np.asarray(design_block["period"] + held_out["period"])
    order = np.argsort(period, kind="stable")
    return {
        name: np.concatenate([_column(design_block, name), _column(held_out, name)])[
            order
        ]
        for name in ("index", "loss")
    }


class _CaseDraws:
    """A user basis-risk model that gives each draw the case's index and loss.

    `draws` maps a number of draws to the index and loss of the case's draws
    of that length: the design (or full) sample's and the fresh draws'. The
    asset's value is 1, so the impact is the loss.
    """

    def __init__(self, draws: Mapping[int, Mapping[str, NDArray[np.float64]]]):
        self.draws = draws

    @property
    def name(self) -> str:
        return "case_draws"

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.name, "draws": sorted(self.draws)}

    def apply(
        self,
        index: NDArray[np.float64],
        hazard: NDArray[np.float64],
        impact: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        del hazard, impact, rng
        draws = self.draws[len(index)]
        return draws["index"].copy(), draws["loss"].copy()


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReturnPeriodExtrapolationWarning)
        yield


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A one-curve canonical dataset of the cases' index, in m/s."""
    metadata = HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="m/s",
        value_semantics="annual maximum wind speed",
        producer="tests",
        source=SourceProvenance(
            provider="example",
            dataset="synthetic wind",
            version="1",
            licence="CC0-1.0",
            attribution="Synthetic curves",
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 500.0),
    )
    row = {
        "cell_index": point_to_cell(*POINT, 7),
        "source_id": "fixture",
        "source_geometry": None,
        "hazard_name": "synthetic_wind",
        "horizon": 2050,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 30.0,
        "curve_scale": 8.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }
    path = tmp_path_factory.mktemp("parity") / "wind.parquet"
    table = pa.Table.from_pylist([row], schema=hazard_arrow_schema(metadata))
    write_hazard_dataset(table, path, metadata, max_workers=1)
    return path


def _plan(
    dataset: Path, case: Mapping[str, Any], draws: _CaseDraws, n_draws: int
) -> ParametricPlan:
    (asset_id,) = case["sample"]["asset_ids"]
    assets = pa.table(
        {
            "asset_id": [asset_id],
            "longitude": [POINT[0]],
            "latitude": [POINT[1]],
            "value_usd": [1.0],
        }
    )
    request = (
        HazardDataset.local(dataset)
        .for_assets(assets)
        .select(horizons=[2050], pathways=["ssp245"])
        .impact(
            PiecewiseLinearImpact([20.0, 60.0], [0.0, 1.0]),
            name="ramp",
            value_unit="1",
            value_semantics="damage ratio",
        )
    )
    plan: ParametricPlan = (
        PLAN.for_evaluation(request, value_column="value_usd", value_unit="USD")
        .monte_carlo(n_draws, seed=1)
        .basis_risk(draws)
        .covers(Cover.from_document(case["cover"]))
        .objective(
            objective_from_document(case["objective"]),
            pricing=pricing_from_document(case["pricing"]),
            budget=budget_from_document(case["budget"]),
        )
    )
    return plan


def _scale(case: Mapping[str, Any]) -> float:
    return float(max(sum(row) for row in case["sample"]["loss"]))


def _check_metrics(
    row: Mapping[str, Any], expected: Mapping[str, Any], case: Mapping[str, Any]
) -> None:
    """The rule of the design parity test: a tolerance, and either atom."""
    scale = _scale(case)
    relative = case["tolerances"]["metrics"]
    assert set(expected) <= set(row)
    for name, value in expected.items():
        got = row[name]
        if value is None:
            assert got is None, name
            continue
        floor = 1e-9 if name.startswith(UNITLESS) else 1e-9 * scale
        assert got is not None, name
        atoms = value["either"] if isinstance(value, dict) else [value]
        assert any(abs(got - atom) <= relative * abs(atom) + floor for atom in atoms), (
            name,
            got,
            value,
        )


def _check_case(result: Any, case: Mapping[str, Any], fold: int) -> None:
    """The plan's design and rows on the case's fold are the case's."""
    expected = case["expected"]
    tolerances = case["tolerances"]
    scale = _scale(case)
    # The plan's rows have the columns of MetricOptions(), the cases' default.
    assert case["metric_options"] is None
    (asset_id,) = case["sample"]["asset_ids"]
    designs = [d for d in result.designs if d.cover.asset_id == asset_id]
    found = designs[fold]
    # The design block is the case's sample, period for period.
    train = found.design_sample
    assert train.period.tolist() == case["sample"]["period"]
    np.testing.assert_array_equal(train.index[:, 0], _column(case["sample"], "index"))
    np.testing.assert_array_equal(train.loss[:, 0], _column(case["sample"], "loss"))
    # The design: objective value, parameters (exact here), premium.
    assert case["compare_parameters"] == "exact" and expected["unique"]
    power = 2 if case["objective"]["kind"] == "mse" else 1
    target = expected["objective_value"]
    assert abs(found.objective_value - target) <= (
        tolerances["objective_value"] * abs(target) + 1e-9 * scale**power
    )
    assert set(found.unidentified) == set(expected["unidentified"])
    for name, value in expected["parameters"].items():
        if name not in expected["unidentified"]:
            allowance = tolerances["parameters"] * scale
            assert abs(found.parameters[name] - value) <= allowance, name
    assert abs(found.premium - expected["premium"]) <= (
        tolerances["premium"] * abs(expected["premium"]) + 1e-9 * scale
    )
    # The function route on the plan's design block gives the plan's design.
    by_hand = design(
        Cover.from_document(case["cover"]),
        train,
        objective_from_document(case["objective"]),
        pricing=pricing_from_document(case["pricing"]),
        budget=budget_from_document(case["budget"]),
    )
    assert found.design_id == by_hand.design_id
    assert found.parameters == by_hand.parameters
    # The rows of the design, in sample and out of sample, with its premium.
    rows = [
        row
        for row in result.table.to_pylist()
        if row["design_id"] == found.design_id and row["fold"] == fold
    ]
    assert sorted(row["split"] for row in rows) == ["in_sample", "out_of_sample"]
    for row in rows:
        assert row["premium_basis"] == "design"
        assert row["premium"] == found.premium
    in_sample = next(row for row in rows if row["split"] == "in_sample")
    _check_metrics(in_sample, expected["metrics"], case)
    out = next(row for row in rows if row["split"] == "out_of_sample")
    assert expected["out_of_sample"]["fold"] == fold
    assert expected["out_of_sample"]["premium_basis"] == "design"
    _check_metrics(out, expected["out_of_sample"]["metrics"], case)


def test_the_split_cases_are_present() -> None:
    assert sorted(case["case_id"] for case in KFOLD_CASES) == sorted(
        [f"step-mse-pure-individual-synthetic52-kfold5-fold{i}" for i in range(5)]
        + ["step-wcvar-b0_9-coc-individual-synthetic52-kfold5-fold0"]
    )
    assert sorted(case["case_id"] for case in FRESH_CASES) == [
        "step-mse-pure-individual-synthetic52-fresh",
        "step-wcvar-b0_9-coc-individual-synthetic52-fresh",
    ]


@pytest.mark.parametrize("case", KFOLD_CASES, ids=lambda case: case["case_id"])
def test_kfold_on_curves_reproduces_the_case(
    dataset: Path, case: dict[str, Any], tmp_path: Path
) -> None:
    split = case["split"]
    full = _full(case)
    n = len(full["index"])
    plan = _plan(dataset, case, _CaseDraws({n: full}), n).split(KFold(split["k"]))
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    (sample,) = result.samples.values()
    np.testing.assert_array_equal(sample.index[:, 0], full["index"])
    assert sample.provenance.source_kind == "monte_carlo"
    assert len(result.designs) == split["k"]
    _check_case(result, case, split["fold"])
    # The held-out block is the case's evaluation sample: its first and last
    # period.
    out = [
        entry
        for entry in result.metadata.setup.evaluations
        if entry["split"] == "out_of_sample" and entry["fold"] == split["fold"]
    ]
    assert [(entry["method"], entry["details"]) for entry in out] == [
        ("kfold", {"k": split["k"], "plan_seed": 1})
    ]
    held_out = next(
        entry
        for entry in result.metadata.setup.samples
        if entry["sample_hash"] == out[0]["sample_hash"]
    )
    periods = case["evaluation_sample"]["period"]
    assert held_out["periods"] == [periods[0], periods[-1]]


@pytest.mark.parametrize("case", FRESH_CASES, ids=lambda case: case["case_id"])
def test_fresh_draws_on_curves_reproduce_the_case(
    dataset: Path, case: dict[str, Any], tmp_path: Path
) -> None:
    split = case["split"]
    design_draws = {name: _column(case["sample"], name) for name in ("index", "loss")}
    fresh_draws = {
        name: _column(case["evaluation_sample"], name) for name in ("index", "loss")
    }
    n, n_fresh = len(design_draws["index"]), split["n_draws"]
    assert n != n_fresh  # the model tells the two apart by their length
    draws = _CaseDraws({n: design_draws, n_fresh: fresh_draws})
    plan = _plan(dataset, case, draws, n).split(FreshDraws(n_fresh, seed=2))
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    _check_case(result, case, 0)
    assert result.fresh_samples is not None
    (fresh,) = result.fresh_samples.values()
    assert fresh.period.tolist() == case["evaluation_sample"]["period"]
    np.testing.assert_array_equal(fresh.loss[:, 0], fresh_draws["loss"])
    out = next(
        row for row in result.table.to_pylist() if row["split"] == "out_of_sample"
    )
    assert out["sample_hash"] == fresh.content_hash
    methods = {
        (entry["method"], entry.get("leakage_check"))
        for entry in result.metadata.setup.evaluations
        if entry["split"] == "out_of_sample"
    }
    assert methods == {("fresh_draws", "passed")}
