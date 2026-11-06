"""Designs: parity with the golden fixtures, closed forms, routes and reports.

The fixture cases are read as JSON with plain numpy, as
`tests/fixtures/parametric/README.md` describes; no reference code is
imported. Every design case of one asset in the folder is judged, so new
cases (other families, the grid route) are picked up when they are copied.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    CostOfCapitalLoad,
    Cover,
    DesignError,
    DesignOptions,
    DesignResult,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    IndexDefinition,
    InfeasibleDesignError,
    LeakageError,
    Limit,
    LinearConstraints,
    LinearPayout,
    MetricOptions,
    OutcomeSample,
    PiecewiseLinearPayout,
    PurePremium,
    SampleError,
    SampleProvenance,
    SolverUnavailableError,
    StepPayout,
    UnboundedDesignError,
    UserObjective,
    WeightedCVaR,
    _solver,
    compare,
    design,
    evaluate,
    holdout,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.constraints import budget_from_document, limit_from_document
from crc_sdk.parametric.design import DESIGN_SEMANTICS_VERSION, Fill
from crc_sdk.parametric.evaluate import DesignedCover
from crc_sdk.parametric.index import Tail
from crc_sdk.parametric.objectives import objective_from_document
from crc_sdk.parametric.parameters import _DefinitionModel
from crc_sdk.parametric.pricing import pricing_from_document

CASES = Path(__file__).parent / "fixtures" / "parametric" / "cases"
UNITLESS = (
    "coverage_ratio",
    "payout_to_loss_ratio",
    "correlation",
    "tail_coverage_",
    "cvar_uncovered_to_loss_",
    "false_negative_rate_",
    "false_positive_rate_",
)


# -- fixtures ----------------------------------------------------------------


def _case(path: Path) -> dict[str, Any]:
    case: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return case


def _design_cases() -> list[Path]:
    paths = []
    for path in sorted(CASES.glob("*.json")):
        case = _case(path)
        if case["kind"] == "design" and case["scope"] == "individual":
            paths.append(path)
    return paths


DESIGN_CASES = _design_cases()


def _block_sample(
    block: Mapping[str, Any], name: str, *, lineage: str | None = None
) -> OutcomeSample:
    return OutcomeSample(
        asset_ids=tuple(block["asset_ids"]),
        period=np.asarray(block["period"], dtype=np.int64),
        probability=np.asarray(block["probability"], dtype=np.float64),
        index=np.asarray(block["index"], dtype=np.float64),
        loss=np.asarray(block["loss"], dtype=np.float64),
        hazard=None,
        index_unit=block["index_unit"],
        loss_unit=block["loss_unit"],
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": name},),
            semantics_source="caller",
            lineage=lineage,
        ),
    )


def _full_sample(case: Mapping[str, Any]) -> OutcomeSample:
    """The full sample of a holdout or k-fold case: both blocks, by period."""
    design_block, held_out = case["sample"], case["evaluation_sample"]
    rows = sorted(
        zip(
            design_block["period"] + held_out["period"],
            design_block["index"] + held_out["index"],
            design_block["loss"] + held_out["loss"],
            strict=True,
        )
    )
    n = len(rows)
    block = {
        **design_block,
        "period": [row[0] for row in rows],
        "index": [row[1] for row in rows],
        "loss": [row[2] for row in rows],
        "probability": [1.0 / n] * n,
    }
    return _block_sample(block, case["case_id"])


def _case_samples(
    case: Mapping[str, Any],
) -> tuple[OutcomeSample, OutcomeSample | None]:
    """The design sample and, out of sample, the evaluation sample.

    A holdout or k-fold case is split again from its full sample by the
    module's own split, which must return the case's two blocks. A
    fresh-draw case's draws come from another source, so the two samples get
    different lineages.
    """
    split = case["split"]
    if split is None:
        return _block_sample(case["sample"], case["case_id"]), None
    if split["kind"] == "fresh_draws":
        design_sample = _block_sample(
            case["sample"], case["case_id"], lineage=f"{case['case_id']}/design"
        )
        fresh = _block_sample(
            case["evaluation_sample"],
            f"{case['case_id']}/fresh",
            lineage=f"{case['case_id']}/fresh",
        )
        assert fresh.period.tolist() == list(range(split["n_draws"]))
        return design_sample, fresh
    full = _full_sample(case)
    if split["kind"] == "holdout":
        train, test = holdout(full, train_fraction=split["train_fraction"])
    else:
        assert split["kind"] == "kfold", split
        train, test = year_block_kfold(full, k=split["k"])[split["fold"]]
    for got, block in ((train, case["sample"]), (test, case["evaluation_sample"])):
        assert got.period.tolist() == block["period"]
        np.testing.assert_allclose(got.probability, block["probability"], rtol=1e-12)
    return train, test


def _scale(case: Mapping[str, Any]) -> float:
    return float(max(sum(row) for row in case["sample"]["loss"]))


def _check_metrics(
    row: Mapping[str, Any], expected: Mapping[str, Any], case: Mapping[str, Any]
) -> None:
    scale = _scale(case)
    relative = case["tolerances"]["metrics"]
    for name, value in expected.items():
        got = row[name]
        if value is None:
            assert got is None, name
            continue
        floor = 1e-9 if name.startswith(UNITLESS) else 1e-9 * scale
        assert got is not None, name
        # A quantile on a step: either atom passes.
        atoms = value["either"] if isinstance(value, dict) else [value]
        assert any(abs(got - atom) <= relative * abs(atom) + floor for atom in atoms), (
            name,
            got,
            value,
        )


def _metric_options(case: Mapping[str, Any]) -> MetricOptions:
    options = case.get("metric_options")
    return MetricOptions() if options is None else MetricOptions(**options)


def _design_case(case: Mapping[str, Any], sample: OutcomeSample) -> DesignResult:
    return design(
        Cover.from_document(case["cover"]),
        sample,
        objective_from_document(case["objective"]),
        pricing=pricing_from_document(case["pricing"]),
        budget=budget_from_document(case["budget"]),
        limit=limit_from_document(case["limit"]),
        options=DesignOptions(ranges=True),
    )


def _check_fill_lowest(result: DesignResult, template: Cover) -> None:
    """Unidentified amounts take the smallest value the constraints allow."""
    payout = template.payout
    assert hasattr(payout, "amount_names")
    parameters = payout.parameters()
    previous: float | None = None
    for name in payout.amount_names():
        value = result.cover.payout.parameters()[name]
        assert isinstance(value, float)
        if name in result.unidentified:
            free = parameters[name]
            assert isinstance(free, Free)
            monotone = bool(getattr(payout, "monotone", False))
            floor = free.lower
            if monotone and previous is not None:
                floor = max(floor, previous)
            assert value == floor, name
        previous = value


def test_the_design_cases_are_present() -> None:
    names = {path.stem for path in DESIGN_CASES}
    objectives = ("mse", "eu", "wcvar-b0_9", "wcvar-b0_99")
    pricings = ("pure", "coc")
    datasets = ("synthetic", "thesis")
    in_sample = {
        f"step-{o}-{p}-individual-{d}"
        for o, p, d in itertools.product(objectives, pricings, datasets)
    }
    out_of_sample = {
        f"step-{o}-{p}-individual-{d}-holdout"
        for o, p, d in itertools.product(("mse", "wcvar-b0_9"), pricings, datasets)
    }
    newer = {
        "binary-upper-mse-pure-individual-synthetic52",
        "binary-upper-wcvar-b0_9-coc-individual-synthetic52",
        "binary-upper-grid-eu-pure-individual-synthetic52",
        "binary-upper-grid-mse-coc-individual-synthetic52",
        "step-lower-wcvar-b0_9-pure-individual-handmade",
        "linear-upper-grid-wcvar-b0_9-pure-individual-handmade",
        "piecewise-linear-upper-mse-pure-individual-handmade",
        "piecewise-linear-upper-wcvar-b0_9-coc-individual-handmade",
    }
    kfold = {
        f"step-mse-pure-individual-synthetic52-kfold5-fold{i}" for i in range(5)
    } | {"step-wcvar-b0_9-coc-individual-synthetic52-kfold5-fold0"}
    fresh = {
        "step-mse-pure-individual-synthetic52-fresh",
        "step-wcvar-b0_9-coc-individual-synthetic52-fresh",
    }
    assert in_sample | out_of_sample | newer | kfold | fresh <= names
    assert len(DESIGN_CASES) >= 40


def _split_cases(kind: str) -> list[dict[str, Any]]:
    cases = [_case(path) for path in DESIGN_CASES]
    return [case for case in cases if (case["split"] or {}).get("kind") == kind]


def test_the_kfold_cases_split_one_full_sample() -> None:
    cases = _split_cases("kfold")
    assert len(cases) == 6
    full = {_full_sample(case).content_hash for case in cases}
    assert len(full) == 1
    # The MSE design has every fold: its held-out blocks partition the sample.
    mse = sorted(
        (case for case in cases if case["objective"]["kind"] == "mse"),
        key=lambda case: case["split"]["fold"],
    )
    assert [case["split"]["fold"] for case in mse] == list(range(5))
    blocks = [case["evaluation_sample"]["period"] for case in mse]
    assert [period for block in blocks for period in block] == list(range(52))
    assert [len(block) for block in blocks] == [11, 11, 10, 10, 10]


@pytest.mark.parametrize(
    "case_id",
    [
        "step-mse-pure-individual-synthetic52-kfold5-fold0",
        "step-wcvar-b0_9-coc-individual-synthetic52-kfold5-fold0",
    ],
)
def test_a_held_out_fold_reaches_the_filled_level(case_id: str) -> None:
    # Fold 0 holds out the one draw in the last bin (60 m/s and above), so
    # the design leaves levels[5] unidentified, and the fill rule sets the
    # payout of that held-out draw: the previous level.
    case = _case(CASES / f"{case_id}.json")
    sample, held_out = _case_samples(case)
    assert held_out is not None
    result = _design_case(case, sample)
    assert result.unidentified == ("levels[5]",)
    parameters = result.cover.payout.parameters()
    assert parameters["levels[5]"] == parameters["levels[4]"]
    index = held_out.index[:, 0]
    reached = index >= 60.0
    assert reached.sum() == 1
    payout = result.cover.payout_values(index[reached])[0]
    assert payout == parameters["levels[4]"]


def test_fresh_draws_are_another_source() -> None:
    cases = _split_cases("fresh_draws")
    assert len(cases) == 2
    for case in cases:
        sample, fresh = _case_samples(case)
        assert fresh is not None and len(fresh) == case["split"]["n_draws"]
        # The draw numbers overlap the design years; only the lineage tells
        # the two sources apart.
        assert set(fresh.period.tolist()) <= set(sample.period.tolist())
        result = _design_case(case, sample)
        row = evaluate(result, fresh, split="out_of_sample").to_pylist()[0]
        assert row["premium"] == result.premium
        same_source = _block_sample(
            case["evaluation_sample"], "fresh", lineage=sample.lineage
        )
        with pytest.raises(LeakageError):
            evaluate(result, same_source, split="out_of_sample")


@pytest.mark.parametrize("path", DESIGN_CASES, ids=lambda path: path.stem)
def test_design_parity(path: Path) -> None:
    case = _case(path)
    expected = case["expected"]
    tolerances = case["tolerances"]
    scale = _scale(case)
    sample, held_out = _case_samples(case)
    template = Cover.from_document(case["cover"])
    result = _design_case(case, sample)

    # 1. The objective value.
    power = 2 if case["objective"]["kind"] == "mse" else 1
    target = expected["objective_value"]
    assert abs(result.objective_value - target) <= (
        tolerances["objective_value"] * abs(target) + 1e-9 * scale**power
    )

    # 2. The parameters.
    assert set(result.unidentified) == set(expected["unidentified"])
    identified = {
        name: value
        for name, value in expected["parameters"].items()
        if name not in expected["unidentified"]
    }
    allowance = tolerances["parameters"] * scale
    if case["compare_parameters"] == "exact":
        for name, value in identified.items():
            assert abs(result.parameters[name] - value) <= allowance, name
    elif case["compare_parameters"] == "within_ranges":
        assert result.ranges is not None
        for name, value in identified.items():
            low, high = result.ranges[name]
            assert low - allowance <= value <= high + allowance, name
    else:
        assert case["compare_parameters"] == "none"
    if result.options.fill == "lowest":
        _check_fill_lowest(result, template)
    shape = [
        name
        for name in template.free_parameters()
        if name not in getattr(template.payout, "amount_names", tuple)()
    ]
    for name in shape:
        # A chosen grid value is compared exactly.
        assert result.parameters[name] == expected["parameters"][name], name
    if expected.get("profile") is not None:
        assert result.route == "grid"
        profile = _profile(result)
        assert len(profile) == len(expected["profile"])
        for row, entry in zip(profile, expected["profile"], strict=True):
            for name, value in entry["parameters"].items():
                assert row[name] == value
            assert row["objective"] is not None
            target = entry["objective_value"]
            assert abs(row["objective"] - target) <= (
                tolerances["objective_value"] * abs(target) + 1e-9 * scale**power
            )
        assert result.ties == 0

    # 3. The premium and the metrics.
    pricing = pricing_from_document(case["pricing"])
    options = _metric_options(case)
    # A k-fold design block is a block of its fold; every other sample is fold 0.
    split = case["split"] or {}
    fold = split["fold"] if split.get("kind") == "kfold" else 0
    if expected["unique"]:
        evaluated: Cover | DesignResult = result
        row = evaluate(
            result, sample, split="in_sample", fold=fold, options=options
        ).to_pylist()[0]
        assert row["premium_basis"] == "design"
        assert row["design_id"] == result.design_id
        premium = result.premium
    else:
        evaluated = template.with_values(expected["parameters"])
        row = evaluate(
            evaluated,
            sample,
            split="in_sample",
            fold=fold,
            pricing=pricing,
            options=options,
        ).to_pylist()[0]
        premium = row["premium"]
    assert abs(premium - expected["premium"]) <= (
        tolerances["premium"] * abs(expected["premium"]) + 1e-9 * scale
    )
    _check_metrics(row, expected["metrics"], case)

    # 4. The module's uniqueness never claims more than the case's.
    if expected["unique"] is False:
        assert result.unique is not True
    if expected["unique"] and case["compare_parameters"] == "exact":
        assert result.unique is True

    # Out of sample, with the design premium.
    if held_out is not None:
        out = expected["out_of_sample"]
        assert out["premium_basis"] == "design"
        assert out["fold"] == fold
        if expected["unique"]:
            table = evaluate(
                result,
                held_out,
                split="out_of_sample",
                fold=out["fold"],
                options=options,
            )
        else:
            table = evaluate(
                evaluated,
                held_out,
                split="out_of_sample",
                fold=out["fold"],
                premium=expected["premium"],
                design_sample=sample,
                options=options,
            )
        row = table.to_pylist()[0]
        assert row["split"] == "out_of_sample" and row["fold"] == out["fold"]
        assert row["premium_basis"] == ("design" if expected["unique"] else "stated")
        _check_metrics(row, out["metrics"], case)


def test_parity_routes_and_flags() -> None:
    for path in DESIGN_CASES:
        case = _case(path)
        sample, _ = _case_samples(case)
        result = _design_case(case, sample)
        kind = case["objective"]["kind"]
        amounts = "qp" if kind == "mse" else "lp"
        assert result.amounts == amounts
        if case["expected"].get("profile") is None:
            assert result.route == amounts
            assert result.profile is None
        else:
            assert result.route == "grid"
        assert result.ties == 0
        assert result.solver is not None and result.solver["name"] == "highs"
        if kind == "weighted_cvar":
            beta = case["objective"]["beta"]
            n = len(case["sample"]["period"])
            assert result.tail_outcomes == pytest.approx((1 - beta) * n)
            assert result.degenerate_tail is None
        else:
            assert result.tail_outcomes is None and result.degenerate_tail is None
        zero = sum(1 for row in case["sample"]["loss"] if row[0] == 0) / len(
            case["sample"]["loss"]
        )
        assert result.p_zero_loss == pytest.approx(zero, abs=1e-12)


# -- small samples -----------------------------------------------------------


def _profile(result: DesignResult) -> list[dict[str, Any]]:
    assert result.profile is not None
    rows: list[dict[str, Any]] = result.profile.to_pylist()
    return rows


def _sample(
    index: Sequence[float],
    loss: Sequence[float],
    *,
    probability: Sequence[float] | None = None,
    asset: str = "A",
    index_unit: str = "m/s",
    loss_unit: str = "EUR",
) -> OutcomeSample:
    n = len(index)
    return OutcomeSample(
        asset_ids=(asset,),
        period=np.arange(n, dtype=np.int64),
        probability=(
            np.full(n, 1.0 / n)
            if probability is None
            else np.asarray(probability, dtype=np.float64)
        ),
        index=np.asarray(index, dtype=np.float64),
        loss=np.asarray(loss, dtype=np.float64),
        hazard=None,
        index_unit=index_unit,
        loss_unit=loss_unit,
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "test"},),
            semantics_source="caller",
        ),
    )


INDEX = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
# Four years: no loss below the first breakpoint; band 0 holds the years with
# losses 10 and 30, band 1 the year with loss 40.
SAMPLE = _sample([0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 30.0, 40.0])


def _step(
    levels: Sequence[float | Free],
    breakpoints: Sequence[float | Free] = (1.0, 2.5),
) -> Cover:
    return Cover(
        INDEX, StepPayout(breakpoints, levels), payout_unit="EUR", asset_id="A"
    )


def test_expected_uncovered_closed_form() -> None:
    # Per unit of premium, band 1 (loss 40) and band 0 up to 10 cover 1 unit
    # of E[U]; band 0 from 10 to 30 covers 1/2. A budget of 20 (4 x 20 = 80
    # in summed payouts) buys band 0 to 10 and band 1 to 40, then band 0 to 20.
    result = design(
        _step([Free(), Free()]),
        SAMPLE,
        ExpectedUncovered(),
        budget=Budget(amount=20.0),
        options=DesignOptions(ranges=True),
    )
    assert result.route == "lp" and result.amounts == "lp"
    assert result.parameters == pytest.approx({"levels[0]": 20.0, "levels[1]": 40.0})
    assert result.objective_value == pytest.approx(2.5)
    assert result.premium == pytest.approx(20.0)
    assert result.budget_bound == 20.0 and result.budget_use == pytest.approx(1.0)
    assert result.unique is True
    assert result.ranges is not None
    for name, value in result.parameters.items():
        low, high = result.ranges[name]
        assert low <= value + 1e-9 and value <= high + 1e-9
        assert high - low < 1e-4


def test_expected_uncovered_without_ranges_reports_no_uniqueness() -> None:
    result = design(
        _step([Free(), Free()]), SAMPLE, ExpectedUncovered(), budget=Budget(amount=20.0)
    )
    assert result.unique is None and result.ranges is None
    assert result.solver_options["ranging"] is None


def test_non_unique_lp_optimum_is_reported_with_ranges() -> None:
    # Losses 10 and 20 in band 0, 40 in band 1, a budget of 5: every unit of
    # premium covers one unit of E[U] in either band, so every split of the
    # budget with levels[0] <= levels[1] is optimal.
    sample = _sample([0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 20.0, 40.0])
    result = design(
        _step([Free(), Free()]),
        sample,
        ExpectedUncovered(),
        budget=Budget(amount=5.0),
        options=DesignOptions(ranges=True),
    )
    assert result.objective_value == pytest.approx(12.5)
    assert result.unique is False
    assert result.ranges is not None
    low, high = result.ranges["levels[0]"]
    assert low == pytest.approx(0.0, abs=1e-6) and high == pytest.approx(20 / 3)
    low, high = result.ranges["levels[1]"]
    assert low == pytest.approx(20 / 3) and high == pytest.approx(20.0)
    row = evaluate(result, sample, split="in_sample").to_pylist()[0]
    assert row["unique"] is False


# -- ranging on one solver instance ------------------------------------------


NON_UNIQUE = _sample([0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 20.0, 40.0])


def test_the_ranging_lps_share_one_solver_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh: list[tuple[Mapping[str, Any] | None, bool]] = []
    solved: list[_solver.Solution] = []
    sessions: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    solve_lp, session = _solver.solve_lp, _solver.LPSession

    def counted_solve(
        programme: _solver.Programme,
        options: Mapping[str, Any] | None = None,
        *,
        keep_basis: bool = False,
    ) -> _solver.Solution:
        fresh.append((options, keep_basis))
        solved.append(solve_lp(programme, options, keep_basis=keep_basis))
        return solved[-1]

    def counted_session(*args: Any, **kwargs: Any) -> _solver.LPSession:
        sessions.append((args, kwargs))
        return session(*args, **kwargs)

    monkeypatch.setattr(_solver, "solve_lp", counted_solve)
    monkeypatch.setattr(_solver, "LPSession", counted_session)
    result = design(
        _step([Free(), Free()]),
        NON_UNIQUE,
        ExpectedUncovered(),
        budget=Budget(amount=5.0),
        options=DesignOptions(ranges=True),
    )
    # The design LP is solved on its own and keeps its basis; the 4 ranging
    # LPs of each of the 2 amounts share one instance, which starts from that
    # basis, and are still counted as solves.
    assert fresh == [(None, True)]
    assert len(sessions) == 1
    args, kwargs = sessions[0]
    assert args[1] == _solver.RANGING_OPTIONS
    assert solved[0].basis is not None
    assert kwargs["basis"] is solved[0].basis
    assert result.solves == 1 + 4 * 2
    assert result.ranges is not None
    assert result.ranges["levels[1]"] == pytest.approx((20 / 3, 20.0))


def _small_lp(cost: Sequence[float], upper: float | None = None) -> _solver.Programme:
    """`min cost @ x` s.t. `x0 + x1 >= 1`, `0 <= x <= 1`, and `x0 + 2 x1 <= upper`."""
    rows, columns, values = [0, 0], [0, 1], [1.0, 1.0]
    row_lower, row_upper = [1.0], [math.inf]
    if upper is not None:
        rows, columns, values = rows + [1, 1], columns + [0, 1], values + [1.0, 2.0]
        row_lower, row_upper = row_lower + [-math.inf], row_upper + [upper]
    matrix = _solver.sparse_columns(
        np.array(rows), np.array(columns), np.array(values), (len(row_lower), 2)
    )
    return _solver.Programme(
        cost=np.asarray(cost, dtype=np.float64),
        col_lower=np.zeros(2),
        col_upper=np.ones(2),
        matrix=matrix,
        row_lower=np.array(row_lower),
        row_upper=np.array(row_upper),
    )


def test_a_session_solves_as_a_fresh_instance_would() -> None:
    session = _solver.LPSession(_small_lp([1.0, 2.0]), _solver.RANGING_OPTIONS)
    assert session.n_rows == 1
    # Each step's optimum is a unique vertex, so both solves must find it.
    steps: list[tuple[list[float], float | None, float]] = [
        ([1.0, 2.0], None, 1.0),  # x = (1, 0)
        ([0.5, -1.0], None, -1.0),  # x = (0, 1)
        ([0.5, -1.0], 1.5, -0.25),  # x = (0.5, 0.5)
        ([0.5, -1.0], 1.2, 0.2),  # x = (0.8, 0.2)
        ([-1.0, -1.0], 1.2, -1.1),  # x = (1, 0.1)
    ]
    row: int | None = None
    for cost, upper, optimum in steps:
        if upper is not None:
            if row is None:
                row = session.add_row(np.array([1.0, 2.0]), -math.inf, math.inf)
                assert row == 1 and session.n_rows == 2
            session.change_row_bounds(row, -math.inf, upper)
        session.change_cost(np.array(cost))
        warm = session.solve()
        cold = _solver.solve_lp(_small_lp(cost, upper), _solver.RANGING_OPTIONS)
        assert warm.status == cold.status == "optimal"
        assert warm.objective == pytest.approx(optimum, abs=1e-12)
        assert cold.objective == pytest.approx(optimum, abs=1e-12)
        assert warm.values is not None and cold.values is not None
        np.testing.assert_allclose(warm.values, cold.values, atol=1e-12)
    # An option for one solve, here presolve off, does not stay set.
    assert session.solve({"presolve": "off"}).status == "optimal"
    assert session.solve().status == "optimal"


def test_a_session_starts_from_a_given_basis() -> None:
    programme = _small_lp([1.0, 2.0])
    solution = _solver.solve_lp(programme, keep_basis=True)
    assert solution.status == "optimal" and solution.basis is not None
    assert (solution.basis.n_columns, solution.basis.n_rows) == (2, 1)
    assert _solver.solve_lp(programme).basis is None
    infeasible = _solver.solve_lp(_small_lp([1.0, 2.0], 0.5), keep_basis=True)
    assert infeasible.status == "infeasible" and infeasible.basis is None
    # The basis is optimal: a session that starts from it needs no iteration,
    # where one that starts from scratch needs some (both without presolve).
    off = {"presolve": "off"}
    cold = _solver.LPSession(programme, _solver.RANGING_OPTIONS).solve(off)
    seeded = _solver.LPSession(programme, _solver.RANGING_OPTIONS, basis=solution.basis)
    warm = seeded.solve(off)
    assert cold.iterations > 0 and warm.iterations == 0
    assert warm.values is not None and solution.values is not None
    np.testing.assert_allclose(warm.values, solution.values, atol=1e-12)
    # A row added to a seeded session enters the basis as a basic row, and
    # the LP then solves as a fresh instance would: x = (0.8, 0.2).
    seeded = _solver.LPSession(programme, _solver.RANGING_OPTIONS, basis=solution.basis)
    row = seeded.add_row(np.array([1.0, 2.0]), -math.inf, math.inf)
    seeded.change_row_bounds(row, -math.inf, 1.2)
    seeded.change_cost(np.array([0.5, -1.0]))
    warm = seeded.solve()
    fresh = _solver.solve_lp(_small_lp([0.5, -1.0], 1.2), _solver.RANGING_OPTIONS)
    assert warm.status == fresh.status == "optimal"
    assert warm.objective == pytest.approx(0.2, abs=1e-12)
    assert warm.values is not None and fresh.values is not None
    np.testing.assert_allclose(warm.values, fresh.values, atol=1e-12)


def test_a_session_rejects_bad_changes() -> None:
    hessian = _solver.sparse_columns(
        np.array([0, 1]), np.array([0, 1]), np.array([1.0, 1.0]), (2, 2)
    )
    lp = _small_lp([1.0, 2.0])
    qp = _solver.Programme(
        cost=lp.cost,
        col_lower=lp.col_lower,
        col_upper=lp.col_upper,
        matrix=lp.matrix,
        row_lower=lp.row_lower,
        row_upper=lp.row_upper,
        hessian=hessian,
    )
    with pytest.raises(ValueError, match="hessian"):
        _solver.LPSession(qp)
    with pytest.raises(ValueError, match="rejected the option"):
        _solver.LPSession(lp, {"no_such_option": 1})
    basis = _solver.solve_lp(lp, keep_basis=True).basis
    with pytest.raises(ValueError, match="basis is for 2 columns and 1 rows"):
        _solver.LPSession(_small_lp([1.0, 2.0], 1.5), basis=basis)
    empty = _solver.Basis(2, 1, _solver.require().HighsBasis())
    with pytest.raises(ValueError, match="rejected the basis"):
        _solver.LPSession(lp, basis=empty)
    session = _solver.LPSession(lp)
    with pytest.raises(ValueError, match="one entry per column"):
        session.change_cost(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        session.change_cost(np.array([1.0, math.nan]))
    with pytest.raises(ValueError, match="one coefficient per column"):
        session.add_row(np.zeros(1), 0.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        session.add_row(np.array([1.0, math.inf]), 0.0, 1.0)
    with pytest.raises(ValueError, match="row 1"):
        session.change_row_bounds(1, 0.0, 1.0)
    with pytest.raises(ValueError, match="rejected the option"):
        session.solve({"no_such_option": 1})


@pytest.mark.parametrize("status", ["infeasible_or_unbounded", "error"])
def test_a_failed_ranging_solve_raises(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    calls: list[Mapping[str, Any] | None] = []

    def failed(
        self: _solver.LPSession, options: Mapping[str, Any] | None = None
    ) -> _solver.Solution:
        calls.append(options)
        return _solver.Solution(status, None, None, f"patched {status}", 0, 0.0)  # type: ignore[arg-type]

    monkeypatch.setattr(_solver.LPSession, "solve", failed)
    with pytest.raises(DesignError) as raised:
        design(
            _step([Free(), Free()]),
            NON_UNIQUE,
            ExpectedUncovered(),
            budget=Budget(amount=5.0),
            options=DesignOptions(ranges=True),
        )
    assert type(raised.value) is DesignError
    assert raised.value.status == status
    assert raised.value.solver_message == f"patched {status}"
    # An ambiguous status is solved again without presolve.
    expected = [None, {"presolve": "off"}] if status != "error" else [None]
    assert calls == expected


def test_mse_closed_forms() -> None:
    # Without a budget each level is its band's mean loss.
    free = design(_step([Free(), Free()]), SAMPLE, MSE())
    assert free.route == "qp"
    assert free.parameters == pytest.approx({"levels[0]": 20.0, "levels[1]": 40.0})
    assert free.objective_value == pytest.approx(50.0)
    assert free.unique is True
    # With a binding budget of 10 the Lagrange conditions give 20/3 and 80/3.
    bound = design(_step([Free(), Free()]), SAMPLE, MSE(), budget=Budget(amount=10.0))
    assert bound.parameters == pytest.approx(
        {"levels[0]": 20 / 3, "levels[1]": 80 / 3}, rel=1e-9
    )
    assert bound.objective_value == pytest.approx(550 / 3, rel=1e-9)
    assert bound.premium == pytest.approx(10.0, rel=1e-9)


def test_weighted_cvar_matches_a_brute_force_search() -> None:
    objective = WeightedCVaR(0.7, 0.5)
    result = design(
        _step([Free(upper=40.0), Free(upper=40.0)]),
        SAMPLE,
        objective,
        budget=Budget(amount=12.0),
    )
    loss = SAMPLE.loss[:, 0]
    best = math.inf
    for low, high in itertools.product(np.arange(0.0, 40.25, 0.25), repeat=2):
        if low > high:
            continue
        payout = np.array([0.0, low, low, high])
        if payout.mean() > 12.0 + 1e-12:
            continue
        best = min(best, objective.value(loss, payout, SAMPLE.probability))
    assert result.objective_value <= best + 1e-9
    assert result.premium <= 12.0 * (1 + 1e-9)
    payout = result.cover.payout_values(SAMPLE.index[:, 0])
    assert objective.value(loss, payout, SAMPLE.probability) == pytest.approx(
        result.objective_value
    )


def test_cost_of_capital_budget_binds_on_the_loaded_premium() -> None:
    pricing = CostOfCapitalLoad(0.1, 0.75)
    result = design(
        _step([Free(), Free()]),
        SAMPLE,
        ExpectedUncovered(),
        pricing=pricing,
        budget=Budget(ratio=0.5),
    )
    payout = result.cover.payout_values(SAMPLE.index[:, 0])
    premium = pricing.premium(payout, SAMPLE.probability)
    assert result.premium == pytest.approx(premium)
    assert result.expected_loss == pytest.approx(20.0)
    assert result.budget_bound == pytest.approx(10.0)
    assert premium == pytest.approx(10.0, rel=1e-9)


@pytest.mark.parametrize(
    "objective", [MSE(), ExpectedUncovered(), WeightedCVaR(0.9, 0.5)]
)
def test_other_families_on_the_lp_and_qp_routes(
    objective: MSE | ExpectedUncovered | WeightedCVaR,
) -> None:
    # index 0..3: the linear basis from attachment 1 to exhaustion 3 is
    # 0, 0, 0.5, 1; a budget of 7.5 allows a limit of 20.
    linear = Cover(
        INDEX, LinearPayout(1.0, 3.0, Free()), payout_unit="EUR", asset_id="A"
    )
    result = design(linear, SAMPLE, objective, budget=Budget(amount=7.5))
    assert result.parameters["limit"] == pytest.approx(20.0)
    binary = Cover(INDEX, BinaryPayout(2.0, Free()), payout_unit="EUR")
    result = design(binary, SAMPLE, objective, budget=Budget(amount=10.0))
    assert result.parameters["amount"] == pytest.approx(20.0)
    piecewise = Cover(
        INDEX,
        PiecewiseLinearPayout([1.0, 3.0], [Free(), Free()]),
        payout_unit="EUR",
    )
    result = design(piecewise, SAMPLE, objective, budget=Budget(amount=10.0))
    payout = result.cover.payout_values(SAMPLE.index[:, 0])
    assert payout.mean() <= 10.0 * (1 + 1e-9)
    assert result.route == ("qp" if isinstance(objective, MSE) else "lp")


def test_lower_tail_index() -> None:
    drought = IndexDefinition("spei3", "1", tail="lower", aggregation="annual_minimum")
    sample = _sample([0.0, -1.0, -2.0, -3.0], [0.0, 10.0, 30.0, 40.0], index_unit="1")
    cover = Cover(
        drought,
        StepPayout([-1.0, -2.5], [Free(), Free()], tail="lower"),
        payout_unit="EUR",
    )
    result = design(cover, sample, ExpectedUncovered(), budget=Budget(amount=20.0))
    assert result.parameters == pytest.approx({"levels[0]": 20.0, "levels[1]": 40.0})


def test_limit_caps_the_amounts() -> None:
    result = design(_step([Free(), Free()]), SAMPLE, MSE(), limit=Limit(25.0))
    assert result.parameters == pytest.approx({"levels[0]": 20.0, "levels[1]": 25.0})
    assert result.limit == {"amount": 25.0}
    payout = result.cover.payout_values(SAMPLE.index[:, 0])
    assert payout.max() <= 25.0


def test_fixed_amounts_stay_fixed() -> None:
    result = design(_step([5.0, Free()]), SAMPLE, MSE())
    assert result.parameters == {"levels[1]": pytest.approx(40.0)}
    assert result.cover.payout.parameters()["levels[0]"] == 5.0


def test_small_amounts_are_set_to_zero_and_listed() -> None:
    # A tiny budget: the solver's levels are far below 1e-9 of the scale.
    result = design(
        _step([Free(), Free()]),
        SAMPLE,
        ExpectedUncovered(),
        budget=Budget(amount=1e-12),
    )
    assert all(value == 0.0 for value in result.parameters.values())
    assert set(result.zeroed) <= {"levels[0]", "levels[1]"}


# -- unidentified amounts ----------------------------------------------------


def _gap_template(last_upper: float | None = 50.0) -> Cover:
    # Band 1 (2 <= x < 2.5) holds no year.
    return _step(
        [Free(upper=50.0), Free(upper=50.0), Free(upper=last_upper)],
        breakpoints=(1.0, 2.0, 2.5),
    )


GAP_SAMPLE = _sample([0.0, 1.0, 3.0], [0.0, 10.0, 40.0])


def test_fill_lowest_takes_the_previous_level() -> None:
    result = design(_gap_template(), GAP_SAMPLE, MSE())
    assert result.unidentified == ("levels[1]",)
    assert result.fill == "lowest"
    assert result.parameters == pytest.approx(
        {"levels[0]": 10.0, "levels[1]": 10.0, "levels[2]": 40.0}
    )


def test_fill_highest_takes_the_next_level() -> None:
    result = design(
        _gap_template(), GAP_SAMPLE, MSE(), options=DesignOptions(fill="highest")
    )
    assert result.parameters["levels[1]"] == pytest.approx(40.0)
    assert result.to_record()["fill"] == "highest"


def test_fill_highest_of_the_last_band_needs_a_bound() -> None:
    template = _step([Free(), Free()], breakpoints=(1.0, 5.0))
    with pytest.raises(DesignError, match="upper bound") as raised:
        design(template, SAMPLE, MSE(), options=DesignOptions(fill="highest"))
    assert raised.value.asset_id == "A"
    filled = design(
        template,
        SAMPLE,
        MSE(),
        limit=Limit(60.0),
        options=DesignOptions(fill="highest"),
    )
    assert filled.parameters["levels[1]"] == 60.0


class CappedStep(StepPayout):
    """A step payout whose levels sum to at most a stated total."""

    kind: ClassVar[str] = "test_capped_step"
    total: ClassVar[float] = 70.0

    def __init__(
        self,
        breakpoints: Sequence[float | Free],
        levels: Sequence[float | Free],
        *,
        tail: Tail = "upper",
        monotone: bool = True,
    ) -> None:
        super().__init__(breakpoints, levels, tail=tail, monotone=monotone)

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        base = super().amount_constraints(limit)
        row = np.ones((1, base.n_amounts))
        return LinearConstraints(
            a_ub=np.vstack([base.a_ub, row]),
            b_ub=np.append(base.b_ub, self.total),
            lower=base.lower,
            upper=base.upper,
        )


@pytest.mark.parametrize(("fill", "expected"), [("lowest", 10.0), ("highest", 20.0)])
def test_fill_under_a_general_constraint_solves_an_lp(
    fill: Fill, expected: float
) -> None:
    template = Cover(
        INDEX,
        CappedStep(
            (1.0, 2.0, 2.5), [Free(upper=50.0), Free(upper=50.0), Free(upper=50.0)]
        ),
        payout_unit="EUR",
    )
    # MSE wants 10 and 40; the sum cap of 70 leaves at most 20 for the empty
    # band, so its highest value is bounded by the cap, not by levels[2].
    result = design(template, GAP_SAMPLE, MSE(), options=DesignOptions(fill=fill))
    levels = result.parameters
    assert levels["levels[0]"] == pytest.approx(10.0)
    assert levels["levels[2]"] == pytest.approx(40.0)
    assert sum(levels.values()) <= 70.0 + 1e-9
    assert levels["levels[1]"] == pytest.approx(expected, abs=1e-6)
    assert levels["levels[0]"] <= levels["levels[1]"] <= levels["levels[2]"] + 1e-9


# -- the grid ----------------------------------------------------------------


def test_free_thresholds_are_searched_on_their_grid() -> None:
    template = Cover(
        INDEX,
        LinearPayout(Free(0.0, 2.0, grid=[0.5, 1.5]), 3.0, Free()),
        payout_unit="EUR",
    )
    # Under a binding budget every payout below the loss has the same E[U],
    # so the tail term decides between the two attachments.
    result = design(template, SAMPLE, WeightedCVaR(0.9, 0.5), budget=Budget(amount=7.5))
    assert result.route == "grid" and result.amounts == "lp"
    profile = _profile(result)
    assert [row["attachment"] for row in profile] == [0.5, 1.5]
    assert all(row["status"] == "optimal" for row in profile)
    separate = [
        design(
            Cover(INDEX, LinearPayout(a, 3.0, Free()), payout_unit="EUR"),
            SAMPLE,
            WeightedCVaR(0.9, 0.5),
            budget=Budget(amount=7.5),
        )
        for a in (0.5, 1.5)
    ]
    for row, single in zip(profile, separate, strict=True):
        assert row["objective"] == pytest.approx(single.objective_value)
    best = min(separate, key=lambda item: item.objective_value)
    assert result.objective_value == pytest.approx(best.objective_value)
    assert (
        result.parameters["attachment"] == best.cover.payout.parameters()["attachment"]
    )
    assert result.ties == 0
    assert result.grid == {"candidates": 2, "optimal": 2}


def test_grid_ties_go_to_the_first_candidate() -> None:
    # Both breakpoints split the years the same way, so they tie.
    template = _step(
        [Free(), Free()], breakpoints=(Free(0.0, 1.0, grid=[0.5, 0.8]), 2.5)
    )
    result = design(template, SAMPLE, MSE(), options=DesignOptions(ranges=True))
    assert result.route == "grid" and result.amounts == "qp"
    assert result.ties == 1
    assert result.unique is False
    assert result.parameters["breakpoints[0]"] == 0.5
    row = evaluate(result, SAMPLE, split="in_sample").to_pylist()[0]
    assert row["ties"] == 1 and row["unique"] is False


def test_unordered_grid_candidates_are_skipped() -> None:
    template = _step(
        [Free(), Free()],
        breakpoints=(
            Free(0.0, 3.0, grid=[0.5, 2.0, 3.0]),
            Free(1.0, 2.5, grid=[1.0, 2.5]),
        ),
    )
    result = design(template, SAMPLE, MSE())
    statuses = [row["status"] for row in _profile(result)]
    assert statuses.count("invalid") == 3
    assert result.grid is not None and result.grid["invalid"] == 3


def test_the_grid_cap() -> None:
    template = _step([Free(), Free()], breakpoints=(Free(0.0, 1.0, grid=5), 2.5))
    with pytest.raises(DesignError, match="max_grid_points"):
        design(template, SAMPLE, MSE(), options=DesignOptions(max_grid_points=4))


def test_user_objective_searches_every_free_parameter() -> None:
    def mean_absolute(loss: Any, payout: Any, probability: Any) -> float:
        return float(np.sum(probability * np.abs(loss - payout)))

    objective = UserObjective(mean_absolute, name="mae", parameters={"power": 1})
    grid = [0.0, 10.0, 20.0, 30.0, 40.0]
    template = _step([Free(0.0, 40.0, grid=grid), Free(0.0, 40.0, grid=grid)])
    result = design(template, SAMPLE, objective, budget=Budget(amount=15.0))
    assert result.route == "grid" and result.amounts == "grid"
    assert result.solver is None and result.unique is None
    best = math.inf
    for low, high in itertools.product(grid, repeat=2):
        payout = np.array([0.0, low, low, high])
        if low <= high and payout.mean() <= 15.0:
            best = min(
                best, mean_absolute(SAMPLE.loss[:, 0], payout, SAMPLE.probability)
            )
    assert result.objective_value == pytest.approx(best)
    assert result.objective == {
        "kind": "user",
        "name": "mae",
        "parameters": {"power": 1},
    }
    statuses = {row["status"] for row in _profile(result)}
    assert {"feasible", "invalid", "over_budget"} <= statuses
    assert len(_profile(result)) == 25


def test_user_objective_needs_grids() -> None:
    objective = UserObjective(lambda loss, payout, p: 0.0, name="zero")
    with pytest.raises(DesignError, match="grid"):
        design(_step([Free(), Free()]), SAMPLE, objective)


class Saturating(_DefinitionModel):
    """A test family that is not linear in its amounts: `amount * x / (x + half)`."""

    kind: ClassVar[str] = "test_saturating"
    amount: Any
    half: Any
    tail: Tail = "upper"

    def __init__(self, amount: Any, half: Any) -> None:
        super().__init__(amount=amount, half=half)

    @property
    def index_names(self) -> tuple[str, ...]:
        return ("index",)

    def parameters(self) -> dict[str, float | Free]:
        return {"amount": self.amount, "half": self.half}

    def with_values(self, values: Mapping[str, float]) -> Self:
        merged = {**self.parameters(), **values}
        return type(self)(merged["amount"], merged["half"])

    def evaluate(self, index: Any) -> Any:
        x = np.maximum(np.asarray(index, dtype=np.float64), 0.0)
        return self.amount * x / (x + self.half)

    def to_document(self) -> dict[str, Any]:
        def document(value: Any) -> Any:
            return value.to_document() if isinstance(value, Free) else value

        return {
            "kind": self.kind,
            "amount": document(self.amount),
            "half": document(self.half),
            "tail": self.tail,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        return cls(document["amount"], document["half"])


def test_a_non_linear_family_is_searched_on_a_grid_without_a_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_solver, "require", _missing_solver)
    cover = Cover(
        INDEX,
        Saturating(Free(0.0, 60.0, grid=7), Free(0.5, 2.0, grid=[0.5, 1.0, 2.0])),
        payout_unit="EUR",
    )
    result = design(cover, SAMPLE, MSE(), limit=Limit(45.0))
    assert result.route == "grid" and result.solver is None
    payout = result.cover.payout_values(SAMPLE.index[:, 0])
    assert payout.max() <= 45.0
    statuses = [row["status"] for row in _profile(result)]
    assert "over_limit" in statuses


# -- errors ------------------------------------------------------------------


def _missing_solver() -> Any:
    raise SolverUnavailableError(
        "designing a cover needs the HiGHS solver: `pip install crc-sdk[parametric]`",
        extra="parametric",
    )


def test_a_missing_solver_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "highspy", None)
    with pytest.raises(
        SolverUnavailableError, match=r"crc-sdk\[parametric\]"
    ) as raised:
        design(_step([Free(), Free()]), SAMPLE, MSE())
    assert raised.value.extra == "parametric"
    assert isinstance(raised.value, ImportError)


def test_infeasible_designs_raise() -> None:
    with pytest.raises(InfeasibleDesignError) as raised:
        design(
            _step([Free(lower=10.0), Free(lower=10.0)]),
            SAMPLE,
            ExpectedUncovered(),
            budget=Budget(amount=1.0),
        )
    assert raised.value.asset_id == "A"
    assert raised.value.status == "infeasible"
    assert raised.value.solver_message


def test_a_fixed_amount_above_the_limit_is_infeasible() -> None:
    with pytest.raises(InfeasibleDesignError, match="levels\\[0\\]") as raised:
        design(_step([30.0, Free()]), SAMPLE, MSE(), limit=Limit(20.0))
    assert raised.value.asset_id == "A"


def test_no_feasible_grid_candidate_raises() -> None:
    template = _step(
        [Free(lower=10.0), Free(lower=10.0)],
        breakpoints=(Free(0.0, 1.0, grid=[0.5, 1.0]), 2.5),
    )
    with pytest.raises(InfeasibleDesignError, match="no grid candidate"):
        design(template, SAMPLE, ExpectedUncovered(), budget=Budget(amount=1.0))


def test_unbounded_payouts_need_a_budget() -> None:
    with pytest.raises(DesignError, match="budget or bounded amounts"):
        design(_step([Free(), Free()]), SAMPLE, ExpectedUncovered())
    with pytest.raises(DesignError, match="budget or bounded amounts"):
        design(_step([Free(), Free()]), SAMPLE, WeightedCVaR(0.5, 0.9))
    bounded = design(
        _step([Free(upper=100.0), Free(upper=100.0)]), SAMPLE, ExpectedUncovered()
    )
    assert bounded.objective_value == pytest.approx(0.0, abs=1e-9)


def _patched(status: str, calls: list[Mapping[str, Any] | None]) -> Any:
    def solve(
        programme: _solver.Programme,
        options: Mapping[str, Any] | None = None,
        *,
        keep_basis: bool = False,
    ) -> _solver.Solution:
        calls.append(options)
        return _solver.Solution(status, None, None, f"patched {status}", 3, 0.0)  # type: ignore[arg-type]

    return solve


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("unbounded", UnboundedDesignError),
        ("infeasible", InfeasibleDesignError),
        ("error", DesignError),
    ],
)
def test_solver_statuses_raise_typed_errors(
    monkeypatch: pytest.MonkeyPatch, status: str, error: type[DesignError]
) -> None:
    calls: list[Mapping[str, Any] | None] = []
    monkeypatch.setattr(_solver, "solve_lp", _patched(status, calls))
    with pytest.raises(error) as raised:
        design(
            _step([Free(), Free()]),
            SAMPLE,
            ExpectedUncovered(),
            budget=Budget(amount=5.0),
        )
    assert type(raised.value) is error
    assert raised.value.status == status
    assert raised.value.solver_message == f"patched {status}"
    assert raised.value.asset_id == "A"


def test_an_ambiguous_status_is_solved_again_without_presolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, Any] | None] = []
    monkeypatch.setattr(_solver, "solve_qp", _patched("infeasible_or_unbounded", calls))
    with pytest.raises(DesignError) as raised:
        design(_step([Free(), Free()]), SAMPLE, MSE())
    assert len(calls) == 2
    assert calls[1] == {"presolve": "off"}
    assert raised.value.status == "infeasible_or_unbounded"


def test_a_solution_that_breaks_the_budget_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _solver.solve_lp

    def inflated(
        programme: _solver.Programme,
        options: Mapping[str, Any] | None = None,
        *,
        keep_basis: bool = False,
    ) -> _solver.Solution:
        solution = real(programme, options, keep_basis=keep_basis)
        assert solution.values is not None
        values = solution.values.copy()
        values[:2] *= 1.01
        return _solver.Solution(
            "optimal",
            values,
            solution.objective,
            solution.message,
            solution.iterations,
            solution.seconds,
        )

    monkeypatch.setattr(_solver, "solve_lp", inflated)
    with pytest.raises(DesignError, match="exceeds the budget") as raised:
        design(
            _step([Free(), Free()]),
            SAMPLE,
            ExpectedUncovered(),
            budget=Budget(amount=20.0),
        )
    assert raised.value.status == "infeasible_solution"


def test_the_template_must_suit_the_sample() -> None:
    with pytest.raises(ValueError, match="no free parameters"):
        design(_step([1.0, 2.0]), SAMPLE, MSE())
    other_unit = Cover(INDEX, StepPayout([1.0], [Free()]), payout_unit="USD")
    with pytest.raises(SampleError, match="pays in"):
        design(other_unit, SAMPLE, MSE())
    other_index = Cover(
        IndexDefinition("wind", "km/h", aggregation="annual_maximum"),
        StepPayout([1.0], [Free()]),
        payout_unit="EUR",
    )
    with pytest.raises(SampleError, match="index"):
        design(other_index, SAMPLE, MSE())
    other_asset = Cover(
        INDEX, StepPayout([1.0], [Free()]), payout_unit="EUR", asset_id="B"
    )
    with pytest.raises(SampleError, match="asset"):
        design(other_asset, SAMPLE, MSE())


@pytest.mark.parametrize(
    "arguments",
    [
        {"template": "cover"},
        {"sample": "sample"},
        {"objective": "mse"},
        {"pricing": "pure"},
        {"budget": 0.8},
        {"limit": 10.0},
        {"options": {"ranges": True}},
    ],
)
def test_arguments_of_the_wrong_type(arguments: dict[str, Any]) -> None:
    call: dict[str, Any] = {
        "template": _step([Free(), Free()]),
        "sample": SAMPLE,
        "objective": MSE(),
        **arguments,
    }
    template, sample, objective = (
        call.pop("template"),
        call.pop("sample"),
        call.pop("objective"),
    )
    with pytest.raises(TypeError):
        design(template, sample, objective, **call)


def test_design_options_validation() -> None:
    for bad in (
        {"max_grid_points": 0},
        {"feasibility_tolerance": 0.0},
        {"fill": "middle"},
        {"ranges": 1},
    ):
        with pytest.raises(ValueError):
            DesignOptions(**bad)
    assert DesignOptions().to_document() == {
        "feasibility_tolerance": 1e-7,
        "fill": "lowest",
        "max_grid_points": 10_000,
        "ranges": False,
    }


# -- degenerate tails --------------------------------------------------------


def test_degenerate_tail_flags() -> None:
    template = _step([Free(), Free()])
    budget = Budget(amount=10.0)
    zero = design(template, SAMPLE, WeightedCVaR(0.9, 0.2), budget=budget)
    assert zero.p_zero_loss == 0.25
    assert zero.degenerate_tail == "expected_uncovered"
    single = design(template, SAMPLE, WeightedCVaR(0.9, 0.8), budget=budget)
    assert single.tail_outcomes == pytest.approx(0.8)
    assert single.degenerate_tail == "single_outcome"
    regular = design(template, SAMPLE, WeightedCVaR(0.9, 0.5), budget=budget)
    assert regular.tail_outcomes == pytest.approx(2.0)
    assert regular.degenerate_tail is None
    row = evaluate(single, SAMPLE, split="in_sample").to_pylist()[0]
    assert row["degenerate_tail"] == "single_outcome"


# -- the record, evaluation and determinism ----------------------------------


def test_a_design_is_a_designed_cover_for_evaluate(tmp_path: Path) -> None:
    train, test = holdout(
        _sample(
            np.linspace(0.0, 3.0, 20).tolist(), np.linspace(0.0, 40.0, 20).tolist()
        ),
        0.7,
    )
    result = design(
        _step([Free(), Free()]),
        train,
        WeightedCVaR(0.9, 0.9),
        budget=Budget(ratio=0.8),
        options=DesignOptions(ranges=True),
    )
    assert isinstance(result, DesignedCover)
    assert result.sample_hash == train.content_hash
    table = compare([result], test, split="out_of_sample")
    row = table.to_pylist()[0]
    assert row["design_id"] == result.design_id
    assert row["premium_basis"] == "design" and row["premium"] == result.premium
    metadata = EvaluationMetadata.from_table(table)
    assert metadata.designs[0]["design_id"] == result.design_id
    assert metadata.designs[0]["route"] == "lp"
    assert metadata.solvers == (dict(result.solver or {}),)
    write_comparison(table, tmp_path / "designs.parquet")
    written = EvaluationMetadata.from_table(pq.read_table(tmp_path / "designs.parquet"))
    assert written.designs[0]["ranges"] == result.to_record()["ranges"]


def test_the_record_is_plain_json() -> None:
    result = design(
        _step([Free(), Free()]),
        SAMPLE,
        ExpectedUncovered(),
        pricing=CostOfCapitalLoad(0.06, 0.75),
        budget=Budget(ratio=0.8),
        options=DesignOptions(ranges=True),
    )
    record = result.to_record()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["solver"] == {"name": "highs", "version": record["solver"]["version"]}
    assert record["solve"]["status"] == "optimal"
    assert (
        record["solve"]["options"]["ranging"]["primal_feasibility_tolerance"] == 1e-10
    )
    assert record["solve"]["solves"] == result.solves > 1
    assert record["objective"] == {"kind": "expected_uncovered"}
    assert record["pricing"] == {"kind": "cost_of_capital", "q": 0.75, "rate": 0.06}
    assert record["budget"] == {"ratio": 0.8}
    assert record["sample_hash"] == SAMPLE.content_hash
    assert record["template_hash"] == _step([Free(), Free()]).definition_hash
    assert record["cover_hash"] == result.cover.definition_hash
    assert set(record["parameters"]) == {"levels[0]", "levels[1]"}


def test_designs_are_deterministic_and_ids_follow_the_inputs() -> None:
    template = _step([Free(), Free()])
    first = design(template, SAMPLE, WeightedCVaR(0.9, 0.5), budget=Budget(amount=10.0))
    second = design(
        template, SAMPLE, WeightedCVaR(0.9, 0.5), budget=Budget(amount=10.0)
    )
    assert first.design_id == second.design_id
    assert first.parameters == second.parameters
    assert first.cover == second.cover
    ids = {
        first.design_id,
        design(
            template, SAMPLE, WeightedCVaR(0.9, 0.6), budget=Budget(amount=10.0)
        ).design_id,
        design(
            template, SAMPLE, WeightedCVaR(0.9, 0.5), budget=Budget(amount=11.0)
        ).design_id,
        design(
            template,
            SAMPLE,
            WeightedCVaR(0.9, 0.5),
            budget=Budget(amount=10.0),
            options=DesignOptions(ranges=True),
        ).design_id,
        design(
            template,
            SAMPLE,
            WeightedCVaR(0.9, 0.5),
            budget=Budget(amount=10.0),
            pricing=PurePremium(),
            limit=Limit(100.0),
        ).design_id,
    }
    assert len(ids) == 5
    assert all(item.startswith("sha256:") for item in ids)


def test_the_label_is_kept() -> None:
    template = Cover(
        INDEX, StepPayout([1.0, 2.5], [Free(), Free()]), payout_unit="EUR", label="mine"
    )
    result = design(template, SAMPLE, MSE())
    assert result.cover.label == "mine"
    assert result.cover.free_parameters() == ()
    assert DESIGN_SEMANTICS_VERSION == "crc-parametric-design/1"
