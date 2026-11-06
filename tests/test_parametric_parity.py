"""Parity of the metrics with the golden fixtures in tests/fixtures/parametric.

The cases are read as JSON with plain numpy, as their README describes; no
reference code is imported. Evaluation cases compare the premium and every
metric of a fixed cover. Out-of-sample cases resolve the designed cover with
the case's parameters and compare its metrics on the design block (in sample)
and on the held-out block (out of sample, with the design premium).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import (
    Cover,
    LeakageError,
    OutcomeSample,
    SampleProvenance,
    evaluate,
    holdout,
)
from crc_sdk.parametric.metrics import MetricOptions, metric_columns
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
NEW_METRICS = {
    "coverage_ratio",
    "tail_coverage_b0_9",
    "tail_coverage_b0_95",
    "tail_coverage_b0_99",
    "cvar_uncovered_to_loss_b0_9",
    "cvar_uncovered_to_loss_b0_95",
    "cvar_uncovered_to_loss_b0_99",
    "false_negative_rate_l0",
    "false_positive_rate_l0",
}
ALL_METRICS = {column.name for column in metric_columns(MetricOptions())}


def _case(path: Path) -> dict[str, Any]:
    case: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return case


def _paths(pattern: str) -> list[Path]:
    return sorted(CASES.glob(pattern))


EVALUATE_CASES = _paths("*-fixed-*.json")
HOLDOUT_CASES = _paths("*-holdout.json")


def _sample(block: dict[str, Any], name: str, *, lineage: str) -> OutcomeSample:
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


def _scale(case: dict[str, Any]) -> float:
    return float(max(sum(row) for row in case["sample"]["loss"]))


def _check_metrics(
    row: dict[str, Any], expected: dict[str, Any], case: dict[str, Any]
) -> None:
    scale = _scale(case)
    relative = case["tolerances"]["metrics"]
    for name, value in expected.items():
        got = row[name]
        floor = 1e-9 if name.startswith(UNITLESS) else 1e-9 * scale
        if value is None:
            assert got is None, name
            continue
        if isinstance(value, dict):
            # A quantile on a step: either atom passes the fixture; the
            # module's rule gives the lower one, so check that one.
            lower, upper = value["either"]
            assert lower <= upper
            value = lower
        assert got is not None, name
        assert abs(got - value) <= relative * abs(value) + floor, (name, got, value)


def test_the_fixture_files_are_present() -> None:
    assert len(EVALUATE_CASES) == 17
    assert len(HOLDOUT_CASES) == 8


@pytest.mark.parametrize("path", EVALUATE_CASES, ids=lambda path: path.stem)
def test_evaluation_cases(path: Path) -> None:
    case = _case(path)
    assert case["kind"] == "evaluate" and case["scope"] == "individual"
    sample = _sample(case["sample"], case["case_id"], lineage=case["case_id"])
    cover = Cover.from_document(case["cover"])
    table = evaluate(
        cover,
        sample,
        split="in_sample",
        pricing=pricing_from_document(case["pricing"]),
    )
    row = table.to_pylist()[0]
    assert row["premium_basis"] == "evaluation"
    premium = case["expected"]["premium"]
    assert abs(row["premium"] - premium) <= (
        case["tolerances"]["premium"] * abs(premium) + 1e-9 * _scale(case)
    )
    expected = case["expected"]["metrics"]
    if case["source"] == "closed_form":
        assert set(expected) == ALL_METRICS
    else:
        assert set(expected) == ALL_METRICS - NEW_METRICS
    _check_metrics(row, expected, case)


@dataclass(frozen=True)
class _Design:
    """A designed cover as a design would return it (the parity stand-in)."""

    cover: Cover
    design_id: str
    premium: float
    design_sample: OutcomeSample
    unique: bool | None
    ties: int | None = None
    degenerate_tail: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {"design_id": self.design_id, "source": "parity fixture"}


def _holdout(case: dict[str, Any]) -> tuple[OutcomeSample, OutcomeSample, Cover]:
    design_block, held_out = case["sample"], case["evaluation_sample"]
    n = len(design_block["period"]) + len(held_out["period"])
    full = OutcomeSample(
        asset_ids=tuple(design_block["asset_ids"]),
        period=np.asarray(design_block["period"] + held_out["period"], dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=np.asarray(design_block["index"] + held_out["index"], dtype=np.float64),
        loss=np.asarray(design_block["loss"] + held_out["loss"], dtype=np.float64),
        hazard=None,
        index_unit=design_block["index_unit"],
        loss_unit=design_block["loss_unit"],
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": case["case_id"]},),
            semantics_source="caller",
        ),
    )
    train, test = holdout(full, train_fraction=case["split"]["train_fraction"])
    assert train.period.tolist() == design_block["period"]
    assert test.period.tolist() == held_out["period"]
    np.testing.assert_allclose(train.probability, design_block["probability"])
    np.testing.assert_allclose(test.probability, held_out["probability"])
    template = Cover.from_document(case["cover"])
    return train, test, template.with_values(case["expected"]["parameters"])


@pytest.mark.parametrize("path", HOLDOUT_CASES, ids=lambda path: path.stem)
def test_out_of_sample_cases(path: Path) -> None:
    case = _case(path)
    assert case["split"] == {"kind": "holdout", "train_fraction": 0.7}
    expected = case["expected"]
    out = expected["out_of_sample"]
    assert out["fold"] == 0 and out["premium_basis"] == "design"
    train, test, cover = _holdout(case)
    design = _Design(
        cover=cover,
        design_id="sha256:" + "0" * 64,
        premium=expected["premium"],
        design_sample=train,
        unique=expected["unique"],
    )

    in_sample = evaluate(design, train, split="in_sample").to_pylist()[0]
    assert in_sample["premium_basis"] == "design"
    _check_metrics(in_sample, expected["metrics"], case)
    priced = evaluate(
        cover,
        train,
        split="in_sample",
        pricing=pricing_from_document(case["pricing"]),
        design_sample=train,
    ).to_pylist()[0]
    assert priced["premium_basis"] == "evaluation"
    _check_metrics(priced, expected["metrics"], case)

    table = evaluate(design, test, split="out_of_sample")
    row = table.to_pylist()[0]
    assert row["split"] == "out_of_sample"
    assert row["fold"] == out["fold"]
    assert row["premium_basis"] == out["premium_basis"]
    assert row["unique"] is expected["unique"]
    assert row["premium"] == expected["premium"]
    _check_metrics(row, out["metrics"], case)

    stated = evaluate(
        cover,
        test,
        split="out_of_sample",
        premium=expected["premium"],
        design_sample=train,
    ).to_pylist()[0]
    assert stated["premium_basis"] == "stated"
    _check_metrics(stated, out["metrics"], case)

    with pytest.raises(LeakageError):
        evaluate(design, train, split="out_of_sample")


def test_the_held_out_bin_takes_the_filled_level() -> None:
    # In the thesis holdout cases no design year reaches levels[4] (50-55 m/s)
    # and one held-out year does; its payout follows the fill rule.
    case = _case(CASES / "step-wcvar-b0_9-pure-individual-thesis-holdout.json")
    assert "levels[4]" in case["expected"]["unidentified"]
    _, test, cover = _holdout(case)
    index = test.index[:, 0]
    reached = (index >= 50.0) & (index < 55.0)
    assert reached.sum() == 1
    parameters = case["expected"]["parameters"]
    assert cover.payout_values(index[reached])[0] == parameters["levels[4]"]
    assert parameters["levels[4]"] == parameters["levels[3]"]
