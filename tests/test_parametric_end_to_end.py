"""End to end, offline: from synthetic sources to a written comparison file.

- the quick start of `docs/parametric.md`, executed as written;
- the function route from a synthetic canonical file of fitted curves;
- the plan from a synthetic annual-extremes file, with k-fold splits.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact

from crc_sdk.connectors import (
    hazard_arrow_schema,
    read_hazard_dataset,
    read_hazard_metadata,
    write_hazard_dataset,
)
from crc_sdk.geometry import point_to_cell
from crc_sdk.parametric import (
    MSE,
    Budget,
    ComonotoneIndexWarning,
    CostOfCapitalLoad,
    Cover,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    ImpactNoise,
    IndexDefinition,
    KFold,
    LinearPayout,
    LossModel,
    MetricOptions,
    ParametricPlan,
    StepPayout,
    WeightedCVaR,
    compare,
    design,
    sample_curve,
    to_long,
    write_comparison,
)
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

DOCS = Path(__file__).resolve().parents[1] / "docs" / "parametric.md"
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
OPTIONS = MetricOptions(betas=(0.9, 0.99), return_periods=(10, 50))


def _python_blocks(heading: str) -> list[str]:
    """The ```python blocks of one section of the docs page, in order."""
    level = len(heading) - len(heading.lstrip("#"))
    blocks: list[str] = []
    inside = False
    fence: str | None = None
    code: list[str] = []
    for line in DOCS.read_text(encoding="utf-8").splitlines():
        if fence is not None:
            if line.startswith("```"):
                if fence == "python" and inside:
                    blocks.append("\n".join(code) + "\n")
                fence, code = None, []
            else:
                code.append(line)
            continue
        if line.startswith("```"):
            fence = line[3:].strip()
            continue
        hashes = len(line) - len(line.lstrip("#"))
        if hashes and line[hashes : hashes + 1] == " ":
            if line == heading:
                inside = True
            elif hashes <= level:
                inside = False
    return blocks


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
def _quiet_comonotone() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        yield


def test_the_docs_quick_start_runs_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    quick_start = _python_blocks("## Quick start")
    by_hand = _python_blocks("### From per-year values")
    assert len(quick_start) == 2 and len(by_hand) == 1
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {}
    for block in (*quick_start, *by_hand):
        exec(compile(block, str(DOCS), "exec"), namespace)
    printed = capsys.readouterr().out
    before, after = printed.split("Level:")[1:3]
    assert "Basis: annual, one draw is one year." in before
    assert "the evaluation sample, not known yet" in before
    assert (
        "Split: out of sample, fold 0, a holdout with train fraction 0.7: "
        "evaluated on 12 recorded years (2009 to 2020), the held-out block; "
        "designed on 28 recorded years (1981 to 2008), the design block; the "
        "leakage check passed"
    ) in after
    assert "Design step (expected_uncovered) (" in after
    assert "Basis risk: impact noise" in after

    result = namespace["result"]
    assert (tmp_path / "comparison.parquet").is_file()
    assert result.row_count == 8  # three designs and one cover, in and out of sample
    out_of_sample = namespace["out_of_sample"]
    assert out_of_sample.num_rows == 4
    assert out_of_sample.column("cover_label").to_pylist() == [
        "step (mse)",
        "step (expected_uncovered)",
        "step (weighted_cvar alpha=0.9 beta=0.9)",
        "linear",
    ]
    assert out_of_sample.column("premium_basis").to_pylist() == [
        "design",
        "design",
        "design",
        "evaluation",
    ]
    sample = result.samples["warehouse-a"]
    assert sample.period[0] == 1981 and sample.period[-1] == 2020
    out = [
        e for e in result.metadata.setup.evaluations if e["split"] == "out_of_sample"
    ]
    assert out[0]["method"] == "holdout"
    assert out[0]["details"] == {"train_fraction": 0.7, "plan_seed": 42}
    # The function route by hand gives the plan's sample and design.
    assert namespace["sample"].content_hash == sample.content_hash
    by_hand_design = namespace["designed"]
    assert by_hand_design.parameters == result.designs[1].parameters
    written = pq.read_table(tmp_path / "by-hand.parquet")
    assert written.num_rows == 2
    planned = result.table.filter(
        pc.equal(result.table.column("split"), "out_of_sample")
    ).slice(1, 1)
    assert written.column("expected_uncovered")[0].as_py() == pytest.approx(
        planned.column("expected_uncovered")[0].as_py(), rel=1e-12
    )


def _canonical_file(path: Path) -> HazardDatasetMetadata:
    """Write a small canonical dataset: one Gumbel curve in each of two cells."""
    metadata = HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset="rx1day",
            version="2026.1",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
            checksum="sha256:" + "0" * 64,
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 500.0),
    )
    rows = [
        {
            "cell_index": point_to_cell(longitude, 50.94, 7),
            "source_id": f"fixture-{k}",
            "source_geometry": None,
            "hazard_name": "rx1day",
            "horizon": 2050,
            "pathway": "ssp245",
            "curve_kind": "fitted",
            "curve_type": "gumbel_r",
            "curve_shape": None,
            "curve_location": location,
            "curve_scale": 12.0,
            "curve_atom_probability": None,
            "curve_atom_location": None,
        }
        for k, (longitude, location) in enumerate(((6.96, 50.0), (7.50, 55.0)))
    ]
    write_hazard_dataset(
        pa.Table.from_pylist(rows, schema=hazard_arrow_schema()),
        path,
        metadata,
        max_workers=1,
    )
    return metadata


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_the_function_route_from_a_canonical_file(tmp_path: Path) -> None:
    path = tmp_path / "rx1day.parquet"
    written = _canonical_file(path)
    metadata = read_hazard_metadata(path)
    curves = read_hazard_dataset(path)
    cell = point_to_cell(6.96, 50.94, metadata.h3_resolution)
    rows = curves.filter(pc.equal(curves.column("cell_index"), cell)).to_pylist()
    assert len(rows) == 1
    loss_model = LossModel(RAMP, value=2e6, value_unit="EUR")

    def draws(seed: int) -> Any:
        return sample_curve(
            rows[0],
            metadata=metadata,
            n_draws=2_000,
            rng=np.random.default_rng(seed),
            loss_model=loss_model,
            basis_risk=ImpactNoise(0.3, upper=1.0),
            asset_id="warehouse-a",
            details={"file_sha256": "sha256:" + "ab" * 32},
        )

    design_draws, fresh_draws = draws(1), draws(2)
    pricing = CostOfCapitalLoad(0.06, 0.99)
    designs = [
        design(
            _step(), design_draws, objective, pricing=pricing, budget=Budget(ratio=1.0)
        )
        for objective in (MSE(), ExpectedUncovered(), WeightedCVaR(0.9, 0.99))
    ]
    covers: list[Any] = [*designs, _linear()]
    in_sample = compare(
        covers,
        design_draws,
        split="in_sample",
        pricing=pricing,
        options=OPTIONS,
        design_sample=design_draws,
    )
    out = compare(
        covers,
        fresh_draws,
        split="out_of_sample",
        pricing=pricing,
        options=OPTIONS,
        design_sample=design_draws,
    )
    table = pa.concat_tables([in_sample, out])
    output = tmp_path / "comparison.parquet"
    write_comparison(
        table,
        output,
        metadata=EvaluationMetadata.merge(
            EvaluationMetadata.from_table(in_sample), EvaluationMetadata.from_table(out)
        ),
    )

    read = pq.read_table(output)
    assert read.num_rows == 8
    assert read.column("split").to_pylist() == ["in_sample"] * 4 + ["out_of_sample"] * 4
    stored = EvaluationMetadata.from_table(read)
    methods = {e["split"]: e["method"] for e in stored.setup.evaluations}
    assert methods == {"in_sample": "in_sample", "out_of_sample": "fresh_draws"}
    hazard = stored.sources[0]
    assert hazard["kind"] == "curve"
    assert hazard["metadata"] == written.model_dump(mode="json")
    assert hazard["details"] == {"file_sha256": "sha256:" + "ab" * 32}
    assert stored.licences[0]["licence"] == "CC-BY-4.0"
    assert {d["design_id"] for d in stored.designs} == {d.design_id for d in designs}
    assert stored.solvers and stored.solvers[0]["name"] == "highs"
    long = to_long(read)
    assert set(long.column("column").to_pylist()) >= {
        "coverage_ratio",
        "tail_coverage_b0_99",
        "cvar_uncovered_b0_99",
        "loss_rp50",
    }


def test_the_plan_from_an_annual_extremes_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    years = np.arange(1971, 2021)
    cells = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
    annual = pa.table(
        {
            "year": pa.array(np.repeat(years, len(cells)), pa.int32()),
            "latitude": np.tile([lat for lat, _ in cells], len(years)),
            "longitude": np.tile([lon for _, lon in cells], len(years)),
            "value": rng.gumbel(50.0, 12.0, len(years) * len(cells)),
        }
    )
    source, assets_file = tmp_path / "annual.parquet", tmp_path / "assets.parquet"
    pq.write_table(annual, source)
    pq.write_table(
        pa.table(
            {
                "asset_id": ["warehouse-a", "depot-b"],
                "longitude": [6.96, 7.10],
                "latitude": [50.94, 51.10],
                "value_eur": [2e6, 8e5],
            }
        ),
        assets_file,
    )
    plan = (
        ParametricPlan.for_annual_extremes(
            source,
            assets_file,
            impact=RAMP,
            unit="mm/day",
            provenance=SourceProvenance(
                provider="example", dataset="synthetic rx1day", licence="CC0-1.0"
            ),
            value_column="value_eur",
            value_unit="EUR",
            seed=11,
        )
        .basis_risk(ImpactNoise(0.25, upper=1.0))
        .covers(_step(), _linear())
        .objective(
            ExpectedUncovered(),
            WeightedCVaR(alpha=0.9, beta=0.9),
            pricing=CostOfCapitalLoad(0.06, 0.99),
            budget=Budget(ratio=1.0),
        )
        .split(KFold(5))
        .metrics(OPTIONS)
    )
    output = tmp_path / "out" / "comparison.parquet"
    result = plan.write_parquet(output)

    read = pq.read_table(output)
    assert read.num_rows == 2 * 5 * 2 * 3  # assets, folds, splits, covers
    stored = EvaluationMetadata.from_table(read)
    assert stored == result.metadata
    folds = {
        (row["asset_id"], row["split"], row["fold"])
        for row in read.select(["asset_id", "split", "fold"]).to_pylist()
    }
    assert len(folds) == 2 * 5 * 2
    # Every held-out year is held out once per asset.
    held_out = {
        entry["sample_hash"]: entry["periods"]
        for entry in stored.setup.samples
        if "evaluation" in entry["roles"]
    }
    assert held_out
    assert stored.assets == {
        "count": 2,
        "id_column": "asset_id",
        "values_sha256": result.metadata.assets["values_sha256"],
    }
    hazard = next(e for e in stored.sources if e["role"] == "hazard")
    assert hazard["annual_table"]["file_sha256"].startswith("sha256:")
    assert hazard["series"]["kind"] == "nearest_cell"
    assert hazard["series"]["cell"] == {"latitude": 50.875, "longitude": 6.875}
    assert {d["route"] for d in stored.designs} == {"lp"}
    assert len(result.designs) == 2 * 5 * 2
    assert all(
        d.design_sample.content_hash
        in {s["sample_hash"] for s in stored.setup.samples if "design" in s["roles"]}
        for d in result.designs
    )
