"""EvaluationMetadata: JSON round trip, merging, the table check, setup_text."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.parametric import (
    AnnualSeries,
    BinaryPayout,
    Budget,
    CostOfCapitalLoad,
    Cover,
    DesignResult,
    EvaluationError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    ImpactNoise,
    IndexDefinition,
    LinearPayout,
    LossModel,
    MetricOptions,
    NoBasisRisk,
    OutcomeSample,
    SampleProvenance,
    WeightedCVaR,
    compare,
    design,
    evaluate,
    holdout,
    sample_annual,
    sample_curve,
    setup_text,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.evaluate import DesignedCover
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY, EvaluationSetup
from crc_sdk.types import EnsembleDescriptor, HazardDatasetMetadata, SourceProvenance

UNTYPED_SETUP_TEXT: Any = setup_text

INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
COVER = Cover(INDEX, BinaryPayout(100.0, 400.0), payout_unit="EUR", label="binary")


def _sample(*, start: int = 2000, loss_unit: str = "EUR") -> OutcomeSample:
    n = 20
    index = np.linspace(10.0, 150.0, n)
    return OutcomeSample(
        asset_ids=("warehouse-a",),
        period=np.arange(start, start + n),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=np.where(index > 60.0, index * 5.0, 0.0),
        hazard=None,
        index_unit="mm/day",
        loss_unit=loss_unit,
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": f"table {start}"},),
            semantics_source="caller",
        ),
    )


def _metadata(table: Any) -> EvaluationMetadata:
    return EvaluationMetadata.from_table(table)


def test_json_round_trip() -> None:
    metadata = _metadata(evaluate(COVER, _sample(), split="in_sample"))
    assert EvaluationMetadata.from_json(metadata.to_json()) == metadata
    assert EvaluationMetadata.from_json(metadata.to_json().encode()) == metadata


@pytest.mark.parametrize("text", ["{", "[]", '{"schema_version": "2.0"}'])
def test_invalid_json_raises(text: str) -> None:
    with pytest.raises(EvaluationError):
        EvaluationMetadata.from_json(text)


def test_non_finite_values_and_repeated_designs_raise() -> None:
    document = _metadata(evaluate(COVER, _sample(), split="in_sample")).model_dump()
    with pytest.raises(EvaluationError, match="plain, finite JSON"):
        EvaluationMetadata(**{**document, "sources": ({"x": np.nan},)})
    with pytest.raises(EvaluationError, match="needs a design_id"):
        EvaluationMetadata(**{**document, "designs": ({"route": "lp"},)})
    record = {"design_id": "sha256:" + "1" * 64}
    with pytest.raises(EvaluationError, match="repeat a design_id"):
        EvaluationMetadata(**{**document, "designs": (record, record)})
    # pydantic's own entry point keeps raising its ValidationError, a ValueError.
    with pytest.raises(ValueError):
        EvaluationMetadata.model_validate({**document, "designs": (record, record)})


def test_invalid_fields_raise_an_evaluation_error() -> None:
    document = _metadata(evaluate(COVER, _sample(), split="in_sample")).model_dump()
    with pytest.raises(EvaluationError, match="invalid EvaluationMetadata: platform"):
        EvaluationMetadata(**{**document, "platform": 3})
    with pytest.raises(EvaluationError, match="unknown"):
        EvaluationMetadata(**{**document, "unknown": 1})
    setup = {**document["setup"], "samples": ({"sample_hash": "sha256:" + "2" * 64},)}
    with pytest.raises(EvaluationError, match="needs a sample_hash and roles"):
        EvaluationMetadata(**{**document, "setup": setup})
    with pytest.raises(EvaluationError, match="needs a sample_hash, a split"):
        EvaluationSetup(metric_options={}, samples=(), evaluations=({"split": "x"},))


def test_merge_combines_without_repeats() -> None:
    first = _metadata(evaluate(COVER, _sample(), split="in_sample"))
    second = _metadata(evaluate(COVER, _sample(start=2100), split="in_sample"))
    merged = EvaluationMetadata.merge(first, second, first)
    assert len(merged.setup.samples) == 2
    assert len(merged.setup.evaluations) == 2
    assert len(merged.covers) == 1
    assert len(merged.sources) == 2
    assert merged.assets["count"] == 1
    assert merged.created_at == max(first.created_at, second.created_at)
    assert EvaluationMetadata.merge(first) == first


def test_merge_rejects_disagreeing_metadata() -> None:
    base = _metadata(evaluate(COVER, _sample(), split="in_sample"))
    other_options = _metadata(
        evaluate(
            COVER, _sample(), split="in_sample", options=MetricOptions(betas=(0.5,))
        )
    )
    with pytest.raises(EvaluationError, match="metric options"):
        EvaluationMetadata.merge(base, other_options)
    usd_cover = Cover(INDEX, BinaryPayout(100.0, 400.0), payout_unit="USD")
    usd = _metadata(evaluate(usd_cover, _sample(loss_unit="USD"), split="in_sample"))
    with pytest.raises(EvaluationError, match="payout unit"):
        EvaluationMetadata.merge(base, usd)
    with pytest.raises(EvaluationError, match="numpy_version"):
        EvaluationMetadata.merge(base, base.model_copy(update={"numpy_version": "0"}))
    with pytest.raises(EvaluationError, match="asset records"):
        EvaluationMetadata.merge(
            base, base.model_copy(update={"assets": {"count": 1, "id_column": "id"}})
        )
    record = {"design_id": "sha256:" + "1" * 64, "route": "lp"}
    one = base.model_copy(update={"designs": (record,)})
    two = base.model_copy(update={"designs": ({**record, "route": "qp"},)})
    with pytest.raises(EvaluationError, match="two different records"):
        EvaluationMetadata.merge(one, two)
    with pytest.raises(EvaluationError):
        EvaluationMetadata.merge()
    with pytest.raises(TypeError):
        EvaluationMetadata.merge(base, base.model_dump())  # type: ignore[arg-type]


def test_check_table_finds_undescribed_rows() -> None:
    table = compare([COVER], _sample(), split="in_sample")
    metadata = _metadata(table)
    metadata.check_table(table)
    other = _metadata(compare([COVER], _sample(start=2100), split="in_sample"))
    with pytest.raises(EvaluationError, match="sample"):
        other.check_table(table)
    with pytest.raises(EvaluationError, match="cover"):
        metadata.model_copy(update={"covers": ()}).check_table(table)
    with pytest.raises(EvaluationError, match="lacks the columns"):
        metadata.check_table(table.drop_columns(["fold"]))
    with pytest.raises(EvaluationError, match="repeats a row key"):
        metadata.check_table(pa.concat_tables([table, table]))
    designed = table.set_column(
        table.column_names.index("design_id"),
        "design_id",
        pa.array(["sha256:" + "3" * 64]),
    )
    with pytest.raises(EvaluationError, match="design"):
        metadata.check_table(designed)
    evaluations = tuple({**entry, "fold": 1} for entry in metadata.setup.evaluations)
    setup = metadata.setup.model_copy(update={"evaluations": evaluations})
    with pytest.raises(EvaluationError, match="no evaluation"):
        metadata.model_copy(update={"setup": setup}).check_table(table)


def test_from_table_needs_the_key() -> None:
    with pytest.raises(EvaluationError, match=EVALUATION_METADATA_KEY):
        EvaluationMetadata.from_table(pa.table({"x": [1]}))


# -- setup_text ----------------------------------------------------------------

LINEAR = Cover(
    INDEX, LinearPayout(60.0, 140.0, 800.0), payout_unit="EUR", label="linear"
)
DESIGN_ID = "sha256:" + "1" * 64


@dataclass(frozen=True)
class _Design:
    """A designed cover with a design record, as `design` returns one."""

    cover: Cover
    design_sample: OutcomeSample
    premium: float = 77495.11012350091
    design_id: str = DESIGN_ID
    unique: bool | None = None
    ties: int | None = 0
    degenerate_tail: str | None = None
    record: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "cover_hash": self.cover.definition_hash,
            "objective": {"alpha": 0.5, "beta": 0.95, "kind": "weighted_cvar"},
            "pricing": {"kind": "pure"},
            "budget": {"ratio": 1.0},
            "limit": {"amount": 1e6},
            "premium": self.premium,
            "route": "lp",
            "unique": self.unique,
            "ties": self.ties,
            "unidentified": [],
            "degenerate_tail": self.degenerate_tail,
            "solver": {"name": "stub", "version": "0"},
            **self.record,
        }


def _lines(text: str) -> dict[str, list[str]]:
    """The lines of a setup text by their label (`Split`, `Design`, ...)."""
    lines: dict[str, list[str]] = {}
    for line in text.splitlines():
        lines.setdefault(line.split(":")[0].split(" (")[0], []).append(line)
    return lines


TABLE_SAMPLE = (
    "Sample: rows of a user table, each a year as the caller states. Basis risk: "
    "not modelled; the index and the loss are used as given. Loss: from the "
    "table, in EUR."
)
TABLE_SOURCE = (
    'Source: the table "table 2000", with provider, dataset, horizon and pathway '
    "not recorded."
)
PRICING = (
    "Pricing: the pure premium, P = E[payout], on the evaluation sample, for the "
    "covers given as is; a design keeps its design premium, and premium_basis "
    "says which premium a row uses."
)
METRICS = (
    "Metrics: beta 0.9, 0.95, 0.99; return periods 2, 5, 10, 25, 50, 100, 250 "
    "years; loss thresholds 0 EUR; payout threshold 0 EUR."
)


def test_setup_text_of_an_in_sample_comparison() -> None:
    metadata = _metadata(compare([COVER], _sample(), split="in_sample"))
    assert setup_text(metadata) == "\n".join(
        [
            "Level: asset (warehouse-a). Scenario: baseline. Basis: annual, one "
            "draw is one year.",
            "Split: in sample, fold 0: evaluated on the design sample itself, 20 "
            "years (periods 2000 to 2019). In-sample numbers show the fit, not "
            "out-of-sample performance.",
            TABLE_SAMPLE,
            TABLE_SOURCE,
            PRICING,
            METRICS,
        ]
    )


def _labelled(result: DesignResult, label: str) -> DesignResult:
    """The design with its cover relabelled, as the parametric plan labels it."""
    cover = result.cover
    relabelled = Cover(
        cover.index,
        cover.payout,
        payout_unit=cover.payout_unit,
        asset_id=cover.asset_id,
        label=label,
    )
    return dataclasses.replace(result, cover=relabelled)


def test_setup_text_keeps_each_designs_label_when_two_share_a_cover() -> None:
    # Under a binding budget, E[U] and the weighted CVaR choose the same amount,
    # so the two designs resolve to the same cover with different design ids.
    sample = _sample()
    template = Cover(
        INDEX, BinaryPayout(100.0, Free()), payout_unit="EUR", label="binary"
    )
    designs = [
        _labelled(
            design(template, sample, ExpectedUncovered(), budget=Budget(ratio=0.2)),
            "binary (expected_uncovered)",
        ),
        _labelled(
            design(template, sample, WeightedCVaR(0.9, 0.9), budget=Budget(ratio=0.2)),
            "binary (weighted_cvar alpha=0.9 beta=0.9)",
        ),
    ]
    eu, wcvar = designs
    assert eu.cover.definition_hash == wcvar.cover.definition_hash
    assert eu.design_id != wcvar.design_id
    assert eu.budget_use == pytest.approx(1.0) == wcvar.budget_use

    table = compare(list(designs), sample, split="in_sample")
    metadata = _metadata(table)
    assert [record["cover_label"] for record in metadata.designs] == [
        "binary (expected_uncovered)",
        "binary (weighted_cvar alpha=0.9 beta=0.9)",
    ]
    lines = _lines(setup_text(metadata))["Design binary"]
    eu_id = eu.design_id.removeprefix("sha256:")[:12]
    wcvar_id = wcvar.design_id.removeprefix("sha256:")[:12]
    assert lines[0].startswith(
        f"Design binary (expected_uncovered) ({eu_id}): objective: E[U], the "
        "expected uncovered loss;"
    )
    assert lines[1].startswith(
        f"Design binary (weighted_cvar alpha=0.9 beta=0.9) ({wcvar_id}): "
        "objective: weighted CVaR"
    )
    # The comparison rows are those of each design evaluated alone.
    assert table.column("cover_label").to_pylist() == [
        "binary (expected_uncovered)",
        "binary (weighted_cvar alpha=0.9 beta=0.9)",
    ]
    for row, each in zip(table.to_pylist(), designs, strict=True):
        alone = evaluate(each, sample, split="in_sample").to_pylist()[0]
        assert row == alone

    # Records written without the label: an ambiguous cover hash gives no label,
    # rather than the wrong one.
    document = metadata.model_dump(mode="json")
    for record in document["designs"]:
        del record["cover_label"]
    unlabelled = setup_text(EvaluationMetadata.model_validate(document))
    assert f"Design {eu_id}: objective: E[U]" in unlabelled
    assert f"Design {wcvar_id}: objective: weighted CVaR" in unlabelled


def test_setup_text_of_a_holdout_comparison() -> None:
    train, test = holdout(_sample(), train_fraction=0.75)
    covers: list[Cover | DesignedCover] = [_Design(LINEAR, train), COVER]
    table = compare(covers, test, split="out_of_sample", design_sample=train)
    assert setup_text(_metadata(table)) == "\n".join(
        [
            "Level: asset (warehouse-a). Scenario: baseline. Basis: annual, one "
            "draw is one year.",
            "Split: out of sample, fold 0, a holdout with train fraction 0.75: "
            "evaluated on 5 years (periods 2015 to 2019), the held-out block; "
            "designed on 15 years (periods 2000 to 2014), the design block; the "
            "leakage check passed: the two share no period of one source.",
            TABLE_SAMPLE,
            TABLE_SOURCE,
            PRICING,
            "Design linear (111111111111): objective: weighted CVaR, alpha "
            "CVaR_beta(U) + (1 - alpha) E[U] with alpha 0.5 and beta 0.95; "
            "pricing: the pure premium, P = E[payout]; budget: premium at most 1 "
            "x E[L] of the design sample; limit: 1,000,000 EUR; design premium: "
            "77,495.1 EUR; route: lp; unique optimum: not checked; ties: 0; "
            "tail: not degenerate.",
            METRICS,
        ]
    )


@pytest.mark.parametrize("split", ["in_sample", "out_of_sample"])
def test_setup_text_states_every_item_of_the_publication_rule(split: str) -> None:
    # Portfolio or asset, basis, basis-risk model, pricing and split.
    train, test = holdout(_sample())
    sample = train if split == "in_sample" else test
    table = evaluate(_Design(LINEAR, train), sample, split=split)  # type: ignore[arg-type]
    text = setup_text(_metadata(table))
    lines = _lines(text)
    assert lines["Level"][0].startswith("Level: asset (warehouse-a).")
    assert "Basis: annual, one draw is one year." in lines["Level"][0]
    assert "Basis risk: " in lines["Sample"][0]
    assert lines["Pricing"][0].startswith("Pricing: the pure premium")
    assert lines["Split"][0].startswith(f"Split: {split.replace('_', ' ')}, fold 0")
    assert "not stated" not in text and "not known" not in text


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_setup_text_carries_the_notes_of_curve_samples() -> None:
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
        probability_semantics=None,
        return_period_support=(2.0, 50.0),
        ensemble=EnsembleDescriptor(pooling="pooled", models=5),
    )
    sample = sample_curve(
        FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0),
        metadata=metadata,
        n_draws=400,
        rng=np.random.default_rng(1),
        loss_model=LossModel(
            PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
            value=1e6,
            value_unit="EUR",
        ),
        basis_risk=ImpactNoise(0.3, upper=1.0),
        asset_id="warehouse-a",
        assume_semantics="annual_exceedance",
    )
    train, test = holdout(sample)
    table = evaluate(_Design(LINEAR, train), test, split="out_of_sample")
    lines = _lines(setup_text(_metadata(table)))
    assert lines["Split"] == [
        "Split: out of sample, fold 0, a holdout with train fraction 0.7: "
        "evaluated on 120 simulated years (draws 280 to 399), the held-out block; "
        "designed on 280 simulated years (draws 0 to 279), the design block; the "
        "leakage check passed: the two share no period of one source."
    ]
    assert lines["Sample"] == [
        "Sample: Monte Carlo draws from a hazard curve with annual_exceedance "
        "semantics, as the caller states. Basis risk: impact noise, a mean-one "
        "lognormal factor with sigma 0.3, the noisy impact capped at 1. Loss: "
        "impact function PiecewiseLinearImpact times a stated asset value, in EUR."
    ]
    notes = lines["Sample notes"][0]
    counts = sample.provenance.details["outside_support"]
    below, above = counts["below"], counts["above"]
    assert below > 0 and above > 0
    for note in sample.provenance.notes:
        assert note in notes
    assert notes.count("The dataset declares no probability semantics") == 1
    assert f"{below} of 400 draws are more frequent than the shortest" in notes
    assert f"{above} of 400 draws are rarer than the longest" in notes
    assert "pools an ensemble" in notes


def test_setup_text_of_per_year_samples() -> None:
    hazard = AnnualSeries(
        np.arange(2000, 2006),
        np.array([50.0, np.nan, 100.0, 70.0, 40.0, 90.0]),
        "mm/day",
        {"source": "station A"},
    )
    index = AnnualSeries(
        np.arange(2000, 2006),
        np.array([45.0, 60.0, 95.0, 75.0, 35.0, 80.0]),
        "mm/day",
        {"source": "station B"},
    )
    loss = LossModel(
        PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
        value_column="value",
        value_unit="EUR",
    ).for_asset(pa.table({"asset_id": ["warehouse-a"], "value": [2e6]}), "warehouse-a")
    sample = sample_annual(
        hazard,
        index=index,
        loss_model=loss,
        basis_risk=NoBasisRisk(),
        asset_id="warehouse-a",
        missing="drop_year",
    )
    lines = _lines(setup_text(_metadata(evaluate(LINEAR, sample, split="in_sample"))))
    assert lines["Split"][0].startswith(
        "Split: in sample, fold 0: evaluated on the design sample itself, 5 "
        "recorded years (2000 to 2005)."
    )
    assert lines["Sample"] == [
        "Sample: per-year values, one row per recorded year. Basis risk: no model. "
        "Loss: impact function PiecewiseLinearImpact times the asset value in "
        "column value, in EUR."
    ]
    assert lines["Source"] == [
        'Source: the hazard series "station A", with provider, dataset, horizon '
        'and pathway not recorded; the index series "station B", with provider, '
        "dataset, horizon and pathway not recorded."
    ]
    notes = lines["Sample notes"][0]
    assert "no trend is removed" in notes
    assert "1 year without a value was dropped: 2001." in notes
    assert "The index is read from another series than the loss." in notes


def test_setup_text_names_each_split_method() -> None:
    sample = _sample()
    train, test = year_block_kfold(sample, 4)[2]
    kfold = _lines(
        setup_text(
            _metadata(
                evaluate(
                    LINEAR, test, split="out_of_sample", fold=2, design_sample=train
                )
            )
        )
    )["Split"][0]
    assert kfold.startswith(
        "Split: out of sample, fold 2 of a 4-fold split over blocks of years: "
        "evaluated on 5 years (periods 2010 to 2014), the held-out block; "
        "designed on 15 years (periods 2000 to 2019), every block but the "
        "held-out one;"
    )

    def drawn(lineage: str) -> OutcomeSample:
        return OutcomeSample(
            **{
                **_fields(_sample()),
                "provenance": SampleProvenance(
                    source_kind="monte_carlo",
                    sources=(),
                    semantics_source="dataset",
                    probability_semantics="annual_value_distribution",
                    lineage=lineage,
                ),
            }
        )

    fresh = _lines(
        setup_text(
            _metadata(
                evaluate(
                    LINEAR,
                    drawn("seed 2"),
                    split="out_of_sample",
                    design_sample=drawn("seed 1"),
                )
            )
        )
    )
    assert (
        "fold 0, on fresh Monte Carlo draws: evaluated on 20 simulated years"
        in (fresh["Split"][0])
    )
    assert (
        "annual_value_distribution semantics, as the dataset declares"
        in (fresh["Sample"][0])
    )
    other = evaluate(
        LINEAR, _sample(start=2100), split="out_of_sample", design_sample=_sample()
    )
    assert "on a separate sample from another source:" in setup_text(_metadata(other))
    shared = SampleProvenance(
        source_kind="table", sources=(), semantics_source="caller", lineage="one"
    )
    early = OutcomeSample(**{**_fields(_sample()), "provenance": shared})
    late = OutcomeSample(**{**_fields(_sample(start=2050)), "provenance": shared})
    disjoint = evaluate(LINEAR, late, split="out_of_sample", design_sample=early)
    assert "on periods of the same source the design sample lacks:" in setup_text(
        _metadata(disjoint)
    )


def _fields(sample: OutcomeSample) -> dict[str, Any]:
    return {
        "asset_ids": sample.asset_ids,
        "period": sample.period,
        "probability": sample.probability,
        "index": sample.index,
        "loss": sample.loss,
        "hazard": None,
        "index_unit": sample.index_unit,
        "loss_unit": sample.loss_unit,
        "semantics": sample.semantics,
        "provenance": sample.provenance,
    }


def test_setup_text_writes_shared_statements_once() -> None:
    train, test = holdout(_sample())
    design = _Design(LINEAR, train)
    in_sample = _metadata(evaluate(design, train, split="in_sample"))
    out = _metadata(evaluate(design, test, split="out_of_sample"))
    lines = _lines(setup_text(EvaluationMetadata.merge(in_sample, out)))
    assert [line.split(",")[0] for line in lines["Split"]] == [
        "Split: in sample",
        "Split: out of sample",
    ]
    assert len(lines["Sample"]) == len(lines["Pricing"]) == 1
    assert len(lines["Design linear"]) == 1
    loaded = _metadata(
        evaluate(
            LINEAR,
            test,
            split="out_of_sample",
            design_sample=train,
            pricing=CostOfCapitalLoad(0.1, 0.99),
        )
    )
    plain = _metadata(evaluate(LINEAR, train, split="in_sample"))
    lines = _lines(setup_text(EvaluationMetadata.merge(plain, loaded)))
    assert lines["Split"][0].endswith(" Pricing: the pure premium, P = E[payout].")
    assert lines["Split"][1].endswith(
        " Pricing: a cost-of-capital load, P = (1 - 0.1) E[payout] + 0.1 "
        "CVaR_0.99(payout)."
    )
    assert lines["Pricing"][0].startswith("Pricing: as each split states,")


def test_setup_text_describes_samples_that_differ() -> None:
    first = _metadata(evaluate(COVER, _sample(), split="in_sample"))
    usd = _sample(start=2100)
    other = _metadata(evaluate(COVER, usd, split="in_sample", options=MetricOptions()))
    drawn = OutcomeSample(
        **{
            **_fields(_sample(start=3000)),
            "provenance": SampleProvenance(
                source_kind="monte_carlo",
                sources=(),
                semantics_source="dataset",
                probability_semantics="annual_exceedance",
            ),
        }
    )
    third = _metadata(evaluate(COVER, drawn, split="in_sample"))
    lines = _lines(setup_text(EvaluationMetadata.merge(first, other, third)))
    assert lines["Sample of 20 years"] == [
        "Sample of 20 years (periods 2000 to 2019): " + TABLE_SAMPLE[8:],
        "Sample of 20 years (periods 2100 to 2119): " + TABLE_SAMPLE[8:],
    ]
    assert lines["Sample of 20 simulated years"][0].startswith(
        "Sample of 20 simulated years (draws 3000 to 3019): Monte Carlo draws"
    )


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"unique": True, "ties": 3}, "unique optimum: yes; ties: 3"),
        ({"unique": False, "ties": None}, "unique optimum: no; ties: not counted"),
        (
            {"degenerate_tail": "expected_uncovered"},
            "degenerate tail: beta is at most the probability of a zero loss, so "
            "the objective is a multiple of E[U].",
        ),
        (
            {"degenerate_tail": "single_outcome"},
            "degenerate tail: the tail holds at most one draw.",
        ),
        ({"degenerate_tail": "other"}, "degenerate tail: other."),
        (
            {"record": {"unidentified": ["level_1", "level_2"], "fill": "highest"}},
            "amounts no design year reaches: level_1 and level_2, set to the "
            "highest value allowed",
        ),
        (
            {"record": {"budget": {"amount": 1234567.891}, "limit": None}},
            "budget: premium at most 1,234,570 EUR; no limit;",
        ),
        ({"record": {"budget": None}}, "; no budget; limit: 1,000,000 EUR;"),
        (
            {"record": {"objective": {"kind": "mse"}}},
            "objective: MSE, E[(L - payout)^2]",
        ),
        (
            {"record": {"objective": {"kind": "expected_uncovered"}}},
            "objective: E[U], the expected uncovered loss; pricing",
        ),
        (
            {
                "record": {
                    "objective": {"kind": "user", "name": "f", "parameters": {"a": 1}}
                }
            },
            'objective: the user objective f ({"a": 1})',
        ),
        (
            {
                "record": {
                    "pricing": {"kind": "cost_of_capital", "q": 0.99, "rate": 0.1}
                }
            },
            "pricing: a cost-of-capital load, P = (1 - 0.1) E[payout] + 0.1 "
            "CVaR_0.99(payout);",
        ),
        ({"premium": 0.000012345678}, "design premium: 0.0000123457 EUR"),
    ],
)
def test_setup_text_states_the_design_facts(
    fields: dict[str, Any], expected: str
) -> None:
    train, test = holdout(_sample())
    design = _Design(LINEAR, train, **fields)
    text = setup_text(_metadata(evaluate(design, test, split="out_of_sample")))
    assert expected in _lines(text)["Design linear"][0]


def test_setup_text_without_a_label_names_the_design_id() -> None:
    train, test = holdout(_sample())
    unlabelled = Cover(INDEX, LinearPayout(60.0, 140.0, 800.0), payout_unit="EUR")
    text = setup_text(
        _metadata(evaluate(_Design(unlabelled, train), test, split="out_of_sample"))
    )
    assert "\nDesign 111111111111: objective" in text
    assert "tail: not degenerate" in text


def test_setup_text_is_deterministic() -> None:
    train, test = holdout(_sample())
    metadata = _metadata(evaluate(_Design(LINEAR, train), test, split="out_of_sample"))
    text = setup_text(metadata)
    later = metadata.model_copy(update={"created_at": "2099-01-01T00:00:00+00:00"})
    assert setup_text(later) == text
    assert setup_text(EvaluationMetadata.from_json(metadata.to_json())) == text
    again = _metadata(evaluate(_Design(LINEAR, train), test, split="out_of_sample"))
    assert setup_text(again) == text


def test_setup_text_of_a_setup_alone() -> None:
    train, test = holdout(_sample())
    metadata = _metadata(evaluate(_Design(LINEAR, train), test, split="out_of_sample"))
    text = setup_text(metadata)
    alone = setup_text(metadata.setup)
    # A setup alone has no design or source records.
    assert alone == "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("Design ", "Source: "))
    )
    assert setup_text(metadata.setup.model_dump(mode="json")) == alone


def test_setup_text_of_a_setup_known_before_drawing() -> None:
    # Before execution a setup knows its models and split, but no hashes.
    setup: dict[str, Any] = {
        "basis": "annual",
        "metric_options": {"betas": [0.99]},
        "samples": [
            {
                "asset_ids": ["warehouse-a"],
                "basis_risk": {"kind": "impact_noise", "sigma": 0.25, "upper": None},
                "draws": 10000,
                "loss_unit": "EUR",
                "probability_semantics": "annual_exceedance",
                "roles": ["design", "evaluation"],
                "scenario": "baseline",
                "semantics_source": "dataset",
                "source_kind": "monte_carlo",
            }
        ],
        "evaluations": [
            {
                "details": {"train_fraction": 0.7},
                "fold": 0,
                "method": "holdout",
                "pricing": {"kind": "pure"},
                "split": "out_of_sample",
            }
        ],
    }
    assert setup_text(setup) == "\n".join(
        [
            "Level: asset (warehouse-a). Scenario: baseline. Basis: annual, one "
            "draw is one year.",
            "Split: out of sample, fold 0, a holdout with train fraction 0.7: "
            "evaluated on the evaluation sample, not known yet; designed on the "
            "design sample, not known yet; no leakage check is recorded.",
            "Sample: Monte Carlo draws from a hazard curve with annual_exceedance "
            "semantics, as the dataset declares. Basis risk: impact noise, a "
            "mean-one lognormal factor with sigma 0.25. Loss: model not stated, "
            "in EUR.",
            "Pricing: the pure premium, P = E[payout], on the evaluation sample, "
            "for the covers given as is; a design keeps its design premium, and "
            "premium_basis says which premium a row uses.",
            "Metrics: beta 0.99; return periods not stated; loss thresholds not "
            "stated; payout threshold not stated.",
        ]
    )
    assert setup_text({}) == "\n".join(
        [
            "Level: asset, the assets not stated. Scenario: not stated. Basis: not "
            "stated.",
            "Split: not known.",
            "Sample: not known.",
            "Pricing: not stated, on the evaluation sample, for the covers given as "
            "is; a design keeps its design premium, and premium_basis says which "
            "premium a row uses.",
            "Metrics: the options are not stated.",
        ]
    )


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ({"samples": "x"}, r"setup\.samples must be a list"),
        ({"samples": ["x"]}, r"setup\.samples must be a list of objects"),
        ({"samples": [{"draws": "many"}]}, r"setup\.samples\[0\]\.draws"),
        ({"samples": [{"draws": True}]}, r"setup\.samples\[0\]\.draws"),
        ({"samples": [{"periods": [1, 2, 3]}]}, "periods must be two integers"),
        ({"samples": [{"notes": [1]}]}, "notes must be a list of text"),
        ({"samples": [{"basis_risk": "noise"}]}, "basis_risk must be an object"),
        ({"samples": [{"loss_model": "ramp"}]}, "loss_model must be an object"),
        (
            {"samples": [{"basis_risk": {"kind": "impact_noise", "sigma": "a"}}]},
            r"basis_risk\.sigma must be a number",
        ),
        ({"evaluations": [{"fold": True}]}, r"setup\.evaluations\[0\]\.fold"),
        (
            {"evaluations": [{"details": {"plan_seed": "42"}}]},
            r"setup\.evaluations\[0\]\.details\.plan_seed has the wrong type",
        ),
        (
            {"evaluations": [{"details": {"fresh_draws_seed": True}}]},
            r"details\.fresh_draws_seed has the wrong type",
        ),
        ({"evaluations": [{"pricing": "pure"}]}, "pricing must be an object"),
        (
            {"evaluations": [{"pricing": {"kind": "cost_of_capital", "q": 0.9}}]},
            r"pricing\.rate must be a number",
        ),
        (
            {"evaluations": [{"split": "out_of_sample", "design_samples": "a"}]},
            "design_samples must be a list",
        ),
        ({"metric_options": {"betas": 0.9}}, "betas must be a list of numbers"),
        ({"metric_options": {"betas": [float("nan")]}}, "must be finite"),
        ({"metric_options": {"payout_threshold": "0"}}, "must be a number"),
    ],
)
def test_setup_text_rejects_malformed_setups(
    setup: dict[str, Any], message: str
) -> None:
    with pytest.raises(EvaluationError, match=message):
        setup_text(setup)


@pytest.mark.parametrize("value", [None, "setup", ["setup"], 3])
def test_setup_text_takes_metadata_or_a_setup(value: Any) -> None:
    with pytest.raises(TypeError, match="setup_text takes"):
        UNTYPED_SETUP_TEXT(value)


# -- setup_text: the source, a comparison table, the seeds, grouped notes -------

CURVE_METADATA = HazardDatasetMetadata(
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
    ),
    creation_version="1",
    probability_semantics="annual_exceedance",
    return_period_support=(2.0, 50.0),
)
CURVE_ROW: dict[str, Any] = {
    "cell_index": int("871fa1999ffffff", 16),
    "source_id": "fixture",
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
CURVE_SOURCE = (
    "Source: the hazard curve rx1day from provider fixture, dataset rx1day, "
    "horizon 2050 and pathway ssp245."
)


def _draws(seed: int, curve: Any = None) -> OutcomeSample:
    """400 simulated years from the fixture curve, drawn with `seed`."""
    return sample_curve(
        CURVE_ROW if curve is None else curve,
        metadata=CURVE_METADATA,
        n_draws=400,
        rng=np.random.default_rng(seed),
        loss_model=LossModel(
            PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
            value=1e6,
            value_unit="EUR",
        ),
        basis_risk=ImpactNoise(0.3, upper=1.0),
        asset_id="warehouse-a",
    )


def _beyond(sample: OutcomeSample) -> tuple[int, int]:
    """A curve sample's draws below and above the fitted support."""
    counts = sample.provenance.details["outside_support"]
    return counts["below"], counts["above"]


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning",
    "ignore::crc_sdk.parametric.ComonotoneIndexWarning",
)
def test_setup_text_names_the_source() -> None:
    curve = _metadata(evaluate(LINEAR, _draws(1), split="in_sample"))
    assert _lines(setup_text(curve))["Source"] == [CURVE_SOURCE]

    # A curve given as a distribution has no canonical row, so no hazard,
    # horizon or pathway is recorded.
    distribution = FittedDistribution.from_parameters(
        "gumbel_r", location=50.0, scale=10.0
    )
    bare = _draws(1, distribution)
    assert _lines(setup_text(_metadata(evaluate(LINEAR, bare, split="in_sample"))))[
        "Source"
    ] == [
        "Source: the hazard curve from provider fixture and dataset rx1day, with "
        "horizon and pathway not recorded."
    ]

    # A per-year series names its provider and dataset; it has no horizon.
    series = AnnualSeries(
        np.arange(2000, 2004),
        np.array([50.0, 70.0, 100.0, 40.0]),
        "mm/day",
        {"source": {"provider": "fixture", "dataset": "station rx1day"}},
    )
    annual = sample_annual(
        series,
        loss_model=LossModel(
            PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
            value=1e6,
            value_unit="EUR",
        ),
        basis_risk=NoBasisRisk(),
        asset_id="warehouse-a",
    )
    assert _lines(setup_text(_metadata(evaluate(LINEAR, annual, split="in_sample"))))[
        "Source"
    ] == [
        "Source: the hazard series from provider fixture and dataset station "
        "rx1day, with horizon and pathway not recorded."
    ]

    # Distinct records are listed once each, in first-seen order.
    table = _metadata(evaluate(LINEAR, _sample(), split="in_sample"))
    merged = setup_text(EvaluationMetadata.merge(curve, table, curve))
    assert _lines(merged)["Source"] == [
        CURVE_SOURCE[:-1] + "; " + TABLE_SOURCE.removeprefix("Source: ")
    ]
    # No record: the text says so.
    nothing = curve.model_copy(update={"sources": ()})
    assert _lines(setup_text(nothing))["Source"] == ["Source: not recorded."]


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"role": True}, r"sources\[0\]\.role has the wrong type"),
        ({"kind": "curve", "metadata": "x"}, r"sources\[0\]\.metadata has the wrong"),
        (
            {"kind": "curve", "metadata": {"source": {"provider": 1.5}}},
            r"sources\[0\]\.provider has the wrong type",
        ),
        (
            {"kind": "curve", "metadata": {}, "row": {"horizon": True}},
            r"sources\[0\]\.row\.horizon has the wrong type",
        ),
        ({"series": {"source": 3}}, r"sources\[0\]\.series\.source has the wrong"),
    ],
)
def test_setup_text_rejects_malformed_source_records(
    record: dict[str, Any], message: str
) -> None:
    metadata = _metadata(evaluate(COVER, _sample(), split="in_sample"))
    with pytest.raises(EvaluationError, match=message):
        setup_text(metadata.model_copy(update={"sources": (record,)}))


def test_setup_text_takes_a_comparison_table(tmp_path: Path) -> None:
    train, test = holdout(_sample())
    covers: list[Cover | DesignedCover] = [_Design(LINEAR, train), COVER]
    table = compare(covers, test, split="out_of_sample", design_sample=train)
    expected = setup_text(EvaluationMetadata.from_table(table))
    assert setup_text(table) == expected

    # A file written by write_comparison, read back.
    path = tmp_path / "comparison.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert setup_text(read) == expected
    # Rows and columns taken from the table keep its metadata.
    some = read.select(["cover_label", "expected_uncovered"]).slice(0, 1)
    assert setup_text(some) == expected

    with pytest.raises(EvaluationError, match=EVALUATION_METADATA_KEY):
        setup_text(pa.table({"x": [1]}))
    broken = read.replace_schema_metadata({EVALUATION_METADATA_KEY: "{"})
    with pytest.raises(EvaluationError, match="not valid JSON"):
        setup_text(broken)
    with pytest.raises(EvaluationError, match="invalid EvaluationMetadata"):
        setup_text(read.replace_schema_metadata({EVALUATION_METADATA_KEY: "{}"}))


def _seeded(metadata: EvaluationMetadata, **details: int) -> EvaluationMetadata:
    """The metadata with `details` added to every evaluation, as a plan adds."""
    evaluations = tuple(
        {**entry, "details": {**(entry.get("details") or {}), **details}}
        for entry in metadata.setup.evaluations
    )
    setup = metadata.setup.model_copy(update={"evaluations": evaluations})
    return metadata.model_copy(update={"setup": setup})


def test_setup_text_states_the_seeds() -> None:
    train, test = holdout(_sample())
    design = _Design(LINEAR, train)
    in_sample = _metadata(evaluate(design, train, split="in_sample"))
    out = _metadata(evaluate(design, test, split="out_of_sample"))
    # The function route takes generators, so it records no seed.
    assert "Seeds" not in _lines(setup_text(EvaluationMetadata.merge(in_sample, out)))

    # A plan's seeds: the plan's on every evaluation, the fresh draws' on the
    # fresh-draw evaluations. Written once when every evaluation agrees.
    shared = EvaluationMetadata.merge(
        _seeded(in_sample, plan_seed=42),
        _seeded(out, plan_seed=42, fresh_draws_seed=7),
    )
    lines = _lines(setup_text(shared))
    assert lines["Seeds"] == ["Seeds: the plan's seed 42; the fresh draws' seed 7."]
    assert not any("Seeds" in line for line in lines["Split"])
    text = setup_text(shared)
    assert text.index("\nSeeds: ") > text.rindex("\nSplit: ")
    assert text.index("\nSeeds: ") < text.index("\nSample: ")
    assert setup_text(shared.setup).count("Seeds: the plan's seed 42;") == 1

    # Different seeds: each split states its own.
    differ = EvaluationMetadata.merge(
        _seeded(in_sample, plan_seed=42), _seeded(out, plan_seed=5)
    )
    lines = _lines(setup_text(differ))
    assert "Seeds" not in lines
    assert lines["Split"][0].endswith(" Seeds: the plan's seed 42.")
    assert lines["Split"][1].endswith(" Seeds: the plan's seed 5.")


@pytest.mark.filterwarnings(
    "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
)
def test_setup_text_writes_one_support_note_with_each_samples_counts() -> None:
    design_draws, fresh = _draws(1), _draws(2)
    table = evaluate(_Design(LINEAR, design_draws), fresh, split="out_of_sample")
    lines = _lines(setup_text(_metadata(table)))
    assert lines["Split"][0].startswith(
        "Split: out of sample, fold 0, on fresh Monte Carlo draws:"
    )
    (below, above), (fresh_below, fresh_above) = _beyond(design_draws), _beyond(fresh)
    assert min(below, above, fresh_below, fresh_above) > 0
    assert (below, above) != (fresh_below, fresh_above)
    assert lines["Sample notes"] == [
        f"Sample notes: {below} of 400 draws in the design sample and "
        f"{fresh_below} of 400 in the fresh draws are more frequent than the "
        "shortest fitted return period (2 years): the curve is extrapolated "
        f"there. {above} of 400 draws in the design sample and {fresh_above} of "
        "400 in the fresh draws are rarer than the longest fitted return period "
        "(50 years): the curve is extrapolated there."
    ]
    # Both samples come from one source, named once.
    assert lines["Source"] == [CURVE_SOURCE]


FREQUENT = (
    "draws are more frequent than the shortest fitted return period (2 years): "
    "the curve is extrapolated there."
)
RARE = (
    "draws are rarer than the longest fitted return period (50 years): the curve "
    "is extrapolated there."
)
POOLED = (
    "The curve pools an ensemble: a draw mixes model spread with year-to-year "
    "variability."
)


def _noted(
    digit: str, roles: list[str], lineage: str, *notes: str, asset: str = "a"
) -> dict[str, Any]:
    """A setup sample that carries `notes`."""
    return {
        "asset_ids": [asset],
        "draws": 400,
        "lineage": lineage,
        "notes": list(notes),
        "roles": roles,
        "sample_hash": "sha256:" + digit * 64,
        "source_kind": "monte_carlo",
    }


def _fresh(digit: str) -> dict[str, Any]:
    """A fresh-draw evaluation of sample `digit`."""
    return {
        "fold": 0,
        "method": "fresh_draws",
        "sample_hash": "sha256:" + digit * 64,
        "split": "out_of_sample",
    }


@pytest.mark.parametrize(
    ("samples", "evaluations", "expected"),
    [
        pytest.param(
            [
                _noted("1", ["design", "evaluation"], "x", f"210 of 400 {FREQUENT}"),
                _noted("2", ["evaluation"], "y", f"210 of 400 {FREQUENT}"),
            ],
            [_fresh("2")],
            f"210 of 400 {FREQUENT}",
            id="the same counts: one note, as it is",
        ),
        pytest.param(
            [
                _noted("1", ["design"], "x", f"210 of 400 {FREQUENT}", POOLED),
                _noted("2", ["evaluation"], "x", f"210 of 400 {FREQUENT}", POOLED),
            ],
            [],
            f"210 of 400 {FREQUENT} {POOLED}",
            id="blocks of one sample: its notes, as they are",
        ),
        pytest.param(
            [
                _noted(
                    "1", ["design"], "x", f"210 of 400 {FREQUENT}", f"9 of 400 {RARE}"
                ),
                _noted("2", ["evaluation"], "y", f"190 of 400 {FREQUENT}", POOLED),
            ],
            [_fresh("2")],
            f"210 of 400 draws in the design sample and 190 of 400 in the fresh "
            f"{FREQUENT} 9 of 400 draws in the design sample "
            f"{RARE.removeprefix('draws ')} {POOLED}",
            id="a note only one sample carries names its sample",
        ),
        pytest.param(
            [
                _noted("1", ["design"], "w", f"210 of 400 {FREQUENT}", asset="a"),
                _noted("2", ["evaluation"], "x", f"190 of 400 {FREQUENT}", asset="a"),
                _noted("3", ["design"], "y", f"205 of 400 {FREQUENT}", asset="b"),
                _noted("4", ["evaluation"], "z", f"195 of 400 {FREQUENT}", asset="b"),
            ],
            [_fresh("2"), _fresh("4")],
            "210 of 400 draws in the design sample for a, 190 of 400 in the fresh "
            "draws for a, 205 of 400 in the design sample for b and 195 of 400 in "
            f"the fresh draws for b {FREQUENT[6:]}",
            id="several assets",
        ),
        pytest.param(
            [
                _noted("1", ["design"], "x", f"210 of 400 {FREQUENT}", asset="a"),
                _noted("2", ["evaluation"], "x", f"210 of 400 {FREQUENT}", asset="a"),
                _noted("3", ["design"], "y", f"205 of 400 {FREQUENT}", asset="b"),
                _noted("4", ["evaluation"], "y", f"205 of 400 {FREQUENT}", asset="b"),
            ],
            [],
            f"210 of 400 draws for a and 205 of 400 for b {FREQUENT[6:]}",
            id="blocks of several assets",
        ),
        pytest.param(
            [
                _noted("1", ["design"], "x", f"210 of 400 {FREQUENT}"),
                _noted("2", ["evaluation"], "x", f"210 of 400 {FREQUENT}"),
                _noted("3", ["evaluation"], "y", f"190 of 400 {FREQUENT}"),
            ],
            [_fresh("3")],
            "210 of 400 draws in the sample split into blocks and 190 of 400 in "
            f"the fresh {FREQUENT}",
            id="blocks and fresh draws",
        ),
        pytest.param(
            [
                _noted("1", ["design"], "x", f"210 of 400 {FREQUENT}"),
                _noted("2", ["design"], "y", f"190 of 400 {FREQUENT}"),
            ],
            [],
            "210 of 400 draws in the design sample (sample 111111111111) and 190 "
            "of 400 in the design sample (sample 222222222222) "
            f"{FREQUENT[6:]}",
            id="two samples that read the same get their hashes",
        ),
    ],
)
def test_setup_text_groups_counted_notes_per_sample(
    samples: list[dict[str, Any]], evaluations: list[dict[str, Any]], expected: str
) -> None:
    setup = {"samples": samples, "evaluations": evaluations}
    assert _lines(setup_text(setup))["Sample notes"] == [f"Sample notes: {expected}"]
