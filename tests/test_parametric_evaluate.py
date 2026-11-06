"""evaluate, compare, write_comparison and to_long."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    BinaryPayout,
    CostOfCapitalLoad,
    Cover,
    EvaluationError,
    EvaluationMetadata,
    Free,
    IndexDefinition,
    LeakageError,
    LinearPayout,
    MetricOptions,
    OutcomeSample,
    PurePremium,
    SampleError,
    SampleProvenance,
    StepPayout,
    UnresolvedParameterError,
    compare,
    evaluate,
    holdout,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.evaluate import DesignedCover
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY, IDENTITY_COLUMNS

UNTYPED_EVALUATE: Any = evaluate
UNTYPED_COMPARE: Any = compare
UNTYPED_WRITE: Any = write_comparison
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")


def _provenance(**fields: Any) -> SampleProvenance:
    base: dict[str, Any] = {
        "source_kind": "table",
        "sources": ({"role": "table", "name": "test table", "licence": "CC0-1.0"},),
        "semantics_source": "caller",
        "licences": ({"role": "table", "source": "test table", "licence": "CC0-1.0"},),
    }
    return SampleProvenance(**{**base, **fields})


def _sample(
    n: int = 40,
    *,
    start: int = 2000,
    asset: str = "warehouse-a",
    provenance: SampleProvenance | None = None,
    **overrides: Any,
) -> OutcomeSample:
    index = np.linspace(10.0, 150.0, n)
    fields: dict[str, Any] = {
        "asset_ids": (asset,),
        "period": np.arange(start, start + n),
        "probability": np.full(n, 1.0 / n),
        "index": index,
        "loss": np.where(index > 60.0, (index - 60.0) * 10.0, 0.0),
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": provenance or _provenance(),
    }
    fields.update(overrides)
    sample: OutcomeSample = OutcomeSample(**fields)
    return sample


def _linear(label: str | None = "linear", asset: str | None = "warehouse-a") -> Cover:
    return Cover(
        INDEX,
        LinearPayout(60.0, 140.0, 800.0),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


def _binary(label: str | None = "binary") -> Cover:
    return Cover(
        INDEX,
        BinaryPayout(100.0, 400.0),
        payout_unit="EUR",
        asset_id="warehouse-a",
        label=label,
    )


@dataclass(frozen=True)
class _Design:
    """A designed cover, as a design would return it."""

    cover: Cover
    design_sample: OutcomeSample
    premium: float = 120.0
    design_id: str = "sha256:" + "1" * 64
    unique: bool | None = True
    ties: int | None = 0
    degenerate_tail: str | None = None
    record: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "route": "lp",
            "solver": {"name": "stub", "version": "0"},
            **self.record,
        }


# -- one cover ---------------------------------------------------------------


def test_evaluate_returns_one_row_with_the_identity_columns() -> None:
    sample = _sample()
    table = evaluate(_linear(), sample, split="in_sample")
    assert table.num_rows == 1
    assert tuple(table.column_names[: len(IDENTITY_COLUMNS)]) == IDENTITY_COLUMNS
    assert table.schema.field("fold").type == pa.int32()
    assert table.schema.field("ties").type == pa.int32()
    assert table.schema.field("unique").type == pa.bool_()
    assert table.schema.field("coverage_ratio").type == pa.float64()
    row = table.to_pylist()[0]
    assert row["cover_label"] == "linear"
    assert row["cover_hash"] == _linear().definition_hash
    assert row["design_id"] is None
    assert (row["asset_id"], row["level"], row["scenario"]) == (
        "warehouse-a",
        "asset",
        "baseline",
    )
    assert (row["split"], row["fold"]) == ("in_sample", 0)
    assert row["sample_hash"] == sample.content_hash
    assert row["premium_basis"] == "evaluation"
    assert row["unique"] is None and row["ties"] is None
    assert row["degenerate_tail"] is None
    payout = _linear().payout_values(sample.index[:, 0])
    assert row["expected_payout"] == pytest.approx(payout.mean())
    assert row["premium"] == pytest.approx(payout.mean())


def test_the_label_defaults_to_the_hash_prefix() -> None:
    cover = _linear(label=None, asset=None)
    row = evaluate(cover, _sample(), split="in_sample").to_pylist()[0]
    assert row["cover_label"] == cover.definition_hash[7:19]
    assert row["asset_id"] == "warehouse-a"


def test_pricing_and_a_stated_premium() -> None:
    sample = _sample()
    payout = _linear().payout_values(sample.index[:, 0])
    loaded = evaluate(
        _linear(), sample, split="in_sample", pricing=CostOfCapitalLoad(0.1, 0.9)
    ).to_pylist()[0]
    assert loaded["premium"] > payout.mean()
    stated = evaluate(_linear(), sample, split="in_sample", premium=50).to_pylist()[0]
    free = evaluate(_linear(), sample, split="in_sample", premium=0).to_pylist()[0]
    assert stated["premium_basis"] == "stated"
    assert stated["premium"] == 50.0
    for name in ("var_net_loss_b0_9", "cvar_net_loss_b0_99", "net_loss_rp10"):
        assert stated[name] == pytest.approx(free[name] + 50.0)
    assert stated["var_uncovered_b0_9"] == free["var_uncovered_b0_9"]


def test_options_change_the_columns() -> None:
    table = evaluate(
        _linear(),
        _sample(),
        split="in_sample",
        options=MetricOptions(betas=(0.8,), return_periods=(20,), loss_thresholds=(1,)),
    )
    metrics = table.column_names[len(IDENTITY_COLUMNS) :]
    assert "tail_coverage_b0_8" in metrics and "loss_rp20" in metrics
    assert "false_negative_rate_l1" in metrics and "var_loss_b0_9" not in metrics
    metadata = EvaluationMetadata.from_table(table)
    assert set(metadata.columns) == set(metrics)
    assert metadata.setup.metric_options["betas"] == [0.8]


# -- several covers ----------------------------------------------------------


def test_compare_puts_covers_in_rows_and_metrics_in_columns() -> None:
    sample = _sample()
    table = compare([_linear(), _binary()], sample, split="in_sample")
    assert table.column("cover_label").to_pylist() == ["linear", "binary"]
    single = evaluate(_binary(), sample, split="in_sample").to_pylist()[0]
    assert table.to_pylist()[1] == single
    metadata = EvaluationMetadata.from_table(table)
    assert [entry.hash for entry in metadata.covers] == [
        _linear().definition_hash,
        _binary().definition_hash,
    ]
    assert metadata.covers[0].document == _linear().to_document()


def test_compare_rejects_a_repeated_cover() -> None:
    with pytest.raises(EvaluationError, match="repeat a row key"):
        compare([_linear(), _linear(label="again")], _sample(), split="in_sample")


def test_the_same_cover_as_given_and_as_designed_is_two_rows() -> None:
    sample = _sample()
    design = _Design(_linear(), design_sample=sample)
    table = compare([_linear(), design], sample, split="in_sample")
    assert table.column("design_id").to_pylist() == [None, design.design_id]
    assert table.column("premium_basis").to_pylist() == ["evaluation", "design"]
    assert table.column("unique").to_pylist() == [None, True]
    assert table.column("ties").to_pylist() == [None, 0]


@pytest.mark.parametrize("covers", ["linear", _linear(), [object()]])
def test_compare_rejects_invalid_cover_lists(covers: Any) -> None:
    with pytest.raises(TypeError):
        UNTYPED_COMPARE(covers, _sample(), split="in_sample")
    with pytest.raises(EvaluationError, match="at least one cover"):
        compare([], _sample(), split="in_sample")


# -- checks between cover and sample -----------------------------------------


def test_a_template_raises() -> None:
    template = Cover(
        INDEX, LinearPayout(60.0, 140.0, Free(0.0, 1e3)), payout_unit="EUR"
    )
    with pytest.raises(UnresolvedParameterError, match="limit"):
        evaluate(template, _sample(), split="in_sample")


@pytest.mark.parametrize(
    ("cover", "message"),
    [
        (Cover(INDEX, LinearPayout(60.0, 140.0, 800.0), payout_unit="USD"), "pays in"),
        (
            Cover(
                IndexDefinition("rx1day", "mm", aggregation="annual_maximum"),
                LinearPayout(60.0, 140.0, 800.0),
                payout_unit="EUR",
            ),
            "index is in",
        ),
        (_linear(asset="warehouse-b"), "written for asset"),
    ],
)
def test_units_and_assets_must_match(cover: Cover, message: str) -> None:
    with pytest.raises(SampleError, match=message):
        evaluate(cover, _sample(), split="in_sample")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"split": "held_out"}, "split must be one of"),
        ({"split": None}, "split must be one of"),
        ({"split": "in_sample", "fold": -1}, "fold must be"),
        ({"split": "in_sample", "fold": True}, "fold must be"),
        ({"split": "in_sample", "fold": 1.0}, "fold must be"),
        ({"split": "in_sample", "fold": 2**31}, "fold must be"),
        ({"split": "in_sample", "premium": -1.0}, "premium must be finite"),
        ({"split": "in_sample", "premium": float("nan")}, "premium must be finite"),
        ({"split": "in_sample", "premium": "10"}, "premium must be a number"),
    ],
)
def test_invalid_arguments_raise_evaluation_errors(
    arguments: dict[str, Any], message: str
) -> None:
    with pytest.raises(EvaluationError, match=message):
        UNTYPED_EVALUATE(_linear(), _sample(), **arguments)
    if "premium" not in arguments:
        with pytest.raises(EvaluationError, match=message):
            UNTYPED_COMPARE([_linear()], _sample(), **arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"pricing": {"kind": "pure"}},
        {"options": {"betas": (0.9,)}},
        {"design_sample": "train"},
    ],
)
def test_arguments_of_the_wrong_type_raise_type_errors(
    arguments: dict[str, Any],
) -> None:
    with pytest.raises(TypeError):
        UNTYPED_EVALUATE(_linear(), _sample(), split="in_sample", **arguments)
    with pytest.raises(TypeError):
        UNTYPED_EVALUATE(_linear(), np.ones(3), split="in_sample")


# -- in and out of sample ----------------------------------------------------


def test_holdout_rows_use_the_design_premium_out_of_sample() -> None:
    train, test = holdout(_sample(), train_fraction=0.75)
    design = _Design(_linear(), design_sample=train, premium=123.0)
    row = evaluate(design, test, split="out_of_sample").to_pylist()[0]
    assert (row["split"], row["fold"], row["premium_basis"]) == (
        "out_of_sample",
        0,
        "design",
    )
    assert row["premium"] == 123.0
    metadata = EvaluationMetadata.from_table(
        evaluate(design, test, split="out_of_sample")
    )
    evaluation = metadata.setup.evaluations[0]
    assert evaluation["method"] == "holdout"
    assert evaluation["details"] == {"train_fraction": 0.75}
    assert evaluation["leakage_check"] == "passed"
    assert evaluation["design_samples"] == [train.content_hash]
    roles = {entry["sample_hash"]: entry["roles"] for entry in metadata.setup.samples}
    assert roles == {test.content_hash: ["evaluation"], train.content_hash: ["design"]}
    assert metadata.designs[0]["route"] == "lp"
    assert metadata.solvers == ({"name": "stub", "version": "0"},)


def test_a_cover_given_as_is_out_of_sample_needs_its_design_sample() -> None:
    train, test = holdout(_sample())
    with pytest.raises(SampleError, match="needs the design sample"):
        evaluate(_linear(), test, split="out_of_sample")
    row = evaluate(
        _linear(), test, split="out_of_sample", design_sample=train
    ).to_pylist()[0]
    assert row["premium_basis"] == "evaluation"
    assert row["design_id"] is None


def test_leakage_raises() -> None:
    sample = _sample()
    train, test = holdout(sample)
    design = _Design(_linear(), design_sample=train)
    with pytest.raises(LeakageError):
        evaluate(design, sample, split="out_of_sample")
    with pytest.raises(LeakageError):
        evaluate(design, train, split="out_of_sample")
    with pytest.raises(LeakageError):
        evaluate(_linear(), sample, split="out_of_sample", design_sample=train)
    with pytest.raises(LeakageError):
        compare([_binary(), design], train, split="out_of_sample", design_sample=train)


def test_in_sample_means_the_design_sample() -> None:
    train, test = holdout(_sample())
    design = _Design(_linear(), design_sample=train)
    with pytest.raises(SampleError, match="in sample means"):
        evaluate(design, test, split="in_sample")
    with pytest.raises(SampleError, match="in sample means"):
        evaluate(_linear(), test, split="in_sample", design_sample=train)
    row = evaluate(design, train, split="in_sample").to_pylist()[0]
    assert row["premium_basis"] == "design"
    metadata = EvaluationMetadata.from_table(evaluate(design, train, split="in_sample"))
    assert metadata.setup.samples[0]["roles"] == ["design", "evaluation"]
    assert metadata.setup.evaluations[0]["method"] == "in_sample"


def test_a_conflicting_design_sample_raises() -> None:
    train, test = holdout(_sample())
    design = _Design(_linear(), design_sample=train)
    with pytest.raises(EvaluationError, match="differs from the design's own"):
        evaluate(design, test, split="out_of_sample", design_sample=test)
    with pytest.raises(EvaluationError, match="differs from the design's own"):
        compare([_binary(), design], test, split="out_of_sample", design_sample=test)
    row = evaluate(
        design, test, split="out_of_sample", design_sample=train
    ).to_pylist()[0]
    assert row["premium_basis"] == "design"


def test_kfold_rows_carry_their_fold() -> None:
    folds = year_block_kfold(_sample(), 4)
    train, test = folds[2]
    design = _Design(_linear(), design_sample=train)
    table = evaluate(design, test, split="out_of_sample", fold=2)
    assert table.column("fold").to_pylist() == [2]
    evaluation = EvaluationMetadata.from_table(table).setup.evaluations[0]
    assert (evaluation["method"], evaluation["details"]) == ("kfold", {"k": 4})
    with pytest.raises(EvaluationError, match="fold 2"):
        evaluate(design, test, split="out_of_sample")
    in_sample = evaluate(design, train, split="in_sample", fold=2)
    assert in_sample.column("fold").to_pylist() == [2]


def test_fresh_draws_and_other_samples() -> None:
    drawn = _provenance(source_kind="monte_carlo", semantics_source="dataset")
    design_sample = _sample(start=0, provenance=drawn)
    fresh = _sample(
        start=0,
        provenance=_provenance(
            source_kind="monte_carlo", semantics_source="dataset", lineage="seed 2"
        ),
    )
    design = _Design(_linear(), design_sample=design_sample)
    evaluation = EvaluationMetadata.from_table(
        evaluate(design, fresh, split="out_of_sample")
    ).setup.evaluations[0]
    assert evaluation["method"] == "fresh_draws"
    other = _sample(start=0, provenance=_provenance(lineage="another table"))
    evaluation = EvaluationMetadata.from_table(
        evaluate(design, other, split="out_of_sample")
    ).setup.evaluations[0]
    assert evaluation["method"] == "separate_sample"
    shared = _provenance(lineage="one series")
    early = _sample(start=1950, provenance=shared)
    late = _sample(start=2000, provenance=shared)
    evaluation = EvaluationMetadata.from_table(
        evaluate(_linear(), late, split="out_of_sample", design_sample=early)
    ).setup.evaluations[0]
    assert evaluation["method"] == "disjoint_periods"
    with pytest.raises(LeakageError):
        evaluate(
            _linear(),
            _sample(start=1980, provenance=shared),
            split="out_of_sample",
            design_sample=early,
        )


def test_the_design_sample_must_hold_the_same_asset() -> None:
    other = _sample(asset="warehouse-b")
    design = _Design(_linear(asset=None), design_sample=other)
    with pytest.raises(SampleError, match="different assets"):
        evaluate(design, _sample(start=2100), split="out_of_sample")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"design_id": ""}, "needs a design_id"),
        ({"premium": "120"}, "design premium must be a number"),
        ({"premium": -1.0}, "design premium must be finite"),
        ({"premium": float("inf")}, "design premium must be finite"),
        ({"unique": 1}, "unique must be a bool"),
        ({"ties": -1}, "ties must be an integer"),
        ({"ties": True}, "ties must be an integer"),
        ({"degenerate_tail": 3}, "degenerate_tail must be a string"),
        ({"record": {"design_id": "sha256:" + "2" * 64}}, "names the design_id"),
    ],
)
def test_invalid_designs_raise(overrides: dict[str, Any], message: str) -> None:
    train, test = holdout(_sample())
    fields: dict[str, Any] = {"cover": _linear(), "design_sample": train, **overrides}
    design: Any = _Design(**fields)
    assert isinstance(design, DesignedCover)
    with pytest.raises(EvaluationError, match=message):
        evaluate(design, test, split="out_of_sample")


# -- metadata, writing and the tidy form -------------------------------------


def test_the_metadata_records_the_setup() -> None:
    sample = _sample()
    table = evaluate(_linear(), sample, split="in_sample")
    raw = table.schema.metadata[EVALUATION_METADATA_KEY.encode()]
    document = json.loads(raw)
    assert list(document) == sorted(document)
    metadata = EvaluationMetadata.from_table(table)
    assert metadata.producer == "crc_sdk.parametric"
    assert metadata.schema_version == "1.0"
    assert metadata.numpy_version == np.__version__
    assert metadata.setup.basis == "annual"
    entry = metadata.setup.samples[0]
    assert entry["sample_hash"] == sample.content_hash
    assert entry["draws"] == 40 and entry["periods"] == [2000, 2039]
    assert entry["source_kind"] == "table"
    assert metadata.setup.evaluations[0]["pricing"] == {"kind": "pure"}
    assert metadata.columns["coverage_ratio"].unit == "1"
    assert metadata.columns["expected_loss"].unit == "EUR"
    assert metadata.sources[0]["sample_hash"] == sample.content_hash
    assert metadata.licences[0]["licence"] == "CC0-1.0"
    assert metadata.assets == {"count": 1, "id_column": None, "values_sha256": None}
    assert metadata.designs == () and metadata.solvers == ()


def test_write_comparison_round_trips(tmp_path: Path) -> None:
    table = compare([_linear(), _binary()], _sample(), split="in_sample")
    path = tmp_path / "comparison.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.to_pylist() == table.to_pylist()
    assert EvaluationMetadata.from_table(read) == EvaluationMetadata.from_table(table)
    assert [p.name for p in tmp_path.iterdir()] == ["comparison.parquet"]
    write_comparison(table, str(path))  # replaces the file


def test_write_comparison_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = evaluate(_linear(), _sample(), split="in_sample")
    path = tmp_path / "comparison.parquet"
    path.write_bytes(b"previous")

    def failing_write(output: Any, where: Any, **_: Any) -> None:
        Path(where).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", failing_write)
    with pytest.raises(OSError, match="disk full"):
        write_comparison(table, path)
    assert path.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["comparison.parquet"]


def test_write_comparison_checks_the_table(tmp_path: Path) -> None:
    table = evaluate(_linear(), _sample(), split="in_sample")
    with pytest.raises(EvaluationError, match="metric columns differ"):
        write_comparison(
            table.append_column("extra", pa.array([1.0])), tmp_path / "a.parquet"
        )
    with pytest.raises(EvaluationError, match="no 'crc.parametric.evaluation'"):
        write_comparison(table.replace_schema_metadata({}), tmp_path / "b.parquet")
    with pytest.raises(EvaluationError, match="local files"):
        write_comparison(table, "s3://bucket/comparison.parquet")
    with pytest.raises(TypeError):
        UNTYPED_WRITE(table.to_pylist(), tmp_path / "c.parquet")
    with pytest.raises(TypeError):
        UNTYPED_WRITE(table, tmp_path / "d.parquet", metadata={"setup": {}})
    assert list(tmp_path.iterdir()) == []


def test_rows_of_several_evaluations_in_one_file(tmp_path: Path) -> None:
    train, test = holdout(_sample())
    design = _Design(_linear(), design_sample=train)
    covers: list[Cover | DesignedCover] = [design, _binary()]
    in_sample = compare(covers, train, split="in_sample", design_sample=train)
    out = compare(covers, test, split="out_of_sample", design_sample=train)
    table = pa.concat_tables([in_sample, out])
    with pytest.raises(
        EvaluationError, match="no evaluation of sample|not in the metadata"
    ):
        write_comparison(table, tmp_path / "x.parquet")
    metadata = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(in_sample), EvaluationMetadata.from_table(out)
    )
    path = tmp_path / "comparison.parquet"
    write_comparison(table, path, metadata=metadata)
    read = pq.read_table(path)
    assert read.num_rows == 4
    assert read.column("split").to_pylist() == [
        "in_sample",
        "in_sample",
        "out_of_sample",
        "out_of_sample",
    ]
    stored = EvaluationMetadata.from_table(read)
    assert len(stored.setup.evaluations) == 2
    roles = {entry["sample_hash"]: entry["roles"] for entry in stored.setup.samples}
    assert roles == {
        train.content_hash: ["design", "evaluation"],
        test.content_hash: ["evaluation"],
    }
    assert len(stored.designs) == 1 and len(stored.covers) == 2


def test_to_long_gives_one_row_per_cover_and_metric() -> None:
    table = compare([_linear(), _binary()], _sample(), split="in_sample")
    long = to_long(table)
    n_metrics = table.num_columns - len(IDENTITY_COLUMNS)
    assert long.num_rows == 2 * n_metrics
    assert long.column_names == [
        *IDENTITY_COLUMNS,
        "column",
        "metric",
        "parameter",
        "parameter_value",
        "unit",
        "value",
    ]
    rows = {(row["cover_label"], row["column"]): row for row in long.to_pylist()}
    tail = rows[("binary", "tail_coverage_b0_99")]
    assert (tail["metric"], tail["parameter"], tail["parameter_value"]) == (
        "tail_coverage",
        "beta",
        0.99,
    )
    assert tail["unit"] == "1"
    assert tail["value"] == table.column("tail_coverage_b0_99").to_pylist()[1]
    rmse = rows[("linear", "rmse")]
    assert rmse["parameter"] is None and rmse["unit"] == "EUR"
    with pytest.raises(EvaluationError, match="no 'crc.parametric.evaluation'"):
        to_long(table.replace_schema_metadata({}))
    with pytest.raises(EvaluationError, match="metric columns differ"):
        to_long(table.drop_columns(["rmse"]))
    with pytest.raises(TypeError):
        to_long(table.to_pylist())


def test_null_metrics_are_null_in_the_table() -> None:
    sample = _sample(loss=np.zeros(40))
    row = evaluate(_linear(), sample, split="in_sample").to_pylist()[0]
    assert row["coverage_ratio"] is None and row["correlation"] is None
    assert row["expected_loss"] == 0.0


def test_step_covers_evaluate_on_bins() -> None:
    cover = Cover(
        INDEX,
        StepPayout([60.0, 100.0], [100.0, 300.0]),
        payout_unit="EUR",
        asset_id="warehouse-a",
    )
    sample = _sample(n=4, index=np.array([50.0, 60.0, 99.0, 100.0]))
    row = evaluate(cover, sample, split="in_sample", pricing=PurePremium()).to_pylist()[
        0
    ]
    assert row["expected_payout"] == pytest.approx((0 + 100 + 100 + 300) / 4)
