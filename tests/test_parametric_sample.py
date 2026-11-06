"""Outcome samples: the object, its hash and table, splits and user tables."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    ComonotoneIndexWarning,
    LeakageError,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    holdout,
    year_block_kfold,
)
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.sample import SAMPLE_METADATA_KEY, _check_out_of_sample
from crc_sdk.types import SourceProvenance

UNTYPED_SAMPLE: Any = OutcomeSample  # for calls with deliberately invalid arguments
UNTYPED_PROVENANCE: Any = SampleProvenance


def _provenance(**fields: Any) -> SampleProvenance:
    base: dict[str, Any] = {
        "source_kind": "table",
        "sources": ({"role": "table"},),
        "semantics_source": "caller",
    }
    return SampleProvenance(**{**base, **fields})


def _fields(n: int = 10, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "asset_ids": ("warehouse-a",),
        "period": np.arange(2000, 2000 + n),
        "probability": np.full(n, 1.0 / n),
        "index": np.linspace(10.0, 100.0, n),
        "loss": (np.arange(n) % 3).astype(np.float64),
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": _provenance(),
    }
    fields.update(overrides)
    return fields


def _sample(n: int = 10, **overrides: Any) -> OutcomeSample:
    sample: OutcomeSample = UNTYPED_SAMPLE(**_fields(n, **overrides))
    return sample


def _table(**overrides: Any) -> Any:
    columns: dict[str, Any] = {
        "year": pa.array([2003, 2001, 2002, 2000], pa.int64()),
        "asset": ["a1"] * 4,
        "rain": [80.0, 20.0, 45.0, 60.0],
        "loss": [5.0, 2.0, 0.0, 3.0],
    }
    columns.update(overrides)
    return pa.table(columns)


# -- the object --------------------------------------------------------------


def test_arrays_are_read_only_copies_with_an_asset_axis() -> None:
    index = np.linspace(10.0, 100.0, 10)
    sample = _sample(index=index)
    index[0] = -1.0
    assert sample.index.shape == (10, 1) and sample.loss.shape == (10, 1)
    assert sample.index[0, 0] == 10.0
    for array in (sample.period, sample.probability, sample.index, sample.loss):
        assert not array.flags.writeable
    assert sample.period.dtype == np.int64
    assert len(sample) == 10
    assert sample.scenario == "baseline"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"asset_ids": ("a", "b")}, "exactly one asset"),
        ({"asset_ids": ()}, "exactly one asset"),
        ({"asset_ids": (" a",)}, "asset id"),
        ({"asset_ids": "a"}, "sequence"),
        ({"period": np.arange(10.0)}, "integers"),
        ({"period": np.array([0, 1, 2, 3, 4, 4, 6, 7, 8, 9])}, "strictly increasing"),
        ({"probability": np.full(10, 0.1001)}, "sum to 1"),
        ({"probability": np.r_[0.0, np.full(9, 1 / 9)]}, "positive"),
        ({"probability": np.full(9, 1 / 9)}, "shape"),
        ({"index": np.ones((10, 2))}, "shape"),
        ({"index": np.r_[np.nan, np.ones(9)]}, "index must be finite"),
        ({"loss": np.r_[-1.0, np.ones(9)]}, "non-negative"),
        ({"loss": np.r_[np.inf, np.ones(9)]}, "loss must be finite"),
        ({"hazard": np.ones(9)}, "hazard"),
        ({"index_unit": ""}, "index_unit"),
        ({"semantics": "event"}, "reserved"),
        ({"scenario": ""}, "scenario"),
        ({"provenance": {"source_kind": "table"}}, "SampleProvenance"),
    ],
)
def test_invalid_fields_raise(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(SampleError, match=message):
        _sample(**overrides)


def test_the_content_hash_covers_the_long_table_and_its_units() -> None:
    sample = _sample(n=3, hazard=np.array([1.0, 2.0, 3.0]))
    expected = content_hash(
        {
            "asset_id": ["warehouse-a"] * 3,
            "hazard": [1.0, 2.0, 3.0],
            "index": [10.0, 55.0, 100.0],
            "index_unit": "mm/day",
            "loss": [0.0, 1.0, 2.0],
            "loss_unit": "EUR",
            "period": [2000, 2001, 2002],
            "probability": [1 / 3] * 3,
            "scenario": "baseline",
            "schema_version": "1.0",
            "semantics": "annual",
            "semantics_version": "crc-parametric-sample/1",
        }
    )
    assert sample.content_hash == expected
    # Pinned: a change here changes every stored sample hash.
    assert sample.content_hash == (
        "sha256:1da887bc6a43dbfe6a440836726462500df4141d51dd40b7493525a3ee54fd74"
    )
    assert _sample(n=3, hazard=np.array([1.0, 2.0, 3.0])).content_hash == expected
    for change in (
        {"loss_unit": "USD"},
        {"index_unit": "mm"},
        {"scenario": "ssp585"},
        {"hazard": None},
        {"loss": np.array([0.0, 1.0, 2.5])},
    ):
        changed = _sample(n=3, **{"hazard": np.array([1.0, 2.0, 3.0]), **change})
        assert changed.content_hash != expected
    other_provenance = _provenance(notes=("another note",))
    assert (
        _sample(
            n=3, hazard=np.array([1.0, 2.0, 3.0]), provenance=other_provenance
        ).content_hash
        == expected
    )


def test_to_table_is_the_long_table_with_its_metadata(tmp_path: Path) -> None:
    sample = _sample(n=4)
    table = sample.to_table()
    assert table.schema.names == [
        "scenario",
        "period",
        "asset_id",
        "probability",
        "index",
        "loss",
        "hazard",
    ]
    assert table.column("period").to_pylist() == [2000, 2001, 2002, 2003]
    assert table.column("loss").to_pylist() == [0.0, 1.0, 2.0, 0.0]
    assert table.column("hazard").null_count == 4
    metadata = json.loads(table.schema.metadata[SAMPLE_METADATA_KEY.encode()])
    assert metadata["content_hash"] == sample.content_hash
    assert metadata["index_unit"] == "mm/day" and metadata["loss_unit"] == "EUR"
    assert metadata["provenance"]["source_kind"] == "table"
    # Written and read back by the caller, the table rebuilds the same sample.
    path = tmp_path / "sample.parquet"
    pq.write_table(table, path)
    read = pq.read_table(path)
    assert SAMPLE_METADATA_KEY.encode() in read.schema.metadata
    rebuilt = OutcomeSample.from_table(
        read,
        period="period",
        asset_id="asset_id",
        index="index",
        loss="loss",
        probability="probability",
        semantics="annual",
        index_unit="mm/day",
        loss_unit="EUR",
    )
    assert rebuilt.content_hash == sample.content_hash


def test_select_rescales_and_records_its_parent() -> None:
    sample = _sample()
    selected = sample.select([2001, 2005, 2003])
    assert selected.period.tolist() == [2001, 2003, 2005]
    np.testing.assert_allclose(selected.probability, [1 / 3] * 3)
    assert selected.provenance.parent == sample.content_hash
    assert selected.lineage == sample.lineage
    assert selected.provenance.selection == ({"kind": "select", "periods": 3},)
    nested = selected.select([2003])
    assert nested.provenance.parent == sample.content_hash
    assert len(nested.provenance.selection) == 2


@pytest.mark.parametrize(
    ("periods", "message"),
    [([1999], "holds no periods"), ([2001, 2001], "repeat"), ([], "non-empty")],
)
def test_select_refuses_unknown_or_repeated_periods(
    periods: list[int], message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        _sample().select(periods)


# -- splits ------------------------------------------------------------------


def test_holdout_keeps_the_latest_periods_out() -> None:
    sample = _sample()
    design, held_out = holdout(sample, train_fraction=0.7)
    assert design.period.tolist() == list(range(2000, 2007))
    assert held_out.period.tolist() == [2007, 2008, 2009]
    for block in (design, held_out):
        assert abs(block.probability.sum() - 1.0) < 1e-12
        assert block.provenance.parent == sample.content_hash
        assert block.lineage == sample.lineage
    assert design.provenance.selection[-1] == {
        "kind": "holdout",
        "train_fraction": 0.7,
        "fold": 0,
        "block": "train",
    }
    assert held_out.provenance.selection[-1]["block"] == "test"


@pytest.mark.parametrize(("n", "fraction", "n_design"), [(10, 0.7, 7), (100, 0.29, 29)])
def test_holdout_rounds_as_the_reference_splits(
    n: int, fraction: float, n_design: int
) -> None:
    # floor(fraction * n + 1e-9): 0.29 * 100 is 28.999999999999996 in binary64.
    design, held_out = holdout(_sample(n=n), fraction)
    assert (len(design), len(held_out)) == (n_design, n - n_design)


@pytest.mark.parametrize("fraction", [0.0, 1.0, 0.05, -0.5, float("nan"), True])
def test_holdout_needs_two_non_empty_blocks(fraction: Any) -> None:
    with pytest.raises(SampleError, match="train_fraction"):
        holdout(_sample(), fraction)


def test_year_block_kfold_partitions_the_periods() -> None:
    sample = _sample()
    folds = year_block_kfold(sample, 3)
    assert [held.period.tolist() for _, held in folds] == [
        [2000, 2001, 2002, 2003],
        [2004, 2005, 2006],
        [2007, 2008, 2009],
    ]
    for fold, (design, held) in enumerate(folds):
        assert sorted(design.period.tolist() + held.period.tolist()) == list(
            range(2000, 2010)
        )
        assert held.provenance.selection[-1] == {
            "kind": "kfold",
            "k": 3,
            "fold": fold,
            "block": "test",
        }
        _check_out_of_sample(design, held)


@pytest.mark.parametrize("k", [1, 11, True, 2.0])
def test_year_block_kfold_needs_a_valid_k(k: Any) -> None:
    with pytest.raises(SampleError, match="k must be"):
        year_block_kfold(_sample(), k)


def test_out_of_sample_checks_detect_shared_periods() -> None:
    sample = _sample()
    design, held_out = holdout(sample)
    _check_out_of_sample(design, held_out)
    with pytest.raises(LeakageError, match="shares 7 periods"):
        _check_out_of_sample(design, sample)
    with pytest.raises(LeakageError):
        _check_out_of_sample(sample, sample)
    # Another source over the same years shares no rows.
    other = _sample(loss=np.ones(10))
    assert other.lineage != sample.lineage
    _check_out_of_sample(sample, other)
    with pytest.raises(TypeError):
        _check_out_of_sample(sample, "sample")  # type: ignore[arg-type]


# -- the provenance ----------------------------------------------------------


def test_provenance_is_read_only_and_plain_json() -> None:
    provenance = _provenance(
        details={"counts": [1, 2], "nested": {"a": None}},
        notes=["one"],
        licences=({"licence": "CC-BY-4.0"},),
    )
    with pytest.raises(TypeError):
        provenance.details["counts"] = 3  # type: ignore[index]
    assert provenance.details["counts"] == (1, 2)
    document = provenance.to_document()
    assert document["details"] == {"counts": [1, 2], "nested": {"a": None}}
    assert document["notes"] == ["one"]
    json.dumps(document)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"source_kind": "events"}, "source_kind"),
        ({"semantics_source": "guess"}, "semantics_source"),
        ({"details": {"x": float("nan")}}, "plain JSON"),
        ({"details": {"x": object()}}, "plain JSON"),
        ({"sources": {"role": "table"}}, "sequence of mappings"),
        ({"rng": ({"step": "hazard"},)}, "rng record"),
        ({"notes": "one"}, "notes"),
        ({"lineage": 3}, "lineage"),
    ],
)
def test_invalid_provenance_raises(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(SampleError, match=message):
        _provenance(**fields)


def test_a_sample_without_draws_has_no_generator() -> None:
    with pytest.raises(SampleError, match="without a random generator"):
        _provenance().generator()


# -- user tables -------------------------------------------------------------


def test_from_table_sorts_by_period_and_records_the_source() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample = OutcomeSample.from_table(
            _table(),
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
            source=SourceProvenance(
                provider="me", dataset="claims", licence="CC0-1.0", attribution="me"
            ),
        )
    assert sample.asset_ids == ("a1",)
    assert sample.period.tolist() == [2000, 2001, 2002, 2003]
    assert sample.index[:, 0].tolist() == [60.0, 20.0, 45.0, 80.0]
    assert sample.loss[:, 0].tolist() == [3.0, 2.0, 0.0, 5.0]
    np.testing.assert_allclose(sample.probability, 0.25)
    provenance = sample.provenance
    assert provenance.source_kind == "table"
    assert provenance.semantics_source == "caller"
    assert provenance.loss_model is None and provenance.basis_risk is None
    assert provenance.sources[0]["content_hash"] == sample.content_hash
    assert provenance.sources[0]["source"]["dataset"] == "claims"
    assert provenance.licences[0]["licence"] == "CC0-1.0"
    assert provenance.details["probability"] == "uniform"


def test_from_table_reads_optional_columns_and_record_batches() -> None:
    table = _table(
        p=[0.1, 0.2, 0.3, 0.4],
        wind=[1.0, 2.0, 3.0, 4.0],
        asset=pa.array([7, 7, 7, 7], pa.int32()),
    )
    sample = OutcomeSample.from_table(
        table.to_batches(),
        period="year",
        asset_id="asset",
        index="rain",
        loss="loss",
        probability="p",
        hazard="wind",
        semantics="annual",
        index_unit="mm/day",
        loss_unit="EUR",
        source="claims-2024",
    )
    assert sample.asset_ids == ("7",)
    assert sample.probability.tolist() == [0.4, 0.2, 0.3, 0.1]
    assert sample.hazard is not None and sample.hazard[:, 0].tolist() == [
        4.0,
        2.0,
        3.0,
        1.0,
    ]
    assert sample.provenance.details["probability"] == "column"
    assert sample.provenance.sources[0]["source"] == {"name": "claims-2024"}


def test_a_table_read_from_parquet(tmp_path: Path) -> None:
    path = tmp_path / "losses.parquet"
    pq.write_table(_table(), path)
    sample = OutcomeSample.from_table(
        pq.read_table(path),
        period="year",
        asset_id="asset",
        index="rain",
        loss="loss",
        semantics="annual",
        index_unit="mm/day",
        loss_unit="EUR",
    )
    assert len(sample) == 4
    with pytest.raises(TypeError, match="no I/O"):
        OutcomeSample.from_table(
            path,
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
        )


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (_table().drop(["loss"]), "no column 'loss'"),
        (_table(loss=[1.0, None, 2.0, 3.0]), "null"),
        (_table(year=[2000.0, 2001.0, 2002.0, 2003.0]), "integers"),
        (_table(year=[2000, 2001, 2001, 2003]), "more than once"),
        (_table(asset=["a1", "a2", "a1", "a1"]), "exactly one asset"),
        (_table(loss=[1.0, -2.0, 2.0, 3.0]), "non-negative"),
        (_table(rain=[1.0, float("inf"), 2.0, 3.0]), "finite"),
        (_table(rain=["x", "y", "z", "w"]), "numeric"),
    ],
)
def test_invalid_tables_raise(table: Any, message: str) -> None:
    with pytest.raises(SampleError, match=message):
        OutcomeSample.from_table(
            table,
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
        )


def test_a_table_needs_annual_semantics() -> None:
    with pytest.raises(SampleError, match="reserved"):
        OutcomeSample.from_table(
            _table(),
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            semantics="event",  # type: ignore[arg-type]
            index_unit="mm/day",
            loss_unit="EUR",
        )


def test_a_table_whose_loss_follows_its_index_warns() -> None:
    table = _table(loss=[8.0, 2.0, 4.5, 6.0])
    with pytest.warns(ComonotoneIndexWarning, match="comonotone"):
        sample = OutcomeSample.from_table(
            table,
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
        )
    assert any("comonotone" in note for note in sample.provenance.notes)


@pytest.mark.filterwarnings("ignore::crc_sdk.parametric.ComonotoneIndexWarning")
def test_tables_from_one_named_source_share_a_lineage() -> None:
    def build(table: Any, source: str | None) -> OutcomeSample:
        return OutcomeSample.from_table(
            table,
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
            source=source,
        )

    early = _table().filter(pa.compute.less(pa.compute.field("year"), 2002))
    late = _table().filter(pa.compute.greater_equal(pa.compute.field("year"), 2001))
    named_early, named_late = build(early, "claims"), build(late, "claims")
    assert named_early.lineage == named_late.lineage
    with pytest.raises(LeakageError):
        _check_out_of_sample(named_early, named_late)
    # Without a name, a table's identity is its content.
    assert build(early, None).lineage != build(late, None).lineage


def test_tables_in_other_arrow_forms() -> None:
    def build(table: Any) -> OutcomeSample:
        return OutcomeSample.from_table(
            table,
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
            source={"name": "claims", "licence": "CC0-1.0"},
        )

    table = _table(loss=[5.0, 2.0, 7.0, 3.0])
    expected = build(table).content_hash
    assert build(table.to_batches()[0]).content_hash == expected
    assert build(pa.RecordBatchReader.from_batches(table.schema, table.to_batches()))
    assert build(table.to_pydict()).content_hash == expected
    assert build(table).provenance.licences[0]["licence"] == "CC0-1.0"
    invalid_tables: tuple[Any, ...] = (42, [1, 2], iter([]))
    for invalid in invalid_tables:
        with pytest.raises(TypeError):
            build(invalid)
    with pytest.raises(TypeError, match="source"):
        OutcomeSample.from_table(
            table,
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
            source=3,
        )
