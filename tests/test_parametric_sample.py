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
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.parametric import (
    ComonotoneIndexWarning,
    ImpactNoise,
    LeakageError,
    LossModel,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    holdout,
    sample_curve,
    year_block_kfold,
)
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.sample import (
    SAMPLE_METADATA_KEY,
    SAMPLE_SCHEMA_VERSION,
    SAMPLE_SEMANTICS_VERSION,
    _check_out_of_sample,
    _hash_document,
)
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

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


def _list_based_table(sample: OutcomeSample) -> Any:
    """The long table built from Python lists, as `to_table` first did."""
    document = _hash_document(
        asset_ids=sample.asset_ids,
        period=sample.period,
        probability=sample.probability,
        index=sample.index,
        loss=sample.loss,
        hazard=sample.hazard,
        index_unit=sample.index_unit,
        loss_unit=sample.loss_unit,
        semantics=sample.semantics,
        scenario=sample.scenario,
    )
    rows = len(document["period"])
    metadata = {
        "asset_ids": list(sample.asset_ids),
        "content_hash": sample.content_hash,
        "index_unit": sample.index_unit,
        "loss_unit": sample.loss_unit,
        "provenance": sample.provenance.to_document(),
        "scenario": sample.scenario,
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "semantics": sample.semantics,
        "semantics_version": SAMPLE_SEMANTICS_VERSION,
    }
    schema = pa.schema(
        [
            ("scenario", pa.string()),
            ("period", pa.int64()),
            ("asset_id", pa.string()),
            ("probability", pa.float64()),
            ("index", pa.float64()),
            ("loss", pa.float64()),
            ("hazard", pa.float64()),
        ],
        metadata={
            SAMPLE_METADATA_KEY: json.dumps(
                metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        },
    )
    hazard = (
        pa.nulls(rows, pa.float64())
        if document["hazard"] is None
        else pa.array(document["hazard"], pa.float64())
    )
    return pa.Table.from_arrays(
        [
            pa.array([sample.scenario] * rows, pa.string()),
            pa.array(document["period"], pa.int64()),
            pa.array(document["asset_id"], pa.string()),
            pa.array(document["probability"], pa.float64()),
            pa.array(document["index"], pa.float64()),
            pa.array(document["loss"], pa.float64()),
            hazard,
        ],
        schema=schema,
    )


@pytest.mark.parametrize("with_hazard", [False, True])
def test_to_table_equals_the_list_based_table(with_hazard: bool) -> None:
    rng = np.random.default_rng(7)
    n = 50
    sample = _sample(
        n=n,
        probability=np.full(n, 1.0 / n),
        index=rng.gamma(2.0, 20.0, n),
        loss=np.where(np.arange(n) % 4 == 0, 0.0, rng.gamma(2.0, 5.0, n)),
        hazard=rng.gamma(2.0, 25.0, n) if with_hazard else None,
        scenario="ssp245",
    )
    design, held_out = holdout(sample, train_fraction=0.7)
    for block in (sample, design, held_out, sample.select([2003, 2001, 2040])):
        table = block.to_table()
        expected = _list_based_table(block)
        assert table.schema.equals(expected.schema, check_metadata=True)
        assert table.schema.metadata == expected.schema.metadata
        assert table.equals(expected, check_metadata=True)
        for ours, theirs in zip(table.columns, expected.columns, strict=True):
            assert ours.type == theirs.type
            assert ours.to_pylist() == theirs.to_pylist()
        rebuilt = OutcomeSample.from_written_table(table)
        assert rebuilt.content_hash == block.content_hash
        np.testing.assert_array_equal(rebuilt.loss, block.loss)
        assert (rebuilt.hazard is None) is (not with_hazard)


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


def _curve_sample(n: int = 200, seed: int = 42) -> OutcomeSample:
    """A Monte Carlo sample, whose provenance has every kind of field."""
    metadata = HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset="rx1day",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 200.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sample_curve(
            FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0),
            metadata=metadata,
            n_draws=n,
            rng=np.random.default_rng(seed),
            loss_model=LossModel(
                PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
                value=1e6,
                value_unit="EUR",
            ),
            basis_risk=ImpactNoise(0.3, upper=1.0),
            asset_id="warehouse-a",
            details={"sha256": "0" * 64},
        )


def _assert_same_sample(read: OutcomeSample, sample: OutcomeSample) -> None:
    assert read.content_hash == sample.content_hash
    assert read.asset_ids == sample.asset_ids
    for name in ("period", "probability", "index", "loss"):
        np.testing.assert_array_equal(getattr(read, name), getattr(sample, name))
        assert getattr(read, name).dtype == getattr(sample, name).dtype
    if sample.hazard is None:
        assert read.hazard is None
    else:
        assert read.hazard is not None
        np.testing.assert_array_equal(read.hazard, sample.hazard)
    for name in ("index_unit", "loss_unit", "semantics", "scenario", "lineage"):
        assert getattr(read, name) == getattr(sample, name)
    assert read.provenance.to_document() == sample.provenance.to_document()


def test_a_written_sample_reads_back_with_its_provenance(tmp_path: Path) -> None:
    sample = _curve_sample()
    assert sample.hazard is None and sample.provenance.rng
    table = sample.to_table()
    _assert_same_sample(OutcomeSample.from_written_table(table), sample)
    _assert_same_sample(OutcomeSample.from_written_table(table.to_batches()[0]), sample)
    path = tmp_path / "sample.parquet"
    pq.write_table(table, path)
    read = OutcomeSample.from_written_table(pq.read_table(path))
    _assert_same_sample(read, sample)
    # The generator state survives, so the sample can be rebuilt from the file.
    rebuilt = _curve_sample(seed=0)
    assert rebuilt.content_hash != sample.content_hash
    state = read.provenance.generator().bit_generator.state
    assert state == np.random.default_rng(42).bit_generator.state
    with pytest.raises(TypeError, match="no I/O"):
        OutcomeSample.from_written_table(path)


def test_written_splits_keep_their_lineage(tmp_path: Path) -> None:
    sample = _sample(
        n=10, hazard=np.linspace(1.0, 2.0, 10), scenario="ssp585", index_unit="m"
    )
    design, held_out = holdout(sample)
    path = tmp_path / "held_out.parquet"
    pq.write_table(held_out.to_table(), path)
    read = OutcomeSample.from_written_table(pq.read_table(path))
    _assert_same_sample(read, held_out)
    assert read.provenance.parent == sample.content_hash
    _check_out_of_sample(design, read)
    with pytest.raises(LeakageError):
        _check_out_of_sample(sample, read)


def _rewritten(table: Any, **changes: Any) -> Any:
    """The table with its sample metadata changed: None deletes a key."""
    document = json.loads(table.schema.metadata[SAMPLE_METADATA_KEY.encode()])
    for key, value in changes.items():
        if value is None:
            del document[key]
        else:
            document[key] = value
    return table.replace_schema_metadata(
        {SAMPLE_METADATA_KEY: json.dumps(document, sort_keys=True)}
    )


def _with_column(table: Any, name: str, values: Any) -> Any:
    position = table.schema.get_field_index(name)
    return table.set_column(position, table.schema.field(position), values)


def _written_cases() -> list[tuple[str, Any, str]]:
    table = _sample(n=4, hazard=np.array([1.0, 2.0, 3.0, 4.0])).to_table()
    provenance = json.loads(table.schema.metadata[SAMPLE_METADATA_KEY.encode()])[
        "provenance"
    ]
    return [
        ("no metadata", table.replace_schema_metadata(None), "no 'crc.parametric"),
        (
            "not JSON",
            table.replace_schema_metadata({SAMPLE_METADATA_KEY: "{"}),
            "not valid JSON",
        ),
        (
            "not an object",
            table.replace_schema_metadata({SAMPLE_METADATA_KEY: "[]"}),
            "must be an object",
        ),
        ("a missing key", _rewritten(table, loss_unit=None), r"lacks \['loss_unit'\]"),
        ("an unknown key", _rewritten(table, extra=1), r"unknown keys \['extra'\]"),
        ("another schema", _rewritten(table, schema_version="2.0"), "'2.0'"),
        (
            "other semantics",
            _rewritten(table, semantics_version="crc-parametric-sample/2"),
            "semantics_version",
        ),
        ("no assets", _rewritten(table, asset_ids=[]), "non-empty list"),
        ("three assets", _rewritten(table, asset_ids=["a", "b", "c"]), "multiple"),
        ("a bad scenario", _rewritten(table, scenario=["x"]), "scenario"),
        ("another unit", _rewritten(table, index_unit="mm"), "content hash"),
        (
            "another scenario",
            _rewritten(table, scenario="ssp585"),
            r"scenario column holds \['baseline'\]",
        ),
        (
            "unknown provenance",
            _rewritten(table, provenance={**provenance, "extra": 1}),
            "unknown fields",
        ),
        (
            "partial provenance",
            _rewritten(
                table,
                provenance={k: v for k, v in provenance.items() if k != "sources"},
            ),
            "incomplete",
        ),
        ("no provenance", _rewritten(table, provenance=["x"]), "must be an object"),
        ("an edited loss", _with_column(table, "loss", pa.array([9.0] * 4)), "content"),
        ("reordered rows", table.take([3, 2, 1, 0]), "strictly increasing"),
        ("filtered rows", table.slice(1), "sum to 1"),
        ("a missing column", table.drop(["index"]), "no column 'index'"),
        (
            "another asset",
            _with_column(table, "asset_id", pa.array(["a", "a", "b", "a"])),
            "asset_id column",
        ),
        (
            "a partial hazard",
            _with_column(table, "hazard", pa.array([1.0, None, 3.0, 4.0])),
            "null in 1 of 4",
        ),
        (
            "two assets",
            _rewritten(
                _with_column(table, "asset_id", pa.array(["a", "b", "a", "b"])),
                asset_ids=["b", "a"],
            ),
            "differs between assets",
        ),
    ]


@pytest.mark.parametrize(
    ("table", "message"),
    [case[1:] for case in _written_cases()],
    ids=[case[0] for case in _written_cases()],
)
def test_written_tables_that_were_changed_are_rejected(
    table: Any, message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        OutcomeSample.from_written_table(table)


def test_a_written_table_of_two_assets_is_refused_until_phase_two() -> None:
    table = _sample(n=4).to_table()
    two = pa.concat_tables([table, table]).sort_by([("period", "ascending")])
    two = _with_column(two, "asset_id", pa.array(["a", "b"] * 4))
    two = _with_column(two, "probability", pa.array([0.25] * 8))
    with pytest.raises(SampleError, match="exactly one asset"):
        OutcomeSample.from_written_table(_rewritten(two, asset_ids=["b", "a"]))


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
