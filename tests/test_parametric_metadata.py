"""EvaluationMetadata: JSON round trip, merging and the table check."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    BinaryPayout,
    Cover,
    EvaluationMetadata,
    IndexDefinition,
    MetricOptions,
    OutcomeSample,
    SampleProvenance,
    compare,
    evaluate,
)
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY

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
    with pytest.raises(ValueError):
        EvaluationMetadata.from_json(text)


def test_non_finite_values_and_repeated_designs_raise() -> None:
    document = _metadata(evaluate(COVER, _sample(), split="in_sample")).model_dump()
    with pytest.raises(ValueError):
        EvaluationMetadata.model_validate({**document, "sources": ({"x": np.nan},)})
    with pytest.raises(ValueError):
        EvaluationMetadata.model_validate({**document, "designs": ({"route": "lp"},)})
    record = {"design_id": "sha256:" + "1" * 64}
    with pytest.raises(ValueError):
        EvaluationMetadata.model_validate({**document, "designs": (record, record)})


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
    with pytest.raises(ValueError, match="metric options"):
        EvaluationMetadata.merge(base, other_options)
    usd_cover = Cover(INDEX, BinaryPayout(100.0, 400.0), payout_unit="USD")
    usd = _metadata(evaluate(usd_cover, _sample(loss_unit="USD"), split="in_sample"))
    with pytest.raises(ValueError, match="payout unit"):
        EvaluationMetadata.merge(base, usd)
    with pytest.raises(ValueError, match="numpy_version"):
        EvaluationMetadata.merge(base, base.model_copy(update={"numpy_version": "0"}))
    with pytest.raises(ValueError, match="asset records"):
        EvaluationMetadata.merge(
            base, base.model_copy(update={"assets": {"count": 1, "id_column": "id"}})
        )
    record = {"design_id": "sha256:" + "1" * 64, "route": "lp"}
    one = base.model_copy(update={"designs": (record,)})
    two = base.model_copy(update={"designs": ({**record, "route": "qp"},)})
    with pytest.raises(ValueError, match="two different records"):
        EvaluationMetadata.merge(one, two)
    with pytest.raises(ValueError):
        EvaluationMetadata.merge()
    with pytest.raises(TypeError):
        EvaluationMetadata.merge(base, base.model_dump())  # type: ignore[arg-type]


def test_check_table_finds_undescribed_rows() -> None:
    table = compare([COVER], _sample(), split="in_sample")
    metadata = _metadata(table)
    metadata.check_table(table)
    other = _metadata(compare([COVER], _sample(start=2100), split="in_sample"))
    with pytest.raises(ValueError, match="sample"):
        other.check_table(table)
    with pytest.raises(ValueError, match="cover"):
        metadata.model_copy(update={"covers": ()}).check_table(table)
    with pytest.raises(ValueError, match="lacks the columns"):
        metadata.check_table(table.drop_columns(["fold"]))
    with pytest.raises(ValueError, match="repeats a row key"):
        metadata.check_table(pa.concat_tables([table, table]))
    designed = table.set_column(
        table.column_names.index("design_id"),
        "design_id",
        pa.array(["sha256:" + "3" * 64]),
    )
    with pytest.raises(ValueError, match="design"):
        metadata.check_table(designed)
    evaluations = tuple({**entry, "fold": 1} for entry in metadata.setup.evaluations)
    setup = metadata.setup.model_copy(update={"evaluations": evaluations})
    with pytest.raises(ValueError, match="no evaluation"):
        metadata.model_copy(update={"setup": setup}).check_table(table)


def test_from_table_needs_the_key() -> None:
    with pytest.raises(ValueError, match=EVALUATION_METADATA_KEY):
        EvaluationMetadata.from_table(pa.table({"x": [1]}))
