"""Quantile evidence located by H3 cells, and one list of knots per curve."""

from __future__ import annotations

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import QuantileCurve, Tail, Temperature, TemperatureUnit
from h3.api import basic_int as h3  # type: ignore[import-untyped]

from crc_sdk import Dataset, Probability, QuantileColumns, SupportKind, Variable

VARIABLE = Variable(
    "lab:annual-maximum-temperature",
    measurement=Temperature(unit=TemperatureUnit.CELSIUS),
)
ANNUAL = Probability.annual(tail=Tail.UPPER)
LEVELS = (0.9, 0.1, 0.5)
CELLS = [h3.latlng_to_cell(lat, 10.0, 5) for lat in (40.0, 45.0, 50.0)]


def wide(values: list[list[float]] | None = None) -> pa.Table:
    return pa.table(
        {
            "support_id": [f"h3:{cell:x}" for cell in CELLS],
            "hex_id": pa.array(CELLS, pa.uint64()),
            "knots": pa.array(
                values or [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0], [9.0, 7.0, 8.0]],
                pa.list_(pa.float32()),
            ),
        }
    )


def test_one_list_per_curve_matches_one_row_per_knot() -> None:
    lists = Dataset.from_quantiles(
        wide(),
        variable=VARIABLE,
        probability=ANNUAL,
        columns=QuantileColumns(cell="hex_id", value="knots", probabilities=LEVELS),
    ).read()
    knots = Dataset.from_quantiles(
        {
            "support_id": np.repeat([f"h3:{cell:x}" for cell in CELLS], 3),
            "hex_id": np.repeat(np.array(CELLS, dtype=np.uint64), 3),
            "probability": list(LEVELS) * 3,
            "value": [3.0, 1.0, 2.0, 6.0, 4.0, 5.0, 9.0, 7.0, 8.0],
        },
        variable=VARIABLE,
        probability=ANNUAL,
        columns=QuantileColumns(cell="hex_id"),
    ).read()

    assert lists.table.equals(knots.table)
    assert lists.metadata.spatial.supports == (SupportKind.CELL,)
    assert lists.metadata.spatial.index_resolution == 5
    assert lists.table["cell_id"].to_pylist() == CELLS
    assert lists.table["geometry"].null_count == 3
    first = lists.table["curve"][0].as_py()["quantile_curve"]
    assert first == {"probabilities": [0.1, 0.5, 0.9], "values": [1.0, 2.0, 3.0]}
    assert QuantileCurve(first["probabilities"], first["values"]).quantile(0.5) == 2.0


def test_cell_located_curves_need_one_valid_resolution() -> None:
    columns = QuantileColumns(cell="hex_id", value="knots", probabilities=LEVELS)
    mixed = wide().set_column(
        1,
        "hex_id",
        pa.array([CELLS[0], CELLS[1], h3.cell_to_parent(CELLS[2], 4)], pa.uint64()),
    )
    with pytest.raises(ValueError, match="one H3 resolution"):
        Dataset.from_quantiles(
            mixed, variable=VARIABLE, probability=ANNUAL, columns=columns
        )
    invalid = wide().set_column(1, "hex_id", pa.array([1, 2, 3], pa.uint64()))
    with pytest.raises(ValueError, match="valid H3 cell"):
        Dataset.from_quantiles(
            invalid, variable=VARIABLE, probability=ANNUAL, columns=columns
        )


def test_knot_lists_hold_one_finite_value_per_probability() -> None:
    columns = QuantileColumns(cell="hex_id", value="knots", probabilities=LEVELS)
    for values, message in (
        ([[3.0, 1.0], [6.0, 4.0, 5.0], [9.0, 7.0, 8.0]], "one value per probability"),
        ([[3.0, 1.0, float("nan")], [6.0, 4.0, 5.0], [9.0, 7.0, 8.0]], "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            Dataset.from_quantiles(
                wide(values), variable=VARIABLE, probability=ANNUAL, columns=columns
            )
    with pytest.raises(ValueError, match="must not be empty"):
        QuantileColumns(value="knots", probabilities=())
    with pytest.raises(ValueError, match="distinct"):
        QuantileColumns(value="knots", probabilities=(0.1, 0.5, 0.1))
    with pytest.raises(TypeError, match="sequence of numbers"):
        QuantileColumns(value="knots", probabilities=(True, False))
    # NumPy levels are accepted at runtime and stored as a tuple of floats.
    levels = np.array([0.9, 0.1, 0.5], dtype=np.float32)
    columns = QuantileColumns(value="knots", probabilities=levels)  # type: ignore[arg-type]
    assert columns.probabilities == tuple(float(level) for level in levels)
    scalars = QuantileColumns(
        value="knots",
        probabilities=(np.float32(0.5), 0.9),  # type: ignore[arg-type]
    )
    assert scalars.probabilities == (0.5, 0.9)


def test_only_the_columns_a_mapping_reads_must_exist_and_differ() -> None:
    # Longitude, latitude and probability are not read here, so their default
    # names neither need to exist nor to differ from the columns that are.
    columns = QuantileColumns(cell="hex_id", value="probability", probabilities=LEVELS)
    table = wide().rename_columns(["support_id", "hex_id", "probability"])
    result = Dataset.from_quantiles(
        table, variable=VARIABLE, probability=ANNUAL, columns=columns
    ).read()
    assert result.table.num_rows == 3
    with pytest.raises(ValueError, match="distinct"):
        QuantileColumns(cell="support_id")


def test_any_unordered_curve_is_rejected_with_the_framework_message() -> None:
    columns = QuantileColumns(cell="hex_id", value="knots", probabilities=LEVELS)
    # Later curves may lie below earlier ones; only knots within a curve count.
    Dataset.from_quantiles(
        wide([[9.0, 7.0, 8.0], [3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]),
        variable=VARIABLE,
        probability=ANNUAL,
        columns=columns,
    )
    with pytest.raises(ValueError, match="quantile evidence requires"):
        Dataset.from_quantiles(
            wide([[3.0, 1.0, 2.0], [6.0, 4.0, 7.0], [9.0, 7.0, 8.0]]),
            variable=VARIABLE,
            probability=ANNUAL,
            columns=columns,
        )


def test_vectorized_knot_checks_agree_with_quantile_curve() -> None:
    from crc_sdk._table_input import _check_curves

    rng = np.random.default_rng(7)
    for _ in range(400):
        counts = rng.integers(1, 5, size=rng.integers(1, 5))
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        q = np.concatenate([np.sort(rng.choice(5, n)) / 4.0 for n in counts])
        values = np.concatenate([np.sort(rng.integers(0, 3, n)) for n in counts])
        values = values.astype(np.float64)
        if rng.random() < 0.3:
            values[rng.integers(len(values))] += rng.choice([-1.0, 1.0])
        if rng.random() < 0.1:
            q[rng.integers(len(q))] = rng.choice([-0.25, 1.25])
        expected = None
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            try:
                QuantileCurve(q[start:end].tolist(), values[start:end].tolist())
            except ValueError as error:
                expected = str(error)
                break
        if expected is None:
            _check_curves(q, values, offsets)
        else:
            with pytest.raises(ValueError) as raised:
                _check_curves(q, values, offsets)
            assert str(raised.value) == expected
