from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk.connectors import HurdleFitPolicy, OSClimateIngestPolicy
from crc_sdk.connectors.parquet import hazard_arrow_schema, read_hazard_dataset
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import OSClimateSelectionSpec, curve_quantiles_at, tile_bounds
from crc_sdk.workflows.tiling import run_tiled_canonicalization


def test_tile_bounds_splits_exact_grid() -> None:
    tiles = tile_bounds((0.0, 0.0, 4.0, 2.0), 2.0)
    assert tiles == (
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 0.0, 4.0, 2.0),
    )


def test_tile_bounds_keeps_a_remainder_tile() -> None:
    tiles = tile_bounds((0.0, 0.0, 5.0, 1.0), 2.0)
    assert tiles == (
        (0.0, 0.0, 2.0, 1.0),
        (2.0, 0.0, 4.0, 1.0),
        (4.0, 0.0, 5.0, 1.0),
    )


def test_tile_bounds_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        tile_bounds((0.0, 0.0, 1.0, 1.0), 0.0)
    with pytest.raises(ValueError, match="min_lon"):
        tile_bounds((1.0, 0.0, 0.0, 1.0), 1.0)


def _metadata() -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=5,
        value_unit="metres",
        value_semantics="flood depth",
        producer="tests",
        source=SourceProvenance(provider="fixture", dataset="flood"),
        creation_version="1",
    )


def _row(cell_index: int, source_id: str) -> dict[str, Any]:
    return {
        "cell_index": cell_index,
        "source_id": source_id,
        "source_geometry": None,
        "hazard_name": "flood",
        "horizon": 1980,
        "pathway": "historical",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 1.0,
        "curve_scale": 2.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


class _FakeSelection:
    pass


class _FakeResource:
    def resolve(self, *, scenario: str, year: int) -> _FakeSelection:
        return _FakeSelection()


class _FakeStream:
    def __init__(self, table: pa.Table, metadata: HazardDatasetMetadata) -> None:
        self._table = table
        self.metadata = metadata

    def read_all(self) -> pa.Table:
        return self._table


class _FakeProvider:
    """Tile (0,0,1,1) yields rows; every other tile yields none."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def select(self, **kwargs: Any) -> _FakeResource:
        return _FakeResource()

    def canonicalize(
        self, selection: _FakeSelection, policy: OSClimateIngestPolicy, *, bounds: Any
    ) -> _FakeStream:
        metadata = _metadata()
        if bounds == (0.0, 0.0, 1.0, 1.0):
            table = pa.Table.from_pylist(
                [_row(1, "a"), _row(2, "b")], schema=hazard_arrow_schema()
            )
        else:
            table = pa.Table.from_pylist([], schema=hazard_arrow_schema())
        return _FakeStream(table, metadata)


def test_run_tiled_canonicalization_skips_empty_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.OSClimateProvider", _FakeProvider
    )
    spec = OSClimateSelectionSpec(
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        model_gcm="historical",
        scenario="historical",
        year=1980,
    )
    policy = OSClimateIngestPolicy(
        h3_resolution=5,
        family="gumbel_r",
        producer="tests",
        creation_version="1",
        hurdle=HurdleFitPolicy(atom_probability=0.5),
    )
    shards = run_tiled_canonicalization(
        spec,
        policy,
        bounds=(0.0, 0.0, 2.0, 1.0),
        output_dir=tmp_path,
        tile_degrees=1.0,
        max_workers=1,
    )
    assert len(shards) == 1
    table = read_hazard_dataset(shards[0])
    assert table.num_rows == 2
    assert set(table["cell_index"].to_pylist()) == {1, 2}


def test_curve_quantiles_at_reconstructs_expected_values() -> None:
    table = pa.Table.from_pylist(
        [_row(1, "a"), _row(2, "b")], schema=hazard_arrow_schema()
    )
    values = curve_quantiles_at(table, 0.9, max_workers=1)
    assert len(values) == 2
    assert all(isinstance(value, float) for value in values)
    assert values[0] == values[1]  # identical curve params


def test_curve_quantiles_at_empty_table_returns_empty_list() -> None:
    table = pa.Table.from_pylist([], schema=hazard_arrow_schema())
    assert curve_quantiles_at(table, 0.9) == []
