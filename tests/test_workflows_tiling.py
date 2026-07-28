from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from shapely.geometry import box  # type: ignore[import-untyped]

from crc_sdk.connectors import HurdleFitPolicy, OSClimateIngestPolicy
from crc_sdk.connectors.parquet import hazard_arrow_schema, read_hazard_dataset
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import OSClimateSelectionSpec, curve_quantiles_at, tile_bounds
from crc_sdk.workflows.tiling import _tile_owns_point, run_tiled_canonicalization


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


def _row(
    cell_index: int, source_id: str, *, centroid: tuple[float, float] | None = None
) -> dict[str, Any]:
    geometry = None
    if centroid is not None:
        longitude, latitude = centroid
        geometry = box(
            longitude - 0.01, latitude - 0.01, longitude + 0.01, latitude + 0.01
        ).wkb
    return {
        "cell_index": cell_index,
        "source_id": source_id,
        "source_geometry": geometry,
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


def test_tile_owns_point_assigns_shared_edge_to_the_far_tile() -> None:
    aoi = (0.0, 0.0, 4.0, 2.0)
    tile_a = (0.0, 0.0, 2.0, 2.0)
    tile_b = (2.0, 0.0, 4.0, 2.0)

    # Exactly on the shared edge: only the higher tile claims it.
    assert _tile_owns_point(2.0, 1.0, tile_a, aoi) is False
    assert _tile_owns_point(2.0, 1.0, tile_b, aoi) is True

    # Interior points stay with their own tile.
    assert _tile_owns_point(1.0, 1.0, tile_a, aoi) is True
    assert _tile_owns_point(3.0, 1.0, tile_b, aoi) is True


def test_tile_owns_point_closes_the_true_aoi_edge() -> None:
    aoi = (0.0, 0.0, 4.0, 2.0)
    tile_b = (2.0, 0.0, 4.0, 2.0)

    # The AOI's own far edge must still be claimed by the last tile.
    assert _tile_owns_point(4.0, 2.0, tile_b, aoi) is True


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


def _spec() -> OSClimateSelectionSpec:
    return OSClimateSelectionSpec(
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        model_gcm="historical",
        scenario="historical",
        year=1980,
    )


def _policy() -> OSClimateIngestPolicy:
    return OSClimateIngestPolicy(
        h3_resolution=5,
        family="gumbel_r",
        producer="tests",
        creation_version="1",
        hurdle=HurdleFitPolicy(atom_probability=0.5),
    )


class _OverlappingEdgeProvider:
    """Simulates real ``_pixel_window`` behavior: the pixel straddling the

    tile_a/tile_b edge (centroid at lon=1.0) is conservatively fit and
    emitted by *both* neighboring tiles, exactly like the real ZarrRaster
    would for a pixel whose footprint overlaps a shared tile boundary.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    def select(self, **kwargs: Any) -> _FakeResource:
        return _FakeResource()

    def canonicalize(
        self, selection: _FakeSelection, policy: OSClimateIngestPolicy, *, bounds: Any
    ) -> _FakeStream:
        boundary = _row(99, "boundary", centroid=(1.0, 0.5))
        if bounds == (0.0, 0.0, 1.0, 1.0):
            rows = [_row(1, "interior-a", centroid=(0.5, 0.5)), boundary]
        elif bounds == (1.0, 0.0, 2.0, 1.0):
            rows = [boundary, _row(2, "interior-b", centroid=(1.5, 0.5))]
        else:
            rows = []
        return _FakeStream(
            pa.Table.from_pylist(rows, schema=hazard_arrow_schema()), _metadata()
        )


def test_run_tiled_canonicalization_dedupes_shared_edge_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.OSClimateProvider", _OverlappingEdgeProvider
    )
    shards = run_tiled_canonicalization(
        _spec(),
        _policy(),
        bounds=(0.0, 0.0, 2.0, 1.0),
        output_dir=tmp_path,
        tile_degrees=1.0,
        max_workers=1,
    )
    assert len(shards) == 2
    combined = pa.concat_tables([read_hazard_dataset(path) for path in shards])
    # Without dedup this would be 4 (the boundary pixel counted by both tiles).
    assert combined.num_rows == 3
    assert combined["source_id"].to_pylist().count("boundary") == 1


class _BoundsMissProvider:
    """Tile (1,0,2,1) falls entirely outside the (fake) raster's coverage."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def select(self, **kwargs: Any) -> _FakeResource:
        return _FakeResource()

    def canonicalize(
        self, selection: _FakeSelection, policy: OSClimateIngestPolicy, *, bounds: Any
    ) -> Any:
        if bounds == (1.0, 0.0, 2.0, 1.0):
            return _RaisingStream("bounds do not intersect the raster")
        rows = [_row(1, "a", centroid=(0.5, 0.5))]
        return _FakeStream(
            pa.Table.from_pylist(rows, schema=hazard_arrow_schema()), _metadata()
        )


class _RaisingStream:
    def __init__(self, message: str) -> None:
        self._message = message

    def read_all(self) -> pa.Table:
        raise ValueError(self._message)


def test_run_tiled_canonicalization_skips_tiles_outside_raster_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.OSClimateProvider", _BoundsMissProvider
    )
    shards = run_tiled_canonicalization(
        _spec(),
        _policy(),
        bounds=(0.0, 0.0, 2.0, 1.0),
        output_dir=tmp_path,
        tile_degrees=1.0,
        max_workers=1,
    )
    assert len(shards) == 1
    assert read_hazard_dataset(shards[0]).num_rows == 1


class _OtherErrorProvider:
    """A genuine fit failure should still propagate, not be swallowed."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def select(self, **kwargs: Any) -> _FakeResource:
        return _FakeResource()

    def canonicalize(
        self, selection: _FakeSelection, policy: OSClimateIngestPolicy, *, bounds: Any
    ) -> Any:
        return _RaisingStream("quantile optimizer did not converge")


def test_run_tiled_canonicalization_reraises_unrelated_value_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.OSClimateProvider", _OtherErrorProvider
    )
    with pytest.raises(ValueError, match="did not converge"):
        run_tiled_canonicalization(
            _spec(),
            _policy(),
            bounds=(0.0, 0.0, 1.0, 1.0),
            output_dir=tmp_path,
            tile_degrees=1.0,
            max_workers=1,
        )


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
