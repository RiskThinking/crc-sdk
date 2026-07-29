from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from shapely.geometry import box  # type: ignore[import-untyped]

from crc_sdk.connectors import HurdleFitPolicy, OSClimateIngestPolicy
from crc_sdk.connectors.parquet import (
    hazard_arrow_schema,
    read_hazard_dataset,
    write_hazard_dataset,
)
from crc_sdk.types import CurveParameters, HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import (
    OSClimateSelectionSpec,
    curve_quantiles_at,
    stream_curve_quantiles_to_parquet,
    tile_bounds,
)
from crc_sdk.workflows.tiling import (
    _canonicalize_tile,
    _tile_owns_point,
    run_tiled_canonicalization,
)


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


def test_tile_owns_point_keeps_pixels_spilling_past_the_aoi_edge() -> None:
    # A pixel merely overlapping the AOI (not centered inside it) can have a
    # centroid outside the AOI entirely; the one tile that could have fetched
    # it must keep it rather than drop it looking for a neighbor to own it.
    aoi = (0.0, 0.0, 4.0, 2.0)
    tile_a = (0.0, 0.0, 2.0, 2.0)  # westmost/southmost: shares the AOI's true edge
    assert _tile_owns_point(-0.2, 1.0, tile_a, aoi) is True
    assert _tile_owns_point(1.0, -0.2, tile_a, aoi) is True


def test_tile_owns_point_single_tile_keeps_everything() -> None:
    aoi = (0.0, 0.0, 4.0, 2.0)
    # A single tile spans the whole AOI, so every side is a true outer edge
    # with no neighbor to hand anything off to.
    assert _tile_owns_point(-1.0, -1.0, aoi, aoi) is True
    assert _tile_owns_point(5.0, 3.0, aoi, aoi) is True


def test_tile_owns_point_middle_tile_stays_bounded_on_both_sides() -> None:
    aoi = (0.0, 0.0, 6.0, 1.0)
    tile_middle = (2.0, 0.0, 4.0, 1.0)
    # Both edges are internal (shared with a real neighbor), so the middle
    # tile must not fall back to the AOI-edge escape hatch on either side.
    assert _tile_owns_point(2.0, 0.5, tile_middle, aoi) is True
    assert _tile_owns_point(1.9, 0.5, tile_middle, aoi) is False
    assert _tile_owns_point(4.0, 0.5, tile_middle, aoi) is False
    assert _tile_owns_point(3.9, 0.5, tile_middle, aoi) is True


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


def test_canonicalize_tile_forwards_validate_max_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_canonicalize_tile must pass validate_max_workers straight through to
    write_hazard_dataset's max_workers — that's the only thing preventing a
    nested ProcessPoolExecutor when this runs inside the outer tile pool."""
    monkeypatch.setattr("crc_sdk.workflows.tiling.OSClimateProvider", _FakeProvider)
    captured: list[Any] = []

    def _spy_write_hazard_dataset(
        table: pa.Table,
        output_path: Path,
        metadata: HazardDatasetMetadata,
        *,
        max_workers: int | None = None,
    ) -> str | Path:
        captured.append(max_workers)
        return write_hazard_dataset(
            table, output_path, metadata, max_workers=max_workers
        )

    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.write_hazard_dataset", _spy_write_hazard_dataset
    )

    _canonicalize_tile(
        (0.0, 0.0, 1.0, 1.0),
        tmp_path / "tile.parquet",
        spec=_spec(),
        policy=_policy(),
        aoi_bounds=(0.0, 0.0, 1.0, 1.0),
        provider_kwargs={},
        validate_max_workers=1,
    )
    assert captured == [1]

    captured.clear()
    _canonicalize_tile(
        (0.0, 0.0, 1.0, 1.0),
        tmp_path / "tile2.parquet",
        spec=_spec(),
        policy=_policy(),
        aoi_bounds=(0.0, 0.0, 1.0, 1.0),
        provider_kwargs={},
        # validate_max_workers omitted entirely -> defaults to None (auto)
    )
    assert captured == [None]


def test_run_tiled_canonicalization_parallel_path_pins_validate_max_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real (non-fake) ProcessPoolExecutor path must request
    validate_max_workers=1 for every tile — verified without actually
    spawning subprocesses, since a nested pool wouldn't reliably fail loudly
    on every platform, it would just silently fan out too many processes."""
    captured: dict[str, Any] = {}

    class _FakeExecutor:
        def __init__(
            self, max_workers: int | None = None, mp_context: Any = None
        ) -> None:
            captured["max_workers"] = max_workers
            captured["mp_context"] = mp_context

        def __enter__(self) -> _FakeExecutor:
            return self

        def __exit__(self, *exc_info: Any) -> None:
            return None

        def map(self, func: Any, *iterables: Any) -> Any:
            captured["func"] = func
            captured["iterables"] = [list(iterable) for iterable in iterables]
            return iter(())

    monkeypatch.setattr("crc_sdk.workflows.tiling.ProcessPoolExecutor", _FakeExecutor)

    run_tiled_canonicalization(
        _spec(),
        _policy(),
        bounds=(0.0, 0.0, 4.0, 1.0),
        output_dir=tmp_path,
        tile_degrees=2.0,
        max_workers=2,
    )
    # validate_max_workers=1 is bound once via functools.partial, not mapped
    # per tile — only the genuinely varying args (tile, output_path) are.
    assert isinstance(captured["func"], partial)
    assert captured["func"].func is _canonicalize_tile
    assert captured["func"].keywords["validate_max_workers"] == 1
    assert len(captured["iterables"]) == 2
    # DuckDB connections/readers are not fork-safe: every pool must force
    # "spawn" so workers never inherit a live connection via fork.
    assert captured["mp_context"].get_start_method() == "spawn"


class _EdgeSpillProvider:
    """A single tile whose only pixel overlaps the AOI but centers outside it."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def select(self, **kwargs: Any) -> _FakeResource:
        return _FakeResource()

    def canonicalize(
        self, selection: _FakeSelection, policy: OSClimateIngestPolicy, *, bounds: Any
    ) -> _FakeStream:
        # centroid at (-0.2, 0.5) is outside bounds=(0,0,1,1) on the west side,
        # exactly like a pixel _pixel_window conservatively included because
        # it merely overlaps that edge rather than sitting inside it.
        rows = [_row(1, "spills-west", centroid=(-0.2, 0.5))]
        return _FakeStream(
            pa.Table.from_pylist(rows, schema=hazard_arrow_schema()), _metadata()
        )


def test_run_tiled_canonicalization_keeps_pixels_spilling_past_aoi_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.OSClimateProvider", _EdgeSpillProvider
    )
    shards = run_tiled_canonicalization(
        OSClimateSelectionSpec(
            hazard_type="RiverineInundation",
            indicator_id="flood_depth",
            model_gcm="historical",
            scenario="historical",
            year=1980,
        ),
        OSClimateIngestPolicy(
            h3_resolution=5,
            family="gumbel_r",
            producer="tests",
            creation_version="1",
            hurdle=HurdleFitPolicy(atom_probability=0.5),
        ),
        bounds=(0.0, 0.0, 1.0, 1.0),
        output_dir=tmp_path,
        max_workers=1,
    )
    assert len(shards) == 1
    assert read_hazard_dataset(shards[0]).num_rows == 1


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


class _FakePoolExecutor:
    """Captures ProcessPoolExecutor construction args without spawning."""

    def __init__(self, max_workers: int | None = None, mp_context: Any = None) -> None:
        self.captured: dict[str, Any] = {
            "max_workers": max_workers,
            "mp_context": mp_context,
        }
        _fake_pool_calls.append(self.captured)

    def __enter__(self) -> _FakePoolExecutor:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def map(self, func: Any, *iterables: Any) -> Any:
        return (func(*args) for args in zip(*iterables))


_fake_pool_calls: list[dict[str, Any]] = []


def test_curve_quantiles_at_uses_detected_cpu_count_not_os_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constrained container's cgroup/affinity cap must bound the pool
    size here too, not just DuckDB's own thread count — os.cpu_count()
    ignores cgroup quotas entirely and can over-spawn workers."""
    _fake_pool_calls.clear()
    monkeypatch.setattr("crc_sdk.workflows.tiling.detected_cpu_count", lambda: 1)
    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.ProcessPoolExecutor", _FakePoolExecutor
    )
    table = pa.Table.from_pylist(
        [_row(index, "a") for index in range(3)], schema=hazard_arrow_schema()
    )
    # chunk_rows=1 forces the multi-chunk path even with only 3 rows, and
    # max_workers is left unset so the detected_cpu_count() stub is exercised.
    curve_quantiles_at(table, 0.9, chunk_rows=1)
    assert not _fake_pool_calls  # detected_cpu_count() stubbed to 1 -> no pool


def test_curve_quantiles_at_pool_forces_spawn_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DuckDB connections are not fork-safe; every pool must force spawn."""
    _fake_pool_calls.clear()
    monkeypatch.setattr("crc_sdk.workflows.tiling.detected_cpu_count", lambda: 4)
    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.ProcessPoolExecutor", _FakePoolExecutor
    )
    table = pa.Table.from_pylist(
        [_row(index, "a") for index in range(3)], schema=hazard_arrow_schema()
    )
    curve_quantiles_at(table, 0.9, chunk_rows=1)
    assert len(_fake_pool_calls) == 1
    assert _fake_pool_calls[0]["mp_context"].get_start_method() == "spawn"


_SOURCE_SCHEMA = pa.schema(
    [
        ("cell_index", pa.int64()),
        ("province", pa.string()),
        ("curve_kind", pa.string()),
        ("curve_type", pa.string()),
        ("curve_shape", pa.float64()),
        ("curve_location", pa.float64()),
        ("curve_scale", pa.float64()),
        ("curve_atom_probability", pa.float64()),
        ("curve_atom_location", pa.float64()),
    ]
)


def _curve_row(cell_index: int, province: str, location: float) -> dict[str, Any]:
    return {
        "cell_index": cell_index,
        "province": province,
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": location,
        "curve_scale": 2.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


def _expected_depth(location: float, probability: float) -> float:
    return float(
        CurveParameters(
            curve_kind="fitted",
            curve_type="gumbel_r",
            curve_shape=None,
            curve_location=location,
            curve_scale=2.0,
            curve_atom_probability=None,
            curve_atom_location=None,
        )
        .to_distribution()
        .quantiles([probability])[0]
    )


def test_stream_curve_quantiles_to_parquet_matches_direct_computation(
    tmp_path: Path,
) -> None:
    rows = [_curve_row(index, "P", float(index)) for index in range(5)]
    con = duckdb.connect()
    con.register("source", pa.Table.from_pylist(rows, schema=_SOURCE_SCHEMA))
    output = tmp_path / "depths.parquet"

    written = stream_curve_quantiles_to_parquet(
        con,
        "SELECT * FROM source",
        0.9,
        output,
        passthrough_columns=["cell_index", "province"],
        batch_rows=2,  # forces 3 batches over 5 rows
        max_workers=1,
    )

    assert written == 5
    result = pq.read_table(output).to_pylist()
    assert {row["cell_index"] for row in result} == set(range(5))
    for row in result:
        assert row["province"] == "P"
        assert row["depth_m"] == pytest.approx(
            _expected_depth(float(row["cell_index"]), 0.9)
        )


def test_stream_curve_quantiles_to_parquet_parallel_matches_sequential(
    tmp_path: Path,
) -> None:
    rows = [_curve_row(index, "P", float(index)) for index in range(10)]

    sequential_path = tmp_path / "sequential.parquet"
    con = duckdb.connect()
    con.register("source", pa.Table.from_pylist(rows, schema=_SOURCE_SCHEMA))
    stream_curve_quantiles_to_parquet(
        con,
        "SELECT * FROM source",
        0.9,
        sequential_path,
        passthrough_columns=["cell_index", "province"],
        batch_rows=3,
        max_workers=1,
    )

    parallel_path = tmp_path / "parallel.parquet"
    con2 = duckdb.connect()
    con2.register("source", pa.Table.from_pylist(rows, schema=_SOURCE_SCHEMA))
    stream_curve_quantiles_to_parquet(
        con2,
        "SELECT * FROM source",
        0.9,
        parallel_path,
        passthrough_columns=["cell_index", "province"],
        batch_rows=3,
        max_workers=2,
        chunk_rows=2,
    )

    sequential = sorted(
        pq.read_table(sequential_path).to_pylist(), key=lambda row: row["cell_index"]
    )
    parallel = sorted(
        pq.read_table(parallel_path).to_pylist(), key=lambda row: row["cell_index"]
    )
    assert sequential == parallel


def test_stream_curve_quantiles_to_parquet_pool_uses_detected_cpu_count_and_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live DuckDB Arrow reader is open in this process when the pool is
    built — the pool must force spawn (never inherit it via fork), and its
    size must come from the cgroup-aware detector, not os.cpu_count()."""
    _fake_pool_calls.clear()
    monkeypatch.setattr("crc_sdk.workflows.tiling.detected_cpu_count", lambda: 4)
    monkeypatch.setattr(
        "crc_sdk.workflows.tiling.ProcessPoolExecutor", _FakePoolExecutor
    )
    rows = [_curve_row(index, "P", float(index)) for index in range(5)]
    con = duckdb.connect()
    con.register("source", pa.Table.from_pylist(rows, schema=_SOURCE_SCHEMA))
    output = tmp_path / "depths.parquet"

    written = stream_curve_quantiles_to_parquet(
        con,
        "SELECT * FROM source",
        0.9,
        output,
        passthrough_columns=["cell_index", "province"],
        batch_rows=2,
        # max_workers left unset -> exercises the detected_cpu_count() stub.
    )

    assert written == 5
    assert len(_fake_pool_calls) == 1
    assert _fake_pool_calls[0]["max_workers"] == 4
    assert _fake_pool_calls[0]["mp_context"].get_start_method() == "spawn"


def test_stream_curve_quantiles_to_parquet_empty_source_writes_valid_file(
    tmp_path: Path,
) -> None:
    con = duckdb.connect()
    con.register("source", pa.Table.from_pylist([], schema=_SOURCE_SCHEMA))
    output = tmp_path / "empty.parquet"

    written = stream_curve_quantiles_to_parquet(
        con,
        "SELECT * FROM source",
        0.9,
        output,
        passthrough_columns=["cell_index", "province"],
    )

    assert written == 0
    table = pq.read_table(output)
    assert table.num_rows == 0
    assert table.column_names == ["cell_index", "province", "depth_m"]
