"""Tile-parallel raster canonicalization and curve evaluation for area-scale ingest.

DuckDB/Arrow already parallelize the join and aggregation stages of a hazard-to-
geography pipeline. The one stage they do not touch is per-pixel/per-cell pure
Python work: fitting a curve per raster pixel (:func:`canonicalize_os_climate`)
and reconstructing a curve per canonical row (:class:`CurveParameters`). Both
are parallel across independent pixels/rows, so this module splits them across
a process pool instead of leaving them single-threaded.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from shapely import wkb  # type: ignore[import-untyped]

from crc_sdk.connectors import (
    OSClimateIngestPolicy,
    write_hazard_dataset,
)
from crc_sdk.connectors.duckdb import Bounds, RuntimeResources, detected_cpu_count
from crc_sdk.providers.os_climate import OSClimateProvider
from crc_sdk.types import CurveParameters

# DuckDB connections/Arrow readers are not fork-safe: a live connection open
# in the parent (e.g. the reader in stream_curve_quantiles_to_parquet) would
# be inherited mid-state by fork-based workers, which DuckDB does not
# support. Every pool below forces "spawn" so workers always start from a
# clean interpreter, regardless of the platform default (Linux forks by
# default; macOS/Windows already spawn).
_MP_CONTEXT = multiprocessing.get_context("spawn")


@dataclass(frozen=True)
class OSClimateSelectionSpec:
    """Picklable description of one OS-Climate resource selection.

    A live ``OSClimateProvider``/``OSClimateSelection`` cannot cross a process
    boundary (each worker needs its own S3 filesystem handle), so tiled
    canonicalization takes this plain, picklable spec instead and resolves a
    fresh provider/selection inside each worker.
    """

    hazard_type: str
    indicator_id: str
    model_gcm: str
    scenario: str
    year: int


def tile_bounds(bounds: Bounds, tile_degrees: float) -> tuple[Bounds, ...]:
    """Split ``bounds`` into a grid of at-most-``tile_degrees``-wide tiles."""
    if tile_degrees <= 0:
        raise ValueError("tile_degrees must be positive")
    min_lon, min_lat, max_lon, max_lat = bounds
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bounds must be (min_lon, min_lat, max_lon, max_lat)")

    tiles: list[Bounds] = []
    lat = min_lat
    while lat < max_lat:
        lat_end = min(lat + tile_degrees, max_lat)
        lon = min_lon
        while lon < max_lon:
            lon_end = min(lon + tile_degrees, max_lon)
            tiles.append((lon, lat, lon_end, lat_end))
            lon = lon_end
        lat = lat_end
    return tuple(tiles)


def _tile_owns_point(
    longitude: float, latitude: float, tile: Bounds, aoi_bounds: Bounds
) -> bool:
    """Assign a source pixel to exactly one tile when a shared edge bisects it.

    ``ZarrRaster._pixel_window`` conservatively expands each tile's world
    bounds outward to the nearest whole pixels, so a pixel overlapping a
    queried boundary is fit and its centroid can legitimately land on either
    side of that boundary — not just at internal tile-to-tile edges, but at
    the AOI's own outer edges too (a pixel need only overlap the AOI, not be
    centered inside it). A side is only tested against the tile's own bound
    when a neighboring tile actually shares that edge (``tile_min > aoi_min``
    / ``tile_max < aoi_max``); the AOI's true outer edges are left unbounded
    on that side, since no other tile's query can reach there to duplicate
    it. A single tile spanning the whole AOI is therefore unbounded on every
    side and keeps everything unfiltered.
    """
    tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat = tile
    aoi_min_lon, aoi_min_lat, aoi_max_lon, aoi_max_lat = aoi_bounds
    longitude_ok = (longitude >= tile_min_lon or tile_min_lon <= aoi_min_lon) and (
        longitude < tile_max_lon or tile_max_lon >= aoi_max_lon
    )
    latitude_ok = (latitude >= tile_min_lat or tile_min_lat <= aoi_min_lat) and (
        latitude < tile_max_lat or tile_max_lat >= aoi_max_lat
    )
    return longitude_ok and latitude_ok


def _owns_row(geometry: bytes | None, tile: Bounds, aoi_bounds: Bounds) -> bool:
    if geometry is None:
        return True
    longitude, latitude = wkb.loads(geometry).centroid.coords[0]
    return _tile_owns_point(longitude, latitude, tile, aoi_bounds)


def _drop_shared_edge_duplicates(
    table: pa.Table, tile: Bounds, aoi_bounds: Bounds
) -> pa.Table:
    """Keep only rows whose source pixel centroid this tile owns."""
    owned = [
        _owns_row(geometry, tile, aoi_bounds)
        for geometry in table.column("source_geometry").to_pylist()
    ]
    return table.filter(pa.array(owned))


def _canonicalize_tile(
    tile: Bounds,
    output_path: Path,
    *,
    spec: OSClimateSelectionSpec,
    policy: OSClimateIngestPolicy,
    aoi_bounds: Bounds,
    provider_kwargs: Mapping[str, Any],
    validate_max_workers: int | None = None,
) -> Path | None:
    """Canonicalize one tile. ``tile``/``output_path`` vary per call; bind the
    rest once via :func:`functools.partial` (see :func:`run_tiled_canonicalization`)
    rather than passing them — and a bare ``None`` for ``validate_max_workers``
    — through every call site.
    """
    provider = OSClimateProvider(**provider_kwargs)
    resource = provider.select(
        hazard_type=spec.hazard_type,
        indicator_id=spec.indicator_id,
        model_gcm=spec.model_gcm,
    )
    selection = resource.resolve(scenario=spec.scenario, year=spec.year)
    stream = provider.canonicalize(selection, policy, bounds=tile)
    try:
        table = stream.read_all()
    except ValueError as error:
        # A tile can legitimately fall entirely outside the raster's own
        # coverage (e.g. an AOI wider than a regional-only hazard raster);
        # that's an empty tile to skip, not a fit failure to propagate.
        if "bounds do not intersect" in str(error):
            return None
        raise
    if table.num_rows == 0:
        return None
    table = _drop_shared_edge_duplicates(table, tile, aoi_bounds)
    if table.num_rows == 0:
        return None
    # validate_max_workers=1 when called from inside the outer tile pool:
    # otherwise validate_hazard_table's own chunk-parallel validation would
    # nest a second ProcessPoolExecutor inside each tile worker, fanning out
    # tile_workers x validation_workers processes instead of just using the
    # outer pool.
    write_hazard_dataset(
        table, output_path, stream.metadata, max_workers=validate_max_workers
    )
    return output_path


def run_tiled_canonicalization(
    spec: OSClimateSelectionSpec,
    policy: OSClimateIngestPolicy,
    bounds: Bounds,
    output_dir: str | Path,
    *,
    tile_degrees: float = 2.0,
    max_workers: int | None = None,
    provider_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Canonicalize one OS-Climate raster over an AOI as parallel tile shards.

    Splits ``bounds`` into a ``tile_degrees`` grid and canonicalizes each tile
    in its own worker process. Tiles are skipped (no shard written) rather
    than raising when they have no fittable pixels, and also when they fall
    entirely outside the raster's own coverage. Pixels straddling a shared
    tile edge are conservatively fit by both neighboring tiles but attributed
    to exactly one shard, so a plain glob read never double-counts, e.g.
    ``read_parquet(f"{output_dir}/*.parquet")``; ``cell_index`` values are
    stable across tiles, so downstream joins don't need to know about shards.

    ``max_workers`` defaults to the host's detected CPU count (via
    :meth:`RuntimeResources.detect`, consistent with the rest of the SDK's
    resource-aware defaults). A single tile, or ``max_workers=1``, runs
    in-process with no pool at all. Parallelism is bounded by tile count, not
    just ``max_workers``: a ``tile_degrees`` coarse enough to produce fewer
    tiles than workers leaves some workers idle regardless of how many are
    requested — pick ``tile_degrees`` so the resulting grid has at least
    ``max_workers`` tiles for full utilization.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tiles = tile_bounds(bounds, tile_degrees)
    shard_paths = tuple(
        output / f"tile_{index:04d}.parquet" for index in range(len(tiles))
    )
    kwargs = dict(provider_kwargs or {})
    workers = max_workers or RuntimeResources.detect(output).cpus

    if len(tiles) == 1 or workers <= 1:
        # No outer pool here, so each tile's own write is free to let
        # validation parallelize itself (validate_max_workers defaults to
        # None -> auto).
        sequential_tile = partial(
            _canonicalize_tile,
            spec=spec,
            policy=policy,
            aoi_bounds=bounds,
            provider_kwargs=kwargs,
        )
        sequential = (
            sequential_tile(tile, path) for tile, path in zip(tiles, shard_paths)
        )
        return tuple(path for path in sequential if path is not None)

    # validate_max_workers=1: avoid nesting a second pool inside each tile worker.
    parallel_tile = partial(
        _canonicalize_tile,
        spec=spec,
        policy=policy,
        aoi_bounds=bounds,
        provider_kwargs=kwargs,
        validate_max_workers=1,
    )
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=_MP_CONTEXT
    ) as executor:
        parallel = executor.map(parallel_tile, tiles, shard_paths)
        return tuple(path for path in parallel if path is not None)


def _curve_quantiles(
    records: Sequence[Mapping[str, Any]], probability: float
) -> list[float]:
    return [
        float(
            CurveParameters(
                curve_kind=row["curve_kind"],
                curve_type=row["curve_type"],
                curve_shape=row["curve_shape"],
                curve_location=row["curve_location"],
                curve_scale=row["curve_scale"],
                curve_atom_probability=row["curve_atom_probability"],
                curve_atom_location=row["curve_atom_location"],
            )
            .to_distribution()
            .quantiles([probability])[0]
        )
        for row in records
    ]


def _evaluate_in_chunks(
    records: Sequence[Mapping[str, Any]],
    probability: float,
    chunk_rows: int,
    executor: ProcessPoolExecutor | None,
) -> list[float]:
    """Evaluate ``records`` directly, or chunked across an already-open pool.

    Takes an *existing* executor (or none) rather than owning its lifetime,
    so a caller processing many batches (:func:`stream_curve_quantiles_to_parquet`)
    can reuse one pool across all of them instead of paying process-spawn
    overhead per batch.
    """
    if not records or executor is None or len(records) <= chunk_rows:
        return _curve_quantiles(records, probability)
    chunks = [
        records[start : start + chunk_rows]
        for start in range(0, len(records), chunk_rows)
    ]
    results = executor.map(partial(_curve_quantiles, probability=probability), chunks)
    return [value for chunk in results for value in chunk]


def curve_quantiles_at(
    table: Any,
    probability: float,
    *,
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> list[float]:
    """Reconstruct each canonical row's curve and evaluate one quantile.

    Curve reconstruction is Pydantic validation plus a per-row Rust call, the
    one step in a hazard join that DuckDB/Arrow don't vectorize. Rows are
    chunked across a process pool; single-chunk or ``max_workers=1`` runs
    in-process with no pool.
    """
    records = table.to_pylist()
    if not records:
        return []
    workers = max_workers or detected_cpu_count()
    if workers <= 1 or len(records) <= chunk_rows:
        return _curve_quantiles(records, probability)
    with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CONTEXT) as executor:
        return _evaluate_in_chunks(records, probability, chunk_rows, executor)


_CURVE_COLUMNS = (
    "curve_kind",
    "curve_type",
    "curve_shape",
    "curve_location",
    "curve_scale",
    "curve_atom_probability",
    "curve_atom_location",
)


def stream_curve_quantiles_to_parquet(
    con: Any,
    source_sql: str,
    probability: float,
    output_path: str | Path,
    *,
    passthrough_columns: Sequence[str],
    batch_rows: int = 50_000,
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> int:
    """Stream a curve-bearing query through quantile evaluation, writing incrementally.

    ``source_sql`` must select the seven canonical curve columns
    (``curve_kind`` .. ``curve_atom_location``) plus whatever
    ``passthrough_columns`` should ride along (e.g. ``cell_index``,
    ``province``) — DuckDB does the join/filtering that produces this query
    out-of-core, spilling to disk under memory pressure on its own.

    Curve reconstruction is the one step in a hazard pipeline DuckDB/Arrow
    can't vectorize, so rows are pulled via DuckDB's own Arrow batch reader
    (``batch_rows`` at a time) rather than materialized in one Python/pandas
    structure; each batch is evaluated and appended to ``output_path``
    immediately, so peak memory is bounded by one batch, not the query's full
    result size, however large the upstream join is. One process pool is
    opened for the whole stream (not per batch) when ``max_workers`` calls
    for one, so spawn overhead is paid once rather than per batch. Returns
    the row count written. The output schema (``passthrough_columns`` +
    ``depth_m``) is fixed from the query's own schema up front, so a
    zero-row source still produces a valid, correctly-typed empty Parquet
    file rather than none.
    """
    reader = con.execute(source_sql).to_arrow_reader(batch_rows)
    output_schema = pa.schema(
        [reader.schema.field(name) for name in passthrough_columns]
        + [pa.field("depth_m", pa.float64())]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = max_workers or detected_cpu_count()
    written = 0

    def _drain(executor: ProcessPoolExecutor | None) -> None:
        nonlocal written
        with pq.ParquetWriter(output, output_schema, compression="zstd") as writer:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                curve_table = pa.Table.from_arrays(
                    [batch.column(name) for name in _CURVE_COLUMNS],
                    names=list(_CURVE_COLUMNS),
                )
                depths = _evaluate_in_chunks(
                    curve_table.to_pylist(), probability, chunk_rows, executor
                )
                out_batch = pa.RecordBatch.from_arrays(
                    [batch.column(name) for name in passthrough_columns]
                    + [pa.array(depths, type=pa.float64())],
                    schema=output_schema,
                )
                writer.write_batch(out_batch)
                written += out_batch.num_rows

    if workers <= 1:
        _drain(None)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=_MP_CONTEXT
        ) as executor:
            _drain(executor)
    return written
