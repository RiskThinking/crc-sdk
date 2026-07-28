"""Tile-parallel raster canonicalization and curve evaluation for area-scale ingest.

DuckDB/Arrow already parallelize the join and aggregation stages of a hazard-to-
geography pipeline. The one stage they do not touch is per-pixel/per-cell pure
Python work: fitting a curve per raster pixel (:func:`canonicalize_os_climate`)
and reconstructing a curve per canonical row (:class:`CurveParameters`). Both
are parallel across independent pixels/rows, so this module splits them across
a process pool instead of leaving them single-threaded.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crc_sdk.connectors import (
    OSClimateIngestPolicy,
    write_hazard_dataset,
)
from crc_sdk.connectors.duckdb import Bounds, RuntimeResources
from crc_sdk.providers.os_climate import OSClimateProvider
from crc_sdk.types import CurveParameters


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


def _canonicalize_tile(
    spec: OSClimateSelectionSpec,
    policy: OSClimateIngestPolicy,
    bounds: Bounds,
    output_path: Path,
    provider_kwargs: Mapping[str, Any],
) -> Path | None:
    provider = OSClimateProvider(**provider_kwargs)
    resource = provider.select(
        hazard_type=spec.hazard_type,
        indicator_id=spec.indicator_id,
        model_gcm=spec.model_gcm,
    )
    selection = resource.resolve(scenario=spec.scenario, year=spec.year)
    stream = provider.canonicalize(selection, policy, bounds=bounds)
    table = stream.read_all()
    if table.num_rows == 0:
        return None
    write_hazard_dataset(table, output_path, stream.metadata)
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
    in its own worker process — tiles with no fittable pixels are skipped
    rather than written empty. Read the result back with a glob, e.g.
    ``read_parquet(f"{output_dir}/*.parquet")``; ``cell_index`` values are
    stable across tiles, so downstream joins don't need to know about shards.

    ``max_workers`` defaults to the host's detected CPU count (via
    :meth:`RuntimeResources.detect`, consistent with the rest of the SDK's
    resource-aware defaults). A single tile, or ``max_workers=1``, runs
    in-process with no pool at all.
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
        sequential = (
            _canonicalize_tile(spec, policy, tile, path, kwargs)
            for tile, path in zip(tiles, shard_paths)
        )
        return tuple(path for path in sequential if path is not None)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        parallel = executor.map(
            _canonicalize_tile,
            [spec] * len(tiles),
            [policy] * len(tiles),
            tiles,
            shard_paths,
            [kwargs] * len(tiles),
        )
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
    chunks = [
        records[start : start + chunk_rows]
        for start in range(0, len(records), chunk_rows)
    ]
    workers = max_workers or os.cpu_count() or 1

    if len(chunks) == 1 or workers <= 1:
        return _curve_quantiles(records, probability)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_curve_quantiles, chunks, [probability] * len(chunks))
        return [value for chunk in results for value in chunk]
