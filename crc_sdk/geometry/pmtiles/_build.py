"""Orchestration: budget check, one tiling pass, local-scratch-then-publish.

This is the private execution engine behind :meth:`PMTilesBuild.write`. It
always tiles to a local scratch file first (tippecanoe has no other mode),
then publishes to the requested destination -- an atomic rename for a local
path, an ``fsspec`` upload for a remote one -- since that publish step has no
DuckDB equivalent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crc_sdk.connectors.duckdb import DuckDBConnection, default_work_dir

from . import _geojson_sql
from ._process import SubprocessPipeOptions, TippecanoeProcess
from .binaries import require_tippecanoe
from .budget import (
    TilingBudget,
    check_tiling_budget,
    measure_source,
    resolve_temp_to_input_factor,
)
from .presets import ZoomRange, tippecanoe_command

if TYPE_CHECKING:
    from .archive import PMTilesBuild, PMTilesResult

logger = logging.getLogger(__name__)

# Arrow batch size for the combined feature query -- matches
# `stream_curve_quantiles_wide_to_parquet`'s own default, a reasonable
# bound on how much of the upstream scan is ever materialized at once.
BATCH_ROWS = 50_000


def _is_remote(destination: str) -> bool:
    return "://" in destination


def build_pmtiles_archive(build: PMTilesBuild, output: str) -> PMTilesResult:
    """Execute one :class:`PMTilesBuild` request, returning its result."""
    from .archive import PMTilesResult

    work_root = (
        Path(build.work_dir) if build.work_dir is not None else default_work_dir()
    )
    work_root.mkdir(parents=True, exist_ok=True)

    tippecanoe_bin = require_tippecanoe()

    # An explicit connection means the caller is already in control -- only
    # build (and resource-tune) one, and only touch its thread count, when
    # they didn't supply their own.
    owns_connection = build.con is None
    connection: Any = (
        build.con
        or DuckDBConnection.for_analytics(
            work_root, extensions=("spatial", "httpfs", "h3")
        ).connect()
    )

    scratch_dir = work_root / "pmtiles-scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        budget = TilingBudget.detect(
            work_root,
            tippecanoe_threads=build.tippecanoe_threads,
            duckdb_threads=build.duckdb_threads,
            scratch_fraction=build.scratch_fraction,
        )
        if owns_connection:
            connection.execute(f"SET threads={budget.duckdb_threads}")

        total_bytes = 0
        total_features = 0
        for layer in build.layers:
            layer_bytes, layer_features = measure_source(
                layer.source, con=connection, work_dir=work_root
            )
            total_bytes += layer_bytes
            total_features += layer_features
        if total_features == 0:
            # Confirmed empirically: tippecanoe exits non-zero ("did not
            # read any valid geometries") on zero fed features regardless of
            # flags, and the file it leaves behind despite that failure is
            # not even a valid PMTiles archive (it's a bare SQLite/mbtiles
            # shell). There is nothing tippecanoe can produce here, so this
            # is the caller's decision, not something to paper over.
            raise ValueError(
                f"no features to tile across {len(build.layers)} layer(s) -- "
                f"nothing to write for {output!r}"
            )
        check_tiling_budget(
            total_bytes,
            total_features,
            budget,
            temp_to_input_factor=resolve_temp_to_input_factor(
                build.temp_to_input_factor
            ),
        )

        effective_zooms = ZoomRange(
            min(layer.zooms.minimum for layer in build.layers),
            max(layer.zooms.maximum for layer in build.layers),
        )
        single_layer_name = build.layers[0].name if len(build.layers) == 1 else None

        tippecanoe_temp = scratch_dir / "tippecanoe-temp"
        tippecanoe_temp.mkdir(parents=True, exist_ok=True)
        local_output = scratch_dir / "output.pmtiles"
        local_output.unlink(missing_ok=True)
        environment = {"TIPPECANOE_MAX_THREADS": str(budget.tippecanoe_threads)}
        options = (
            SubprocessPipeOptions(idle_timeout_seconds=build.idle_timeout_seconds)
            if build.idle_timeout_seconds is not None
            else SubprocessPipeOptions()
        )

        layer_sources = [
            _geojson_sql.LayerSource(
                source=layer.source,
                layer=layer.name,
                minzoom=layer.zooms.minimum,
                maxzoom=layer.zooms.maximum,
                property_columns=layer.property_columns,
            )
            for layer in build.layers
        ]
        query = _geojson_sql.build_combined_query(connection, layer_sources)
        # tippecanoe applies one set of algorithmic flags per process -- a
        # single build spanning layers with genuinely different presets
        # (e.g. a density-dropping point layer alongside a lossless polygon
        # layer) uses the FIRST layer's preset for the whole process; only
        # each feature's layer name/zoom window varies per-layer (via the
        # `tippecanoe.layer`/`minzoom`/`maxzoom` tag in `_geojson_sql`).
        # Callers needing genuinely distinct algorithms per layer should
        # build separate archives and combine them with `tile-join`
        # (`require_tile_join`) -- that is archive-combining, not
        # GeoParquet -> PMTiles, and outside this primitive's scope.
        effective_preset = build.layers[0].preset
        command = tippecanoe_command(
            effective_preset,
            tippecanoe_bin,
            local_output,
            layer=single_layer_name,
            zooms=effective_zooms,
            temp_dir=tippecanoe_temp,
        )
        reader = connection.execute(query).to_arrow_reader(BATCH_ROWS)
        with TippecanoeProcess(
            command, environment=environment, options=options
        ) as process:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                # The record separator and trailing newline are already part
                # of each `feature` string (baked in by `_geojson_sql`, in
                # SQL) -- no per-row Python formatting here, just a batched
                # concatenation.
                payload = "".join(batch.column("feature").to_pylist())
                process.write(payload.encode("utf-8"))

        _publish(local_output, output)

        return PMTilesResult(
            output=output,
            layers=tuple(layer.name for layer in build.layers),
            tippecanoe_threads=budget.tippecanoe_threads,
            duckdb_threads=budget.duckdb_threads,
        )
    finally:
        if owns_connection:
            connection.close()
        import shutil

        shutil.rmtree(scratch_dir, ignore_errors=True)


def _publish(local_output: Path, destination: str) -> None:
    """Move the finished local archive to its final (local or remote) home."""
    if _is_remote(destination):
        import fsspec  # type: ignore[import-untyped]

        filesystem, remote_path = fsspec.core.url_to_fs(destination)
        parent = os.path.dirname(remote_path)
        if parent:
            filesystem.makedirs(parent, exist_ok=True)
        filesystem.put_file(str(local_output), remote_path)
        return
    final_path = Path(destination)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(local_output, final_path)
