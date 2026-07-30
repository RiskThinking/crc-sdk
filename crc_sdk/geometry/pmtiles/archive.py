"""Fluent public entry point: GeoParquet -> PMTiles, one archive per build."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .presets import POLYGONS, TippecanoePreset, ZoomRange, coerce_zoom_range

__all__ = ["ZoomRange", "PMTilesLayer", "PMTilesBuild", "PMTilesResult"]


@dataclass(frozen=True)
class PMTilesLayer:
    """One named layer: a GeoParquet source, its zoom window, and its preset.

    ``source`` is one file or one glob (local, ``s3://``, or ``gs://``) --
    DuckDB's own ``read_parquet`` accepts one path/glob string per relation,
    not a list, so a caller with genuinely disjoint files should
    pre-consolidate into a glob-matchable layout.
    """

    source: str
    name: str
    zooms: ZoomRange
    preset: TippecanoePreset = POLYGONS

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layer name must not be empty")
        if not self.source:
            raise ValueError("layer source must not be empty")


@dataclass(frozen=True)
class PMTilesResult:
    """Summary of one completed :meth:`PMTilesBuild.write` call."""

    output: str
    layers: tuple[str, ...]
    tippecanoe_threads: int
    duckdb_threads: int


@dataclass(frozen=True)
class PMTilesBuild:
    """Immutable, composable request to build one PMTiles archive.

    ``.layer()``/``.add_layer()`` are literally the same method under two
    names -- both start a build and both append to one, so a caller never
    has to think about which name applies at which point in a chain::

        PMTilesBuild().layer(hive_glob, name="buildings", zooms=(11, 14)).write(out)

        (PMTilesBuild()
            .layer(points_glob, name="points", zooms=range(0, 11), preset=POINTS)
            .add_layer(polygons_glob, name="buildings", zooms=(11, 14))
            .write(out))

    Defaults to one streaming tiling pass over every layer combined; raises
    ``ValueError`` before spawning tippecanoe if a pre-flight budget check
    (see :mod:`crc_sdk.geometry.pmtiles.budget`) estimates the source won't
    fit the available scratch disk -- this primitive does not auto-shard an
    oversized source (provision more disk, or narrow the run's scope).
    """

    layers: tuple[PMTilesLayer, ...] = ()
    con: Any | None = None
    work_dir: str | Path | None = None
    tippecanoe_threads: int | None = None
    duckdb_threads: int | None = None
    idle_timeout_seconds: float | None = None

    def layer(
        self,
        source: str | Path,
        *,
        name: str,
        zooms: ZoomRange | Iterable[int],
        preset: TippecanoePreset = POLYGONS,
    ) -> PMTilesBuild:
        """Return a new build with one more layer appended."""
        new_layer = PMTilesLayer(
            source=str(source),
            name=name,
            zooms=coerce_zoom_range(zooms),
            preset=preset,
        )
        return replace(self, layers=self.layers + (new_layer,))

    add_layer = layer

    def with_resources(
        self,
        *,
        con: Any | None = None,
        work_dir: str | Path | None = None,
        tippecanoe_threads: int | None = None,
        duckdb_threads: int | None = None,
        idle_timeout_seconds: float | None = None,
    ) -> PMTilesBuild:
        """Return a new build with resource/connection overrides applied."""
        return replace(
            self,
            con=con if con is not None else self.con,
            work_dir=work_dir if work_dir is not None else self.work_dir,
            tippecanoe_threads=(
                tippecanoe_threads
                if tippecanoe_threads is not None
                else self.tippecanoe_threads
            ),
            duckdb_threads=(
                duckdb_threads if duckdb_threads is not None else self.duckdb_threads
            ),
            idle_timeout_seconds=(
                idle_timeout_seconds
                if idle_timeout_seconds is not None
                else self.idle_timeout_seconds
            ),
        )

    def write(self, output: str | Path) -> PMTilesResult:
        """Tile every layer into one PMTiles archive at ``output``.

        Always tiles to a local scratch file first (tippecanoe has no other
        mode), then publishes to ``output``: an atomic rename for a local
        path, an ``fsspec`` upload for a remote one (``s3://``, ``gs://``,
        any fsspec URI).
        """
        if not self.layers:
            raise ValueError("at least one layer() is required before write()")
        from ._build import build_pmtiles_archive

        return build_pmtiles_archive(self, str(output))
