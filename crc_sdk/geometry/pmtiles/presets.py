"""Tippecanoe flag presets for building PMTiles archives.

Named, reusable, testable recipes replacing gen_pmtiles_v2's three
near-duplicate command builders (points/polygons/areas), with every tunable
promoted to a dataclass field so a caller overrides one flag via
``dataclasses.replace`` instead of hand-writing the whole argv list.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ZoomRange:
    """A tippecanoe ``-Z``/``-z`` zoom window, validated once at construction."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if not (0 <= self.minimum <= self.maximum <= 24):
            raise ValueError(
                "zoom range must satisfy 0 <= minimum <= maximum <= 24, got "
                f"({self.minimum}, {self.maximum})"
            )


def coerce_zoom_range(zooms: ZoomRange | Iterable[int]) -> ZoomRange:
    """Normalize ``zooms`` into a :class:`ZoomRange`.

    Accepts a :class:`ZoomRange` (passed through), a ``(min, max)``
    tuple/list (used as-is, not sorted -- reversed bounds are still a
    genuine error `ZoomRange` will raise on), or any other iterable of ints
    (a ``range``, a ``set``, ...), which takes ``min(values)``/``max(values)``.
    This is purely a call-site convenience so ``zooms=(0, 10)`` or
    ``zooms=range(0, 11)`` both work without importing ``ZoomRange`` for the
    common case.
    """
    if isinstance(zooms, ZoomRange):
        return zooms
    values = list(zooms)
    if not values:
        raise ValueError("zooms must not be empty")
    if len(values) == 2:
        return ZoomRange(int(values[0]), int(values[1]))
    return ZoomRange(int(min(values)), int(max(values)))


@dataclass(frozen=True)
class TippecanoePreset:
    """A named bundle of tippecanoe flags for one tiling profile.

    Every field mirrors one flag family from gen_pmtiles_v2's proven
    ``point_tippecanoe_command``/``polygon_tippecanoe_command``/
    ``area_tippecanoe_command`` builders. ``extra_args`` is the escape hatch
    for anything not worth its own field (fixed companion flags, or a
    one-off override), appended last so it always wins.
    """

    name: str
    hilbert: bool = True
    single_precision: bool = True
    no_tile_stats: bool = True
    exclude_columns: tuple[str, ...] = ()
    simplify_only_low_zooms: bool = False
    no_line_simplification: bool = False
    detect_shared_borders: bool = False
    no_tiny_polygon_reduction_at_maximum_zoom: bool = False
    drop_densest_as_needed: bool = False
    drop_smallest_as_needed: bool = False
    no_tile_size_limit: bool = False
    no_feature_limit: bool = False
    maximum_tile_bytes: int | None = None
    maximum_tile_features: int | None = None
    accumulate_attribute: Mapping[str, str] | None = None
    base_zoom: str | None = None
    drop_rate: str | None = None
    drop_denser: int | None = None
    preserve_point_density_threshold: int | None = None
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.no_tile_size_limit and self.maximum_tile_bytes is not None:
            raise ValueError(
                "no_tile_size_limit and maximum_tile_bytes are mutually exclusive"
            )
        if self.no_feature_limit and self.maximum_tile_features is not None:
            raise ValueError(
                "no_feature_limit and maximum_tile_features are mutually exclusive"
            )


# Density-dropping point layer: drops points at high zoom before dropping
# attributes, sums the count-like columns across dropped points so totals
# stay correct (`accumulate_attribute` is filled in per-call, since which
# columns are sums vs. maxes is caller/schema-specific).
POINTS = TippecanoePreset(
    name="points",
    drop_densest_as_needed=True,
    maximum_tile_bytes=350_000,
    maximum_tile_features=50_000,
    base_zoom="g",
    drop_rate="g",
    drop_denser=100,
    preserve_point_density_threshold=64,
    extra_args=("--limit-base-zoom-to-maximum-zoom",),
)

# Lossless polygon layer: every input footprint survives to the maximum
# zoom untouched. Appropriate when every feature carries information that
# must not be silently thinned (e.g. a scored building footprint) -- opt
# into POLYGONS_CAPPED instead for deliberate low-zoom size thinning.
POLYGONS = TippecanoePreset(
    name="polygons",
    simplify_only_low_zooms=True,
    no_tiny_polygon_reduction_at_maximum_zoom=True,
    no_tile_size_limit=True,
    no_feature_limit=True,
)

# Same polygon profile, but bounded to a tile-byte budget via
# drop-smallest-as-needed -- use only when deliberate thinning is
# acceptable for this layer.
POLYGONS_CAPPED = replace(
    POLYGONS,
    name="polygons_capped",
    no_tile_size_limit=False,
    no_feature_limit=False,
    drop_smallest_as_needed=True,
    maximum_tile_bytes=350_000,
)

# Lossless H3-hex/area layer: exact cell boundaries with shared-border
# detection, since adjacent hex cells simplifying independently would
# create visible seams in a continuous hazard surface.
AREAS = TippecanoePreset(
    name="areas",
    no_line_simplification=True,
    detect_shared_borders=True,
    no_tiny_polygon_reduction_at_maximum_zoom=True,
    no_tile_size_limit=True,
    no_feature_limit=True,
)


def tippecanoe_command(
    preset: TippecanoePreset,
    tippecanoe_bin: str,
    output: Path,
    *,
    layer: str | None,
    zooms: ZoomRange,
    temp_dir: Path,
    force: bool = True,
) -> list[str]:
    """Render ``preset`` into a tippecanoe argv reading GeoJSONSeq from stdin.

    ``layer=None`` omits ``-l`` entirely -- the right choice for a genuinely
    multi-layer build, where each feature already carries its own
    ``tippecanoe.layer`` property (see ``_geojson_sql``) and tippecanoe reads
    layer names from that instead. ``zooms`` here is the *process-wide* span;
    a multi-layer build passes the union of every layer's own zoom window,
    with each feature's narrower window applied via the same per-feature tag.

    Pure string-building, no I/O and no dependency on ``tippecanoe`` actually
    being installed -- the caller resolves the real binary path
    (:func:`crc_sdk.geometry.pmtiles.require_tippecanoe`) separately, which
    keeps this fully unit-testable without the binary present.
    """
    command = [tippecanoe_bin, "-P"]
    if force:
        command.append("-f")
    command += ["-o", str(output)]
    if layer is not None:
        command += ["-l", layer]
    command += [
        f"-Z{zooms.minimum}",
        f"-z{zooms.maximum}",
        "-t",
        str(temp_dir),
    ]
    if preset.hilbert:
        command.append("--hilbert")
    if preset.single_precision:
        command.append("--single-precision")
    if preset.no_tile_stats:
        command.append("--no-tile-stats")
    for column in preset.exclude_columns:
        command += ["-x", column]
    if preset.simplify_only_low_zooms:
        command.append("--simplify-only-low-zooms")
    if preset.no_line_simplification:
        command.append("--no-line-simplification")
    if preset.detect_shared_borders:
        command.append("--detect-shared-borders")
    if preset.no_tiny_polygon_reduction_at_maximum_zoom:
        command.append("--no-tiny-polygon-reduction-at-maximum-zoom")
    if preset.no_tile_size_limit:
        command.append("--no-tile-size-limit")
    elif preset.maximum_tile_bytes is not None:
        command.append(f"--maximum-tile-bytes={preset.maximum_tile_bytes}")
    if preset.no_feature_limit:
        command.append("--no-feature-limit")
    elif preset.maximum_tile_features is not None:
        command.append(f"--maximum-tile-features={preset.maximum_tile_features}")
    if preset.drop_densest_as_needed:
        command.append("--drop-densest-as-needed")
    if preset.drop_smallest_as_needed:
        command.append("--drop-smallest-as-needed")
    if preset.accumulate_attribute:
        command.append(
            "--accumulate-attribute="
            + json.dumps(dict(preset.accumulate_attribute), separators=(",", ":"))
        )
    if preset.base_zoom is not None:
        command.append(f"--base-zoom={preset.base_zoom}")
    if preset.drop_rate is not None:
        command.append(f"--drop-rate={preset.drop_rate}")
    if preset.drop_denser is not None:
        command.append(f"--drop-denser={preset.drop_denser}")
    if preset.preserve_point_density_threshold is not None:
        command.append(
            "--preserve-point-density-threshold="
            f"{preset.preserve_point_density_threshold}"
        )
    command.extend(preset.extra_args)
    return command
