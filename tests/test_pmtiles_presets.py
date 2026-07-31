from dataclasses import replace
from pathlib import Path

import pytest

from crc_sdk.geometry.pmtiles.presets import (
    AREAS,
    POINTS,
    POLYGONS,
    POLYGONS_CAPPED,
    TippecanoePreset,
    ZoomRange,
    coerce_zoom_range,
    tippecanoe_command,
)


def test_zoom_range_rejects_out_of_order_bounds() -> None:
    with pytest.raises(ValueError):
        ZoomRange(10, 5)


def test_zoom_range_rejects_out_of_spec_bounds() -> None:
    with pytest.raises(ValueError):
        ZoomRange(0, 25)


def test_coerce_zoom_range_passes_through_zoom_range() -> None:
    zooms = ZoomRange(2, 8)
    assert coerce_zoom_range(zooms) is zooms


def test_coerce_zoom_range_treats_two_element_sequence_as_min_max() -> None:
    assert coerce_zoom_range((0, 10)) == ZoomRange(0, 10)
    assert coerce_zoom_range([0, 10]) == ZoomRange(0, 10)


def test_coerce_zoom_range_takes_min_max_of_other_iterables() -> None:
    assert coerce_zoom_range(range(0, 11)) == ZoomRange(0, 10)
    assert coerce_zoom_range({3, 7, 1}) == ZoomRange(1, 7)


def test_coerce_zoom_range_rejects_empty_iterable() -> None:
    with pytest.raises(ValueError):
        coerce_zoom_range([])


def test_preset_rejects_conflicting_tile_size_flags() -> None:
    with pytest.raises(ValueError):
        TippecanoePreset(name="bad", no_tile_size_limit=True, maximum_tile_bytes=1000)


def test_preset_rejects_conflicting_feature_limit_flags() -> None:
    with pytest.raises(ValueError):
        TippecanoePreset(name="bad", no_feature_limit=True, maximum_tile_features=1000)


def test_tippecanoe_command_names_layer_and_zoom_flags() -> None:
    command = tippecanoe_command(
        POLYGONS,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer="buildings",
        zooms=ZoomRange(11, 14),
        temp_dir=Path("/tmp/tc"),
    )
    assert command[0] == "tippecanoe"
    assert "-o" in command and "/tmp/out.pmtiles" in command
    assert "-l" in command and "buildings" in command
    assert "-Z11" in command
    assert "-z14" in command


def test_tippecanoe_command_omits_layer_flag_when_none() -> None:
    command = tippecanoe_command(
        POLYGONS,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer=None,
        zooms=ZoomRange(0, 14),
        temp_dir=Path("/tmp/tc"),
    )
    assert "-l" not in command


def test_points_preset_renders_density_dropping_flags() -> None:
    command = tippecanoe_command(
        POINTS,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer="points",
        zooms=ZoomRange(0, 10),
        temp_dir=Path("/tmp/tc"),
    )
    assert "--drop-densest-as-needed" in command
    assert "--maximum-tile-bytes=350000" in command
    assert "--base-zoom=g" in command
    assert "--limit-base-zoom-to-maximum-zoom" in command


def test_polygons_preset_is_lossless_by_default() -> None:
    command = tippecanoe_command(
        POLYGONS,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer="buildings",
        zooms=ZoomRange(11, 14),
        temp_dir=Path("/tmp/tc"),
    )
    assert "--no-tile-size-limit" in command
    assert "--no-feature-limit" in command
    assert "--maximum-tile-bytes" not in " ".join(command)


def test_polygons_capped_bounds_tile_bytes_instead() -> None:
    command = tippecanoe_command(
        POLYGONS_CAPPED,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer="buildings",
        zooms=ZoomRange(11, 14),
        temp_dir=Path("/tmp/tc"),
    )
    assert "--drop-smallest-as-needed" in command
    assert "--maximum-tile-bytes=350000" in command
    assert "--no-tile-size-limit" not in command


def test_areas_preset_is_lossless_with_shared_border_detection() -> None:
    command = tippecanoe_command(
        AREAS,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer="hazard_areas",
        zooms=ZoomRange(0, 12),
        temp_dir=Path("/tmp/tc"),
    )
    assert "--no-line-simplification" in command
    assert "--detect-shared-borders" in command
    assert "--no-tile-size-limit" in command


def test_replace_overrides_a_single_field_without_disturbing_others() -> None:
    custom = replace(POLYGONS, exclude_columns=("building_id",))
    command = tippecanoe_command(
        custom,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer="buildings",
        zooms=ZoomRange(11, 14),
        temp_dir=Path("/tmp/tc"),
    )
    assert command.count("-x") == 1
    assert "building_id" in command
    assert "--no-tile-size-limit" in command


def test_extra_args_are_appended_last() -> None:
    custom = replace(POLYGONS, extra_args=("--force-feature-limit",))
    command = tippecanoe_command(
        custom,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer="buildings",
        zooms=ZoomRange(11, 14),
        temp_dir=Path("/tmp/tc"),
    )
    assert command[-1] == "--force-feature-limit"


def test_accumulate_attribute_is_rendered_as_json() -> None:
    custom = replace(
        POINTS, accumulate_attribute={"impacted_buildings": "sum", "depth_m": "max"}
    )
    command = tippecanoe_command(
        custom,
        "tippecanoe",
        Path("/tmp/out.pmtiles"),
        layer="points",
        zooms=ZoomRange(0, 10),
        temp_dir=Path("/tmp/tc"),
    )
    accumulate_args = [
        arg for arg in command if arg.startswith("--accumulate-attribute=")
    ]
    assert len(accumulate_args) == 1
    assert '"impacted_buildings":"sum"' in accumulate_args[0]
    assert '"depth_m":"max"' in accumulate_args[0]
