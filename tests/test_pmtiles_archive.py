import shutil
from pathlib import Path

import duckdb
import pytest

from crc_sdk.geometry.pmtiles import POINTS, POLYGONS, PMTilesBuild
from crc_sdk.geometry.pmtiles.budget import TilingBudget

pytestmark = pytest.mark.skipif(
    shutil.which("tippecanoe") is None,
    reason="tippecanoe is not installed on PATH -- see require_tippecanoe()",
)

_PMTILES_MAGIC = b"PMTiles"


def _write_points(path: Path, count: int = 20) -> None:
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute(
        f"""
        COPY (SELECT i AS id, ST_Point(i * 0.01, i * 0.01) AS geometry
              FROM range(1, {count + 1}) t(i))
        TO '{path}' (FORMAT PARQUET)
        """
    )


def _write_polygons(path: Path, count: int = 5) -> None:
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute(
        f"""
        COPY (SELECT i AS building_id,
                     ST_Buffer(ST_Point(i * 0.02, i * 0.02), 0.001) AS geometry
              FROM range(1, {count + 1}) t(i))
        TO '{path}' (FORMAT PARQUET)
        """
    )


def test_single_layer_build_writes_a_valid_pmtiles_archive(tmp_path: Path) -> None:
    source = tmp_path / "points.parquet"
    _write_points(source)
    output = tmp_path / "out.pmtiles"

    result = (
        PMTilesBuild()
        .layer(str(source), name="places", zooms=(0, 10), preset=POINTS)
        .write(str(output))
    )

    assert output.is_file()
    assert output.read_bytes()[:7] == _PMTILES_MAGIC
    assert result.layers == ("places",)
    assert result.output == str(output)


def test_layer_and_add_layer_are_interchangeable(tmp_path: Path) -> None:
    points = tmp_path / "points.parquet"
    polygons = tmp_path / "polygons.parquet"
    _write_points(points, count=10)
    _write_polygons(polygons, count=3)

    via_layer_then_add_layer = (
        PMTilesBuild()
        .layer(str(points), name="points", zooms=range(0, 11), preset=POINTS)
        .add_layer(str(polygons), name="buildings", zooms=(11, 14), preset=POLYGONS)
    )
    via_add_layer_then_layer = (
        PMTilesBuild()
        .add_layer(str(points), name="points", zooms=range(0, 11), preset=POINTS)
        .layer(str(polygons), name="buildings", zooms=(11, 14), preset=POLYGONS)
    )
    assert via_layer_then_add_layer.layers == via_add_layer_then_layer.layers


def test_multi_layer_build_combines_into_one_archive(tmp_path: Path) -> None:
    points = tmp_path / "points.parquet"
    polygons = tmp_path / "polygons.parquet"
    _write_points(points, count=8)
    _write_polygons(polygons, count=4)
    output = tmp_path / "multi.pmtiles"

    result = (
        PMTilesBuild()
        .layer(str(points), name="points", zooms=range(0, 11), preset=POINTS)
        .add_layer(str(polygons), name="buildings", zooms=(11, 14), preset=POLYGONS)
        .write(str(output))
    )

    assert output.is_file()
    assert output.read_bytes()[:7] == _PMTILES_MAGIC
    assert set(result.layers) == {"points", "buildings"}


def test_write_publishes_to_a_remote_fsspec_destination(tmp_path: Path) -> None:
    fsspec = pytest.importorskip("fsspec")
    source = tmp_path / "points.parquet"
    _write_points(source, count=5)

    memory_fs = fsspec.filesystem("memory")
    memory_fs.store.clear()
    destination = "memory://pmtiles-test/out.pmtiles"

    result = (
        PMTilesBuild()
        .layer(str(source), name="places", zooms=(0, 10), preset=POINTS)
        .write(destination)
    )

    assert result.output == destination
    assert memory_fs.exists("pmtiles-test/out.pmtiles")
    with memory_fs.open("pmtiles-test/out.pmtiles", "rb") as handle:
        assert handle.read(7) == _PMTILES_MAGIC


def test_write_raises_on_zero_features_without_spawning_a_broken_archive(
    tmp_path: Path,
) -> None:
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    source = tmp_path / "empty.parquet"
    con.execute(
        f"""
        COPY (SELECT 1 AS id, ST_Point(0, 0) AS geometry FROM range(0))
        TO '{source}' (FORMAT PARQUET)
        """
    )
    output = tmp_path / "empty.pmtiles"

    with pytest.raises(ValueError, match="no features to tile"):
        PMTilesBuild().layer(
            str(source), name="empty", zooms=(0, 10), preset=POINTS
        ).write(str(output))


def test_write_raises_before_spawning_tippecanoe_when_budget_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "points.parquet"
    _write_points(source, count=10)
    output = tmp_path / "out.pmtiles"

    def _tiny_budget(*args: object, **kwargs: object) -> TilingBudget:
        return TilingBudget(
            tippecanoe_threads=1,
            duckdb_threads=1,
            free_disk_bytes=1,
            safe_scratch_bytes=1,
        )

    monkeypatch.setattr(
        "crc_sdk.geometry.pmtiles._build.TilingBudget.detect",
        staticmethod(_tiny_budget),
    )
    with pytest.raises(ValueError, match="scratch disk"):
        PMTilesBuild().layer(
            str(source), name="places", zooms=(0, 10), preset=POINTS
        ).write(str(output))
    assert not output.exists()


def test_zooms_accepts_tuple_range_and_set(tmp_path: Path) -> None:
    source = tmp_path / "points.parquet"
    _write_points(source, count=3)

    for zooms in [(0, 5), range(0, 6), {0, 5}]:
        output = tmp_path / f"out-{id(zooms)}.pmtiles"
        result = (
            PMTilesBuild()
            .layer(str(source), name="places", zooms=zooms, preset=POINTS)
            .write(str(output))
        )
        assert output.is_file()
        assert result.layers == ("places",)


def test_tippecanoe_threads_default_to_detected_cpu_count(tmp_path: Path) -> None:
    from crc_sdk.connectors.duckdb import detected_cpu_count

    source = tmp_path / "points.parquet"
    _write_points(source, count=3)
    output = tmp_path / "out.pmtiles"

    result = (
        PMTilesBuild()
        .layer(str(source), name="places", zooms=(0, 10), preset=POINTS)
        .write(str(output))
    )
    assert result.tippecanoe_threads == detected_cpu_count()
    assert result.duckdb_threads == detected_cpu_count()


def test_tippecanoe_threads_env_override_is_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRC_TIPPECANOE_THREADS", "1")
    source = tmp_path / "points.parquet"
    _write_points(source, count=3)
    output = tmp_path / "out.pmtiles"

    result = (
        PMTilesBuild()
        .layer(str(source), name="places", zooms=(0, 10), preset=POINTS)
        .write(str(output))
    )
    assert result.tippecanoe_threads == 1
