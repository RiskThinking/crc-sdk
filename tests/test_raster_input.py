"""User-supplied quantile rasters share canonical source ownership and science."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import crc_framework as cf
import numpy as np
import pytest
from shapely import from_wkb  # type: ignore[import-untyped]

from crc_sdk import (
    Bounds,
    Cache,
    Dataset,
    Probability,
    Runtime,
    ValidationError,
    Variable,
    _source_io,
)
from crc_sdk._schema import table_to_rows

_VARIABLE = Variable(
    "company:water", measurement=cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
)
_PROBABILITY = Probability.annual(tail=cf.Tail.UPPER)


def _write(
    path: Path,
    values: Any,
    *,
    bands: int = 1,
    longitude: float = 24.0,
    crs: str | None = "EPSG:4326",
) -> None:
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=bands,
        dtype="float64",
        crs=crs,
        transform=rasterio.transform.from_origin(longitude, 60, 0.1, 0.1),
        nodata=-9999,
    ) as output:
        for band in range(1, bands + 1):
            output.write(np.asarray(values, dtype=np.float64), band)


@pytest.fixture
def rasters(tmp_path: Path) -> dict[float, Path]:
    paths = {q: tmp_path / f"q-{q}.tif" for q in (0.5, 0.9, 0.99)}
    for offset, (q, path) in enumerate(paths.items()):
        values = np.arange(1, 7, dtype=np.float64).reshape(2, 3) + offset
        values[0, 2] = -9999
        if q == 0.99:
            values[0, 0] = -9999
        else:
            values[0, 1] = -9999
        _write(path, values)
    return paths


def _recipe(paths: dict[float, Path]) -> Dataset:
    return Dataset.from_raster(paths, variable=_VARIABLE, probability=_PROBABILITY)


def test_raster_quantiles_do_not_require_annual_probability_meaning(
    rasters: dict[float, Path],
) -> None:
    recipe = Dataset.from_raster(rasters, variable=_VARIABLE)
    assert recipe.inspect().series[0].probability is None
    assert recipe.read().metadata.spec.series[0].probability is None


def test_raster_builder_and_inspection_are_pure_immutable_and_empty_no_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {0.9: tmp_path / "missing-1.tif", 0.99: tmp_path / "missing-2.tif"}
    recipe = _recipe(paths)
    paths.clear()
    monkeypatch.setattr(
        _source_io, "_download", lambda *a: pytest.fail("unexpected raster acquisition")
    )
    assert recipe.explain().operations
    assert recipe.inspect().series[0].variable == _VARIABLE
    assert recipe.select(variables=[]).read().table.num_rows == 0
    monkeypatch.undo()
    with pytest.raises(FileNotFoundError):
        recipe.read()


def test_raster_pixels_keep_partial_knots_and_report_skipped_supports(
    rasters: dict[float, Path],
) -> None:
    result = _recipe(rasters).read(runtime=Runtime(batch_rows=2))
    rows = table_to_rows(result.table, result.metadata)
    assert len(rows) == 4
    assert isinstance(rows[0].curve, cf.QuantileCurve)
    assert rows[0].curve.probability_interval == (0.5, 0.9)
    assert rows[0].curve.quantile([0.5, 0.9]) == pytest.approx([1, 2])
    assert from_wkb(rows[0].geometry).bounds == pytest.approx((24, 59.9, 24.1, 60))
    assert result.report.counts.requested == 6
    assert result.report.counts.produced == 4
    assert result.report.counts.skipped == 2
    assert result.report.counts.no_data == 0
    assert result.report.complete
    assert result.metadata.source is not None
    assert len(result.metadata.source.resources) == 3


def test_raster_input_offline_write_reopen_and_resource_cleanup(
    rasters: dict[float, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe(rasters)
    cache = tmp_path / "cache"
    original = recipe.read(runtime=Runtime(cache=Cache.reuse(cache)))
    for path in rasters.values():
        path.unlink()
    monkeypatch.setattr(
        _source_io, "_download", lambda *a: pytest.fail("offline acquisition")
    )
    offline = Runtime(cache=Cache.offline(cache))
    replay = recipe.read(runtime=offline)
    assert replay.table.equals(original.table, check_metadata=False)
    saved = recipe.write(tmp_path / "water.parquet", runtime=offline)
    assert type(saved) is Dataset
    assert saved.path is not None
    assert (
        Dataset.open(saved.path)
        .read()
        .table.equals(original.table, check_metadata=False)
    )


def test_raster_area_selection_and_native_read_windows_are_bounded(
    rasters: dict[float, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    rasterio = pytest.importorskip("rasterio")
    original = rasterio.open
    windows = []

    class Reader:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __getattr__(self, name: str) -> Any:
            return getattr(self.value, name)

        def __enter__(self) -> Reader:
            self.value.__enter__()
            return self

        def __exit__(self, *args: Any) -> None:
            self.value.__exit__(*args)

        def read(self, *args: Any, **kwargs: Any) -> Any:
            window = kwargs["window"]
            windows.append(window.width * window.height)
            return self.value.read(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", lambda *a, **kw: Reader(original(*a, **kw)))
    result = (
        _recipe(rasters)
        .select(area=Bounds(west=24.01, south=59.91, east=24.09, north=59.99))
        .read(runtime=Runtime(batch_rows=1))
    )
    assert result.table.num_rows == 1
    assert windows == [1] * 3


@pytest.mark.parametrize("fault", ["grid", "bands", "crs", "decreasing"])
def test_malformed_raster_sources_raise_typed_validation(
    rasters: dict[float, Path], fault: str
) -> None:
    target = rasters[0.99]
    values = [[3, 4, 5], [6, 7, 8]]
    _write(
        target,
        [[-10] * 3] * 2 if fault == "decreasing" else values,
        bands=2 if fault == "bands" else 1,
        longitude=25 if fault == "grid" else 24,
        crs=None if fault == "crs" else "EPSG:4326",
    )
    with pytest.raises(ValidationError) as error:
        _recipe(rasters).read()
    assert error.value.report.valid is False
    assert any(d.code == "invalid_source_data" for d in error.value.report.diagnostics)


@pytest.mark.parametrize(
    "probabilities",
    [
        {0.9: "a"},
        {-0.1: "a", 0.9: "b"},
        {float("nan"): "a", 0.9: "b"},
        {True: "a", 0.9: "b"},
    ],
)
def test_raster_probabilities_are_explicit_and_valid(
    probabilities: dict[float, str],
) -> None:
    with pytest.raises(ValueError):
        Dataset.from_raster(probabilities, variable=_VARIABLE, probability=_PROBABILITY)


def test_partial_raster_read_closes_all_native_handles(
    rasters: dict[float, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rasterio = pytest.importorskip("rasterio")
    original = rasterio.open
    opened: list[Any] = []

    def open_source(*args: Any, **kwargs: Any) -> Any:
        reader = original(*args, **kwargs)
        opened.append(reader)
        return reader

    monkeypatch.setattr(rasterio, "open", open_source)
    execution = _recipe(rasters).execute(runtime=Runtime(batch_rows=1))
    with execution:
        result = execution.read(limit=1)
        assert result.table.num_rows == 1
        assert not result.report.complete
    assert len(opened) == 3
    assert all(source.closed for source in opened)
