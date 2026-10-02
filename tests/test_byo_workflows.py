"""Bring-your-own-data plans: table, Zarr and GeoTIFF onboarding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution

from crc_sdk.connectors import CurveFitIngestPolicy
from crc_sdk.fitting import CDFCurveFitPolicy
from crc_sdk.types import SourceProvenance, TemporalWindow
from crc_sdk.workflows import BYOPlan, HazardDataset

GUMBEL = FittedDistribution.from_parameters("gumbel_r", location=20.0, scale=5.0)
P11 = np.linspace(0.0, 1.0, 11)
CELLS = list(range(599024279241097215, 599024279241097215 + 3))
RETURN_PERIODS = (2, 5, 10, 100, 1000)


def _quantiles(scale: float) -> list[float]:
    # Interior probabilities only: the 0 and 1 knots are unbounded for a Gumbel.
    clipped = np.clip(P11, 1e-3, 1 - 1e-3)
    return [float(GUMBEL.quantiles([p])[0]) * scale for p in clipped]


def _table() -> pa.Table:
    return pa.table(
        {
            "hex_id": pa.array(CELLS, pa.uint64()),
            "index_name": ["rx1day"] * 3,
            "year": pa.array([2050] * 3, pa.int32()),
            "pathway": ["ssp245"] * 3,
            "cdf_quantiles": pa.array(
                [_quantiles(1.0), _quantiles(1.1), _quantiles(1.2)],
                pa.list_(pa.float64()),
            ),
        }
    )


def _cdf_policy(**overrides: Any) -> CDFCurveFitPolicy:
    values: dict[str, Any] = {
        "h3_resolution": 5,
        "family": "gumbel_r",
        "value_unit": "mm",
        "value_semantics": "annual maximum",
        "producer": "tests",
        "source": SourceProvenance(provider="fixture", dataset="byo", version="v1"),
        "source_id": "fixture",
        "prefetch": False,
        "max_workers": 1,
        "probability_semantics": "annual_value_distribution",
        "temporal_window": TemporalWindow(kind="time_invariant", reference_year=2010),
    }
    values.update(overrides)
    return CDFCurveFitPolicy(**values)


def _assert_canonical(dataset: HazardDataset) -> None:
    metadata = dataset.metadata()
    assert metadata.schema_version == "1.3"
    assert metadata.probability_semantics == "annual_value_distribution"
    assert metadata.temporal_window is not None
    assert metadata.temporal_window.kind == "time_invariant"
    assert metadata.value_unit == "mm"
    table = pq.read_table(dataset.provider.source)
    assert sorted(table["cell_index"].to_pylist()) == sorted(
        str(cell) if table["cell_index"].type == pa.string() else cell for cell in CELLS
    )
    assert set(table["hazard_name"].to_pylist()) == {"rx1day"}


def test_table_from_arrow_roundtrips_policy_metadata(tmp_path: Path) -> None:
    plan = HazardDataset.from_table(
        _table(), policy=_cdf_policy(), probabilities=P11.tolist()
    )
    assert isinstance(plan, BYOPlan)
    dataset = plan.materialize(tmp_path / "out.parquet")
    _assert_canonical(dataset)
    assert dataset.materialization is not None
    assert dataset.materialization.canonical_rows == 3
    assert dataset.materialization.source_version == "v1"


def test_table_from_parquet_and_csv(tmp_path: Path) -> None:
    table = _table()
    parquet = tmp_path / "in.parquet"
    pq.write_table(table, parquet)
    _assert_canonical(
        HazardDataset.from_table(
            parquet, policy=_cdf_policy(), probabilities=P11.tolist()
        ).materialize(tmp_path / "a.parquet")
    )

    # DuckDB reads list columns from CSV in `[1.0, 2.0]` form.
    csv = tmp_path / "in.csv"
    rows = ["hex_id,index_name,year,pathway,cdf_quantiles"]
    for cell, quantiles in zip(CELLS, table["cdf_quantiles"].to_pylist()):
        rows.append(f'{cell},rx1day,2050,ssp245,"{quantiles}"')
    csv.write_text("\n".join(rows) + "\n")
    _assert_canonical(
        HazardDataset.from_table(
            csv, policy=_cdf_policy(), probabilities=P11.tolist()
        ).materialize(tmp_path / "b.parquet")
    )


def test_table_policy_options_flow_through(tmp_path: Path) -> None:
    sidecar = tmp_path / "diag.parquet"
    plan = HazardDataset.from_table(_table(), probabilities=P11.tolist()).canonicalize(
        policy=_cdf_policy(diagnostics=sidecar, lower_bound=0.0)
    )
    dataset = plan.materialize(tmp_path / "out.parquet")
    assert sidecar.is_file()
    assert dataset.metadata().fitting is not None


def test_table_explain_is_lazy_and_policy_required(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.parquet"
    plan = HazardDataset.from_table(missing, probabilities=P11.tolist())
    assert "not set" in str(plan.explain())
    configured = plan.canonicalize(policy=_cdf_policy())
    details = configured.explain(format="json")
    assert isinstance(details, dict) and details["source"] == "table"
    assert not (tmp_path / "canonical").exists()
    with pytest.raises(ValueError, match="policy is required"):
        plan.materialize(tmp_path / "out.parquet")
    with pytest.raises(ValueError, match="no spatial window"):
        plan.for_area((0, 0, 1, 1))
    with pytest.raises(TypeError, match="CDFCurveFitPolicy"):
        plan.canonicalize(
            policy=CurveFitIngestPolicy(
                h3_resolution=5, family="gumbel_r", producer="t"
            )
        )
    with pytest.raises(ValueError, match="at most one"):
        HazardDataset.from_table(
            missing, probabilities=P11.tolist(), return_periods=[2, 5]
        )


def test_table_return_period_axis(tmp_path: Path) -> None:
    periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    values = [10.0, 14.0, 17.0, 21.0, 24.0, 27.0]
    table = pa.table(
        {
            "hex_id": pa.array(CELLS[:1], pa.uint64()),
            "index_name": ["rx1day"],
            "year": pa.array([2050], pa.int32()),
            "pathway": ["ssp245"],
            "cdf_quantiles": pa.array([values], pa.list_(pa.float64())),
        }
    )
    dataset = HazardDataset.from_table(
        table, policy=_cdf_policy(), return_periods=periods
    ).materialize(tmp_path / "rp.parquet")
    assert dataset.materialization is not None
    assert dataset.materialization.canonical_rows == 1


def test_ensure_materialized_uses_deterministic_cache(tmp_path: Path) -> None:
    plan = HazardDataset.from_table(
        _table(), policy=_cdf_policy(), probabilities=P11.tolist()
    )
    with pytest.raises(ValueError, match="persistent cache"):
        plan.ensure_materialized()
    cached = plan.cache(tmp_path / "cache")
    first = cached.ensure_materialized()
    assert Path(first.provider.source).parent == tmp_path / "cache" / "canonical"
    mtime = Path(first.provider.source).stat().st_mtime_ns
    second = cached.ensure_materialized()
    assert second.provider.source == first.provider.source
    assert Path(second.provider.source).stat().st_mtime_ns == mtime
    changed = cached.canonicalize(policy=_cdf_policy(lower_bound=0.0))
    assert changed.ensure_materialized().provider.source != first.provider.source
    with pytest.raises(ValueError, match="reuse or refresh"):
        plan.cache(tmp_path, mode="offline")
    assert (
        cached.for_assets(
            pa.table({"asset_id": ["a"], "longitude": [0.0], "latitude": [0.0]})
        ).plan
        is cached
    )


def _write_zarr(path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    array = zarr.open_array(
        str(path),
        mode="w",
        shape=(len(RETURN_PERIODS), 2, 2),
        chunks=(len(RETURN_PERIODS), 2, 2),
        dtype="float64",
    )
    base = np.asarray([0.2, 0.5, 1.0, 2.0, 3.0])
    array[:] = base[:, None, None] * np.asarray([[1.0, 1.5], [2.0, 2.5]])
    array.attrs["index_name"] = "Return period"
    array.attrs["index_values"] = list(RETURN_PERIODS)
    array.attrs["transform_mat3x3"] = [0.01, 0.0, 0.0, 0.0, -0.01, 0.02]


def _ingest_policy(**overrides: Any) -> CurveFitIngestPolicy:
    values: dict[str, Any] = {
        "h3_resolution": 7,
        "family": "gumbel_r",
        "producer": "tests",
        "value_semantics": "flood depth",
        "source_version": "r1",
        "on_fit_failure": "skip",
    }
    values.update(overrides)
    return CurveFitIngestPolicy(**values)


def test_zarr_plan_is_lazy_and_canonicalizes(tmp_path: Path) -> None:
    pytest.importorskip("shapely")
    store = tmp_path / "cube.zarr"
    plan = HazardDataset.from_zarr(
        store,
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        scenario="historical",
        year=2020,
        units="m",
        policy=_ingest_policy(),
    )
    # Nothing exists yet, so explain() and for_area() must not open the store.
    assert "zarr" in str(plan.explain())
    windowed = plan.for_area((0.0, 0.0, 0.01, 0.02))
    details = windowed.explain(format="json")
    assert isinstance(details, dict)
    assert details["area"] == (0.0, 0.0, 0.01, 0.02)
    assert not store.exists()

    _write_zarr(store)
    full = plan.materialize(tmp_path / "full.parquet")
    part = windowed.materialize(tmp_path / "part.parquet")
    meta = full.metadata()
    assert meta.value_unit == "m"
    assert meta.source.provider == "byo-zarr"
    assert meta.source.version == "r1"
    assert meta.return_period_support is None or meta.return_period_support
    assert full.materialization is not None and part.materialization is not None
    assert 0 < part.materialization.canonical_rows < full.materialization.canonical_rows
    with pytest.raises(TypeError, match="CurveFitIngestPolicy"):
        plan.canonicalize(policy=_cdf_policy())


def _write_geotiffs(directory: Path) -> dict[int, Path]:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin  # type: ignore[import-untyped]

    values = {2: 0.2, 5: 0.5, 10: 1.0, 100: 2.0, 1000: 3.0}
    paths = {}
    for period, value in values.items():
        path = directory / f"RP{period}.tif"
        array = np.asarray([[value, value * 2.0], [value * 1.5, value * 3.0]])
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(0.0, 0.02, 0.01, 0.01),
            nodata=-9999.0,
        ) as target:
            target.write(array.astype("float32"), 1)
        paths[period] = path
    return paths


def test_raster_plan_canonicalizes_geotiffs(tmp_path: Path) -> None:
    pytest.importorskip("shapely")
    paths = _write_geotiffs(tmp_path)
    plan = HazardDataset.from_raster(
        paths,
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        scenario="historical",
        year=2020,
        units="m",
        policy=_ingest_policy(),
    )
    assert "RP1000" in str(plan.explain())
    sequence = HazardDataset.from_raster(
        [paths[period] for period in sorted(paths)],
        return_periods=sorted(paths),
        hazard_type="RiverineInundation",
        indicator_id="flood_depth",
        scenario="historical",
        year=2020,
        units="m",
        policy=_ingest_policy(),
    )
    assert sequence.source == plan.source
    full = plan.materialize(tmp_path / "full.parquet")
    part = plan.for_area((0.0, 0.0, 0.01, 0.02)).materialize(tmp_path / "part.parquet")
    meta = full.metadata()
    assert meta.source.provider == "byo-raster"
    assert meta.value_unit == "m"
    assert meta.return_period_support == (2.0, 1000.0)
    assert full.materialization is not None and part.materialization is not None
    assert 0 < part.materialization.canonical_rows < full.materialization.canonical_rows


def test_raster_inputs_are_validated(tmp_path: Path) -> None:
    common: dict[str, Any] = {
        "hazard_type": "h",
        "indicator_id": "i",
        "scenario": "s",
        "year": 0,
        "units": "m",
    }
    with pytest.raises(ValueError, match="four source return periods"):
        HazardDataset.from_raster({2: "a.tif", 5: "b.tif"}, **common)
    with pytest.raises(ValueError, match="one return period per"):
        HazardDataset.from_raster(["a.tif"], **common)
    with pytest.raises(ValueError, match="unique"):
        HazardDataset.from_raster(
            ["a", "b", "c", "d"], return_periods=[2, 2, 5, 10], **common
        )


def test_deferred_policy_applies_its_tail_to_return_periods(tmp_path: Path) -> None:
    # Lower tail: probability 1/T rises as T falls, so periods run downward.
    periods = [25, 10, 5, 2]
    table = pa.table(
        {
            "hex_id": pa.array(CELLS[:1], pa.uint64()),
            "index_name": ["spei"],
            "year": pa.array([2050], pa.int32()),
            "pathway": ["ssp245"],
            "cdf_quantiles": pa.array(
                [[-1.9, -1.4, -1.0, -0.5]], pa.list_(pa.float64())
            ),
        }
    )
    lower = _cdf_policy(return_period_tail="lower")

    def parameters(plan: Any, name: str) -> list[Any]:
        out = plan.materialize(tmp_path / name)
        rows = pq.read_table(out.provider.source)
        return [
            rows[column].to_pylist() for column in ("curve_location", "curve_scale")
        ]

    upfront = HazardDataset.from_table(table, policy=lower, return_periods=periods)
    deferred = HazardDataset.from_table(table, return_periods=periods).canonicalize(
        policy=lower
    )
    upper = HazardDataset.from_table(
        table, policy=_cdf_policy(), return_periods=periods
    )
    assert parameters(deferred, "deferred.parquet") == parameters(
        upfront, "upfront.parquet"
    )
    # Under the default upper tail the same periods give a decreasing axis.
    with pytest.raises(ValueError, match="strictly increasing"):
        parameters(upper, "upper.parquet")
