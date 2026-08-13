from __future__ import annotations

import json
from pathlib import Path

import h5netcdf  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.connectors import read_hazard_metadata
from crc_sdk.providers.jrc_edo import EDO_DATASETS, EDODataset
from crc_sdk.workflows import (
    EDOCanonicalizationPlan,
    EDODroughtPolicy,
    HazardDataset,
    ReturnPeriodExtrapolationWarning,
)

_YEARS = tuple(range(2000, 2006))


def _write_year(path: Path, index: int) -> None:
    base = 0.10 + index * 0.04
    data: np.ndarray = np.empty((2, 2, 2), dtype=np.float32)
    data[0] = np.array(
        [[base, base + 0.01], [base + 0.02, base + 0.03]], dtype=np.float32
    )
    data[1] = data[0] + 0.2
    with h5netcdf.File(path, "w") as output:
        output.dimensions = {"time": 2, "lat": 2, "lon": 2}
        output.create_variable(
            "lat", ("lat",), dtype=np.float64, data=np.array([1.0, 0.0])
        )
        output.create_variable(
            "lon", ("lon",), dtype=np.float64, data=np.array([10.0, 11.0])
        )
        output.create_variable(
            "sminx", ("time", "lat", "lon"), dtype=np.float32, data=data
        )


def _dataset(directory: Path) -> EDODataset:
    directory.mkdir(parents=True, exist_ok=True)
    for index, year in enumerate(_YEARS):
        _write_year(directory / f"smi_{year}.nc", index)
    return EDODataset(
        name="test-edo-smi",
        base_url=str(directory),
        filename_template="smi_{year}.nc",
        variable="sminx",
        version="1.0.0",
    )


def _plan(tmp_path: Path) -> EDOCanonicalizationPlan:
    return (
        HazardDataset.edo("test", version="1.0.0")
        .for_area((9.5, -0.5, 11.5, 1.5))
        .years(_YEARS)
        .cache(tmp_path / "cache", mode="reuse")
        .canonicalize(
            policy=EDODroughtPolicy.curated(
                h3_resolution=7,
                minimum_years=4,
            )
        )
    )


def test_builder_is_lazy_and_explain_does_not_resolve_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(EDO_DATASETS, "test", _dataset(tmp_path / "source"))
    plan = _plan(tmp_path)

    details = plan.explain(format="json")

    assert details["resolved_version"] is None
    assert details["requested_years"] == _YEARS
    assert details["return_period_tail"] == "lower"
    assert not (tmp_path / "cache" / "manifest.json").exists()


def test_prefetch_caches_annual_minima_and_offline_materializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(EDO_DATASETS, "test", _dataset(tmp_path / "source"))
    plan = _plan(tmp_path)

    prefetched = plan.prefetch()
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    cached = Path(manifest["resources"][0]["local"])
    with h5netcdf.File(cached) as source:
        assert source.variables["sminx"].shape == (1, 2, 2)
    hazard = plan.cache(tmp_path / "cache", mode="offline").materialize(
        tmp_path / "drought.parquet"
    )

    assert prefetched.cache_misses == len(_YEARS)
    assert hazard.materialization is not None
    assert hazard.materialization.source_cache_hits == len(_YEARS)
    metadata = read_hazard_metadata(hazard.provider.source)
    assert metadata.return_period_tail == "lower"
    assert metadata.return_period_support == pytest.approx(
        ((len(_YEARS) + 0.12) / (len(_YEARS) - 0.44), (len(_YEARS) + 0.12) / 0.56)
    )


def test_one_chain_uses_lower_tail_and_warns_on_extrapolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(EDO_DATASETS, "test", _dataset(tmp_path / "source"))
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "longitude": [10.0],
            "latitude": [1.0],
        }
    )

    with pytest.warns(ReturnPeriodExtrapolationWarning, match="25.0"):
        result = (
            _plan(tmp_path)
            .for_assets(assets)
            .select(hazard_names=["Drought"])
            .return_periods([2, 25])
            .write_parquet(tmp_path / "portfolio.parquet")
        )

    row = pq.read_table(result.output).to_pylist()[0]
    assert row["value_rp25"] < row["value_rp2"]
    metadata = json.loads(
        (pq.read_schema(result.output).metadata or {})[b"crc.hazard.evaluation"]
    )
    assert metadata["return_period_tail"] == "lower"


def test_curated_policy_requires_a_defensible_record_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(EDO_DATASETS, "test", _dataset(tmp_path / "source"))
    plan = (
        HazardDataset.edo("test", version="1.0.0")
        .for_area((9.5, -0.5, 11.5, 1.5))
        .years(_YEARS)
        .cache(tmp_path / "cache", mode="reuse")
        .canonicalize(policy="curated")
    )

    with pytest.raises(ValueError, match="at least 20 complete years"):
        plan.prefetch()
