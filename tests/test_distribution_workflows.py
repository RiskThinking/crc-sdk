from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, HurdleDistribution
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from crc_sdk.connectors import (
    hazard_arrow_schema,
    write_hazard_dataset,
)
from crc_sdk.geometry import point_to_cell
from crc_sdk.providers import LocalProvider
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import (
    PORTFOLIO_METADATA_KEY,
    AssetPortfolio,
    CellColumn,
    ExecutionOptions,
    HazardDataset,
    PointColumns,
    curve_parameters_from_row,
    curve_quantiles,
    distribution_from_hazard_row,
    return_period_value_columns,
    return_periods_to_probabilities,
)

LONGITUDE = 6.9603
LATITUDE = 50.9375
H3_RESOLUTION = 7


def _metadata() -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=H3_RESOLUTION,
        value_unit="metres",
        value_semantics="flood depth",
        producer="tests",
        source=SourceProvenance(provider="fixture", dataset="flood"),
        creation_version="1",
    )


def _row(
    *,
    cell_index: int | None = None,
    source_id: str = "source-a",
    source_geometry: bytes | None = None,
    horizon: int = 2050,
    pathway: str = "ssp585",
    curve_kind: str = "fitted",
    curve_location: float = 2.0,
) -> dict[str, Any]:
    hurdle = curve_kind == "hurdle"
    return {
        "cell_index": cell_index
        if cell_index is not None
        else point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION),
        "source_id": source_id,
        "source_geometry": source_geometry,
        "hazard_name": "flood",
        "horizon": horizon,
        "pathway": pathway,
        "curve_kind": curve_kind,
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": curve_location,
        "curve_scale": 3.0,
        "curve_atom_probability": 0.5 if hurdle else None,
        "curve_atom_location": 0.0 if hurdle else None,
    }


def _table(*rows: dict[str, Any]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=hazard_arrow_schema())


def _provider(tmp_path: Path, *rows: dict[str, Any]) -> LocalProvider:
    source = tmp_path / "hazard.parquet"
    write_hazard_dataset(
        _table(*rows),
        source,
        _metadata(),
        max_workers=1,
    )
    return LocalProvider(source)


def test_row_utilities_reconstruct_fitted_and_hurdle_curves() -> None:
    fitted = distribution_from_hazard_row(_row())
    hurdle = distribution_from_hazard_row(_table(_row(curve_kind="hurdle")))

    assert isinstance(fitted, FittedDistribution)
    assert isinstance(hurdle, HurdleDistribution)
    assert curve_parameters_from_row(_row()).curve_type == "gumbel_r"


def test_return_periods_map_to_probabilities_and_wide_columns() -> None:
    periods = [25, 50, 100, 250, 500, 1000, 2.5]

    probabilities = return_periods_to_probabilities(periods)
    columns = return_period_value_columns(periods)

    assert probabilities == pytest.approx(
        [0.96, 0.98, 0.99, 0.996, 0.998, 0.999, 0.6]
    )
    assert columns == (
        "value_rp25",
        "value_rp50",
        "value_rp100",
        "value_rp250",
        "value_rp500",
        "value_rp1000",
        "value_rp2_5",
    )


@pytest.mark.parametrize(
    "periods",
    [
        [],
        [1],
        [0],
        [float("inf")],
        [25, 25.0],
    ],
)
def test_return_periods_reject_invalid_values(periods: list[float]) -> None:
    with pytest.raises(ValueError):
        return_periods_to_probabilities(periods)


def test_curve_quantiles_evaluates_all_probabilities_per_row() -> None:
    table = _table(
        _row(source_id="fitted"),
        _row(source_id="hurdle", curve_kind="hurdle"),
    )
    probabilities = return_periods_to_probabilities([25, 100])

    values = curve_quantiles(table, probabilities, max_workers=1)

    assert len(values) == 2
    assert all(len(row) == 2 for row in values)
    expected = [
        tuple(
            float(value)
            for value in distribution_from_hazard_row(table, row_index=index).quantiles(
                probabilities
            )
        )
        for index in range(2)
    ]
    assert values == pytest.approx(expected)


def test_asset_portfolio_infers_point_location_and_passthrough_columns() -> None:
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "longitude": [LONGITUDE],
            "latitude": [LATITUDE],
            "sector": ["energy"],
        }
    )

    portfolio = AssetPortfolio(assets)

    assert portfolio.location == PointColumns()
    assert portfolio.passthrough_columns == ("sector",)


def test_asset_portfolio_accepts_explicit_nonstandard_columns() -> None:
    assets = pa.table(
        {
            "uuid": ["asset-a"],
            "x": [LONGITUDE],
            "y": [LATITUDE],
            "region": ["west"],
        }
    )

    portfolio = AssetPortfolio(
        assets,
        id_column="uuid",
        location=PointColumns(longitude="x", latitude="y"),
    )

    assert portfolio.passthrough_columns == ("region",)


def test_evaluate_cell_portfolio_writes_wide_scenario_rows(
    tmp_path: Path,
) -> None:
    cell_a = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    cell_b = point_to_cell(7.5, 51.0, H3_RESOLUTION)
    provider = _provider(
        tmp_path,
        _row(cell_index=cell_a, source_id="a", curve_location=1.0),
        _row(cell_index=cell_b, source_id="b", curve_location=4.0),
        _row(
            cell_index=cell_a,
            source_id="a-2080",
            horizon=2080,
            curve_location=6.0,
        ),
    )
    assets = pa.table(
        {
            "asset_id": ["asset-a", "asset-b"],
            "hazard_cell": pa.array([cell_a, cell_b], type=pa.uint64()),
            "sector": ["energy", "finance"],
        }
    )
    output = tmp_path / "evaluated.parquet"

    result = (
        HazardDataset.local(provider.source)
        .for_assets(
            AssetPortfolio(
                assets,
                location=CellColumn("hazard_cell"),
            )
        )
        .select(horizons=[2050])
        .return_periods([25, 100])
        .write_parquet(
            output,
            execution=ExecutionOptions(max_workers=1),
        )
    )

    assert result.row_count == 2
    assert result.value_columns == ("value_rp25", "value_rp100")
    table = pq.read_table(output)
    assert table.column_names == [
        "asset_id",
        "sector",
        "cell_index",
        "hazard_name",
        "horizon",
        "pathway",
        "source_id",
        "spatial_match",
        "value_rp25",
        "value_rp100",
    ]
    assert set(table["asset_id"].to_pylist()) == {"asset-a", "asset-b"}
    assert set(table["horizon"].to_pylist()) == {2050}
    assert set(table["spatial_match"].to_pylist()) == {"h3_cell"}

    encoded = (pq.read_schema(output).metadata or {})[
        PORTFOLIO_METADATA_KEY.encode()
    ]
    metadata = json.loads(encoded)
    assert metadata["value_unit"] == "metres"
    assert metadata["value_semantics"] == "flood depth"
    assert metadata["return_periods"][1] == {
        "column": "value_rp100",
        "probability": 0.99,
        "return_period": 100.0,
    }


def test_evaluate_point_portfolio_refines_geometry_and_preserves_coordinates(
    tmp_path: Path,
) -> None:
    containing = Polygon(
        [
            (LONGITUDE - 0.01, LATITUDE - 0.01),
            (LONGITUDE + 0.01, LATITUDE - 0.01),
            (LONGITUDE + 0.01, LATITUDE + 0.01),
            (LONGITUDE - 0.01, LATITUDE + 0.01),
        ]
    )
    elsewhere = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    provider = _provider(
        tmp_path,
        _row(source_id="match", source_geometry=containing.wkb),
        _row(source_id="other", source_geometry=elsewhere.wkb),
    )
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "longitude": [LONGITUDE],
            "latitude": [LATITUDE],
        }
    )
    output = tmp_path / "point.parquet"

    result = (
        HazardDataset(provider)
        .for_assets(assets)
        .return_periods([100])
        .write_parquet(
            output,
            execution=ExecutionOptions(max_workers=1),
        )
    )

    assert result.row_count == 1
    row = pq.read_table(output).to_pylist()[0]
    assert row["source_id"] == "match"
    assert row["spatial_match"] == "exact_geometry"
    assert row["longitude"] == LONGITUDE
    assert row["latitude"] == LATITUDE


def test_evaluate_point_portfolio_marks_null_wkb_as_cell_match(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path, _row())
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "longitude": [LONGITUDE],
            "latitude": [LATITUDE],
        }
    )
    output = tmp_path / "point-cell-level.parquet"

    (
        HazardDataset(provider)
        .for_assets(assets)
        .return_periods([100])
        .write_parquet(
            output,
            execution=ExecutionOptions(max_workers=1),
        )
    )

    assert pq.read_table(output)["spatial_match"].to_pylist() == ["h3_cell"]


def test_evaluate_portfolio_rejects_ambiguous_sources(tmp_path: Path) -> None:
    cell = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    provider = _provider(
        tmp_path,
        _row(source_id="a"),
        _row(source_id="b"),
    )
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "cell_index": pa.array([cell], type=pa.uint64()),
        }
    )

    with pytest.raises(LookupError, match="multiple source curves"):
        (
            HazardDataset(provider)
            .for_assets(assets)
            .return_periods([100])
            .write_parquet(
                tmp_path / "ambiguous.parquet",
                execution=ExecutionOptions(max_workers=1),
            )
        )


def test_evaluate_portfolio_rejects_missing_asset_scenarios(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path, _row())
    other_cell = point_to_cell(0.0, 0.0, H3_RESOLUTION)
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "cell_index": pa.array([other_cell], type=pa.uint64()),
        }
    )

    with pytest.raises(LookupError, match="missing hazard curves"):
        (
            HazardDataset(provider)
            .for_assets(assets)
            .return_periods([100])
            .write_parquet(
                tmp_path / "missing.parquet",
                execution=ExecutionOptions(max_workers=1),
            )
        )


def test_evaluate_portfolio_empty_assets_writes_typed_empty_file(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path, _row())
    assets = pa.table(
        {
            "asset_id": pa.array([], type=pa.string()),
            "cell_index": pa.array([], type=pa.uint64()),
        }
    )
    output = tmp_path / "empty.parquet"

    result = (
        HazardDataset(provider)
        .for_assets(assets)
        .return_periods([25, 100])
        .write_parquet(
            output,
            execution=ExecutionOptions(max_workers=1),
        )
    )

    assert result.row_count == 0
    assert pq.read_table(output).column_names[-2:] == [
        "value_rp25",
        "value_rp100",
    ]
