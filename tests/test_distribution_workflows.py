from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    FittedDistribution,
    HurdleDistribution,
    LinearImpact,
    TransformContext,
)
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
    ImpactContextColumns,
    PointColumns,
    ReturnPeriodExtrapolationWarning,
    curve_parameters_from_row,
    curve_quantiles,
    distribution_from_hazard_row,
    return_period_value_columns,
    return_periods_to_probabilities,
)

LONGITUDE = 6.9603
LATITUDE = 50.9375
H3_RESOLUTION = 7


class ContextAwareImpact:
    def __init__(self) -> None:
        self.context = TransformContext(
            continent="Europe",
            building_type="fallback",
            historic_mean=1.0,
        )

    def evaluate(
        self,
        values: Any,
        *,
        context: TransformContext | None = None,
    ) -> Any:
        assert context is not None
        assert context.cell is not None
        assert context.historic_mean is not None
        building_adjustment = 100.0 if context.building_type == "warehouse" else 0.0
        continent_adjustment = 10.0 if context.continent == "Europe" else 0.0
        return (
            np.asarray(values)
            + context.historic_mean
            + building_adjustment
            + continent_adjustment
            + context.cell % 10
        )


def _metadata(
    *,
    return_period_tail: Literal["upper", "lower"] = "upper",
    return_period_support: tuple[float, float] | None = None,
) -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=H3_RESOLUTION,
        return_period_tail=return_period_tail,
        return_period_support=return_period_support,
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


def test_row_utilities_reconstruct_point_mass_curve() -> None:
    row = _row(curve_kind="point_mass")
    row.update(
        curve_type="point_mass",
        curve_shape=None,
        curve_location=0.0,
        curve_scale=0.0,
        curve_atom_probability=1.0,
        curve_atom_location=0.0,
    )

    distribution = distribution_from_hazard_row(row)

    assert distribution.quantiles([0.01, 0.5, 0.99]).tolist() == [0.0, 0.0, 0.0]


def test_return_periods_map_to_probabilities_and_wide_columns() -> None:
    periods = [25, 50, 100, 250, 500, 1000, 2.5]

    probabilities = return_periods_to_probabilities(periods)
    columns = return_period_value_columns(periods)

    assert probabilities == pytest.approx([0.96, 0.98, 0.99, 0.996, 0.998, 0.999, 0.6])
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


def test_return_periods_lower_tail_is_the_upper_tails_complement() -> None:
    periods = [10, 100]

    lower = return_periods_to_probabilities(periods, tail="lower")
    upper = return_periods_to_probabilities(periods, tail="upper")

    assert lower == pytest.approx([0.1, 0.01])
    assert upper == pytest.approx([0.9, 0.99])
    # Same period, opposite tail: probabilities are complements of each other.
    assert lower == pytest.approx([1.0 - value for value in upper])


def test_return_periods_reject_unknown_tail() -> None:
    with pytest.raises(ValueError, match="tail"):
        return_periods_to_probabilities([10], tail="sideways")  # type: ignore[arg-type]


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
            "cell_index": [point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)],
            "horizon": [2030],
            "pathway": ["asset-scenario"],
            "curve_location": [1_000.0],
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

    encoded = (pq.read_schema(output).metadata or {})[PORTFOLIO_METADATA_KEY.encode()]
    metadata = json.loads(encoded)
    assert metadata["value_unit"] == "metres"
    assert metadata["value_semantics"] == "flood depth"
    assert metadata["return_periods"][1] == {
        "column": "value_rp100",
        "probability": 0.99,
        "return_period": 100.0,
    }


def test_portfolio_warns_when_flood_period_exceeds_source_support(
    tmp_path: Path,
) -> None:
    source = tmp_path / "supported-hazard.parquet"
    write_hazard_dataset(
        _table(_row()),
        source,
        _metadata(return_period_support=(10.0, 500.0)),
        max_workers=1,
    )
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "cell_index": pa.array(
                [_row()["cell_index"]],
                type=pa.uint64(),
            ),
        }
    )

    with pytest.warns(ReturnPeriodExtrapolationWarning, match="1000.0"):
        (
            HazardDataset.local(source)
            .for_assets(assets)
            .return_periods([250, 1000])
            .write_parquet(
                tmp_path / "extrapolated.parquet",
                execution=ExecutionOptions(max_workers=1),
            )
        )


def test_evaluate_portfolio_applies_inline_event_impact_and_metadata(
    tmp_path: Path,
) -> None:
    cell = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    provider = _provider(tmp_path, _row(cell_index=cell))
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "cell_index": pa.array([cell], type=pa.uint64()),
        }
    )
    output = tmp_path / "inline-impact.parquet"

    (
        HazardDataset(provider)
        .for_assets(assets)
        .return_periods([25, 100])
        .impact(
            lambda exposure: np.clip(exposure / 2.0, 0.0, 4.0),
            name="depth_damage_ratio",
            value_unit="fraction",
            value_semantics="damage ratio",
        )
        .write_parquet(output)
    )

    probabilities = return_periods_to_probabilities([25, 100])
    exposure = distribution_from_hazard_row(_row()).quantiles(probabilities)
    row = pq.read_table(output).to_pylist()[0]
    assert [row["value_rp25"], row["value_rp100"]] == pytest.approx(
        np.clip(exposure / 2.0, 0.0, 4.0)
    )
    encoded = (pq.read_schema(output).metadata or {})[PORTFOLIO_METADATA_KEY.encode()]
    metadata = json.loads(encoded)
    assert metadata["value_unit"] == "fraction"
    assert metadata["value_semantics"] == "damage ratio"
    assert metadata["interpretation"] == "event_aligned"
    assert metadata["source_value_unit"] == "metres"
    assert metadata["source_value_semantics"] == "flood depth"
    assert metadata["impact"] == {
        "context_columns": {},
        "name": "depth_damage_ratio",
        "type": "CallableImpact",
    }


def test_evaluate_portfolio_preserves_decreasing_event_order(
    tmp_path: Path,
) -> None:
    cell = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    provider = _provider(tmp_path, _row(cell_index=cell))
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "cell_index": pa.array([cell], type=pa.uint64()),
        }
    )
    output = tmp_path / "decreasing-impact.parquet"

    (
        HazardDataset(provider)
        .for_assets(assets)
        .return_periods([25, 100])
        .impact(
            LinearImpact(
                slope=-1.0,
                intercept=20.0,
                minimum=None,
                maximum=None,
            ),
            name="remaining_capacity",
            value_unit="units",
            value_semantics="remaining capacity",
        )
        .write_parquet(
            output,
            execution=ExecutionOptions(max_workers=1),
        )
    )

    row = pq.read_table(output).to_pylist()[0]
    assert row["value_rp25"] > row["value_rp100"]


def test_evaluate_portfolio_builds_complete_context_from_hidden_asset_columns(
    tmp_path: Path,
) -> None:
    cell_a = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    cell_b = point_to_cell(7.5, 51.0, H3_RESOLUTION)
    provider = _provider(
        tmp_path,
        _row(cell_index=cell_a, source_id="a"),
        _row(cell_index=cell_b, source_id="b"),
    )
    assets = pa.table(
        {
            "asset_id": ["asset-a", "asset-b"],
            "cell_index": pa.array([cell_a, cell_b], type=pa.uint64()),
            "asset_type": ["warehouse", "office"],
            "baseline": [2.0, 3.0],
        }
    )
    output = tmp_path / "context-impact.parquet"

    (
        HazardDataset(provider)
        .for_assets(
            AssetPortfolio(
                assets,
                location=CellColumn(),
                passthrough_columns=(),
            )
        )
        .return_periods([100])
        .impact(
            ContextAwareImpact(),
            context=ImpactContextColumns(
                building_type="asset_type",
                historic_mean="baseline",
            ),
            name="context_impact",
            value_unit="units",
            value_semantics="context-adjusted impact",
        )
        .write_parquet(
            output,
            execution=ExecutionOptions(max_workers=1),
        )
    )

    rows = {row["asset_id"]: row for row in pq.read_table(output).to_pylist()}
    exposure = float(distribution_from_hazard_row(_row()).quantiles([0.99])[0])
    assert rows["asset-a"]["value_rp100"] == pytest.approx(
        exposure + 2.0 + 100.0 + 10.0 + cell_a % 10
    )
    assert rows["asset-b"]["value_rp100"] == pytest.approx(
        exposure + 3.0 + 10.0 + cell_b % 10
    )
    assert "asset_type" not in rows["asset-a"]
    metadata = json.loads(
        (pq.read_schema(output).metadata or {})[PORTFOLIO_METADATA_KEY.encode()]
    )
    assert metadata["impact"]["context_columns"] == {
        "building_type": "asset_type",
        "historic_mean": "baseline",
    }


def test_evaluate_portfolio_rejects_parallel_inline_lambda(tmp_path: Path) -> None:
    cell = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    provider = _provider(tmp_path, _row(cell_index=cell))
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "cell_index": pa.array([cell], type=pa.uint64()),
        }
    )

    with pytest.raises(ValueError, match="not picklable"):
        (
            HazardDataset(provider)
            .for_assets(assets)
            .return_periods([100])
            .impact(
                lambda exposure: exposure,
                name="identity",
                value_unit="metres",
                value_semantics="flood depth",
            )
            .write_parquet(
                tmp_path / "parallel-lambda.parquet",
                execution=ExecutionOptions(max_workers=2),
            )
        )


def test_evaluate_portfolio_runs_picklable_impact_in_process_pool(
    tmp_path: Path,
) -> None:
    cell_a = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    cell_b = point_to_cell(7.5, 51.0, H3_RESOLUTION)
    provider = _provider(
        tmp_path,
        _row(cell_index=cell_a, source_id="a"),
        _row(cell_index=cell_b, source_id="b"),
    )
    assets = pa.table(
        {
            "asset_id": ["asset-a", "asset-b"],
            "cell_index": pa.array([cell_a, cell_b], type=pa.uint64()),
        }
    )
    output = tmp_path / "parallel-impact.parquet"

    (
        HazardDataset(provider)
        .for_assets(assets)
        .return_periods([100])
        .impact(
            LinearImpact(slope=0.5),
            name="linear_impact",
            value_unit="fraction",
            value_semantics="linear impact",
        )
        .write_parquet(
            output,
            execution=ExecutionOptions(max_workers=2, chunk_rows=1),
        )
    )

    assert pq.read_table(output).num_rows == 2


def test_evaluate_portfolio_validates_impact_context_columns(
    tmp_path: Path,
) -> None:
    cell = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    provider = _provider(tmp_path, _row(cell_index=cell))
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "cell_index": pa.array([cell], type=pa.uint64()),
        }
    )

    with pytest.raises(ValueError, match="missing columns"):
        (
            HazardDataset(provider)
            .for_assets(assets)
            .return_periods([100])
            .impact(
                LinearImpact(slope=1.0),
                context=ImpactContextColumns(country="country"),
                name="linear_impact",
                value_unit="fraction",
                value_semantics="linear impact",
            )
            .write_parquet(
                tmp_path / "missing-context.parquet",
                execution=ExecutionOptions(max_workers=1),
            )
        )


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
            "cell_index": [point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)],
            "horizon": [2030],
            "pathway": ["asset-scenario"],
            "value_rp100": [-1.0],
            "curve_kind": ["fitted"],
            "curve_type": ["gumbel_r"],
            "curve_shape": [None],
            "curve_location": [1_000.0],
            "curve_scale": [1.0],
            "curve_atom_probability": [None],
            "curve_atom_location": [None],
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
    assert row["cell_index"] == point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
    assert row["horizon"] == 2050
    assert row["pathway"] == "ssp585"
    expected = distribution_from_hazard_row(_table(_row())).quantiles([0.99])[0]
    assert row["value_rp100"] == pytest.approx(expected)


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


def test_evaluate_point_portfolio_rejects_overlapping_interiors(
    tmp_path: Path,
) -> None:
    geometry = Polygon(
        [
            (LONGITUDE - 0.01, LATITUDE - 0.01),
            (LONGITUDE + 0.01, LATITUDE - 0.01),
            (LONGITUDE + 0.01, LATITUDE + 0.01),
            (LONGITUDE - 0.01, LATITUDE + 0.01),
        ]
    ).wkb
    provider = _provider(
        tmp_path,
        _row(source_id="a", source_geometry=geometry),
        _row(source_id="b", source_geometry=geometry),
    )
    assets = pa.table(
        {
            "asset_id": ["asset-a"],
            "longitude": [LONGITUDE],
            "latitude": [LATITUDE],
        }
    )

    with pytest.raises(LookupError, match="multiple source curves"):
        (
            HazardDataset(provider)
            .for_assets(assets)
            .return_periods([100])
            .write_parquet(
                tmp_path / "ambiguous-point.parquet",
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


def test_evaluate_impact_empty_assets_writes_output_metadata(tmp_path: Path) -> None:
    provider = _provider(tmp_path, _row())
    assets = pa.table(
        {
            "asset_id": pa.array([], type=pa.string()),
            "cell_index": pa.array([], type=pa.uint64()),
        }
    )
    output = tmp_path / "empty-impact.parquet"

    result = (
        HazardDataset(provider)
        .for_assets(assets)
        .return_periods([100])
        .impact(
            LinearImpact(slope=0.5),
            name="linear_impact",
            value_unit="fraction",
            value_semantics="linear impact",
        )
        .write_parquet(
            output,
            execution=ExecutionOptions(max_workers=1),
        )
    )

    assert result.row_count == 0
    assert pq.read_table(output).column_names[-1] == "value_rp100"
    metadata = json.loads(
        (pq.read_schema(output).metadata or {})[PORTFOLIO_METADATA_KEY.encode()]
    )
    assert metadata["interpretation"] == "event_aligned"
