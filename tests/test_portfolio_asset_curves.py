"""PortfolioEvaluation.asset_curves(): each asset's matched canonical curve rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from crc_sdk.connectors import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.geometry import point_to_cell
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance, TemporalWindow
from crc_sdk.workflows import (
    AssetPortfolio,
    CellColumn,
    ExecutionOptions,
    HazardDataset,
    PointColumns,
    curve_parameters_from_row,
    distribution_from_hazard_row,
    return_periods_to_probabilities,
)
from crc_sdk.workflows.distributions import HorizonExtrapolationWarning

LONGITUDE = 6.9603
LATITUDE = 50.9375
H3_RESOLUTION = 7
CELL_A = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)
CELL_B = point_to_cell(7.5, 51.0, H3_RESOLUTION)
CURVE_COLUMNS = [
    "curve_kind",
    "curve_type",
    "curve_shape",
    "curve_location",
    "curve_scale",
    "curve_atom_probability",
    "curve_atom_location",
    "curve_probabilities",
    "curve_values",
]
KEY_COLUMNS = [
    "cell_index",
    "hazard_name",
    "horizon",
    "pathway",
    "source_id",
    "spatial_match",
]


def _metadata(provider: str = "fixture", **update: Any) -> HazardDatasetMetadata:
    metadata = HazardDatasetMetadata(
        h3_resolution=H3_RESOLUTION,
        value_unit="metres",
        value_semantics="flood depth",
        producer="tests",
        source=SourceProvenance(provider=provider, dataset="flood"),
        creation_version="1",
    )
    return metadata.model_copy(update=update)


def _row(
    *,
    cell_index: int = CELL_A,
    source_id: str = "source-a",
    source_geometry: bytes | None = None,
    horizon: int = 2050,
    pathway: str = "ssp585",
    curve_kind: str = "fitted",
    curve_location: float = 2.0,
) -> dict[str, Any]:
    no_data = curve_kind == "no_data"
    return {
        "cell_index": cell_index,
        "source_id": source_id,
        "source_geometry": source_geometry,
        "hazard_name": "flood",
        "horizon": horizon,
        "pathway": pathway,
        "curve_kind": curve_kind,
        "curve_type": "insufficient_informative_support" if no_data else "gumbel_r",
        "curve_shape": None,
        "curve_location": None if no_data else curve_location,
        "curve_scale": None if no_data else 3.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


def _dataset(
    tmp_path: Path,
    *rows: dict[str, Any],
    metadata: HazardDatasetMetadata | None = None,
    name: str = "hazard.parquet",
) -> HazardDataset:
    stated = metadata or _metadata()
    source = tmp_path / name
    table = pa.Table.from_pylist(list(rows), schema=hazard_arrow_schema(stated))
    write_hazard_dataset(table, source, stated, max_workers=1)
    return HazardDataset.local(source)


def _cell_assets(*cells: int, ids: tuple[str, ...] = ("asset-b", "asset-a")) -> Any:
    return pa.table(
        {
            "asset_id": list(ids[: len(cells)]),
            "cell_index": pa.array(list(cells), type=pa.uint64()),
            "sector": ["energy", "finance"][: len(cells)],
        }
    )


def test_cell_assets_get_one_curve_row_per_scenario_sorted(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        _row(cell_index=CELL_A, source_id="a", curve_location=1.0),
        _row(cell_index=CELL_B, source_id="b", curve_location=4.0),
        _row(cell_index=CELL_A, source_id="a-2080", horizon=2080, curve_location=6.0),
        _row(cell_index=CELL_B, source_id="b-2080", horizon=2080, curve_location=7.0),
    )
    request = dataset.for_assets(_cell_assets(CELL_B, CELL_A))

    curves = request.asset_curves()

    assert isinstance(curves, pa.Table)
    assert curves.column_names == ["asset_id", "sector", *KEY_COLUMNS, *CURVE_COLUMNS]
    keys = [
        (row["asset_id"], row["horizon"], row["source_id"])
        for row in curves.to_pylist()
    ]
    assert keys == [
        ("asset-a", 2050, "a"),
        ("asset-a", 2080, "a-2080"),
        ("asset-b", 2050, "b"),
        ("asset-b", 2080, "b-2080"),
    ]
    assert curves["cell_index"].type == pa.uint64()
    assert set(curves["spatial_match"].to_pylist()) == {"h3_cell"}
    assert curves["sector"].to_pylist() == ["finance", "finance", "energy", "energy"]
    first = curves.to_pylist()[0]
    assert curve_parameters_from_row(first) == curve_parameters_from_row(
        _row(cell_index=CELL_A, source_id="a", curve_location=1.0)
    )
    # The selection applies.
    selected = request.select(horizons=[2080]).asset_curves()
    assert selected["source_id"].to_pylist() == ["a-2080", "b-2080"]


def test_curve_rows_give_the_values_write_parquet_writes(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        _row(cell_index=CELL_A, source_id="a", curve_location=1.0),
        _row(cell_index=CELL_B, source_id="b", curve_location=4.0),
    )
    request = dataset.for_assets(_cell_assets(CELL_B, CELL_A)).return_periods([25, 100])
    output = tmp_path / "evaluated.parquet"
    request.write_parquet(output, execution=ExecutionOptions(max_workers=1))
    written = {row["asset_id"]: row for row in pq.read_table(output).to_pylist()}

    curves = request.asset_curves()

    probabilities = return_periods_to_probabilities([25, 100])
    assert curves.num_rows == len(written) == 2
    for row in curves.to_pylist():
        expected = written[row["asset_id"]]
        values = distribution_from_hazard_row(row).quantiles(probabilities)
        assert list(values) == pytest.approx(
            [expected["value_rp25"], expected["value_rp100"]], rel=1e-12
        )
        for name in KEY_COLUMNS:
            assert row[name] == expected[name]


def test_point_assets_are_refined_by_geometry_and_keep_coordinates(
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
    dataset = _dataset(
        tmp_path,
        _row(source_id="match", source_geometry=containing.wkb),
        _row(source_id="other", source_geometry=elsewhere.wkb),
    )
    assets = pa.table({"site": ["asset-a"], "lon": [LONGITUDE], "lat": [LATITUDE]})
    portfolio = AssetPortfolio(
        assets, id_column="site", location=PointColumns("lon", "lat")
    )

    curves = dataset.for_assets(portfolio).asset_curves()

    assert curves.column_names == ["site", "lon", "lat", *KEY_COLUMNS, *CURVE_COLUMNS]
    row = curves.to_pylist()[0]
    assert curves.num_rows == 1
    assert row["source_id"] == "match"
    assert row["spatial_match"] == "exact_geometry"
    assert (row["lon"], row["lat"]) == (LONGITUDE, LATITUDE)
    assert row["cell_index"] == CELL_A


def test_no_data_rows_are_returned_as_they_are(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        _row(cell_index=CELL_A, source_id="dry", curve_kind="no_data"),
        _row(cell_index=CELL_B, source_id="b"),
    )

    curves = dataset.for_assets(_cell_assets(CELL_B, CELL_A)).asset_curves()

    rows = {row["asset_id"]: row for row in curves.to_pylist()}
    assert rows["asset-a"]["curve_kind"] == "no_data"
    assert rows["asset-a"]["curve_type"] == "insufficient_informative_support"
    assert rows["asset-a"]["curve_location"] is None
    assert curve_parameters_from_row(rows["asset-a"]).curve_kind == "no_data"
    assert rows["asset-b"]["curve_kind"] == "fitted"


def test_missing_and_multiple_matches_raise_as_the_join_does(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, _row(source_id="a"), _row(source_id="b"))
    with pytest.raises(LookupError, match="multiple source curves"):
        dataset.for_assets(_cell_assets(CELL_A)).asset_curves()
    other = _dataset(tmp_path, _row(), name="other.parquet")
    with pytest.raises(LookupError, match="missing hazard curves"):
        other.for_assets(_cell_assets(point_to_cell(0.0, 0.0, 7))).asset_curves()
    repeated = pa.table(
        {
            "asset_id": ["a", "a"],
            "cell_index": pa.array([CELL_A, CELL_A], type=pa.uint64()),
        }
    )
    with pytest.raises(ValueError, match="must be unique"):
        other.for_assets(repeated).asset_curves()


def test_jrc_cell_inputs_with_several_curves_raise_where_write_parquet_aggregates(
    tmp_path: Path,
) -> None:
    dataset = _dataset(
        tmp_path,
        _row(source_id="tile-1", curve_location=1.0),
        _row(source_id="tile-2", curve_location=3.0),
        metadata=_metadata(provider="jrc"),
    )
    request = dataset.for_assets(_cell_assets(CELL_A))
    output = tmp_path / "aggregated.parquet"
    result = request.return_periods([100]).write_parquet(
        output, execution=ExecutionOptions(max_workers=1)
    )
    assert result.row_count == 1
    assert pq.read_table(output)["spatial_match"].to_pylist() == ["h3_cell_aggregate"]

    with pytest.raises(LookupError, match="multiple source curves"):
        request.asset_curves()


def test_empty_assets_give_a_typed_empty_table(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, _row())
    assets = pa.table(
        {
            "asset_id": pa.array([], type=pa.string()),
            "cell_index": pa.array([], type=pa.uint64()),
        }
    )

    curves = dataset.for_assets(
        AssetPortfolio(assets, location=CellColumn())
    ).asset_curves()

    assert curves.num_rows == 0
    assert curves.column_names == ["asset_id", *KEY_COLUMNS, *CURVE_COLUMNS]


@pytest.mark.parametrize("schema_version", ["1.0", "1.1"])
def test_legacy_schemas_return_their_own_curve_columns(
    tmp_path: Path, schema_version: Literal["1.0", "1.1"]
) -> None:
    dataset = _dataset(
        tmp_path, _row(), metadata=_metadata(schema_version=schema_version)
    )

    curves = dataset.for_assets(_cell_assets(CELL_A)).asset_curves()

    assert "curve_probabilities" not in curves.column_names
    assert curves.column_names[-1] == "curve_atom_location"
    assert distribution_from_hazard_row(curves).quantiles([0.5])[0] == pytest.approx(
        distribution_from_hazard_row(_row()).quantiles([0.5])[0]
    )


def test_a_given_connection_is_used_and_left_open(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, _row())
    con = duckdb.connect()
    try:
        curves = dataset.for_assets(_cell_assets(CELL_A)).asset_curves(
            execution=ExecutionOptions(connection=con)
        )
        assert curves.num_rows == 1
        # The connection stays open, and the asset relation is unregistered.
        assert con.execute("SELECT 1").fetchone() == (1,)
        names = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        assert not [name for name in names if name.startswith("_crc_assets_")]
    finally:
        con.close()


def test_asset_curves_writes_nothing_and_warns_outside_the_window(
    tmp_path: Path,
) -> None:
    window = TemporalWindow(kind="window", start_year=2040, end_year=2060)
    dataset = _dataset(
        tmp_path, _row(horizon=2080), metadata=_metadata(temporal_window=window)
    )
    before = sorted(tmp_path.rglob("*"))
    request = dataset.for_assets(_cell_assets(CELL_A)).select(horizons=[2080])

    with pytest.warns(HorizonExtrapolationWarning, match="outside"):
        curves = request.asset_curves()

    assert curves.num_rows == 1
    assert sorted(tmp_path.rglob("*")) == before
