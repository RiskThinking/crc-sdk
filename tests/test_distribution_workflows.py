from __future__ import annotations

from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, HurdleDistribution
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from crc_sdk.connectors import hazard_arrow_schema
from crc_sdk.geometry import point_to_cell
from crc_sdk.types import (
    HazardDatasetMetadata,
    HazardQuery,
    SourceProvenance,
)
from crc_sdk.workflows import (
    curve_parameters_from_row,
    distribution_from_hazard_row,
    sample_hazard_at_cell,
    sample_hazard_at_point,
    sample_hazard_row,
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
    source_id: str = "source-a",
    source_geometry: bytes | None = None,
    horizon: int = 2050,
    pathway: str = "ssp585",
    curve_kind: str = "fitted",
) -> dict[str, Any]:
    hurdle = curve_kind == "hurdle"
    return {
        "cell_index": point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION),
        "source_id": source_id,
        "source_geometry": source_geometry,
        "hazard_name": "flood",
        "horizon": horizon,
        "pathway": pathway,
        "curve_kind": curve_kind,
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 2.0,
        "curve_scale": 3.0,
        "curve_atom_probability": 0.5 if hurdle else None,
        "curve_atom_location": 0.0 if hurdle else None,
    }


def _table(*rows: dict[str, Any]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=hazard_arrow_schema())


class _Provider:
    def __init__(self, table: pa.Table):
        self.table = table
        self.last_query: HazardQuery | None = None

    def list_hazards(self) -> tuple[str, ...]:
        return ("flood",)

    def metadata(self, hazard_name: str) -> HazardDatasetMetadata:
        if hazard_name != "flood":
            raise LookupError(hazard_name)
        return _metadata()

    def read(self, query: HazardQuery) -> pa.Table:
        self.last_query = query
        rows = [
            row
            for row in self.table.to_pylist()
            if row["hazard_name"] == query.hazard_name
            and (query.horizon is None or row["horizon"] == query.horizon)
            and (query.pathway is None or row["pathway"] == query.pathway)
            and (
                query.cell_index is None
                or row["cell_index"] == query.cell_index
            )
        ]
        return _table(*rows)


def test_row_utilities_reconstruct_fitted_and_hurdle_curves() -> None:
    fitted = distribution_from_hazard_row(_row())
    hurdle = distribution_from_hazard_row(_table(_row(curve_kind="hurdle")))

    assert isinstance(fitted, FittedDistribution)
    assert isinstance(hurdle, HurdleDistribution)
    assert curve_parameters_from_row(_row()).curve_type == "gumbel_r"


def test_sample_hazard_row_is_seeded_and_supports_table_row_index() -> None:
    table = _table(_row(source_id="a"), _row(source_id="b", horizon=2080))

    first = sample_hazard_row(table, row_index=1, size=32, seed=7)
    second = sample_hazard_row(table, row_index=1, size=32, seed=7)

    assert first.samples.shape == (32,)
    assert (first.samples == second.samples).all()
    assert first.parameters.curve_location == 2.0


@pytest.mark.parametrize("size", [0, -1])
def test_sample_hazard_row_rejects_non_positive_size(size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_hazard_row(_row(), size=size)


def test_sample_hazard_row_rejects_non_integer_size() -> None:
    with pytest.raises(TypeError, match="integer"):
        sample_hazard_row(_row(), size=1.5)  # type: ignore[arg-type]


def test_sample_hazard_at_cell_filters_and_returns_context() -> None:
    provider = _Provider(_table(_row(curve_kind="hurdle")))
    cell_index = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)

    result = sample_hazard_at_cell(
        provider,
        "flood",
        cell_index,
        horizon=2050,
        pathway="ssp585",
        size=16,
        seed=11,
    )

    assert result.samples.shape == (16,)
    assert result.cell_index == cell_index
    assert result.source_id == "source-a"
    assert result.value_unit == "metres"
    assert isinstance(result.distribution, HurdleDistribution)
    assert provider.last_query == HazardQuery(
        hazard_name="flood",
        horizon=2050,
        pathway="ssp585",
        cell_index=cell_index,
    )


def test_sample_hazard_at_cell_rejects_no_match() -> None:
    provider = _Provider(_table(_row()))
    other_cell = point_to_cell(0.0, 0.0, H3_RESOLUTION)

    with pytest.raises(LookupError, match=f"cell {other_cell}"):
        sample_hazard_at_cell(provider, "flood", other_cell)


def test_sample_hazard_at_cell_rejects_ambiguous_matches() -> None:
    provider = _Provider(
        _table(
            _row(source_id="source-a"),
            _row(source_id="source-b"),
        )
    )
    cell_index = point_to_cell(LONGITUDE, LATITUDE, H3_RESOLUTION)

    with pytest.raises(LookupError, match="matches multiple 'flood' curves"):
        sample_hazard_at_cell(provider, "flood", cell_index)


def test_sample_hazard_at_point_refines_geometry_and_returns_context() -> None:
    containing = Polygon(
        [
            (LONGITUDE - 0.01, LATITUDE - 0.01),
            (LONGITUDE + 0.01, LATITUDE - 0.01),
            (LONGITUDE + 0.01, LATITUDE + 0.01),
            (LONGITUDE - 0.01, LATITUDE + 0.01),
        ]
    )
    provider = _Provider(_table(_row(source_geometry=containing.wkb)))

    result = sample_hazard_at_point(
        provider,
        "flood",
        LONGITUDE,
        LATITUDE,
        horizon=2050,
        pathway="ssp585",
        size=16,
        seed=11,
    )

    assert result.samples.shape == (16,)
    assert result.spatial_match == "exact_geometry"
    assert result.value_unit == "metres"
    assert result.value_semantics == "flood depth"
    assert result.source_id == "source-a"
    assert provider.last_query is not None
    assert provider.last_query.cell_index == result.cell_index


def test_sample_hazard_at_point_reports_geometry_free_cell_match() -> None:
    result = sample_hazard_at_point(
        _Provider(_table(_row())),
        "flood",
        LONGITUDE,
        LATITUDE,
        size=8,
        seed=3,
    )

    assert result.spatial_match == "h3_cell"


def test_sample_hazard_at_point_rejects_no_exact_match() -> None:
    elsewhere = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    provider = _Provider(_table(_row(source_geometry=elsewhere.wkb)))

    with pytest.raises(LookupError, match="no 'flood' curve"):
        sample_hazard_at_point(
            provider,
            "flood",
            LONGITUDE,
            LATITUDE,
        )


def test_sample_hazard_at_point_rejects_ambiguous_matches() -> None:
    provider = _Provider(
        _table(
            _row(source_id="source-a"),
            _row(source_id="source-b"),
        )
    )

    with pytest.raises(LookupError, match="multiple 'flood' curves"):
        sample_hazard_at_point(
            provider,
            "flood",
            LONGITUDE,
            LATITUDE,
        )


@pytest.mark.parametrize(
    ("longitude", "latitude", "message"),
    [
        (181.0, 0.0, "longitude"),
        (0.0, 91.0, "latitude"),
    ],
)
def test_sample_hazard_at_point_validates_coordinates(
    longitude: float, latitude: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_hazard_at_point(
            _Provider(_table(_row())),
            "flood",
            longitude,
            latitude,
        )
