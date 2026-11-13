"""Index definitions and their locations."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from crc_sdk.parametric import (
    AtAsset,
    AtCell,
    AtPoint,
    AtStation,
    CoverDefinitionError,
    IndexDefinition,
)

CELL = 0x8A2A1072B59FFFF


def test_defaults() -> None:
    index = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
    assert index.tail == "upper"
    assert index.location == AtAsset()
    assert index.source is None
    assert index.to_document() == {
        "aggregation": "annual_maximum",
        "hazard_name": "rx1day",
        "location": {"kind": "asset"},
        "source": None,
        "tail": "upper",
        "unit": "mm/day",
    }


@pytest.mark.parametrize(
    ("location", "document"),
    [
        pytest.param(AtAsset(), {"kind": "asset"}, id="asset"),
        pytest.param(
            AtPoint(6.96, 50.94, "cologne"),
            {"kind": "point", "latitude": 50.94, "longitude": 6.96, "name": "cologne"},
            id="point",
        ),
        pytest.param(
            AtPoint(-180, 90),
            {"kind": "point", "latitude": 90.0, "longitude": -180.0, "name": None},
            id="point-edges",
        ),
        pytest.param(
            AtStation("USW00094728"),
            {"kind": "station", "station_id": "USW00094728"},
            id="station",
        ),
        pytest.param(
            AtCell(CELL), {"kind": "cell", "cell_index": "8a2a1072b59ffff"}, id="cell"
        ),
    ],
)
def test_locations_round_trip(location: Any, document: dict[str, Any]) -> None:
    index = IndexDefinition(
        "spei",
        "1",
        tail="lower",
        location=location,
        aggregation="annual_minimum",
        source="era5",
    )
    assert index.to_document()["location"] == document
    assert IndexDefinition.from_document(index.to_document()) == index


def test_a_cell_takes_the_integer_or_the_string() -> None:
    assert AtCell(CELL) == AtCell("8a2a1072b59ffff") == AtCell("8A2A1072B59FFFF")
    assert AtCell(CELL).cell_index == "8a2a1072b59ffff"


def test_a_location_may_be_given_as_its_document() -> None:
    document = {"kind": "point", "longitude": 1.0, "latitude": 2.0, "name": None}
    index = IndexDefinition(
        "rx1day",
        "mm/day",
        location=document,  # type: ignore[arg-type]
        aggregation="annual_maximum",
    )
    assert index.location == AtPoint(1.0, 2.0)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: AtPoint(181.0, 0.0), id="longitude"),
        pytest.param(lambda: AtPoint(0.0, -90.5), id="latitude"),
        pytest.param(lambda: AtPoint(float("nan"), 0.0), id="nan"),
        pytest.param(lambda: AtPoint(0.0, 0.0, ""), id="empty-name"),
        pytest.param(lambda: AtStation(" station"), id="station-whitespace"),
        pytest.param(lambda: AtStation(""), id="station-empty"),
        pytest.param(lambda: AtCell(5), id="cell-short"),
        pytest.param(lambda: AtCell(0), id="cell-zero"),
        pytest.param(lambda: AtCell(2**64), id="cell-too-large"),
        pytest.param(lambda: AtCell(True), id="cell-bool"),
        pytest.param(lambda: AtCell("8a2a1072b59fffg"), id="cell-not-hex"),
        pytest.param(lambda: AtCell(1.5), id="cell-float"),  # type: ignore[arg-type]
    ],
)
def test_invalid_locations_raise(build: Any) -> None:
    with pytest.raises(CoverDefinitionError):
        build()


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"hazard_name": ""}, id="empty-hazard"),
        pytest.param({"hazard_name": "rx1day "}, id="hazard-whitespace"),
        pytest.param({"unit": ""}, id="empty-unit"),
        pytest.param({"unit": 1}, id="unit-not-text"),
        pytest.param({"tail": "both"}, id="tail"),
        pytest.param({"aggregation": ""}, id="empty-aggregation"),
        pytest.param({"source": ""}, id="empty-source"),
        pytest.param({"location": "asset"}, id="location-text"),
        pytest.param({"location": {"kind": "region"}}, id="location-kind"),
        pytest.param({"location": {"kind": "point", "longitude": 1.0}}, id="point-doc"),
    ],
)
def test_invalid_index_definitions_raise(arguments: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "hazard_name": "rx1day",
        "unit": "mm/day",
        "aggregation": "annual_maximum",
    }
    values.update(arguments)
    with pytest.raises(CoverDefinitionError):
        IndexDefinition(
            values.pop("hazard_name"),
            values.pop("unit"),
            aggregation=values.pop("aggregation"),
            **values,
        )


@pytest.mark.parametrize(
    "document",
    [
        pytest.param({"hazard_name": "rx1day", "unit": "mm/day"}, id="missing"),
        pytest.param(
            {
                "aggregation": "annual_maximum",
                "hazard_name": "rx1day",
                "unit": "mm/day",
                "window": "1991-2020",
            },
            id="unknown",
        ),
        pytest.param(["rx1day"], id="not-an-object"),
    ],
)
def test_invalid_index_documents_raise(document: Any) -> None:
    with pytest.raises(CoverDefinitionError):
        IndexDefinition.from_document(document)


def test_index_definitions_are_frozen_and_hashable() -> None:
    index = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
    assert hash(index) == hash(
        IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
    )
    with pytest.raises(ValidationError, match="frozen"):
        index.unit = "mm"
