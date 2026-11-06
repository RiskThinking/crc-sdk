"""What a parametric cover pays on: the index definition and its location."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, field_validator

from crc_sdk.impacts._types import FiniteFloat, Identifier
from crc_sdk.parametric.errors import CoverDefinitionError
from crc_sdk.parametric.parameters import _DefinitionModel

__all__ = [
    "AtAsset",
    "AtCell",
    "AtPoint",
    "AtStation",
    "IndexDefinition",
    "Location",
    "Tail",
]

Tail = Literal["upper", "lower"]
"""Which side of the index is severe: larger values (`"upper"`) or smaller."""

_H3_CELL = re.compile(r"^[0-9a-f]{15}$")


class AtAsset(_DefinitionModel):
    """The index is read at the cover's asset."""

    kind: Literal["asset"] = "asset"

    def __init__(self) -> None:
        super().__init__()


class AtPoint(_DefinitionModel):
    """The index is read at a WGS84 point, for example a reference site."""

    kind: Literal["point"] = "point"
    longitude: Annotated[FiniteFloat, Field(ge=-180.0, le=180.0)]
    latitude: Annotated[FiniteFloat, Field(ge=-90.0, le=90.0)]
    name: Identifier | None = None

    def __init__(
        self, longitude: float, latitude: float, name: str | None = None
    ) -> None:
        super().__init__(longitude=longitude, latitude=latitude, name=name)


class AtStation(_DefinitionModel):
    """The index is read at an observation station."""

    kind: Literal["station"] = "station"
    station_id: Identifier

    def __init__(self, station_id: str) -> None:
        super().__init__(station_id=station_id)


class AtCell(_DefinitionModel):
    """The index is read in one H3 cell.

    `cell_index` takes the canonical unsigned integer (the `cell_index`
    column of a canonical dataset) or its 15-digit hexadecimal string. The
    document stores the string, because JSON numbers are binary64 floats and
    cannot hold every 64-bit cell index.
    """

    kind: Literal["cell"] = "cell"
    cell_index: str

    def __init__(self, cell_index: int | str) -> None:
        super().__init__(cell_index=cell_index)

    @field_validator("cell_index", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise CoverDefinitionError("cell_index must be an H3 cell, not a bool")
        if isinstance(value, int | np.integer):
            if not 0 < int(value) < 2**64:
                raise CoverDefinitionError(
                    "cell_index must be an unsigned 64-bit H3 cell"
                )
            value = format(int(value), "x")
        if not isinstance(value, str) or not _H3_CELL.fullmatch(value.lower()):
            raise CoverDefinitionError(
                f"cell_index must be an H3 cell (15 hexadecimal digits), got {value!r}"
            )
        return value.lower()


Location = Annotated[
    AtAsset | AtPoint | AtStation | AtCell, Field(discriminator="kind")
]
"""Where the index is read: `AtAsset`, `AtPoint`, `AtStation` or `AtCell`."""

_LOCATIONS: dict[str, type[AtAsset | AtPoint | AtStation | AtCell]] = {
    "asset": AtAsset,
    "point": AtPoint,
    "station": AtStation,
    "cell": AtCell,
}


def _location_from_document(document: Mapping[str, Any]) -> Any:
    kind = document.get("kind")
    location = _LOCATIONS.get(kind) if isinstance(kind, str) else None
    if location is None:
        raise CoverDefinitionError(
            f"location kind must be one of {sorted(_LOCATIONS)}, got {kind!r}"
        )
    return location._validated(
        {key: value for key, value in document.items() if key != "kind"}
    )


class IndexDefinition(_DefinitionModel):
    """The index a cover pays on.

    `hazard_name` and `unit` name the hazard and its unit, as a canonical
    dataset's `hazard_name` and `value_unit` do. `tail` says which side is
    severe: `"upper"` for hazards where larger values are worse (rainfall,
    wind), `"lower"` where smaller values are worse (a drought index); it
    follows the dataset's `return_period_tail`. `location` says where the
    index is read (`AtAsset()` by default). `aggregation` says how one year's
    values become the year's index, for example `"annual_maximum"`.
    `source` optionally names the data source.

    Raises `CoverDefinitionError` for empty names or units, an unknown tail
    or an invalid location.
    """

    hazard_name: Identifier
    unit: Identifier
    tail: Tail = "upper"
    location: Location = AtAsset()
    aggregation: Identifier
    source: Identifier | None = None

    def __init__(
        self,
        hazard_name: str,
        unit: str,
        *,
        tail: Tail = "upper",
        location: AtAsset | AtPoint | AtStation | AtCell = AtAsset(),
        aggregation: str,
        source: str | None = None,
    ) -> None:
        super().__init__(
            hazard_name=hazard_name,
            unit=unit,
            tail=tail,
            location=location,
            aggregation=aggregation,
            source=source,
        )

    @field_validator("location", mode="before")
    @classmethod
    def _read_location(cls, value: Any) -> Any:
        return _location_from_document(value) if isinstance(value, Mapping) else value

    def to_document(self) -> dict[str, Any]:
        """Return the definition as plain JSON types, as the cover stores it."""
        document: dict[str, Any] = self.model_dump(mode="json")
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Read a definition written by `to_document`."""
        return cls._validated(document)
