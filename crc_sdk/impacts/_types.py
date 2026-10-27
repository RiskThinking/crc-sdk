"""Strict scalar types and the frozen base model for impact documents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    StringConstraints,
)

IdentifierPattern = r"^\S(?:[\s\S]*\S)?$"

Identifier = Annotated[
    str,
    Strict(),
    StringConstraints(min_length=1, pattern=IdentifierPattern),
]
Text = Annotated[str, Strict(), StringConstraints(min_length=1)]
Sha256 = Annotated[str, Strict(), StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ContentHash = Annotated[
    str,
    Strict(),
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
H3Cell = Annotated[str, Strict(), StringConstraints(pattern=r"^[0-9a-f]{15}$")]


def _positive_zero(value: float) -> float:
    """Normalize IEEE negative zero to positive zero."""
    return value + 0.0


FiniteFloat = Annotated[
    float,
    Strict(),
    Field(allow_inf_nan=False),
    AfterValidator(_positive_zero),
]
NonNegativeFloat = Annotated[
    float,
    Strict(),
    Field(allow_inf_nan=False, ge=0),
    AfterValidator(_positive_zero),
]
PositiveFloat = Annotated[
    float,
    Strict(),
    Field(allow_inf_nan=False, gt=0),
    AfterValidator(_positive_zero),
]


class StrictFrozenModel(BaseModel):
    """Reject unknown fields and keep instances immutable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        revalidate_instances="never",
    )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> dict[str, Any]:
        schema: dict[str, Any] = handler(core_schema)
        if (
            schema.get("type") == "object"
            and "properties" in schema
            and "required" not in schema
        ):
            schema["required"] = []
        return schema


def thaw(value: Any) -> Any:
    """Return plain JSON types, keeping unknown mapping keys."""
    if isinstance(value, BaseModel):
        return thaw(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    return value
