"""Payout parameters: fixed values and `Free` parameters with bounds.

A payout parameter is either a float, a fixed contract term, or `Free`, a
value a design chooses within bounds. A cover with any `Free` parameter is a
template: it has a document and a hash, but it pays nothing until its free
parameters are resolved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError, model_validator

from crc_sdk.impacts._types import FiniteFloat, StrictFrozenModel
from crc_sdk.parametric.errors import CoverDefinitionError

__all__ = ["Free"]


def _location(parts: Sequence[int | str]) -> str:
    text = ""
    for part in parts:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else str(part)
    return text


def _definition_error(error: ValidationError) -> CoverDefinitionError:
    """Turn a pydantic validation error into a `CoverDefinitionError`."""
    messages: list[str] = []
    for item in error.errors(include_url=False):
        location = _location(item["loc"])
        original = (item.get("ctx") or {}).get("error")
        message = (
            str(original) if isinstance(original, CoverDefinitionError) else item["msg"]
        )
        messages.append(f"{location}: {message}" if location else message)
    return CoverDefinitionError(f"invalid {error.title}: {'; '.join(messages)}")


class _DefinitionModel(StrictFrozenModel):
    """Frozen model whose validation failures raise `CoverDefinitionError`.

    Subclasses define `__init__` with positional parameters named as their
    fields. Pydantic calls that `__init__` with the fields as keywords
    whenever it validates a mapping, so documents are read through
    `_validated`, which checks the keys first.
    """

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as error:
            raise _definition_error(error) from error

    @classmethod
    def _validated(cls, data: Any) -> Self:
        """Build the model from a mapping of its fields, as a document has them."""
        name = cls.__name__
        if not isinstance(data, Mapping):
            raise CoverDefinitionError(f"a {name} document must be an object")
        fields = cls.model_fields
        unknown = sorted(str(key) for key in data if key not in fields)
        missing = [
            field
            for field, info in fields.items()
            if info.is_required() and field not in data
        ]
        if unknown or missing:
            raise CoverDefinitionError(
                f"a {name} document has unknown fields {unknown} "
                f"and lacks the fields {missing}"
            )
        return cls(**data)


def _real(value: Any) -> float | None:
    """Return `value` as a float if it is a real number (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


class Free(_DefinitionModel):
    """A payout parameter that a design chooses, within bounds.

    `lower` and `upper` bound the value; `upper=None` leaves it unbounded
    above. `grid` lists the candidate values a grid search tries: an integer
    gives that many evenly spaced values from `lower` to `upper` (both
    finite), and a sequence gives the values themselves, strictly increasing
    and within the bounds. A grid given as a count is stored as its values,
    so the cover document holds only numbers.

    The payout decides which bounds a parameter needs. Amounts (payout units)
    need `lower >= 0`; `upper` may be `None`, and then a limit or a budget
    bounds the design. Thresholds (index units) need a finite `upper` and a
    grid.

    Raises `CoverDefinitionError` for non-finite bounds, `lower > upper`, or
    an invalid grid.
    """

    lower: FiniteFloat = 0.0
    upper: FiniteFloat | None = None
    grid: tuple[FiniteFloat, ...] | None = None

    def __init__(
        self,
        lower: float = 0.0,
        upper: float | None = None,
        *,
        grid: int | np.integer[Any] | Sequence[float] | NDArray[Any] | None = None,
    ) -> None:
        super().__init__(lower=lower, upper=upper, grid=grid)

    @model_validator(mode="before")
    @classmethod
    def _expand_grid(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        count = data.get("grid")
        if isinstance(count, bool) or not isinstance(count, int | np.integer):
            return data
        lower, upper = _real(data.get("lower", 0.0)), _real(data.get("upper"))
        if lower is None or upper is None or not math.isfinite(lower + upper):
            raise CoverDefinitionError("a grid given as a count needs finite bounds")
        if count < 2 or not lower < upper:
            raise CoverDefinitionError(
                "a grid given as a count needs at least 2 points and lower < upper"
            )
        values = np.linspace(lower, upper, int(count)).tolist()
        return {**data, "grid": tuple(values)}

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.upper is not None and self.lower > self.upper:
            raise CoverDefinitionError(
                f"lower must be <= upper, got lower={self.lower}, upper={self.upper}"
            )
        if self.grid is not None:
            grid = self.grid
            if not grid:
                raise CoverDefinitionError("grid must hold at least one value")
            if any(later <= earlier for earlier, later in zip(grid, grid[1:])):
                raise CoverDefinitionError("grid values must be strictly increasing")
            above = self.upper is not None and grid[-1] > self.upper
            if grid[0] < self.lower or above:
                raise CoverDefinitionError(
                    f"grid values must lie within [{self.lower}, {self.upper}]"
                )
        return self

    def contains(self, value: float) -> bool:
        """Whether `value` lies within the bounds (the grid is not checked)."""
        return self.lower <= value and (self.upper is None or value <= self.upper)

    def to_document(self) -> dict[str, Any]:
        """Return the document form, `{"free": {"grid", "lower", "upper"}}`."""
        grid = None if self.grid is None else list(self.grid)
        return {"free": {"grid": grid, "lower": self.lower, "upper": self.upper}}

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Free:
        """Read the document form written by `to_document`."""
        if not isinstance(document, Mapping) or set(document) != {"free"}:
            raise CoverDefinitionError(
                'a free parameter document must be {"free": {...}}'
            )
        return cls._validated(document["free"])


def _resolve_values(
    parameters: Mapping[str, float | Free], values: Mapping[str, float]
) -> dict[str, float]:
    """Check values for free parameters and return them as floats.

    Every name must be a free parameter of `parameters`, and every value a
    finite number within that parameter's bounds. Raises
    `CoverDefinitionError` otherwise.
    """
    if not isinstance(values, Mapping):
        raise CoverDefinitionError("values must map parameter names to numbers")
    checked: dict[str, float] = {}
    for name, raw in values.items():
        if name not in parameters:
            known = ", ".join(parameters) or "none"
            raise CoverDefinitionError(
                f"unknown parameter {name!r}; the parameters are: {known}"
            )
        parameter = parameters[name]
        if not isinstance(parameter, Free):
            raise CoverDefinitionError(
                f"parameter {name!r} is fixed; only free parameters take values"
            )
        value = _real(raw)
        if value is None or not math.isfinite(value):
            raise CoverDefinitionError(
                f"the value of {name!r} must be a finite number, got {raw!r}"
            )
        if not parameter.contains(value):
            raise CoverDefinitionError(
                f"the value {value} of {name!r} is outside "
                f"[{parameter.lower}, {parameter.upper}]"
            )
        checked[name] = value + 0.0
    return checked
