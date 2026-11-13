"""Payout functions: the protocols, the built-in families and their registry.

A payout function maps one year's index value to the year's payout. Every
family pays more as the index moves to the severe side. With
`tail="upper"` larger index values are more severe; with `tail="lower"`
smaller ones are. Thresholds are given in index units in order of increasing
severity: increasing for `"upper"`, decreasing for `"lower"`. Internally a
family works on the severity `s = x` (upper) or `s = -x` (lower).

- `LinearPayout(attachment, exhaustion, limit)`: `limit * clip((s - a) /
  (e - a), 0, 1)`.
- `StepPayout(breakpoints, levels)`: `levels[k]` for `b_k <= s < b_{k+1}`,
  the last level at or beyond the last breakpoint.
- `PiecewiseLinearPayout(knots, levels)`: the points `(c_k, levels[k])`
  joined by straight lines, the last level at or beyond the last knot.
- `BinaryPayout(trigger, amount)`: `amount` for `s >= t`.

Thresholds (attachment, exhaustion, breakpoints, knots, trigger) are in index
units; amounts (limit, levels, amount) in payout units.

- Every family pays nothing below its first threshold.
- A severity equal to a threshold belongs to the more severe side (half-open
  bins).
- Amounts are non-negative. `monotone=True`, the default for steps and
  piecewise linear payouts, also requires non-decreasing levels.

Each parameter is a float (fixed) or `Free` (chosen by a design). Parameter
names are flat strings: `attachment`, `levels[2]`, `knots[1]`, ... A user
family implements `PayoutFunction` (and `LinearInAmounts` for payouts linear
in their amounts) and is made readable from cover documents with
`register_payout`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import (
    Annotated,
    Any,
    ClassVar,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BeforeValidator, Strict, model_validator

from crc_sdk.parametric.errors import (
    CoverDefinitionError,
    SampleError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from crc_sdk.parametric.index import Tail
from crc_sdk.parametric.parameters import Free, _DefinitionModel, _resolve_values

__all__ = [
    "BinaryPayout",
    "LinearConstraints",
    "LinearInAmounts",
    "LinearPayout",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "StepPayout",
    "register_payout",
]

FloatArray = NDArray[np.float64]


@runtime_checkable
class PayoutFunction(Protocol):
    """What every payout function provides.

    `kind` is the registry key written into the cover document. `tail` is
    the severe side of the index; a cover requires it to equal its index's.
    `index_names` names the index components the payout reads, in the order
    of the columns of a multi-component index; Phase 1 covers read one,
    `("index",)`.

    `parameters()` returns every parameter by flat name, in a fixed order,
    each a float or `Free`. `with_values` returns a copy with values for
    some free parameters. `evaluate` maps index values to payouts, same
    shape and non-negative, and needs every parameter resolved. The document
    holds `kind` and every parameter, with `Free` ones in their document
    form, so that `from_document(to_document())` rebuilds the payout.
    """

    kind: ClassVar[str]

    @property
    def tail(self) -> Tail: ...

    @property
    def index_names(self) -> tuple[str, ...]: ...

    def parameters(self) -> Mapping[str, float | Free]: ...

    def with_values(self, values: Mapping[str, float]) -> Self: ...

    def evaluate(self, index: FloatArray) -> FloatArray: ...

    def to_document(self) -> dict[str, Any]: ...

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self: ...


@runtime_checkable
class LinearInAmounts(PayoutFunction, Protocol):
    """A payout that is linear in its amounts once its thresholds are fixed.

    The payout is `basis(index) @ amounts`, with the amounts in
    `amount_names()` order. `basis` returns one row per index value and one
    column per amount, and needs the thresholds resolved; the amounts may
    still be free. `amount_constraints(limit)` states the bounds and linear
    constraints on the amounts, with every amount at most `limit` when one
    is given. Designs use these to solve for the amounts as a linear or
    quadratic programme.
    """

    def amount_names(self) -> tuple[str, ...]: ...

    def basis(self, index: FloatArray) -> FloatArray: ...

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints: ...


def _float_array(name: str, values: ArrayLike, ndim: int) -> FloatArray:
    try:
        array = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CoverDefinitionError(f"{name} must be numeric") from error
    if array.ndim != ndim:
        raise CoverDefinitionError(f"{name} must be {ndim}-D, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class LinearConstraints:
    """Linear constraints on an amount vector `a`.

    `a_ub @ a <= b_ub` and `lower <= a <= upper`, where `upper` is `inf` for
    an amount without an upper bound. `a_ub` has one row per constraint and
    one column per amount. The arrays are read-only copies.

    Raises `CoverDefinitionError` for mismatched shapes or non-finite values
    (other than an infinite upper bound).
    """

    a_ub: FloatArray
    b_ub: FloatArray
    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        a_ub = _float_array("a_ub", self.a_ub, 2)
        b_ub = _float_array("b_ub", self.b_ub, 1)
        lower = _float_array("lower", self.lower, 1)
        upper = _float_array("upper", self.upper, 1)
        n_amounts = lower.shape[0]
        if upper.shape != (n_amounts,) or a_ub.shape[1:] != (n_amounts,):
            raise CoverDefinitionError(
                "lower, upper and the columns of a_ub must have one entry per amount"
            )
        if b_ub.shape != (a_ub.shape[0],):
            raise CoverDefinitionError("b_ub must have one entry per row of a_ub")
        if not (np.isfinite(a_ub).all() and np.isfinite(b_ub).all()):
            raise CoverDefinitionError("a_ub and b_ub must be finite")
        if not np.isfinite(lower).all() or np.isnan(upper).any():
            raise CoverDefinitionError("lower must be finite and upper not NaN")
        if np.isneginf(upper).any():
            raise CoverDefinitionError("upper must not be -inf")
        object.__setattr__(self, "a_ub", a_ub)
        object.__setattr__(self, "b_ub", b_ub)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def n_amounts(self) -> int:
        """The number of amounts the constraints are over."""
        return int(self.lower.shape[0])


def _index_values(index: ArrayLike, *, ndim: int | None = None) -> FloatArray:
    """Return index values as a float array; raise `SampleError` if invalid."""
    try:
        values = np.asarray(index, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SampleError("index values must be numeric") from error
    if ndim is not None and values.ndim != ndim:
        raise SampleError(f"index values must be {ndim}-D, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise SampleError("index values must be finite")
    return values


def _parameter_value(value: Any) -> Any:
    if isinstance(value, Free):
        return value
    if isinstance(value, Mapping):
        return Free.from_document(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CoverDefinitionError(f"must be a number or Free, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise CoverDefinitionError(f"must be finite, got {number}")
    return number + 0.0


def _parameter_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise CoverDefinitionError(f"must be a sequence, got {value!r}")
    return value


Parameter = Annotated[float | Free, BeforeValidator(_parameter_value)]
"""A fixed value or a `Free` parameter."""

Parameters = Annotated[tuple[Parameter, ...], BeforeValidator(_parameter_list)]
"""A sequence of parameters, such as the levels of a step payout."""

StrictBool = Annotated[bool, Strict()]


def _severity_candidates(value: float | Free, sign: float) -> list[float]:
    """The severities a threshold can take: its value, or its grid."""
    if isinstance(value, Free):
        return sorted(sign * point for point in value.grid or ())
    return [sign * value]


def _check_thresholds(
    names: Sequence[str], values: Sequence[float | Free], tail: Tail, what: str
) -> None:
    """Thresholds must be strictly increasing in severity for some choice."""
    for name, value in zip(names, values, strict=True):
        if isinstance(value, Free) and (value.upper is None or value.grid is None):
            raise CoverDefinitionError(
                f"{name} is a threshold in index units: a free threshold needs a "
                "finite upper bound and a grid"
            )
    sign = 1.0 if tail == "upper" else -1.0
    previous = -math.inf
    for value in values:
        later = [s for s in _severity_candidates(value, sign) if s > previous]
        if not later:
            direction = "increasing" if tail == "upper" else "decreasing"
            choice = (
                "; no choice of the free values is ordered"
                if any(isinstance(item, Free) for item in values)
                else ""
            )
            raise CoverDefinitionError(
                f"{what} must be strictly {direction} for tail={tail!r}{choice}"
            )
        previous = later[0]


def _check_amounts(
    names: Sequence[str], values: Sequence[float | Free], *, monotone: bool, what: str
) -> None:
    """Amounts must be non-negative and, if monotone, non-decreasing."""
    for name, value in zip(names, values, strict=True):
        lowest = value.lower if isinstance(value, Free) else value
        if lowest < 0:
            bound = "the lower bound of " if isinstance(value, Free) else ""
            raise CoverDefinitionError(
                f"{bound}{name} must be >= 0: amounts are non-negative"
            )
    if not monotone:
        return
    previous = 0.0
    for value in values:
        if isinstance(value, Free):
            lowest = max(previous, value.lower)
            if value.grid is not None:
                above = [point for point in value.grid if point >= lowest]
                feasible = bool(above)
                lowest = above[0] if above else lowest
            else:
                feasible = value.upper is None or value.upper >= lowest
        else:
            feasible = value >= previous
            lowest = value
        if not feasible:
            raise CoverDefinitionError(
                f"{what} must be non-decreasing when monotone=True"
            )
        previous = lowest


def _location(name: str) -> tuple[str, int | None]:
    field, _, rest = name.partition("[")
    return (field, int(rest[:-1])) if rest else (field, None)


class _BuiltinPayout(_DefinitionModel):
    """Shared parameter handling of the built-in families."""

    kind: ClassVar[str]
    _threshold_fields: ClassVar[tuple[str, ...]]
    _amount_fields: ClassVar[tuple[str, ...]]

    tail: Tail = "upper"

    def __init__(self, **data: Any) -> None:
        # Explicit, so type checkers do not take pydantic's field-only signature.
        super().__init__(**data)

    @property
    def index_names(self) -> tuple[str, ...]:
        """The index components the payout reads: `("index",)`."""
        return ("index",)

    def _field_items(self, fields: Sequence[str]) -> list[tuple[str, float | Free]]:
        items: list[tuple[str, float | Free]] = []
        for field in fields:
            value = getattr(self, field)
            if isinstance(value, tuple):
                items.extend((f"{field}[{i}]", item) for i, item in enumerate(value))
            else:
                items.append((field, value))
        return items

    def parameters(self) -> dict[str, float | Free]:
        """Every parameter by flat name: thresholds first, then amounts."""
        return dict(self._field_items(self._threshold_fields + self._amount_fields))

    def amount_names(self) -> tuple[str, ...]:
        """The names of the amounts, in the column order of `basis`."""
        return tuple(name for name, _ in self._field_items(self._amount_fields))

    def with_values(self, values: Mapping[str, float]) -> Self:
        """Return a copy with values for some of the free parameters.

        Each name must be a free parameter, and each value a finite number
        within its bounds (the grid is not enforced). The copy is validated
        again, so the resolved thresholds must be ordered. Raises
        `CoverDefinitionError` otherwise.
        """
        checked = _resolve_values(self.parameters(), values)
        data: dict[str, Any] = {
            field: list(value) if isinstance(value, tuple) else value
            for field, value in self._fields_data().items()
        }
        for name, value in checked.items():
            field, position = _location(name)
            if position is None:
                data[field] = value
            else:
                data[field][position] = value
        return type(self)._validated(data)

    def _fields_data(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in type(self).model_fields}

    def _unresolved(self, fields: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            name for name, value in self._field_items(fields) if isinstance(value, Free)
        )

    def _require_resolved(self, fields: Sequence[str]) -> None:
        free = self._unresolved(fields)
        if free:
            raise UnresolvedParameterError(
                f"the {self.kind} payout has free parameters: {', '.join(free)}"
            )

    def _sign(self) -> float:
        return 1.0 if self.tail == "upper" else -1.0

    def _thresholds(self) -> FloatArray:
        """The resolved thresholds as severities, strictly increasing."""
        self._require_resolved(self._threshold_fields)
        values = [value for _, value in self._field_items(self._threshold_fields)]
        return self._sign() * np.asarray(values, dtype=np.float64)

    def _amounts(self) -> FloatArray:
        self._require_resolved(self._amount_fields)
        values = [value for _, value in self._field_items(self._amount_fields)]
        return np.asarray(values, dtype=np.float64)

    def _monotone(self) -> bool:
        return False

    def evaluate(self, index: FloatArray) -> FloatArray:
        """Return the payout of each index value, same shape.

        Raises `UnresolvedParameterError` if a parameter is free, and
        `SampleError` for non-numeric or non-finite index values.
        """
        self._require_resolved(self._threshold_fields + self._amount_fields)
        values = _index_values(index)
        severity = self._sign() * values.reshape(-1)
        payout = self._payout(severity, self._thresholds(), self._amounts())
        return payout.reshape(values.shape)

    def basis(self, index: FloatArray) -> FloatArray:
        """Return the basis matrix, `(n, len(amount_names()))`.

        The payout is `basis(index) @ amounts`. Rows are non-negative and
        sum to at most 1, so every payout is at most the largest amount.
        Needs the thresholds resolved; raises `UnresolvedParameterError`
        otherwise, and `SampleError` unless `index` is 1-D and finite.
        """
        self._require_resolved(self._threshold_fields)
        severity = self._sign() * _index_values(index, ndim=1)
        return self._basis(severity, self._thresholds())

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        """Return the bounds and constraints on the amounts.

        A fixed amount is bounded by its value on both sides; a free one by
        its `Free` bounds. `limit` caps every amount; with `monotone=True`
        each level is at most the next. Raises `CoverDefinitionError` for a
        negative or non-finite limit.
        """
        items = self._field_items(self._amount_fields)
        lower = np.array(
            [value.lower if isinstance(value, Free) else value for _, value in items]
        )
        upper = np.array(
            [
                (math.inf if value.upper is None else value.upper)
                if isinstance(value, Free)
                else value
                for _, value in items
            ]
        )
        if limit is not None:
            cap = None if isinstance(limit, bool) else _parameter_or_none(limit)
            if cap is None or cap < 0:
                raise CoverDefinitionError(
                    f"limit must be a finite number >= 0, got {limit!r}"
                )
            upper = np.minimum(upper, cap)
        n_amounts = len(items)
        if self._monotone() and n_amounts > 1:
            a_ub = np.zeros((n_amounts - 1, n_amounts))
            rows = np.arange(n_amounts - 1)
            a_ub[rows, rows] = 1.0
            a_ub[rows, rows + 1] = -1.0
        else:
            a_ub = np.zeros((0, n_amounts))
        return LinearConstraints(
            a_ub=a_ub, b_ub=np.zeros(a_ub.shape[0]), lower=lower, upper=upper
        )

    def to_document(self) -> dict[str, Any]:
        """Return `kind` and every field; `Free` parameters in document form."""
        document: dict[str, Any] = {"kind": self.kind}
        for field, value in self._fields_data().items():
            document[field] = _document_value(value)
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Read a payout written by `to_document`.

        Raises `CoverDefinitionError` for another `kind`, unknown or missing
        fields, or invalid values.
        """
        if not isinstance(document, Mapping):
            raise CoverDefinitionError("a payout document must be an object")
        if document.get("kind") != cls.kind:
            raise CoverDefinitionError(
                f"expected a payout of kind {cls.kind!r}, got {document.get('kind')!r}"
            )
        return cls._validated(
            {key: value for key, value in document.items() if key != "kind"}
        )

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        raise NotImplementedError

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        raise NotImplementedError


def _parameter_or_none(value: Any) -> float | None:
    if not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _document_value(value: Any) -> Any:
    if isinstance(value, Free):
        return value.to_document()
    if isinstance(value, tuple):
        return [_document_value(item) for item in value]
    return value


class LinearPayout(_BuiltinPayout):
    """Pays in proportion to the severity between attachment and exhaustion.

    The payout is `limit * clip((s - a) / (e - a), 0, 1)` in severity `s`:
    nothing at or below the `attachment`, the full `limit` at or beyond the
    `exhaustion`, linear in between. `attachment` and `exhaustion` are in
    index units, with the exhaustion more severe (larger for `tail="upper"`,
    smaller for `"lower"`). `limit` is in payout units and non-negative.

    Raises `CoverDefinitionError` for invalid parameters.
    """

    kind: ClassVar[str] = "linear"
    _threshold_fields: ClassVar[tuple[str, ...]] = ("attachment", "exhaustion")
    _amount_fields: ClassVar[tuple[str, ...]] = ("limit",)

    attachment: Parameter
    exhaustion: Parameter
    limit: Parameter

    def __init__(
        self,
        attachment: float | Free,
        exhaustion: float | Free,
        limit: float | Free,
        *,
        tail: Tail = "upper",
    ) -> None:
        super().__init__(
            attachment=attachment, exhaustion=exhaustion, limit=limit, tail=tail
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_thresholds(
            ("attachment", "exhaustion"),
            (self.attachment, self.exhaustion),
            self.tail,
            "attachment and exhaustion",
        )
        _check_amounts(("limit",), (self.limit,), monotone=False, what="limit")
        return self

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        payout: FloatArray = amounts[0] * self._basis(severity, thresholds)[:, 0]
        return payout

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        attachment, exhaustion = float(thresholds[0]), float(thresholds[1])
        share = (severity - attachment) / (exhaustion - attachment)
        basis: FloatArray = np.clip(share, 0.0, 1.0).reshape(-1, 1)
        return basis


class StepPayout(_BuiltinPayout):
    """Pays a level per band of severity.

    With breakpoints b_0 < ... < b_{n-1} in severity, a year pays `levels[k]`
    when b_k <= s < b_{k+1}, `levels[n-1]` when s >= b_{n-1}, and nothing
    below b_0, the attachment. A severity equal to a breakpoint falls into
    the band above it. `breakpoints` are in index units, increasing for
    `tail="upper"` and decreasing for `"lower"`; `levels` are in payout
    units, one per breakpoint. `monotone=True` requires non-decreasing
    levels.

    Raises `CoverDefinitionError` for invalid parameters.
    """

    kind: ClassVar[str] = "step"
    _threshold_fields: ClassVar[tuple[str, ...]] = ("breakpoints",)
    _amount_fields: ClassVar[tuple[str, ...]] = ("levels",)

    breakpoints: Parameters
    levels: Parameters
    monotone: StrictBool = True

    def __init__(
        self,
        breakpoints: Sequence[float | Free] | NDArray[np.float64],
        levels: Sequence[float | Free] | NDArray[np.float64],
        *,
        tail: Tail = "upper",
        monotone: bool = True,
    ) -> None:
        super().__init__(
            breakpoints=breakpoints, levels=levels, tail=tail, monotone=monotone
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_banded(self, "breakpoints", minimum=1)
        return self

    def _monotone(self) -> bool:
        return self.monotone

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        band = np.searchsorted(thresholds, severity, side="right") - 1
        return np.where(band >= 0, amounts[np.maximum(band, 0)], 0.0)

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        band = np.searchsorted(thresholds, severity, side="right") - 1
        matrix = np.zeros((severity.shape[0], thresholds.shape[0]))
        rows = np.flatnonzero(band >= 0)
        matrix[rows, band[rows]] = 1.0
        return matrix


class PiecewiseLinearPayout(_BuiltinPayout):
    """Pays along straight lines between knots.

    With knots c_0 < ... < c_{m-1} in severity (m >= 2), a year pays nothing
    below c_0, the linear interpolation of the points (c_k, levels[k]) on
    [c_0, c_{m-1}], and `levels[m-1]` at or above c_{m-1}. At c_0 the payout
    is `levels[0]`, so a positive first level is a jump at the attachment.
    `knots` are in index units, increasing for `tail="upper"` and decreasing
    for `"lower"`; `levels` are in payout units, one per knot.
    `monotone=True` requires non-decreasing levels.

    Raises `CoverDefinitionError` for invalid parameters.
    """

    kind: ClassVar[str] = "piecewise_linear"
    _threshold_fields: ClassVar[tuple[str, ...]] = ("knots",)
    _amount_fields: ClassVar[tuple[str, ...]] = ("levels",)

    knots: Parameters
    levels: Parameters
    monotone: StrictBool = True

    def __init__(
        self,
        knots: Sequence[float | Free] | NDArray[np.float64],
        levels: Sequence[float | Free] | NDArray[np.float64],
        *,
        tail: Tail = "upper",
        monotone: bool = True,
    ) -> None:
        super().__init__(knots=knots, levels=levels, tail=tail, monotone=monotone)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_banded(self, "knots", minimum=2)
        return self

    def _monotone(self) -> bool:
        return self.monotone

    @staticmethod
    def _segments(
        severity: FloatArray, knots: FloatArray
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], FloatArray, NDArray[np.intp]]:
        """Rows inside [c_0, c_{m-1}), their segment and weight; rows beyond."""
        segment = np.searchsorted(knots, severity, side="right") - 1
        last = knots.shape[0] - 1
        inside = np.flatnonzero((segment >= 0) & (segment < last))
        left = segment[inside]
        weight = (severity[inside] - knots[left]) / (knots[left + 1] - knots[left])
        beyond = np.flatnonzero(segment >= last)
        return inside, left, weight, beyond

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        inside, left, weight, beyond = self._segments(severity, thresholds)
        low, high = amounts[left], amounts[left + 1]
        # low + w * (high - low) is exact at a flat segment and monotone in w;
        # the clip keeps rounding from leaving the segment's range.
        line = np.clip(
            low + weight * (high - low), np.minimum(low, high), np.maximum(low, high)
        )
        payout = np.zeros(severity.shape[0])
        payout[inside] = line
        payout[beyond] = amounts[-1]
        return payout

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        inside, left, weight, beyond = self._segments(severity, thresholds)
        matrix = np.zeros((severity.shape[0], thresholds.shape[0]))
        matrix[inside, left] = 1.0 - weight
        matrix[inside, left + 1] = weight
        matrix[beyond, -1] = 1.0
        return matrix


class BinaryPayout(_BuiltinPayout):
    """Pays a fixed amount when the index reaches the trigger.

    A year pays `amount` when its severity is at or beyond `trigger` (at or
    above it for `tail="upper"`, at or below it for `"lower"`), and nothing
    otherwise. `trigger` is in index units, `amount` in payout units and
    non-negative.

    Raises `CoverDefinitionError` for invalid parameters.
    """

    kind: ClassVar[str] = "binary"
    _threshold_fields: ClassVar[tuple[str, ...]] = ("trigger",)
    _amount_fields: ClassVar[tuple[str, ...]] = ("amount",)

    trigger: Parameter
    amount: Parameter

    def __init__(
        self, trigger: float | Free, amount: float | Free, *, tail: Tail = "upper"
    ) -> None:
        super().__init__(trigger=trigger, amount=amount, tail=tail)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_thresholds(("trigger",), (self.trigger,), self.tail, "trigger")
        _check_amounts(("amount",), (self.amount,), monotone=False, what="amount")
        return self

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        payout: FloatArray = np.where(severity >= thresholds[0], amounts[0], 0.0)
        return payout

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        basis: FloatArray = (severity >= thresholds[0]).astype(np.float64)
        return basis.reshape(-1, 1)


def _check_banded(
    payout: StepPayout | PiecewiseLinearPayout, field: str, *, minimum: int
) -> None:
    thresholds: tuple[float | Free, ...] = getattr(payout, field)
    if len(thresholds) < minimum:
        raise CoverDefinitionError(
            f"{field} must hold at least {minimum} value{'s' if minimum > 1 else ''}"
        )
    if len(payout.levels) != len(thresholds):
        raise CoverDefinitionError(
            f"levels must hold one value per entry of {field}: "
            f"got {len(payout.levels)} levels and {len(thresholds)} {field}"
        )
    names = [f"{field}[{i}]" for i in range(len(thresholds))]
    _check_thresholds(names, thresholds, payout.tail, field)
    _check_amounts(
        [f"levels[{i}]" for i in range(len(payout.levels))],
        payout.levels,
        monotone=payout.monotone,
        what="levels",
    )


_KIND = re.compile(r"^\S(?:.*\S)?$")
_PAYOUT_METHODS = ("parameters", "with_values", "evaluate", "to_document")
_REGISTRY: dict[str, type[Any]] = {}

PayoutClass = TypeVar("PayoutClass", bound=type[Any])


def register_payout(cls: PayoutClass) -> PayoutClass:
    """Register a payout class under its `kind`, so covers can read it back.

    Use it as a class decorator on a user family that implements
    `PayoutFunction`. `Cover.from_json` looks the document's `kind` up here.
    Registering the same class again does nothing. Raises
    `CoverDefinitionError` if `cls` is not a class with a non-empty string
    `kind` and the protocol's methods, or if another class already holds its
    `kind`.
    """
    if not isinstance(cls, type):
        raise CoverDefinitionError(f"register_payout takes a class, got {cls!r}")
    kind = getattr(cls, "kind", None)
    if not isinstance(kind, str) or not _KIND.fullmatch(kind):
        raise CoverDefinitionError(
            f"{cls.__qualname__}.kind must be a non-empty string without "
            "surrounding whitespace"
        )
    missing = [
        name
        for name in (*_PAYOUT_METHODS, "from_document")
        if not callable(getattr(cls, name, None))
    ]
    if missing:
        raise CoverDefinitionError(
            f"{cls.__qualname__} does not implement PayoutFunction: "
            f"missing {', '.join(missing)}"
        )
    registered = _REGISTRY.get(kind)
    if registered is not None and registered is not cls:
        raise CoverDefinitionError(
            f"payout kind {kind!r} is already registered by {registered.__qualname__}"
        )
    _REGISTRY[kind] = cls
    return cls


def _payout_reader(kind: Any) -> Callable[[Mapping[str, Any]], Any]:
    """Return the registered `from_document` of `kind`."""
    if not isinstance(kind, str):
        raise CoverDefinitionError(
            f"a payout document needs a string kind, got {kind!r}"
        )
    cls = _REGISTRY.get(kind)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownPayoutKindError(
            f"unknown payout kind {kind!r}; registered kinds: {known}. "
            "Register a user family with register_payout."
        )
    reader: Callable[[Mapping[str, Any]], Any] = cls.from_document
    return reader


for _family in (LinearPayout, StepPayout, PiecewiseLinearPayout, BinaryPayout):
    register_payout(_family)
del _family
