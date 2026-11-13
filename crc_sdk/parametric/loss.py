"""The loss model: an impact function of the hazard, times the asset value.

`LossModel(impact, ...)` computes one year's loss from the hazard at the
asset: `loss = impact(hazard) * value`. `impact` takes the types
`PortfolioEvaluation.impact(...)` takes:

- an `ImpactExpression` (crc-sdk's or crc-framework's), which carries its own
  context and records its `definition_hash`;
- a crc-framework `ImpactFunction`: `PiecewiseLinearImpact`, a registry
  impact such as `impacts.for_factor("cyclone", context=...)`, or any object
  with `evaluate(values, *, context=None)`;
- a vectorised callable from hazard values to impacts.

The value is an asset column (`value_column`, read by `for_asset`), a stated
number (`value`), or 1, which gives the loss as the damage ratio in
`value_unit="1"`. Context columns for registry impacts work as in
`PortfolioEvaluation.impact(..., context=ImpactContextColumns(...))`: the
asset's columns fill the framework's `TransformContext`.

An `ImpactExpression` declares the hazard unit it is calibrated for
(`calibration.exposure_unit`). `check_hazard_unit(unit)`, or the `hazard_unit=`
argument of `impact_values` and `evaluate`, compares the unit of the hazard
values with it, as `crc_sdk.impacts.check_exposure` compares units, and raises
`SampleError` on a mismatch rather than computing losses from values in the
wrong unit; `sample_curve` and `sample_annual` call it before drawing. Other
impacts declare no unit, so they are not checked. The exception is a
registry `ClimateImpact` (from `crc_sdk.impacts.for_factor`): its unit is the
factor's calibration in the registry manifest. A factor whose branches
declare different units is not checked.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from crc_framework import (
    CallableImpact,
    ClimateImpact,
    ImpactFunction,
    TransformContext,
    impact_registry_manifest,
)
from crc_framework import ImpactExpression as FrameworkImpactExpression
from numpy.typing import ArrayLike, NDArray

from crc_sdk.impacts.calibration import units_equivalent
from crc_sdk.impacts.config import RegistryImpactReference
from crc_sdk.impacts.errors import ImpactCalibrationError
from crc_sdk.impacts.expression import ImpactExpression
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.errors import SampleError
from crc_sdk.parametric.sample import _arrow_table, _identifier
from crc_sdk.workflows.portfolio import AssetPortfolio, CellColumn, ImpactContextColumns

__all__ = ["LossModel"]

FloatArray = NDArray[np.float64]

_CONTEXT_FIELDS = ("cell", "country", "continent", "building_type", "historic_mean")


def _context_document(context: TransformContext | None) -> dict[str, Any] | None:
    if context is None:
        return None
    document: dict[str, Any] = {}
    for name in _CONTEXT_FIELDS:
        value = getattr(context, name)
        if name == "cell" and value is not None:
            value = format(int(value), "x")
        document[name] = value
    return document


def _exposure_unit(impact: ImpactExpression | FrameworkImpactExpression) -> str:
    """Return the hazard unit an impact expression's calibration declares."""
    if isinstance(impact, ImpactExpression):
        return str(impact.calibration.exposure_unit)
    return str(impact.to_config()["calibration"]["exposure_unit"])


def _registry_exposure_unit(factor: str) -> str | None:
    """Return the hazard unit of a registry factor, or `None` if it has no one.

    The unit is the factor's calibration in the installed registry manifest
    (`RegistryImpactReference.pinned(factor).calibration()`; the manifest is
    bundled with `crc_framework`, so there is no file or network access). A
    factor whose calibration depends on a branch (for example
    `cooling_degree_days`) has a unit if every branch declares the same one;
    branches that disagree, or a factor the manifest does not list, give
    `None`: the unit cannot be settled without context, so the impact is
    not checked rather than guessed.
    """
    reference = RegistryImpactReference.pinned(factor)
    try:
        return str(reference.calibration().exposure_unit)
    except ImpactCalibrationError as error:
        if error.code != "branch_required":
            return None
    entry = next(
        (
            item
            for item in impact_registry_manifest()["factors"]
            if item["factor"] == factor
        ),
        None,
    )
    if entry is None:
        return None
    units = {
        str(reference.calibration(branch).exposure_unit)
        for branch in entry.get("branch_calibrations") or {}
    }
    return units.pop() if len(units) == 1 else None


def _finite_non_negative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SampleError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise SampleError(f"{name} must be finite and non-negative, got {value!r}")
    return number


class LossModel:
    """Loss = impact(hazard) × asset value.

    `impact` is an `ImpactExpression`, a crc-framework `ImpactFunction` or a
    vectorised callable (module docstring). `name` labels it in the
    provenance; by default the registry factor, the expression's model kind,
    or the callable's name.

    The value is one of:

    - `value_column`: the column of the asset table holding the asset's
      value; bind the model to an asset with `for_asset` before sampling;
    - `value`: a finite, non-negative number;
    - neither: 1, so the loss is the damage ratio; `value_unit` must then be
      `"1"`.

    `value_unit` is the unit of the loss, for example a currency; it must
    equal a cover's `payout_unit` for designs and evaluations. `context` is
    `ImpactContextColumns` (asset columns that fill the framework's
    `TransformContext`, read by `for_asset`) or a fixed `TransformContext`.
    Expressions carry their own context, so they take none.

    `exposure_unit` is the hazard unit an expression is calibrated for, and
    `check_hazard_unit(unit)` checks hazard values against it.

    The provenance records the impact's type, name, `definition_hash` and
    `exposure_unit` (for expressions), the value's source and unit, and a
    SHA-256 of the (asset id, value) pair, not the value itself.

    Raises `SampleError` for invalid values, units or contexts, and
    `TypeError` for an impact of another type.
    """

    __slots__ = (
        "_asset_id",
        "_bound_context",
        "_context",
        "_evaluate",
        "_exposure_unit",
        "_hash",
        "_impact",
        "_kind",
        "_name",
        "_type",
        "_value",
        "_value_column",
        "_value_unit",
    )

    def __init__(
        self,
        impact: Any,
        *,
        value_column: str | None = None,
        value: float | None = None,
        value_unit: str = "1",
        context: ImpactContextColumns | TransformContext | None = None,
        name: str | None = None,
    ) -> None:
        evaluate: Callable[[FloatArray, TransformContext | None], Any]
        if isinstance(impact, (ImpactExpression, FrameworkImpactExpression)):
            kind = "expression"
            default_name = str(impact.model_kind)
            definition_hash: str | None = str(impact.definition_hash)
            exposure_unit: str | None = _exposure_unit(impact)
            expression = impact

            def evaluate(values: FloatArray, context: TransformContext | None) -> Any:
                del context
                return expression.evaluate(values)

        elif isinstance(impact, ImpactFunction):
            kind = "function"
            factor = getattr(impact, "factor", None)
            default_name = str(factor) if factor is not None else type(impact).__name__
            definition_hash = None
            exposure_unit = (
                _registry_exposure_unit(str(factor))
                if isinstance(impact, ClimateImpact)
                else None
            )
            function = impact

            def evaluate(values: FloatArray, context: TransformContext | None) -> Any:
                return function.evaluate(values, context=context)

        elif callable(impact):
            kind = "callable"
            default_name = str(
                getattr(impact, "__qualname__", None) or type(impact).__name__
            )
            definition_hash = None
            exposure_unit = None
            wrapped = CallableImpact(impact)

            def evaluate(values: FloatArray, context: TransformContext | None) -> Any:
                del context
                return wrapped.evaluate(values)

        else:
            raise TypeError(
                "impact must be an ImpactExpression, an ImpactFunction or a "
                f"vectorised callable, got {type(impact).__name__}"
            )
        if context is not None and not isinstance(
            context, (ImpactContextColumns, TransformContext)
        ):
            raise TypeError(
                "context must be ImpactContextColumns or a TransformContext, got "
                f"{type(context).__name__}"
            )
        has_context = isinstance(context, TransformContext) or (
            isinstance(context, ImpactContextColumns) and bool(context.items())
        )
        if has_context and kind == "expression":
            raise SampleError(
                "impact expressions carry their own context; do not pass a context"
            )
        if has_context and kind == "callable":
            raise SampleError(
                "a callable impact takes no context; wrap it in an ImpactFunction"
            )
        if value_column is not None and value is not None:
            raise SampleError("give value_column or value, not both")
        if value_column is not None:
            _identifier("value_column", value_column)
        unit = _identifier("value_unit", value_unit)
        number = None if value is None else _finite_non_negative("value", value)
        if value_column is None and value is None and unit != "1":
            raise SampleError(
                f"without a value the loss is the damage ratio, in unit '1'; got "
                f"value_unit={unit!r}"
            )
        self._impact = impact
        self._kind = kind
        self._evaluate = evaluate
        self._type = type(impact).__name__
        self._name = default_name if name is None else _identifier("name", name)
        self._hash = definition_hash
        self._exposure_unit = exposure_unit
        self._value_column = value_column
        self._value = number
        self._value_unit = unit
        self._context = context
        self._bound_context: TransformContext | None = (
            context if isinstance(context, TransformContext) else None
        )
        self._asset_id: str | None = None

    @property
    def impact(self) -> Any:
        """The impact function, as given."""
        return self._impact

    @property
    def name(self) -> str:
        """The impact's label in the provenance."""
        return self._name

    @property
    def exposure_unit(self) -> str | None:
        """The hazard unit the impact is calibrated for, if it declares one.

        An `ImpactExpression`'s `calibration.exposure_unit`; a registry
        `ClimateImpact`'s unit from the registry manifest (all branches of a
        branch-dependent factor must agree). Any other `ImpactFunction`, a
        callable, or a registry factor whose branches disagree declares none:
        `None`.
        """
        return self._exposure_unit

    @property
    def value_column(self) -> str | None:
        """The asset column holding the value, if the value comes from one."""
        return self._value_column

    @property
    def value_unit(self) -> str:
        """The unit of the loss."""
        return self._value_unit

    @property
    def context(self) -> ImpactContextColumns | TransformContext | None:
        """The context columns or fixed context, as given."""
        return self._context

    @property
    def asset_id(self) -> str | None:
        """The asset the model is bound to by `for_asset`, if any."""
        return self._asset_id

    def _needs_asset(self) -> bool:
        return self._value_column is not None or (
            isinstance(self._context, ImpactContextColumns)
            and bool(self._context.items())
        )

    def for_asset(self, assets: Any, asset_id: str) -> LossModel:
        """Return the model bound to one asset's row of an asset table.

        `assets` is an `AssetPortfolio` (its `id_column` names the id column)
        or an in-memory table with an `asset_id` column. The value comes from
        `value_column`, and the context columns fill the framework's
        `TransformContext`, on top of the impact's own context; an asset
        located by `CellColumn` also gives the context's `cell`. Null
        context values keep the impact's own.

        Raises `SampleError` if the asset is missing or appears more than
        once (as crc-sdk's asset joins do), a column is missing, or the value
        is null, negative or not finite; `TypeError` for a file path (this
        module does no I/O).
        """
        _identifier("asset_id", asset_id)
        if isinstance(assets, AssetPortfolio):
            data, id_column, location = assets.data, assets.id_column, assets.location
        else:
            data, id_column, location = assets, "asset_id", None
        table = _arrow_table(data, "the asset table")
        if id_column not in table.column_names:
            raise SampleError(f"the asset table has no id column {id_column!r}")
        ids = table.column(id_column)
        if not (pa.types.is_string(ids.type) or pa.types.is_large_string(ids.type)):
            ids = ids.cast(pa.string())
        rows = table.filter(pc.equal(ids, asset_id))
        if rows.num_rows == 0:
            raise SampleError(f"asset {asset_id!r} is not in the asset table")
        if rows.num_rows > 1:
            raise SampleError(
                f"asset {asset_id!r} appears {rows.num_rows} times in the asset table"
            )
        row = rows.to_pylist()[0]

        def column(name: str) -> Any:
            if name not in row:
                raise SampleError(f"the asset table has no column {name!r}")
            return row[name]

        bound = self._copy()
        bound._asset_id = asset_id
        if self._value_column is not None:
            raw = column(self._value_column)
            if raw is None:
                raise SampleError(
                    f"asset {asset_id!r} has no value in column {self._value_column!r}"
                )
            bound._value = _finite_non_negative(f"the value of asset {asset_id!r}", raw)
        if isinstance(self._context, ImpactContextColumns) and self._context.items():
            base = getattr(self._impact, "context", None)
            fields: dict[str, Any] = {
                name: getattr(base, name)
                if isinstance(base, TransformContext)
                else None
                for name in _CONTEXT_FIELDS
            }
            if isinstance(location, CellColumn):
                cell = column(location.name)
                if cell is not None:
                    fields["cell"] = int(cell)
            for field_name, column_name in self._context.items():
                found = column(column_name)
                if found is None:
                    continue
                fields[field_name] = (
                    float(found) if field_name == "historic_mean" else str(found)
                )
            bound._bound_context = TransformContext(**fields)
        return bound

    def _copy(self) -> LossModel:
        copy = LossModel.__new__(LossModel)
        for slot in LossModel.__slots__:
            setattr(copy, slot, getattr(self, slot))
        return copy

    def _require_bound(self) -> None:
        if self._needs_asset() and self._asset_id is None:
            raise SampleError(
                "the loss model reads asset columns; bind it with "
                "for_asset(assets, asset_id) first"
            )

    def check_hazard_unit(self, unit: str) -> None:
        """Raise `SampleError` unless hazard values in `unit` suit the impact.

        `unit` is the unit of the hazard values the model reads, such as a
        curve's `value_unit` or an `AnnualSeries`' `unit`. It must equal the
        impact's `exposure_unit` as `crc_sdk.impacts.check_exposure` compares
        units: equal after the framework's unit aliases (`metres` is `m`),
        case sensitive, with no conversion. An impact that declares no unit
        (a custom `ImpactFunction`, a callable, or a registry factor whose
        branches disagree on the unit) is not checked, and its document
        records `exposure_unit: None`.
        """
        unit = _identifier("hazard_unit", unit)
        expected = self._exposure_unit
        if expected is not None and not units_equivalent(unit, expected):
            raise SampleError(
                f"the hazard values are in {unit!r}, but impact {self._name!r} is "
                f"calibrated for hazard values in {expected!r}; convert the hazard "
                "or use an impact calibrated in its unit"
            )

    def impact_values(
        self, hazard: ArrayLike, *, hazard_unit: str | None = None
    ) -> FloatArray:
        """Return the impact of each hazard value, usually a damage ratio.

        The result has the hazard's shape. With `hazard_unit`, the unit of
        the values is checked first (`check_hazard_unit`). Raises
        `SampleError` for a unit the impact is not calibrated for, if the
        model needs an asset and is not bound, or if the impact is negative
        or not finite.
        """
        if hazard_unit is not None:
            self.check_hazard_unit(hazard_unit)
        self._require_bound()
        values = np.asarray(hazard, dtype=np.float64)
        if not np.isfinite(values).all():
            raise SampleError("hazard values must be finite")
        impact = np.asarray(
            self._evaluate(values, self._bound_context), dtype=np.float64
        )
        if impact.shape != values.shape:
            raise SampleError(
                f"impact {self._name!r} returned shape {impact.shape} for hazard "
                f"values of shape {values.shape}"
            )
        if not np.isfinite(impact).all() or (impact < 0).any():
            raise SampleError(
                f"impact {self._name!r} returned negative or non-finite values"
            )
        return impact

    def loss_values(self, impact: ArrayLike) -> FloatArray:
        """Return impact × value, in `value_unit`.

        Raises `SampleError` if the value comes from an asset column and the
        model is not bound, or the loss is negative or not finite.
        """
        self._require_bound()
        values = np.asarray(impact, dtype=np.float64)
        value = 1.0 if self._value is None else self._value
        with np.errstate(over="ignore", invalid="ignore"):
            loss = values * value
        if not np.isfinite(loss).all() or (loss < 0).any():
            raise SampleError("losses must be finite and non-negative")
        return loss

    def evaluate(
        self, hazard: ArrayLike, *, hazard_unit: str | None = None
    ) -> FloatArray:
        """Return the loss of each hazard value: `loss_values(impact_values(...))`."""
        return self.loss_values(self.impact_values(hazard, hazard_unit=hazard_unit))

    def _document(self, asset_id: str | None) -> dict[str, Any]:
        if self._value_column is not None:
            source = "column"
        elif self._value is not None:
            source = "constant"
        else:
            source = "none"
        if isinstance(self._context, ImpactContextColumns):
            context: dict[str, Any] | None = {
                "columns": dict(self._context.items()),
                "resolved": _context_document(self._bound_context),
            }
        elif isinstance(self._context, TransformContext):
            context = {"fixed": _context_document(self._context)}
        else:
            context = None
        owner = self._asset_id or asset_id
        value_hash = (
            content_hash({"asset_id": owner, "value": self._value})
            if self._value is not None
            else None
        )
        return {
            "impact": {
                "kind": self._kind,
                "type": self._type,
                "name": self._name,
                "definition_hash": self._hash,
                "exposure_unit": self._exposure_unit,
            },
            "value": {
                "source": source,
                "column": self._value_column,
                "unit": self._value_unit,
                "asset_value_hash": value_hash,
            },
            "context": context,
            "asset_id": owner,
        }

    def to_document(self) -> dict[str, Any]:
        """Return the loss model's provenance document (no asset values)."""
        return self._document(None)

    def __repr__(self) -> str:
        bound = f", asset_id={self._asset_id!r}" if self._asset_id else ""
        return f"LossModel({self._name!r}, unit={self._value_unit!r}{bound})"
