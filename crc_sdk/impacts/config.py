"""Frozen impact definition and registry reference documents.

Document entry points parse with the framework and raise
``crc_framework.ImpactError``. The model is then built from the canonical
document. ``model_dump_json`` is not canonical; use ``to_json``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from crc_framework import ImpactExpression as FrameworkExpression
from crc_framework import RegistryImpactReference as FrameworkReference
from crc_framework import TransformContext, impact_registry_manifest
from pydantic import (
    Field,
    PrivateAttr,
    RootModel,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from crc_sdk.impacts._types import (
    ContentHash,
    FiniteFloat,
    H3Cell,
    Identifier,
    IdentifierPattern,
    NonNegativeFloat,
    PositiveFloat,
    Sha256,
    StrictFrozenModel,
    Text,
    thaw,
)
from crc_sdk.impacts.errors import (
    ImpactCalibrationError,
    ImpactModelError,
    impact_error,
)

RegistryFactor = Literal[
    "daily_freezethaw_cycles",
    "frost_days",
    "rx1day",
    "cyclone",
    "wind_max_daily_max",
    "fwi",
    "hot_days",
    "cflood",
    "rflood",
    "inundation",
    "spei",
    "cooling_degree_days",
    "maximum_consecutive_dry_days",
]
FallbackPolicy = Literal["error", "legacy_defaults"]
_IDENTIFIER = re.compile(IdentifierPattern)


def _finite(value: Any, path: str) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise impact_error("non_finite_number", path, "number is not finite")
    if isinstance(value, dict):
        return {key: _finite(item, f"{path}/{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item, f"{path}/{index}") for index, item in enumerate(value)]
    return value


def _encode(value: Any) -> str:
    return json.dumps(
        _finite(thaw(value), ""),
        allow_nan=False,
        separators=(",", ":"),
    )


class ImpactCalibration(StrictFrozenModel):
    """Declared meaning of a function's argument and response."""

    exposure_variable: Identifier
    exposure_unit: Identifier
    impact_metric: Identifier
    impact_unit: Identifier


class ImpactClamp(StrictFrozenModel):
    """Final numerical limits applied once after the raw response."""

    minimum: FiniteFloat | None = None
    maximum: FiniteFloat | None = None


class LinearModel(StrictFrozenModel):
    kind: Literal["linear"]
    slope: FiniteFloat
    intercept: FiniteFloat = 0.0


class SigmoidModel(StrictFrozenModel):
    kind: Literal["sigmoid"]
    midpoint: FiniteFloat
    steepness: FiniteFloat
    offset: FiniteFloat = 0.0
    amplitude: NonNegativeFloat = 1.0
    origin_adjustment: Literal["none", "subtract_at_zero"] = "none"
    zero_below: FiniteFloat | None = None


class TabulatedPoint(StrictFrozenModel):
    x: FiniteFloat
    y: FiniteFloat


class TabulatedModel(StrictFrozenModel):
    kind: Literal["tabulated"]
    points: tuple[TabulatedPoint, ...] = Field(min_length=2)
    interpolation: Literal["linear", "step_left", "step_right"] = "linear"
    outside_domain: Literal["clamp", "error"] = "clamp"

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_points(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        coerced: list[Any] = []
        for index, point in enumerate(value):
            if isinstance(point, (list, tuple)) and not isinstance(point, Mapping):
                if len(point) != 2:
                    raise impact_error(
                        "invalid_value",
                        f"/model/points/{index}",
                        "a point needs x and y",
                    )
                coerced.append({"x": point[0], "y": point[1]})
            else:
                coerced.append(point)
        return coerced


class BinnedModel(StrictFrozenModel):
    kind: Literal["binned"]
    bin_edges: tuple[FiniteFloat, ...] = Field(min_length=2)
    values: tuple[FiniteFloat, ...] = Field(min_length=1)
    outside_domain: Literal["clamp", "error"] = "clamp"


class SpeiSnapshot(StrictFrozenModel):
    kind: Literal["spei"]
    water_consumption_asset: NonNegativeFloat
    original_water_unit_cost: NonNegativeFloat


class CoolingSnapshot(StrictFrozenModel):
    kind: Literal["cooling"]
    cooling_cost: NonNegativeFloat
    historic_mean: PositiveFloat


class CropVector(StrictFrozenModel):
    maize: FiniteFloat
    rice: FiniteFloat
    wheat: FiniteFloat


class CropWeights(StrictFrozenModel):
    maize: NonNegativeFloat
    rice: NonNegativeFloat
    wheat: NonNegativeFloat


class CropSnapshot(StrictFrozenModel):
    kind: Literal["crop"]
    weights: CropWeights
    country: CropVector
    yearly: CropVector
    index: CropVector
    historic_mean: FiniteFloat


ImpactModel = Annotated[
    LinearModel
    | SigmoidModel
    | TabulatedModel
    | BinnedModel
    | SpeiSnapshot
    | CoolingSnapshot
    | CropSnapshot,
    Field(discriminator="kind"),
]


class UserDefinedOrigin(StrictFrozenModel):
    kind: Literal["user_defined"]


class PublishedOrigin(StrictFrozenModel):
    kind: Literal["published"]
    source_id: Identifier


class ImportedOrigin(StrictFrozenModel):
    kind: Literal["imported"]
    source_id: Identifier
    importer_name: Identifier
    importer_version: Identifier
    importer_build_sha256: Sha256 | None = None


class DerivedOrigin(StrictFrozenModel):
    kind: Literal["derived"]
    description: Text


class ImpactContext(StrictFrozenModel):
    """Registry context. Persisted cells are 15 lowercase hex characters.

    ``supplied`` returns keys that carry a value.
    """

    cell: H3Cell | None = None
    country: Identifier | None = None
    continent: Identifier | None = None
    building_type: Identifier | None = None
    historic_mean: FiniteFloat | None = None

    def supplied(self) -> Mapping[str, str | float]:
        dumped = self.model_dump(mode="json")
        return {
            key: dumped[key] for key in self.model_fields_set if dumped[key] is not None
        }

    @classmethod
    def from_transform_context(cls, context: TransformContext) -> ImpactContext:
        """Copy fields that carry a value."""
        values: dict[str, Any] = {}
        if context.cell is not None:
            values["cell"] = f"{int(context.cell):015x}"
        if context.country is not None:
            values["country"] = context.country
        if context.continent is not None:
            values["continent"] = context.continent
        if context.building_type is not None:
            values["building_type"] = context.building_type
        if context.historic_mean is not None:
            values["historic_mean"] = context.historic_mean
        return cls(**values)


_NULLABLE_OVERRIDE_KEYS = frozenset({"maximum_damage", "minimum_damage", "zero_below"})


class RegistryOverrides(StrictFrozenModel):
    """Supplied override keys. Absence and explicit null are different.

    Python defaults represent absence. The emitted schema does not advertise
    ``null`` defaults, because a missing key is not stored as null.
    """

    cooling_cost: FiniteFloat | None = None
    intercept: FiniteFloat | None = None
    maximum_damage: FiniteFloat | None = None
    midpoint: FiniteFloat | None = None
    minimum_damage: FiniteFloat | None = None
    slope: FiniteFloat | None = None
    steepness: FiniteFloat | None = None
    water_unit_cost: FiniteFloat | None = None
    zero_below: FiniteFloat | None = None

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> dict[str, Any]:
        schema: dict[str, Any] = handler(core_schema)
        nullable = {"anyOf": [{"type": "null"}, {"type": "number"}]}
        number = {"type": "number"}
        for name in schema.get("properties", {}):
            schema["properties"][name] = (
                nullable if name in _NULLABLE_OVERRIDE_KEYS else number
            )
        schema["required"] = []
        return schema

    @model_serializer(mode="wrap")
    def _supplied_only(self, handler: Any) -> dict[str, Any]:
        dumped = handler(self)
        return {key: dumped[key] for key in self.model_fields_set if key in dumped}

    def supplied(self) -> Mapping[str, float | None]:
        dumped = self.model_dump(mode="json")
        return {key: dumped[key] for key in self.model_fields_set}


class CompositeLookupKey(RootModel[tuple[tuple[Identifier, Identifier], ...]]):
    """Sorted string fields of a composite registry lookup key."""

    @model_validator(mode="before")
    @classmethod
    def _from_object(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        if not value:
            raise impact_error(
                "invalid_value",
                "",
                "a composite key needs at least one field",
            )
        pairs: list[tuple[str, str]] = []
        for key, item in value.items():
            if not isinstance(key, str) or _IDENTIFIER.fullmatch(key) is None:
                raise impact_error("invalid_value", f"/{key}", "invalid key")
            if not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None:
                raise impact_error(
                    "invalid_type" if not isinstance(item, str) else "invalid_value",
                    f"/{key}",
                    "invalid composite key value",
                )
            pairs.append((key, item))
        return tuple(sorted(pairs))

    @model_serializer(mode="plain")
    def _as_object(self) -> dict[str, str]:
        return {key: value for key, value in self.root}

    def as_mapping(self) -> Mapping[str, str]:
        from types import MappingProxyType

        return MappingProxyType(dict(self.root))

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: Any, handler: Any
    ) -> dict[str, Any]:
        del core_schema, handler
        identifier = {
            "type": "string",
            "minLength": 1,
            "pattern": IdentifierPattern,
        }
        return {
            "type": "object",
            "minProperties": 1,
            "propertyNames": identifier,
            "additionalProperties": identifier,
        }


class RegistryLookup(StrictFrozenModel):
    table_id: Identifier
    table_sha256: Sha256
    requested_key: Identifier | CompositeLookupKey
    resolved_key: Identifier | CompositeLookupKey
    status: Literal["exact", "fallback"]


class RegistryOrigin(StrictFrozenModel):
    kind: Literal["registry"]
    registry_id: Identifier
    registry_revision: Identifier
    registry_manifest_sha256: Sha256
    resolver_semantics_version: Literal["crc-impact-registry/1"]
    factor: RegistryFactor
    context: ImpactContext
    overrides: RegistryOverrides
    fallback_policy: FallbackPolicy
    lookups: tuple[RegistryLookup, ...]


ImpactOrigin = (
    UserDefinedOrigin
    | PublishedOrigin
    | ImportedOrigin
    | DerivedOrigin
    | RegistryOrigin
)


class ImpactSource(StrictFrozenModel):
    id: Identifier
    kind: Literal["publication", "dataset", "code", "definition"]
    uri: Identifier
    revision: Identifier | None = None
    content_sha256: Sha256 | None = None


class ImpactProvenance(StrictFrozenModel):
    origin: ImpactOrigin = Field(discriminator="kind")
    sources: tuple[ImpactSource, ...] = ()
    derived_from: tuple[ContentHash, ...] = Field(
        default=(),
        json_schema_extra={"uniqueItems": True},
    )

    @field_validator("derived_from")
    @classmethod
    def _sort_parents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(value))


class _NativeDocument(StrictFrozenModel):
    """A document whose canonical form and hash come from the framework."""

    _native: Any = PrivateAttr(default=None)

    @classmethod
    def parse_text(cls, text: str) -> Any:
        raise NotImplementedError

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls._from_native(cls.parse_text(text))

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        return cls._from_native(cls.parse_text(_encode(obj)), **kwargs)

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> Self:
        del kwargs
        if isinstance(json_data, (bytes, bytearray)):
            text = bytes(json_data).decode("utf-8")
        else:
            text = json_data
        return cls.from_json(text)

    @classmethod
    def _from_native(cls, native: Any, **kwargs: Any) -> Self:
        canonical = json.loads(native.to_json())
        try:
            model = super().model_validate(canonical, **kwargs)
        except ValidationError as error:
            raise ImpactModelError(str(error)) from error
        object.__setattr__(model, "_native", native)
        return model

    @model_validator(mode="after")
    def _matches_canonical(self) -> Self:
        native = type(self).parse_text(_encode(self.model_dump(mode="json")))
        canonical = json.loads(native.to_json())
        dumped = self.model_dump(mode="json")
        if dumped != canonical:
            raise ImpactModelError(
                f"{type(self).__name__} dump is not the canonical document"
            )
        object.__setattr__(self, "_native", native)
        return self

    def to_json(self) -> str:
        return str(self._native.to_json())

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if update:
            data = self.model_dump(mode="json")
            data.update(thaw(dict(update)))
            return type(self).model_validate(data)
        return super().model_copy(deep=deep)

    def with_changes(self, **fields: Any) -> Self:
        return self.model_copy(update=fields)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        return hash(self.to_json())

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (type(self).from_json, (self.to_json(),))


class ImpactDefinition(_NativeDocument):
    """Immutable impact function: model, calibration, clamp, and provenance."""

    schema_version: Literal["1.0"]
    semantics_version: Literal["crc-impact/1"]
    calibration: ImpactCalibration
    model: ImpactModel
    clamp: ImpactClamp | None = None
    provenance: ImpactProvenance

    @classmethod
    def parse_text(cls, text: str) -> FrameworkExpression:
        return FrameworkExpression.from_json(text)

    @field_validator("clamp", mode="before")
    @classmethod
    def _normalize_clamp(cls, value: Any) -> Any:
        if value is None:
            return None
        minimum: Any
        maximum: Any
        if isinstance(value, ImpactClamp):
            minimum, maximum = value.minimum, value.maximum
        elif isinstance(value, Mapping):
            minimum = value.get("minimum")
            maximum = value.get("maximum")
        else:
            return value
        if minimum is None and maximum is None:
            return None
        return value

    @property
    def definition_hash(self) -> str:
        return str(self._native.definition_hash)

    def derive(self, description: str, **changes: Any) -> ImpactDefinition:
        """Return a derived definition whose parent is this definition."""
        data = self.model_dump(mode="json")
        data.update({key: thaw(value) for key, value in changes.items()})
        provenance = data["provenance"]
        parents = list(provenance.get("derived_from", []))
        if self.definition_hash not in parents:
            parents.append(self.definition_hash)
        data["provenance"] = {
            "origin": {"kind": "derived", "description": description},
            "sources": provenance.get("sources", []),
            "derived_from": parents,
        }
        return ImpactDefinition.model_validate(data)


class RegistryImpactReference(_NativeDocument):
    """Pinned registry selection. Resolve it to get an impact definition."""

    schema_version: Literal["1.0"]
    registry_id: Identifier
    registry_revision: Identifier
    registry_manifest_sha256: Sha256
    resolver_semantics_version: Literal["crc-impact-registry/1"]
    factor: RegistryFactor
    overrides: RegistryOverrides = RegistryOverrides()
    context_defaults: ImpactContext = ImpactContext()
    fallback_policy: FallbackPolicy = "error"

    @classmethod
    def parse_text(cls, text: str) -> FrameworkReference:
        return FrameworkReference.from_json(text)

    @classmethod
    def pinned(
        cls,
        factor: RegistryFactor | str,
        *,
        overrides: RegistryOverrides | Mapping[str, Any] | None = None,
        context_defaults: (
            ImpactContext | TransformContext | Mapping[str, Any] | None
        ) = None,
        fallback_policy: FallbackPolicy = "error",
    ) -> RegistryImpactReference:
        """Pin the installed registry. Nothing persisted means ``latest``."""
        framework = FrameworkReference.for_factor(
            factor,
            context_defaults=_context_argument(context_defaults),
            overrides=_override_argument(overrides),
            fallback_policy=fallback_policy,
        )
        return cls.from_json(framework.to_json())

    @property
    def reference_hash(self) -> str:
        return str(self._native.reference_hash)

    def resolve(
        self,
        context: ImpactContext | TransformContext | Mapping[str, Any] | None = None,
    ) -> Any:
        from crc_sdk.impacts.expression import ImpactExpression

        framework = self._native.resolve(_context_argument(context))
        return ImpactExpression._wrap(framework)

    def calibration(self, branch: str | None = None) -> ImpactCalibration:
        """Verified calibration for this factor, from the installed manifest."""
        manifest = impact_registry_manifest()
        meta = next(
            (item for item in manifest["factors"] if item["factor"] == self.factor),
            None,
        )
        if meta is None:
            raise ImpactCalibrationError(
                "unknown_exposure_semantics",
                f"factor {self.factor!r} is not in the installed registry manifest",
            )
        branches = dict(meta.get("branch_calibrations") or {})
        if branch is not None:
            if branch not in branches:
                raise ImpactCalibrationError(
                    "unknown_branch",
                    f"{branch} is not a branch of {self.factor}",
                )
            return ImpactCalibration.model_validate(thaw(branches[branch]))
        if branches:
            responses = {
                (item["impact_metric"], item["impact_unit"])
                for item in branches.values()
            }
            if len(responses) > 1:
                raise ImpactCalibrationError(
                    "branch_required",
                    f"{self.factor} response calibration depends on the branch",
                )
        return ImpactCalibration.model_validate(thaw(meta["calibration"]))


def _context_argument(
    value: ImpactContext | TransformContext | Mapping[str, Any] | None,
) -> Any:
    if isinstance(value, ImpactContext):
        return dict(value.supplied())
    if isinstance(value, TransformContext):
        return dict(ImpactContext.from_transform_context(value).supplied())
    if isinstance(value, Mapping):
        return {key: item for key, item in value.items() if item is not None}
    return value


def _override_argument(
    value: RegistryOverrides | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, RegistryOverrides):
        return dict(value.supplied())
    return dict(value)
