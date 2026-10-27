"""SDK facade over the framework impact expression.

Numerical work, canonical JSON, and definition hashes belong to the framework.
``to_config`` returns the frozen SDK definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from crc_framework import (
    ImpactEvaluationRecord,
    TabulatedDistribution,
    TransformContext,
)
from crc_framework import ImpactExpression as FrameworkExpression
from crc_framework.distributions import Distribution
from crc_framework.transforms.base import ImpactResult, ImpactValues

from crc_sdk.impacts._types import thaw
from crc_sdk.impacts.config import (
    ImpactCalibration,
    ImpactClamp,
    ImpactContext,
    ImpactDefinition,
    ImpactProvenance,
    RegistryOverrides,
    TabulatedPoint,
)

T = TypeVar("T")


@dataclass(frozen=True)
class ImpactEvaluationResult(Generic[T]):
    """A value and the record of how it was obtained."""

    value: T
    record: ImpactEvaluationRecord


def _context_argument(value: Any) -> Any:
    if isinstance(value, ImpactContext):
        return value.model_dump(mode="json")
    return value


def _override_argument(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, RegistryOverrides):
        return dict(value.supplied())
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _points(points: Sequence[Any]) -> list[Any]:
    converted: list[Any] = []
    for point in points:
        if isinstance(point, TabulatedPoint):
            converted.append(point.model_dump(mode="json"))
        elif isinstance(point, Mapping):
            converted.append(dict(point))
        else:
            converted.append({"x": point[0], "y": point[1]})
    return converted


def _definition(
    model: Mapping[str, Any],
    calibration: ImpactCalibration | Mapping[str, str],
    clamp: ImpactClamp | Mapping[str, Any] | None,
    provenance: ImpactProvenance | Mapping[str, Any] | None,
) -> ImpactDefinition:
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "semantics_version": "crc-impact/1",
        "calibration": thaw(calibration),
        "model": thaw(model),
        "provenance": (
            thaw(provenance)
            if provenance is not None
            else {"origin": {"kind": "user_defined"}}
        ),
    }
    if clamp is not None:
        document["clamp"] = thaw(clamp)
    return ImpactDefinition.model_validate(document)


class ImpactExpression:
    """Typed facade. Evaluation and hashing delegate to the framework."""

    def __init__(
        self, framework: FrameworkExpression, definition: ImpactDefinition
    ) -> None:
        self._framework = framework
        self._definition = definition

    @classmethod
    def _wrap(cls, framework: FrameworkExpression) -> ImpactExpression:
        definition = ImpactDefinition.from_json(framework.to_json())
        return cls(framework, definition)

    @classmethod
    def _from_definition(cls, definition: ImpactDefinition) -> ImpactExpression:
        return cls(definition._native, definition)

    @classmethod
    def linear(
        cls,
        slope: float,
        *,
        intercept: float = 0.0,
        calibration: ImpactCalibration | Mapping[str, str],
        clamp: ImpactClamp | Mapping[str, Any] | None = None,
        provenance: ImpactProvenance | Mapping[str, Any] | None = None,
    ) -> ImpactExpression:
        return cls._from_definition(
            _definition(
                {"kind": "linear", "slope": slope, "intercept": intercept},
                calibration,
                clamp,
                provenance,
            )
        )

    @classmethod
    def sigmoid(
        cls,
        *,
        midpoint: float,
        steepness: float,
        calibration: ImpactCalibration | Mapping[str, str],
        offset: float = 0.0,
        amplitude: float = 1.0,
        origin_adjustment: str = "none",
        zero_below: float | None = None,
        clamp: ImpactClamp | Mapping[str, Any] | None = None,
        provenance: ImpactProvenance | Mapping[str, Any] | None = None,
    ) -> ImpactExpression:
        return cls._from_definition(
            _definition(
                {
                    "kind": "sigmoid",
                    "midpoint": midpoint,
                    "steepness": steepness,
                    "offset": offset,
                    "amplitude": amplitude,
                    "origin_adjustment": origin_adjustment,
                    "zero_below": zero_below,
                },
                calibration,
                clamp,
                provenance,
            )
        )

    @classmethod
    def tabulated(
        cls,
        points: Sequence[Any],
        *,
        calibration: ImpactCalibration | Mapping[str, str],
        interpolation: str = "linear",
        outside_domain: str = "clamp",
        clamp: ImpactClamp | Mapping[str, Any] | None = None,
        provenance: ImpactProvenance | Mapping[str, Any] | None = None,
    ) -> ImpactExpression:
        return cls._from_definition(
            _definition(
                {
                    "kind": "tabulated",
                    "points": _points(points),
                    "interpolation": interpolation,
                    "outside_domain": outside_domain,
                },
                calibration,
                clamp,
                provenance,
            )
        )

    @classmethod
    def binned(
        cls,
        bin_edges: Sequence[float],
        values: Sequence[float],
        *,
        calibration: ImpactCalibration | Mapping[str, str],
        outside_domain: str = "clamp",
        clamp: ImpactClamp | Mapping[str, Any] | None = None,
        provenance: ImpactProvenance | Mapping[str, Any] | None = None,
    ) -> ImpactExpression:
        return cls._from_definition(
            _definition(
                {
                    "kind": "binned",
                    "bin_edges": [float(edge) for edge in bin_edges],
                    "values": [float(value) for value in values],
                    "outside_domain": outside_domain,
                },
                calibration,
                clamp,
                provenance,
            )
        )

    @classmethod
    def for_factor(
        cls,
        factor: Any,
        *,
        context: Any = None,
        context_defaults: Any = None,
        overrides: Any = None,
        fallback_policy: str = "error",
        registry_revision: str | None = None,
        registry_manifest_sha256: str | None = None,
    ) -> ImpactExpression:
        framework = FrameworkExpression.for_factor(
            factor,
            context=_context_argument(context),
            context_defaults=_context_argument(context_defaults),
            overrides=_override_argument(overrides),
            fallback_policy=fallback_policy,
            registry_revision=registry_revision,
            registry_manifest_sha256=registry_manifest_sha256,
        )
        return cls._wrap(framework)

    @classmethod
    def from_config(
        cls, config: ImpactDefinition | Mapping[str, Any]
    ) -> ImpactExpression:
        if isinstance(config, ImpactDefinition):
            definition = config
        else:
            definition = ImpactDefinition.model_validate(config)
        return cls._from_definition(definition)

    @classmethod
    def from_json(cls, text: str) -> ImpactExpression:
        return cls._from_definition(ImpactDefinition.from_json(text))

    def to_config(self) -> ImpactDefinition:
        return self._definition

    def to_json(self) -> str:
        return self._framework.to_json()

    @property
    def definition_hash(self) -> str:
        return self._framework.definition_hash

    @property
    def model_kind(self) -> str:
        return self._framework.model_kind

    @property
    def monotonicity(self) -> str:
        return self._framework.monotonicity

    @property
    def calibration(self) -> ImpactCalibration:
        return self._definition.calibration

    @property
    def provenance(self) -> ImpactProvenance:
        return self._definition.provenance

    @property
    def framework_expression(self) -> FrameworkExpression:
        return self._framework

    def evaluate(self, values: ImpactValues) -> ImpactResult:
        return self._framework.evaluate(values)

    def evaluate_with_record(
        self, values: ImpactValues, *, source: Mapping[str, Any] | None = None
    ) -> ImpactEvaluationResult[ImpactResult]:
        recorded = self._framework.evaluate_with_record(values, source=source)
        return ImpactEvaluationResult(recorded.value, recorded.record)

    def expected_value(self, source: Any, *, method: Any = "auto") -> float:
        return float(self._framework.expected_value(source, method=method))

    def expected_value_with_record(
        self,
        source: Any,
        *,
        method: Any = "auto",
        snapshot: Mapping[str, Any] | None = None,
    ) -> ImpactEvaluationResult[float]:
        recorded = self._framework.expected_value_with_record(
            source, method=method, snapshot=snapshot
        )
        return ImpactEvaluationResult(float(recorded.value), recorded.record)

    def transform_distribution(self, source: Any) -> Any:
        return self._framework.transform_distribution(source)

    def transform_distribution_with_record(
        self, source: Any, *, snapshot: Mapping[str, Any] | None = None
    ) -> ImpactEvaluationResult[Any]:
        recorded = self._framework.transform_distribution_with_record(
            source, snapshot=snapshot
        )
        return ImpactEvaluationResult(recorded.value, recorded.record)

    def __call__(
        self,
        distribution: Distribution,
        *,
        probabilities: Sequence[float] | None = None,
        context: TransformContext | None = None,
    ) -> TabulatedDistribution:
        return self._framework(
            distribution, probabilities=probabilities, context=context
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImpactExpression):
            return NotImplemented
        return self.definition_hash == other.definition_hash

    def __hash__(self) -> int:
        return hash(self.definition_hash)

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (type(self).from_json, (self.to_json(),))

    def __repr__(self) -> str:
        return f"ImpactExpression({self.model_kind}, {self.definition_hash})"
