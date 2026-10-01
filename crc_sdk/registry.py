"""Optional hazard registry (ADR-0004).

The SDK core is hazard-agnostic. A registry lets a caller attach hazard
knowledge -- units, tail direction, eligibility defaults -- without the fitter
hard-coding any of it. Nothing consults a registry unless one is passed in.

``public_registry()`` returns a seed of the openly defined ETCCDI indices. A
private catalogue registers its own specs into a
registry of its own and passes that to the fitting policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "EligibilityDefaults",
    "HazardRegistry",
    "HazardSpec",
    "public_registry",
]


@dataclass(frozen=True)
class EligibilityDefaults:
    """Per-hazard defaults for the informative-support screens.

    ``minimum_informative_value=None`` means *no floor* for this hazard -- the
    value population is signed or tiny-scaled, and a shared positive floor
    would exclude the whole distribution. ``None`` for the two count fields
    means "no opinion; use the fit policy's value".
    """

    minimum_informative_value: float | None = None
    minimum_informative_knots: int | None = None
    minimum_distinct_informative_values: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "minimum_informative_knots",
            "minimum_distinct_informative_values",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class HazardSpec:
    """What the registry knows about one hazard name."""

    name: str
    unit: str
    value_semantics: str
    tail: Literal["upper", "lower"] = "upper"
    block: str | None = None
    aliases: tuple[str, ...] = ()
    eligibility: EligibilityDefaults | None = None
    preferred_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not self.value_semantics:
            raise ValueError("name, unit and value_semantics must be non-empty")
        if self.tail not in ("upper", "lower"):
            raise ValueError("tail must be 'upper' or 'lower'")
        if self.name in self.aliases:
            raise ValueError("a spec must not alias its own name")


@dataclass
class HazardRegistry:
    """Mutable name -> :class:`HazardSpec` mapping with alias resolution."""

    _specs: dict[str, HazardSpec] = field(default_factory=dict)
    _aliases: dict[str, str] = field(default_factory=dict)

    def register(self, spec: HazardSpec, *, replace: bool = False) -> HazardSpec:
        taken = {spec.name, *spec.aliases}
        for label in taken:
            owner = self._aliases.get(label) or (
                label if label in self._specs else None
            )
            # ``replace`` swaps a spec of the same name; it never takes over a
            # label (name or alias) that belongs to a different spec.
            if owner is not None and owner != spec.name:
                raise ValueError(f"{label!r} is already registered for {owner!r}")
        if spec.name in self._specs and not replace:
            raise ValueError(f"hazard {spec.name!r} is already registered")
        if replace:
            self.unregister(spec.name)
        self._specs[spec.name] = spec
        for alias in spec.aliases:
            self._aliases[alias] = spec.name
        return spec

    def register_all(
        self, specs: Iterable[HazardSpec], *, replace: bool = False
    ) -> None:
        for spec in specs:
            self.register(spec, replace=replace)

    def unregister(self, name: str) -> None:
        spec = self._specs.pop(name, None)
        if spec is not None:
            for alias in spec.aliases:
                if self._aliases.get(alias) == name:
                    del self._aliases[alias]

    def get(self, name: str) -> HazardSpec | None:
        """Return the spec for a name or alias, or ``None`` if unknown."""
        if name in self._specs:
            return self._specs[name]
        canonical = self._aliases.get(name)
        return self._specs[canonical] if canonical is not None else None

    def resolve(self, name: str) -> HazardSpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"unknown hazard {name!r}")
        return spec

    def eligibility(self, name: str) -> EligibilityDefaults | None:
        spec = self.get(name)
        return None if spec is None else spec.eligibility

    def merged(self, other: HazardRegistry) -> HazardRegistry:
        """Return a new registry holding both; clashing names raise."""
        result = HazardRegistry()
        result.register_all(self)
        result.register_all(other)
        return result

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __iter__(self) -> Iterator[HazardSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


# Signed temperature extremes legitimately go below any positive floor (polar
# TXx, mid-latitude TNn), so a shared eligibility floor must not apply.
_SIGNED = EligibilityDefaults(minimum_informative_value=None)


def public_registry() -> HazardRegistry:
    """A fresh registry seeded with openly defined ETCCDI indices."""
    registry = HazardRegistry()
    registry.register_all(
        (
            HazardSpec(
                "txx",
                "degC",
                "annual maximum of daily maximum temperature",
                block="annual",
                aliases=("TXx",),
                eligibility=_SIGNED,
            ),
            HazardSpec(
                "txn",
                "degC",
                "annual minimum of daily maximum temperature",
                tail="lower",
                block="annual",
                aliases=("TXn",),
                eligibility=_SIGNED,
            ),
            HazardSpec(
                "tnx",
                "degC",
                "annual maximum of daily minimum temperature",
                block="annual",
                aliases=("TNx",),
                eligibility=_SIGNED,
            ),
            HazardSpec(
                "tnn",
                "degC",
                "annual minimum of daily minimum temperature",
                tail="lower",
                block="annual",
                aliases=("TNn",),
                eligibility=_SIGNED,
            ),
            HazardSpec(
                "rx1day",
                "mm/day",
                "annual maximum one-day precipitation",
                block="annual",
                aliases=("Rx1day",),
            ),
            HazardSpec(
                "rx5day",
                "mm",
                "annual maximum five-day precipitation",
                block="annual",
                aliases=("Rx5day",),
            ),
            HazardSpec(
                "sdii",
                "mm/day",
                "annual mean precipitation on wet days",
                block="annual",
                aliases=("SDII",),
            ),
            HazardSpec(
                "cdd",
                "days",
                "annual maximum consecutive dry days",
                block="annual",
                aliases=("CDD",),
            ),
            HazardSpec(
                "cwd",
                "days",
                "annual maximum consecutive wet days",
                block="annual",
                aliases=("CWD",),
            ),
        )
    )
    return registry
