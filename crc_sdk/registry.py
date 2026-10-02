"""Optional hazard registry (ADR-0004).

The SDK core is hazard-agnostic. A registry lets a caller attach hazard
knowledge -- units, tail direction, eligibility defaults -- without the fitter
hard-coding any of it. Nothing consults a registry unless one is passed in.

``public_registry()`` returns a seed of the openly defined ETCCDI indices
(and, on request, the OS-Climate hazard indicators). A private catalogue
registers its own specs into a registry of its own and passes that to the
fitting policy. ``HazardRegistry.export_catalog`` publishes only an allowlist
of per-hazard fields, so any new spec field is private by default.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any, Literal

__all__ = [
    "CATALOG_PUBLIC_FIELDS",
    "EligibilityDefaults",
    "HazardRegistry",
    "HazardSpec",
    "os_climate_registry",
    "public_registry",
]

# What a user needs to interpret and trust a curve. Eligibility floors and
# preferred families describe how a curve was tuned and stay private; a field
# is exported only if it is listed here (or named explicitly by the caller).
CATALOG_PUBLIC_FIELDS: tuple[str, ...] = (
    "name",
    "unit",
    "value_semantics",
    "tail",
    "block",
    "aliases",
)


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

    def export_catalog(
        self, fields: Iterable[str] = CATALOG_PUBLIC_FIELDS
    ) -> list[dict[str, Any]]:
        """Export the registry as plain dicts holding only allowlisted fields.

        One dict per hazard, sorted by name. Tuples become lists so the result
        is JSON-ready. ``fields`` must be :class:`HazardSpec` field names;
        anything else raises ``ValueError``. The default
        (:data:`CATALOG_PUBLIC_FIELDS`) omits ``eligibility`` and
        ``preferred_families``, and so does any field added to a spec later.
        """
        allowed = tuple(fields)
        known = {f.name for f in dataclass_fields(HazardSpec)}
        unknown = [name for name in allowed if name not in known]
        if unknown:
            raise ValueError(f"unknown catalogue fields: {unknown}")
        catalog: list[dict[str, Any]] = []
        for spec in sorted(self._specs.values(), key=lambda item: item.name):
            entry: dict[str, Any] = {}
            for name in allowed:
                value = getattr(spec, name)
                if name == "eligibility" and value is not None:
                    value = asdict(value)
                entry[name] = list(value) if isinstance(value, tuple) else value
            catalog.append(entry)
        return catalog

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


def public_registry(*, os_climate: bool = False) -> HazardRegistry:
    """A fresh registry seeded with openly defined ETCCDI indices.

    ``os_climate=True`` also registers :func:`os_climate_registry`.
    """
    registry = HazardRegistry()
    if os_climate:
        registry.register_all(os_climate_registry())
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
            HazardSpec(
                "fd",
                "days",
                "annual count of days with daily minimum temperature below 0 degC",
                block="annual",
                aliases=("FD",),
            ),
            HazardSpec(
                "su",
                "days",
                "annual count of days with daily maximum temperature above 25 degC",
                block="annual",
                aliases=("SU",),
            ),
            HazardSpec(
                "tr",
                "days",
                "annual count of days with daily minimum temperature above 20 degC",
                block="annual",
                aliases=("TR",),
            ),
            HazardSpec(
                "dtr",
                "degC",
                "annual mean of daily temperature range (maximum minus minimum)",
                block="annual",
                aliases=("DTR",),
            ),
            HazardSpec(
                "prcptot",
                "mm",
                "annual total precipitation on wet days (at least 1 mm)",
                block="annual",
                aliases=("PRCPTOT",),
            ),
            HazardSpec(
                "r10mm",
                "days",
                "annual count of days with precipitation of at least 10 mm",
                block="annual",
                aliases=("R10mm",),
            ),
            HazardSpec(
                "r20mm",
                "days",
                "annual count of days with precipitation of at least 20 mm",
                block="annual",
                aliases=("R20mm",),
            ),
            HazardSpec(
                "r95p",
                "mm",
                "annual total precipitation on days above the 95th percentile "
                "of wet days in the 1961-1990 base period",
                block="annual",
                aliases=("R95p",),
            ),
        )
    )
    return registry


def os_climate_registry() -> HazardRegistry:
    """Seed specs for OS-Climate ``hazard_type`` / ``indicator_id`` pairs.

    Names are ``"<hazard_type>/<indicator_id>"`` as they appear in the
    OS-Climate inventory (``crc_sdk.providers.os_climate``); units are those the
    SDK's OS-Climate fixtures and ingest use. Only indicators whose meaning is
    unambiguous from the inventory are listed. The ChronicHeat name keeps the
    inventory's ``{temp_c}`` template for the threshold in degC.
    """
    registry = HazardRegistry()
    registry.register_all(
        (
            HazardSpec(
                "Wind/max_speed",
                "m/s",
                "maximum wind speed",
            ),
            HazardSpec(
                "RiverineInundation/flood_depth",
                "metres",
                "riverine flood inundation depth",
            ),
            HazardSpec(
                "ChronicHeat/days_tas/above/{temp_c}c",
                "days/year",
                "days per year with mean daily air temperature above {temp_c} degC",
            ),
        )
    )
    return registry
