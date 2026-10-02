"""Pathway label crosswalk (ADR-0002).

Pathway labels are the source's own: SSPs, warming-level bands,
``SV``/``RT3``/``Hot House``/``Paris``/``NDC``, and a baseline spelled
``historic`` (crc-framework) or ``historical`` (EDO). This module maps every
known spelling to one canonical id and, where crc-framework has one, to its
``PATHWAYS`` id. It never rewrites labels: writes keep the source spelling,
and :func:`resolve_pathway` returns the label exactly as it was given.

Matching ignores case and whitespace. Unknown labels warn
(:class:`UnknownPathwayWarning`); ``strict=True`` rejects them.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

from crc_framework import PATHWAYS

__all__ = [
    "PathwayResolution",
    "UnknownPathwayWarning",
    "pathways_equivalent",
    "register_pathway",
    "resolve_pathway",
]


class UnknownPathwayWarning(UserWarning):
    """A pathway label is not in the crosswalk registry."""


@dataclass(frozen=True)
class PathwayResolution:
    """The outcome of resolving one pathway label.

    ``label`` is the caller's spelling, untouched. ``framework_id`` is the
    crc-framework ``PATHWAYS`` id, or ``None`` when the framework has no such
    pathway (or the label is unknown).
    """

    label: str
    canonical_id: str
    framework_id: int | None
    known: bool


def _key(label: str) -> str:
    return re.sub(r"\s+", "", label).casefold()


# normalised key -> (canonical id, framework id)
_REGISTRY: dict[str, tuple[str, int | None]] = {}


def _add(canonical: str, framework_id: int | None, aliases: Iterable[str]) -> None:
    labels = (canonical, *aliases)
    # Validate every label first so a clash leaves the registry untouched.
    for label in labels:
        existing = _REGISTRY.get(_key(label))
        if existing is not None and existing[0] != canonical:
            raise ValueError(f"{label!r} is already registered for {existing[0]!r}")
    for label in labels:
        _REGISTRY[_key(label)] = (canonical, framework_id)


def register_pathway(
    canonical_id: str,
    *,
    framework_id: int | None = None,
    aliases: Iterable[str] = (),
) -> None:
    """Register a private pathway (or extra aliases of an existing one).

    A label already owned by a different canonical id raises ``ValueError``;
    re-registering the same canonical id only adds aliases.
    """
    if not canonical_id.strip():
        raise ValueError("canonical_id must be non-empty")
    existing = _REGISTRY.get(_key(canonical_id))
    if existing is not None and framework_id is None:
        framework_id = existing[1]
    _add(canonical_id.strip(), framework_id, aliases)


# crc-framework spellings are the canonical ids, so `historic` is canonical
# and `historical` its alias.
for _pathway in PATHWAYS:
    _add(_pathway.name, _pathway.id, ())
_add("historic", _REGISTRY[_key("historic")][1], ("historical",))
# Not in crc-framework PATHWAYS: known to the SDK but with no framework id.
_add("ssp534", None, ("ssp534-over", "ssp534over"))
_add("RT3", None, ())


def resolve_pathway(label: str, *, strict: bool = False) -> PathwayResolution:
    """Resolve a pathway label to its canonical id and framework id.

    Unknown labels warn with :class:`UnknownPathwayWarning` and resolve to
    themselves (stripped) with ``known=False``; ``strict=True`` raises
    ``ValueError`` instead.
    """
    if not isinstance(label, str):
        raise TypeError("pathway label must be a string")
    entry = _REGISTRY.get(_key(label))
    if entry is not None:
        return PathwayResolution(label, entry[0], entry[1], True)
    message = f"unknown pathway label {label!r}"
    if strict:
        raise ValueError(message)
    warnings.warn(message, UnknownPathwayWarning, stacklevel=2)
    return PathwayResolution(label, label.strip(), None, False)


def pathways_equivalent(a: str, b: str) -> bool:
    """Whether two labels denote the same pathway (never warns).

    Unknown labels are equivalent only when they match ignoring case and
    whitespace.
    """
    ea, eb = _REGISTRY.get(_key(a)), _REGISTRY.get(_key(b))
    if ea is None or eb is None:
        return _key(a) == _key(b)
    return ea[0] == eb[0]
