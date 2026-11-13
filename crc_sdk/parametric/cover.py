"""The parametric cover: an index, a payout function and the contract terms."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from crc_sdk.parametric._canonical import canonical_json, content_hash
from crc_sdk.parametric.errors import CoverDefinitionError, UnresolvedParameterError
from crc_sdk.parametric.index import IndexDefinition
from crc_sdk.parametric.parameters import Free, _resolve_values
from crc_sdk.parametric.payouts import PayoutFunction, _index_values, _payout_reader

__all__ = ["COVER_SCHEMA_VERSION", "COVER_SEMANTICS_VERSION", "Cover"]

COVER_SCHEMA_VERSION = "1.0"
"""The `schema_version` of cover documents."""

COVER_SEMANTICS_VERSION = "crc-parametric-cover/1"
"""The `semantics_version` of cover documents."""

_DOCUMENT_KEYS = frozenset(
    {
        "asset_id",
        "index",
        "payout",
        "payout_unit",
        "schema_version",
        "semantics_version",
    }
)
_IDENTIFIER = re.compile(r"^\S(?:[\s\S]*\S)?$")


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CoverDefinitionError(
            f"{name} must be a non-empty string without surrounding whitespace, "
            f"got {value!r}"
        )
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise CoverDefinitionError(f"cover JSON repeats the key {key!r}")
        document[key] = value
    return document


def _reject_constant(name: str) -> NoReturn:
    raise CoverDefinitionError(f"cover JSON must not contain {name}")


class Cover:
    """A parametric cover: what it pays on, how much, and for whom.

    `index` is the `IndexDefinition` the cover pays on and `payout` a payout
    function (`LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`,
    `BinaryPayout`, or a registered user family) whose `tail` equals the
    index's. `payout_unit` is the unit of the payouts, for example a
    currency. `asset_id` names the asset the cover is written for; the same
    terms on another asset are another contract. `label` is a display name
    only.

    A cover with `Free` parameters is a template: it has a document and a
    hash, but `payout_values` raises until `with_values` (or a design)
    resolves it.

    The cover document holds every term except the label, as canonical JSON
    (`to_json`). `definition_hash` is `"sha256:"` plus the SHA-256 of that
    text, the format of `ImpactExpression.definition_hash`. Covers compare
    and hash by `definition_hash`, so two covers with the same terms are
    equal whatever their labels, and they pickle through their JSON.

    Raises `CoverDefinitionError` for an invalid index, payout, unit, asset
    id or label, or a payout whose tail differs from the index's.
    """

    __slots__ = (
        "_asset_id",
        "_document",
        "_hash",
        "_index",
        "_json",
        "_label",
        "_payout",
        "_payout_unit",
    )

    def __init__(
        self,
        index: IndexDefinition,
        payout: PayoutFunction,
        *,
        payout_unit: str,
        asset_id: str | None = None,
        label: str | None = None,
    ) -> None:
        if not isinstance(index, IndexDefinition):
            raise CoverDefinitionError(
                f"index must be an IndexDefinition, got {type(index).__name__}"
            )
        if not isinstance(payout, PayoutFunction):
            raise CoverDefinitionError(
                f"payout must implement PayoutFunction, got {type(payout).__name__}"
            )
        if payout.tail != index.tail:
            raise CoverDefinitionError(
                f"the payout's tail {payout.tail!r} differs from the index's "
                f"tail {index.tail!r}"
            )
        if tuple(payout.index_names) != ("index",):
            raise CoverDefinitionError(
                "a cover reads one index component, index_names == ('index',); "
                f"got {payout.index_names!r}"
            )
        payout_document = payout.to_document()
        if (
            not isinstance(payout_document, Mapping)
            or payout_document.get("kind") != payout.kind
        ):
            raise CoverDefinitionError(
                f"the payout document must be an object whose kind is {payout.kind!r}"
            )
        unit = _identifier("payout_unit", payout_unit)
        asset = None if asset_id is None else _identifier("asset_id", asset_id)
        document: dict[str, Any] = {
            "asset_id": asset,
            "index": index.to_document(),
            "payout": dict(payout_document),
            "payout_unit": unit,
            "schema_version": COVER_SCHEMA_VERSION,
            "semantics_version": COVER_SEMANTICS_VERSION,
        }
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise CoverDefinitionError(
                f"label must be a non-empty string, got {label!r}"
            )
        try:
            text = canonical_json(document)
        except (TypeError, ValueError) as error:
            raise CoverDefinitionError(
                f"the cover document has no canonical JSON form: {error}"
            ) from error
        self._index = index
        self._payout = payout
        self._payout_unit = unit
        self._asset_id = asset
        self._label = label
        self._document = document
        self._json = text
        self._hash = content_hash(document)

    @property
    def index(self) -> IndexDefinition:
        """What the cover pays on."""
        return self._index

    @property
    def payout(self) -> PayoutFunction:
        """The payout function and its parameters."""
        return self._payout

    @property
    def payout_unit(self) -> str:
        """The unit of the payouts."""
        return self._payout_unit

    @property
    def asset_id(self) -> str | None:
        """The asset the cover is written for, if named."""
        return self._asset_id

    @property
    def label(self) -> str | None:
        """The display name; not part of the document or the hash."""
        return self._label

    @property
    def definition_hash(self) -> str:
        """`"sha256:<hex>"` of the canonical cover document."""
        return self._hash

    def free_parameters(self) -> tuple[str, ...]:
        """The names of the free parameters, in the payout's order."""
        return tuple(
            name
            for name, value in self._payout.parameters().items()
            if isinstance(value, Free)
        )

    def with_values(self, values: Mapping[str, float]) -> Cover:
        """Return the cover with values for some or all free parameters.

        Each name must be a free parameter and each value a finite number
        within its bounds. The index, unit, asset and label are kept. Raises
        `CoverDefinitionError` otherwise, or if the resolved payout is
        invalid (for example, unordered thresholds).
        """
        checked = _resolve_values(self._payout.parameters(), values)
        return Cover(
            self._index,
            self._payout.with_values(checked),
            payout_unit=self._payout_unit,
            asset_id=self._asset_id,
            label=self._label,
        )

    def payout_values(self, index_values: ArrayLike) -> NDArray[np.float64]:
        """Return the payout for each index value, in `payout_unit`.

        The result has the shape of `index_values`. Raises
        `UnresolvedParameterError` on a template, and `SampleError` for
        non-numeric or non-finite index values.
        """
        free = self.free_parameters()
        if free:
            raise UnresolvedParameterError(
                f"the cover is a template; resolve its free parameters first: "
                f"{', '.join(free)}"
            )
        values = _index_values(index_values)
        payout = np.asarray(self._payout.evaluate(values), dtype=np.float64)
        if payout.shape != values.shape:
            raise CoverDefinitionError(
                f"payout {self._payout.kind!r} returned shape {payout.shape} "
                f"for index values of shape {values.shape}"
            )
        if not np.isfinite(payout).all() or (payout < 0).any():
            raise CoverDefinitionError(
                f"payout {self._payout.kind!r} returned negative or non-finite values"
            )
        return payout

    def to_document(self) -> dict[str, Any]:
        """Return a copy of the cover document, as plain JSON types."""
        return copy.deepcopy(self._document)

    def to_json(self) -> str:
        """Return the canonical JSON text of the cover document."""
        return self._json

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], *, label: str | None = None
    ) -> Cover:
        """Read a cover document, as `to_document` returns it.

        The payout's `kind` is looked up among the registered payout
        classes. Raises `UnknownPayoutKindError` for an unregistered kind,
        and `CoverDefinitionError` for missing or unknown keys, another
        schema or semantics version, or invalid terms.
        """
        if not isinstance(document, Mapping):
            raise CoverDefinitionError("a cover document must be an object")
        keys = set(document)
        if keys != _DOCUMENT_KEYS:
            missing = ", ".join(sorted(_DOCUMENT_KEYS - keys)) or "none"
            unknown = ", ".join(sorted(map(str, keys - _DOCUMENT_KEYS))) or "none"
            raise CoverDefinitionError(
                f"a cover document has the keys {sorted(_DOCUMENT_KEYS)}; "
                f"missing: {missing}; unknown: {unknown}"
            )
        if document["schema_version"] != COVER_SCHEMA_VERSION:
            raise CoverDefinitionError(
                f"unsupported cover schema_version {document['schema_version']!r}"
            )
        if document["semantics_version"] != COVER_SEMANTICS_VERSION:
            raise CoverDefinitionError(
                f"unsupported cover semantics_version {document['semantics_version']!r}"
            )
        payout_document = document["payout"]
        if not isinstance(payout_document, Mapping):
            raise CoverDefinitionError("a payout document must be an object")
        payout = _payout_reader(payout_document.get("kind"))(payout_document)
        return cls(
            IndexDefinition.from_document(document["index"]),
            payout,
            payout_unit=document["payout_unit"],
            asset_id=document["asset_id"],
            label=label,
        )

    @classmethod
    def from_json(cls, text: str, *, label: str | None = None) -> Cover:
        """Read a cover from its JSON text, as `to_json` writes it.

        `label` sets the display name, which the JSON does not hold. Raises
        `CoverDefinitionError` for invalid JSON (including repeated keys,
        NaN and infinity) and as `from_document` does.
        """
        if not isinstance(text, str):
            raise CoverDefinitionError(f"cover JSON must be a str, got {type(text)}")
        try:
            document = json.loads(
                text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except json.JSONDecodeError as error:
            raise CoverDefinitionError(f"cover JSON is not valid: {error}") from error
        return cls.from_document(document, label=label)

    @classmethod
    def _from_pickle(cls, text: str, label: str | None) -> Cover:
        return cls.from_json(text, label=label)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cover):
            return NotImplemented
        return self._hash == other._hash

    def __hash__(self) -> int:
        return hash(self._hash)

    def __reduce__(self) -> tuple[Any, tuple[str, str | None]]:
        return (type(self)._from_pickle, (self._json, self._label))

    def __repr__(self) -> str:
        label = f"{self._label!r}, " if self._label is not None else ""
        return f"Cover({label}{self._payout.kind}, {self._hash})"
