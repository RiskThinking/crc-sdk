"""Covers: payouts, templates, the cover document, its JSON and its hash."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from crc_sdk.parametric import (
    AtCell,
    AtPoint,
    AtStation,
    BinaryPayout,
    Cover,
    CoverDefinitionError,
    Free,
    IndexDefinition,
    LinearPayout,
    PiecewiseLinearPayout,
    SampleError,
    StepPayout,
    UnknownPayoutKindError,
    UnresolvedParameterError,
    register_payout,
)
from crc_sdk.parametric.payouts import _REGISTRY

FloatArray = NDArray[np.float64]
UNTYPED_COVER: Any = Cover  # for calls with deliberately invalid arguments

RX1DAY = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
SPEI = IndexDefinition(
    "spei",
    "1",
    tail="lower",
    location=AtPoint(-0.1276, 51.5072, "london"),
    aggregation="annual_minimum",
    source="era5",
)
TEMPLATE = Cover(
    RX1DAY,
    StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
    payout_unit="EUR",
    asset_id="warehouse-a",
)
RESOLVED = TEMPLATE.with_values({"levels[0]": 1e3, "levels[1]": 5e3, "levels[2]": 2e4})

# Pinned: a change to the document layout or the number spelling changes every
# stored hash, so it must be deliberate (a new semantics_version).
TEMPLATE_JSON = (
    '{"asset_id":"warehouse-a","index":{"aggregation":"annual_maximum",'
    '"hazard_name":"rx1day","location":{"kind":"asset"},"source":null,'
    '"tail":"upper","unit":"mm/day"},"payout":{"breakpoints":[60.0,80.0,100.0],'
    '"kind":"step","levels":[{"free":{"grid":null,"lower":0.0,"upper":null}},'
    '{"free":{"grid":null,"lower":0.0,"upper":null}},'
    '{"free":{"grid":null,"lower":0.0,"upper":null}}],"monotone":true,'
    '"tail":"upper"},"payout_unit":"EUR","schema_version":"1.0",'
    '"semantics_version":"crc-parametric-cover/1"}'
)
TEMPLATE_HASH = (
    "sha256:999d2ad0b657c7bb42c60b48c8d18387c9c69924096c1113d85d44a3c69a3f29"
)
LOWER_TAIL = Cover(
    SPEI,
    PiecewiseLinearPayout([-1.0, -1.5, -2.0], [1e-7, 2.5e5, 1e16], tail="lower"),
    payout_unit="GBP",
    asset_id="farm-7",
)
LOWER_TAIL_JSON = (
    '{"asset_id":"farm-7","index":{"aggregation":"annual_minimum",'
    '"hazard_name":"spei","location":{"kind":"point","latitude":51.5072,'
    '"longitude":-0.1276,"name":"london"},"source":"era5","tail":"lower",'
    '"unit":"1"},"payout":{"kind":"piecewise_linear","knots":[-1.0,-1.5,-2.0],'
    '"levels":[1e-7,250000.0,1e+16],"monotone":true,"tail":"lower"},'
    '"payout_unit":"GBP","schema_version":"1.0",'
    '"semantics_version":"crc-parametric-cover/1"}'
)
LOWER_TAIL_HASH = (
    "sha256:e60f7a393b4f5cef855872d1f4a69225f01471133c57413cdc2232b68cfcc070"
)


def test_the_cover_document() -> None:
    free = {"free": {"grid": None, "lower": 0.0, "upper": None}}
    assert TEMPLATE.to_document() == {
        "asset_id": "warehouse-a",
        "index": {
            "aggregation": "annual_maximum",
            "hazard_name": "rx1day",
            "location": {"kind": "asset"},
            "source": None,
            "tail": "upper",
            "unit": "mm/day",
        },
        "payout": {
            "breakpoints": [60.0, 80.0, 100.0],
            "kind": "step",
            "levels": [free, free, free],
            "monotone": True,
            "tail": "upper",
        },
        "payout_unit": "EUR",
        "schema_version": "1.0",
        "semantics_version": "crc-parametric-cover/1",
    }


@pytest.mark.parametrize(
    ("cover", "text", "digest"),
    [
        pytest.param(TEMPLATE, TEMPLATE_JSON, TEMPLATE_HASH, id="template"),
        pytest.param(LOWER_TAIL, LOWER_TAIL_JSON, LOWER_TAIL_HASH, id="exponents"),
    ],
)
def test_canonical_json_and_hash_are_stable(
    cover: Cover, text: str, digest: str
) -> None:
    assert cover.to_json() == text
    assert cover.definition_hash == digest
    assert digest == "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


COVERS = [
    TEMPLATE,
    RESOLVED,
    LOWER_TAIL,
    Cover(
        IndexDefinition(
            "wind_max_daily_max",
            "m/s",
            location=AtStation("USW00094728"),
            aggregation="annual_maximum",
        ),
        LinearPayout(Free(20.0, 40.0, grid=5), 50.0, Free(upper=1e6)),
        payout_unit="USD",
    ),
    Cover(
        IndexDefinition(
            "rflood",
            "m",
            location=AtCell(0x8A2A1072B59FFFF),
            aggregation="annual_maximum",
        ),
        BinaryPayout(0.5, 1.0),
        payout_unit="1",
        asset_id="site-12",
        label="flood trigger",
    ),
    Cover(
        SPEI,
        StepPayout([-1.0, -2.0], [Free(), Free()], tail="lower", monotone=False),
        payout_unit="EUR",
    ),
]


@pytest.mark.parametrize("cover", COVERS, ids=lambda c: c.payout.kind)
def test_json_and_document_round_trips(cover: Cover) -> None:
    restored = Cover.from_json(cover.to_json())
    assert restored == cover
    assert restored.to_json() == cover.to_json()
    assert restored.definition_hash == cover.definition_hash
    assert restored.payout == cover.payout
    assert restored.index == cover.index
    assert (restored.payout_unit, restored.asset_id) == (
        cover.payout_unit,
        cover.asset_id,
    )
    assert Cover.from_document(cover.to_document()) == cover


@pytest.mark.parametrize("cover", COVERS, ids=lambda c: c.payout.kind)
def test_pickles_through_json_and_keeps_the_label(cover: Cover) -> None:
    restored = pickle.loads(pickle.dumps(cover))
    assert restored == cover
    assert restored.label == cover.label
    assert restored.to_json() == cover.to_json()


def test_non_canonical_json_reads_to_the_same_cover() -> None:
    document = json.loads(TEMPLATE_JSON)
    document["payout"]["breakpoints"] = [60, 80, 100]
    text = json.dumps(document, indent=2, sort_keys=False)
    restored = Cover.from_json(text)
    assert restored.to_json() == TEMPLATE_JSON
    assert restored.definition_hash == TEMPLATE_HASH


def test_the_label_is_not_in_the_hash_and_the_asset_is() -> None:
    labelled = Cover(
        RX1DAY,
        TEMPLATE.payout,
        payout_unit="EUR",
        asset_id="warehouse-a",
        label="10-year flood step",
    )
    assert labelled == TEMPLATE
    assert labelled.definition_hash == TEMPLATE.definition_hash
    assert labelled.label == "10-year flood step"
    assert "label" not in labelled.to_json()
    assert Cover.from_json(labelled.to_json()).label is None
    assert Cover.from_json(labelled.to_json(), label="x").label == "x"
    other_asset = Cover(
        RX1DAY, TEMPLATE.payout, payout_unit="EUR", asset_id="warehouse-b"
    )
    no_asset = Cover(RX1DAY, TEMPLATE.payout, payout_unit="EUR")
    assert len({TEMPLATE, other_asset, no_asset}) == 3


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(
            Cover(RX1DAY, TEMPLATE.payout, payout_unit="USD", asset_id="warehouse-a"),
            id="unit",
        ),
        pytest.param(
            Cover(
                IndexDefinition("rx1day", "mm", aggregation="annual_maximum"),
                TEMPLATE.payout,
                payout_unit="EUR",
                asset_id="warehouse-a",
            ),
            id="index-unit",
        ),
        pytest.param(
            Cover(
                IndexDefinition(
                    "rx1day",
                    "mm/day",
                    location=AtPoint(7.0, 51.0),
                    aggregation="annual_maximum",
                ),
                TEMPLATE.payout,
                payout_unit="EUR",
                asset_id="warehouse-a",
            ),
            id="location",
        ),
        pytest.param(
            Cover(
                RX1DAY,
                StepPayout([60.0, 80.0, 100.5], [Free(), Free(), Free()]),
                payout_unit="EUR",
                asset_id="warehouse-a",
            ),
            id="breakpoint",
        ),
        pytest.param(
            Cover(
                RX1DAY,
                StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free(upper=1.0)]),
                payout_unit="EUR",
                asset_id="warehouse-a",
            ),
            id="bound",
        ),
        pytest.param(
            Cover(
                RX1DAY,
                StepPayout(
                    [60.0, 80.0, 100.0], [Free(), Free(), Free()], monotone=False
                ),
                payout_unit="EUR",
                asset_id="warehouse-a",
            ),
            id="monotone",
        ),
    ],
)
def test_every_term_is_in_the_hash(changed: Cover) -> None:
    assert changed != TEMPLATE
    assert changed.definition_hash != TEMPLATE.definition_hash


def test_equality_with_other_types() -> None:
    assert TEMPLATE != TEMPLATE.to_json()
    assert TEMPLATE.__eq__(object()) is NotImplemented


def test_repr_names_the_kind_and_the_hash() -> None:
    assert repr(TEMPLATE) == f"Cover(step, {TEMPLATE_HASH})"
    labelled = Cover(RX1DAY, TEMPLATE.payout, payout_unit="EUR", label="a")
    assert repr(labelled).startswith("Cover('a', step, sha256:")


# -- payouts and templates -------------------------------------------------------


def test_payout_values() -> None:
    values = [0.0, 59.9, 60.0, 79.9, 80.0, 99.9, 100.0, 250.0]
    paid = RESOLVED.payout_values(values)
    assert paid.dtype == np.float64
    assert paid.tolist() == [0.0, 0.0, 1e3, 1e3, 5e3, 5e3, 2e4, 2e4]
    grid = np.array([[0.0, 60.0], [80.0, 100.0]])
    assert RESOLVED.payout_values(grid).tolist() == [[0.0, 1e3], [5e3, 2e4]]
    assert LOWER_TAIL.payout_values([-0.5, -1.0, -2.0, -3.0]).tolist() == [
        0.0,
        1e-7,
        1e16,
        1e16,
    ]


@pytest.mark.parametrize(
    "bad", [[math.nan], [math.inf], ["sixty"], [[1.0], [2.0, 3.0]]]
)
def test_invalid_index_values_raise(bad: Any) -> None:
    with pytest.raises(SampleError):
        RESOLVED.payout_values(bad)


def test_templates_do_not_pay() -> None:
    assert TEMPLATE.free_parameters() == ("levels[0]", "levels[1]", "levels[2]")
    assert RESOLVED.free_parameters() == ()
    with pytest.raises(UnresolvedParameterError, match="levels\\[0\\], levels\\[1\\]"):
        TEMPLATE.payout_values([70.0])


def test_with_values_keeps_the_terms_and_returns_a_new_cover() -> None:
    labelled = Cover(
        RX1DAY, TEMPLATE.payout, payout_unit="EUR", asset_id="a-1", label="mine"
    )
    partial = labelled.with_values({"levels[1]": 2.0})
    assert partial.free_parameters() == ("levels[0]", "levels[2]")
    assert (partial.index, partial.payout_unit, partial.asset_id, partial.label) == (
        RX1DAY,
        "EUR",
        "a-1",
        "mine",
    )
    assert labelled.free_parameters() == ("levels[0]", "levels[1]", "levels[2]")
    assert partial != labelled


@pytest.mark.parametrize(
    ("values", "error"),
    [
        pytest.param({"levels[3]": 1.0}, CoverDefinitionError, id="unknown"),
        pytest.param({"breakpoints[0]": 1.0}, CoverDefinitionError, id="fixed"),
        pytest.param({"levels[0]": -1.0}, CoverDefinitionError, id="outside"),
        pytest.param(
            {"levels[0]": 2.0, "levels[1]": 1.0}, CoverDefinitionError, id="order"
        ),
    ],
)
def test_with_values_validates(
    values: dict[str, float], error: type[Exception]
) -> None:
    with pytest.raises(error):
        TEMPLATE.with_values(values)


# -- invalid covers --------------------------------------------------------------


class _Fraction:
    """A user family: a fraction of the index, with configurable defects."""

    kind = "test_cover_fraction"

    def __init__(
        self,
        fraction: float,
        *,
        index_names: tuple[str, ...] = ("index",),
        kind: str | None = None,
        offset: float = 0.0,
        reshape: bool = False,
    ) -> None:
        self.fraction = fraction
        self._index_names = index_names
        self._document_kind = kind or self.kind
        self.offset = offset
        self.reshape = reshape

    @property
    def tail(self) -> str:
        return "upper"

    @property
    def index_names(self) -> tuple[str, ...]:
        return self._index_names

    def parameters(self) -> dict[str, float | Free]:
        return {"fraction": self.fraction}

    def with_values(self, values: Any) -> _Fraction:
        return self

    def evaluate(self, index: FloatArray) -> FloatArray:
        paid = np.maximum(index, 0.0) * self.fraction + self.offset
        return paid.reshape(-1, 1) if self.reshape else paid

    def to_document(self) -> dict[str, Any]:
        return {"kind": self._document_kind, "fraction": self.fraction}

    @classmethod
    def from_document(cls, document: Any) -> _Fraction:
        return cls(document["fraction"])


def test_a_registered_user_family_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    cover = Cover(RX1DAY, _Fraction(0.5), payout_unit="1")  # type: ignore[arg-type]
    assert cover.payout_values([-1.0, 2.0]).tolist() == [0.0, 1.0]
    with pytest.raises(UnknownPayoutKindError, match="test_cover_fraction"):
        Cover.from_json(cover.to_json())
    monkeypatch.delitem(_REGISTRY, _Fraction.kind, raising=False)
    register_payout(_Fraction)
    restored = Cover.from_json(cover.to_json())
    assert restored == cover
    assert isinstance(restored.payout, _Fraction)
    monkeypatch.delitem(_REGISTRY, _Fraction.kind)


@pytest.mark.parametrize(
    ("payout", "message"),
    [
        pytest.param(_Fraction(0.5, offset=-1.0), "negative", id="negative"),
        pytest.param(_Fraction(0.5, offset=math.inf), "non-finite", id="inf"),
        pytest.param(_Fraction(0.5, reshape=True), "shape", id="shape"),
    ],
)
def test_user_payouts_are_checked_when_they_pay(payout: Any, message: str) -> None:
    cover = Cover(RX1DAY, payout, payout_unit="1")
    with pytest.raises(CoverDefinitionError, match=message):
        cover.payout_values([1.0, 2.0])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        pytest.param({"index": "rx1day"}, "IndexDefinition", id="index"),
        pytest.param({"payout": lambda x: x}, "PayoutFunction", id="callable"),
        pytest.param({"index": SPEI}, "tail", id="tail-mismatch"),
        pytest.param(
            {"payout": _Fraction(0.5, index_names=("rain", "wind"))},
            "one index component",
            id="two-components",
        ),
        pytest.param(
            {"payout": _Fraction(0.5, kind="other")}, "kind", id="document-kind"
        ),
        pytest.param({"payout": _Fraction(math.nan)}, "canonical", id="nan-document"),
        pytest.param({"payout_unit": ""}, "payout_unit", id="empty-unit"),
        pytest.param({"payout_unit": " EUR"}, "payout_unit", id="unit-whitespace"),
        pytest.param({"payout_unit": None}, "payout_unit", id="unit-none"),
        pytest.param({"asset_id": ""}, "asset_id", id="empty-asset"),
        pytest.param({"asset_id": 7}, "asset_id", id="asset-not-text"),
        pytest.param({"label": ""}, "label", id="empty-label"),
        pytest.param({"label": 3}, "label", id="label-not-text"),
    ],
)
def test_invalid_covers_raise(arguments: dict[str, Any], message: str) -> None:
    values: dict[str, Any] = {
        "index": RX1DAY,
        "payout": RESOLVED.payout,
        "payout_unit": "EUR",
    }
    values.update(arguments)
    with pytest.raises(CoverDefinitionError, match=message):
        UNTYPED_COVER(values.pop("index"), values.pop("payout"), **values)


def _document(**changes: Any) -> str:
    document = json.loads(TEMPLATE_JSON)
    document.update(changes)
    return json.dumps(document)


@pytest.mark.parametrize(
    ("text", "error", "message"),
    [
        pytest.param("{", CoverDefinitionError, "not valid", id="syntax"),
        pytest.param("[1.0]", CoverDefinitionError, "object", id="list"),
        pytest.param(
            '{"asset_id":null,"asset_id":null}',
            CoverDefinitionError,
            "repeats",
            id="dup",
        ),
        pytest.param(
            TEMPLATE_JSON.replace("60.0", "NaN"), CoverDefinitionError, "NaN", id="nan"
        ),
        pytest.param(
            TEMPLATE_JSON.replace("60.0", "-Infinity"),
            CoverDefinitionError,
            "Infinity",
            id="infinity",
        ),
        pytest.param(
            _document(label="x"), CoverDefinitionError, "unknown: label", id="key"
        ),
        pytest.param(
            json.dumps(
                {k: v for k, v in json.loads(TEMPLATE_JSON).items() if k != "index"}
            ),
            CoverDefinitionError,
            "missing: index",
            id="missing",
        ),
        pytest.param(
            _document(schema_version="2.0"), CoverDefinitionError, "schema", id="schema"
        ),
        pytest.param(
            _document(semantics_version="crc-parametric-cover/2"),
            CoverDefinitionError,
            "semantics",
            id="semantics",
        ),
        pytest.param(
            _document(payout=[1.0]), CoverDefinitionError, "payout", id="payout"
        ),
        pytest.param(
            _document(payout={"kind": "layered"}),
            UnknownPayoutKindError,
            "layered",
            id="unknown-kind",
        ),
        pytest.param(
            _document(payout={"kind": "step", "breakpoints": [1.0], "levels": [-1.0]}),
            CoverDefinitionError,
            ">= 0",
            id="invalid-payout",
        ),
        pytest.param(
            _document(index={"hazard_name": "rx1day"}),
            CoverDefinitionError,
            "IndexDefinition",
            id="invalid-index",
        ),
        pytest.param(
            _document(payout_unit=""), CoverDefinitionError, "payout_unit", id="unit"
        ),
    ],
)
def test_invalid_cover_json_raises(
    text: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        Cover.from_json(text)


def test_from_json_needs_text_and_from_document_an_object() -> None:
    with pytest.raises(CoverDefinitionError, match="str"):
        Cover.from_json(TEMPLATE_JSON.encode())  # type: ignore[arg-type]
    with pytest.raises(CoverDefinitionError, match="object"):
        Cover.from_document([])  # type: ignore[arg-type]


def test_to_document_returns_a_copy() -> None:
    document = TEMPLATE.to_document()
    document["payout"]["levels"][0]["free"]["lower"] = 5.0
    assert TEMPLATE.to_document()["payout"]["levels"][0]["free"]["lower"] == 0.0
    assert TEMPLATE.to_json() == TEMPLATE_JSON
