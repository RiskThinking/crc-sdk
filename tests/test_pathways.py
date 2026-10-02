from __future__ import annotations

import warnings

import pytest
from crc_framework import PATHWAYS

from crc_sdk import pathways
from crc_sdk.pathways import (
    UnknownPathwayWarning,
    pathways_equivalent,
    register_pathway,
    resolve_pathway,
)


def test_every_framework_pathway_resolves_to_its_id() -> None:
    for pathway in PATHWAYS:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = resolve_pathway(pathway.name)
        assert result.known and result.framework_id == pathway.id
        assert result.canonical_id == pathway.name


def test_historic_and_historical_share_one_canonical_id_but_keep_labels() -> None:
    a, b = resolve_pathway("historic"), resolve_pathway("historical")
    assert a.canonical_id == b.canonical_id == "historic"
    assert a.framework_id == b.framework_id is not None
    assert (a.label, b.label) == ("historic", "historical")
    assert pathways_equivalent("historic", "Historical")


def test_matching_is_case_and_whitespace_tolerant_label_preserved() -> None:
    result = resolve_pathway("  Hot  house ")
    assert result.canonical_id == "Hot House" and result.label == "  Hot  house "
    assert resolve_pathway("< 2 Degrees").canonical_id == "<2 degrees"
    assert resolve_pathway("SSP585").canonical_id == "ssp585"


def test_known_without_framework_id() -> None:
    for label in ("RT3", "ssp534", "ssp534-over"):
        result = resolve_pathway(label)
        assert result.known and result.framework_id is None
    assert pathways_equivalent("ssp534", "SSP534-over")


def test_unknown_warns_or_raises_when_strict() -> None:
    with pytest.warns(UnknownPathwayWarning, match="mystery"):
        result = resolve_pathway(" mystery ")
    assert not result.known and result.framework_id is None
    assert result.label == " mystery " and result.canonical_id == "mystery"
    with pytest.raises(ValueError, match="mystery"):
        resolve_pathway("mystery", strict=True)


def test_equivalence_never_warns_and_distinguishes() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not pathways_equivalent("ssp126", "ssp245")
        assert pathways_equivalent("zzz", "ZZ Z")
        assert not pathways_equivalent("zzz", "ssp126")


def test_register_private_pathway_and_conflicts() -> None:
    try:
        register_pathway("Private-X", framework_id=99, aliases=("px",))
        result = resolve_pathway("PX", strict=True)
        assert result.canonical_id == "Private-X" and result.framework_id == 99
        register_pathway("Private-X", aliases=("px2",))  # extra alias only
        assert resolve_pathway("px2").framework_id == 99
        with pytest.raises(ValueError, match="already registered"):
            register_pathway("other", aliases=("historical",))
        assert "other" not in {v[0] for v in pathways._REGISTRY.values()}
        with pytest.raises(ValueError, match="non-empty"):
            register_pathway(" ")
    finally:
        for key in [k for k, v in pathways._REGISTRY.items() if v[0] == "Private-X"]:
            del pathways._REGISTRY[key]
