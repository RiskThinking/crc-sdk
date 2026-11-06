"""The parity fixtures' README is true of the cases it describes.

`tests/fixtures/parametric/README.md` gives the cases' licence by the suffix
of their ids, with a count for each group of suffixes, and the number of
cases copied, with a count for each kind. The cases carry no licence field,
so the README is their licence marker; these tests keep it true of
`tests/fixtures/parametric/cases/`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "parametric"
README = FIXTURES / "README.md"
CASES = FIXTURES / "cases"
LICENCE_HINT = "update the Licence section of tests/fixtures/parametric/README.md"
ORIGIN_HINT = "update the case list under Origin in tests/fixtures/parametric/README.md"


class _Group(NamedTuple):
    """One group of id suffixes in the Licence section, and its count."""

    spdx: str
    data_sets: tuple[str, ...]
    suffixes: tuple[str, ...]
    stated: int


@pytest.fixture(scope="module")
def cases() -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for path in sorted(CASES.glob("*.json")):
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        assert document["case_id"] == path.stem, path
        documents[path.stem] = document
    assert documents, "tests/fixtures/parametric/cases/ holds no case"
    return documents


def _section(title: str) -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(
        rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    assert match is not None, f"the README has no {title} section"
    return match.group(1)


def _bullets(section: str) -> list[str]:
    """The section's top-level bullets, each on one line."""
    bullets: list[str] = []
    for line in section.splitlines():
        if line.startswith("- "):
            bullets.append(line[2:])
        elif bullets and line.startswith("  "):
            bullets[-1] += " " + line.strip()
        elif line.strip():
            bullets.append("")  # a paragraph ends the list's last bullet
    return [bullet for bullet in bullets if bullet]


def _groups() -> list[_Group]:
    """The Licence section's groups of id suffixes, with their counts.

    Each bullet opens with its data sets and licence, "**`a` and `b`:
    SPDX**", and lists their cases in a sentence: "These are the cases whose
    ids end in `-a` or `-a-holdout` (18 cases), in `-b` (13) ...". Each
    count closes the group of the code spans before it; the parenthesis may
    go on to name cases, which are not suffixes. `<i>` stands for a fold
    number.
    """
    groups: list[_Group] = []
    for bullet in _bullets(_section("Licence")):
        head = re.match(r"\*\*(.+?): ([A-Za-z0-9.+-]+)\*\*", bullet)
        if head is None:
            continue
        data_sets = tuple(re.findall(r"`([^`]+)`", head.group(1)))
        sentence = re.search(
            r"These are the cases whose ids end in (.+?\(\d+[^)]*\))\.", bullet
        )
        assert sentence is not None, (bullet[:60], LICENCE_HINT)
        pending: list[str] = []
        for token in re.finditer(r"\((\d+)[^)]*\)|`([^`]+)`", sentence.group(1)):
            if token.group(2) is not None:
                pending.append(token.group(2))
                continue
            assert pending, (token.group(0), LICENCE_HINT)
            groups.append(
                _Group(head.group(2), data_sets, tuple(pending), int(token.group(1)))
            )
            pending = []
        assert not pending, (pending, LICENCE_HINT)
    return groups


def _pattern(suffix: str) -> re.Pattern[str]:
    return re.compile("[0-9]+".join(map(re.escape, suffix.split("<i>"))) + r"\Z")


def _data_set(suffix: str) -> str:
    return suffix.removeprefix("-").split("-", 1)[0]


def test_the_licence_section_counts_every_case(
    cases: dict[str, dict[str, Any]],
) -> None:
    """Every case is in exactly one group, and each group's count is true.

    A case's licence is that of its data set, the part of its id after the
    scope, so the group a case's id ends in must be its data set's, and that
    data set must be one the group's licence bullet names. A change that adds
    cases updates the counts and the total; one that adds a data set names
    it, with its licence, before its cases are committed.
    """
    groups = _groups()
    assert groups, ("the Licence section lists no cases", LICENCE_HINT)
    spdx = {group.spdx for group in groups}
    assert spdx == {"CC0-1.0", "CC-BY-4.0"}, (spdx, LICENCE_HINT)
    for group in groups:
        for suffix in group.suffixes:
            assert suffix.startswith("-"), (suffix, LICENCE_HINT)
            assert _data_set(suffix) in group.data_sets, (group, LICENCE_HINT)
        assert len({_data_set(suffix) for suffix in group.suffixes}) == 1, group
    found: list[list[str]] = [[] for _ in groups]
    used: set[str] = set()
    for case_id in cases:
        hits = [
            (number, suffix)
            for number, group in enumerate(groups)
            for suffix in group.suffixes
            if _pattern(suffix).search(case_id)
        ]
        assert len({number for number, _ in hits}) == 1, (case_id, hits, LICENCE_HINT)
        number = hits[0][0]
        scope = re.search(r"-(?:individual|joint)-([^-]+)", case_id)
        assert scope is not None, case_id
        assert _data_set(groups[number].suffixes[0]) == scope.group(1), (
            case_id,
            LICENCE_HINT,
        )
        found[number].append(case_id)
        used.update(suffix for _, suffix in hits)
    for group, matched in zip(groups, found, strict=True):
        assert group.stated == len(matched), (
            group.suffixes,
            group.stated,
            len(matched),
            LICENCE_HINT,
        )
    unused = {suffix for group in groups for suffix in group.suffixes} - used
    assert not unused, (unused, LICENCE_HINT)
    assert sum(group.stated for group in groups) == len(cases), LICENCE_HINT


def test_the_origin_section_counts_the_cases(cases: dict[str, dict[str, Any]]) -> None:
    """The number of cases copied, and the counts of its list, are true.

    "The N cases copied" is the number of files in `cases/`; the counts that
    open the list's bullets add up to it; and the evaluation, holdout,
    k-fold and fresh-draw counts are those of the cases' `kind` and `split`.
    """
    section = _section("Origin")
    total = re.search(r"The (\d+) cases copied", section)
    assert total is not None, ORIGIN_HINT
    assert int(total.group(1)) == len(cases), (int(total.group(1)), ORIGIN_HINT)
    stated: dict[str, int] = {}
    counts: list[int] = []
    for bullet in _bullets(section):
        opening = re.match(r"(\d+) (.+?) cases\b", bullet)
        if opening is None:
            continue
        counts.append(int(opening.group(1)))
        stated[opening.group(2)] = int(opening.group(1))
    assert sum(counts) == len(cases), (counts, ORIGIN_HINT)

    def splits(kind: str) -> int:
        return sum(
            1 for case in cases.values() if (case["split"] or {}).get("kind") == kind
        )

    evaluations = sum(1 for case in cases.values() if case["kind"] == "evaluate")
    expected = {
        "evaluation": evaluations,
        "out-of-sample": splits("holdout"),
        "k-fold": splits("kfold"),
        "fresh-draw": splits("fresh_draws"),
    }
    for kind, count in expected.items():
        assert stated.get(kind) == count, (kind, stated.get(kind), count, ORIGIN_HINT)
    in_sample_designs = sum(
        1
        for case in cases.values()
        if case["kind"] == "design" and case["split"] is None
    )
    others = sum(stated.values()) - sum(expected.values())
    assert others == in_sample_designs, (others, in_sample_designs, ORIGIN_HINT)
