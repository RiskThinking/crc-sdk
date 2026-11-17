"""Verify installed paired wheels and their packaged release declarations."""

from __future__ import annotations

import importlib.metadata
import json
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

import crc_framework
from crc_framework import _core

import crc_sdk


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    for module in (crc_framework, crc_sdk):
        if module.__file__ is None or not Path(
            module.__file__
        ).resolve().is_relative_to(prefix):
            raise SystemExit(f"wheel verification imported a checkout: {module}")

    ledgers = []
    for package in ("crc_framework", "crc_sdk"):
        root = files(package).joinpath("contracts", "api")
        ledger = json.loads(root.joinpath("release.json").read_text(encoding="utf-8"))
        contract = json.loads(
            root.joinpath("public-api.json").read_text(encoding="utf-8")
        )
        if not contract:
            raise SystemExit(f"empty Python contract in {package}")
        ledgers.append(ledger)
    if ledgers[0] != ledgers[1]:
        raise SystemExit("framework and SDK release ledgers disagree")

    versions = ledgers[0]["packages"]
    for distribution, key in (("crc-framework", "framework"), ("crc-sdk", "sdk")):
        actual = importlib.metadata.version(distribution)
        if actual != versions[key]:
            raise SystemExit(
                f"{distribution}: expected {versions[key]}, found {actual}"
            )
    if _core.__version__ != versions["framework"]:
        raise SystemExit("native framework version differs from its distribution")

    requirement = next(
        value
        for value in importlib.metadata.requires("crc-sdk") or ()
        if value.startswith("crc-framework")
    )
    specifiers = requirement.replace(" ", "").removeprefix("crc-framework").strip("()")
    expected = versions["framework_requirement"].removeprefix("crc-framework")
    if set(specifiers.split(",")) != set(expected.split(",")):
        raise SystemExit(
            f"SDK wheel has an incorrect framework requirement: {requirement}"
        )
    for source, target in (
        ("schemas/dataset-2.0.schema.json", "dataset/dataset-2.0.schema.json"),
        ("fixtures/dataset-2.0.json", "dataset/fixtures.json"),
        ("taxonomy/taxonomy-1.0.schema.json", "taxonomy/taxonomy-1.0.schema.json"),
        ("taxonomy/fixtures.json", "taxonomy/fixtures.json"),
    ):
        if (
            files("crc_sdk").joinpath("contracts", source).read_bytes()
            != files("crc_framework").joinpath("contracts", target).read_bytes()
        ):
            raise SystemExit(f"installed data contracts disagree: {source}")

    variable = crc_sdk.Variable("user:installed-depth")
    dataset = crc_sdk.Dataset.from_quantiles(
        {
            "support_id": ["a", "a"],
            "longitude": [0, 0],
            "latitude": [0, 0],
            "probability": [0.1, 0.9],
            "value": [1, 2],
        },
        variable=variable,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dataset.parquet"
        with dataset.execute() as execution:
            assert execution.validate().valid
            assert execution.read().table.num_rows == 1
            saved = execution.write(path)
        assert saved.read().report.complete
        assert crc_sdk.Dataset.open(path).read().report.valid
    print(
        f"Installed framework {versions['framework']} and SDK {versions['sdk']}: "
        "versions, shared contracts, execution, and persistence verified"
    )


if __name__ == "__main__":
    main()
