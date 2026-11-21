"""Annual loss persistence and pairing on known offline books."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import crc_framework as cf
import numpy
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from crc_sdk._dataset_schema import (
    ExecutionCounts,
    ExecutionReport,
    ResourceIdentity,
    SourceIdentity,
    _dump,
)
from crc_sdk._declarations import ScientificValue
from crc_sdk._errors import ValidationError
from crc_sdk._loss_artifact import (
    _DOCUMENT,
    _DOCUMENT_DIGEST,
    LossArtifact,
    LossResult,
    _sha256,
    make_loss_result,
    write_loss_artifact,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _sample(
    *,
    end: float = 0.25,
    stream: int = 0,
    include_components: bool = False,
    currency: str = "EUR",
    trials: int = 15,
    vulnerability: str = "c",
    retrieved_day: int = 1,
) -> LossResult:
    curve = cf.QuantileCurve([0, 0.5, 1], [0, 0, end])
    interests = (
        cf.InsuredInterest(
            "dock",
            longitude=10,
            latitude=40,
            total_insured_value=100,
            damage_curve=curve,
        ),
        cf.InsuredInterest(
            "store",
            longitude=11,
            latitude=41,
            total_insured_value=200,
            damage_curve=curve,
        ),
    )
    sample = cf.simulate_annual_losses(
        interests,
        spec=cf.AnnualLossSpec(trials=trials, damage_cv=0),
        secondary_stream=stream,
        include_components=include_components,
    )
    source = SourceIdentity(
        "local",
        "flood",
        resources=(
            ResourceIdentity(
                "file:///hazard.parquet",
                _digest("d"),
                "CC0-1.0",
                "Provider",
                datetime(2026, 1, retrieved_day, tzinfo=UTC),
            ),
        ),
    )
    evidence: dict[str, ScientificValue] = {
        "damage_response": "damage",
        "response_model_sha256": _digest("b"),
        "vulnerability_sha256": _digest(vulnerability),
        "member_id": None,
        "variable_id": "flood-depth",
        "scenario_id": None,
        "period_id": None,
        "series": {
            "variable": {"id": "flood-depth"},
            "scenario": None,
            "period": None,
            "probability": {"kind": "annual", "tail": "upper"},
        },
        "probability_grid": [0, 0.5, 1],
        "tail_completion": None,
        "matched_support_count": 1,
        "matched_supports_sha256": _digest("e"),
        "resolved_impacts": [
            {
                "resolution_id": _digest("b"),
                "impact_sha256": _digest("b"),
                "asset_count": 2,
            }
        ],
        "input_resources": [
            {
                "member_id": None,
                "resource_id": None,
                "source": None,
                "transformations": [],
            }
        ],
    }
    return make_loss_result(
        sample,
        peril="flood",
        currency=currency,
        valuation_basis="replacement cost",
        portfolio_sha256=_digest("a"),
        input_sha256=_digest("1" if end == 0.25 else "2"),
        evidence=evidence,
        report=ExecutionReport(
            True,
            True,
            counts=ExecutionCounts(requested=6, produced=6, matched_assets=2),
            source_identities=(source,),
        ),
    )


def _assert_schema(document: dict[str, Any]) -> None:
    schema_resources: dict[str, Resource[Any]] = {}
    for package in ("crc_sdk", "crc_framework"):
        root = files(package).joinpath("contracts")
        schemas = list(root.iterdir())
        for entry in schemas:
            if entry.is_dir():
                for schema in entry.iterdir():
                    if schema.name.endswith(".schema.json"):
                        value = json.loads(schema.read_text(encoding="utf-8"))
                        schema_resources[value["$id"]] = Resource.from_contents(value)
    resource = schema_resources[
        "https://climateriskcommons.org/schemas/annual-loss/1.0"
    ]
    Draft202012Validator.check_schema(resource.contents)
    validator = Draft202012Validator(
        resource.contents,
        registry=Registry().with_resources(schema_resources.items()),
        format_checker=FormatChecker(),
    )
    validator.validate(document)


@pytest.mark.parametrize("components", (False, True))
def test_loss_parquet_roundtrip_and_json_schema(
    tmp_path: Path, *, components: bool
) -> None:
    expected = _sample(include_components=components, trials=4100)
    path = tmp_path / "loss.parquet"
    saved = write_loss_artifact(path, expected)
    assert saved.validate().valid is True
    reopened = LossArtifact.open(path)
    assert reopened.report == expected.report
    actual = reopened.read()
    assert actual.metadata == expected.metadata
    assert actual.metadata.series.variable.id == "flood-depth"
    assert actual.metadata.damage_response == "damage"
    assert actual.summary(levels=[0.95]) == expected.summary(levels=[0.95])
    assert actual.trial_ids == range(4100)
    numpy.testing.assert_array_equal(actual.totals, expected.totals)
    with pytest.raises(ValueError, match="WRITEABLE"):
        actual.totals.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        actual.metadata.run_id = "forged"  # type: ignore[misc]
    if components:
        assert expected.components is not None and actual.components is not None
        assert tuple(actual.components) == ("dock", "store")
        for identifier in actual.components:
            numpy.testing.assert_array_equal(
                actual.components[identifier], expected.components[identifier]
            )
            assert not actual.components[identifier].flags.writeable
    else:
        assert actual.components is None
    rows = pq.read_table(path).to_pylist()
    _assert_schema({"metadata": json.loads(reopened.metadata.to_json()), "rows": rows})


def test_run_identity_and_paired_reopened_comparison(tmp_path: Path) -> None:
    baseline = _sample()
    augmented = _sample(include_components=True, retrieved_day=2)
    assert baseline.metadata.run_id == augmented.metadata.run_id
    assert baseline.metadata.totals_sha256 == augmented.metadata.totals_sha256
    assert augmented.metadata.components_sha256 is not None
    assert baseline.report != augmented.report
    future = _sample(end=0.5, stream=4)
    assert future.metadata.run_id != baseline.metadata.run_id
    path = tmp_path / "future.parquet"
    write_loss_artifact(path, future)
    loading = LossArtifact.open(path).read().compare(baseline)
    assert loading.baseline_run_id == baseline.metadata.run_id
    assert loading.future_run_id == future.metadata.run_id
    assert loading.trial_ids == baseline.trial_ids
    numpy.testing.assert_array_equal(
        loading.trial_differences, future.totals - baseline.totals
    )
    assert loading.average_annual_loss_change == pytest.approx(
        baseline.summary().average_annual_loss
    )
    with pytest.raises(ValueError, match="secondary_stream must differ"):
        future.compare(future)
    with pytest.raises(ValueError, match="currency"):
        future.compare(_sample(currency="USD"))
    with pytest.raises(ValueError, match="vulnerability_sha256"):
        future.compare(_sample(vulnerability="f"))
    with pytest.raises(ValueError, match="spec"):
        future.compare(_sample(trials=12))


def test_loss_write_preserves_destination_until_verified(tmp_path: Path) -> None:
    original = _sample()
    path = tmp_path / "loss.parquet"
    write_loss_artifact(path, original)
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_loss_artifact(path, _sample(end=0.5))

    def fail(completed: int, total: int) -> None:
        assert completed == total
        raise RuntimeError("callback stopped the write")

    with pytest.raises(RuntimeError, match="stopped"):
        write_loss_artifact(path, _sample(end=0.5), overwrite=True, progress=fail)
    assert path.read_bytes() == original_bytes
    assert tuple(tmp_path.iterdir()) == (path,)


def test_loss_artifact_detects_schema_footer_and_payload_corruption(
    tmp_path: Path,
) -> None:
    result = _sample()
    path = tmp_path / "loss.parquet"
    write_loss_artifact(path, result)
    with pq.ParquetFile(path) as artifact:
        footer = artifact.metadata.metadata
        assert footer is not None
        footer = {key: value for key, value in footer.items() if key != b"ARROW:schema"}
        table = artifact.read().replace_schema_metadata(footer)
    modified = table.set_column(0, table.schema.field(0), pa.array(range(1, 16)))
    pq.write_table(modified, path)
    opened = LossArtifact.open(path)
    with pytest.raises(ValidationError, match="trial IDs"):
        opened.read()
    assert opened.validate().valid is False
    modified = table.set_column(1, table.schema.field(1), pa.array([0.0] * 15))
    pq.write_table(modified, path)
    with pytest.raises(ValidationError, match="totals checksum"):
        opened.read()
    oversized = table.set_column(
        1,
        table.schema.field(1),
        pa.array([result.metadata.total_insured_value + 1] * 15),
    )
    pq.write_table(oversized, path)
    with pytest.raises(ValidationError, match="exceed total insured value"):
        opened.read()
    forged = json.loads(footer[_DOCUMENT])
    forged["peril"] = "other-peril"
    document = _dump(forged).encode()
    pq.write_table(table.replace_schema_metadata({**footer, _DOCUMENT: document}), path)
    with pytest.raises(ValidationError, match="metadata checksum"):
        opened.read()
    pq.write_table(
        table.replace_schema_metadata(
            {
                **footer,
                _DOCUMENT: document,
                _DOCUMENT_DIGEST: _sha256(document).encode(),
            }
        ),
        path,
    )
    with pytest.raises(ValidationError, match="run identity"):
        opened.read()

    forged = json.loads(footer[_DOCUMENT])
    forged["report"]["counts"]["produced"] -= 1
    document = _dump(forged).encode()
    pq.write_table(
        table.replace_schema_metadata(
            {
                **footer,
                _DOCUMENT: document,
                _DOCUMENT_DIGEST: _sha256(document).encode(),
            }
        ),
        path,
    )
    with pytest.raises(ValidationError, match="complete insured book"):
        opened.read()
