"""Current CRC Open indexes, exact science, and verified dataset resources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import DepthAboveGround, LengthUnit, Tail

from crc_sdk import (
    Cache,
    CRCOpen,
    Dataset,
    Period,
    Probability,
    Runtime,
    ValidationError,
    Variable,
)


def release(tmp_path: Path) -> tuple[CRCOpen, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_quantiles(
        {
            "support_id": ["harbour"] * 2,
            "longitude": [24.0] * 2,
            "latitude": [60.0] * 2,
            "probability": [0.9, 0.99],
            "value": [0.3, 0.85],
        },
        variable=Variable(
            "company:depth", measurement=DepthAboveGround(unit=LengthUnit.METRE)
        ),
        probability=Probability.annual(tail=Tail.UPPER),
        period=Period.window(2000, 2020),
    ).write(tmp_path / "depth.parquet")
    assert dataset.path is not None and dataset.metadata is not None
    index = tmp_path / "_INDEX.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "id": "fixture-release",
                "datasets": [
                    {
                        "id": "depth",
                        "uri": dataset.path.name,
                        "sha256": "sha256:"
                        + hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
                        "metadata": json.loads(dataset.metadata.to_json()),
                    }
                ],
            }
        )
    )
    return CRCOpen(catalog=index), index, dataset.path


def test_index_discovery_does_not_read_bulk_and_roundtrips_exact_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe, index, artifact = release(tmp_path)
    original = artifact.read_bytes()
    artifact.unlink()
    inspection = recipe.inspect()
    reference = inspection.series[0]
    assert reference.release == inspection.releases[0]
    assert reference.probability == Probability.annual(tail=Tail.UPPER)
    assert reference.variable.id == "company:depth"
    with pytest.raises(FileNotFoundError):
        recipe.validate()
    artifact.write_bytes(original)
    report = recipe.validate()
    assert report.valid and report.counts.produced == 1
    assert report.source_identities[0].release is not None
    assert report.source_identities[0].release.sha256 == reference.release.sha256
    selected = recipe.select(series=reference)
    result = selected.read(runtime=Runtime(cache=Cache.reuse(tmp_path / "cache")))
    assert result.report.valid
    assert result.table.num_rows == 1
    assert result.metadata.source is not None
    assert len(result.metadata.source.resources) == 2
    assert all(
        r.license and r.attribution and r.retrieved_at
        for r in result.metadata.source.resources
    )
    index.unlink()
    artifact.unlink()
    replay = selected.read(runtime=Runtime(cache=Cache.offline(tmp_path / "cache")))
    assert replay.table.equals(result.table, check_metadata=False)


def test_current_pointer_checks_both_hash_and_identity(tmp_path: Path) -> None:
    _, index, _ = release(tmp_path)
    pointer = tmp_path / "_CURRENT.json"
    document = {
        "schema_version": "1.0",
        "release": {
            "id": "fixture-release",
            "uri": index.name,
            "sha256": "sha256:" + hashlib.sha256(index.read_bytes()).hexdigest(),
        },
    }
    pointer.write_text(json.dumps(document))
    assert CRCOpen(catalog=pointer).read().report.valid
    document["release"]["id"] = "wrong"  # type: ignore[index]
    pointer.write_text(json.dumps(document))
    with pytest.raises(ValidationError, match="pointer identity"):
        CRCOpen(catalog=pointer).inspect()


def test_exact_typed_selectors_reject_conflicting_science(tmp_path: Path) -> None:
    recipe, _, _ = release(tmp_path)
    reference = recipe.inspect().series[0]
    conflicting = Variable(
        reference.variable.id, measurement=DepthAboveGround(unit=LengthUnit.CENTIMETRE)
    )
    with pytest.raises(ValueError, match="scientific|science|definition"):
        recipe.select(variables=conflicting).read()
    assert reference.period is not None
    assert recipe.select(periods=reference.period).read().table.num_rows == 1
    with pytest.raises(ValueError, match="unknown"):
        recipe.select(variables="unknown").read()


def test_dataset_checksum_and_header_must_match_index(tmp_path: Path) -> None:
    recipe, index, artifact = release(tmp_path)
    artifact.write_bytes(b"corrupt")
    with pytest.raises(OSError, match="content identity"):
        recipe.read()
    # Even checksum-correct Parquet is rejected if indexed science differs.
    recipe, index, artifact = release(tmp_path / "second")
    document = json.loads(index.read_text())
    document["datasets"][0]["metadata"]["spec"]["variables"][0]["measurement"][
        "unit"
    ] = "cm"
    index.write_text(json.dumps(document))
    with pytest.raises(ValidationError, match="header differs"):
        recipe.read()
    assert pq.ParquetFile(artifact).metadata.num_rows == 1


def test_historical_indexes_are_not_admitted_and_empty_does_not_acquire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "_CATALOG.json"
    index.write_text(json.dumps({"schema_version": "1.3", "pathways": []}))
    with pytest.raises(ValidationError):
        CRCOpen(catalog=index).read()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("empty selection acquired a source")

    monkeypatch.setattr("crc_sdk._source_io._download", forbidden)
    recipe = CRCOpen().select(variables=[])
    assert recipe.read().report.complete
    assert recipe.inspect().series == ()
    assert recipe.explain().operations
