"""Current CRC Open indexes, exact science, and verified dataset resources."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    DepthAboveGround,
    DiagnosticSeverity,
    DistributionFamily,
    LengthUnit,
    SampleFit,
    Tail,
)

from crc_sdk import (
    Assets,
    Bounds,
    Cache,
    CRCOpen,
    Dataset,
    ExecutionDiagnostic,
    IdentityRename,
    Period,
    Probability,
    ResourceIdentity,
    Runtime,
    Samples,
    SourceIdentity,
    SpatialMatch,
    Stage,
    ValidationError,
    Variable,
)
from crc_sdk._crc_open import release_extent


def _extent(path: Path) -> dict[str, float] | None:
    extent = release_extent(path)
    return None if extent is None else dataclasses.asdict(extent)


def _release_index(tmp_path: Path, *datasets: Dataset) -> Path:
    resources = []
    for dataset in datasets:
        assert dataset.path is not None and dataset.metadata is not None
        resources.append(
            {
                "id": dataset.path.stem,
                "uri": dataset.path.name,
                "sha256": "sha256:"
                + hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
                "metadata": json.loads(dataset.metadata.to_json()),
                "extent": _extent(dataset.path),
            }
        )
    index = tmp_path / "_INDEX.json"
    index.write_text(
        json.dumps(
            {"schema_version": "1.1", "id": "fixture-release", "datasets": resources}
        )
    )
    return index


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
    index = _release_index(tmp_path, dataset)
    assert dataset.path is not None
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
    result = CRCOpen(catalog=pointer).read()
    assert result.report.valid
    assert result.metadata.source is not None
    assert {item.uri for item in result.metadata.source.resources} == {
        pointer.as_uri(),
        index.as_uri(),
        (tmp_path / "depth.parquet").as_uri(),
    }
    document["release"]["id"] = "wrong"  # type: ignore[index]
    pointer.write_text(json.dumps(document))
    with pytest.raises(ValidationError, match="pointer identity"):
        CRCOpen(catalog=pointer).inspect()


def test_release_terminals_keep_verified_dataset_reports(tmp_path: Path) -> None:
    upstream = SourceIdentity(
        "fixture",
        "observations",
        resources=(
            ResourceIdentity(
                "https://example.org/observations",
                "sha256:" + "a" * 64,
                license="CC-BY-4.0",
                attribution="Fixture author",
                retrieved_at=datetime(2025, 1, 1, tzinfo=UTC),
            ),
        ),
    )
    warning = ExecutionDiagnostic(
        "incomplete_years",
        DiagnosticSeverity.WARNING,
        Stage.REDUCTION,
        "Three observation years failed coverage",
    )
    datasets = []
    for name, longitude in (("harbour", 24.0), ("airport", 22.0)):
        payload = Dataset.from_quantiles(
            {
                "support_id": [name] * 2,
                "longitude": [longitude] * 2,
                "latitude": [60.0] * 2,
                "probability": [0.9, 0.99],
                "value": [0.3, 0.85],
            },
            variable=Variable("company:depth"),
        ).read()
        original_report = replace(
            payload.report,
            counts=replace(
                payload.report.counts,
                requested=2,
                skipped=1,
                usable_years=8,
                incomplete_years=3,
            ),
            diagnostics=(warning,),
        )
        metadata = replace(payload.metadata, report=original_report, source=upstream)
        path = tmp_path / f"{name}.parquet"
        pq.write_table(
            payload.table.replace_schema_metadata(
                {b"crc.dataset": metadata.to_json().encode()}
            ),
            path,
        )
        datasets.append(Dataset.open(path))

    recipe = CRCOpen(catalog=_release_index(tmp_path, *datasets))
    report = recipe.validate()
    assert report.valid
    assert report.counts.requested == 4
    assert report.counts.produced == 2
    assert report.counts.skipped == 2
    assert report.counts.usable_years == 16
    assert report.counts.incomplete_years == 6
    assert report.diagnostics == (warning,)
    assert report.source_identities[1:] == (upstream,)

    selected = recipe.select(area=Bounds(west=23.0, south=59.0, east=25.0, north=61.0))
    with selected.execute() as execution:
        preview = execution.read(limit=1)
        saved = execution.write(tmp_path / "selected.parquet")
    # The airport file's indexed extent misses the area, so it does not
    # participate and its report is not combined.
    assert preview.report.counts.produced == 1
    assert preview.report.counts.skipped == 1
    assert preview.report.diagnostics == (warning,)
    assert preview.report.source_identities[1:] == (upstream,)
    assert saved.report == preview.report
    assert saved.path is not None
    reopened = Dataset.open(saved.path).read().report
    assert reopened.counts == preview.report.counts
    assert reopened.diagnostics == preview.report.diagnostics
    assert reopened.source_identities[:2] == preview.report.source_identities


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


def _converted_fit(variable_id: str, support: str, *, unit: LengthUnit) -> Dataset:
    variable = Variable(
        variable_id, measurement=DepthAboveGround(unit=LengthUnit.METRE)
    )
    return (
        Samples.from_table(
            {
                "support_id": [support] * 5,
                "longitude": [24.0] * 5,
                "latitude": [60.0] * 5,
                "time": [2000, 2001, 2002, 2003, 2004],
                "value": [1.0, 2.0, 4.0, 5.0, 8.0],
            },
            variable=variable,
        )
        .fit(SampleFit(family=DistributionFamily.GUMBEL_RIGHT))
        .convert_units(variable=variable, unit=unit)
    )


def test_release_replays_fitted_conversions_across_distinct_series_and_shards(
    tmp_path: Path,
) -> None:
    first = (
        _converted_fit("company:harbour", "first", unit=LengthUnit.CENTIMETRE)
        .rename(variables={"company:harbour": "company:coast"})
        .write(tmp_path / "coast.parquet")
    )
    shard = (
        _converted_fit("company:harbour", "second", unit=LengthUnit.CENTIMETRE)
        .rename(variables={"company:harbour": "company:coast"})
        .write(tmp_path / "coast-shard.parquet")
    )
    other = _converted_fit("company:river", "third", unit=LengthUnit.MILLIMETRE).write(
        tmp_path / "river.parquet"
    )
    assert first.metadata is not None and other.metadata is not None
    source = CRCOpen(catalog=_release_index(tmp_path, first, shard, other))
    reference = source.inspect().series[0]
    assert reference.variable.id == "company:coast"
    published = source.read()
    assert published.report.valid is True
    assert published.metadata.provenance == (
        *first.metadata.provenance,
        *other.metadata.provenance,
    )
    assert published.table["support_id"].to_pylist() == ["first", "second", "third"]
    written = source.write(tmp_path / "replayed.parquet").read()
    assert written.table.equals(published.table, check_metadata=False)
    assert written.metadata.provenance == published.metadata.provenance

    assert other.path is not None
    other.path.unlink()
    selected = source.select(series=reference)
    result = selected.read()
    assert result.table["support_id"].to_pylist() == ["first", "second"]
    assert result.metadata.provenance == first.metadata.provenance

    assets = Assets({"asset_id": ["site"], "longitude": [24.0], "latitude": [60.0]})
    evaluated = (
        selected.evaluate(
            assets, spatial=SpatialMatch.covering(preferred_supports=["first"])
        )
        .probabilities([0.9])
        .read()
    )
    assert evaluated.report.valid is True
    assert evaluated.table["support_id"].to_pylist() == ["first"]


def test_release_combines_atomic_rename_projected_across_payloads(
    tmp_path: Path,
) -> None:
    first = (
        _converted_fit("company:coast", "coast", unit=LengthUnit.CENTIMETRE)
        .rename(variables={"company:coast": "company:river"})
        .write(tmp_path / "coast.parquet")
    )
    second = (
        _converted_fit("company:river", "river", unit=LengthUnit.MILLIMETRE)
        .rename(variables={"company:river": "company:coast"})
        .write(tmp_path / "river.parquet")
    )
    result = CRCOpen(catalog=_release_index(tmp_path, first, second)).read()
    assert result.report.valid is True
    assert result.table["variable_id"].to_pylist() == ["company:river", "company:coast"]
    rename = result.metadata.provenance[-1]
    assert isinstance(rename, IdentityRename)
    assert rename.mapping == {
        "company:coast": "company:river",
        "company:river": "company:coast",
    }


def test_release_inspects_series_with_conflicting_combined_history(
    tmp_path: Path,
) -> None:
    variable = "company:depth"
    first = (
        _converted_fit(variable, "first", unit=LengthUnit.CENTIMETRE)
        .declare(period=Period.window(2000, 2004))
        .write(tmp_path / "first.parquet")
    )
    second = (
        _converted_fit(variable, "second", unit=LengthUnit.MILLIMETRE)
        .convert_units(variable=variable, unit=LengthUnit.CENTIMETRE)
        .declare(period=Period.window(2005, 2009))
        .write(tmp_path / "second.parquet")
    )
    source = CRCOpen(catalog=_release_index(tmp_path, first, second))
    references = source.inspect().series
    assert len(references) == 2
    with pytest.raises(
        ValidationError, match="Conflicting release provenance for variable"
    ):
        source.read()
    assert source.select(series=references[1]).read().table[
        "support_id"
    ].to_pylist() == ["second"]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("zero_rows", "produced count disagrees with the Parquet footer"),
        ("field", "Arrow schema contradicts its indexed declaration"),
        ("incomplete", "requires a complete, valid report"),
        ("no_data", "report counts disagree with the consumed rows"),
    ],
)
def test_selected_release_payload_requires_complete_consistent_counts_and_schema(
    tmp_path: Path, damage: str, message: str
) -> None:
    recipe, index, artifact = release(tmp_path)
    table = pq.read_table(artifact)
    document = json.loads(index.read_text())
    indexed = document["datasets"][0]
    if damage == "zero_rows":
        table = table.slice(0, 0)
    elif damage == "field":
        table = table.drop(["failure"])
    else:
        report = indexed["metadata"]["report"]
        if damage == "incomplete":
            report.update(complete=False, valid=None)
        else:
            report["counts"]["no_data"] = 1
        table = table.replace_schema_metadata(
            {b"crc.dataset": json.dumps(indexed["metadata"]).encode()}
        )
    pq.write_table(table, artifact)
    indexed["sha256"] = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    index.write_text(json.dumps(document))

    assert recipe.inspect().series
    assert recipe.validate().valid is False
    with pytest.raises(ValidationError, match=message):
        recipe.read(limit=1)
    with pytest.raises(ValidationError, match=message):
        recipe.select(area=Bounds(west=23.0, south=59.0, east=25.0, north=61.0)).read()


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


def test_release_rows_repeated_across_datasets_are_rejected(tmp_path: Path) -> None:
    _, index, artifact = release(tmp_path)
    document = json.loads(index.read_text())
    copy = artifact.with_name("copy.parquet")
    copy.write_bytes(artifact.read_bytes())
    document["datasets"].append(
        {**document["datasets"][0], "id": "copy", "uri": copy.name}
    )
    index.write_text(json.dumps(document))
    with pytest.raises(ValidationError, match="duplicate scientific row key"):
        CRCOpen(catalog=index).read()
    # Equal files fetched at once on threads share one owned snapshot.
    for _ in range(20):
        with pytest.raises(ValidationError, match="duplicate scientific row key"):
            CRCOpen(catalog=index).read(runtime=Runtime(workers=4))


def test_release_files_outside_the_selected_area_are_not_fetched(
    tmp_path: Path,
) -> None:
    recipe, index, artifact = release(tmp_path)
    document = json.loads(index.read_text())
    extent = document["datasets"][0]["extent"]
    assert extent is not None
    artifact.write_bytes(b"not the indexed file")

    outside = recipe.select(area=Bounds(west=-10.0, south=-10.0, east=-5.0, north=-5.0))
    assert outside.read().table.num_rows == 0
    variable = recipe.inspect().series[0].variable.id
    assert outside.select(variables=[variable]).read().table.num_rows == 0
    inside = Bounds(
        west=extent["west"],
        south=extent["south"],
        east=extent["east"],
        north=extent["north"],
    )
    with pytest.raises(OSError, match="content identity"):
        recipe.select(area=inside).read()


def test_evaluation_fetches_only_files_that_can_cover_its_assets(
    tmp_path: Path,
) -> None:
    recipe, _, artifact = release(tmp_path)
    original = artifact.read_bytes()
    artifact.write_bytes(b"not the indexed file")

    away = Assets({"asset_id": ["away"], "longitude": [-7.0], "latitude": [-7.0]})
    result = recipe.evaluate(away).probabilities([0.9]).read()
    assert result.table["failure"][0].as_py()["code"] == "no_covering_support"
    here = Assets({"asset_id": ["here"], "longitude": [24.0], "latitude": [60.0]})
    with pytest.raises(OSError, match="content identity"):
        recipe.evaluate(here).probabilities([0.9]).read()

    artifact.write_bytes(original)
    both = Assets(
        {
            "asset_id": ["away", "here"],
            "longitude": [-7.0, 24.0],
            "latitude": [-7.0, 60.0],
        }
    )
    evaluated = recipe.evaluate(both).probabilities([0.9]).read()
    assert evaluated.table["value"].to_pylist() == [None, 0.3]


def test_evaluation_keeps_series_whose_only_file_is_not_read(tmp_path: Path) -> None:
    def dataset(name: str, longitude: float, period: Period) -> Dataset:
        return Dataset.from_quantiles(
            {
                "support_id": [name] * 2,
                "longitude": [longitude] * 2,
                "latitude": [60.0] * 2,
                "probability": [0.9, 0.99],
                "value": [0.3, 0.85],
            },
            variable=Variable(
                "company:depth", measurement=DepthAboveGround(unit=LengthUnit.METRE)
            ),
            probability=Probability.annual(tail=Tail.UPPER),
            period=period,
        ).write(tmp_path / f"{name}.parquet")

    near = dataset("near", 24.0, Period.window(2000, 2020))
    far = dataset("far", -100.0, Period.window(2030, 2050))
    index = _release_index(tmp_path, near, far)
    assert far.path is not None
    far.path.write_bytes(b"never read")
    assets = Assets({"asset_id": ["site"], "longitude": [24.0], "latitude": [60.0]})
    result = CRCOpen(catalog=index).evaluate(assets).probabilities([0.9]).read()
    rows = sorted(
        result.table.select(["period_id", "value", "failure"]).to_pylist(),
        key=lambda row: row["value"] is None,
    )
    assert [row["value"] for row in rows] == [0.3, None]
    assert rows[1]["failure"]["code"] == "no_covering_support"
    assert {row["period_id"] for row in rows} == {
        Period.window(2000, 2020).id,
        Period.window(2030, 2050).id,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda document: document.update(schema_version="1.0"), "current release"),
        (lambda document: document["datasets"][0].pop("extent"), "missing"),
        (
            lambda document: document["datasets"][0].update(
                extent={"west": 0.0, "south": 10.0, "east": 1.0, "north": 5.0}
            ),
            "extent",
        ),
    ],
)
def test_release_index_requires_current_extents(
    tmp_path: Path, change: Callable[[dict[str, Any]], object], message: str
) -> None:
    _, index, _ = release(tmp_path)
    document = json.loads(index.read_text())
    change(document)
    index.write_text(json.dumps(document))
    with pytest.raises(ValidationError, match=message):
        CRCOpen(catalog=index).read()


def _depth(path: Path, variable: str, value: float) -> Dataset:
    return Dataset.from_quantiles(
        {
            "support_id": ["harbour"] * 2,
            "longitude": [24.0] * 2,
            "latitude": [60.0] * 2,
            "probability": [0.9, 0.99],
            "value": [value, value + 0.5],
        },
        variable=Variable(
            variable, measurement=DepthAboveGround(unit=LengthUnit.METRE)
        ),
        probability=Probability.annual(tail=Tail.UPPER),
        period=Period.window(2000, 2020),
    ).write(path, overwrite=True)


def _publish(tmp_path: Path, *datasets: Dataset) -> Path:
    index = _release_index(tmp_path, *datasets)
    pointer = tmp_path / "_CURRENT.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "release": {
                    "id": "fixture-release",
                    "uri": index.name,
                    "sha256": "sha256:"
                    + hashlib.sha256(index.read_bytes()).hexdigest(),
                },
            }
        )
    )
    return pointer


def test_a_superseded_reuse_cache_names_the_release_and_the_fix(
    tmp_path: Path,
) -> None:
    first = _depth(tmp_path / "first.parquet", "company:first", 0.3)
    second = _depth(tmp_path / "second.parquet", "company:second", 0.3)
    pointer = _publish(tmp_path, first, second)
    cache = tmp_path / "cache"
    recipe = CRCOpen(catalog=pointer)
    reused = recipe.select(variables="company:first").read(
        runtime=Runtime(cache=Cache.reuse(cache))
    )
    assert reused.report.valid
    # The publisher replaces the release; the cache never held `second`.
    _publish(
        tmp_path, first, _depth(tmp_path / "second.parquet", "company:second", 0.6)
    )
    with pytest.raises(OSError, match="release fixture-release in the reuse cache"):
        recipe.select(variables="company:second").read(
            runtime=Runtime(cache=Cache.reuse(cache))
        )
    refreshed = recipe.select(variables="company:second").read(
        runtime=Runtime(cache=Cache.refresh(cache))
    )
    assert refreshed.report.valid
    assert (
        recipe.select(variables="company:second")
        .read(runtime=Runtime(cache=Cache.reuse(cache)))
        .table.equals(refreshed.table)
    )


def test_files_acquired_on_worker_threads_read_like_one_worker(
    tmp_path: Path,
) -> None:
    datasets = [
        _depth(tmp_path / f"{name}.parquet", f"company:{name}", value)
        for name, value in (("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4))
    ]
    recipe = CRCOpen(catalog=_publish(tmp_path, *datasets))
    serial = recipe.read(runtime=Runtime(cache=Cache.reuse(tmp_path / "one")))
    parallel = recipe.read(
        runtime=Runtime(cache=Cache.reuse(tmp_path / "three"), workers=3)
    )
    assert parallel.table.equals(serial.table)
    assert parallel.report.valid is serial.report.valid is True
    assert parallel.report.counts == serial.report.counts
    assert [r.uri for s in parallel.report.source_identities for r in s.resources] == [
        r.uri for s in serial.report.source_identities for r in s.resources
    ]
