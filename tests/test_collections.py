"""Collection composition, typed manifests, and publication conformance."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from crc_sdk import (
    Bounds,
    Cache,
    CanonicalReader,
    CollectionMember,
    CollectionMetadata,
    CollectionResource,
    CollectionSelection,
    Dataset,
    DatasetCollection,
    DatasetMetadata,
    DataSpec,
    ExecutionReport,
    FitFailure,
    FittingDiagnostic,
    Period,
    Probability,
    QuantileColumns,
    ReleaseReference,
    Runtime,
    Samples,
    SeriesSpec,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
    ValidationError,
    Variable,
    _collection_bundle,
)
from crc_sdk._collection_bundle import MANIFEST
from crc_sdk._collection_schema import _collection_report, collection_metadata_schema
from crc_sdk._runtime import ProgressEvent
from crc_sdk._schema import rows_to_table, table_to_rows

HERE = Path(__file__).resolve().parents[1]


def _variable(
    name: str = "company:depth",
    unit: cf.LengthUnit = cf.LengthUnit.METRE,
    label: str | None = None,
) -> Variable:
    return Variable(name, measurement=cf.DepthAboveGround(unit=unit), label=label)


def _dataset(
    *variables: Variable,
    support: str = "point",
    longitude: float = 1,
) -> Dataset:
    declarations = variables or (_variable(),)
    n = len(declarations)
    return Dataset.from_quantiles(
        {
            "support_id": [support] * (2 * n),
            "longitude": [longitude] * (2 * n),
            "latitude": [2.0] * (2 * n),
            "probability": [0.8, 0.95] * n,
            "value": [0.4, 1.0] * n,
            "variable": [v.id for v in declarations for _ in range(2)],
        },
        spec=DataSpec(
            declarations,
            series=tuple(
                SeriesSpec(
                    variable,
                    period=Period.window(1991, 2020),
                    probability=Probability.annual(tail=cf.Tail.UPPER),
                )
                for variable in declarations
            ),
        ),
        columns=QuantileColumns(variable_id="variable"),
    )


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("counts", {"requested": 1000, "skipped": 999}),
        (
            "diagnostics",
            [
                {
                    "code": "fabricated",
                    "message": "No such event",
                    "severity": "warning",
                    "stage": "validation",
                }
            ],
        ),
        (
            "source_identities",
            [{"provider": "local", "product": "fabricated", "resources": []}],
        ),
    ],
)
def test_bundle_reconciles_the_whole_report_with_member_resources(
    tmp_path: Path, field: str, forged: object
) -> None:
    path = tmp_path / "bundle"
    DatasetCollection({"member": _dataset()}).write(path)
    manifest = path / MANIFEST
    payload = json.loads(manifest.read_text())
    if field == "counts":
        payload["report"][field].update(forged)
    else:
        payload["report"][field] = forged
    manifest.write_text(json.dumps(payload))
    reopened = DatasetCollection.open(path)
    with pytest.raises(ValidationError, match=f"collection report {field}"):
        _ = reopened.report
    with pytest.raises(ValidationError, match=f"collection report {field}"):
        reopened.read()
    assert reopened.validate().valid is False


class _Reader(CanonicalReader):
    def __init__(
        self, dataset: Dataset, *, source: SourceIdentity | None = None
    ) -> None:
        self.result = dataset.read()
        self.source = source
        self.inspections = 0
        self.streams = 0
        self.delivered = 0
        self.closed = 0

    def inspect(self) -> DatasetMetadata:
        self.inspections += 1
        return (
            replace(self.result.metadata, source=self.source)
            if self.source is not None
            else self.result.metadata
        )

    @contextmanager
    def batches(self, *, batch_rows: int = 65536) -> Iterator[Iterator[pa.RecordBatch]]:
        self.streams += 1

        def emit() -> Iterator[pa.RecordBatch]:
            for batch in self.result.table.to_batches(max_chunksize=batch_rows):
                self.delivered += batch.num_rows
                yield batch

        try:
            yield emit()
        finally:
            self.closed += 1


def test_normalized_read_keeps_members_and_reuses_shared_resource(
    tmp_path: Path,
) -> None:
    source = _Reader(_dataset())
    collection = DatasetCollection({"reference": source, "projected": source})
    assert source.inspections == 0
    with collection.execute() as query:
        assert query.validate().valid is True
        result = query.read()
        persisted = query.write(tmp_path / "saved")
    assert source.inspections == source.streams == source.closed == 1
    assert source.delivered == 1
    assert result.table["member_id"].to_pylist() == ["reference", "projected"]
    resource_ids = result.table["resource_id"].to_pylist()
    assert resource_ids[0] == resource_ids[1]
    assert len(list((tmp_path / "saved" / "resources").iterdir())) == 1
    assert persisted.read().table.equals(result.table)
    assert persisted.report == result.report


def test_collection_inspection_and_early_streaming_stay_lazy() -> None:
    reader = _Reader(_dataset(_variable("a"), _variable("b")))
    recipe = DatasetCollection({"member": reader})
    assert recipe.explain().operations
    assert reader.inspections == reader.streams == 0
    info = recipe.inspect()
    assert isinstance(info.metadata, CollectionMetadata)
    assert {ref.variable.id for ref in info.series} == {"a", "b"}
    assert reader.inspections == 1 and reader.streams == 0
    with recipe.batches(batch_rows=1, runtime=Runtime(batch_rows=1)) as stream:
        batch = next(iter(stream))
        assert batch.num_rows == 1
        assert reader.delivered == 1
        assert not stream.report.complete
    assert not stream.report.complete and reader.closed == 1


def test_explicit_empty_selection_avoids_discovery_and_bulk_work(
    tmp_path: Path,
) -> None:
    reader = _Reader(_dataset())
    selection = DatasetCollection({"plain": reader}).select(
        members="plain", variables=[]
    )
    report = selection.write(tmp_path / "empty").read()
    assert reader.inspections == reader.streams == 0
    assert report.table.num_rows == 0 and report.report.valid is True
    assert report.metadata.selection == CollectionSelection(
        members=("plain",),
        variables=(),
    )
    assert report.metadata.members == (CollectionMember(id="plain", resource_ids=()),)
    assert report.metadata.resources == ()


def test_selected_bundle_moves_and_reopens_offline_with_typed_selection(
    tmp_path: Path,
) -> None:
    first = _dataset(_variable(), longitude=1)
    second = _dataset(_variable(), longitude=4)
    path = tmp_path / "source.parquet"
    first.write(path)
    reference = first.inspect().series[0]
    area = Bounds(west=0, south=0, east=2, north=3)
    recipe = (
        DatasetCollection({"source": Dataset.open(path), "other": second})
        .select(members="source", series=reference, area=area)
        .rename(variables={reference.variable.id: "company:hazard"})
    )
    with recipe.execute() as execution:
        direct = execution.read()
        saved = execution.write(tmp_path / "bundle")
    assert isinstance(saved, DatasetCollection)
    assert saved.path is not None
    moved = saved.path.rename(tmp_path / "moved")
    path.unlink()
    opened = DatasetCollection.open(moved)
    assert opened.metadata is not None
    assert opened.metadata.selection.series is not None
    assert opened.metadata.selection.series[0].variable.id == "company:hazard"
    assert opened.metadata.selection.area == area
    restored = opened.read(runtime=Runtime(cache=Cache.offline(tmp_path / "cache")))
    assert restored.table.equals(direct.table)
    assert restored.report == direct.report
    assert restored.metadata.members == direct.metadata.members
    assert (
        opened.select(members=None, series=None, area=None)
        .read()
        .table.equals(direct.table)
    )


@pytest.mark.parametrize(
    ("selector", "value", "message"),
    [
        ("members", ["missing"], "saved member selection"),
        ("members", [], "saved member selection"),
        ("variables", ["missing"], "contradicts the saved selection"),
        ("variables", [], "empty collection selection"),
        ("area", {"west": 5, "south": 5, "east": 7, "north": 7}, "saved area"),
    ],
)
def test_bundle_rejects_rows_outside_its_saved_selection(
    tmp_path: Path, selector: str, value: object, message: str
) -> None:
    path = tmp_path / "bundle"
    DatasetCollection({"first": _dataset(), "second": _dataset()}).write(path)
    manifest = path / MANIFEST
    document = json.loads(manifest.read_text())
    document["selection"][selector] = value
    manifest.write_text(json.dumps(document))

    with pytest.raises(ValidationError, match=message):
        DatasetCollection.open(path).read()


def test_selecting_one_member_still_checks_integrity_of_the_bundle_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bundle"
    saved = DatasetCollection(
        {"first": _dataset(), "second": _dataset(longitude=4)}
    ).write(path)
    assert saved.metadata is not None
    expected = saved.metadata.resources[1].path
    assert expected is not None
    (path / expected).unlink()

    with pytest.raises(FileNotFoundError):
        DatasetCollection.open(path).select(members="first").read()


def test_full_terminals_verify_unselected_resource_checksums(tmp_path: Path) -> None:
    path = tmp_path / "bundle"
    saved = DatasetCollection(
        {"first": _dataset(), "second": _dataset(longitude=4)}
    ).write(path)
    assert saved.metadata is not None
    expected = saved.metadata.resources[1].path
    assert expected is not None
    resource = path / expected
    resource.write_bytes(resource.read_bytes() + b"changed")

    selected = DatasetCollection.open(path).select(members="first")
    preview = selected.read(limit=1)
    assert preview.table.num_rows == 1
    assert preview.report.complete is False
    with pytest.raises(OSError, match="checksum mismatch"):
        selected.read()
    with pytest.raises(OSError, match="checksum mismatch"):
        selected.validate()
    with pytest.raises(OSError, match="checksum mismatch"):
        selected.write(tmp_path / "selected")
    with pytest.raises(OSError, match="checksum mismatch"):
        saved.select(members=[]).validate()
    assert not (tmp_path / "selected").exists()


def test_only_participating_conflicts_are_reconciled() -> None:
    identifiers = {
        "first": _dataset(_variable(label="first description"), _variable("isolated")),
        "second": _dataset(_variable(unit=cf.LengthUnit.CENTIMETRE)),
    }
    source = DatasetCollection(identifiers)
    assert source.validate().valid is False
    with pytest.raises(ValidationError, match="conflicting scientific"):
        source.read()
    assert source.select(members="second").read().table.num_rows == 1
    assert source.select(variables="isolated").read().table.num_rows == 1
    assert (
        source.select(variables="isolated").select(variables=None).validate().valid
        is False
    )
    with pytest.raises(ValueError, match="unknown requested"):
        source.select(members="missing").read()
    with pytest.raises(ValueError, match="unknown requested"):
        source.select(variables="missing").read()
    equal_science = DatasetCollection(
        {
            "first": _dataset(_variable(label="original")),
            "second": _dataset(_variable(label="different")),
        }
    ).read()
    assert equal_science.table.num_rows == 2
    assert len(equal_science.metadata.spec.series) == 1


def test_rename_rejects_global_collision_with_an_unchanged_member() -> None:
    original = DatasetCollection(
        {
            "a": _dataset(_variable("a")),
            "b": _dataset(_variable("b")),
        }
    )
    with pytest.raises(ValueError, match="collides"):
        original.rename(variables={"a": "b"}).read()
    renamed = original.select(members="a").rename(variables={"a": "b"}).read()
    assert renamed.table["variable_id"].to_pylist() == ["b"]


def test_qualified_series_match_source_not_collection_member_and_round_trip(
    tmp_path: Path,
) -> None:
    release = ReleaseReference("release-one", "sha256:" + "a" * 64)

    def reader(model: str, member: str) -> CanonicalReader:
        return _Reader(
            _dataset(),
            source=SourceIdentity(
                provider="project",
                product="depth",
                model=model,
                member=member,
                release=release,
            ),
        )

    collection = DatasetCollection(
        {
            "baseline": reader("model-a", "r1"),
            "other-model": reader("model-b", "r1"),
            "other-member": reader("model-a", "r2"),
        }
    )
    references = collection.inspect().series
    assert [(item.model, item.member) for item in references] == [
        ("model-a", "r1"),
        ("model-b", "r1"),
        ("model-a", "r2"),
    ]
    reference = references[0]
    recipe = collection.select(series=reference)
    assert recipe.read().table["member_id"].to_pylist() == ["baseline"]
    written = recipe.write(tmp_path / "qualified")
    assert written.path is not None
    saved = DatasetCollection.open(written.path)
    assert saved.metadata is not None
    assert saved.metadata.selection.series == (reference,)
    assert saved.read().table["member_id"].to_pylist() == ["baseline"]
    with pytest.raises(ValueError, match="model or member differs"):
        collection.select(series=replace(reference, member="missing")).read()


def _install_shards(
    directory: Path,
    datasets: tuple[Dataset, ...],
) -> DatasetCollection:
    directory.mkdir()
    (directory / "resources").mkdir()
    resources = []
    for number, dataset in enumerate(datasets):
        relative = f"resources/{number:064x}.parquet"
        path = directory / relative
        persisted = dataset.write(path)
        assert persisted.metadata is not None
        resources.append(
            CollectionResource(
                id=relative,
                metadata=persisted.metadata,
                path=relative,
                sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    manifest = CollectionMetadata(
        members=(
            CollectionMember(
                id="member",
                resource_ids=tuple(resource.id for resource in resources),
            ),
        ),
        resources=tuple(resources),
        selection=CollectionSelection(),
        report=_collection_report(
            (resource.metadata.report for resource in resources),
            complete=True,
        ),
    )
    (directory / MANIFEST).write_text(manifest.to_json())
    return DatasetCollection.open(directory)


def test_sharded_member_rejects_scientific_overlap(tmp_path: Path) -> None:
    recipe = _install_shards(tmp_path / "invalid", (_dataset(), _dataset()))
    report = recipe.validate(runtime=Runtime(batch_rows=1))
    assert report.valid is False
    assert any("duplicate scientific row key" in d.message for d in report.diagnostics)
    with pytest.raises(ValidationError, match="duplicate scientific row"):
        recipe.write(tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.*"))


def test_disjoint_shards_are_not_pooled_into_an_artificial_resource(
    tmp_path: Path,
) -> None:
    recipe = _install_shards(
        tmp_path / "shards",
        (_dataset(support="one"), _dataset(support="two")),
    )
    result = recipe.read()
    assert result.table["member_id"].to_pylist() == ["member", "member"]
    assert len(set(result.table["resource_id"].to_pylist())) == 2
    assert recipe.write(tmp_path / "copy").read().table.equals(result.table)


def test_disjoint_fitted_shards_retain_their_own_unit_conversion(
    tmp_path: Path,
) -> None:
    variable = _variable()

    def fitted(support: str) -> Dataset:
        return (
            Samples.from_table(
                {
                    "support_id": [support] * 5,
                    "longitude": [1.0] * 5,
                    "latitude": [2.0] * 5,
                    "time": [2000, 2001, 2002, 2003, 2004],
                    "value": [1.0, 2.0, 4.0, 5.0, 8.0],
                },
                variable=variable,
            )
            .fit(cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT))
            .convert_units(variable=variable, unit=cf.LengthUnit.CENTIMETRE)
        )

    recipe = _install_shards(tmp_path / "converted", (fitted("one"), fitted("two")))
    assert recipe.validate().valid is True
    assert recipe.read().table["support_id"].to_pylist() == ["one", "two"]


def test_disjoint_shards_support_distinct_spatial_contracts(tmp_path: Path) -> None:
    point = _dataset(support="concrete point")
    result = point.read()
    cell = replace(
        table_to_rows(result.table)[0],
        support_id="delivery cell",
        geometry=None,
        cell_id=h3.latlng_to_cell(30, 50, 5),
    )
    cell_metadata = replace(
        result.metadata,
        spatial=SpatialContract(supports=(SupportKind.CELL,), index_resolution=5),
    )
    cells = Dataset._collected(rows_to_table([cell], cell_metadata), cell_metadata)
    recipe = _install_shards(tmp_path / "mixed_spatial", (point, cells))
    restored = recipe.read()
    assert restored.report.complete and restored.report.valid is True
    assert restored.table["support_id"].to_pylist() == [
        "concrete point",
        "delivery cell",
    ]
    assert restored.table["geometry"].null_count == 1


def test_hash_is_checked_on_owned_resource_even_if_cache_contains_old_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bundle"
    persisted = DatasetCollection({"a": _dataset()}).write(destination)
    resource = next((destination / "resources").iterdir())
    cache = tmp_path / "cache"
    persisted.read(runtime=Runtime(cache=Cache.reuse(cache)))
    resource.write_bytes(resource.read_bytes() + b"changed")
    with pytest.raises(OSError, match="checksum mismatch"):
        persisted.read(runtime=Runtime(cache=Cache.reuse(cache)))


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="atomic directory exchange"
)
def test_overwrite_preserves_existing_bundle_on_failure(tmp_path: Path) -> None:
    source = DatasetCollection({"a": _dataset()})
    path = tmp_path / "bundle"
    source.write(path)
    files_before = {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }
    with pytest.raises(FileExistsError):
        source.write(path)

    def stop(event: ProgressEvent) -> None:
        if event.stage is Stage.WRITING:
            raise RuntimeError("abort publication")

    with pytest.raises(RuntimeError, match="abort publication"):
        source.select(members=[]).write(
            path,
            overwrite=True,
            runtime=Runtime(progress=stop),
        )
    files_after = {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }
    assert files_after == files_before
    assert not list(tmp_path.glob(".bundle.*"))
    empty = source.select(members=[]).write(path, overwrite=True).read()
    assert not empty.table.num_rows
    assert empty.metadata.selection.members == ()


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="atomic directory exchange"
)
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_overwrite_preserves_a_directory_moved_into_destination_during_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback_fails: bool
) -> None:
    recipe = DatasetCollection({"first": _dataset()})
    path = tmp_path / "bundle"
    recipe.write(path)
    moved = tmp_path / "moved"
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes").write_text("keep")
    original = _collection_bundle._rename_bundle

    def interleave(staged: Path, target: Path, *, exchange: bool) -> None:
        if not moved.exists():
            target.rename(moved)
            other.rename(target)
        elif rollback_fails:
            raise OSError("rollback unavailable")
        original(staged, target, exchange=exchange)

    monkeypatch.setattr(_collection_bundle, "_rename_bundle", interleave)
    with pytest.raises(
        OSError, match="collection (destination changed|rollback failed)"
    ):
        recipe.write(path, overwrite=True)

    if rollback_fails:
        (retained,) = tmp_path.glob(".bundle.*/bundle/notes")
        assert retained.read_text() == "keep"
    else:
        assert (path / "notes").read_text() == "keep"
        assert not list(tmp_path.glob(".bundle.*"))
    assert DatasetCollection.open(moved).validate().valid is True


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="atomic directory exchange"
)
def test_overwrite_keeps_bundle_published_by_another_writer(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    recipe = DatasetCollection({"first": _dataset()})
    recipe.write(target)
    replacement = DatasetCollection({"second": _dataset()})
    complete = False

    def publish_replacement(event: ProgressEvent) -> None:
        nonlocal complete
        if not complete and event.stage is Stage.WRITING:
            replacement.write(target, overwrite=True)
            complete = True

    with pytest.raises(OSError, match="collection destination changed"):
        recipe.write(
            target, overwrite=True, runtime=Runtime(progress=publish_replacement)
        )

    assert DatasetCollection.open(target).read().table["member_id"].to_pylist() == [
        "second"
    ]
    assert not list(tmp_path.glob(".bundle.*"))


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="atomic directory exchange"
)
def test_overwrite_preserves_new_files_in_the_displaced_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "bundle"
    recipe = DatasetCollection({"original": _dataset()})
    recipe.write(path)
    original = _collection_bundle._retire_bundle
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)

    def add_file(staging: Path, bundle: _collection_bundle._ExistingBundle) -> None:
        file = os.open(
            "resources/notes", os.O_WRONLY | os.O_CREAT | os.O_EXCL, dir_fd=directory
        )
        with os.fdopen(file, "w") as stream:
            stream.write("keep")
        original(staging, bundle)

    monkeypatch.setattr(_collection_bundle, "_retire_bundle", add_file)
    try:
        published = recipe.select(members=[]).write(path, overwrite=True)
    finally:
        os.close(directory)

    assert published.validate().valid is True
    (retained,) = tmp_path.glob(".bundle.*/bundle/resources/notes")
    assert retained.read_text() == "keep"
    assert str(retained.parent.parent) in caplog.text


def test_refuses_to_replace_arbitrary_or_extended_directory(tmp_path: Path) -> None:
    recipe = DatasetCollection({"a": _dataset()})
    unrelated = tmp_path / "unowned"
    unrelated.mkdir()
    (unrelated / "notes").write_text("keep")
    with pytest.raises(ValueError, match="existing complete collection"):
        recipe.write(unrelated, overwrite=True)
    assert (unrelated / "notes").read_text() == "keep"
    saved = tmp_path / "bundle"
    recipe.write(saved)
    (saved / "notes").write_text("keep")
    with pytest.raises(ValueError, match="outside"):
        recipe.write(saved, overwrite=True)
    assert (saved / "notes").read_text() == "keep"


def test_generated_collection_contract_and_shared_cases() -> None:
    sdk = files("crc_sdk").joinpath("contracts")
    framework = files("crc_framework").joinpath("contracts")
    artifact = sdk.joinpath("schemas", "collection-1.0.schema.json")
    schema = json.loads(artifact.read_text())
    assert schema == collection_metadata_schema()
    references = []
    for directory in (
        framework.joinpath("scientific"),
        framework.joinpath("numerical"),
    ):
        for artifact in directory.iterdir():
            if artifact.name.endswith(".schema.json"):
                document = json.loads(artifact.read_text())
                references.append((document["$id"], Resource.from_contents(document)))
    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
        registry=Registry().with_resources(references),
    )
    cases = json.loads(sdk.joinpath("fixtures", "collection-1.0.json").read_text())
    for case in cases["accepted"]:
        document = case["document"]
        validator.validate(document)
        assert (
            json.loads(CollectionMetadata.from_json(json.dumps(document)).to_json())
            == document
        )
    for case in cases["rejected"]:
        document = case["document"]
        assert bool(list(validator.iter_errors(document))) is case["structural"]
        with pytest.raises((TypeError, ValueError)):
            CollectionMetadata.from_json(json.dumps(document))


def test_collected_normalized_result_retains_resource_and_science() -> None:
    source = DatasetCollection({"a": _dataset(), "b": _dataset()}).read()
    collected = DatasetCollection._collected(source.table, source.metadata)
    result = collected.read()
    assert result.table.equals(source.table)
    assert result.metadata.members == source.metadata.members
    assert result.report == source.report
    with pytest.raises(ValueError, match="complete valid"):
        metadata = replace(source.metadata, report=ExecutionReport(False, None))
        table = source.table.replace_schema_metadata(
            {
                b"crc.collection": metadata.to_json().encode(),
            }
        )
        DatasetCollection._collected(table, metadata)


def _short_sample() -> Samples:
    return Samples.from_table(
        {
            "support_id": ["gauge", "gauge"],
            "longitude": [1.0, 1.0],
            "latitude": [2.0, 2.0],
            "time": [1999, 2000],
            "value": [0.5, 1.0],
        },
        variable=_variable(),
    )


@pytest.mark.parametrize("policy", [FitFailure.KEEP_NO_DATA, FitFailure.SKIP])
def test_collection_persists_failed_fit_records_by_policy(
    tmp_path: Path,
    policy: FitFailure,
) -> None:
    fitted = _short_sample().fit(
        cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT),
        on_failure=policy,
    )
    collection = DatasetCollection({"past": fitted, "future": fitted})
    with collection.execute() as execution:
        report = execution.validate()
        assert report.complete and report.valid
        assert report.counts.no_data == (2 if policy is FitFailure.KEEP_NO_DATA else 0)
        assert report.counts.skipped == (2 if policy is FitFailure.SKIP else 0)
        assert len(report.diagnostics) == 1
        diagnostic = report.diagnostics[0]
        assert isinstance(diagnostic, FittingDiagnostic)
        assert diagnostic.record.evidence_count == 2
        collected = execution.read()
        artifact = execution.write(tmp_path / "with_diagnostics")
    assert isinstance(artifact, DatasetCollection)
    reopened = artifact.read()
    assert reopened.table.equals(collected.table)
    assert reopened.report == report


def test_collection_strict_fit_failure_keeps_typed_report() -> None:
    fitted = _short_sample().fit(
        cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT)
    )
    collection = DatasetCollection({"baseline": fitted})
    report = collection.validate()
    assert report.valid is False
    assert len(report.diagnostics) == 1
    assert isinstance(report.diagnostics[0], FittingDiagnostic)
    assert report.diagnostics[0].record.evidence_count == 2
    with pytest.raises(ValidationError) as caught:
        collection.read()
    assert caught.value.report == report
