"""Real recipe execution, persistence, reuse, and selection invariants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    DepthAboveGround,
    DistributionFamily,
    LengthUnit,
    QuantileCurve,
    Ratio,
    RatioUnit,
    SampleEstimator,
    SampleFit,
    Tail,
)

from crc_sdk import (
    EDO,
    ERA5,
    CRCOpen,
    Layer,
    NOAASeaLevel,
    Observations,
    Projections,
    USDACropland,
    _recipes,
)
from crc_sdk._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionReport,
    IdentityKind,
    IdentityRename,
    ResourceIdentity,
    SourceIdentity,
)
from crc_sdk._declarations import (
    Calendar,
    DataSpec,
    Period,
    Probability,
    SeriesSpec,
    Variable,
)
from crc_sdk._errors import CapabilityError, ValidationError
from crc_sdk._evidence import (
    EvidenceKind,
    EvidenceMetadata,
    evidence_arrow_schema,
    fit_sample_table,
)
from crc_sdk._graph import ExecutionScope, Node, NodeStream
from crc_sdk._inspection import Bounds, Inspection
from crc_sdk._recipes import CanonicalReader, Dataset, Samples
from crc_sdk._runtime import Runtime
from crc_sdk._schema import metadata_from_table, table_to_rows
from crc_sdk._selection import SelectionState
from crc_sdk._source_io import SourceNode
from crc_sdk._table_input import QuantileColumns, normalize_samples


def raw_samples() -> dict[str, list[object]]:
    return {
        "support_id": ["station-a"] * 5,
        "longitude": [-79.38] * 5,
        "latitude": [43.65] * 5,
        "time": [2000, 2001, 2002, 2003, 2004],
        "value": [1.0, 3.0, 2.0, 5.0, 4.0],
    }


def sample_recipe() -> Samples:
    return Samples.from_table(
        raw_samples(), variable=Variable("my:rainfall", measurement=Ratio())
    )


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        (CRCOpen(), Dataset),
        (ERA5(), Samples),
        (EDO(), Observations),
        (NOAASeaLevel(), Projections),
        (USDACropland(), Layer),
    ],
)
def test_scientific_transforms_return_the_canonical_data_kind(
    source: Dataset | Samples | Observations | Projections | Layer,
    kind: type[Dataset]
    | type[Samples]
    | type[Observations]
    | type[Projections]
    | type[Layer],
) -> None:
    assert type(source.declare()) is kind
    assert type(source.rename(variables={"source:value": "company:value"})) is kind
    assert (
        type(source.convert_units(variable="source:value", unit=LengthUnit.METRE))
        is kind
    )


def test_inherited_user_input_constructors_return_canonical_kinds() -> None:
    assert (
        type(ERA5.from_table(raw_samples(), variable=Variable("user:rain"))) is Samples
    )
    supplied = CRCOpen.from_quantiles(
        {
            "support_id": ["a", "a"],
            "longitude": [0, 0],
            "latitude": [0, 0],
            "probability": [0.9, 0.99],
            "value": [1, 2],
        },
        variable=Variable("user:depth"),
    )
    assert type(supplied) is Dataset


def test_transformed_and_fitted_inspection_uses_declared_source_metadata() -> None:
    metadata = sample_recipe().read().metadata

    class InspectedSource(SourceNode[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]]):
        key = "inspection-without-data"
        state = SelectionState()

        def selected(self, state: SelectionState) -> InspectedSource:
            return self

        def inspection(self, scope: ExecutionScope) -> Inspection:
            return Inspection(schema=evidence_arrow_schema(metadata))

        def open(
            self, scope: ExecutionScope
        ) -> NodeStream[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]]:
            raise AssertionError("inspection opened source data")

    recipe = Samples._create(InspectedSource(), operations=())
    transformed = recipe.convert_units(
        variable="my:rainfall",
        unit=RatioUnit.PERCENT,
    ).rename(variables={"my:rainfall": "company:value"})
    inspection = transformed.fit(
        SampleFit(family=DistributionFamily.GUMBEL_RIGHT),
    ).inspect()
    assert inspection.series[0].variable.id == "company:value"
    assert inspection.series[0].measurement == Ratio(unit=RatioUnit.PERCENT)
    declared = inspection.metadata
    assert isinstance(declared, DatasetMetadata)
    assert len(declared.provenance) == 2


def fit_specification() -> SampleFit:
    return SampleFit(
        family=DistributionFamily.GUMBEL_RIGHT,
        estimator=SampleEstimator.L_MOMENTS,
    )


def quantile_recipe(*, scale: float = 1.0) -> Dataset:
    return Dataset.from_quantiles(
        {
            "support_id": ["site-a"] * 3,
            "longitude": [-79.38] * 3,
            "latitude": [43.65] * 3,
            "probability": [0.5, 0.9, 0.99],
            "value": [0.2 * scale, 0.8 * scale, 1.6 * scale],
        },
        variable=Variable(
            "my:depth", measurement=DepthAboveGround(unit=LengthUnit.METRE)
        ),
        probability=Probability.annual(tail=Tail.UPPER),
    )


def test_samples_fit_executes_its_complete_upstream_graph_automatically() -> None:
    result = sample_recipe().fit(fit_specification()).read()
    assert result.report.complete and result.report.valid
    rows = table_to_rows(result.table)
    assert len(rows) == 1 and rows[0].fit is not None
    assert rows[0].fit.evidence_count == rows[0].fit.used_count == 5
    assert rows[0].fit.specification == fit_specification()
    assert result.metadata.spec.variables[0].id == "my:rainfall"
    assert result.metadata.spec.series[0].scenario is None
    assert result.metadata.spec.series[0].period is None


def test_quantile_constructor_uses_supplied_evidence_without_any_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("A supplied quantile curve must not run a fit")

    monkeypatch.setattr(_recipes, "fit_sample_table", unexpected_fit)
    monkeypatch.setattr("crc_framework.fit_quantiles", unexpected_fit)
    result = quantile_recipe().read()
    row = table_to_rows(result.table)[0]
    assert isinstance(row.curve, QuantileCurve)
    assert row.curve.probabilities.tolist() == [0.5, 0.9, 0.99]
    assert row.curve.values.tolist() == [0.2, 0.8, 1.6]
    assert row.fit is None


def test_explicit_execution_reuses_source_and_fit_for_all_terminals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = sample_recipe()
    source = samples._node
    calls = {"source": 0, "fit": 0}

    class Counted(Node[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]]):
        key = source.key

        def open(
            self, scope: ExecutionScope
        ) -> NodeStream[EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]]:
            calls["source"] += 1
            return source.open(scope)

    samples = Samples._create(Counted(), operations=samples._operations)
    original = fit_sample_table

    def counted(*args: object, **kwargs: object) -> object:
        calls["fit"] += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_recipes, "fit_sample_table", counted)
    recipe = samples.fit(fit_specification())
    with recipe.execute(runtime=Runtime(batch_rows=2)) as execution:
        report = execution.validate()
        collected = execution.read()
        saved = execution.write(tmp_path / "rainfall.parquet")
        assert report.complete and report.valid
        assert collected.table.num_rows == 1
        assert saved.report is not None and saved.report.complete
        assert calls == {"source": 1, "fit": 1}
    assert report.complete and collected.report.complete
    assert collected.table.num_rows == 1
    assert saved.read().table.num_rows == 1


def test_saved_dataset_and_samples_reopen_with_science_and_reports(
    tmp_path: Path,
) -> None:
    depth = quantile_recipe()
    dataset = depth.write(tmp_path / "depth.parquet")
    assert dataset.path == tmp_path / "depth.parquet"
    assert dataset.metadata is not None
    assert dataset.metadata.spec.variables[0].measurement == DepthAboveGround(
        unit=LengthUnit.METRE
    )
    result = Dataset.open(dataset.path).read()
    assert result.report.complete and result.report.valid
    assert table_to_rows(result.table)[0].curve is not None
    sample_artifact = sample_recipe().write(tmp_path / "samples.parquet")
    assert sample_artifact.metadata is not None
    assert sample_artifact.path is not None
    reopened = Samples.open(sample_artifact.path)
    evidence = reopened.read()
    assert evidence.table.num_rows == 5
    assert evidence.metadata.spec == sample_artifact.metadata.spec
    assert evidence.report.complete and evidence.report.valid
    assert reopened.fit(fit_specification()).read().table.num_rows == 1


def test_collected_samples_fit_without_reopening_original_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.parquet"
    sample_recipe().write(path)
    collected = Samples.open(path).read()
    path.unlink()
    result = collected.fit(fit_specification()).read()
    assert result.table.num_rows == 1
    assert result.report.complete


def test_previews_are_incomplete_and_cannot_become_fit_evidence() -> None:
    samples = sample_recipe()
    preview = samples.read(limit=2, runtime=Runtime(batch_rows=1))
    assert preview.table.num_rows == 2
    assert not preview.report.complete and preview.report.valid is None
    with pytest.raises(CapabilityError, match="incomplete"):
        preview.fit(fit_specification())
    # Observing all rows is insufficient until EOF/global checks are observed.
    boundary = samples.read(limit=5, runtime=Runtime(batch_rows=1))
    assert boundary.table.num_rows == 5 and not boundary.report.complete
    exhausted = samples.read(limit=6, runtime=Runtime(batch_rows=1))
    assert exhausted.table.num_rows == 5 and exhausted.report.complete
    with samples.execute(runtime=Runtime(batch_rows=1)) as execution:
        assert execution.validate().complete
        complete_replay = execution.read(limit=5)
        partial_replay = execution.read(limit=2)
    assert complete_replay.report.complete
    assert not partial_replay.report.complete


def test_empty_selection_skips_fitting_and_selector_clear_restores_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("An empty selection must not fit any evidence")

    samples = sample_recipe()
    monkeypatch.setattr(_recipes, "fit_sample_table", unexpected_fit)
    empty = samples.fit(fit_specification()).select(variables=[])
    result = empty.read()
    assert result.table.num_rows == 0 and result.report.complete
    assert result.metadata.spec.series == ()
    assert samples.select(variables=[]).select().read().table.num_rows == 0
    assert (
        samples.select(variables=[]).select(variables=None).read().table.num_rows == 5
    )
    assert samples.select(variables="my:rainfall").read().table.num_rows == 5


def test_unknown_or_conflicting_scientific_selectors_raise() -> None:
    samples = sample_recipe()
    with pytest.raises(ValueError, match="unknown"):
        samples.select(variables="my:missing").read()
    with pytest.raises(ValueError, match="conflicting"):
        samples.select(
            variables=Variable("my:rainfall", measurement=Ratio(unit=RatioUnit.PERCENT))
        ).read()
    with pytest.raises(ValueError, match="unknown"):
        samples.select(stations="station-missing").read()


def test_valid_area_or_time_with_no_observations_is_typed_empty() -> None:
    samples = sample_recipe()
    result = samples.select(
        stations="station-a", area=Bounds(west=10, south=10, east=11, north=11)
    ).read()
    assert result.table.num_rows == 0 and result.report.complete and result.report.valid
    result = samples.select(time=Period.window(1900, 1910)).read()
    assert result.table.num_rows == 0 and result.report.complete and result.report.valid


def test_native_calendar_dates_and_time_selection_survive_persistence(
    tmp_path: Path,
) -> None:
    table = raw_samples()
    table["time"] = [
        "2000-02-30",
        "2001-02-30",
        "2002-02-30",
        "2003-02-30",
        "2004-02-30",
    ]
    recipe = Samples.from_table(
        table,
        variable=Variable("my:rainfall"),
        period=Period.window(2000, 2004, calendar=Calendar.DAYS_360),
    )
    saved = recipe.write(tmp_path / "native-calendar.parquet")
    assert saved.path is not None
    selected = (
        Samples.open(saved.path)
        .select(time=Period.window(2001, 2002, calendar=Calendar.DAYS_360))
        .read()
    )
    assert selected.table["time"].to_pylist() == ["2001-02-30", "2002-02-30"]
    with pytest.raises(ValueError, match="calendar"):
        recipe.select(
            time=Period.window(2001, 2002, calendar=Calendar.GREGORIAN)
        ).read()


def test_actual_observation_training_window_is_persisted_separately(
    tmp_path: Path,
) -> None:
    from crc_sdk._dataset_schema import ObservationTimeSelection

    declared = Period.window(2000, 2004, calendar=Calendar.DAYS_360)
    training = Period.window(2001, 2004, calendar=Calendar.DAYS_360)
    samples = Samples.from_table(
        raw_samples(), variable=Variable("my:rainfall"), period=declared
    )
    fitted = samples.select(time=training).fit(fit_specification())
    result = fitted.read()
    assert result.metadata.spec.series[0].period == declared
    selections = [
        item
        for item in result.metadata.provenance
        if isinstance(item, ObservationTimeSelection)
    ]
    assert selections == [ObservationTimeSelection(time=training)]
    row = table_to_rows(result.table)[0]
    assert row.fit is not None and row.fit.evidence_count == 4
    saved = fitted.write(tmp_path / "training-window.parquet")
    assert saved.path is not None
    reopened = Dataset.open(saved.path).read()
    assert reopened.metadata.provenance == result.metadata.provenance
    assert reopened.metadata.spec.series[0].period == declared


def test_same_computation_across_independent_recipes_shares_one_graph_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = sample_recipe()
    first = samples.fit(fit_specification())
    second = samples.fit(fit_specification())
    assert first is not second and first._node.key == second._node.key
    calls = 0
    original = fit_sample_table

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_recipes, "fit_sample_table", counted)
    with ExecutionScope() as scope:
        assert sum(batch.num_rows for batch in scope.iter_batches(first._node)) == 1
        assert sum(batch.num_rows for batch in scope.iter_batches(second._node)) == 1
    assert calls == 1


def test_recipe_is_distinct_from_a_stable_canonical_reader() -> None:
    assert not isinstance(quantile_recipe(), CanonicalReader)


def test_partial_stream_reports_and_rows_outlive_scope_without_full_claim() -> None:
    recipe = sample_recipe()
    with recipe.batches(runtime=Runtime(batch_rows=1)) as stream:
        first = next(iter(stream))
        snapshot = stream.report
        directory = stream._scope.directory
        assert directory.exists()
        assert first.num_rows == 1 and not snapshot.complete
        assert stream.metadata.spec.variables[0].id == "my:rainfall"
    assert not directory.exists()
    assert first["value"].to_pylist() == [1.0]
    assert not stream.report.complete and stream.report.valid is None
    assert not snapshot.complete


def test_saved_handles_pin_content_and_execution_replay_keeps_consumed_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "depth.parquet"
    saved = quantile_recipe().write(path)
    latest = Dataset.open(path)
    with latest.execute() as execution:
        assert execution.validate().valid
        original = table_to_rows(execution.read().table)[0]
        quantile_recipe(scale=10).write(path, overwrite=True)
        same = table_to_rows(execution.read().table)[0]
        assert original._document() == same._document()
    with pytest.raises(OSError, match="identity mismatch"):
        saved.read()
    changed = table_to_rows(Dataset.open(path).read().table)[0]
    assert isinstance(changed.curve, QuantileCurve)
    assert changed.curve.values.tolist() == [2.0, 8.0, 16.0]


def _write_cross_batch_duplicate(path: Path) -> None:
    table, metadata = normalize_samples(raw_samples(), variable=Variable("my:rainfall"))
    duplicated = pa.concat_tables([table, table])
    metadata = replace(
        metadata,
        report=ExecutionReport(
            True, True, counts=ExecutionCounts(requested=10, produced=10)
        ),
    )
    duplicated = duplicated.replace_schema_metadata(
        evidence_arrow_schema(metadata).metadata
    )
    pq.write_table(duplicated, path, row_group_size=5)


def test_sample_global_integrity_is_checked_across_batch_boundaries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-samples.parquet"
    _write_cross_batch_duplicate(path)
    recipe = Samples.open(path)
    report = recipe.validate(runtime=Runtime(batch_rows=5))
    assert report.valid is False and not report.complete
    assert any("duplicate" in diagnostic.message for diagnostic in report.diagnostics)
    with pytest.raises(ValidationError) as raised:
        recipe.read(runtime=Runtime(batch_rows=5))
    assert raised.value.report.valid is False


def test_failed_validation_preserves_destination_and_cleans_staging(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-samples.parquet"
    _write_cross_batch_duplicate(source)
    destination = tmp_path / "existing.parquet"
    destination.write_bytes(b"untouched")
    with pytest.raises(ValidationError):
        Samples.open(source).write(
            destination, overwrite=True, runtime=Runtime(batch_rows=5)
        )
    assert destination.read_bytes() == b"untouched"
    assert not list(tmp_path.glob(".existing.parquet.*"))


def test_malformed_persisted_data_is_a_report_but_missing_files_remain_io_errors(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.parquet"
    malformed.write_bytes(b"This file is not a Parquet dataset")
    report = Dataset.open(malformed).validate()
    assert report.valid is False
    assert report.diagnostics
    with pytest.raises(ValidationError) as raised:
        Dataset.open(malformed).read()
    assert raised.value.report.valid is False
    with pytest.raises(FileNotFoundError):
        Dataset.open(tmp_path / "missing.parquet").validate()


def test_unsupported_destination_and_no_overwrite_fail_before_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid destinations must fail before fitting")

    recipe = sample_recipe().fit(fit_specification())
    monkeypatch.setattr(_recipes, "fit_sample_table", unexpected_fit)
    destination = tmp_path / "existing.parquet"
    destination.write_bytes(b"untouched")
    with pytest.raises(FileExistsError):
        recipe.write(destination)
    with pytest.raises(ValueError, match="local"):
        recipe.write("s3://bucket/result.parquet")
    assert destination.read_bytes() == b"untouched"


def test_unrelated_operational_failures_keep_their_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("storage is unavailable")

    recipe = sample_recipe().fit(fit_specification())
    monkeypatch.setattr(_recipes, "fit_sample_table", broken)
    with pytest.raises(OSError, match="storage is unavailable"):
        recipe.validate()


def test_renaming_and_unit_conversion_change_exact_identities_and_actual_values() -> (
    None
):
    depth = quantile_recipe().rename(variables={"my:depth": "company:water"})
    result = depth.convert_units(
        variable="company:water", unit=LengthUnit.CENTIMETRE
    ).read()
    row = table_to_rows(result.table)[0]
    assert row.variable_id == "company:water"
    assert row.curve is not None and row.curve.quantile(0.9) == pytest.approx(80)
    assert result.metadata.spec.variables[0].measurement == DepthAboveGround(
        unit=LengthUnit.CENTIMETRE
    )
    assert len(result.metadata.provenance) == 2


def test_effective_rename_mapping_includes_unchanged_participating_identity() -> None:
    first, second = Variable("a"), Variable("b")
    recipe = Dataset.from_quantiles(
        {
            "support_id": ["a"] * 3 + ["b"] * 3,
            "longitude": [0.0] * 3 + [1.0] * 3,
            "latitude": [0.0] * 6,
            "probability": [0.5, 0.9, 0.99] * 2,
            "value": [1.0, 2.0, 3.0] * 2,
            "variable": ["a"] * 3 + ["b"] * 3,
        },
        spec=DataSpec([first, second], series=[SeriesSpec(first), SeriesSpec(second)]),
        columns=QuantileColumns(variable_id="variable"),
    )
    with pytest.raises(ValueError, match="injective|collid"):
        recipe.rename(variables={"a": "b"}).read()
    assert (
        recipe.select(variables="a").rename(variables={"a": "b"}).read().table.num_rows
        == 1
    )


def _attributed_samples(
    owner: str, *, expected_count: int = 5
) -> tuple[Samples, EvidenceMetadata]:
    data = raw_samples()
    table, metadata = normalize_samples(
        data,
        variable=Variable(
            "my:rainfall", measurement=Ratio(), label=f"Rainfall supplied by {owner}"
        ),
    )
    digest = (
        "sha256:"
        + hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    )
    source = SourceIdentity(
        provider=owner,
        product=f"{owner}:observations",
        resources=(
            ResourceIdentity(
                uri=f"https://{owner}.example/observations.json",
                sha256=digest,
                license=f"License supplied by {owner}",
                attribution=f"Observation attribution for {owner}",
            ),
        ),
    )
    report = replace(
        metadata.report,
        counts=ExecutionCounts(requested=expected_count, produced=expected_count),
        source_identities=(source,),
    )
    metadata = replace(
        metadata,
        source=source,
        report=report,
        provenance=(
            IdentityRename(
                dimension=IdentityKind.VARIABLE,
                mapping={f"{owner}:rainfall": "my:rainfall"},
            ),
        ),
    )
    table = table.replace_schema_metadata(evidence_arrow_schema(metadata).metadata)
    return Samples._collected(table, metadata), metadata


def test_shared_model_computation_retains_each_requested_source_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, first_metadata = _attributed_samples("source-a")
    second, second_metadata = _attributed_samples("source-b")
    assert first._node.key == second._node.key
    first_fit = first.fit(fit_specification())
    second_fit = second.fit(fit_specification())
    assert first_fit._node.key == second_fit._node.key
    calls = 0
    original = fit_sample_table

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_recipes, "fit_sample_table", counted)
    with ExecutionScope() as scope:
        first_batches = list(scope.iter_batches(first_fit._node))
        second_batches = list(scope.iter_batches(second_fit._node))
        assert sum(batch.num_rows for batch in first_batches) == 1
        assert sum(batch.num_rows for batch in second_batches) == 1
        assert calls == 1
        second_view = scope.metadata(second_fit._node)
        second_report = scope.report(second_fit._node)
        assert second_view.source == second_metadata.source
        assert second_view.spec.variables[0].label == "Rainfall supplied by source-b"
        assert second_view.provenance == second_metadata.provenance
        assert second_report.complete and second_report.valid
        assert second_report.source_identities == (second_metadata.source,)
        assert all(
            diagnostic.source in (None, "source-b:observations")
            for diagnostic in second_report.diagnostics
        )
        # Stable streams and their yielded Arrow headers describe the same view.
        second_table = pa.Table.from_batches(second_batches)
        assert metadata_from_table(second_table).source == second_metadata.source
        assert (
            metadata_from_table(second_table).provenance == second_metadata.provenance
        )
        first_view = scope.metadata(first_fit._node)
        assert first_view.source == first_metadata.source
        assert first_view.provenance == first_metadata.provenance


@pytest.mark.parametrize("invalid_first", [False, True])
def test_source_views_do_not_borrow_or_poison_each_others_validation(
    invalid_first: bool,
) -> None:
    valid, _ = _attributed_samples("source-a")
    invalid, _ = _attributed_samples("source-b", expected_count=6)
    with ExecutionScope() as scope:
        if not invalid_first:
            assert sum(batch.num_rows for batch in scope.iter_batches(valid._node)) == 5
            assert (
                scope.report(valid._node).complete and scope.report(valid._node).valid
            )
        assert sum(batch.num_rows for batch in scope.iter_batches(invalid._node)) == 5
        invalid_report = scope.report(invalid._node)
        assert invalid_report.valid is False
        assert any("count" in item.message for item in invalid_report.diagnostics)
        assert sum(batch.num_rows for batch in scope.iter_batches(valid._node)) == 5
        assert scope.report(valid._node).complete and scope.report(valid._node).valid
    with pytest.raises(ValidationError):
        invalid.read()
