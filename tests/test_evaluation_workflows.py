"""Public workflow conformance across collection and direct evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk import (
    Assets,
    Bounds,
    Dataset,
    DatasetCollection,
    DataSpec,
    Distance,
    EvaluationArtifact,
    EvaluationMetadata,
    Period,
    Probability,
    QuantileColumns,
    QueryKind,
    ReleaseReference,
    Runtime,
    Samples,
    SeriesSpec,
    SourceIdentity,
    SpatialMatch,
    Stage,
    ValidationError,
    Variable,
)
from crc_sdk._dataset_schema import DatasetMetadata
from crc_sdk._graph import ExecutionScope, Node, NodeStream
from crc_sdk._schema import dataset_arrow_schema

DEPTH = cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
VARIABLE = Variable("company:water-depth", measurement=DEPTH)
WINDOW = Period.window(2041, 2060)


def data(
    probability: Probability | None = None,
    *,
    longitudes: tuple[float, ...] = (0.0,),
    coordinates: tuple[float, float] = (0.9, 0.99),
) -> Dataset:
    return Dataset.from_quantiles(
        {
            "support_id": [
                f"site-{i}" for i in range(len(longitudes)) for _ in range(2)
            ],
            "longitude": [value for value in longitudes for _ in range(2)],
            "latitude": [0.0] * (2 * len(longitudes)),
            "probability": list(coordinates) * len(longitudes),
            "value": [1.0, 2.0] * len(longitudes),
        },
        variable=VARIABLE,
        period=WINDOW,
        probability=probability,
    )


def annual() -> Dataset:
    return data(Probability.annual(tail=cf.Tail.UPPER))


def test_large_finite_return_period_inverts_the_direct_upper_tail(
    tmp_path: Path,
) -> None:
    depth = Dataset.from_quantiles(
        {
            "support_id": ["site"] * 4,
            "longitude": [0.0] * 4,
            "latitude": [0.0] * 4,
            "probability": [0.5, 0.75, 0.9, 0.99],
            "value": [1.0, 2.0, 3.0, 4.0],
        },
        variable=VARIABLE,
        probability=Probability.annual(tail=cf.Tail.UPPER),
    ).fit(cf.QuantileFit(family=cf.DistributionFamily.GUMBEL_RIGHT))
    periods = [1e12, 1e16, 1e18]
    law = cf.fit_quantiles(
        cf.QuantileCurve([0.5, 0.75, 0.9, 0.99], [1.0, 2.0, 3.0, 4.0]),
        cf.QuantileFit(family=cf.DistributionFamily.GUMBEL_RIGHT),
    ).distribution
    model = json.loads(law.to_json())
    expected = [
        model["location"] - model["scale"] * math.log(-math.log1p(-1 / period))
        for period in periods
    ]
    evaluation = depth.evaluate(sites(0)).return_periods(periods)
    assert evaluation.read().table["value"].to_pylist() == pytest.approx(
        expected, rel=1e-13
    )
    path = tmp_path / "tail.parquet"
    evaluation.write(path)
    assert EvaluationArtifact.open(path).read().table[
        "value"
    ].to_pylist() == pytest.approx(expected, rel=1e-13)


def test_upper_return_period_distinguishes_the_end_of_supplied_evidence() -> None:
    last = Dataset.from_quantiles(
        {
            "support_id": ["site", "site"],
            "longitude": [0.0, 0.0],
            "latitude": [0.0, 0.0],
            "probability": [0.5, 1.0],
            "value": [-1e300, 0.0],
        },
        variable=VARIABLE,
        probability=Probability.annual(tail=cf.Tail.UPPER),
    )
    result = last.evaluate(sites(0)).return_periods([1e18]).read()
    assert result.table["value"].to_pylist() == pytest.approx([-2e282], rel=1e-13)


def sites(*longitudes: float) -> Assets:
    return Assets(
        {
            "asset_id": list(range(len(longitudes))),
            "longitude": list(longitudes),
            "latitude": [0.0] * len(longitudes),
            "portfolio": ["A"] * len(longitudes),
        },
        columns=["portfolio"],
    )


def test_empty_assets_execute_and_validate_the_upstream_fit() -> None:
    samples = Samples.from_table(
        {
            "support_id": ["S"] * 4,
            "longitude": [0.0] * 4,
            "latitude": [0.0] * 4,
            "time": [2000, 2001, 2002, 2003],
            "value": [1.0] * 4,
        },
        variable=VARIABLE,
        probability=Probability.annual(tail=cf.Tail.UPPER),
    )
    fit = samples.fit(cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT))
    empty_assets = Assets(
        pa.table(
            {
                "asset_id": pa.array([], type=pa.string()),
                "longitude": pa.array([], type=pa.float64()),
                "latitude": pa.array([], type=pa.float64()),
            }
        )
    )
    failed = fit.evaluate(empty_assets).return_periods([10])
    with pytest.raises(ValidationError, match="samples fitting failed"):
        failed.read()
    report = failed.validate()
    assert report.valid is False
    assert report.diagnostics == fit.validate().diagnostics
    valid = annual().evaluate(empty_assets).return_periods([10]).read()
    assert valid.report.complete and valid.report.valid
    assert valid.table.num_rows == 0


def test_retained_arrow_temporal_values_survive_batched_eval_and_replay(
    tmp_path: Path,
) -> None:
    attributes = pa.table(
        {
            "asset_id": ["site"],
            "longitude": [0.0],
            "latitude": [0.0],
            "opened": pa.array([123], type=pa.timestamp("ns", tz="UTC")),
            "duration": pa.array([987], type=pa.duration("ns")),
            "snapshots": pa.array([[456]], type=pa.list_(pa.timestamp("ns"))),
        }
    )
    assets = Assets(attributes, columns=["opened", "duration", "snapshots"])
    recipe = annual().evaluate(assets).probabilities([0.9, 0.99])
    runtime = Runtime(batch_rows=1)
    table = recipe.read(runtime=runtime).table
    path = tmp_path / "nanoseconds.parquet"
    recipe.write(path, runtime=runtime)
    reopened = EvaluationArtifact.open(path).read(runtime=runtime).table
    assert table.equals(reopened, check_metadata=False)
    attrs = reopened["asset"].combine_chunks()
    assert attrs.field("opened").cast(pa.int64()).to_pylist() == [123, 123]
    assert attrs.field("duration").cast(pa.int64()).to_pylist() == [987, 987]
    assert attrs.field("snapshots").values.cast(pa.int64()).to_pylist() == [456, 456]


def test_direct_collection_and_saved_selection_agree(tmp_path: Path) -> None:
    source = annual()
    assets = sites(0, 0.0001)
    bundle = DatasetCollection({"baseline": source, "future": source}).select(
        members="baseline",
        periods=WINDOW,
    )
    saved = bundle.write(tmp_path / "selected")
    moved = tmp_path / "portable"
    (tmp_path / "selected").rename(moved)
    reopened = DatasetCollection.open(moved)
    assert saved.metadata is not None
    assert saved.metadata.selection.members == ("baseline",)
    direct = source.evaluate(assets).return_periods([100, 10, 2]).read()
    for collection in (bundle, reopened):
        result = collection.evaluate(assets).return_periods([100, 10, 2]).read()
        assert result.table.drop(["member_id", "resource_id"]).equals(
            direct.table.drop(["member_id", "resource_id"]),
            check_metadata=False,
        )
        assert set(result.table["member_id"].to_pylist()) == {"baseline"}
        assert result.report.counts == direct.report.counts
        assert result.report.diagnostics == direct.report.diagnostics
    assert direct.table["value"].to_pylist() == [2, 1, None, None, None, None]
    assert (
        direct.report.counts.matched_assets
        == direct.report.counts.unmatched_assets
        == 1
    )
    assert direct.metadata.query.kind is QueryKind.RETURN_PERIOD
    assert direct.table.schema.field("query_value").metadata[b"crc.unit"] == b"year"


def test_selection_moved_across_evaluate_preserves_rows_and_errors() -> None:
    area = Bounds(west=-0.1, south=-0.1, east=0.1, north=0.1)
    source = data(longitudes=(0, 1))
    assets = sites(0, 1)
    before = source.select(variables=VARIABLE, periods=WINDOW, area=area)
    after = source.evaluate(assets).select(
        variables=VARIABLE, periods=WINDOW, area=area
    )
    first = before.evaluate(assets).probabilities([0.9]).read()
    second = after.probabilities([0.9]).read()
    assert first.table.equals(second.table, check_metadata=True)
    assert list(second.table["value"].to_pylist()) == [1, None]
    for invalid in (
        source.select(variables="unknown").evaluate(assets),
        source.evaluate(assets).select(variables="unknown"),
    ):
        with pytest.raises(ValueError, match="unknown requested identity"):
            invalid.probabilities([0.9]).read()
    cleared = before.evaluate(assets).select(area=None).probabilities([0.9]).read()
    assert cleared.table["value"].to_pylist() == [1, 1]


@pytest.mark.parametrize("use_collection", [False, True])
def test_empty_period_selection_cannot_hide_unknown_variables(
    use_collection: bool,
) -> None:
    dataset = data()
    source = DatasetCollection({"baseline": dataset}) if use_collection else dataset
    assets = sites(0)
    for recipe in (
        source.select(variables="unknown", periods=[]).evaluate(assets),
        source.evaluate(assets).select(variables="unknown", periods=[]),
    ):
        evaluation = recipe.probabilities([0.9])
        with pytest.raises(ValueError, match="unknown requested identity"):
            evaluation.inspect()
        with pytest.raises(ValueError, match="unknown requested identity"):
            evaluation.read()

    before = source.select(variables=VARIABLE, periods=[]).evaluate(assets)
    after = source.evaluate(assets).select(variables=VARIABLE, periods=[])
    first = before.probabilities([0.9]).read().table
    second = after.probabilities([0.9]).read().table
    assert first.num_rows == 0
    assert first.equals(second, check_metadata=True)
    assert first.schema.field("asset_id").type == pa.int64()


def test_empty_member_selection_still_checks_explicit_variable_ids() -> None:
    source = DatasetCollection({"baseline": data()})
    for filtered in (
        source.select(members=[], variables=VARIABLE),
        source.evaluate(sites(0))
        .select(members=[], variables=VARIABLE)
        .probabilities([0.9]),
    ):
        assert filtered.read().table.num_rows == 0
    with pytest.raises(ValueError, match="unknown requested identity"):
        source.select(members=[], variables="unknown").read()


def test_inspection_describes_retained_asset_fields_without_acquiring_assets(
    tmp_path: Path,
) -> None:
    assets = Assets(
        pa.table(
            {
                "asset_id": pa.array([2**64 - 1], type=pa.uint64()),
                "longitude": [0.0],
                "latitude": [0.0],
                "portfolio": ["A"],
            }
        ),
        columns=["portfolio"],
    )
    evaluation = data().evaluate(assets).probabilities([0.9])
    declared = evaluation.inspect().schema
    actual = evaluation.read().table.schema
    assert declared is not None and declared.equals(actual)
    assert declared.field("asset_id").type == pa.uint64()
    assert declared.field("asset").type == pa.struct(
        [pa.field("portfolio", pa.string())]
    )

    deferred = (
        data()
        .evaluate(Assets(tmp_path / "deferred.csv"))
        .probabilities([0.9])
        .inspect()
    )
    assert deferred.schema is None
    assert [requirement.code for requirement in deferred.requirements] == [
        "asset_schema_unresolved"
    ]


def test_response_projection_avoids_conflicts_in_explicit_unrelated_variables() -> None:
    unrelated = Variable("unrelated", measurement=DEPTH)
    conflicting = Variable(
        unrelated.id, measurement=cf.DepthAboveGround(unit=cf.LengthUnit.CENTIMETRE)
    )

    def member(variable: Variable) -> Dataset:
        return Dataset.from_quantiles(
            {
                "support_id": ["site", "site"],
                "longitude": [0.0, 0.0],
                "latitude": [0.0, 0.0],
                "probability": [0.9, 0.99],
                "value": [1.0, 2.0],
            },
            variable=variable,
        )

    collection = DatasetCollection(
        {"depth": annual(), "left": member(unrelated), "right": member(conflicting)}
    )
    model = cf.Impact.linear(slope=2, input=DEPTH, output=cf.Ratio())
    recipe = (
        collection.select(variables=[VARIABLE, unrelated])
        .evaluate(sites(0))
        .probabilities([0.9])
        .impact("ratio", model, variable=VARIABLE)
        .select(responses="ratio")
    )
    result = recipe.read()
    assert result.table["value"].to_pylist() == [2.0]
    assert result.table["member_id"].to_pylist() == ["depth"]
    assert [ref.variable.id for ref in recipe.inspect().series] == [VARIABLE.id]
    with pytest.raises(ValueError, match="unknown requested identity 'typo'"):
        recipe.select(variables=[VARIABLE, "typo"]).read()

    selected = collection.select(
        series=[annual().inspect().series[0], member(unrelated).inspect().series[0]]
    )
    response_only = (
        selected.evaluate(sites(0))
        .probabilities([0.9])
        .impact("ratio", model, variable=VARIABLE)
        .select(responses="ratio")
    )
    assert response_only.read().table["value"].to_pylist() == [2.0]
    renamed = collection.rename(variables={VARIABLE.id: "renamed"})
    response_only = (
        renamed.evaluate(sites(0))
        .probabilities([0.9])
        .impact("ratio", model, variable="renamed")
        .select(responses="ratio")
    )
    assert response_only.read().table["value"].to_pylist() == [2.0]


def test_direct_response_and_variable_selection_scope_inspection_and_results() -> None:
    other = Variable("other", measurement=DEPTH)
    source = Dataset.from_quantiles(
        {
            "support_id": ["site"] * 4,
            "longitude": [0.0] * 4,
            "latitude": [0.0] * 4,
            "probability": [0.5, 0.9] * 2,
            "value": [1.0, 2.0, 3.0, 4.0],
            "variable": [VARIABLE.id, VARIABLE.id, other.id, other.id],
        },
        spec=DataSpec(
            [VARIABLE, other], series=[SeriesSpec(VARIABLE), SeriesSpec(other)]
        ),
        columns=QuantileColumns(variable_id="variable"),
    )
    model = cf.Impact.linear(slope=2, input=DEPTH, output=cf.Ratio())
    recipe = (
        source.evaluate(sites(0))
        .probabilities([0.5])
        .impact("primary", model, variable=VARIABLE)
        .impact("secondary", model, variable=other)
    )

    selected = recipe.select(variables=[VARIABLE, other], responses="primary")
    assert [reference.variable.id for reference in selected.inspect().series] == [
        VARIABLE.id
    ]
    assert selected.read().table["variable_id"].to_pylist() == [VARIABLE.id]
    assert recipe.select(variables=VARIABLE).read().table[
        "response_id"
    ].to_pylist() == [None, "primary"]


def test_named_responses_share_the_input_event_and_survive_reopening(
    tmp_path: Path,
) -> None:
    units = cf.TimeDuration(unit=cf.TimeUnit.DAY)
    falling = cf.Impact.linear(slope=-1, intercept=4, input=DEPTH, output=units)
    rising = cf.Impact.linear(slope=2, input=DEPTH, output=units)
    evaluation = (
        annual()
        .evaluate(sites(0))
        .return_periods([10, 100])
        .impact("recovery", falling, variable=VARIABLE)
        .impact("closure", rising, variable=VARIABLE)
    )
    inspection = evaluation.inspect()
    assert isinstance(inspection.metadata, EvaluationMetadata)
    assert [r.name for r in inspection.metadata.responses] == ["recovery", "closure"]
    result = evaluation.read()
    assert result.table["response_id"].to_pylist() == [
        None,
        None,
        "recovery",
        "recovery",
        "closure",
        "closure",
    ]
    assert result.table["value"].to_pylist() == [1, 2, 3, 2, 2, 4]
    assert result.table["asset"].to_pylist() == [{"portfolio": "A"}] * 6
    assert "return_period (years)" in repr(result)
    assert "recovery" in repr(result)
    selected = evaluation.select(responses="closure").read()
    assert selected.table["value"].to_pylist() == [2, 4]
    assert not selected.metadata.include_exposure
    with pytest.raises(ValueError, match="already exists"):
        evaluation.impact("closure", falling, variable=VARIABLE)
    changed = evaluation.impact("closure", falling, variable=VARIABLE, replace=True)
    assert changed.select(responses="closure").read().table["value"].to_pylist() == [
        3,
        2,
    ]
    bad = evaluation.impact("typo", falling, variable="company:unknown").validate()
    assert bad.valid is False
    assert bad.diagnostics[0].code == "response_variable_missing"
    stored = evaluation.write(tmp_path / "events.parquet")
    read = EvaluationArtifact.open(stored.path).read()
    assert read.table.equals(result.table, check_metadata=True)
    assert read.metadata == stored.metadata


@pytest.mark.parametrize(
    ("include_exposure", "value", "diagnostic"),
    [
        (True, 9999.0, "contradicts its model and exposure"),
        (False, float("inf"), "requires a finite value"),
    ],
)
def test_reopened_artifact_checks_named_response_values(
    tmp_path: Path, include_exposure: bool, value: float, diagnostic: str
) -> None:
    model = cf.Impact.linear(slope=2, input=DEPTH, output=cf.Ratio())
    evaluation = (
        annual()
        .evaluate(sites(0))
        .return_periods([10, 100])
        .impact("closure", model, variable=VARIABLE)
    )
    if not include_exposure:
        evaluation = evaluation.select(responses="closure")
    saved = evaluation.write(tmp_path / "events.parquet")
    runtime = Runtime(batch_rows=1)
    assert EvaluationArtifact.open(saved.path).validate(runtime=runtime).valid is True
    table = pq.read_table(saved.path)
    rows = table.to_pylist()
    assert rows[-1]["value"] == 4.0
    rows[-1]["value"] = value
    changed = tmp_path / "changed.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), changed)

    invalid = EvaluationArtifact.open(changed)
    report = invalid.validate(runtime=runtime)
    assert report.valid is False
    assert any(diagnostic in issue.message for issue in report.diagnostics)
    with pytest.raises(ValidationError, match=diagnostic):
        invalid.read(runtime=runtime)


def test_reopened_artifact_preserves_overflow_as_response_failure(
    tmp_path: Path,
) -> None:
    limit = cf.Impact.linear(slope=1e308, input=DEPTH, output=cf.Ratio())
    evaluation = (
        annual()
        .evaluate(sites(0))
        .return_periods([10, 100])
        .impact("closure", limit, variable=VARIABLE)
    )
    saved = evaluation.write(tmp_path / "events.parquet")
    result = EvaluationArtifact.open(saved.path).read()
    assert result.table["value"].to_pylist() == [1.0, 2.0, 1e308, None]
    assert result.table["failure"][-1].as_py()["code"] == "non_finite_result"
    assert result.report.valid is True


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("release", "source version disagrees"),
        ("diagnostics", "report omits diagnostics"),
    ],
)
def test_reopened_evaluation_checks_source_versions_and_diagnostics(
    tmp_path: Path, change: str, diagnostic: str
) -> None:
    source = SourceIdentity(
        provider="project",
        product="depth",
        release=ReleaseReference("first", "sha256:" + "a" * 64),
    )
    result = annual().read()
    base = tmp_path / "base.parquet"
    metadata = replace(result.metadata, source=source)
    pq.write_table(
        result.table.replace_schema_metadata(dataset_arrow_schema(metadata).metadata),
        base,
    )
    saved = (
        Dataset.open(base)
        .evaluate(sites(0))
        .probabilities([0.5, 0.9])
        .write(tmp_path / "events.parquet")
    )
    assert EvaluationArtifact.open(saved.path).validate().valid is True
    evaluation = pq.read_table(saved.path)
    headers = dict(evaluation.schema.metadata)
    document = json.loads(headers[b"crc.evaluation"])
    if change == "release":
        document["inputs"][0]["metadata"]["source"]["release"]["id"] = "second"
    else:
        document["report"]["diagnostics"] = []
    headers[b"crc.evaluation"] = json.dumps(document).encode()
    altered = tmp_path / "altered.parquet"
    pq.write_table(evaluation.replace_schema_metadata(headers), altered)
    artifact = EvaluationArtifact.open(altered)
    assert artifact.validate().valid is False
    with pytest.raises(ValidationError, match=diagnostic):
        artifact.read()


def test_bounded_nearest_and_lower_tail_queries() -> None:
    source = data(Probability.annual(tail=cf.Tail.LOWER), coordinates=(0.01, 0.1))
    result = (
        source.evaluate(
            sites(0.02, 0.1),
            spatial=SpatialMatch.nearest(max_distance=Distance.kilometres(5)),
        )
        .return_periods([100, 10])
        .read()
    )
    assert result.table["value"].to_pylist() == [1, 2, None, None]
    assert result.table["distance_metres"][0].as_py() == pytest.approx(2224, rel=0.002)
    assert result.table["failure"][2].as_py()["code"] == "outside_max_distance"


def test_write_and_reopen_preserves_covering_support_priority(tmp_path: Path) -> None:
    duplicate_supports = data(
        Probability.annual(tail=cf.Tail.UPPER), longitudes=(0.0, 0.0)
    )
    assets = sites(0.0)
    assert (
        duplicate_supports.evaluate(assets).return_periods([10]).validate().valid
        is False
    )
    policy = SpatialMatch.covering(preferred_supports=["site-1", "site-0"])
    saved = (
        duplicate_supports.evaluate(assets, spatial=policy)
        .return_periods([10])
        .write(tmp_path / "priority.parquet")
    )
    result = EvaluationArtifact.open(saved.path).read()
    assert result.metadata.spatial == policy
    assert result.table["support_id"].to_pylist() == ["site-1"]


@pytest.mark.parametrize("choose_variables", [False, True])
def test_only_named_response_skips_unrelated_member_fit(choose_variables: bool) -> None:
    other = Variable("company:other")
    constant = Samples.from_table(
        {
            "support_id": ["site"] * 5,
            "longitude": [0] * 5,
            "latitude": [0] * 5,
            "time": [2000, 2001, 2002, 2003, 2004],
            "value": [1] * 5,
        },
        variable=other,
    ).fit(cf.SampleFit(family=cf.DistributionFamily.GUMBEL_RIGHT))
    collection = DatasetCollection({"observed": annual(), "unrelated": constant})
    if choose_variables:
        collection = collection.select(variables=[VARIABLE, other])
    model = cf.Impact.linear(slope=2, input=DEPTH, output=cf.Ratio())
    result = (
        collection.evaluate(sites(0))
        .probabilities([0.9])
        .impact("ratio", model, variable=VARIABLE)
        .select(responses="ratio")
        .read()
    )
    assert result.table["value"].to_pylist() == [2]
    assert result.table["member_id"].to_pylist() == ["observed"]
    assert result.report.valid is True


def test_scope_reuses_upstream_and_streams_output(tmp_path: Path) -> None:
    parent = annual()._node
    opens = 0
    evaluated: list[int] = []

    class Counted(Node[DatasetMetadata]):
        key = parent.key

        def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
            nonlocal opens
            opens += 1
            return parent.open(scope)

    source = Dataset._create(Counted(), operations=())

    def progress(event: object) -> None:
        from crc_sdk import ProgressEvent

        assert isinstance(event, ProgressEvent)
        if event.stage is Stage.EVALUATION:
            evaluated.append(event.completed)

    evaluation = source.evaluate(sites(0, 0)).return_periods([100, 10])
    with evaluation.execute(
        runtime=Runtime(batch_rows=1, progress=progress)
    ) as execution:
        with execution.batches() as stream:
            assert next(iter(stream)).num_rows == 1
            assert evaluated == [1]
        assert not stream.report.complete
        assert execution.validate().complete
        expected = execution.read().table
        saved = execution.write(tmp_path / "one-pass.parquet")
    assert isinstance(saved, EvaluationArtifact)
    assert saved.read().table.equals(expected, check_metadata=True)
    assert opens == 1
    assert evaluated == [1, 2, 3, 4]


def test_empty_and_invalid_paths_have_stable_terminal_behavior(tmp_path: Path) -> None:
    missing_assets = Assets(tmp_path / "missing.csv")
    empty = annual().evaluate(missing_assets).return_periods([10]).select(responses=[])
    assert empty.read().table.num_rows == 0
    assert empty.write(tmp_path / "empty.parquet").read().report.valid
    observed = annual().evaluate(sites()).return_periods([10]).read()
    assert observed.table.num_rows == 0
    assert observed.report.valid
    invalid = (
        data(Probability.distribution()).evaluate(missing_assets).return_periods([10])
    )
    report = invalid.validate()
    assert report.valid is False
    assert report.diagnostics[0].code == "invalid_query_science"
    path = tmp_path / "keep.parquet"
    path.write_bytes(b"unchanged")
    with pytest.raises(ValidationError) as caught:
        invalid.write(path, overwrite=True)
    assert caught.value.report == report
    assert path.read_bytes() == b"unchanged"
