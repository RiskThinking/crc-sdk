"""Admit annual loss only from complete selected curves and a fixed point book."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import crc_framework as cf
import duckdb
import pytest
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from crc_sdk import (
    Assets,
    CanonicalReader,
    CapabilityError,
    Dataset,
    DatasetCollection,
    DatasetMetadata,
    Evaluation,
    GeometryColumn,
    LossRun,
    PointColumns,
    PolicyColumns,
    Probability,
    ReleaseReference,
    ResourceIdentity,
    Runtime,
    SourceIdentity,
    Stage,
    ValidationError,
    Variable,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pyarrow as pa  # type: ignore[import-untyped]

_RATIO = cf.Ratio()
_VARIABLE = Variable("custom:flood", measurement=_RATIO)
_ANNUAL = Probability.annual(tail=cf.Tail.UPPER)
_POLICY = PolicyColumns(
    total_insured_value=100.0, currency="USD", valuation_basis="replacement"
)
_SPEC = cf.AnnualLossSpec(trials=10, damage_cv=0.0)


def _hazard(
    values: Sequence[float] = (0.0, 0.5, 1.0),
    *,
    variable: Variable = _VARIABLE,
    probability: Probability = _ANNUAL,
    grid: Sequence[float] = (0.0, 0.5, 1.0),
) -> Dataset:
    return Dataset.from_quantiles(
        {
            "support_id": ["site"] * len(grid),
            "longitude": [0] * len(grid),
            "latitude": [0] * len(grid),
            "probability": grid,
            "value": values,
        },
        variable=variable,
        probability=probability,
    )


def _assets(*ids: str) -> Assets:
    return Assets(
        {
            "asset_id": ids,
            "longitude": [0] * len(ids),
            "latitude": [0] * len(ids),
        },
        columns=[],
    )


def _evaluation(
    source: Dataset | DatasetCollection,
    assets: Assets,
    *,
    impact: cf.Impact | None = None,
) -> Evaluation:
    model = cf.Impact.linear(slope=0.5, input=_RATIO, output=_RATIO)
    return (
        source.evaluate(assets)
        .probabilities([0.0, 0.5, 1.0])
        .impact("ground-up", model if impact is None else impact, variable=_VARIABLE)
    )


def _run(evaluation: Evaluation, *, policy: PolicyColumns = _POLICY) -> LossRun:
    return LossRun(
        evaluation,
        peril="river-flood",
        damage_response="ground-up",
        policy=policy,
        spec=_SPEC,
    )


def test_loss_resolves_exact_annual_member_and_requires_complete_grid() -> None:
    hazard = _hazard()
    collection = DatasetCollection({"baseline": hazard, "future": hazard})
    recipe = _evaluation(collection, _assets("one"))
    report = _run(recipe).validate()
    assert report.valid is False
    assert report.diagnostics[-1].code == "loss_series_selection"
    assert "baseline" in report.diagnostics[-1].message

    chosen = _run(recipe.select(members="baseline")).read()
    assert chosen.metadata.perils["river-flood"].series.member == "baseline"
    assert chosen.report.counts.requested == 3

    distribution = _hazard(probability=Probability.distribution())
    report = _run(_evaluation(distribution, _assets("one"))).validate()
    assert report.valid is False
    assert report.diagnostics[-1].code == "loss_probability_not_annual"

    partial = _hazard((0.3, 0.6), grid=(0.25, 0.75))
    report = _run(_evaluation(partial, _assets("one", "two"))).validate()
    assert report.valid is False
    assert report.counts.requested == report.counts.produced == 6
    assert report.counts.no_data == 4
    assert (
        sum(
            d.code == "query_outside_support" and d.stage is Stage.VALIDATION
            for d in report.diagnostics
        )
        == 2
    )


@pytest.mark.parametrize(
    ("model", "code"),
    [
        (
            cf.Impact.linear(slope=2, input=_RATIO, output=_RATIO),
            "loss_damage_ratio_bounds",
        ),
        (
            cf.Impact.tabulated(
                [(0, 0), (0.5, 0.5), (1, 0.2)], input=_RATIO, output=_RATIO
            ),
            "loss_damage_curve",
        ),
    ],
)
def test_loss_rejects_invalid_damage_evidence(model: cf.Impact, code: str) -> None:
    report = _run(_evaluation(_hazard(), _assets("one"), impact=model)).validate()
    assert report.valid is False
    assert report.diagnostics[-1].code == code
    assert "'one'" in report.diagnostics[-1].message


def test_loss_rejects_nonpoint_wkb_and_nonfraction_output() -> None:
    points = Polygon([(0, 0), (1, 1), (1, 0)])
    areas = Assets(
        {"asset_id": ["area"], "shape": [points.wkb]},
        location=GeometryColumn(name="shape"),
    )
    report = _run(_evaluation(_hazard(), areas)).validate()
    assert report.valid is False
    assert report.diagnostics[-1].code == "loss_location_not_point"

    percent = cf.Impact.linear(
        slope=0.5, input=_RATIO, output=cf.Ratio(unit=cf.RatioUnit.PERCENT)
    )
    with pytest.raises(CapabilityError, match="RatioUnit.FRACTION"):
        _run(_evaluation(_hazard(), _assets("one"), impact=percent))


def test_loss_reports_all_failed_model_contexts_and_unmatched_asset() -> None:
    depth = Variable(
        "custom:depth", measurement=cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
    )
    assets = Assets(
        {
            "asset_id": ["context-a", "context-b", "unmatched"],
            "longitude": [0, 0, 1],
            "latitude": [0, 0, 1],
            "region": [None, None, None],
        },
        columns=[],
    )
    evaluation = (
        _hazard(variable=depth)
        .evaluate(assets)
        .probabilities([0.0, 0.5, 1.0])
        .impact(
            "ground-up",
            cf.Impact.builtin(cf.BuiltinImpact.INUNDATION),
            variable=depth,
            context=cf.InundationContext(
                region=assets.column("region"), asset_type=cf.AssetType.RESIDENTIAL
            ),
        )
    )
    with pytest.raises(ValidationError) as failure:
        _run(evaluation).read()
    report = failure.value.report
    assert (
        report.counts.requested == report.counts.produced == report.counts.no_data == 9
    )
    assert report.counts.unmatched_assets == 1
    for asset_id in ("context-a", "context-b", "unmatched"):
        assert any(
            d.stage is Stage.VALIDATION and asset_id in d.message
            for d in report.diagnostics
        )


def test_policy_columns_resolve_sql_once_and_report_nonfinite_terms() -> None:
    with duckdb.connect() as connection:
        calls = 0

        def observe(value: float) -> float:
            nonlocal calls
            calls += 1
            return value

        connection.create_function("observe_tiv", observe, side_effects=True)
        assets = Assets.sql(
            """
            SELECT i AS asset_id, 0 AS longitude, 0 AS latitude,
                   observe_tiv(100.0 * i) AS tiv, 0.1 AS deductible_fraction
            FROM range(1, 3) AS rows(i)
            """,
            location=PointColumns(),
            columns=[],
        )
        policy = PolicyColumns(
            total_insured_value="tiv",
            currency="EUR",
            valuation_basis="2026 replacement",
            deductible_fraction="deductible_fraction",
        )
        run = _run(_evaluation(_hazard(), assets), policy=policy)
        with run.execute(runtime=Runtime(connection=connection)) as execution:
            assert execution.validate().valid
            result = execution.read()
        assert calls == 2
    assert result.metadata.total_insured_value == 300
    assert result.report.source_identities[0].product == "sql_assets"
    assert result.report.source_identities[0].resources[0].sha256.startswith("sha256:")

    bad = Assets(
        {
            "asset_id": ["bad-value", "bad-fraction"],
            "longitude": [0, 0],
            "latitude": [0, 0],
            "tiv": [float("inf"), 100.0],
            "deductible_fraction": [0.1, 1.25],
        },
        columns=[],
    )
    report = _run(_evaluation(_hazard(), bad), policy=policy).validate()
    assert report.valid is False
    assert any("Invalid terms: 2" in d.message for d in report.diagnostics)
    assert sum(d.code == "loss_policy_terms" for d in report.diagnostics) == 2


def test_loss_fingerprints_are_ordered_and_include_hazard_under_zero_damage() -> None:
    model = cf.Impact.linear(slope=0, input=_RATIO, output=_RATIO)
    first = _run(_evaluation(_hazard(), _assets("one", "two"), impact=model)).read()
    reordered = _run(_evaluation(_hazard(), _assets("two", "one"), impact=model)).read()
    assert first.metadata.run_id == reordered.metadata.run_id
    assert first.metadata.portfolio_sha256 == reordered.metadata.portfolio_sha256
    assert not first.totals.any()

    changed = _run(
        _evaluation(_hazard((0.0, 0.52, 0.9)), _assets("one", "two"), impact=model)
    ).read()
    assert changed.metadata.portfolio_sha256 == first.metadata.portfolio_sha256
    changed_peril = next(iter(changed.metadata.perils.values()))
    first_peril = next(iter(first.metadata.perils.values()))
    assert changed_peril.vulnerability_sha256 == first_peril.vulnerability_sha256
    assert changed_peril.input_sha256 != first_peril.input_sha256
    assert changed.metadata.run_id != first.metadata.run_id
    assert not changed.totals.any()


def test_loss_preserves_source_licence_and_repeats_past_retrieval_timestamps() -> None:
    hazard = _hazard().read()
    checksum = "sha256:" + "f" * 64

    class Reader(CanonicalReader):
        def __init__(self, day: int) -> None:
            timestamp = datetime(2026, 11, day, tzinfo=UTC)
            source = SourceIdentity(
                "example",
                "synthetic-flood",
                ReleaseReference("r1", checksum, timestamp),
                (
                    ResourceIdentity(
                        "example:hazard",
                        checksum,
                        license="CC-BY-4.0",
                        attribution="Example author",
                        retrieved_at=timestamp,
                    ),
                ),
            )
            self.metadata = replace(hazard.metadata, source=source)

        def inspect(self) -> DatasetMetadata:
            return self.metadata

        @contextmanager
        def batches(
            self, *, batch_rows: int = 65536
        ) -> Iterator[Iterator[pa.RecordBatch]]:
            yield iter(hazard.table.to_batches(max_chunksize=batch_rows))

    results = [
        _run(
            _evaluation(DatasetCollection({"historical": Reader(day)}), _assets("one"))
        ).read()
        for day in (1, 2)
    ]
    first, second = results
    assert first.metadata.run_id == second.metadata.run_id
    for day, result in enumerate(results, 1):
        (source,) = result.report.source_identities
        (resource,) = source.resources
        assert resource.license == "CC-BY-4.0"
        assert resource.attribution == "Example author"
        assert resource.sha256 == checksum
        assert resource.retrieved_at == datetime(2026, 11, day, tzinfo=UTC)
