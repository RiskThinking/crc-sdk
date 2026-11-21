"""Offline public workflow for paired insured loss and saved provenance."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from hashlib import file_digest
from pathlib import Path
from unittest.mock import patch

import crc_framework as cf
import numpy as np
import pyarrow.csv as arrow_csv  # type: ignore[import-untyped]
import pytest

from crc_sdk import (
    Assets,
    Dataset,
    DatasetCollection,
    LossArtifact,
    LossRun,
    Period,
    PolicyColumns,
    Probability,
    ProgressEvent,
    Runtime,
    Scenario,
    Stage,
    ValidationError,
    Variable,
)

_DAMAGE = "building-damage"
_PERIL = "river-flood"
_DEPTH = cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
_VARIABLE = Variable("test:water-depth", measurement=_DEPTH)
_RANKS = (0.0, 0.25, 0.75, 0.95, 1.0)
_SITES = {
    "asset_id": ["north", "south"],
    "longitude": [24.0, 24.2],
    "latitude": [60.0, 60.0],
    "value": [400_000.0, 650_000.0],
    "deductible_fraction": [0.02, 0.01],
    "limit": [150_000.0, 400_000.0],
}
_POLICY = PolicyColumns(
    total_insured_value="value",
    currency="EUR",
    valuation_basis="2026 replacement cost",
    deductible_fraction="deductible_fraction",
    limit="limit",
    share=0.8,
)
_SPEC = cf.AnnualLossSpec(trials=500, seed=20261027)


def _hazard(
    scenario: Scenario, period: Period, scale: float, *, ranks: Sequence[float]
) -> Dataset:
    return Dataset.from_quantiles(
        {
            "support_id": ["n"] * len(ranks) + ["s"] * len(ranks),
            "longitude": [24.0] * len(ranks) + [24.2] * len(ranks),
            "latitude": [60.0] * (2 * len(ranks)),
            "probability": list(ranks) * 2,
            "value": [rank * scale for rank in ranks]
            + [rank * scale * 0.8 for rank in ranks],
        },
        variable=_VARIABLE,
        scenario=scenario,
        period=period,
        probability=Probability.annual(tail=cf.Tail.UPPER),
    )


def _recipe(
    member: str = "baseline",
    *,
    stream: int = 0,
    assets: Assets | None = None,
    policy: PolicyColumns = _POLICY,
    components: bool = True,
    future_scale: float = 2.0,
    ranks: Sequence[float] = _RANKS,
    tail_completion: cf.TailCompletion | None = None,
) -> LossRun:
    hazards = DatasetCollection(
        {
            "baseline": _hazard(
                Scenario("historical"), Period.window(1981, 2010), 1.5, ranks=ranks
            ),
            "future": _hazard(
                Scenario("warmer"), Period.window(2041, 2070), future_scale, ranks=ranks
            ),
        }
    )
    evaluation = (
        hazards.evaluate(Assets(_SITES, columns=[]) if assets is None else assets)
        .select(members=member)
        .probabilities(ranks)
        .impact(
            _DAMAGE,
            cf.Impact.builtin(cf.BuiltinImpact.INUNDATION),
            variable=_VARIABLE,
            context=cf.InundationContext(
                region=cf.FloodRegion.EUROPE, asset_type=cf.AssetType.RESIDENTIAL
            ),
        )
    )
    return LossRun(
        evaluation,
        damage_response=_DAMAGE,
        peril=_PERIL,
        policy=policy,
        spec=_SPEC,
        tail_completion=tail_completion,
        secondary_stream=stream,
        include_components=components,
    )


def test_scope_reuses_simulation_and_writes_reopenable_paired_runs(
    tmp_path: Path,
) -> None:
    baseline = _recipe()
    destination = tmp_path / "baseline.parquet"
    with (
        patch.object(
            cf, "simulate_annual_losses", wraps=cf.simulate_annual_losses
        ) as sim,
        baseline.execute(runtime=Runtime(batch_rows=3)) as execution,
    ):
        assert execution.report.valid is None
        report = execution.validate()
        assert report.complete and report.valid is True
        result = execution.read()
        artifact = execution.write(destination)
        assert execution.read() is result
        assert sim.call_count == 1

    assert execution.report == report
    assert artifact.path == destination
    reopened = LossArtifact.open(destination).read()
    assert reopened.metadata.run_id == result.metadata.run_id
    assert reopened.report == report
    assert reopened.summary() == result.summary()
    assert reopened.summary().total_insured_value == 1_050_000.0
    assert reopened.components is not None and set(reopened.components) == {
        "north",
        "south",
    }
    assert reopened.trial_ids == range(_SPEC.trials)
    np.testing.assert_array_equal(reopened.totals, result.totals)

    future_recipe = _recipe("future", stream=1)
    future = future_recipe.write(tmp_path / "future.parquet").read()
    comparison = future.compare(reopened)
    assert future.metadata.run_id != reopened.metadata.run_id
    assert future.metadata.portfolio_sha256 == reopened.metadata.portfolio_sha256
    assert future.metadata.input_sha256 != reopened.metadata.input_sha256
    np.testing.assert_array_equal(
        comparison.trial_differences, future.totals - reopened.totals
    )
    assert future.summary().average_annual_loss > reopened.summary().average_annual_loss

    repeated = _recipe(components=False).read(runtime=Runtime(batch_rows=7))
    assert repeated.metadata.run_id == reopened.metadata.run_id
    assert repeated.components is None
    np.testing.assert_array_equal(repeated.totals, reopened.totals)
    changed_hazard = _recipe("future", stream=1, future_scale=2.1).read()
    assert changed_hazard.metadata.portfolio_sha256 == future.metadata.portfolio_sha256
    assert changed_hazard.metadata.input_sha256 != future.metadata.input_sha256
    assert changed_hazard.metadata.run_id != future.metadata.run_id
    with pytest.raises(ValueError, match="secondary"):
        _recipe("future").read().compare(reopened)


def test_bad_coverage_and_failed_publication_preserve_destination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "losses.parquet"
    path.write_bytes(b"existing output")
    assets = Assets({**_SITES, "longitude": [24.0, 60.0]})
    invalid = _recipe(assets=assets)

    with patch.object(cf, "simulate_annual_losses") as simulate:
        with invalid.execute() as execution:
            report = execution.validate()
            assert report.valid is False
            assert report.counts.unmatched_assets == 1
            assert any("south" in item.message for item in report.diagnostics)
            with pytest.raises(ValidationError) as error:
                execution.write(path, overwrite=True)
            assert error.value.report == report
        assert not simulate.called

    def fail_before_publication(event: ProgressEvent) -> None:
        if event.stage is Stage.WRITING:
            raise RuntimeError("publication stopped")

    with pytest.raises(RuntimeError, match="publication stopped"):
        _recipe().write(
            path, overwrite=True, runtime=Runtime(progress=fail_before_publication)
        )
    assert path.read_bytes() == b"existing output"
    assert list(tmp_path.iterdir()) == [path]

    with patch.object(cf, "simulate_annual_losses") as simulate:
        with pytest.raises(FileExistsError):
            _recipe().write(path)
        assert not simulate.called


def test_recipe_explain_is_lazy_and_tail_completion_is_explicit(tmp_path: Path) -> None:
    unavailable = tmp_path / "assets.csv"
    run = _recipe(assets=Assets(unavailable))
    assert any("Simulate 500" in op.description for op in run.explain().operations)
    assert not unavailable.exists()

    with pytest.raises(ValueError, match="(?i)tail|support|endpoint"):
        _recipe(ranks=_RANKS[1:-1])
    result = _recipe(
        ranks=_RANKS[1:-1], tail_completion=cf.TailCompletion(lower=0.0, upper=0.9)
    ).read()
    assert result.report.valid is True
    assert result.metadata.input_sha256 != _recipe().read().metadata.input_sha256


def test_loss_run_reads_one_file_snapshot_and_records_checksum(tmp_path: Path) -> None:
    path = tmp_path / "book.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(_SITES)
        writer.writerows(zip(*_SITES.values(), strict=True))
    with path.open("rb") as file:
        expected = "sha256:" + file_digest(file, "sha256").hexdigest()

    run = _recipe(assets=Assets(path, columns=[]))
    with patch.object(arrow_csv, "open_csv", wraps=arrow_csv.open_csv) as open_csv:
        with run.execute() as execution:
            assert execution.validate().valid
            result = execution.read()
        assert open_csv.call_count == 1
    assert any(
        resource.sha256 == expected
        for source in result.report.source_identities
        for resource in source.resources
    )
    assert (
        result.metadata.portfolio_sha256 == _recipe().read().metadata.portfolio_sha256
    )
    assert run.read().metadata.run_id == result.metadata.run_id

    missing = _recipe(
        assets=Assets(
            {name: _SITES[name] for name in ("asset_id", "longitude", "latitude")}
        )
    )
    assert any(
        item.code == "asset_policy_columns_required"
        for item in missing.explain().requirements
    )
