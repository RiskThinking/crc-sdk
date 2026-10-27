"""SDK expression facade delegates to the framework expression."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from crc_framework import ImpactExpression as FrameworkExpression
from crc_framework import RegistryImpactReference as FrameworkReference

from crc_sdk.impacts import (
    ImpactContext,
    ImpactEvaluationResult,
    ImpactExpression,
    RegistryImpactReference,
)

CALIBRATION = {
    "exposure_variable": "flood_depth",
    "exposure_unit": "m",
    "impact_metric": "damage_ratio",
    "impact_unit": "1",
}


def test_constructors_match_framework_canonical_json() -> None:
    pairs = [
        (
            ImpactExpression.linear(2.0, intercept=-1.0, calibration=CALIBRATION),
            FrameworkExpression.linear(2.0, intercept=-1.0, calibration=CALIBRATION),
        ),
        (
            ImpactExpression.sigmoid(
                midpoint=1.0, steepness=0.5, calibration=CALIBRATION
            ),
            FrameworkExpression.sigmoid(
                midpoint=1.0, steepness=0.5, calibration=CALIBRATION
            ),
        ),
        (
            ImpactExpression.tabulated(
                [(0.0, 0.0), (1.0, 1.0)], calibration=CALIBRATION
            ),
            FrameworkExpression.tabulated(
                [(0.0, 0.0), (1.0, 1.0)], calibration=CALIBRATION
            ),
        ),
        (
            ImpactExpression.binned(
                [0.0, 1.0, 2.0], [0.1, 0.4], calibration=CALIBRATION
            ),
            FrameworkExpression.binned(
                [0.0, 1.0, 2.0], [0.1, 0.4], calibration=CALIBRATION
            ),
        ),
    ]
    for sdk, framework in pairs:
        assert sdk.to_json() == framework.to_json()
        assert sdk.definition_hash == framework.definition_hash


def test_to_config_keeps_registry_snapshot_provenance() -> None:
    for factor, context in (
        ("spei", {"country": "USA", "building_type": "Commercial buildings"}),
        (
            "cooling_degree_days",
            {
                "country": "USA",
                "building_type": "Agricultural buildings",
                "historic_mean": 410.5,
            },
        ),
        (
            "maximum_consecutive_dry_days",
            {
                "country": "USA",
                "building_type": "Agricultural buildings",
                "historic_mean": 12.0,
            },
        ),
        ("cyclone", {"country": "USA"}),
    ):
        expression = ImpactExpression.for_factor(factor, context=context)
        config = expression.to_config()
        assert config.model.kind == expression.model_kind
        assert config.provenance.origin.kind == "registry"
        assert config.provenance.origin.lookups
        assert ImpactExpression.from_json(expression.to_json()).definition_hash == (
            expression.definition_hash
        )


def test_shapes_and_recorded_results() -> None:
    expression = ImpactExpression.linear(2.0, calibration=CALIBRATION)
    assert expression.evaluate(3.0) == 6.0
    empty = expression.evaluate(np.zeros((0, 2)))
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (0, 2)
    grid = expression.evaluate(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (2, 2)
    assert grid[1, 1] == 8.0
    recorded = expression.evaluate_with_record(np.array([1.0, 2.0]))
    assert isinstance(recorded, ImpactEvaluationResult)
    assert isinstance(recorded.value, np.ndarray)
    assert list(recorded.value) == [2.0, 4.0]
    assert recorded.record.to_dict()["operation"]["kind"] == "point"
    with pytest.raises(TypeError):
        expression.evaluate([1.0], context={"country": "USA"})  # type: ignore[call-arg]


def test_partial_context_forwards_only_supplied_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    original_expression = FrameworkExpression.for_factor
    original_reference = FrameworkReference.for_factor
    original_resolve = FrameworkReference.resolve

    def spy_expression(factor: Any, **kwargs: Any) -> Any:
        seen["expression_context"] = kwargs.get("context")
        seen["expression_defaults"] = kwargs.get("context_defaults")
        return original_expression(factor, **kwargs)

    def spy_reference(factor: Any, **kwargs: Any) -> Any:
        seen["reference_defaults"] = kwargs.get("context_defaults")
        return original_reference(factor, **kwargs)

    def spy_resolve(self: Any, context: Any = None) -> Any:
        seen["resolve_context"] = context
        return original_resolve(self, context)

    monkeypatch.setattr(
        "crc_sdk.impacts.expression.FrameworkExpression.for_factor",
        spy_expression,
    )
    monkeypatch.setattr(
        "crc_sdk.impacts.config.FrameworkReference.for_factor",
        spy_reference,
    )
    monkeypatch.setattr(
        "crc_sdk.impacts.config.FrameworkReference.resolve",
        spy_resolve,
    )

    context = ImpactContext(country="USA")
    defaults = ImpactContext(continent="Europe", country=None)
    ImpactExpression.for_factor(
        "cyclone",
        context=context,
        context_defaults=defaults,
    )
    reference = RegistryImpactReference.pinned("cyclone", context_defaults=defaults)
    reference.resolve(context)

    assert seen["expression_context"] == {"country": "USA"}
    assert seen["expression_defaults"] == {"continent": "Europe", "country": None}
    assert seen["reference_defaults"] == {"continent": "Europe", "country": None}
    assert seen["resolve_context"] == {"country": "USA"}


def test_context_defaults_match_the_framework_reference() -> None:
    defaults = ImpactContext(continent="Europe", building_type="Commercial buildings")
    sdk = ImpactExpression.for_factor("inundation", context_defaults=defaults)
    framework = FrameworkExpression.for_factor(
        "inundation",
        context_defaults={
            "continent": "Europe",
            "building_type": "Commercial buildings",
        },
    )
    assert sdk.definition_hash == framework.definition_hash
    reference = RegistryImpactReference.pinned("inundation", context_defaults=defaults)
    assert reference.resolve().definition_hash == sdk.definition_hash


def test_pickle_reconstructs_without_the_registry(tmp_path: Any) -> None:
    expression = ImpactExpression.tabulated(
        [(0.0, 0.0), (2.0, 1.0)],
        calibration=CALIBRATION,
        clamp={"minimum": 0.0, "maximum": 1.0},
    )
    payload = tmp_path / "impact.pkl"
    payload.write_bytes(pickle.dumps(expression))
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle, sys; from pathlib import Path; "
            "restored = pickle.loads(Path(sys.argv[1]).read_bytes()); "
            "sys.stdout.write(restored.to_json())",
            str(payload),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout == expression.to_json()
