"""Calibrated responses resolve explicitly bound source measurements and context."""

from __future__ import annotations

from dataclasses import dataclass

import crc_framework as cf
import pytest

from crc_sdk._assets import Assets, PointColumns
from crc_sdk._evaluation_responses import Response
from crc_sdk._graph import ExecutionScope

_DEPTH = cf.DepthAboveGround(unit=cf.LengthUnit.METRE)


def test_binding_only_asset_columns_resolve_without_publishing_them() -> None:
    assets = Assets.sql(
        "SELECT 'x' AS asset_id, 0.0 AS longitude, 0.0 AS latitude, "
        "'europe' AS region, 'residential' AS use",
        location=PointColumns(),
        columns=[],
    )
    reference = cf.Impact.builtin(cf.BuiltinImpact.INUNDATION)
    response = Response(
        name="damage",
        variable="user:depth",
        model=reference,
        context=cf.InundationContext(
            region=assets.column("region"),
            asset_type=assets.column("use"),
        ),
    )
    response.bind(_DEPTH)
    cache: dict[tuple[str, str], cf.Impact] = {}
    with ExecutionScope() as scope:
        data = assets._materialize(scope, required_columns=response.columns)
        actual, context = response.resolve(data.context.to_pylist()[0], cache)
        assert context is None
        assert response.resolve(data.context.to_pylist()[0], cache)[0] is actual
        assert data.table.column_names == ["asset_id"]
        assert sorted(data.context.column_names) == ["region", "use"]
    expected = reference.resolve(
        cf.InundationContext(
            region=cf.FloodRegion.EUROPE, asset_type=cf.AssetType.RESIDENTIAL
        )
    )
    assert actual.sha256 == expected.sha256
    assert actual.evaluate(1.0) == expected.evaluate(1.0)


def test_builder_owns_context_binding_names_and_reports_invalid_values() -> None:
    @dataclass
    class Binding:
        name: str

    binding = Binding("region")
    response = Response(
        name="damage",
        variable="user:depth",
        model=cf.Impact.builtin(cf.BuiltinImpact.INUNDATION),
        context=cf.InundationContext(
            region=binding, asset_type=cf.AssetType.RESIDENTIAL
        ),
    )
    binding.name = "different_column"
    assert response.columns == ("region",)
    with pytest.raises(cf.ImpactError, match="assets.column"):
        response.resolve({}, {})
    with pytest.raises(cf.ImpactError, match="missing_context_value"):
        response.resolve({"region": None}, {})
    with pytest.raises(cf.ImpactError, match="invalid_context_value"):
        response.resolve({"region": "Europe"}, {})
    assets = Assets({"asset_id": ["x"], "longitude": [0.0], "latitude": [0.0]})
    with ExecutionScope() as scope:
        with pytest.raises(ValueError, match="Missing bound asset columns"):
            assets._materialize(scope, required_columns=response.columns)


def test_custom_impact_checks_qualifiers_and_explicit_unit_conversion() -> None:
    calibration = cf.WaterSurfaceElevation(unit=cf.LengthUnit.METRE, datum="datum-a")
    response = Response(
        name="hours",
        variable="user:elevation",
        model=cf.Impact.linear(
            slope=2.0, input=calibration, output=cf.TimeDuration(unit=cf.TimeUnit.HOUR)
        ),
    )
    response.bind(calibration)
    with pytest.raises(cf.ImpactError, match="convert_units"):
        response.bind(
            cf.WaterSurfaceElevation(unit=cf.LengthUnit.CENTIMETRE, datum="datum-a")
        )
    with pytest.raises(cf.ImpactError, match="measurement_mismatch"):
        response.bind(
            cf.WaterSurfaceElevation(unit=cf.LengthUnit.METRE, datum="datum-b")
        )


def test_resolved_builtin_models_keep_published_binding_requirements() -> None:
    reference = cf.Impact.builtin(cf.BuiltinImpact.CONSECUTIVE_DRY_DAYS)
    context = cf.CropContext(country="USA", historic_mean=20.0)
    response = Response("damage", "user:dry-days", reference.resolve(context))
    with pytest.raises(cf.ImpactError, match="calibration_unestablished"):
        response.bind(reference.requirements.input)
