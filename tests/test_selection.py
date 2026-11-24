from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import crc_framework as cf
import pytest
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely.geometry import Point, box  # type: ignore[import-untyped]

from crc_sdk._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionReport,
    IdentityKind,
    IdentityRename,
    ReleaseReference,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
)
from crc_sdk._declarations import (
    Calendar,
    DataSpec,
    Period,
    Probability,
    Scenario,
    SeriesSpec,
    Variable,
    scientific_identity,
)
from crc_sdk._inspection import (
    Bounds,
    Capability,
    IndicatorReference,
    Inspection,
    Operation,
    Requirement,
    SeriesReference,
)
from crc_sdk._schema.dataset import DatasetRow, validate_rows
from crc_sdk._selection import (
    SelectionState,
    Unit,
    conversion_metadata,
    conversion_row,
    conversion_spec,
    rename_metadata,
    rename_row,
    rename_spec,
    selected_metadata,
    support_intersects,
)


def definitions() -> DataSpec:
    a = Variable(
        "company:a", measurement=cf.DepthAboveGround(unit=cf.LengthUnit.CENTIMETRE)
    )
    b = Variable(
        "company:b", measurement=cf.Temperature(unit=cf.TemperatureUnit.CELSIUS)
    )
    scenario = Scenario("company:scenario", definition={"rainfall_multiplier": 1.2})
    period = Period.window(2030, 2050, calendar=Calendar.GREGORIAN)
    return DataSpec(
        [a, b, Variable("unused")],
        series=[
            SeriesSpec(
                a,
                scenario=scenario,
                period=period,
                probability=Probability.annual(tail=cf.Tail.UPPER),
            ),
            SeriesSpec(b),
        ],
    )


def metadata(spec: DataSpec, rows: int = 1) -> DatasetMetadata:
    return DatasetMetadata(
        spec=spec,
        spatial=SpatialContract(supports=(SupportKind.POINT,), index_resolution=0),
        report=ExecutionReport(
            complete=True,
            valid=True,
            counts=ExecutionCounts(requested=rows, produced=rows),
        ),
    )


def row(spec: DataSpec, *, index: int = 0) -> DatasetRow:
    series = spec.series[index]
    return DatasetRow(
        variable_id=series.variable.id,
        scenario_id=None if series.scenario is None else series.scenario.id,
        period_id=None if series.period is None else series.period.id,
        support_id="station 1",
        cell_id=h3.latlng_to_cell(0, 0, 0),
        geometry=Point(0, 0).wkb,
        curve=cf.QuantileCurve([0.1, 0.9], [100, 300]),
    )


def test_selection_omitted_clear_empty_and_single_string_have_distinct_meaning() -> (
    None
):
    spec = definitions()
    state = SelectionState().updated(variables="company:a")
    assert len(state.resolve(spec).spec.series) == 1
    assert state.updated().variables == state.variables
    assert len(state.updated(variables=None).resolve(spec).spec.series) == 2
    empty = state.updated(variables=[]).resolve(spec)
    assert empty.empty
    assert empty.spec.variables == ()
    assert empty.spec.series == ()
    assert empty.matches(row(spec)) is False


def test_selection_retains_valid_combinations_and_prunes_unused_descriptors() -> None:
    spec = definitions()
    selected = SelectionState().updated(scenarios=spec.scenarios[0]).resolve(spec)
    assert [v.id for v in selected.spec.variables] == ["company:a"]
    assert selected.spec.periods == spec.periods
    assert selected.spec.series == (spec.series[0],)
    assert selected.matches(row(spec))
    assert not selected.matches(row(spec, index=1))
    assert SelectionState().updated(variables="unused").resolve(spec).empty


@pytest.mark.parametrize(
    "arguments",
    [
        {"variables": "missing"},
        {"scenarios": "missing"},
        {"periods": Period.window(2090, 2100)},
        {"series": SeriesReference(Variable("missing"))},
    ],
)
def test_unknown_selection_fails_before_rows_are_needed(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        SelectionState().updated(**arguments).resolve(definitions())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        {"variables": [None]},
        {"variables": 123},
        {"periods": "2030"},
        {"area": (0, 0, 1, 1)},
        {"stations": [42]},
        {"time": Period.timeless()},
    ],
)
def test_invalid_selector_types_are_rejected(arguments: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        SelectionState().updated(**arguments)  # type: ignore[arg-type]


def test_scientific_identity_is_checked_but_labels_do_not_constrain_selection() -> None:
    spec = definitions()
    renamed_label = Variable(
        "company:a", measurement=spec.variables[0].measurement, label="A label"
    )
    assert SelectionState().updated(variables=renamed_label).resolve(spec).keys
    incompatible = Variable(
        "company:a", measurement=cf.DepthAboveGround(unit=cf.LengthUnit.METRE)
    )
    with pytest.raises(ValueError, match="conflicting scientific"):
        SelectionState().updated(variables=incompatible).resolve(spec)


def test_inspected_series_pins_science_and_release_without_cartesian_product() -> None:
    spec = definitions()
    release = ReleaseReference("release:1", "sha256:" + "0" * 64)
    source = SourceIdentity("provider", "product", release=release)
    item = spec.series[0]
    reference = SeriesReference(item.variable, item.scenario, item.period, release)
    selected = SelectionState().updated(series=reference).resolve(spec, source)
    assert selected.spec.series == (item,)
    assert reference.measurement is item.variable.measurement
    changed = SourceIdentity(
        "provider",
        "product",
        release=ReleaseReference("release:1", "sha256:" + "1" * 64),
    )
    with pytest.raises(ValueError, match="release differs"):
        SelectionState().updated(series=reference).resolve(spec, changed)
    with pytest.raises(ValueError, match="unknown series"):
        SelectionState().updated(
            series=SeriesReference(spec.variables[1], item.scenario, item.period)
        ).resolve(spec)


def test_equivalent_selector_orders_have_one_computation_document() -> None:
    first = SelectionState().updated(variables=["a", "b"], stations=["x", "y"])
    second = SelectionState().updated(variables=["b", "a"], stations=["y", "x"])
    assert first.document() == second.document()
    assert (
        SelectionState().document() != SelectionState().updated(variables=[]).document()
    )


def test_observation_time_filter_is_separate_from_declared_period_selection() -> None:
    spec = definitions()
    selected = SelectionState().updated(time=Period.window(2040, 2041)).resolve(spec)
    item = row(spec)._document()
    item["geometry"] = row(spec).geometry
    item["time"] = (
        "2040-02-30"  # Source-native calendar timestamps need not be Gregorian.
    )
    assert selected.matches_record(item)
    item["time"] = 2030
    assert not selected.matches_record(item)
    with pytest.raises(ValueError, match="calendar contradicts"):
        SelectionState().updated(
            time=Period.window(2040, 2041, calendar=Calendar.DAYS_360)
        ).resolve(spec)


def test_station_empty_selection_is_known_empty_and_single_id_is_exact() -> None:
    spec = definitions()
    assert SelectionState().updated(stations=[]).resolve(spec).empty
    assert (
        SelectionState().updated(stations="station 1").resolve(spec).matches(row(spec))
    )
    assert (
        not SelectionState()
        .updated(stations="Station 1")
        .resolve(spec)
        .matches(row(spec))
    )


@pytest.mark.parametrize("coordinates", [(179, 0), (-179, 0), (180, 0), (-180, 0)])
def test_antimeridian_bounds_cover_both_sides(coordinates: tuple[float, float]) -> None:
    assert support_intersects(
        Bounds(west=170, south=-10, east=-170, north=10),
        geometry=Point(*coordinates).wkb,
        cell_id=None,
    )


def test_source_geometry_cannot_expand_to_its_delivery_cell() -> None:
    cell = h3.latlng_to_cell(0, 0, 0)
    area = Bounds(west=0.1, south=0.1, east=0.2, north=0.2)
    assert support_intersects(area, geometry=None, cell_id=cell)
    assert not support_intersects(area, geometry=Point(0, 0).wkb, cell_id=cell)
    assert not support_intersects(
        area, geometry=box(-0.01, -0.01, 0.01, 0.01).wkb, cell_id=cell
    )


def test_cell_support_crossing_antimeridian_keeps_its_real_small_footprint() -> None:
    cell = h3.latlng_to_cell(10, 179.999, 3)
    assert support_intersects(
        Bounds(west=179.99, south=9, east=-179.99, north=11),
        geometry=None,
        cell_id=cell,
    )
    assert not support_intersects(
        Bounds(west=-1, south=9, east=1, north=11), geometry=None, cell_id=cell
    )


@pytest.mark.parametrize("pole", [-90, 90])
@pytest.mark.parametrize("longitude", [-170, -90, 0, 90, 170])
def test_polar_cell_support_retains_its_cap(pole: int, longitude: int) -> None:
    cell = h3.latlng_to_cell(pole, 0, 0)
    latitude = 89.999 if pole > 0 else -89.999
    assert h3.latlng_to_cell(latitude, longitude, 0) == cell
    assert support_intersects(
        Bounds(
            west=longitude - 0.1,
            south=max(-90, latitude - 0.0001),
            east=longitude + 0.1,
            north=min(90, latitude + 0.0001),
        ),
        geometry=None,
        cell_id=cell,
    )
    assert not support_intersects(
        Bounds(west=-1, south=-1, east=1, north=1), geometry=None, cell_id=cell
    )


@pytest.mark.parametrize(
    "bounds",
    [
        {"west": -181, "south": 0, "east": 1, "north": 2},
        {"west": 0, "south": 3, "east": 1, "north": 2},
        {"west": float("nan"), "south": 0, "east": 1, "north": 2},
        {"west": True, "south": 0, "east": 1, "north": 2},
    ],
)
def test_bounds_are_validated_without_io(bounds: dict[str, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        Bounds(**bounds)


def test_rename_rejects_collisions_with_unchanged_identities() -> None:
    spec = definitions()
    with pytest.raises(ValueError, match="collides"):
        rename_spec(spec, variables={"company:a": "company:b"})
    with pytest.raises(ValueError, match="unknown"):
        rename_spec(spec, scenarios={"absent": "new"})
    swapped, steps = rename_spec(
        spec, variables={"company:a": "company:b", "company:b": "company:a"}
    )
    assert swapped.variables[0].measurement == spec.variables[0].measurement
    assert steps[0].dimension is IdentityKind.VARIABLE


def test_rename_records_identity_only_and_keeps_numerical_data() -> None:
    spec = definitions()
    renamed = rename_metadata(
        metadata(spec),
        variables={"company:a": "custom:depth"},
        scenarios={"company:scenario": "custom:scenario"},
    )
    item = row(spec)
    result = rename_row(
        item,
        variables={"company:a": "custom:depth"},
        scenarios={"company:scenario": "custom:scenario"},
    )
    assert result.curve is item.curve
    assert result.variable_id == "custom:depth"
    assert renamed.provenance == (
        IdentityRename(
            dimension=IdentityKind.VARIABLE, mapping={"company:a": "custom:depth"}
        ),
        IdentityRename(
            dimension=IdentityKind.SCENARIO,
            mapping={"company:scenario": "custom:scenario"},
        ),
    )
    validate_rows([result], renamed)


def test_unit_conversion_records_exact_evidence_values_without_fitting() -> None:
    spec = definitions()
    changed, step = conversion_metadata(
        metadata(spec), variable="company:a", unit=cf.LengthUnit.METRE
    )
    assert step is not None
    original_measurement = spec.variables[0].measurement
    new_measurement = changed.spec.variables[0].measurement
    assert original_measurement is not None and new_measurement is not None
    result = conversion_row(
        row(spec), step, input=original_measurement, output=new_measurement
    )
    assert result.curve is not None
    assert result.curve.quantile(0.1) == 1
    assert result.curve.quantile(0.9) == 3
    assert result.fit is None
    assert changed.provenance == (step,)
    assert scientific_identity(changed.spec.variables[0]) != scientific_identity(
        spec.variables[0]
    )
    validate_rows([result], changed)


def test_converted_fitted_law_retains_original_fit_and_parametric_source() -> None:
    original = Variable(
        "x", measurement=cf.DepthAboveGround(unit=cf.LengthUnit.CENTIMETRE)
    )
    spec = DataSpec([original], series=[SeriesSpec(original)])
    fitted = cf.fit_samples(
        [100, 110, 120, 140, 180, 210],
        cf.SampleFit(
            family=cf.DistributionFamily.GUMBEL_RIGHT,
            constraints=cf.FitConstraints(probability=0.5, minimum_value=80),
        ),
    )
    item = DatasetRow(
        variable_id="x",
        support_id="station 1",
        cell_id=h3.latlng_to_cell(0, 0, 0),
        geometry=Point(0, 0).wkb,
        curve=fitted.distribution,
        fit=fitted.record,
    )
    changed, step = conversion_metadata(
        metadata(spec), variable="x", unit=cf.LengthUnit.METRE
    )
    assert step is not None
    assert original.measurement is not None
    assert changed.spec.variables[0].measurement is not None
    result = conversion_row(
        item,
        step,
        input=original.measurement,
        output=changed.spec.variables[0].measurement,
    )
    assert result.fit is fitted.record
    assert result.curve is not None and item.curve is not None
    assert result.curve.quantile(0.9) == pytest.approx(item.curve.quantile(0.9) / 100)
    validate_rows([result], changed)
    with pytest.raises(ValueError, match="representation contradicts"):
        validate_rows([result], replace(changed, provenance=()))


@pytest.mark.parametrize(
    "measurement,unit,expected",
    [
        (
            cf.Temperature(unit=cf.TemperatureUnit.CELSIUS),
            cf.TemperatureUnit.FAHRENHEIT,
            32,
        ),
        (cf.Ratio(unit=cf.RatioUnit.FRACTION), cf.RatioUnit.PERCENT, 0),
        (cf.TimeDuration(unit=cf.TimeUnit.HOUR), cf.TimeUnit.SECOND, 0),
    ],
)
def test_closed_unit_conversions_have_explicit_linear_meaning(
    measurement: cf.Measurement, unit: Unit, expected: float
) -> None:
    variable = Variable("x", measurement=measurement)
    spec = DataSpec([variable], series=[SeriesSpec(variable)])
    _, step = conversion_spec(spec, variable=variable, unit=unit)
    assert step is not None
    assert step.offset == pytest.approx(expected)


def test_calendar_duration_does_not_guess_fixed_day_length() -> None:
    variable = Variable("x", measurement=cf.TimeDuration(unit=cf.TimeUnit.MONTH))
    spec = DataSpec([variable], series=[SeriesSpec(variable)])
    with pytest.raises(ValueError, match="cannot convert"):
        conversion_spec(spec, variable=variable, unit=cf.TimeUnit.DAY)
    _, step = conversion_spec(spec, variable=variable, unit=cf.TimeUnit.YEAR)
    assert step is not None
    assert step.scale == pytest.approx(1 / 12)
    unchanged, step = conversion_spec(spec, variable=variable, unit=cf.TimeUnit.MONTH)
    assert unchanged is spec and step is None


def test_metadata_selection_prunes_unused_transformation_lineage() -> None:
    spec = definitions()
    converted, _ = conversion_metadata(
        metadata(spec), variable="company:a", unit=cf.LengthUnit.METRE
    )
    renamed = rename_metadata(
        converted, variables={"company:a": "new:a", "company:b": "new:b"}
    )
    selected = SelectionState().updated(variables="new:a").resolve(renamed.spec)
    pruned = selected_metadata(renamed, selected)
    assert pruned.spec.variables[0].id == "new:a"
    assert isinstance(pruned.provenance[-1], IdentityRename)
    assert pruned.provenance[-1].mapping == {"company:a": "new:a"}
    other = selected_metadata(
        renamed, SelectionState().updated(variables="new:b").resolve(renamed.spec)
    )
    assert len(other.provenance) == 1
    assert isinstance(other.provenance[0], IdentityRename)
    assert other.provenance[0].mapping == {"company:b": "new:b"}
    empty = selected_metadata(
        renamed, SelectionState().updated(variables=[]).resolve(renamed.spec)
    )
    assert empty.provenance == ()


def test_inspection_records_are_immutable_and_validate_typed_inventories() -> None:
    requirement = Requirement("select_indicator", "Choose an indicator", "indicator")
    operation = Operation(Stage.SELECTION, "Select exact scientific identities")
    indicator = IndicatorReference(
        "company:depth",
        "Water depth",
        measurement=cf.DepthAboveGround(unit=cf.LengthUnit.METRE),
    )
    inspection = Inspection(
        indicators=(indicator,),
        capabilities=(Capability.READ,),
        requirements=(requirement,),
    )
    assert "company:depth" in repr(inspection)
    with pytest.raises(FrozenInstanceError):
        operation.description = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable typed tuple"):
        Inspection(capabilities=[Capability.READ])  # type: ignore[arg-type]


def test_record_mask_matches_the_per_row_selection() -> None:
    import pyarrow as pa  # type: ignore[import-untyped]

    from crc_sdk._selection import ResolvedSelection

    inside = h3.latlng_to_cell(43.65, -79.38, 5)
    outside = h3.latlng_to_cell(10.0, 10.0, 5)
    rows = [
        {
            "variable_id": v,
            "scenario_id": s,
            "period_id": p,
            "support_id": support,
            "cell_id": cell,
            "geometry": geometry,
        }
        for v in ("crc:a", "crc:b")
        for s in ("ssp585", None)
        for p in ("period:x", None)
        for support, cell, geometry in (
            ("h3:in", inside, None),
            ("h3:out", outside, None),
            ("point:in", inside, Point(-79.38, 43.65).wkb),
            ("point:out", outside, Point(10.0, 10.0).wkb),
        )
    ]
    batch = pa.RecordBatch.from_pylist(rows)
    area = Bounds(west=-79.5, south=43.5, east=-79.2, north=43.8)
    for keys, stations, bounds in (
        ({("crc:a", "ssp585", "period:x")}, None, None),
        ({("crc:a", None, None), ("crc:b", "ssp585", None)}, None, area),
        ({("crc:b", None, "period:x")}, ("h3:in", "point:out"), area),
        (set(), None, area),
    ):
        selection = ResolvedSelection(
            spec=DataSpec([Variable("crc:a")], series=[SeriesSpec(Variable("crc:a"))]),
            keys=frozenset(keys),
            area=bounds,
            stations=stations,
        )
        cache: dict[object, bool] = {}
        expected = [selection.matches_record(row) for row in rows]
        assert selection.record_mask(batch, cache).to_pylist() == expected
        assert selection.record_mask(batch, cache).to_pylist() == expected


def test_record_mask_prefilter_keeps_every_intersecting_cell() -> None:
    import random

    import pyarrow as pa

    from crc_sdk._selection import ResolvedSelection

    rng = random.Random(7)
    cells = {
        h3.latlng_to_cell(rng.uniform(-90, 90), rng.uniform(-180, 180), resolution)
        for resolution in (1, 3, 5, 7)
        for _ in range(400)
    }
    for pentagon in h3.get_pentagons(5):
        cells.update(h3.grid_disk(pentagon, 2))
    areas = [
        Bounds(west=-79.40, south=43.64, east=-79.38, north=43.66),
        Bounds(west=179.5, south=-10.0, east=-179.5, north=10.0),
        Bounds(west=-180.0, south=85.0, east=180.0, north=90.0),
        Bounds(west=-180.0, south=-90.0, east=180.0, north=-80.0),
    ]
    for cell in rng.sample(sorted(cells), 40):
        latitude, longitude = h3.cell_to_latlng(cell)
        size = rng.choice((1e-4, 0.05, 2.0, 20.0))
        areas.append(
            Bounds(
                west=max(-180.0, longitude - size),
                south=max(-90.0, latitude - size),
                east=min(180.0, longitude + size),
                north=min(90.0, latitude + size),
            )
        )
    rows = [
        {
            "variable_id": "crc:a",
            "scenario_id": None,
            "period_id": None,
            "support_id": f"h3:{cell:x}",
            "cell_id": cell,
            "geometry": None,
        }
        for cell in sorted(cells)
    ]
    batch = pa.RecordBatch.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("variable_id", pa.string()),
                ("scenario_id", pa.string()),
                ("period_id", pa.string()),
                ("support_id", pa.string()),
                ("cell_id", pa.uint64()),
                ("geometry", pa.binary()),
            ]
        ),
    )
    for area in areas:
        selection = ResolvedSelection(
            spec=DataSpec([Variable("crc:a")], series=[SeriesSpec(Variable("crc:a"))]),
            keys=frozenset({("crc:a", None, None)}),
            area=area,
        )
        expected = [selection.matches_record(row) for row in rows]
        assert selection.record_mask(batch, {}).to_pylist() == expected
