# Evaluate datasets at assets

Use `evaluate(assets)` to query a dataset at asset locations. The result is
a recipe that can be inspected, read, streamed, or saved. A
`DatasetCollection` supports the same workflow and keeps member identities
on its rows. [Collections](collections.md)

## Query supplied quantiles

This example uses two tables expressed as mappings from column names to lists.
Each row of the depth table supplies one knot of a curve. Rows for the same
`support_id` share a WGS84 longitude and latitude and use the declared variable
and measurement. `probability` is a non-exceedance probability and `value` is
depth in metres above local ground. With these knots, the curve supports
10-year and 100-year upper-tail queries.

The asset table has one row per asset. Here `site_id` is a custom ID column.
One asset shares the station's position and the other is outside point coverage.

```python
from crc_framework import DepthAboveGround, FloodRegion, LengthUnit, Tail
from crc_sdk import Assets, Dataset, Probability, QueryKind, Variable

depth_measurement = DepthAboveGround(unit=LengthUnit.METRE)
depth_variable = Variable("company:flood-depth", measurement=depth_measurement)
depth_table = {
    "support_id": ["station-1", "station-1", "station-1"],
    "longitude": [24.0, 24.0, 24.0],
    "latitude": [60.0, 60.0, 60.0],
    "probability": [0.5, 0.9, 0.99],
    "value": [0.2, 1.0, 1.8],
}
depth = Dataset.from_quantiles(
    depth_table,
    variable=depth_variable,
    probability=Probability.annual(tail=Tail.UPPER),
)

asset_table = {
    "site_id": ["station-warehouse", "nearby-warehouse"],
    "longitude": [24.0, 24.02],
    "latitude": [60.0, 60.0],
    "region": [FloodRegion.EUROPE.value, FloodRegion.EUROPE.value],
    "replacement_cost": [80_000, 120_000],
}
assets = Assets(asset_table, id_column="site_id", columns=["replacement_cost"])

result = depth.evaluate(assets).return_periods([10, 100]).read()
assert result.metadata.query.kind is QueryKind.RETURN_PERIOD
assert set(result.table["period_id"].to_pylist()) == {None}
assert set(result.table["scenario_id"].to_pylist()) == {None}
matched = [row for row in result.table.to_pylist() if row["support_id"] is not None]
assert [(row["query_value"], row["value"]) for row in matched] == [(10, 1.0), (100, 1.8)]
assert result.report.counts.matched_assets == 1
assert result.report.counts.unmatched_assets == 1
```

Each call in the chain describes work until `read()` runs the full recipe.
You can call `probabilities([0.5, 0.9])` to query non-exceedance probabilities
directly. An annual
`return_periods` query uses years and requires each selected series to declare
its annual probability and tail. The upper tail uses `q = 1 - 1 / T` and keeps
`1 / T` directly during inversion, so a finite period can resolve past the
largest floating-point probability below one. Supplied upper-tail knot endpoints
allow a relative comparison tolerance of `1e-12` in the tail probability to
account for conversion rounding. The lower tail uses `q = 1 / T`.
Query order is preserved. A query outside
the supplied quantile interval retains a row with a null value and the
`query_outside_support` failure code. [Scientific declarations](scientific-data.md)

## Choose spatial support

The default matches the actual support. A point covers its own location.
A polygon covers the asset geometry within its declared footprint, including
its boundary. H3 cells provide coverage when the series declares cell support.
Equivalent delivery copies of the same original support are matched once.
Conflicting geometry or models for one support raise a validation error.
Unmatched assets retain rows with a null `support_id` and a failure reason.
At a shared edge, a point belongs to the rectangular grid cell to the north
or east. A point on the outer boundary can still match the adjoining cell.
For other overlaps, use
`SpatialMatch.covering(preferred_supports=["site-survey", "regional-model"])`.
The first ID in the list that covers the asset wins, including on a shared
edge. Unresolved overlaps produce an `overlapping_supports` row failure for
that asset and series. Saved evaluations retain the priority order.

Use `SpatialMatch.nearest` with point assets and point-supported series. Set
an explicit distance cap. Results include the matched support and its
great-circle distance in metres. Ties raise a validation error.

```python
from crc_sdk import Bounds, Distance, SpatialMatch

area = Bounds(west=23.9, south=59.9, east=24.1, north=60.1)
spatial = SpatialMatch.nearest(max_distance=Distance.kilometres(5))
nearby = depth.select(area=area).evaluate(assets, spatial=spatial)
nearby_result = nearby.return_periods([10, 100]).read()
assert nearby_result.report.counts.matched_assets == 2
assert max(nearby_result.table["distance_metres"].to_pylist()) < 5_000
```

Select source variables, scenarios, periods, exact series, or an area before
or after `evaluate`. For example, `depth.select(area=area).evaluate(assets)`
and `depth.evaluate(assets).select(area=area)` apply the same source filter.
A collection evaluation also accepts `members`. Omitted arguments retain
the selection, `None` clears a filter, and an empty sequence selects zero rows.
Unknown IDs raise an error.

## Add named responses

A calibrated response uses the exposure at the requested event rank as its
input. For a 100-year event the SDK first obtains the source's 100-year
exposure, then evaluates each selected response at that exposure. Multiple
responses can use the same variable and keep their individual quantities and
units.

```python
from crc_framework import (
    AssetType, BuiltinImpact, Impact, InundationContext, TimeDuration, TimeUnit,
)
from crc_sdk import EvaluationMetadata

damage = Impact.builtin(BuiltinImpact.INUNDATION)
requirements = damage.requirements
print(requirements.accepted_units)
assert requirements.context_type is InundationContext
downtime = Impact.linear(
    slope=2.0,
    input=depth_measurement,
    output=TimeDuration(unit=TimeUnit.DAY),
)
evaluation = (
    nearby.return_periods([10, 100])
    .impact(
        "damage_fraction",
        damage,
        variable=depth_variable,
        context=InundationContext(
            region=assets.column("region"),
            asset_type=AssetType.RESIDENTIAL,
        ),
    )
    .impact("downtime", downtime, variable=depth_variable)
)
inspection = evaluation.inspect()
declared = inspection.metadata
assert isinstance(declared, EvaluationMetadata)
print(declared.query.kind, declared.query.values)
print({response.name: response.output.unit for response in declared.responses})

responses = evaluation.read()
assert responses.table.num_rows == 12
damage_only = evaluation.select(responses="damage_fraction").read()
assert set(damage_only.table["response_id"].to_pylist()) == {"damage_fraction"}
```

Inspection reports the output schema for in-memory assets. File and SQL asset
schemas become available when the evaluation runs.

The framework checks the input measurement and context against the calibrated
model. `damage.requirements.input` shows the measurement the model expects.
`context_type` names the typed context constructor and `context_fields` lists
its argument names and choices. An asset-column binding reads the column's
values from the input table. The `columns` argument on `Assets` independently
chooses the attributes carried into the nested `asset` field. Here the
`region` values resolve the built-in response and the output carries
`replacement_cost`. The configured site IDs appear in `asset_id`. Retained
Arrow attributes preserve their types and precision in memory and in saved
results, including nested values and nanosecond timestamps and durations.
[Framework measurement and model requirements](https://github.com/RiskThinking/crc-framework-rs/blob/main/docs/measurements.md)

Fire-weather damage applies only in the wildland-urban interface.
`BuiltinImpact.FIRE_WEATHER_INDEX` requires
`WildfireContext(wildland_urban_interface=assets.column("in_wui"))`, with a
boolean asset column. Assets outside the interface get zero damage.

Exposure rows are included by default and have a null `response_id`. Named
response rows identify their definition in `response_id`. A referenced
model's concrete numerical model, context, and asset assignments are stored
in metadata and linked through `resolution_id`. Reopening checks those
assignments even when the bound columns are omitted from the output. Successful
named responses have finite values. Full validation recomputes each named
response and compares its value within a relative tolerance of `1e-12`. With a
named-only selection, each response row retains the source value or failure in
its `exposure` field for validation. Validation also checks retained asset
attributes, recorded source releases, and diagnostics for missing results.
Rows that share a source support agree on its availability and on the exposure
at each query coordinate. Validation checks that exposure quantiles are
nondecreasing in non-exceedance order and that supported queries form one
contiguous range. A matched source-unavailable outcome retains its provider code
and message and has `failure.kind` set to `source_unavailable`. Upstream counts
separately bound matches with source unavailability or failed fits and matches
to supports with curves.
The evaluation report carries acquired input checksums, upstream diagnostics,
skipped counts, and annual completeness counts from the input reports. Reopening
checks those inherited facts against the embedded input metadata.
With covering matches, an evaluation reads only rows whose supports can cover
its assets: CRC Open skips files whose extent misses every asset, and H3 cell
rows outside the assets' cells are not decoded. Input reports therefore count
and diagnose the rows read for these assets, not the whole selected source.
The declared series, and the rows of series no read support covers, are
unchanged.
Use `select(responses=...)` to choose named rows explicitly. Source filters also
limit the responses in the default selection. The returned table has one row
per asset, participating series, requested coordinate, and exposure or named
response.

### Calibrate your own cases

Use `ContextualImpact` when your preferred numerical model varies by an asset's
category. Define a frozen `ResponseContext` dataclass with the fields your
model needs. Case values are exact strings or string-valued enum members.
Each case holds a framework `Impact` with the same input and output measurement.

```python
from dataclasses import dataclass
from enum import Enum

from crc_sdk import ColumnBinding, ContextualImpact, ResponseContext


class FacilityUse(Enum):
    WAREHOUSE = "warehouse"
    OFFICE = "office"


@dataclass(frozen=True)
class FacilityContext(ResponseContext):
    use: FacilityUse | ColumnBinding


days = TimeDuration(unit=TimeUnit.DAY)
warehouse = Impact.linear(slope=2.0, input=depth_measurement, output=days)
office = Impact.tabulated([(0.0, 0.0), (1.0, 1.0), (3.0, 7.0)],
                          input=depth_measurement, output=days)
repair_model = ContextualImpact([
    (FacilityContext(FacilityUse.WAREHOUSE), warehouse),
    (FacilityContext(FacilityUse.OFFICE), office),
])
assert repair_model.requirements.context_type is FacilityContext
repair = nearby.return_periods([10, 100]).impact(
    "repair_days", repair_model, variable=depth_variable,
    context=FacilityContext(FacilityUse.WAREHOUSE),
)
assert repair.read().table.num_rows == 8
```

To resolve a different case per asset, pass `FacilityContext(assets.column("use"))`
when the asset table has a `use` column. An unknown case raises a validation
error. A saved evaluation records the declared cases, the concrete context and
the calibrated model used for each asset. Resolve a concrete context with
`repair_model.resolve(FacilityContext(FacilityUse.WAREHOUSE))` to obtain the
framework `Impact` for numerical operations such as expectation or pushforward.

## Save and reopen

`write` executes the complete recipe, validates the rows, and publishes the
evaluation as Parquet. The saved handle exposes the path and the typed
metadata. Another process can open that path and read the saved result.
Evaluation metadata stores asset IDs once. Resolutions in the artifact JSON
refer to zero-based positions in that asset list. The Python resolution still
exposes the original IDs as `asset_ids`.

```python
from crc_sdk import EvaluationArtifact

with evaluation.execute() as execution:
    report = execution.validate()
    preview = execution.read(limit=4)
    saved = execution.write("events.parquet")
assert report.valid and preview.table.num_rows == 4
assert saved.path is not None
reopened = EvaluationArtifact.open(saved.path)
events = reopened.read()
assert events.report.complete and events.report.valid
```

Use `batches()` for a stream, `read(limit=...)` for a bounded preview, or call
`write()` directly if you only need to save the result. The execution scope
above shares computed rows across validation, preview, and writing.
If the metadata exceeds the Parquet footer limit, split the assets into
smaller evaluations before calling `write()`.
[Execution and persistence](execution.md)
