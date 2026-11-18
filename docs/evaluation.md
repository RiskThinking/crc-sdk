# Evaluate datasets at assets

Use `evaluate(assets)` to query a dataset at asset locations. The result is
a recipe that can be inspected, read, streamed, or saved. A
`DatasetCollection` supports the same workflow and keeps member identities
on its rows. [Collections](collections.md)

## Query supplied quantiles

This example uses two tables expressed as mappings from column names to lists.
Each row of the depth table supplies one quantile at the same station point.
Each row of the asset table supplies a site ID, a WGS84 location, and attributes.

```python
from crc_framework import DepthAboveGround, FloodRegion, LengthUnit, Tail
from crc_sdk import Assets, Dataset, Period, Probability, Variable

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
    period=Period.window(2011, 2030),
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

exposure = depth.evaluate(assets).probabilities([0.5, 0.9])
result = exposure.read()
assert result.report.counts.matched_assets == 1
assert result.report.counts.unmatched_assets == 1
```

The `probabilities` query accepts non-exceedance probabilities. An annual
`return_periods` query uses years and requires each selected series to declare
its annual probability and tail. The upper tail uses `q = 1 - 1 / T` and
the lower tail uses `q = 1 / T`. Query order is preserved. A query outside
the supplied quantile interval retains a row with a null value and the
`query_outside_support` failure code. [Scientific declarations](scientific-data.md)

## Choose spatial support

The default matches the actual support. A point covers its own location.
A polygon covers the asset geometry within its declared footprint, including
its boundary. H3 cells provide coverage when the series declares cell support.
Unmatched assets retain rows with a null `support_id` and a failure reason.
Overlapping covering supports raise a validation error.

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

The framework checks the input measurement and context against the calibrated
model. Inspect `damage.requirements` for its input unit, qualifiers, and
typed context constructor. An asset-column binding reads the column's values
from the input table. The `columns` argument on `Assets` independently
chooses the attributes carried into the nested `asset` field. Here the
`region` values resolve the built-in response and the output carries
`replacement_cost`. The configured site IDs appear in `asset_id`.
[Framework measurement and model requirements](../../crc-framework-rs/docs/measurements.md)

Exposure rows are included by default and have a null `response_id`. Named
response rows identify their definition in `response_id`. A referenced
model's concrete numerical model and context are stored in metadata and
linked through `resolution_id`. Use `select(responses=...)` to choose
named rows explicitly. The returned table has one row per asset, participating
series, requested coordinate, and exposure or named response.

## Save and reopen

`write` executes the complete recipe, validates the rows, and publishes the
evaluation as Parquet. The saved handle exposes the path and the typed
metadata. Another process can open that path and read the saved result.

```python
from crc_sdk import EvaluationArtifact

saved = evaluation.write("events.parquet")
assert saved.path is not None
reopened = EvaluationArtifact.open(saved.path)
events = reopened.read()
assert events.report.complete and events.report.valid
assert events.metadata.schema_version == "2.0"
```

Use `batches()` for a stream or `read(limit=...)` for a bounded preview.
For several terminals over the same work, an `execute(runtime=...)` scope
shares the computed rows. [Execution and persistence](execution.md)
