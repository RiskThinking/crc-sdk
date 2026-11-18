# Dataset collections

[Docs index](README.md) · [Canonical datasets](datasets.md) · [Execution](execution.md)

Use a collection when independent datasets should share one selection or asset
evaluation. Give each scientific member an ID meaningful to your application.
The constructor accepts lazy dataset recipes, opened datasets, and explicit
`CanonicalReader` implementations.

The example supplies two depth curves at one point. Each input table contains
the same probability knots and its own depth values in metres. The series
periods describe the assumed baseline and projection windows.

```python
from crc_framework import DepthAboveGround, LengthUnit, Tail
from crc_sdk import Dataset, DatasetCollection, Period, Probability, Variable

measurement = DepthAboveGround(unit=LengthUnit.METRE)
variable = Variable("company:flood-depth", measurement=measurement)
probability = Probability.annual(tail=Tail.UPPER)
station = {
    "support_id": ["harbour", "harbour"],
    "longitude": [24.0, 24.0],
    "latitude": [60.0, 60.0],
    "probability": [0.9, 0.99],
}
collection = DatasetCollection({
    "baseline": Dataset.from_quantiles(
        {**station, "value": [0.3, 0.85]},
        variable=variable,
        period=Period.window(1981, 2010),
        probability=probability,
    ),
    "projection": Dataset.from_quantiles(
        {**station, "value": [0.6, 1.2]},
        variable=variable,
        period=Period.window(2041, 2070),
        probability=probability,
    ),
})
projection = collection.select(members="projection")
projection_series, = projection.inspect().series
result = projection.select(series=projection_series).read()
assert result.table["member_id"].to_pylist() == ["projection"]
```

The normalized Arrow table adds `member_id` and `resource_id` to the canonical
dataset fields. The member identifies a scientific input. The resource
identifies the physical data read for that member. Several members can share one
resource, and a member in a bundle can span several resources. The collection
keeps each member's rows separate and rejects duplicate scientific row keys
across a member's resources. Inspection metadata describes the member,
resource, and series relationships without running fitting stages.

Collections reconcile the scientific definitions of participating series.
Select one member or the series you need when independent inputs use the same
ID with different definitions. Display labels and acquisition provenance do not
change scientific identity. Rename one dataset before adding it to the
collection when independent data with different definitions needs distinct IDs
in the same selection.

Evaluate all members together and add any named responses to the shared
variable. This example uses illustrative linear functions. Each slope converts
the exposure in metres into a duration in days. Rows keep both the member and
response identities. [Evaluation](evaluation.md)

```python
from crc_framework import Impact, TimeDuration, TimeUnit
from crc_sdk import Assets

assets = Assets({"asset_id": ["warehouse"], "longitude": [24.0], "latitude": [60.0]})
days = TimeDuration(unit=TimeUnit.DAY)
inspection_time = Impact.linear(slope=1.0, input=measurement, output=days)
repair_time = Impact.linear(slope=3.0, input=measurement, output=days)
evaluation = (
    collection.evaluate(assets)
    .return_periods([100])
    .impact("inspection_days", inspection_time, variable=variable)
    .impact("repair_days", repair_time, variable=variable)
    .select(responses=["inspection_days", "repair_days"])
)
with evaluation.batches() as stream:
    rows = [row for batch in stream for row in batch.to_pylist()]
assert stream.report.complete
assert {(row["member_id"], row["response_id"]) for row in rows} == {
    ("baseline", "inspection_days"),
    ("baseline", "repair_days"),
    ("projection", "inspection_days"),
    ("projection", "repair_days"),
}
```

Write a selection to a portable directory containing its typed manifest and
verified canonical Parquet resources.

```python
saved = projection.write("selected-data")
assert saved.path is not None
reopened = DatasetCollection.open(saved.path)
assert reopened.read().metadata.selection.members == ("projection",)
```

Reopening the directory reproduces the saved rows and science, even after the
original sources are removed. The bundle contains the saved selection's data.
Additional selectors act within that snapshot. An explicit empty selection
writes a valid empty bundle and remains empty when reopened.

The writer validates the complete selection before publishing the directory.
Writes default to a new path. To replace a bundle, pass `overwrite=True` on a
filesystem that supports atomic directory exchange. Existing bundle replacement
is available on compatible Linux and macOS filesystems. On Windows, write to a
new directory.

Use an explicit execution when you want to validate, preview, and save with one
source resolution.

```python
with collection.execute() as execution:
    report = execution.validate()
    preview = execution.read(limit=5)
    saved = execution.write("verified-data")
```
