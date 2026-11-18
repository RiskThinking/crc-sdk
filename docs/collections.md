# Dataset collections

[Docs index](README.md) · [Canonical datasets](datasets.md) · [Execution](execution.md)

Use a collection when independent datasets should share one selection or asset
evaluation. Give each scientific member an ID meaningful to your application.
The constructor accepts lazy dataset recipes, opened datasets, and explicit
`CanonicalReader` implementations.

```python
from crc_sdk import Dataset, DatasetCollection

collection = DatasetCollection({
    "baseline": Dataset.open("baseline.parquet"),
    "projection": Dataset.open("projection.parquet"),
})
projection = collection.select(members="projection")
inventory = projection.inspect()
result = projection.select(series=inventory.series[0]).read()
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
change scientific identity. Rename one dataset before adding it to the collection when independent data
with different definitions needs distinct IDs in the same selection.

Write a selection to a portable directory containing its typed manifest and
verified canonical Parquet resources.

```python
saved = collection.select(members="projection").write("selected-data")
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
