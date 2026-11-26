# Reading, modelling, and saving data

[Docs index](README.md) · [Scientific declarations](scientific-data.md)

Recipes describe work. `read()`, `validate()`, `write()`, and entering `batches()`
execute their dependencies automatically. Builders and `explain()` perform no
source I/O. Fitting stays explicit because it changes the scientific model.

## Use supplied quantiles directly

```python
from crc_framework import DepthAboveGround, LengthUnit, Tail
from crc_sdk import Dataset, Probability, QuantileColumns, Variable

table = {
    "station": ["harbour", "harbour"],
    "lon": [24.0, 24.0],
    "lat": [60.0, 60.0],
    "q": [0.90, 0.99],
    "depth_cm": [30.0, 85.0],
}
depth = Dataset.from_quantiles(
    table,
    variable=Variable(
        "company:water-depth",
        measurement=DepthAboveGround(unit=LengthUnit.CENTIMETRE),
    ),
    probability=Probability.annual(tail=Tail.UPPER),
    columns=QuantileColumns(
        support_id="station", longitude="lon", latitude="lat",
        probability="q", value="depth_cm",
    ),
)
assert depth.inspect().series[0].variable.id == "company:water-depth"
result = depth.read()
assert result.report.complete and result.report.valid
```

The supplied interval remains `[0.90, 0.99]`. Reading or saving does not invent
tails or fit a distribution. Input column names are ordinary user data; units
and probability meaning are typed declarations. Mappings of columns to arrays,
Arrow tables, record batches, Arrow stream interchange, and installed pandas
DataFrames use the same input boundary.

## Inspect annual samples, then fit them

Each row supplies an annual maximum temperature in Celsius at one station.
Read a small preview to check the input, then collect the full sample for the
fit. Here the fitted result is written directly without an intermediate
evidence file.

```python
from crc_framework import DistributionFamily, SampleFit, Tail, Temperature, TemperatureUnit
from crc_sdk import Dataset, Probability, Samples, Variable

observations = {
    "support_id": ["station-a"] * 5,
    "longitude": [24.0] * 5,
    "latitude": [60.0] * 5,
    "time": [2000, 2001, 2002, 2003, 2004],
    "value": [24.0, 27.0, 25.0, 31.0, 29.0],
}
samples = Samples.from_table(
    observations,
    variable=Variable(
        "company:annual-maximum-temperature",
        measurement=Temperature(unit=TemperatureUnit.CELSIUS),
    ),
    probability=Probability.annual(tail=Tail.UPPER),
)
preview = samples.read(limit=2)
assert preview.table.num_rows == 2 and not preview.report.complete
evidence = samples.read()
assert evidence.report.complete and evidence.report.valid
fitted = evidence.fit(SampleFit(family=DistributionFamily.GUMBEL_RIGHT))
saved = fitted.write("temperature.parquet")
assert saved.path is not None
assert Dataset.open(saved.path).read().report.valid
```

The collected evidence owns its Arrow values. Fitting that result uses those
values, even after the original execution closes. It never reacquires the
original source. A bounded read can be fitted only if consumption actually
reached the end and completed validation.
An exact limit boundary may remain incomplete because the source has not yet
reported exhaustion. A limit on a fitted recipe bounds output rows; fitting
still uses the complete selected evidence.

`FitFailure.RAISE` is the default. `FitFailure.KEEP_NO_DATA` retains a failed
fit with its authoritative record and warning; `FitFailure.SKIP` records the
omission and counts. Neither policy fabricates a numerical law.
Fitting failures retain their framework execution record on a typed
`FittingDiagnostic.record`, including when `validate()` returns a failure report.
Missing observations are excluded before fitting. The fit record counts the
observations available to the estimator, and the SDK report records incomplete
annual observations or identifies other omissions in its diagnostics.
When a station has no observations in the selected window, its historical
support may be unknown. `FitFailure.KEEP_NO_DATA` retains that fit failure in
the report. A spatial output row requires known support.

## Select and transform explicitly

```python
from crc_sdk import Bounds

selected = depth.select(
    variables="company:water-depth",
    area=Bounds(west=20, south=55, east=30, north=65),
)
empty = selected.select(variables=[]).read()
assert empty.table.num_rows == 0
all_variables = selected.select(variables=None)
metres = all_variables.convert_units(
    variable="company:water-depth", unit=LengthUnit.METRE,
)
renamed = metres.rename(variables={"company:water-depth": "project:depth"})
```

Omitting a selector retains it, `None` clears it, and an empty sequence selects
nothing. Unknown identities raise. A single ID is one identity, never a sequence
of characters. Inspection returns actual series combinations, and
`depth.select(series=depth.inspect().series[0])` keeps that combination and any
pinned release identity.

Spatial filtering uses the original support: a point stays a point and a polygon
keeps its footprint. Its H3 delivery index adds no coverage. Source observation
`time=` on samples is separate from selecting declared `periods=`.

Renaming validates the full participating mapping, including unchanged IDs.
It cannot merge identities or alter measurements. Unit conversion changes the
values and measurement together and records the transformation. `declare()`
fills unknown scientific facts; it rejects contradictions with facts already
declared.

## Reuse an execution or stream it

```python
from crc_sdk import Cache, CacheMode, Runtime

runtime = Runtime(cache=Cache.reuse(".crc-cache"), batch_rows=1024)
description = fitted.explain(runtime=runtime)
assert description.cache.mode is CacheMode.REUSE
with fitted.execute(runtime=runtime) as execution:
    report = execution.validate()
    values = execution.read()
    artifact = execution.write("validated-temperature.parquet")

with depth.batches(runtime=runtime) as stream:
    planned_metadata = stream.metadata
    for batch in stream:
        assert batch.num_rows <= 1024
assert stream.report.complete
```

The scoped `write()` returns the same artifact type as the recipe's `write()`.
An explicit execution resolves each source and distinct modelling node once.
Its checked batch spool supplies later terminals without reacquisition or
refitting. The runtime is selected once, on `execute()`. Separate direct
terminals are separate executions; reading does not silently warm an
unspecified source cache.

Streams yield after batch checks. Duplicate row keys and support IDs naming
different geometry are checked across the whole stream. For the first 32,768
rows the batch that breaks them fails. After that, keys move to a compressed
scratch database in the execution's temporary directory, which needs a few
bytes per row and at most 128 MiB of memory (less under a smaller
`memory_limit`), and a repeat is raised when the stream ends, before its
report can complete. Breaking early leaves an incomplete report and releases
scope-owned resources. Tables, metadata, and
immutable report snapshots remain usable after their execution closes.

`Runtime()` uses ephemeral source storage and one worker. `Cache.reuse(path)`
keeps verified source snapshots in the given directory and acquires missing
resources as needed. `Cache.offline(path)` only reads verified cached
resources and raises if a required resource is missing. `Cache.refresh(path)`
reacquires source data. [Offline replay example](sources.md#discover-a-crc-open-release)

`Runtime(workers=n)` acquires up to n CRC Open release files at once, ahead of
decoding them, and gives DuckDB n threads. Progress callbacks still run on the
consuming thread.

Direct local inputs are read from their owned snapshots. Temporary joins,
replay batches, and modelling state remain execution-owned. Replay batches are
zstd-compressed Arrow files in the execution's temporary directory, removed
when the execution closes. A borrowed
database connection remains caller-owned and keeps its configuration. Memory
limits bound collection and blocking fitting. A selection exceeding the budget
raises `MemoryError`.

`Runtime(progress=callback)` receives typed `ProgressEvent` values. Raising from
the callback cancels execution and closes owned resources.

## Validate and publish atomically

`validate()` consumes the complete selection and returns expected data failures
as an `ExecutionReport`. Other result terminals raise
`ValidationError(report=...)` for the same failures. I/O, programming, and
operational errors retain their exception types. `valid` is true only after
successful complete validation, false for known validation errors, and otherwise
unknown.

Writes accept local filesystem paths, default to no overwrite, and check the
destination before source execution. They stage batches, finish validation,
write final metadata in the Parquet footer, and publish atomically. Failure or
cancellation preserves an existing destination. No separate validation call or
intermediate output artifact is needed. `Dataset.open(path)` and
`Samples.open(path)` reopen the corresponding current format in another process.
Reopened handles expose their saved metadata and report through header-only
discovery; reading data still performs the execution's validation.

When a provider knows the original spatial supports for an empty observation
window, collected results and `Observations.open(path)` preserve those
supports. Annual aggregation can then report incomplete years at each support
after reopening.
