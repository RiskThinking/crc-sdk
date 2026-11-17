# Python interface contract

[Docs index](README.md) · [Development](development.md)

The release contract declares one canonical Python interface. Its source is
[`public-api.json`](../crc_sdk/contracts/api/public-api.json); the independent
[`release.json`](../crc_sdk/contracts/api/release.json) records package and format
versions. Publication requires the installed implementation to conform to both.

Use `crc_sdk` for data access and workflows and `crc_framework` for numerical
distributions, fitting specifications, measurements, units, and impacts. The SDK
does not re-export the framework's numerical interface.

| Work | Canonical interface |
| --- | --- |
| Read a provider | Explicit constructors such as `ERA5`, `CRCOpen`, and `JRC` |
| Supply statistical samples | `Samples.from_table` |
| Supply native time observations | `Observations.from_table` |
| Supply projections or spatial evidence | `Projections.from_table`, `Layer.from_table` |
| Supply quantile curves | `Dataset.from_quantiles` |
| Open canonical data | `Dataset.open` |
| Compose scientific members | `DatasetCollection` |
| Select | `select` with exact typed references or user identities |
| Rename | `rename` with an injective mapping |
| Fit | `fit` with a framework fitting specification |
| Evaluate assets | `evaluate`, then `probabilities` or `return_periods` |
| Add a named response | `impact` |
| Inspect metadata | `inspect` |
| Describe work without I/O | `explain` |
| Collect or preview | `read(limit=None)` |
| Iterate batches | Context-managed `batches` |
| Save | `write(path, overwrite=False, runtime=None)` |
| Validate all selected data | `validate` |
| Reuse one execution | Context-managed `execute` |

Supplied quantile curves are directly queryable. There is no preservation or
preparation call before evaluation. Result terminals execute every upstream
dependency automatically; choosing a statistical fit remains an explicit
scientific operation.

Single-series constructors accept scientific declarations directly. Scenario
and period may be absent. Unknown user quantities require no taxonomy entry or
custom Python class. Model-specific measurement facts are enforced when an
operation needs them.

Closed choices use enums such as `FitFailure`, `Calendar`, and provider-owned
indicator enums. User variable IDs, scenario IDs, column names, paths, and
response names remain ordinary data. Identifiers are exact nonempty strings
without boundary whitespace; no case folding or alias lookup occurs implicitly.

The [source guide](sources.md) describes provider-specific capabilities.
`Sampling`, typed aggregation specifications, and `Coverage` make temporal
reductions explicit. `TimeSeriesColumns`, `ProjectionColumns`, and `LayerColumns`
map the corresponding input kind without unrelated placeholder fields.

`read` returns `ReadResult` with its Arrow table, typed metadata, and immutable
execution report. Completed sample results can be fitted without reacquiring
their original source. Bounded previews report incomplete traversal and cannot
silently become fitting samples.

`write` returns `Dataset`, `Samples`, `Observations`, `Projections`, `Layer`,
`EvaluationArtifact`, or `DatasetCollection` according to the data kind. Each
saved handle has a path and can be reopened by that type's `open` method. A saved
collection is a portable directory bundle containing its selected dependencies.
Only local atomic publication is supported.

An explicit `execute(runtime=...)` scope owns one runtime. Its `read`, `batches`,
`validate`, and `write` methods reuse that scope. `explain(runtime=...)` describes
effective cache and resource settings without I/O.

## Verify declarations and installed packages

```sh
python tools/api_contracts/check.py
python tools/api_contracts/check.py --runtime
python tools/api_contracts/check.py --runtime --package-root /path/to/site-packages
```

The first command checks the allowlist, typed signatures, enum choices,
inheritance, removals, and package/version agreement without importing the SDK
or acquiring data. The runtime command additionally checks the installed
package's resources, exports, constructors, methods, properties, defaults,
annotations, enum members, and removal of superseded import paths. It reports
missing implementation instead of treating a declaration as implemented code.

API snapshots are not generated from the current implementation. Review an
intentional contract change in the authoritative declaration, then change the
implementation and behavior tests together. No separate export-retention
baseline exists.
