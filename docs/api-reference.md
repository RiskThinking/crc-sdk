# Python interface contract

[Docs index](README.md) · [Development](development.md)

Use `crc_sdk` for data access and workflows and `crc_framework` for numerical
distributions, fitting specifications, measurements, units, and impacts.

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
| Simulate annual insured losses | `LossRun` with a named damage response and `PolicyColumns` |
| Combine separately insured perils | `LossRun.separate_perils()` with a framework `PerilDependence` |
| Compare paired annual loss runs | `future.compare(baseline)` on `LossResult` |
| Inspect metadata | `inspect` |
| Describe work without I/O | `explain` |
| Collect or preview | `read(limit=None)` |
| Iterate batches | Context-managed `batches` |
| Save | `write(path, overwrite=False, runtime=None)` |
| Validate all selected data | `validate` |
| Reuse one execution | Context-managed `execute` |

Supplied quantiles can be evaluated directly. Result terminals execute upstream
dependencies automatically. Choosing a statistical fit is an explicit scientific
operation. [Evaluation guide](evaluation.md)

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
`EvaluationArtifact`, `DatasetCollection`, or `LossArtifact` according to the data
kind. Each saved handle has a path and can be reopened by that type's `open`
method. A saved collection is a portable directory bundle containing its
selected dependencies.
Publication is atomic on supported local filesystems. [Collection guide](collections.md)

An explicit `execute(runtime=...)` scope owns one runtime. Its `read`, `batches`,
`validate`, and `write` methods reuse that scope. `explain(runtime=...)` describes
effective cache and resource settings without I/O.

`LossRun.read()` returns immutable trial totals and optional component losses in a
`LossResult`. Components are interests for one peril and perils for a combined
run. `summary()` computes metrics from that sample, and `compare()`
pairs climate runs on their common primary trial stream. A `LossArtifact` can
reopen the same sample for both operations. [Annual insured loss](insurance.md)

## Verify declarations and installed packages

[`public-api.json`](../crc_sdk/contracts/api/public-api.json) declares exports,
signatures, enum members, and removals.
[`release.json`](../crc_sdk/contracts/api/release.json) records package and
artifact versions.

```sh
python tools/api_contracts/check.py
python tools/api_contracts/check.py --runtime
```

Use the first command to validate the declarations and the second to verify an
installation. Check the contract when changing the public interface and update
the implementation and behavior tests together. [Development](development.md#releases)
