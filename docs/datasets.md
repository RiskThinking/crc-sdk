# Canonical datasets

[Docs index](README.md) · [Scientific declarations](scientific-data.md)

Dataset `2.0` stores the science declared for participating series and the
numerical evidence or law for each original spatial support. Historical hazard
schemas and field aliases are rejected.

## Rows

| Field | Meaning |
| --- | --- |
| `variable_id` | Exact measured-variable identity |
| `scenario_id` | Declared scenario identity, or null |
| `period_id` | Content-derived period reference, or null |
| `support_id` | Identity of the original spatial support |
| `cell_id` | Valid H3 delivery index |
| `geometry` | Exact 2D WGS84 source support as WKB, when required |
| `curve` | Current framework distribution or quantile evidence |
| `fit` | Authoritative successful fitting record, when fitted |
| `failure` | Retained failed fitting record with its code and message |

The key is `(variable_id, scenario_id, period_id, support_id, cell_id)`; null
dimensions participate consistently in uniqueness. Every row references a
declared series. A row contains exactly one curve or failure. A supplied
quantile curve has no fictional fit record.

## Numerical and spatial meaning

The numerical payload is the framework's strict tagged union. A point mass
contains one value. Gumbel laws have no shape parameter. Histograms, atoms,
parametric laws, hurdles, completed curves, and transformed laws retain their
actual representations. Completion and conditioning retain the scientific
choice and original evidence. Historical curve dictionaries are rejected.

H3 is a delivery optimization. Source geometry governs coverage; an index does
not broaden a point into an area. Polygon topology and WKB layout are validated.
Repeated support IDs must identify the same actual source support, including
when equivalent WKB encodings use different byte orders.

Arrow uses a tagged struct with only its active numerical branch populated.
Inactive branches remain null. Recursive numerical children and fitting records
use canonical JSON validated by the framework's exact loaders. Parquet round
trips preserve those types and declarations.

## Supply quantile rasters

`Dataset.from_raster` accepts one same-grid, single-band raster per supplied
non-exceedance probability. Each pixel retains its original footprint and
quantile knots. The constructor describes work; terminals acquire verified
snapshots and stream the selected pixels.

```python
from crc_framework import DepthAboveGround, LengthUnit, Tail
from crc_sdk import Dataset, Probability, Variable

depth = Dataset.from_raster(
    {0.90: "depth-q90.tif", 0.99: "depth-q99.tif"},
    variable=Variable(
        "company:flood-depth",
        measurement=DepthAboveGround(unit=LengthUnit.METRE),
    ),
    probability=Probability.annual(tail=Tail.UPPER),
)
```

The probability keys are supplied data. All rasters must declare the same grid
and coordinate system. No fitting or tail completion is implicit. A pixel with
at least two valid knots retains those knots; insufficient pixels are counted
as skipped with a diagnostic.

## Metadata and provenance

Metadata contains variables, scenarios, actual periods, series probability
meaning, source product and member identity, spatial support, typed resource
identities, transformations, and the execution report. Global unit, tail,
window, fitting, and coverage summaries do not override per-series science.
Stored row-key constants and redundant envelope checksums are absent.

Resource hashes use `sha256:` followed by 64 lowercase hexadecimal digits.
Source license, attribution, and timezone-aware retrieval time are recorded when
known. Unit conversions record their actual scale and offset; renaming records
identity changes independently of numerical values.

A complete artifact's produced count equals its row count, and its no-data count
matches retained failures. Failed records identify actual attempts and their
outcomes. They never carry a fabricated numerical curve.

The packaged JSON Schema describes structure and current variants. Typed
validation additionally enforces descriptor foreign keys, generated period
identity, selected-fit consistency, null-key uniqueness, source-support identity,
and report counts. Shared valid and invalid fixtures exercise the same boundary
in Python, Arrow, Rust, and WASM.
