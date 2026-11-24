# Canonical datasets

[Docs index](README.md) · [Scientific declarations](scientific-data.md)

Dataset `2.0` stores the science declared for participating series and the
numerical evidence or law for each original spatial support.

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
| `failure` | Source unavailability or a retained fitting failure, with its code and message |

The key is `(variable_id, scenario_id, period_id, support_id, cell_id)`; null
dimensions participate consistently in uniqueness. Every row references a
declared series. A row contains exactly one curve or failure. A fitting
failure carries its authoritative fit record. Source unavailability carries
`kind: source_unavailable` in JSON. In Arrow, a failure with a null `record`
represents source unavailability. Both failure outcomes count toward `no_data`.

## Numerical and spatial meaning

The numerical payload is the framework's strict tagged union. A point mass
contains one value. Gumbel laws have no shape parameter. Histograms, atoms,
parametric laws, hurdles, completed curves, and transformed laws retain their
actual representations. Completion and conditioning retain the scientific
choice and original evidence.

H3 is a delivery optimization. Source geometry governs coverage; an index does
not broaden a point into an area. Polygon topology and WKB layout are validated.
Repeated support IDs must identify the same actual source support, including
when equivalent WKB encodings use different byte orders.

Arrow uses a tagged struct with only its active numerical branch populated.
Inactive branches remain null. Recursive numerical children and fitting records
use canonical JSON validated by the framework's exact loaders. Parquet round
trips preserve those types and declarations.

## Use the curves directly

Evaluation answers return-period and probability queries for you. For anything
else, such as a CDF, a custom statistic, or a map that evaluates curves in the
browser, `decode_curves` turns the `curve` column of rows you read into
framework objects. Failure rows decode to `None`.

```python
from crc_framework import DepthAboveGround, LengthUnit, Tail
from crc_sdk import Dataset, Probability, Variable, decode_curves

rows = Dataset.from_quantiles(
    {
        "support_id": ["site-a"] * 3,
        "longitude": [6.96] * 3,
        "latitude": [50.94] * 3,
        "probability": [0.5, 0.9, 0.99],
        "value": [0.0, 0.3, 1.2],
    },
    variable=Variable(
        "company:flood-depth", measurement=DepthAboveGround(unit=LengthUnit.METRE)
    ),
    probability=Probability.annual(tail=Tail.UPPER),
).read().table
(curve,) = decode_curves(rows)
assert curve is not None
assert abs(curve.quantile(0.95) - 0.8) < 1e-9
document = curve.to_json()  # canonical JSON, also read by the JavaScript package
```

It works on any `read()` table or `batches()` batch of Dataset 2.0 rows.

## Supply quantile rasters

`Dataset.from_raster` accepts one same-grid, single-band raster per supplied
non-exceedance probability. Each pixel retains its original footprint and
quantile knots. Substitute your own GeoTIFF paths below and install the
`raster` extra to read them. Building the recipe only records the paths.

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
assert depth.explain().operations
```

The probability keys are supplied data. Supply `probability=` when the series
has a declared annual interpretation for return-period queries. All rasters
must declare the same grid and coordinate system. No fitting or tail completion
is implicit. A pixel with
at least two valid knots retains those knots; insufficient pixels are counted
as skipped with a diagnostic.

## Metadata and provenance

Metadata contains variables, scenarios, actual periods, series probability
meaning, source product and member identity, spatial support, typed resource
identities, transformations, and the execution report. Units, annual tails,
time windows, fits, and coverage retain the scope where they were declared
or observed.

Resource hashes use `sha256:` followed by 64 lowercase hexadecimal digits.
Source license, attribution, and timezone-aware retrieval time are recorded when
known. Unit conversions record their actual scale and offset; renaming records
identity changes independently of numerical values. Standard conversions are
checked against the recorded input measurement and the unit coefficients.
Provider precipitation mass flux uses water-equivalent depth, where one
kilogram per square metre corresponds to one millimetre of water.

A complete artifact's produced count equals its row count, and its no-data count
matches retained failures. Each source-unavailable row preserves the known
support and the provider's reason. Each failed fitting row describes its attempt
and outcome.

The packaged JSON Schema describes structure and current variants. Typed
validation additionally enforces descriptor foreign keys, generated period
identity, selected-fit consistency, null-key uniqueness, source-support identity,
and report counts. Shared valid and invalid fixtures exercise the same boundary
in Python, Arrow, Rust, and WASM.
