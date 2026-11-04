# Core concepts

[Docs index](README.md) · [Project README](../README.md)

## Lazy plans

Every source starts with a factory on `HazardDataset` (or a `*Plan` class for
observation sources) and is refined with immutable builder calls. Builders and
`explain()` never touch the network; work happens only at an execution point.

```text
HazardDataset.<source>(...)            # choose the source
  .for_area(bounds) / .for_stations()  # bound the request
  .years(...) / .hazards(...)          # select data
  .cache(dir, mode=...)                # choose cache behaviour
  .canonicalize(policy=...)            # choose how curves are fitted
  .explain()                           # JSON description, no I/O
  .prefetch()                          # execution: populate the source cache
  .materialize("out.parquet")          # execution: write a canonical dataset
  .for_assets(assets)...write_parquet  # execution: evaluate a portfolio
```

Bounds are always WGS84 `(west, south, east, north)`.

### Cache modes

| Mode | Behaviour |
| --- | --- |
| `reuse` (default) | Use verified cached objects; fetch only what is missing |
| `offline` | Never touch the network; fail if anything is missing or fails its checksum |
| `refresh` | Fetch again and replace cached objects (changed bytes under a pinned release are rejected) |
| `stream` | Use temporary storage only; `.cache(None, mode="stream")` |

Cache manifests record source URLs, retrieval times, HTTP validators and
SHA-256 checksums. The usual reproducible pattern is `prefetch()` once, keep the
cache, then rerun with `mode="offline"`.

## Canonical hazard datasets

A canonical dataset is one self-describing Parquet file of fitted curves,
expanded by H3 cell for spatial joins.

- **Row key:** `(hazard_name, horizon, pathway, cell_index, source_id)`.
  `cell_index` is the unsigned H3 join key, not a unique ID. Files are sorted by
  the row key by default.
- **Curve columns:** `curve_kind` selects how `curve_type`, `curve_location`,
  `curve_scale`, `curve_shape`, `curve_atom_*`, `curve_probabilities` and
  `curve_values` are read:

| `curve_kind` | Meaning | Reconstructs as |
| --- | --- | --- |
| `fitted` | Parametric family (`genextreme`, `gumbel_r`, …) | `FittedDistribution` |
| `hurdle` | Point mass (atom) plus a truncated parametric tail | `HurdleDistribution` |
| `point_mass` | Constant value at every probability | `PointMassDistribution` |
| `tabulated` | Linear interpolation of probability/value knots | `TabulatedDistribution` |
| `no_data` | `curve_type` holds the reason; quantiles return null | — |

- **Optional source geometry:** WKB in `source_geometry`. When present, asset
  joins are refined from the H3 cell to an exact `ST_Covers` test.
- **Dataset metadata:** one JSON payload under the Parquet key
  `crc.hazard.metadata`. It holds the schema version, H3 resolution, units,
  value semantics, return-period tail and support, source provenance (licence, attribution, retrieval
  time, checksum), fit policy and creation version.

Read and write with `read_hazard_dataset`, `read_hazard_metadata`,
`write_hazard_dataset` and `write_hazard_stream` (`crc_sdk.connectors`).
`distribution_from_hazard_row` reconstructs a single curve;
`curve_quantiles` evaluates whole tables.

### Schema versions

| Version | Adds |
| --- | --- |
| 1.0 | Fitted and hurdle curves |
| 1.1 | `point_mass` |
| 1.2 | `tabulated` (list columns) and `no_data` |
| 1.3 (current) | Metadata only: `probability_semantics`, `temporal_window`, `ensemble`, source licence/attribution/checksum, fit provenance |

All 1.x files share the same physical columns and row key, and every 1.0–1.2
file reads unchanged. Readers reject unknown future versions. Writers can emit
1.2 for older readers with `schema_version="1.2"`; 1.3-only fields are then
rejected.

### Writing large datasets

`write_hazard_stream(..., ordered=True)` (default) validates unique keys and
globally sorts in DuckDB, spilling to the work directory when it reaches its
memory limit. Lower the connection's `memory_limit` to bound memory.
`ordered=False` appends validated batches in input order and checks uniqueness
on key columns only: it uses less memory, writes larger files (roughly 25–40%
in tests) and keeps the same schema and API.

## Probabilities and return periods

Curves are quantile functions over **non-exceedance** probability `q` in
`[0, 1]`.

- Upper-tail hazards (flood depth, heat): return period `T` maps to
  `q = 1 - 1/T`, so the 100-year value is `ppf(0.99)`.
- Lower-tail hazards (drought indices, cold): `q = 1/T`. Portfolio evaluation
  reads the direction from the dataset's `return_period_tail` metadata.
  `return_periods_to_probabilities(..., tail=...)` takes it explicitly and must
  match the tail used at ingest; `curve_quantiles` takes probabilities only.
- Sources that use the Poisson convention record
  `source_return_period_convention="poisson"`, and ingest normalizes them.
  `return_periods_to_probabilities` always takes the convention explicitly.

`probability_semantics` states what the axis means: `annual_exceedance`,
`annual_value_distribution`, `within_period_percentile`,
`projection_uncertainty` or `estimate_confidence`. Return-period evaluation of
the last three warns (`ProbabilitySemanticsWarning`), or raises with
`strict=True`. Requests outside the fitted source support warn with
`ReturnPeriodExtrapolationWarning`.

## Spatial support

H3 resolution is a delivery choice, not scientific resolution: expanding a
30 km reanalysis pixel to r7 cells adds no local detail. Grid sources use
conservative overlap coverage, so the integer join is a superset that exact
geometry then refines. Station sources keep WKB points and imply no coverage of
surrounding land. The design rationale is in
[design decisions](design-decisions.md#spatial-support).
