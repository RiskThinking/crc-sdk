# ADR-0007: Diagnostics sidecar

Status: Proposed

## Context

`on_fit_failure="skip"` and eligibility screens drop or downgrade rows with no
row-level trace. Operators had to reverse-engineer a 99.9% `no_data` rate from
aggregate counters.

## Decision

Add an **optional persisted Parquet sidecar** for every canonicalization path,
starting with `fit_cdf_quantile_batches` (`diagnostics="path.parquet"` on the
policy). The canonical contract itself is unchanged.

One row per source row (or only the non-standard ones, by option):

`cell_index, source_id, hazard_name, horizon, pathway, outcome, family,
normalized_rmse, maximum_absolute_residual, attempted_families,
failed_families, fallback, reason, treatment`

`outcome` is one of `fitted`, `hurdle`, `point_mass`, `tabulated`, `no_data`,
`skipped`, `rejected`; `reason` carries the `no_data` reason, the rejection
cause (e.g. `missing_axis_labels`) or the skip error.

Block-extrema recipes additionally store the per-cell annual series
with valid and excluded years in a companion table.

The sidecar is written to a partial path and published only on successful
completion, so a half-written sidecar is never mistaken for a result.

## Alternatives rejected

- Extra columns in the canonical file: contract change, bloats consumers.
- Counters only: cannot answer *which* rows were dropped.

## Application

- (a) The production pipeline writes the sidecar next to each output partition
  and records its digest in the run manifest.
- (b) Open adapters write the same schema plus the annual-series table.
