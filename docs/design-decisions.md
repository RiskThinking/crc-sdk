# Design decisions

[Docs index](README.md) · [Project README](../README.md)

These decisions shape canonical schema 1.3 and the open-data adapters. Schema
1.3 is metadata-only: physical columns, row key and sort order are unchanged
from 1.2, and every 1.0–1.2 file reads unchanged.

## Ensemble identity

Spread between climate models is not year-to-year variability, so silently
pooling models answers a different question.

- Dataset metadata may carry an optional `ensemble` object: `pooling`
  (`single_member` | `pooled` | `unknown`), `models`, `members`, and optional
  `scenario`, `downscaling_method`, `bias_adjustment`, `pooling_method`.
  Absent means `unknown`.
- Open adapters write **one model, member, scenario and window per dataset**
  (`single_member`). Compare models across files rather than inside one.
- Pooled datasets are legitimate when labelled `pooled`.
- Rejected: encoding model identity in `source_id` (opaque) or adding a model
  dimension to the row key (a physical schema change).

## Temporal window and pathways

`horizon` and `pathway` are row-key columns whose meaning used to be implicit.

- Optional `temporal_window` metadata: `kind` (`window` | `time_invariant`),
  inclusive `start_year`/`end_year`, `horizon` = centre year
  `(start + end) // 2`, optional `baseline`, `calendar` (`gregorian`, `noleap`,
  `360_day`, `unspecified`), `minimum_complete_years`, and `reference_year` for
  time-invariant data (rows then carry it as `horizon`).
- `crc_sdk.pathways` maps every known label (SSPs, warming-level bands,
  `historic`/`historical` as aliases of one id, and others) to crc-framework
  `PATHWAYS`. Writes keep the **source's own label**; unknown labels warn and
  `strict=True` rejects them. `register_pathway` adds labels.
- Rejected: forcing one spelling at write time (breaks consumers keyed on
  either) or a per-row window column (constant per dataset).

## Probability semantics

A canonical curve is a quantile function, but its probability axis can mean
different things. Evaluating "the 100-year value" on the wrong kind is silently
meaningless.

- Optional `probability_semantics` (absent = unspecified, no warnings):
  `annual_exceedance`, `annual_value_distribution`, `within_period_percentile`,
  `projection_uncertainty`, `estimate_confidence`. Return-period evaluation of
  the last three warns, or raises with `strict=True`.
- Optional `source_return_period_convention`: `one_minus_inverse`
  (`p = 1 - 1/T`, default) or `poisson` (`p = exp(-1/T)`). Ingest normalizes to
  canonical non-exceedance probabilities and records the source convention. The
  two differ little in probability for `T ≥ 10` but noticeably in the tail, so
  the choice is always stated.
- Rejected: inferring semantics from free-text `value_semantics`, or making
  the field required (breaks older files).

## Hazard registry

Hazard knowledge (units, tail, eligibility floors) should not be hand-copied
between repositories, and the fitter must stay hazard-agnostic.

- `crc_sdk.registry` optionally maps `hazard_name` → `HazardSpec`: unit, value
  semantics, tail, block definition, aliases, eligibility defaults
  (`minimum_informative_value`, `minimum_informative_knots`,
  `minimum_distinct_informative_values`) and preferred families.
- Fitting consults a registry only when one is passed. Resolution order:
  per-hazard override in the policy > registry > policy default.
- A public seed covers the ETCCDI indices (TXx, TNn, Rx1day, Rx5day, CDD, …).
  Other catalogues can register into the same mechanism. Names include
  duration and threshold, for example `RainfallIntensity/1h`.

## Spatial support

Canonical rows are keyed by H3 cell, but indexing is separate from scientific
resolution: expanding a ~31 km reanalysis pixel into r7 cells adds no
information.

- Metadata and docs state the native resolution; the H3 resolution is a
  delivery choice. Defaults are r5 for climate indices and r7 for water.
- Grids use conservative H3 overlap coverage. Stations keep their points (an
  area of representativeness, if ever added, must be explicit, distance-capped
  and stored as geometry). Exposure counts must be allocated conservatively,
  never duplicated across cells.

## Fit methods and reproducibility

- `CurveFitProvenance.method` is `quantile_least_squares`, `censored_quantile`,
  `sample_mle` or `sample_lmoments`, with `input_kind`
  (`probability_labelled` | `samples`), optional `initialization` and
  `sample_resampling`.
- Probability-labelled inputs use quantile methods; sample methods are for
  genuine samples only. Fitting return-period vectors as samples is invalid.
- Reproducibility is a tolerance on evaluated quantiles, not bit identity
  across architectures; provenance records `platform` and
  `crc_framework_version`.

Details: [fitting](fitting.md).

## Diagnostics sidecar

`on_fit_failure="skip"` and eligibility screens would otherwise drop or
downgrade rows without a trace. An optional Parquet sidecar records one row per
source row (outcome, family, residuals, attempted and failed families, reason),
written to a partial path and published only on success. Extra canonical
columns were rejected because they would bloat every consumer.
