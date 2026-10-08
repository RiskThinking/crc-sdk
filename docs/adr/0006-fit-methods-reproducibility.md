# ADR-0006: Fit methods and reproducibility

Status: Proposed

## Context

Production fits distributions that already exist (probability-labelled
quantiles), never time series. Open adapters also fit genuine samples (annual
maxima). Treating return-period-labelled depths as observations confuses
quantiles with samples. Floating-point differences between ARM and x86 can
affect Gumbel/GEV/tabulated allocation near optimizer convergence boundaries.

## Decision

- `CurveFitProvenance.method`: `quantile_least_squares` \| `sample_mle` \|
  `sample_lmoments`, plus an optional `initialization` (e.g. `lmoments`), `input_kind`
  (`probability_labelled` \| `samples`) and `sample_resampling` (knot count
  when samples were resampled to a quantile grid).
- **Probability-labelled inputs** (CDF quantiles, return-period vectors)
  **always** use quantile least squares. Sample-based MLE is for genuine
  samples only.
- Genuine samples may use sample MLE (GEV starts from L-moments/PWM, refined
  by penalised MLE), standalone L-moments, or plotting positions + least
  squares. The choice is recorded.
- **Reproducibility is a tolerance on evaluated quantiles, not bit identity**
  across architectures. Provenance records `platform` and `crc_framework`
  version. The tolerance policy is the publication gate.

## Alternatives rejected

- Bit-identical output across architectures: not achievable with iterative
  optimisers near convergence boundaries.
- Fit return-period vectors as samples: statistically invalid.

## Application

- (a) `quantile_least_squares`, `input_kind="probability_labelled"` (or
  `samples` for genuine block records fitted via plotting positions, or
  cyclone/inundation samples resampled to 1001 knots). `input_kind` records
  the original source data rather than an estimator's intermediate knots.
- (b) Block-extrema recipes choose plotting positions + least squares,
  `sample_mle`, or `sample_lmoments` and record it.

## Standalone L-moments estimator

`sample_lmoments` fits genuine block samples using unbiased sample L-moments,
with an explicit GEV or Gumbel family. This is an estimator, not an optimizer
initialization: record `method="sample_lmoments"` and leave `initialization`
unset. It does not use likelihood refinement, implicit family selection,
hurdle fits or quantile residual gates. All ERA5 recipes and generic block-extrema policies default to `genextreme`
with `sample_lmoments`. Unspecified fields in `BlockExtremaPolicy` inherit
the recipe's family and estimator independently. Explicit field values and
fully specified `CurveFitIngestPolicy` objects override these defaults.
L-moment fitting requires crc-framework 0.3.0 or
later; compatible published wheels do not require a local Rust backend checkout.

## Sample-fit validity and explicit fallback

Block policies require observations to lie within fitted GEV support, allowing
machine roundoff at an endpoint. Using the backend's SciPy-compatible shape `c`,
the dimensionless support margin is `1 - c * (x - location) / scale`. This avoids
division by a near-zero shape; `c=0` is the unbounded Gumbel limit.
Block fits also require finite return levels at configurable validation periods
(default 2, 5, 10, 20, 50 and 100 years). Optional physical bounds apply at those
periods, using the policy's upper or lower tail. Return-level gates apply to
both sample and quantile estimators, including the complete hurdle distribution;
the GEV observed-support check is specific to sample estimators. These settings
are persisted in fit provenance; they do not modify parameters or constrain later evaluations.

`fallback_family` explicitly enables a Gumbel L-moment attempt after a GEV
L-moment fitting or validation failure. Invalid short/constant samples are not
rescued. The fallback receives identical validity checks. This option requires
a diagnostics path: per-cell records preserve attempted and failed families,
the reason for switching and the actual fitted family. Provenance uses
`first_acceptable` with both candidate families; canonical rows retain the
selected family. An unsuccessful fallback obeys `on_fit_failure`.
