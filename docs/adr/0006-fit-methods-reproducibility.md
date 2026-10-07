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
  `samples` for cyclone/inundation arrays resampled to 1001 knots).
- (b) Block-extrema recipes choose plotting positions + least squares,
  `sample_mle`, or `sample_lmoments` and record it.

## Standalone L-moments estimator

`sample_lmoments` fits genuine block samples using unbiased sample L-moments,
with an explicit GEV or Gumbel family. This is an estimator, not an optimizer
initialization: record `method="sample_lmoments"` and leave `initialization`
unset. It does not use likelihood refinement, automatic family selection,
hurdle fits or quantile residual gates. ERA5 defaults to `gumbel_r` with
`quantile_least_squares`. L-moment fitting requires crc-framework 0.3.0 or
later; compatible published wheels do not require a local Rust backend checkout.
