# ADR-0006: Fit methods and reproducibility

Status: Proposed

## Context

Production fits distributions that already exist (probability-labelled
quantiles), never time series. Open adapters will fit genuine samples (annual
maxima). A prototype once ran L-moments over return-period-labelled depths as
if they were samples — wrong. ARM and x86 produce slightly different
Gumbel/GEV/tabulated allocation near optimizer convergence boundaries.

## Decision

- `CurveFitProvenance.method`: `quantile_least_squares` \| `sample_mle`, plus an
  optional `initialization` (e.g. `lmoments`), `input_kind`
  (`probability_labelled` \| `samples`) and `sample_resampling` (knot count
  when samples were resampled to a quantile grid).
- **Probability-labelled inputs** (CDF quantiles, return-period vectors)
  **always** use quantile least squares. Sample-based MLE is for genuine
  samples only.
- Genuine samples may use crc-framework's `fit`/`fit_candidates` (GEV starts
  from L-moments/PWM, refined by penalised MLE), or plotting positions +
  least squares. The choice is recorded.
- **Reproducibility is a tolerance on evaluated quantiles, not bit identity**
  across architectures. Provenance records `platform` and `crc_framework`
  version. The tolerance policy is the publication gate.
- Open a crc-framework issue only if the batch API lacks something.

## Alternatives rejected

- Bit-identical output across architectures: not achievable with iterative
  optimisers near convergence boundaries.
- Fit return-period vectors as samples: statistically invalid.

## Application

- (a) `quantile_least_squares`, `input_kind="probability_labelled"` (or
  `samples` for cyclone/inundation arrays resampled to 1001 knots).
- (b) Block-extrema recipes (Phase 1B) choose plotting position + LS or
  `sample_mle` and record it.
