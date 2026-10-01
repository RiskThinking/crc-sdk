# ADR-0003: Probability semantics

Status: Proposed

## Context

A canonical curve is a quantile function, but what its probability axis *means*
differs by source: an annual-maximum return level, the distribution of an
annual indicator across years and members, a within-period percentile, or
ensemble spread. Evaluating "the 100-year value" on the wrong kind is
silently meaningless. The upstream flood labels also mix `1/T` and the Poisson
`1 - exp(-1/T)` conventions, and the ambiguity is unresolved.

## Decision

Add an optional `probability_semantics` metadata field. **Absent means
`unspecified`**: current behaviour, no warnings.

| value | meaning |
|---|---|
| `annual_exceedance` | block-maximum return levels, return-period maps |
| `annual_value_distribution` | distribution of an annual indicator across years × members within a window (what the internal CDFs are); return-period evaluation allowed, interpretation documented |
| `within_period_percentile` | e.g. a daily 99th percentile; **not** a return period |
| `projection_uncertainty` | e.g. AR6 sea-level quantiles, ensemble percentiles |
| `estimate_confidence` | confidence bounds on a return level |

Return-period evaluation on `within_period_percentile` or
`projection_uncertainty` **warns** by default and **refuses** under
`strict=True`.

### T ↔ p convention

Add an optional `source_return_period_convention`:

- `one_minus_inverse`: `p = 1 - 1/T` (non-exceedance, the SDK default)
- `poisson`: `p = exp(-1/T)` (annual exceedance `1 - exp(-1/T)`)

Ingest **normalises explicitly** to canonical non-exceedance probabilities and
records the convention the source used. `return_periods_to_probabilities`
takes the convention as an argument; the tail (`upper`/`lower`) is separate.
For `T >= ~10` the two differ by under 5% in probability but not negligibly in
the tail, so the choice is always stated, never assumed.

## Alternatives rejected

- Infer semantics from `value_semantics` text: free text, not machine-checkable.
- Make it required: breaks 1.0–1.2 files and open-ended bring-your-own data.

## Application

- (a) Internal CDFs: `annual_value_distribution`. Flood RP-labelled depths:
  `annual_exceedance` with the convention resolved during the flood-label
  investigation (the metadata field records the answer once known).
- (b) Open recipes declare it per recipe (block maxima → `annual_exceedance`;
  daily percentile indices → `within_period_percentile`; CMIP6 ensemble
  summaries → `projection_uncertainty`).
