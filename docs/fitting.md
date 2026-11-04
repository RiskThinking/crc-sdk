# Fitting

[Docs index](README.md) · [Project README](../README.md)

The SDK fits two kinds of input. The method is always recorded in
`CurveFitProvenance.method`.

| Input | Example | Default method |
| --- | --- | --- |
| **Probability-labelled quantiles** | CDF tables, return-period depth maps, published IDF rates | `quantile_least_squares` |
| **Genuine samples** | Annual maxima from reanalysis, models or stations | `sample_lmoments` with GEV (`genextreme`) |

Do not fit return-period vectors as samples: they are quantiles, not
observations.

## Quantile tables: `fit_cdf_quantile_batches`

`fit_cdf_quantile_batches` (`crc_sdk.fitting`) canonicalizes Arrow rows of
quantiles without hazard-specific code. `CDFColumnSchema` names the columns, and
the probability axis comes from one of:

- a shared axis passed in (`probabilities=`);
- a per-row list column (`CDFColumnSchema(probabilities=...)`);
- labelled return periods (`CDFColumnSchema(return_periods=...)`), converted
  with the policy's `return_period_convention`;
- raw samples (`CDFColumnSchema(samples=...)`), sorted and resampled to
  `sample_quantile_count` probabilities.

Key `CDFCurveFitPolicy` options:

| Option | Effect |
| --- | --- |
| `family`, `fit_method` | Explicit family and estimator |
| `lower_bound` | Censor values below a physical bound (e.g. depth ≥ 0); fitted mass below it becomes a hurdle atom |
| `fit_method="censored_quantile"`, `censor_location`, `residual_scale_floor_ratio` | Left-censored likelihood for sparse return-period curves (below) |
| `short_axis_action` | Rows with fewer than four interior knots become `tabulated` by default; unlabelled or length-mismatched rows are rejected with a reason |
| `registry`, `minimum_informative_overrides` | Per-hazard eligibility: override > registry > policy default |
| `no_data_rate_threshold`, `strict` | Warn (or raise) when a hazard is mostly `no_data` |
| `diagnostics="path.parquet"` | Row-level sidecar (see [Diagnostics](#diagnostics)) |
| `schema_version` | `"1.3"` (default) or `"1.2"` for older readers |

`HazardDataset.from_table(...)` wraps this in a lazy plan; see
[bring your own data](bring-your-own-data.md).

### Censored return-period curves

Sparse flood and inundation curves often hold zeros at their lower return
periods, meaning "at or below the physical bound". Treating those zeros as
exact values or as a known atom both bias the fit. Use:

```python
CDFCurveFitPolicy(..., family="genextreme", fit_method="censored_quantile", censor_location=0.0)
```

Values at or below `censor_location` are left-censored; positive values are
noisy observations of latent quantiles. The stored curve clamps the latent fit
at the censor location and uses the hurdle representation. A three-parameter
GEV fitted to three positive knots can drive residual variance to zero, so the
optimizer enforces `residual_scale_floor_ratio × robust value scale`
(default `1e-3`) and records both values in provenance. Use an explicit
`HurdleFitPolicy` only when the source defines a known point-mass probability;
the SDK never infers one from sparse knots.

## Annual samples: block extrema

`BlockExtremaPolicy` fits per-block statistics (annual maxima, minima, k-day
sums) from ERA5, CMIP6, EDO and station adapters. `BlockExtremaCurveSource`
(`crc_sdk.connectors.blocks`) is the generic engine for any reader that yields
per-block values.

| Estimator | `fit_method` | Notes |
| --- | --- | --- |
| Sample L-moments (default) | `sample_lmoments` | Unbiased L-moments; `genextreme`, `gumbel_r`, `gumbel_l`; no likelihood refinement |
| Penalised MLE | `sample_mle` | GEV starts from L-moments |
| Plotting positions + least squares | `quantile_least_squares` | Required for quantile residual gates |

Unspecified `family`/`fit_method` fields inherit the recipe defaults
independently; explicit values override them. A `CurveFitIngestPolicy` is fully
explicit and inherits nothing. Incompatible combinations (for example residual
gates with an L-moment fit) are rejected when the plan is built.

```python
from crc_sdk.workflows import BlockExtremaPolicy

BlockExtremaPolicy.curated(h3_resolution=5)                       # GEV + L-moments
BlockExtremaPolicy.curated(family="gumbel_r")                     # Gumbel + L-moments
BlockExtremaPolicy(family="gumbel_r", fit_method="quantile_least_squares", h3_resolution=5)
```

### Validation

Sample fits must place every observation inside the fitted GEV support
(allowing round-off at an endpoint) and produce finite return levels at
`validation_return_periods` (default 2, 5, 10, 20, 50, 100 years). Optional
`minimum_return_value` / `maximum_return_value` bounds apply only at those
periods (set `minimum_return_value=0` for rainfall). They never truncate the
curve. Return-level checks also apply to quantile fits, including whole hurdle
distributions. All settings are stored in fit provenance.

### Explicit Gumbel fallback

```python
BlockExtremaPolicy.curated(
    minimum_return_value=0,
    fallback_family="gumbel_r",
    diagnostics="rx1day-fit-diagnostics.parquet",   # required with a fallback
)
```

Only a failed GEV L-moment fit, or a failed validation, triggers the fallback.
The fallback must pass the same checks, and short or constant records are never
rescued. Provenance lists both families with `selection_metric="first_acceptable"`;
each row records the family actually used.

L-moments alone do not quantify return-level uncertainty. Use resampling or
independent validation before adopting a default for an application.

## Diagnostics

`diagnostics="path.parquet"` writes one row per source row (or only
non-standard rows), published atomically after success:

`cell_index, source_id, hazard_name, horizon, pathway, outcome, family,
normalized_rmse, maximum_absolute_residual, attempted_families,
failed_families, fallback, reason, treatment`

`outcome` is `fitted`, `hurdle`, `point_mass`, `tabulated`, `no_data`,
`skipped` or `rejected`, and `reason` explains the last three. Block-extrema
adapters can also write the annual samples behind every curve
(`plan.annual_extremes()` / `write_annual_extremes`).

## Reproducibility

Fits are reproducible within a tolerance on evaluated quantiles, not bit for
bit across CPU architectures: iterative optimizers can diverge near convergence
boundaries. Provenance records `platform` and the `crc_framework_version`.
