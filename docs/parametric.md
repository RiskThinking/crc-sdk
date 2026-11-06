# Parametric covers

[Docs index](README.md) · [Project README](../README.md)

`crc_sdk.parametric` defines parametric insurance covers on hazard indices: a
cover pays an amount fixed in advance whenever a measured index, such as a
year's maximum daily rainfall, reaches stated thresholds. This page covers the
parts available now: covers, payout functions and their parameters, the
outcome samples they are evaluated on, and the evaluation and comparison of
covers. Design builds on them and will be documented here when it lands.

Everything here works on numpy arrays and does no I/O, except
`write_comparison`, which writes one Parquet file.

## Define a cover and evaluate its payouts

```python
from crc_sdk.parametric import Cover, IndexDefinition, StepPayout

index = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
cover = Cover(
    index,
    StepPayout(breakpoints=[60.0, 80.0, 100.0], levels=[1e3, 5e3, 2e4]),
    payout_unit="EUR",
    asset_id="warehouse-a",
)
cover.payout_values([45.0, 60.0, 85.0, 140.0])   # array([0., 1000., 5000., 20000.])
```

- **Index:** `IndexDefinition(hazard_name, unit, *, tail="upper", location=AtAsset(), aggregation, source=None)`.
  `tail="upper"` means larger values are more severe (rainfall, wind);
  `tail="lower"` means smaller values are (a drought index), as a dataset's
  `return_period_tail` says. `location` is `AtAsset()`, `AtPoint(longitude,
  latitude)`, `AtStation(station_id)` or `AtCell(cell_index)` (an H3 cell).
- **Cover:** `Cover(index, payout, *, payout_unit, asset_id=None, label=None)`.
  The payout's `tail` must equal the index's. `label` is a display name only.

## Payout families

Each family pays nothing below its first threshold and more as the index moves
to the severe side. Thresholds are in index units and listed in order of
increasing severity: increasing for `tail="upper"`, decreasing for
`tail="lower"`. A value equal to a threshold belongs to the more severe side.
Amounts are in payout units and non-negative.

| Family | Parameters | Payout at severity `s` |
| --- | --- | --- |
| `LinearPayout(attachment, exhaustion, limit)` | thresholds a < e; amount `limit` | `limit · clip((s − a)/(e − a), 0, 1)` |
| `StepPayout(breakpoints, levels)` | thresholds b_0 < … < b_{n−1}; one level each | `levels[k]` for b_k ≤ s < b_{k+1}; the last level from b_{n−1} on |
| `PiecewiseLinearPayout(knots, levels)` | thresholds c_0 < … < c_{m−1}, m ≥ 2; one level each | the points (c_k, `levels[k]`) joined linearly; the last level from c_{m−1} on |
| `BinaryPayout(trigger, amount)` | threshold t; `amount` | `amount` for s ≥ t |

`StepPayout` and `PiecewiseLinearPayout` take `monotone=True` by default, which
requires non-decreasing levels; `monotone=False` is allowed. Every built-in
family is linear in its amounts (`LinearInAmounts`): `basis(index) @ amounts`
gives the payout, and `amount_constraints(limit)` states the amounts' bounds.

## Fixed and free parameters

Each parameter is a float (a fixed term) or `Free(lower=0.0, upper=None, *,
grid=None)`, a value a design chooses within bounds. A cover with a free
parameter is a template: it has a document and a hash, but
`payout_values` raises `UnresolvedParameterError` until it is resolved.

```python
from crc_sdk.parametric import Free

template = Cover(index, StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
                 payout_unit="EUR", asset_id="warehouse-a")
template.free_parameters()   # ('levels[0]', 'levels[1]', 'levels[2]')
resolved = template.with_values({"levels[0]": 1e3, "levels[1]": 5e3, "levels[2]": 2e4})
```

- Parameter names are flat: `attachment`, `exhaustion`, `limit`,
  `breakpoints[0]`, `levels[2]`, `knots[1]`, `trigger`, `amount`.
- A free amount needs `lower >= 0`; `upper=None` leaves it unbounded.
- A free threshold needs a finite `upper` and a `grid`: a count of evenly
  spaced values, or the values themselves. A count is stored as its values.
- Invalid parameters raise `CoverDefinitionError` (a `ValueError`); invalid
  index values raise `SampleError`.

## The cover document and its hash

`cover.to_json()` returns the cover document as canonical JSON: the index, the
payout with every parameter, `payout_unit`, `asset_id`, `schema_version` and
`semantics_version`. `cover.definition_hash` is `sha256:` plus the SHA-256 of
that text, in the canonical spelling crc-framework uses for
`ImpactExpression.definition_hash`. The label is not part of the document, so
two covers with the same terms have the same hash; `asset_id` is, so the same
terms on another asset are another contract. `Cover.from_json(text)` reads a
cover back, and covers compare, hash and pickle by their document.

## User-defined payouts

A payout family is any class that implements the `PayoutFunction` protocol
(`kind`, `tail`, `index_names`, `parameters()`, `with_values()`, `evaluate()`,
`to_document()` and `from_document()`). Register it with `@register_payout`
so `Cover.from_json` can read covers that use it; an unregistered `kind`
raises `UnknownPayoutKindError`. A family that is linear in its amounts can
also implement `LinearInAmounts`.

## Outcome samples

Covers are designed and evaluated on an `OutcomeSample`: rows (draws) with a
probability each, and for each row an index value and a loss. One draw is one
year: a cover pays once per draw, on that draw's index, against that draw's
loss. `period` is the year, or the draw number of a simulated sample. A sample
holds one asset; `index` and `loss` are `(n, 1)` arrays, and `hazard` keeps the
value the loss model read when it differs from the index.

| Source | Function | One draw is | Probability |
| --- | --- | --- | --- |
| A fitted curve | `sample_curve(curve, *, metadata, n_draws, rng, loss_model, basis_risk, asset_id)` | a simulated year | `1/n` |
| Per-year values | `sample_annual(AnnualSeries, *, loss_model, basis_risk, asset_id, index=None)` | an observed or modelled year | `1/n` |
| A user table | `OutcomeSample.from_table(table, *, period, asset_id, index, loss, semantics="annual", ...)` | a table row | its `probability` column, or `1/n` |

```python
import numpy as np
from crc_sdk.parametric import ImpactNoise, LossModel, sample_curve
from crc_sdk.impacts import PiecewiseLinearImpact
from crc_sdk.workflows import distribution_from_hazard_row

# row: one canonical curve row; metadata: its dataset's HazardDatasetMetadata
loss_model = LossModel(PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
                       value=2e6, value_unit="EUR")
sample = sample_curve(distribution_from_hazard_row(row), metadata=metadata,
                      n_draws=10_000, rng=np.random.default_rng(42),
                      loss_model=loss_model, basis_risk=ImpactNoise(0.3, upper=1.0),
                      asset_id="warehouse-a")
```

**Curves.** Each draw takes `u = rng.random(n_draws)` and `distribution.quantiles(u)`,
crc-framework's own inverse-CDF route. A draw is a year only if the curve
describes one year's value:

- `probability_semantics` must be `annual_exceedance` or
  `annual_value_distribution`; any other value raises
  `ProbabilitySemanticsError`;
- unspecified semantics raise too, unless the caller states one of the two
  with `assume_semantics=`; a stated value that contradicts the declared one
  raises;
- a hurdle curve keeps its atom (a dry year gets the atom's value), and a
  `no_data` row raises instead of becoming zero;
- draws outside the fitted support are counted: rarer than the longest fitted
  return period warns with `ReturnPeriodExtrapolationWarning`; more frequent
  than the shortest is recorded in the provenance notes. A tabulated curve
  raises outside its knots unless `outside_support="extrapolate"` or
  `"clamp"` is stated.

**Per-year values.** `AnnualSeries.from_table(table, *, year, value, where,
unit)` reads one location of an `annual_extremes()` table (a station with
`where={"station": ...}`), and `AnnualSeries.nearest_cell(table, *, longitude,
latitude, unit)` the grid cell whose centre lies within half a grid step of a
point. A year without a value raises in `sample_annual`, unless
`missing="drop_year"`, which lists the dropped years. No trend is removed.

**Loss model.** `LossModel(impact, *, value_column=None, value=None,
value_unit="1", context=None)` computes `impact(hazard) × value`. `impact` is
an `ImpactExpression`, a framework `ImpactFunction` (such as
`PiecewiseLinearImpact` or a registry impact) or a vectorised callable, as
`PortfolioEvaluation.impact(...)` takes. Without a value the loss is the
damage ratio. A value column, and `ImpactContextColumns` for registry impacts,
are read from the asset's row with `loss_model.for_asset(assets, asset_id)`; a
missing or repeated asset raises.

**Basis risk** is a required argument:

| Model | Index | Loss |
| --- | --- | --- |
| `NoBasisRisk()` | the hazard at the asset | `impact(hazard) × value`; warns with `ComonotoneIndexWarning`, as the loss is then a function of the index |
| `ImpactNoise(sigma, *, upper=None)` | the hazard at the asset | each year's impact times `exp(sigma·Z − sigma²/2)`, capped at `upper` |
| `index=` series in `sample_annual` | another cell, station or hazard | from the hazard series; the dependence is the data's own |

A user model implements the `BasisRisk` protocol (`name`, `to_document()`,
`apply(index, hazard, impact, rng)`). With one generator per call, the hazard
draws come first, then the basis-risk model.

**Splits.** `holdout(sample, train_fraction=0.7)` designs on the first
`floor(train_fraction · n)` periods and holds out the rest, so a per-year
sample is tested on its latest years. `year_block_kfold(sample, k)` holds out
each of `k` contiguous blocks in turn. Monte Carlo draws are independent, so a
fresh sample from another generator is the plainer out-of-sample test.

**Provenance.** `sample.provenance` (`SampleProvenance`) records the source
(dataset metadata with licence, attribution, version and checksum; the series
and its matched cell or station; or the table's name), the semantics and who
stated them, the loss model (impact type, name, `definition_hash`, value
column and unit, not the value itself), the basis-risk model, the generator's
state before each random step (`provenance.generator()` rebuilds it), what was
counted (draws outside the support, atom draws, dropped years) and notes for
the setup text. `sample.content_hash` is `sha256:` plus the SHA-256 of the
canonical long table with its units, semantics and scenario.
`sample.to_table()` returns that long table, with the metadata under the
Parquet key `crc.parametric.sample`.

## Evaluate and compare covers

`evaluate(cover, sample, *, split, ...)` returns a one-row table of a cover's
metrics on a sample; `compare(covers, sample, *, split, ...)` returns one row
per cover, with covers as rows and metrics as columns. A cover is a resolved
`Cover` (a template raises `UnresolvedParameterError`) or a design, which
brings its design sample and its premium. The cover's payout unit, index unit
and asset must be the sample's.

```python
from crc_sdk.parametric import (CostOfCapitalLoad, MetricOptions, compare,
                                holdout, to_long, write_comparison)

design_years, held_out = holdout(sample, train_fraction=0.7)
# cover_a and cover_b were chosen on design_years, so name it
table = compare([cover_a, cover_b], held_out, split="out_of_sample",
                design_sample=design_years, pricing=CostOfCapitalLoad(0.06, 0.99),
                options=MetricOptions(betas=(0.9, 0.99), loss_thresholds=(0.0, 1e5)))
write_comparison(table, "comparison.parquet")   # atomic
tidy = to_long(table)                           # one row per cover and metric
```

### Metrics

On an evaluation sample with draw probabilities p, loss L, payout π, the
uncovered loss U = max(L − π, 0), the residual R = L − π, the premium used P and
the net loss N = L − π + P (the owner's cost of the year with the cover):

| Column | Definition | Null when |
| --- | --- | --- |
| `expected_loss`, `expected_payout`, `expected_uncovered` | E[L], E[π], E[U] | — |
| `premium` | P, the premium used (below) | — |
| `coverage_ratio` | E[min(π, L)] / E[L]: the share of the expected loss that payouts cover; payouts above the loss don't count | E[L] = 0 |
| `payout_to_loss_ratio` | E[π] / E[L]: payouts above the loss count too | E[L] = 0 |
| `tail_coverage_b<β>` | Σ w min(π, L) / Σ w L, with w the CVaR_β weights of L: the share of the loss covered in the worst 1 − β of loss years | Σ w L = 0 |
| `cvar_uncovered_to_loss_b<β>` | CVaR_β(U) / CVaR_β(L): the tail of the uncovered loss against the tail of the uninsured loss | CVaR_β(L) = 0 |
| `false_negative_rate_l<ℓ>` | P(π ≤ ε, L > ℓ) / P(L > ℓ): years with a loss above ℓ and no payout | P(L > ℓ) = 0 |
| `false_positive_rate_l<ℓ>` | P(π > ε, L ≤ ℓ) / P(L ≤ ℓ): years with a payout and a loss of at most ℓ | P(L ≤ ℓ) = 0 |
| `rmse` | sqrt(E[R²]) | — |
| `correlation` | the p-weighted Pearson correlation of L and π | either is constant |
| `var_uncovered_b<β>`, `cvar_uncovered_b<β>` | VaR_β(U), CVaR_β(U) | — |
| `var_net_loss_b<β>`, `cvar_net_loss_b<β>` | VaR_β(N), CVaR_β(N) | — |
| `var_loss_b<β>`, `cvar_loss_b<β>` | VaR_β(L), CVaR_β(L), the same for every cover on a sample | — |
| `loss_rp<T>`, `payout_rp<T>`, `uncovered_rp<T>`, `net_loss_rp<T>` | the value exceeded with probability 1/T: the lower (1 − 1/T)-quantile. T is in years, because a draw is a year | — |

`MetricOptions(betas=(0.9, 0.95, 0.99), return_periods=(2, 5, 10, 25, 50, 100,
250), loss_thresholds=(0.0,), payout_threshold=0.0)` states β, T, ℓ and ε; no
threshold is hidden. ℓ is in loss units; ε is the payout at or below which a
year counts as "no payout". Labels write the number with the point as an
underscore, as crc-sdk's return-period columns do: β = 0.99 gives
`cvar_uncovered_b0_99`, T = 100 gives `loss_rp100`, ℓ = 0 gives `l0`. Repeated
values raise `ValueError`.

**VaR and CVaR.** `VaR_β(X)` is the smallest x with Σ_{X ≤ x} p ≥ β, where the
comparison allows 1e-10 for the rounding of the summed probabilities, so a β
exactly on a step of the distribution gives the lower value on every platform.
`CVaR_β(X) = VaR_β(X) + Σ p (X − VaR_β(X))⁺ / (1 − β)`, the mean of the worst
1 − β of the probability; it does not depend on which value the VaR picks.

**crc-framework's CVaR is a different quantity.** Its `compute_risk` gives VaR
and CVaR over the 2ⁿ branches of independent binary factor outcomes, each with
a downside probability and impact. This module's VaR and CVaR are over the
years of an outcome sample, for one continuous quantity of one cover
(uncovered loss, net loss or loss). The random variable, the outcomes and the
dependence assumptions all differ. The two never share a column and are never
compared.

### Pricing and the premium used

`PurePremium()` prices P = E[π]. `CostOfCapitalLoad(rate, q)` prices
P = (1 − rate) · E[π] + rate · CVaR_q(π), with rate in [0, 1] and q in (0, 1).
The premium a row uses is, in this order:

- the `premium=` argument of `evaluate` (`premium_basis` `"stated"`);
- for a design, the design premium: the contract price, priced on the design
  sample (`"design"`), so out-of-sample rows use the price agreed at design;
- otherwise the premium priced on the evaluation sample with `pricing`
  (`"evaluation"`).

### In and out of sample

In sample means the design sample; out of sample means held-out year blocks
(`holdout`, `year_block_kfold`), fresh Monte Carlo draws, or another sample
that shares no year with the design sample. Every row says which in `split`
(`"in_sample"` or `"out_of_sample"`) and `fold` (0 for a holdout and fresh
draws; the fold for k-fold, which must match the sample's own split).

- An out-of-sample evaluation needs the design sample: a design brings its
  own; for a cover given as is, pass `design_sample=` with the sample its
  terms were chosen on. Without one it raises `SampleError`. A design sample
  that shares a period with the evaluation sample from the same source raises
  `LeakageError`.
- In sample, a known design sample must be the evaluation sample itself.
- The metadata records how the split was made: `holdout` (with the train
  fraction), `kfold` (with k), `fresh_draws` (Monte Carlo samples from
  different generator states), `separate_sample` or `disjoint_periods`.

Published numbers come from out-of-sample rows; in-sample rows show the gap
between the two.

### The comparison table

| Column | Type | Meaning |
| --- | --- | --- |
| `cover_label` | string | The cover's label, or the first 12 hex digits of its hash |
| `cover_hash` | string | `Cover.definition_hash` |
| `design_id` | string, nullable | The design's id; null for a cover given as is |
| `asset_id` | string | The asset |
| `level` | string | `"asset"` |
| `scenario` | string | The sample's scenario, `"baseline"` by default |
| `split` | string | `"in_sample"` or `"out_of_sample"` |
| `fold` | int32 | 0 for a holdout and fresh draws; the fold for k-fold |
| `sample_hash` | string | The evaluation sample's `content_hash` |
| `premium_basis` | string | `"stated"`, `"design"` or `"evaluation"` |
| `unique`, `ties`, `degenerate_tail` | bool, int32, string; nullable | The design's non-uniqueness report; null for a cover given as is |
| one column per metric | float64, nullable | As defined above; null where undefined |

The key (`cover_hash`, `design_id`, `asset_id`, `level`, `scenario`, `split`,
`fold`) is unique; `compare` raises `ValueError` for the same cover twice.
`to_long(table)` gives the tidy form: the identity columns, then `column`,
`metric`, `parameter`, `parameter_value`, `unit` and `value`.

**Metadata.** The table's schema metadata holds one JSON document under the
key `crc.parametric.evaluation`, validated by `EvaluationMetadata` and written
with sorted keys: the crc-sdk, crc-framework and numpy versions, the platform
and the time; the setup (basis, metric options, every sample with its source,
semantics, basis-risk and loss models, and every evaluation with its split and
pricing); every metric column's metric, parameter, unit and definition; every
cover's document; the designs' records; and the samples' sources, licences
and generator states. Asset values are never written. A comparison is derived
data under its inputs' licences, which the metadata lists.

`write_comparison(table, path, *, metadata=None)` checks that the metadata
describes the table, then writes one local Parquet file through a temporary
sibling and `os.replace`, so a failed write leaves no partial file. To write
rows of several calls (in and out of sample, several folds) to one file,
concatenate the tables with `pyarrow.concat_tables` and pass
`metadata=EvaluationMetadata.merge(*(EvaluationMetadata.from_table(t) for t in
tables))`.
