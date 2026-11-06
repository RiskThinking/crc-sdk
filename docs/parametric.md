# Parametric covers

[Docs index](README.md) · [Project README](../README.md)

`crc_sdk.parametric` defines, designs, evaluates and compares parametric
insurance covers on hazard indices. A parametric cover pays an amount fixed in
advance whenever a measured index, such as a year's maximum daily rainfall,
reaches stated thresholds, whatever the loss turns out to be. The gap between
payout and loss is the basis risk.

There are two ways to use the module:

- **The plan.** `ParametricPlan.for_evaluation(request)` takes the portfolio
  evaluation you build with `HazardDataset.local(path).for_assets(assets)
  .select(...).impact(...)`, draws simulated years from each asset's fitted
  hazard curve, and chains the steps to a written comparison. It is a lazy
  crc-sdk plan: builders and `explain()` do no I/O, and the data is read only
  at the execution points `samples()`, `designs()` and `write_parquet()`. The
  same plan also runs on per-year values, such as ERA5's annual extremes
  ([next step](#next-step-per-year-values)).
- **The functions.** Every step is also a function on numpy arrays and Arrow
  tables that does no I/O (`sample_curve`, `sample_annual`, `design`,
  `compare`, ...), for your own tables, fitted curves and chains. Only
  `write_comparison` writes a file.

## Concepts

| Concept | Name | What it is |
| --- | --- | --- |
| Cover | `Cover` | The contract terms: an index, a payout function with its parameters, the payout unit and the asset |
| Index | `IndexDefinition` | What the cover pays on: hazard name, unit, severe tail, location, aggregation |
| Payout function | `LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`, `BinaryPayout`, or your own | Maps a year's index value to its payout |
| Template | a `Cover` with `Free` parameters | Terms a design chooses within stated bounds |
| Outcome sample | `OutcomeSample` | Years (or simulated years), each with a probability, an index value and a loss |
| Loss model | `LossModel` | loss = impact(hazard) × asset value |
| Basis-risk model | `NoBasisRisk`, `ImpactNoise`, or an index series read elsewhere | How the index and the loss differ |
| Objective | `MSE`, `ExpectedUncovered`, `WeightedCVaR`, `UserObjective` | What a design minimises |
| Pricing and constraints | `PurePremium`, `CostOfCapitalLoad`, `Budget`, `Limit` | The premium of a payout, and what a design must respect |
| Design | `design`, `DesignResult` | A template's resolved cover and its design record |
| Comparison | `compare`, a `pyarrow.Table` | Covers as rows, metrics as columns, in and out of sample |
| Plan | `ParametricPlan`, `ParametricResult` | The lazy workflow from a portfolio evaluation's curves (or per-year values) to a written comparison |

"Impact" is crc-sdk's impact value, usually a damage ratio; "loss" is impact ×
value, in the payout's unit; "hazard" is the value the loss model reads at the
asset; "index" is the value the cover pays on. Without a basis-risk model,
index and hazard are the same value. "Portfolio" keeps crc-sdk's meaning, a
set of assets.

## Quick start

Design a step cover on 5,000 years simulated from a fitted rainfall curve,
under three objectives, and compare the designs with a fixed linear cover on
5,000 fresh draws. The canonical dataset (one Gumbel curve in the asset's H3
cell) and the asset are an illustration, not data. The designs need the
optional extra `crc-sdk[parametric]` (the HiGHS solver), and `point_to_cell`
needs `crc-sdk[geometry]`.

```python
import pyarrow as pa
from crc_framework import PiecewiseLinearImpact
from crc_sdk.connectors import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.geometry import point_to_cell
from crc_sdk.parametric import (
    MSE, Budget, Cover, ExpectedUncovered, Free, FreshDraws, ImpactNoise,
    IndexDefinition, LinearPayout, MetricOptions, ParametricPlan, StepPayout,
    WeightedCVaR, setup_text,
)
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import HazardDataset

# A synthetic canonical dataset of annual maximum daily rainfall. Real curves
# come from a source plan: HazardDataset.crc_open(...)...materialize(path).
metadata = HazardDatasetMetadata(
    h3_resolution=7, value_unit="mm/day",
    value_semantics="annual maximum daily rainfall",
    probability_semantics="annual_exceedance",   # the curve of an annual maximum
    return_period_support=(2.0, 500.0),          # the fitted range
    producer="example", creation_version="1",
    source=SourceProvenance(provider="example", dataset="synthetic rx1day",
                            licence="CC0-1.0", attribution="Synthetic data"),
)
curve = {
    "cell_index": point_to_cell(6.96, 50.94, 7), "source_id": "example",
    "source_geometry": None, "hazard_name": "rx1day", "horizon": 2050,
    "pathway": "ssp245", "curve_kind": "fitted", "curve_type": "gumbel_r",
    "curve_shape": None, "curve_location": 50.0, "curve_scale": 12.0,
    "curve_atom_probability": None, "curve_atom_location": None,
}
write_hazard_dataset(pa.Table.from_pylist([curve], schema=hazard_arrow_schema()),
                     "rx1day.parquet", metadata)

assets = pa.table({"asset_id": ["warehouse-a"], "longitude": [6.96],
                   "latitude": [50.94], "value_eur": [2e6]})
request = (
    HazardDataset.local("rx1day.parquet")
    .for_assets(assets)
    .select(hazard_names=["rx1day"], horizons=[2050], pathways=["ssp245"])
    .impact(PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),   # damage ratio
            name="rx1day_damage", value_unit="1", value_semantics="damage ratio")
)

index = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
step = Cover(index, StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
             payout_unit="EUR", label="step")
linear = Cover(index, LinearPayout(60.0, 120.0, 5e5), payout_unit="EUR",
               label="linear")

plan = (
    ParametricPlan.for_evaluation(request, value_column="value_eur",
                                  value_unit="EUR")
    .monte_carlo(5_000, seed=42)               # simulated years from the curve
    .basis_risk(ImpactNoise(0.3, upper=1.0))   # stated, never assumed
    .covers(step, linear)                      # a template and a fixed cover
    .objective(MSE(), ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9),
               budget=Budget(ratio=0.8))
    .split(FreshDraws(5_000, seed=7))          # evaluate on fresh draws
    .metrics(MetricOptions(betas=(0.9,), return_periods=(10,)))
)
print(setup_text(plan.explain(format="json")["setup"]))   # the setup; no I/O

result = plan.write_parquet("comparison.parquet")   # execution; atomic
```

`plan.explain()` gives the whole plan as JSON text; `setup_text` states its
setup in plain sentences. `result.table` is the comparison as written: one row
per cover (the three designs, labelled `"step (mse)"`,
`"step (expected_uncovered)"` and `"step (weighted_cvar alpha=0.9 beta=0.9)"`,
and the linear cover) and split. The out-of-sample rows, on the fresh draws,
are the ones to report, each with its setup:

```python
import pyarrow.compute as pc

table = result.table
out_of_sample = table.filter(pc.equal(table["split"], "out_of_sample"))
out_of_sample.select(["cover_label", "premium_basis", "premium",
                      "expected_uncovered", "coverage_ratio", "tail_coverage_b0_9",
                      "unique"])
print(setup_text(result.metadata))   # the setup of these rows, to print with them
[design.unique for design in result.designs]   # is each design the only optimum?
```

The designs' rows use the design premium (`premium_basis` `"design"`), the
linear cover's rows the premium of its payouts on the fresh draws
(`"evaluation"`). The setup text also states what the curve's fitted range
means for these draws: with a shortest fitted return period of 2 years, about
half of the simulated years are more frequent than it, where the curve is
extrapolated. Any
number read from this table describes this synthetic curve under this setup
and nothing more.

## The parametric plan

```text
ParametricPlan.for_evaluation(request, *, value_column=None, value_unit="1")
  .monte_carlo(n_draws, *, seed, assume_semantics=None, outside_support="raise")
ParametricPlan.for_annual_extremes(source, assets, *, impact, unit, provenance, ...)
# then, on either plan:
  .basis_risk(model)                          # required
  .covers(*covers)                            # resolved covers and templates
  .objective(*objectives, pricing=PurePremium(), budget=None, limit=None,
             options=DesignOptions())         # to design the templates
  .split(FreshDraws(n_draws, seed=...) | Holdout(train_fraction) | KFold(k))
  .metrics(MetricOptions(...))                # optional
  .explain(format="text")                     # no I/O
  .samples() / .designs() / .write_parquet(path)   # execution points
```

### Fitted curves

- **`request`** is the `PortfolioEvaluation` built with
  `HazardDataset.local(path).for_assets(assets).select(...).impact(...)`, the
  request whose `write_parquet` evaluates the assets at return periods. The
  plan reuses its dataset, assets, selection and impact and changes nothing in
  it. Materialize a remote source first: a source plan's `materialize(path)`
  returns the `HazardDataset` to build the request on.
- **The curves.** At execution, `request.asset_curves()` joins each asset to
  its canonical curve as `write_parquet` does (the asset's H3 cell, refined by
  exact geometry where the source has it); an asset without a curve, or with
  several for one scenario, raises. The plan samples one curve per asset, so
  the selection must leave one hazard, horizon and pathway; several raise
  `SampleError`.
- **`.monte_carlo(n_draws, *, seed, assume_semantics=None,
  outside_support="raise")`** is required: each asset's sample is `n_draws`
  simulated years drawn from its curve by `sample_curve`, each with
  probability `1/n_draws`. The rules of [outcome samples](#outcome-samples)
  apply. The dataset's `probability_semantics` must be `annual_exceedance` or
  `annual_value_distribution`; a dataset that declares none is refused unless
  `assume_semantics=` states one of the two, and the setup then says that the
  caller stated it. A `no_data` curve raises. Draws beyond the fitted support
  are counted and named in the setup, and so is a pooled ensemble.
  `outside_support` is what a tabulated curve does with draws outside its
  knots.
- **The loss** is `impact(hazard) × value`: the request's impact (an
  `ImpactExpression` keeps its `definition_hash`) times the asset column
  `value_column` in `value_unit`, or 1 with `value_unit="1"` (the damage
  ratio). A framework impact gets the context `write_parquet` gives it: its
  own, with the asset's matched H3 cell and the request's context columns on
  top, so a registry impact such as `rx1day` needs no cell of its own.
- **The index** is the curve's hazard at the asset, in the dataset's
  `value_unit`: each cover's index must be in that unit, which is checked at
  execution.
- **The assets** are an in-memory table or a local Parquet file, read at
  execution, in row order. Asset values are read for the loss and never
  written.

### Builders

| Builder | What it states |
| --- | --- |
| `.monte_carlo(n_draws, *, seed, ...)` | Fitted curves only, and required there: the simulated years per asset, the plan's seed, the semantics the caller states for a dataset that declares none, and `outside_support` |
| `.basis_risk(model)` | How index and loss differ: `NoBasisRisk()` (warns with `ComonotoneIndexWarning`), `ImpactNoise(sigma, upper=...)` or a `BasisRisk` of your own. Required: no model is assumed |
| `.covers(*covers)` | The covers to compare. A template is designed; a resolved cover is evaluated as it is. A cover written for no asset applies to every asset, as a copy written for that asset; one written for an asset applies to it only. Each cover pays in the loss unit, on an index read at the asset (`AtAsset()`), in the hazard's unit |
| `.objective(*objectives, pricing, budget, limit, options)` | Designs every template under each objective, with `design`'s terms. `pricing` also prices the resolved covers; without `.objective(...)` they are priced with `PurePremium()` |
| `.split(rule)` | `FreshDraws(n_draws, seed=...)` (fitted curves), `Holdout(train_fraction)` or `KFold(k)`, below |
| `.metrics(options)` | The `MetricOptions` of every row |

A designed cover keeps the template's label (or the first 12 hex digits of its
hash) followed by the objective, for example `"step (mse)"`; its hash and
design id identify it.

### Splits

- **No split:** every cover is designed and evaluated on the whole sample,
  and every row is in sample.
- **`FreshDraws(n_draws, seed=...)`** (fitted curves): designs on the plan's
  draws, evaluation on `n_draws` fresh draws from the same curve. Draws are
  independent simulated years, so this is the plain out-of-sample test. Its
  seed must differ from the plan's.
- **`Holdout(train_fraction)`:** each asset's sample is split by `holdout`:
  designs on the first `floor(train_fraction · n)` periods (years, or draws),
  evaluation on the rest, so a per-year sample is tested on its latest years.
- **`KFold(k)`:** `year_block_kfold` holds out each of `k` contiguous blocks of
  periods in turn; every fold has its own designs.

Every design block gets in-sample rows and every held-out block, or the fresh
draws, out-of-sample rows (`split`, `fold`). Resolved covers are evaluated on
the same blocks, with the design block as their design sample: their terms are
taken as chosen without the held-out data. The leakage checks of `compare` run
on every block.

### Seeding

The plan holds an integer seed: `monte_carlo`'s on fitted curves,
`for_annual_extremes(..., seed=...)`'s on per-year values. At execution it
builds `numpy.random.SeedSequence(seed)` and spawns one child generator per
asset, in asset order, so adding an asset at the end leaves the others'
samples as they were. Each child draws the hazard first, then the basis-risk
model. Fresh draws spawn their children from `SeedSequence(FreshDraws.seed)`
in the same way. On per-year values only the basis-risk model draws, so
`ImpactNoise` needs a seed there. Each sample's provenance records its
generator's state before each draw.

### Explain, execute, write

`explain()` returns the plan's setup as far as it is known before execution,
as JSON text (`format="json"` gives a dict), without I/O. It mirrors the
metadata the plan writes: `setup` (the basis, the metric options, one sample per
asset with its loss and basis-risk models, the planned evaluations with their
split, fold and pricing), `covers`, `designs`, `sources`, `assets`, `licences`,
`rng` and `entry`. On fitted curves, `monte_carlo` lists the draws, the
semantics the caller states and the fresh draws; the dataset's own metadata
(its units, semantics and licence) is read only at execution, so `licences` is
`None` there.

| Execution point | Returns |
| --- | --- |
| `.samples()` | One `OutcomeSample` per asset, all draws or years, in asset order |
| `.designs()` | Every `DesignResult`, in asset, fold, cover and objective order |
| `.write_parquet(path)` | A `ParametricResult`: `output`, `table`, `metadata`, `samples`, `designs` and `row_count` |

`write_parquet` writes the rows of every asset, split and fold to one local
Parquet file through a temporary sibling and `os.replace`, and creates the
parent directory. Its metadata, under `crc.parametric.evaluation`, merges that
of every block (below) and adds to each hazard source the source's record: on
fitted curves `canonical_file` (the dataset's path, its SHA-256 and the number
of matched curves; each sample also records its curve row's key and spatial
match), on per-year values `annual_table` (its rows, columns and SHA-256, the
source plan's `explain()`, and for a file its path and SHA-256). Every
evaluation's `details` holds the plan's seed (`"plan_seed"`), every fresh-draw
evaluation's also the fresh draws' seed (`"fresh_draws_seed"`), and `assets`
the count, the id column and a SHA-256 of the (asset id, value) pairs.

## Next step: per-year values

The same plan runs on per-year values: `ParametricPlan.for_annual_extremes`
reads the years of an annual-extremes table (ERA5, CMIP6, NEX-GDDP-CMIP6, a
station, NOAA water levels) at each asset. Each row is a recorded or modelled
year with probability `1/n`, so no curve and no probability semantics are
assumed, and no trend is removed. Here the quick start's covers and asset are
designed on 28 years of a synthetic table and compared on the 12 years held
out:

```python
import numpy as np
from crc_sdk.parametric import Holdout

# The layout of ERA5's and CMIP6's annual_extremes(): one row per year and cell.
rng = np.random.default_rng(1)
years = np.arange(1981, 2021)
annual = pa.table({
    "year": np.repeat(years, 4).astype("int32"),
    "latitude": np.tile([50.875, 50.875, 51.125, 51.125], len(years)),
    "longitude": np.tile([6.875, 7.125, 6.875, 7.125], len(years)),
    "value": rng.gumbel(50.0, 12.0, 4 * len(years)),       # mm/day
})
provenance = SourceProvenance(provider="example", dataset="synthetic rx1day",
                              licence="CC0-1.0", attribution="Synthetic data")

annual_plan = (
    ParametricPlan.for_annual_extremes(
        annual, assets,
        impact=PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),   # damage ratio
        unit="mm/day", value_column="value_eur", value_unit="EUR",
        provenance=provenance, seed=42,
    )
    .basis_risk(ImpactNoise(0.3, upper=1.0))
    .covers(step, linear)
    .objective(MSE(), ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9),
               budget=Budget(ratio=0.8))
    .split(Holdout(0.7))                       # design on 1981-2008, test on 2009-2020
    .metrics(MetricOptions(betas=(0.9,), return_periods=(10,)))
)
annual_result = annual_plan.write_parquet("annual-comparison.parquet")
```

The builders, splits, seeding and outputs are those above.

### Source and assets

- **`source`** is a plan with an `annual_extremes()` execution point (the ERA5
  and CMIP6 canonicalization plans, NEX-GDDP-CMIP6, the station plans, NOAA
  water levels), an Arrow table in one of their layouts, or a local Parquet
  file of one (for example from a station plan's `write_annual_extremes`). It
  is read once per execution.
- **`assets`** is an `AssetPortfolio`, an in-memory table it accepts, or a
  local Parquet file read at execution. The asset order is the table's row
  order. Asset values are read for the loss and never written.
- **`location`** finds each asset's series. `NearestCell()` reads per-cell
  tables (`year`, `latitude`, `longitude`, `value`): the cell whose centre is
  nearest the asset, accepted only within half a grid step in latitude and
  longitude (`AnnualSeries.nearest_cell`); the assets need longitude and
  latitude. `StationColumn("station_id")` reads per-station tables: the asset
  column names each asset's station, matched against the table's `station`
  column (`table_column="station_id"` for NOAA water levels). An asset that
  cannot be located raises; none is dropped.
- **`unit`** is the unit of the table's values. It is checked against the
  impact's calibration (an `ImpactExpression` or a registry impact), against a
  `unit` column of the table, and against the source plan's `explain()`.
- **`provenance`** is a `SourceProvenance` with the source's licence,
  attribution and version, which annual-extremes tables do not carry (for a
  plan that also materializes curves, `HazardDataset.local(path).provenance()`
  of its canonical file). It identifies the series of every sample, so give
  one source the same provenance each time: the leakage checks then link its
  blocks.
- **The loss model** is `LossModel(impact, value_column=..., value_unit=...,
  context=...)`: loss = `impact(hazard) × value`, the value from an asset
  column, or 1 with `value_unit="1"` (the damage ratio). A registry impact
  that needs an H3 cell in its context (such as `rx1day`) raises unless the
  impact, or `context`, is a `TransformContext` with the cell: this route reads
  no cell for an asset.
- **`years`** states the years every series must cover (by default the years
  the table holds); a year without a value raises, unless `missing="drop_year"`
  drops it and lists it. No trend is removed.

## The function route

The plan's steps are functions you can chain by hand, for a table, a curve or
an index of your own.

### From a fitted curve

The quick start's plan, by hand, for its one asset: `request.asset_curves()`
gives the asset's canonical curve row, `sample_curve` draws simulated years
from it, and fresh draws from another generator are the out-of-sample test.
With the plan's generators for its first asset, the samples are the plan's:

```python
import numpy as np
from crc_sdk.parametric import (LossModel, compare, design, sample_curve,
                                write_comparison)

row = request.asset_curves().to_pylist()[0]   # one row per asset and scenario
metadata = request.dataset.metadata()          # units, tail, semantics, licence
loss_model = LossModel(PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
                       value=2e6, value_unit="EUR")

def draws(seed: int):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return sample_curve(row, metadata=metadata, n_draws=5_000, rng=rng,
                        loss_model=loss_model,
                        basis_risk=ImpactNoise(0.3, upper=1.0),
                        asset_id="warehouse-a")

design_draws, fresh_draws = draws(42), draws(7)
designed = design(step, design_draws, ExpectedUncovered(), budget=Budget(ratio=0.8))
table = compare([designed, linear], fresh_draws, split="out_of_sample",
                design_sample=design_draws)   # the fixed cover needs its design sample
write_comparison(table, "by-hand.parquet")   # atomic
```

`curve` may also be a crc-framework distribution, for example
`distribution_from_hazard_row(row)`, or `CurveParameters`. A draw is a year
only if the curve describes one year's value: its `probability_semantics`
must be `annual_exceedance` or `annual_value_distribution` ([outcome
samples](#outcome-samples)).

### From per-year values

The per-year plan, by hand, for its one asset. With the plan's generator for
its first asset, the sample is the plan's:

```python
from crc_sdk.parametric import AnnualSeries, holdout, sample_annual

series = AnnualSeries.nearest_cell(annual, longitude=6.96, latitude=50.94,
                                   unit="mm/day", source=provenance)
rng = np.random.default_rng(np.random.SeedSequence(42).spawn(1)[0])
sample = sample_annual(series, loss_model=loss_model,
                       basis_risk=ImpactNoise(0.3, upper=1.0),
                       asset_id="warehouse-a", rng=rng)
design_years, held_out = holdout(sample, train_fraction=0.7)
designed = design(step, design_years, ExpectedUncovered(), budget=Budget(ratio=0.8))
table = compare([designed, linear], held_out, split="out_of_sample",
                design_sample=design_years)
write_comparison(table, "by-hand-annual.parquet")   # atomic
```

`AnnualSeries.from_table(table, where={"station": ...}, unit=...)` reads a
station instead, and `sample_annual(..., index=other_series)` reads the index
at another cell, station or hazard than the loss.

## Reporting numbers

- Report out-of-sample rows (`split == "out_of_sample"`): held-out years, or
  fresh draws from a curve. In-sample rows measure the fit on the design years
  and show the gap.
- State the setup next to every number: the asset, the basis (one draw is one
  year), the basis-risk model, the pricing and budget, and the split
  (below).
- A design is one member of a set of optimal designs unless `unique` is true
  ([non-unique optima](#non-unique-optima)).
- A comparison holds for its stated setup. The module makes no claim that one
  objective, payout family or cover performs better in general.

### Stating the setup

Every number in a comparison rests on a setup: the asset, the basis, the
split, the samples' source and basis-risk model, the pricing and, for a design,
its objective and constraints. `setup_text(result.metadata)`, or
`setup_text(EvaluationMetadata.from_table(table))` for a table read back,
states that setup in plain text, one statement per line (`Level`, `Split`,
`Sample`, `Sample notes`, `Pricing`, `Design`, `Metrics`), for a notebook or a
report to print next to the numbers. It reads only the metadata, so the same
table always gives the same text, and statements that every split or sample
shares are written once. Before execution, `setup_text(plan.explain(
format="json")["setup"])`, or `setup_text(metadata.setup)`, gives the same
lines without the design records, and says what only execution knows (the
years of each block, the leakage check) is not known yet. Out-of-sample rows'
`Split` line says how the split was made and that the leakage check passed.
The plan's seed is in every evaluation's `details` under `"plan_seed"`, and
the fresh draws' seed under `"fresh_draws_seed"`.

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
missing or repeated asset raises. An `ImpactExpression` declares the hazard
unit it is calibrated for (`loss_model.exposure_unit`). Before drawing,
`sample_curve` and `sample_annual` check the curve's `value_unit` or the hazard
series' unit against it and raise `SampleError` when they differ (compared as
`check_exposure` compares units, through the framework's aliases);
`loss_model.check_hazard_unit(unit)`, or `hazard_unit=` in `evaluate`, is the
same check by hand. Other impacts declare no unit and are not checked.

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
Parquet key `crc.parametric.sample`, and `OutcomeSample.from_written_table(table)`
reads it back, in memory or from `pyarrow.parquet.read_table`, with its
provenance and lineage; a table whose rows no longer match the recorded
content hash raises `SampleError`.

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
values raise `EvaluationError`, a `ValueError`.

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
`fold`) is unique; `compare` raises `EvaluationError` for the same cover
twice. Invalid evaluation arguments, metric options and comparison metadata
raise `EvaluationError`, a `ValueError`; arguments of the wrong type raise
`TypeError`.
`to_long(table)` gives the tidy form: the identity columns, then `column`,
`metric`, `parameter`, `parameter_value`, `unit` and `value`.

**Metadata.** The table's schema metadata holds one JSON document under the
key `crc.parametric.evaluation`, validated by `EvaluationMetadata` and written
with sorted keys: the crc-sdk, crc-framework and numpy versions, the platform
and the time; the setup (basis, metric options, every sample with its source,
semantics, basis-risk and loss models, and every evaluation with its split and
pricing); every metric column's metric, parameter, unit and definition; every
cover's document; the designs' records, each with its cover's label as
evaluated (`cover_label`), so two designs that resolve to the same cover keep
their own labels; and the samples' sources, licences
and generator states. Asset values are never written. A comparison is derived
data under its inputs' licences, which the metadata lists.

`write_comparison(table, path, *, metadata=None)` checks that the metadata
describes the table, then writes one local Parquet file through a temporary
sibling and `os.replace`, so a failed write leaves no partial file. To write
rows of several calls (in and out of sample, several folds) to one file,
concatenate the tables with `pyarrow.concat_tables` and pass
`metadata=EvaluationMetadata.merge(*(EvaluationMetadata.from_table(t) for t in
tables))`.

## Design a cover

`design(template, sample, objective, *, pricing=PurePremium(), budget=None,
limit=None, options=DesignOptions())` chooses a template's `Free` parameters
on a design sample and returns a `DesignResult`: the resolved cover and its
design record. The template must be written for the sample's asset (or for
none) and pay in the sample's loss unit on an index in its index unit.

```python
from crc_sdk.parametric import (Budget, CostOfCapitalLoad, DesignOptions, Free,
                                StepPayout, WeightedCVaR, design, evaluate, holdout)

design_years, held_out = holdout(sample, train_fraction=0.7)
template = Cover(index, StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
                 payout_unit="EUR", asset_id="warehouse-a")
result = design(template, design_years, WeightedCVaR(alpha=0.9, beta=0.95),
                pricing=CostOfCapitalLoad(0.06, 0.99), budget=Budget(ratio=0.8),
                options=DesignOptions(ranges=True))
result.cover, result.parameters, result.objective_value, result.premium
evaluate(result, held_out, split="out_of_sample")   # priced at the design premium
```

The design needs the optional extra `crc-sdk[parametric]` (the HiGHS solver,
`highspy`) on its linear and quadratic routes; without it they raise
`SolverUnavailableError`, an `ImportError` that names the extra. Importing
`crc_sdk.parametric` loads no solver.

### Objectives

On the design sample, with probabilities p, loss L, payout π and the uncovered
loss U = max(L − π, 0):

| Objective | Minimises | Solved as |
| --- | --- | --- |
| `MSE()` | Σ p (L − π)²: paying more than the loss counts as much as paying less | a QP in the amounts |
| `ExpectedUncovered()` | E[U] = Σ p U: paying more than the loss earns nothing | an LP |
| `WeightedCVaR(alpha, beta)` | α · CVaR_β(U) + (1 − α) · E[U], α ∈ [0, 1], β ∈ (0, 1) | an LP, the CVaR in the Rockafellar–Uryasev form min_z z + Σ p (U − z)⁺ / (1 − β) |
| `UserObjective(function, *, name, parameters=None)` | `function(loss, payout, probability)`, a float | a grid search, evaluated in numpy |

The objectives weigh the years differently: MSE penalises payouts that miss
the loss in either direction, E[U] counts every year's shortfall alike, and
the weighted CVaR adds weight on the worst 1 − β of the shortfall. Which suits
a cover depends on what its owner needs; none is better in general. A user
objective's `name` and `parameters` go into the record and the design id; its
function does not, so give each function its own name.

### Pricing and constraints

- The premium is priced on the design sample with `pricing`: `PurePremium()`
  or `CostOfCapitalLoad(rate, q)`. The cost-of-capital premium is convex in
  the payout, so the design stays an LP or a QP; its CVaR enters in the
  Rockafellar–Uryasev form too.
- `Budget(ratio=k)` means premium ≤ k · E[L] on the design sample;
  `Budget(amount=B)` means premium ≤ B, in payout units. Exactly one of the two.
- `Limit(amount)`: no payout exceeds `amount`. For the built-in families it
  bounds every amount (their payout is at most their largest amount).
- The payout family's own constraints hold: amounts ≥ 0, non-decreasing levels
  under `monotone=True`, and every `Free` parameter's bounds.
- E[U] and the weighted CVaR need a budget or bounded amounts (a `Free` upper
  bound or a `Limit`): without either, any payout at least as large as the loss
  is optimal, and `design` raises `DesignError` before solving. MSE needs
  neither.

### Routes

| Template | Objective | Route |
| --- | --- | --- |
| amounts free, thresholds fixed (any built-in family, or a `LinearInAmounts` user family) | E[U], weighted CVaR | one LP (`route="lp"`) |
| the same | MSE | one QP in the amounts (`route="qp"`) |
| free thresholds on grids (attachment, exhaustion, breakpoints, knots, trigger) | any built-in | a grid over the thresholds, with the LP or QP at each point (`route="grid"`) |
| a `UserObjective`, or a payout not linear in its amounts | any | a grid over every free parameter, evaluated in numpy; candidates that break the budget or the limit are dropped |

- **The grid.** Every parameter the grid searches needs `Free(..., grid=...)`.
  The candidates are the Cartesian product of the grids, in lexicographic
  order; unordered thresholds and non-monotone levels are skipped.
  `DesignOptions(max_grid_points=10_000)` caps the product; a larger grid
  raises `DesignError`. On a finite sample the objective is piecewise constant
  or piecewise smooth in the thresholds, so a stated grid is used rather than
  gradients. `DesignResult.profile` is a `pyarrow.Table` with one row per
  candidate: its values, `objective` and `status` (`"optimal"` or `"feasible"`,
  `"infeasible"`, `"invalid"`, `"over_budget"`, `"over_limit"`).
- **Scaling.** Losses and payouts are divided by the largest loss of a draw (1
  if that is at most 1e-6) before solving; results are in payout units.
- **Solver settings.** HiGHS's simplex for the LPs (a vertex solution), its QP
  solver with a Hessian regularisation of 1e-12 for the QPs. The record holds
  the solver's name and version, the options, the status, the iterations and
  the solver time.

### Checks and errors

After every solve numpy recomputes the payout, the objective, the premium, the
budget, the limit and the family's constraints from the returned values, with
the relative tolerance `DesignOptions.feasibility_tolerance` (1e-7, relative to
the larger of the bound and the scale). Nothing is returned that fails them.

| Error | When |
| --- | --- |
| `InfeasibleDesignError` | no payout meets the constraints, including a fixed amount or a lower bound above the limit, and a grid without a feasible candidate |
| `UnboundedDesignError` | the solver reports an unbounded objective |
| `DesignError` | any other solver status, a failed check, a missing grid, a grid above the cap, E[U] or weighted CVaR without a budget or bounds |
| `SolverUnavailableError` | the LP or QP route without `crc-sdk[parametric]` |

The design errors are `RuntimeError`s and carry `asset_id`, `status` and
`solver_message`. Free amounts below 1e-9 of the scale are set to 0 and listed
in `DesignResult.zeroed`, so a rounding residue does not count as a payout.

### Non-unique optima

A design sample often leaves part of a design open, and the result says so
rather than presenting one solver's answer as the design.

- **Unidentified amounts.** An amount whose payout band no design year reaches
  changes nothing on the sample: not the objective, the premium or the budget.
  `DesignResult.unidentified` lists them, and `DesignOptions(fill="lowest")`,
  the default, sets each to the smallest value the constraints allow (under
  `monotone`, the previous level; else its lower bound); `fill="highest"` takes
  the largest, and raises without a bound above. Both keep the design exactly
  optimal, but out of sample a year can fall into such a band, and then the
  rule sets its payout.
- **The LP route.** With `DesignOptions(ranges=True)` every identified amount
  is ranged over the near-optimal designs, those within ε · max(1, |optimum|)
  of the optimum in scaled units, at ε = 1e-9 and ε = 1e-11 (four more LPs per
  amount, at HiGHS feasibility tolerances of 1e-10). An amount is pinned when
  its narrow range is at most half its wide one, or narrower than 1e-7 of the
  scale; `unique` is true when every identified amount is pinned, and
  `DesignResult.ranges` gives the wide ranges in payout units. At a unique
  vertex a range shrinks with ε; on a set of optimal designs it keeps the set's
  width. Without `ranges=True`, `unique` is `None` on the LP route.
- **The QP route.** `unique` is true when H = Bᵀ diag(p) B, over the identified
  amounts, is positive definite (smallest eigenvalue above 1e-10 of the
  largest).
- **The grid.** `ties` counts the other candidates whose objective lies within
  `feasibility_tolerance` · max(1, |best|) of the best (in scaled units; in the
  objective's own units for a user objective); the first in lexicographic order
  wins. `ties` is 0 on the LP and QP routes. With ties, `unique` is false;
  otherwise it is the LP's or QP's report at the chosen candidate, and `None`
  for a grid evaluated in numpy.

**Degenerate tails.** For a weighted CVaR the result records `p_zero_loss` =
P(L = 0), `tail_outcomes` = (1 − β) / p_min, and `degenerate_tail`:
`"expected_uncovered"` when β ≤ `p_zero_loss` (the objective is then a multiple
of E[U]), `"single_outcome"` when the tail holds at most one draw (a minimax
over years), otherwise `None`. They are recorded, not raised.

### The design result

`DesignResult` holds the resolved `cover` and its `parameters` (the designed
value of each free parameter); `template_hash`; `design_id`, `sha256:` plus the
SHA-256 of the canonical JSON of the template hash, the objective, the pricing,
the budget, the limit, the options and the design sample's `content_hash`
(the solver and the time are in the record, not in the id); the
`objective_value`, the design `premium`, `expected_loss`, `budget_bound` and
`budget_use`; the `route`; the solver's identity, status and work; the
non-uniqueness report and the degenerate-tail flags; and the `design_sample`.

`evaluate` and `compare` take a `DesignResult` in place of a cover: its rows
use the design premium (`premium_basis` `"design"`), carry its `design_id`,
`unique`, `ties` and `degenerate_tail`, and its `to_record()` goes into the
metadata's `designs`, with the solver in `solvers`. In-sample rows of a design
measure its fit on the years it was chosen on; out-of-sample rows, on held-out
years or fresh draws, are the ones to report.

## What this version does not do

- **Several assets in one sample.** Each asset is designed and evaluated on its
  own sample. Nothing sums over assets: there are no designs for a goal over
  several assets, and no dependence between assets is assumed.
- **Several hazards in one index.** An index is one hazard at one location.
- **Several scenarios at once.** Every sample is `"baseline"`, and a plan on
  fitted curves samples one hazard, horizon and pathway per asset; the
  comparison's `scenario` column is there for later.
- **Several events within a year.** One draw is one year.
- **Remote requests in the plan.** `for_evaluation` takes a request on a local
  canonical file; materialize a remote source first (its `materialize(path)`
  returns the `HazardDataset`).
- **An index read elsewhere in the plan.** The plan reads the index at each
  asset. On per-year values, `sample_annual(..., index=series)` reads it at
  another cell, station or hazard; on fitted curves, an index elsewhere needs
  a joint distribution of two curves, which the curves do not give.
- **An H3 cell for registry impacts on per-year values.** Give the impact a
  context with the cell, as above. On fitted curves the plan passes each
  asset's matched cell.
- **Pooled k-fold metrics, bootstrap intervals, trend removal.** Each fold has
  its own rows, and years are used as recorded.
