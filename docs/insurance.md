# Simulate annual insured loss

[Docs index](README.md) · [Evaluate datasets](evaluation.md)

`LossRun` applies primary policy terms to simulated annual damage for one peril.
It accepts an evaluation with one selected annual hazard series and an explicit
damage response. The response maps hazard severity to mean physical damage as a
fraction of insured value. Use `Ratio(unit=RatioUnit.FRACTION)` for the model
output. The framework owns the stochastic dependence, secondary damage
uncertainty, and loss statistics.

Each trial draws one annual severity per location and sums losses after
primary terms. This gives a modeled annual aggregate loss, gross of
reinsurance, for annual aggregate exceedance probability (AEP) metrics.

## Set up the book and annual damage model

This example covers two point locations. Each depth table has one row per
support and non-exceedance probability. The two windows use the same insured
interests and vulnerability curve. The amounts are in USD with fixed
replacement values and deductibles across both windows.

```python
from crc_framework import AnnualLossSpec, DepthAboveGround, Impact, LengthUnit, Ratio, RatioUnit, Tail
from crc_sdk import Assets, Dataset, LossArtifact, LossRun, Period, PolicyColumns, Probability, Variable

insured_values = "replacement_value"
deductibles = "per_interest_deductible"
assets = Assets({
    "asset_id": ["warehouse", "depot"],
    "longitude": [24.0, 24.3],
    "latitude": [60.0, 60.1],
    insured_values: [800_000.0, 400_000.0],
    deductibles: [20_000.0, 10_000.0],
})
policy = PolicyColumns(
    total_insured_value=insured_values,
    deductible=deductibles,
    share=0.9,
    currency="USD",
    valuation_basis="2026 replacement value",
)

depth = DepthAboveGround(unit=LengthUnit.METRE)
hazard = Variable("example:flood-depth", measurement=depth)
damage_response = "building_damage_fraction"
damage = Impact.linear(
    slope=0.22,
    minimum=0.0,
    maximum=1.0,
    input=depth,
    output=Ratio(unit=RatioUnit.FRACTION),
)
probabilities = [0.0, 0.5, 0.9, 0.99, 1.0]

def damage_evaluation(period, depth_values):
    width = len(probabilities)
    data = Dataset.from_quantiles(
        {
            "support_id": ["west"] * width + ["east"] * width,
            "longitude": [24.0] * width + [24.3] * width,
            "latitude": [60.0] * width + [60.1] * width,
            "probability": probabilities * 2,
            "value": depth_values,
        },
        variable=hazard,
        period=period,
        probability=Probability.annual(tail=Tail.UPPER),
    )
    return data.evaluate(assets).probabilities(probabilities).impact(
        damage_response, damage, variable=hazard,
    )

baseline_damage = damage_evaluation(
    Period.window(1981, 2010),
    [0.0, 0.0, 0.4, 1.1, 2.0, 0.0, 0.0, 0.2, 1.0, 2.5],
)
future_damage = damage_evaluation(
    Period.window(2041, 2070),
    [0.0, 0.0, 0.7, 1.5, 2.6, 0.0, 0.0, 0.4, 1.3, 3.1],
)
```

Term values in `PolicyColumns` can be an asset column name or a single number
for the whole book. Deductibles and limits are money amounts in the declared
currency. Use `deductible_fraction` instead of `deductible` for a proportion of
each interest's insured value. Shares and deductible fractions are in `[0, 1]`.
The defaults apply no deductible and give a full share. An omitted limit uses
the insured value. Each interest must have a point location in WGS84 and valid
terms.

Choose one series or collection member with `Evaluation.select()` when your
source contains several. Use `probabilities()` with strictly increasing
non-exceedance probabilities. Every requested rank must yield a finite,
nondecreasing damage fraction in `[0, 1]` for every asset. If the supplied grid
omits rank 0 or 1, pass a framework `TailCompletion` with explicit damage
endpoints in `[0, 1]`. It completes the damage curve from the nearest observed
rank to the stated endpoint. You still need a successful evaluation at every
requested rank. A missing asset or failed response prevents simulation and is
reported by `validate()`.

## Simulate, save, and compare

The recipe and `explain()` describe work before any I/O. A context-managed
execution shares the result across its terminals. A successful `validate()`
includes the simulation and keeps the sample ready for `read()` and `write()`
in that scope. Local Parquet publication is atomic, and overwriting requires
`overwrite=True`.

```python
spec = AnnualLossSpec(trials=400, seed=20261027)
peril = "river_flood"
baseline_run = LossRun(
    baseline_damage,
    damage_response=damage_response,
    peril=peril,
    policy=policy,
    spec=spec,
    secondary_stream=0,
    include_components=True,
)
future_run = LossRun(
    future_damage,
    damage_response=damage_response,
    peril=peril,
    policy=policy,
    spec=spec,
    secondary_stream=1,
)

with baseline_run.execute() as execution:
    assert execution.validate().valid
    observed = execution.read()
    saved = execution.write("baseline-loss.parquet")

baseline = LossArtifact.open(saved.path).read()
assert baseline.metadata.run_id == observed.metadata.run_id
assert baseline.components is not None

future = future_run.read()
summary = future.summary(levels=[0.95, 0.99], return_periods=[10, 25, 100])
assert summary.trial_count == spec.trials
assert summary.at(0.99).conditional_value_at_risk >= summary.at(0.99).value_at_risk

change = future.compare(baseline)
print(change.average_annual_loss_change)
```

The sample keeps zero-loss years and stable trial IDs. A saved artifact carries
the simulation specification, source evidence, run and sample identities, and
the execution report. `result.metadata.series` gives the selected member,
variable, scenario, period, and annual probability definition.
`result.metadata.damage_response` names the response and
`result.metadata.evidence` records its resolved models and matching evidence.
`result.report.source_identities` has source checksums and attribution.
`result.report.counts` and `result.report.diagnostics` describe evaluation
coverage and diagnostics.

Reopen a sample to choose new risk levels or return periods without simulating
again. Set `include_components=True` to retain each interest's aligned annual
losses for component tail contributions and the VaR diversification benefit.
The `components` mapping uses string asset IDs, including the string form of
numeric IDs. Component output increases memory and file size with the number
of interests and trials.

A comparison holds the book, coverage terms, currency, valuation basis, peril,
vulnerability assignments, and simulation specification fixed. Matching seeds
and settings give a shared primary rank draw for each trial. Distinct secondary
streams give separate secondary damage draws. The result carries the two run
IDs and trial differences computed as future loss minus baseline loss. The
example uses a small sample. The reference specification uses 20,000 trials.

## Read the Parquet layout

An Annual Loss 1.0 file holds one run and one peril. It stores one row per
zero-based `trial_id`, with a 64-bit floating-point `insured_loss` in the
declared currency. With components enabled, trial 0 holds the ordered
`component_ids`, and each trial contains a corresponding `component_losses`
vector. The metadata and its checksum are
stored in the Parquet footer under `crc.annual_loss` and
`crc.annual_loss.sha256`. `LossArtifact.validate()` checks the metadata,
each trial, and sample checksums. [Annual Loss metadata schema](../crc_sdk/contracts/schemas/annual-loss-1.0.schema.json)
