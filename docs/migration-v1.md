# Migrating to crc-sdk 1.0

[Docs index](README.md) · [Python interface](api-reference.md) · [Release notes](release-notes.md)

crc-sdk 1.0 is a rewrite of the public interface. The ideas are the same (lazy
recipes, canonical Parquet, asset evaluation), but the names, the dataset
format, and the output shape changed. This page maps 0.11 code to 1.0 code.

Examples marked **Before** use crc-sdk 0.11. Examples marked **After** run
against 1.0 and are checked by `tools/check_examples.py`, except those that read
live provider data.

Numerical changes in distributions, fitting, impacts, and risk are covered by
the [crc-framework migration guide](https://github.com/RiskThinking/crc-framework-rs/blob/main/docs/migration-v1.md).
Read its "Changes that alter results" section too.

## Upgrade checklist

1. Install `crc-sdk>=1,<2` with `crc-framework>=1,<2` (Python 3.11+). The
   `geometry` and `agriculture` extras are gone: geometry support is built in,
   and the remaining extras are `raster`, `netcdf`, and `zarr`.
2. Import from `crc_sdk` and `crc_framework` only. Every 0.x submodule
   (`crc_sdk.workflows`, `.providers`, `.connectors`, `.fitting`, `.impacts`,
   `.types`, `.geometry`, `.schema`, …) is removed.
3. Replace `HazardDataset.<source>()` with a provider class such as `CRCOpen`,
   `JRC`, or `ERA5`, or with `Dataset.from_quantiles` / `Dataset.from_raster`
   for your own data.
4. Replace `for_assets(...).return_periods(...).write_parquet(...)` with
   `evaluate(Assets(...)).return_periods(...).write(...)`, and read results in
   the new long format.
5. Replace fit policies with framework specifications passed to `fit(...)`.
6. Regenerate stored datasets. 1.0 reads only Dataset format `2.0`; files
   written by 0.x (formats 1.0–1.4) and 0.x portfolio outputs cannot be opened.
7. Declare units and probability meaning with `Variable(measurement=...)` and
   `Probability.annual(tail=...)` instead of free-text `value_unit`,
   `value_semantics`, and `return_period_tail`.

## Concepts at a glance

| 0.11 | 1.0 |
| --- | --- |
| `HazardDataset` plans, `*Plan` classes | Typed recipes: `Dataset`, `Samples`, `Observations`, `Projections`, `Layer` |
| `canonicalize(policy=...)` fits implicitly during materialization | Supplied quantiles are used as they are; fitting is an explicit `fit(spec)` |
| `materialize(path)`, `write_parquet(path)` | `write(path)`, which returns a reopenable handle |
| `prefetch()`, `ensure_materialized()` | `Runtime(cache=Cache.reuse(path))`, or one `execute()` scope |
| `.cache(dir, mode=...)` | `Runtime(cache=Cache.reuse/offline/refresh(dir))` |
| `ExecutionOptions(connection, batch_rows, max_workers, ...)` | `Runtime(connection=..., batch_rows=..., workers=..., memory_limit=...)` |
| `AssetPortfolio` | `Assets` |
| Bounds tuple `(west, south, east, north)` | `Bounds(west=..., south=..., east=..., north=...)` |
| `for_area(bounds)`, `hazards([...])`, `horizons([...])`, `pathways([...])` | `select(area=..., variables=..., periods=..., scenarios=...)` |
| `hazard_name`, `pathway`, `horizon` | `variable_id`, `scenario_id`, `period_id` |
| `explain()` returns JSON | `explain()` returns a typed `Explanation`; `inspect()` lists what is available |

## Read and evaluate a canonical dataset

`HazardDataset.local(path)` and `LocalProvider` become `Dataset.open(path)`.
Evaluation no longer writes one column per return period: each row is one
asset, series, and query value.

**Before**

```py
import pyarrow as pa
from crc_sdk.workflows import HazardDataset

assets = pa.table({
    "asset_id": ["warehouse-a", "warehouse-b"],
    "longitude": [6.9603, 7.5010],
    "latitude": [50.9375, 51.0030],
    "sector": ["logistics", "manufacturing"],
})
result = (
    HazardDataset.local("flood.parquet")
    .for_assets(assets)
    .select(horizons=[2050], pathways=["ssp585"])
    .return_periods([25, 100])
    .write_parquet("portfolio-flood.parquet")
)
print(result.value_columns)  # ('value_rp25', 'value_rp100')
```

**After**

The first block creates a small Dataset 2.0 file so the example runs on its
own. Replace it with your own file.

```python
from crc_framework import DepthAboveGround, LengthUnit, Tail
from crc_sdk import Dataset, Period, Probability, Scenario, Variable

depth = Variable(
    "company:flood-depth", measurement=DepthAboveGround(unit=LengthUnit.METRE)
)
Dataset.from_quantiles(
    {
        "support_id": ["cell-a"] * 5,
        "longitude": [6.9603] * 5,
        "latitude": [50.9375] * 5,
        "probability": [0.5, 0.8, 0.9, 0.96, 0.99],
        "value": [0.0, 0.1, 0.3, 0.6, 1.2],
    },
    variable=depth,
    scenario=Scenario("ssp585"),
    period=Period.window(2041, 2060),
    probability=Probability.annual(tail=Tail.UPPER),
).write("flood.parquet")
```

```python
from crc_sdk import Assets, EvaluationArtifact

assets = Assets(
    {
        "asset_id": ["warehouse-a", "warehouse-b"],
        "longitude": [6.9603, 7.5010],
        "latitude": [50.9375, 51.0030],
        "sector": ["logistics", "manufacturing"],
    },
    columns=["sector"],  # extra columns are no longer carried by default
)
evaluation = (
    Dataset.open("flood.parquet")
    .select(scenarios="ssp585")
    .evaluate(assets)
    .return_periods([25, 100])
)
saved = evaluation.write("portfolio-flood.parquet")
rows = EvaluationArtifact.open(saved.path).read().table.to_pylist()
for row in rows:
    print(row["asset_id"], row["query_value"], row["value"], row["failure"])
```

What changed in the output:

- One row per asset, series, and query value, with `query_value` and `value`
  columns instead of `value_rp25`, `value_rp100`, ….
- An asset with no covering data keeps its rows, with a null `value` and a
  `failure` such as `no_covering_support`. In 0.x the run raised an error.
- Retained asset attributes are nested in the `asset` column, and only the
  columns you list in `Assets(columns=...)` are kept. They keep their Arrow
  types, such as timestamps and dictionaries, when saved and reopened.
- Metadata is typed (`saved.metadata`) instead of JSON under
  `crc.hazard.evaluation`.

If downstream code expects the old wide layout, pivot the table:

```python
wide = {}
for row in rows:
    columns = wide.setdefault(row["asset_id"], {})
    columns[f"value_rp{int(row['query_value'])}"] = row["value"]
assert abs(wide["warehouse-a"]["value_rp25"] - 0.6) < 1e-9
```

### Asset inputs

| 0.11 | 1.0 |
| --- | --- |
| Arrow table | Arrow table, record batch, mapping of columns, or pandas DataFrame |
| `Path` to Parquet | `str` or `Path` to a CSV or Parquet file |
| `str` treated as DuckDB SQL | `Assets.sql("SELECT ...", location=PointColumns())`; a plain `str` is now a **file path** |
| `AssetPortfolio(data, id_column="site", location=PointColumns("lon", "lat"))` | `Assets(data, id_column="site", location=PointColumns(longitude="lon", latitude="lat"))` |
| `location=CellColumn("h3")` | `location=CellColumn(...)` (unchanged idea) |

Spatial matching now uses the original source footprint. A point source only
matches assets at that point, unless you opt into
`SpatialMatch.nearest(max_distance=Distance.kilometres(...))`. See
[Evaluation](evaluation.md#choose-spatial-support).

## Bring your own quantiles

`HazardDataset.from_table(..., policy=CDFCurveFitPolicy(...))` fitted every row
while writing. In 1.0, `Dataset.from_quantiles` keeps your knots exactly and
can be evaluated directly. Fit only if you want a parametric law.

**Before**

```py
from crc_sdk.fitting import CDFColumnSchema, CDFCurveFitPolicy
from crc_sdk.types import SourceProvenance
from crc_sdk.workflows import HazardDataset

policy = CDFCurveFitPolicy(
    h3_resolution=5, family="gumbel_r", value_unit="mm",
    value_semantics="annual maximum 1-day rainfall", producer="me",
    source=SourceProvenance(provider="lab", dataset="rx1day", version="v1"),
    source_id="lab-rx1day", probability_semantics="annual_value_distribution",
)
plan = HazardDataset.from_table(
    "rx1day.parquet", columns=CDFColumnSchema(), policy=policy,
    probabilities=[0.5, 0.8, 0.9, 0.96, 0.99],
)
hazard = plan.materialize("hazards/rx1day.parquet")
```

**After**

```python
from crc_framework import (
    DistributionFamily, Duration, PrecipitationAmount, QuantileFit, TimeUnit,
)
from crc_sdk import QuantileColumns

rx1day = Variable(
    "lab:rx1day",
    measurement=PrecipitationAmount(
        unit=LengthUnit.MILLIMETRE,
        accumulation_duration=Duration(1, unit=TimeUnit.DAY),
    ),
)
rain = Dataset.from_quantiles(
    {
        "site": ["s1"] * 5,
        "lon": [24.0] * 5,
        "lat": [60.0] * 5,
        "q": [0.5, 0.8, 0.9, 0.96, 0.99],
        "mm": [30.0, 42.0, 50.0, 61.0, 78.0],
    },
    variable=rx1day,
    probability=Probability.annual(tail=Tail.UPPER),
    columns=QuantileColumns(
        support_id="site", longitude="lon", latitude="lat", probability="q", value="mm"
    ),
)
rain.write("rx1day-knots.parquet")  # lossless: the supplied knots
fitted = rain.fit(QuantileFit(family=DistributionFamily.GUMBEL_RIGHT))
fitted.write("rx1day-gumbel.parquet")  # optional parametric fit
```

The input is any table (mapping, Arrow, or pandas) with one row per knot.
`h3_resolution`, `producer`, `source_id`, and the free-text unit and semantics
fields are gone: the support ID and typed measurement carry that meaning, and
the H3 delivery index is chosen by the SDK. Censored and hurdle fits use
`CensoredQuantileFit` and `HurdleQuantileFit`; `FitFailure.KEEP_NO_DATA` or
`FitFailure.SKIP` replace `short_axis_action` and no-data thresholds.

### Rasters and Zarr

`HazardDataset.from_raster` now takes **non-exceedance probabilities** as keys,
not return periods, and does not fit. `from_zarr` and the OS-Climate reader have
no 1.0 replacement; export the arrays to GeoTIFF or to a quantile table.

**Before**

```py
from crc_sdk.connectors import CurveFitIngestPolicy
from crc_sdk.workflows import HazardDataset

plan = HazardDataset.from_raster(
    {10: "rp10.tif", 100: "rp100.tif"},
    hazard_type="RiverineInundation", indicator_id="flood_depth",
    scenario="historical", year=2020, units="m",
    policy=CurveFitIngestPolicy(h3_resolution=8, family="gumbel_r", producer="me"),
)
```

**After**

```python
flood_maps = Dataset.from_raster(
    {1 - 1 / 10: "rp10.tif", 1 - 1 / 100: "rp100.tif"},
    variable=depth,
    probability=Probability.annual(tail=Tail.UPPER),
)
assert flood_maps.explain().operations  # building the recipe does no I/O
```

## Annual extremes and sample fitting

`BlockExtremaPolicy` and the ERA5/CMIP6 canonicalization plans become a sample
recipe followed by an explicit fit. You can read and check the annual values
before fitting.

**Before**

```py
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

plan = HazardDataset.era5("rx1day").for_area((23.9, 59.9, 24.1, 60.1)).years(1991, 2020)
hazard = plan.canonicalize(
    policy=BlockExtremaPolicy.curated(
        minimum_return_value=0,
        fallback_family="gumbel_r",
        diagnostics="rx1day-fit-diagnostics.parquet",
    )
).materialize("annual-rainfall.parquet")
```

**After** (reads live ERA5 data)

<!-- live-example -->
```python
from crc_framework import DistributionFamily, SampleEstimator, SampleFit
from crc_sdk import ERA5, Bounds

samples = ERA5(indicator=ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION).select(
    area=Bounds(west=23.9, south=59.9, east=24.1, north=60.1),
    time=Period.window(1991, 2020),
)
annual = samples.read()  # inspect annual.table before fitting
fitted = annual.fit(SampleFit(
    family=DistributionFamily.GENERALIZED_EXTREME_VALUE,
    estimator=SampleEstimator.L_MOMENTS,
    fallbacks=(SampleFit(family=DistributionFamily.GUMBEL_RIGHT),),
))
fitted.write("annual-rainfall.parquet")
```

The same shape works for your own annual values with `Samples.from_table`; see
[Execution](execution.md#inspect-annual-samples-then-fit-them). The 0.x
diagnostics sidecar (`diagnostics="path.parquet"`) is replaced by the fit record
stored on each row and by `read().report.diagnostics`.

## Impacts in an evaluation

`.impact(fn, name=..., value_unit=..., value_semantics=...)` becomes
`.impact("name", impact, variable=..., context=...)`. The impact is a framework
`Impact`; its output measurement replaces `value_unit`. Named responses are
added as extra rows next to the exposure rows, identified by `response_id`.

| 0.11 | 1.0 |
| --- | --- |
| Lambda or vectorized callable | `Impact.linear`, `Impact.sigmoid`, `Impact.tabulated`, `Impact.binned` |
| `PiecewiseLinearImpact(exposure=xs, impact=ys)` | `Impact.tabulated(list(zip(xs, ys)), input=..., output=...)` |
| `impacts.for_factor("inundation")` | `Impact.builtin(BuiltinImpact.INUNDATION)` |
| `ImpactContextColumns(continent="continent", ...)` | A typed context whose fields reference `assets.column("...")` |
| `ImpactExpression`, `ImpactCalibration`, `check_exposure` | Framework `Impact` with input/output measurements; checked automatically |
| Category-specific curves | `ContextualImpact` with your own `ResponseContext` |

**Before**

```py
from crc_sdk.impacts import PiecewiseLinearImpact, impacts
from crc_sdk.workflows import HazardDataset, ImpactContextColumns

request = HazardDataset.local("flood.parquet").for_assets(assets).return_periods([25, 100])
request.impact(
    PiecewiseLinearImpact(exposure=[0.0, 0.2, 1.0, 2.0], impact=[0.0, 0.0, 0.25, 1.0]),
    name="flood_damage_ratio", value_unit="fraction", value_semantics="damage ratio",
).write_parquet("impact-curve.parquet")
request.impact(
    impacts.for_factor("inundation"),
    context=ImpactContextColumns(continent="continent", building_type="building_type"),
    name="inundation_impact", value_unit="fraction", value_semantics="damage ratio",
).write_parquet("impact-registry.parquet")
```

**After**

```python
from crc_framework import (
    AssetType, BuiltinImpact, FloodRegion, Impact, InundationContext, Ratio,
)

sites = Assets({
    "asset_id": ["warehouse-a"],
    "longitude": [6.9603],
    "latitude": [50.9375],
    "region": [FloodRegion.EUROPE.value],
})
curve = Impact.tabulated(
    [(0.0, 0.0), (0.2, 0.0), (1.0, 0.25), (2.0, 1.0)],
    input=depth.measurement,
    output=Ratio(),
)
responses = (
    Dataset.open("flood.parquet")
    .evaluate(sites)
    .return_periods([25, 100])
    .impact("flood_damage_ratio", curve, variable=depth)
    .impact(
        "inundation_impact",
        Impact.builtin(BuiltinImpact.INUNDATION),
        variable=depth,
        context=InundationContext(
            region=sites.column("region"), asset_type=AssetType.COMMERCIAL
        ),
    )
)
table = responses.read().table
print(table.select(["response_id", "query_value", "value"]).to_pylist())
```

### Built-in models that cannot be bound to data

In 1.0 only `COASTAL_FLOOD`, `RIVER_FLOOD`, `INUNDATION`, and
`FIRE_WEATHER_INDEX` can be used in `.impact(...)`. The other built-in models
(rainfall, wind, cyclone, frost, freeze-thaw, hot days, cooling degree days,
SPEI, consecutive dry days) have calibration facts that are not yet published,
such as the reference period, so the SDK rejects them with a `ValidationError`.

If you used one of these through `impacts.for_factor(...)`, you can still
compute it numerically: evaluate the exposure, resolve the model with a
concrete context, and apply it to the values yourself.

```python
import pyarrow as pa
from crc_framework import CellContext

exposure = rain.evaluate(
    Assets({"asset_id": ["site-1"], "longitude": [24.0], "latitude": [60.0]})
).return_periods([10, 100]).read().table
model = Impact.builtin(BuiltinImpact.MAXIMUM_DAILY_PRECIPITATION)
damage = model.resolve(CellContext(cell=0x851126D3FFFFFFF))
values = exposure["value"].to_numpy(zero_copy_only=False)
with_damage = exposure.append_column("damage_ratio", pa.array(damage.evaluate(values)))
```

This result is not recorded in the evaluation metadata, so keep track of the
model and context you used.

## Providers

Every provider is a class imported from `crc_sdk`. Use `inspect()` to list the
indicators, series, stations, and releases it offers, then pass what you chose
to `select(...)`.

| 0.11 | 1.0 |
| --- | --- |
| `HazardDataset.crc_open()` | `CRCOpen(catalog=...)` (see note below) |
| `HazardDataset.efas()`, `.glofas()` | `JRC(product=JRC.Product.EFAS)`, `JRC(product=JRC.Product.GLOFAS)` |
| `HazardDataset.smi()`, `EDO*Plan` | `EDO(...)` (native dekadal observations) |
| `HazardDataset.era5(recipe)`, `ERA5*Plan` | `ERA5(indicator=ERA5.Indicator...)`; raw data via `.observations()` |
| `HazardDataset.cmip6(...)`, `CMIP6*Plan` | `CMIP6(indicator=...)` with `select(series=...)` |
| `HazardDataset.nex_gddp_cmip6(...)` | `NEX(...)` |
| `HazardDataset.isd()`, `ISDPlan` | `ISD()` |
| `HazardDataset.ghcnh()`, `GHCNhPlan` | `GHCNh()` |
| `HazardDataset.eccc_idf(...)`, `ECCCIDFPlan` | `ECCCIDF(duration=..., province=...)` |
| `ECCCDailyPlan`, `HYDATDailyPlan`, `USGSStreamflowPlan` | `ECCCDaily`, `HYDAT`, `USGS` |
| `CHIRPSMonthlyPlan`, `IBTrACSPlan` | `CHIRPS`, `IBTrACS(agency=...)` |
| `NOAASeaLevelPlan`, `NOAAExtremeWaterLevelPlan` | `NOAASeaLevel`, `NOAAWaterLevels` |
| `AgriculturalLayer` | `USDACropland`, `FieldsOfTheWorld` |
| `OSClimateProvider`, `HazardDataset.from_zarr` | Removed |

CRC Open hazard names are enum members, for example
`CRCOpen.Variable.ANNUAL_MAXIMUM_DAILY_PRECIPITATION` (formerly `"rx1day"`) and
`CRCOpen.Scenario.SSP585`.

**Note:** 0.11 downloaded the public CRC Open release by default. A production
CRC Open release in Dataset 2.0 format is not yet published, so 1.0 needs an
explicit release index: `CRCOpen(catalog="path-or-url/release-index.json")`.
See [Release notes](release-notes.md#crc-open-availability).

**Before**

```py
plan = (
    HazardDataset.crc_open()
    .for_area((-79.5, 43.5, -79.2, 43.8))
    .hazards(["rx1day"])
    .horizons([2050])
)
plan.for_assets(assets).return_periods([10, 100]).write_parquet("rx1day.parquet")
```

**After** (needs a release index)

<!-- live-example -->
```python
from crc_sdk import CRCOpen

source = CRCOpen(catalog="release-index.json").select(
    area=Bounds(west=-79.5, south=43.5, east=-79.2, north=43.8),
    variables=CRCOpen.Variable.ANNUAL_MAXIMUM_DAILY_PRECIPITATION,
)
for series in source.inspect().series:  # choose a period and scenario here
    print(series.scenario, series.period)
source.evaluate(assets).return_periods([10, 100]).write("rx1day.parquet")
```

## Caching and execution

**Before**

```py
from crc_sdk.workflows import ExecutionOptions, HazardDataset

plan = HazardDataset.crc_open().for_area(bounds).hazards(["rx1day"])
plan.cache(".crc-cache", mode="reuse").prefetch()
plan.for_assets(assets).return_periods([100]).write_parquet(
    "out.parquet", execution=ExecutionOptions(max_workers=4, batch_rows=10_000)
)
```

**After**

```python
from crc_sdk import Cache, Runtime

runtime = Runtime(cache=Cache.reuse(".crc-cache"), workers=4, batch_rows=10_000)
recipe = Dataset.open("flood.parquet").evaluate(assets).return_periods([100])
with recipe.execute(runtime=runtime) as execution:
    report = execution.validate()
    execution.write("out.parquet")
assert report.valid
```

The 0.x `stream` cache mode is the 1.0 default (`Runtime()` uses temporary
storage). `Cache.offline(dir)` replays verified downloads without network
access. One `execute()` scope runs each source and fit once and shares the
result across `validate`, `read`, `batches`, and `write`.

## Dataset format 2.0

| 0.x column or field | 2.0 |
| --- | --- |
| `hazard_name` | `variable_id` |
| `pathway` | `scenario_id` (nullable) |
| `horizon` | `period_id` (nullable, derived from a `Period`) |
| `source_id` | `support_id` (identity of the original footprint) |
| `cell_index` | `cell_id` |
| `source_geometry` | `geometry` |
| `curve_kind`, `curve_type`, `curve_*` | `curve` (one tagged struct) plus `fit` and `failure` |
| `no_data` curve kind | `failure` record; no fictional curve |
| `value_unit`, `value_semantics`, `return_period_tail` | Typed `Variable.measurement` and `Probability` in metadata |
| `crc.hazard.metadata` footer key | Typed `DatasetMetadata` (`Dataset.open(path).metadata`) |

There is no converter for 0.x files. Rebuild them from their sources, or read
the old knots with pyarrow and pass them to `Dataset.from_quantiles`.

## Removed without a replacement

- PMTiles building and tiling (`crc_sdk.geometry.pmtiles`,
  `run_tiled_canonicalization`, `tile_bounds`).
- H3, vector, coverage, and administrative-boundary helpers in
  `crc_sdk.geometry`.
- `from_zarr`, `OSClimateProvider`, `LocalProvider`, `HazardQuery`.
- Low-level connectors (`crc_sdk.connectors`, DuckDB raster readers,
  `canonicalize_curve_source`, `DuckDBPipeline`).
- Curve helpers (`curve_quantiles`, `curve_quantiles_at`,
  `distribution_from_hazard_row`, `return_periods_to_probabilities`); use
  `evaluate(...).probabilities(...)` or framework `Distribution.from_json`.
- Re-exported distributions and fitting functions (`crc_sdk.Distribution`,
  `fit_distribution`, …); import them from `crc_framework` under their 1.0 names.
- `ProbabilitySemanticsWarning`, `ReturnPeriodExtrapolationWarning`: invalid
  return-period queries now fail, and out-of-range queries return a row with a
  `query_outside_support` failure.

## New in 1.0

- `DatasetCollection` evaluates several datasets together and saves a portable
  bundle. [Collections](collections.md)
- `Observations` with typed aggregation (`DailyTotal`, `AnnualMaximum`, …),
  `Projections`, and `Layer` for evidence that is not a curve.
  [Sources](sources.md)
- `inspect()` discovery and release pinning, with offline replay of the exact
  bytes you read.
- `SpatialMatch.nearest` with an explicit distance cap, and
  `preferred_supports` for overlapping footprints.
- Named responses with `ContextualImpact`, and `EvaluationArtifact.open` to
  reopen saved evaluations with their full provenance.
  [Evaluation](evaluation.md)
- Bounded previews with `read(limit=...)`, streaming with `batches()`, and
  atomic writes that never leave a half-written file.
  [Execution](execution.md)
