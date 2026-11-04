# Portfolio evaluation and impacts

[Docs index](README.md) · [Project README](../README.md)

## Evaluate assets at return periods

Join every asset to its canonical curve and write one row per asset, hazard,
horizon and pathway, with one value column per return period:

```python
import pyarrow as pa
from crc_sdk.workflows import HazardDataset

assets = pa.table({
    "asset_id": ["warehouse-a", "warehouse-b"],
    "longitude": [6.9603, 7.5010],
    "latitude": [50.9375, 51.0030],
    "sector": ["logistics", "manufacturing"],   # extra columns pass through
})

result = (
    HazardDataset.local("flood.parquet")
    .for_assets(assets)
    .select(horizons=[2050], pathways=["ssp585"])   # optional filters
    .return_periods([25, 100, 500])
    .write_parquet("portfolio-flood.parquet")
)
```

Output columns are `value_rp25`, `value_rp100`, `value_rp500`. Units, value
semantics and the full return-period → probability → column mapping are stored
under the `crc.hazard.evaluation` Parquet metadata key.

Any source plan (`crc_open()`, `efas()`, `era5(...)`, `from_table(...)`, …)
also exposes `.for_assets(...)`; it materializes the canonical file in the plan's
cache first. CRC Open plans require `.for_area(bounds)` so acquisition stays
bounded.

### Asset inputs

- **Arrow table:** registered directly.
- **`Path`:** read as a Parquet file.
- **`str`:** treated as caller-supplied DuckDB SQL.

`asset_id`, `longitude`/`latitude` and `cell_index` are inferred. For other
names wrap the data, for example
`AssetPortfolio(data, id_column="site", location=PointColumns("lon", "lat"))`.
Assets that already carry canonical H3 indexes (`cell_index`, or
`location=CellColumn("h3")`) skip point conversion and exact geometry
refinement.

### Join rules

- Points are converted to the dataset's H3 resolution. If the source row has
  WKB, the match is refined with `ST_Covers(source_geometry, point)`.
  `spatial_match` reports `exact_geometry` or `h3_cell`.
- Several source curves for one asset/hazard/horizon/pathway raise an error;
  they are never silently selected or aggregated.
- Missing asset/scenario matches raise instead of being dropped.
- Evaluation streams bounded Arrow batches to compressed Parquet.

Execution tuning lives in `ExecutionOptions` (`connection`, `batch_rows`,
`max_workers`, `chunk_rows`), passed as `write_parquet(..., execution=...)`.

## Apply an impact function

`.impact(...)` replaces sampled hazard values with event-aligned impacts:
`value_rp100 = impact(hazard_rp100)`. The return period still identifies the
source hazard event. This intentionally differs from transforming a whole
distribution and then taking an impact quantile, which reorders decreasing or
non-monotonic impacts; use crc-framework's distribution interface for that.

```python
import numpy as np
from crc_sdk.impacts import PiecewiseLinearImpact, impacts
from crc_sdk.workflows import ExecutionOptions, HazardDataset, ImpactContextColumns

request = HazardDataset.local("flood.parquet").for_assets(assets).return_periods([25, 100, 250])

# 1. A vectorized callable. Lambdas cannot be pickled, so run them with one worker.
request.impact(
    lambda depth: np.clip(depth / 2.0, 0.0, 1.0),
    name="depth_damage_ratio", value_unit="fraction", value_semantics="damage ratio",
).write_parquet("impact-callable.parquet", execution=ExecutionOptions(max_workers=1))

# 2. A built-in curve.
request.impact(
    PiecewiseLinearImpact(exposure=[0.0, 0.2, 1.0, 2.0], impact=[0.0, 0.0, 0.25, 1.0]),
    name="flood_damage_ratio", value_unit="fraction", value_semantics="damage ratio",
).write_parquet("impact-curve.parquet")

# 3. A registry transform, with context read from asset columns.
request.impact(
    impacts.for_factor("inundation"),
    context=ImpactContextColumns(
        country="country", continent="continent",
        building_type="building_type", historic_mean="historic_mean",
    ),
    name="inundation_impact", value_unit="fraction", value_semantics="damage ratio",
).write_parquet("impact-registry.parquet")
```

The generated H3 `cell_index` is always passed to the impact context. Context
columns are read from each asset even when not retained in the output; stored
registry context fills fields an asset leaves empty. Impact metadata records the
event-aligned interpretation, hazard and output units, function identity and
context mapping.

Framework impact objects and top-level functions run in the process pool.
Lambdas and closures run serially when the worker count is implicit; explicitly
requesting more than one worker for them raises.

## Reproducible impact definitions

An `ImpactExpression` is a persisted, hashable definition, separate from the
workflow binding above:

```python
from crc_sdk.impacts import (
    ImpactCalibration, ImpactExpression, RegistryImpactReference,
    check_exposure, exposure_declaration,
)

calibration = ImpactCalibration(
    exposure_variable="flood_depth", exposure_unit="m",
    impact_metric="damage_ratio", impact_unit="1",
)
impact = ImpactExpression.tabulated(
    [(0.0, 0.0), (0.3, 0.08), (1.0, 0.45), (2.0, 1.0)],
    calibration=calibration,
    clamp={"minimum": 0.0, "maximum": 1.0},
)
restored = ImpactExpression.from_json(impact.to_json())
assert restored.definition_hash == impact.definition_hash   # SHA-256 of canonical JSON

pinned = RegistryImpactReference.pinned("rx1day")             # pinned to the installed registry
check_exposure(pinned.calibration(), exposure_declaration("rx1day"))
```

`exposure_declaration` reads the framework hazard catalog (`rx1day` is mm/day;
`fd` is `frost_days`). Unit aliases such as `days` and `d` match through the
unit catalog. Hazards outside the catalog need an explicit declaration. The
schemas and identities are described in crc-framework's
[impact contracts guide](https://github.com/RiskThinking/crc-framework-rs/blob/main/docs/impact-contracts.md).

## Lower-level helpers

- `curve_quantiles(table, probabilities)` and `curve_quantiles_at(table, q)`
  evaluate canonical rows in parallel chunks.
- `stream_curve_quantiles_to_parquet` / `stream_curve_quantiles_wide_to_parquet`
  stream evaluated quantiles to disk.
- `distribution_from_hazard_row(row)` reconstructs one `crc_framework`
  distribution.
