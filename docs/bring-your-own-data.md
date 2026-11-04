# Bring your own data

[Docs index](README.md) · [Project README](../README.md)

`HazardDataset.from_table`, `from_zarr` and `from_raster` turn local or remote
data into canonical datasets through the same lazy plan (`BYOPlan`):
`canonicalize(policy=...)`, `explain()`, `materialize(output)`,
`ensure_materialized()` and `for_assets(...)`. Nothing is opened or fitted until
an execution point. `ensure_materialized()` and `for_assets(...)` need
`.cache(directory)`, which gives a deterministic canonical path
(`mode="refresh"` rewrites it). Local files, directories and globs are
fingerprinted by file size and modification time; remote URLs are
identified by path only, so use `mode="refresh"` after an upstream change.

```text
external raster/table -> source curves and geometry -> family fit
  -> conservative intersecting H3 cells -> canonical Arrow -> Parquet
```

## Tables of quantiles

Parquet, CSV, any DuckDB-readable path, or an Arrow table keyed by H3 cell:

```python
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
    probabilities=[i / 10 for i in range(11)],   # or return_periods=[...]
)
hazard = plan.materialize("hazards/rx1day.parquet")
```

The full policy reference is in [fitting](fitting.md).

## Return-period Zarr arrays

Requires `crc-sdk[zarr,geometry]`. The array carries `index_name` (containing
"return period"), `index_values` and `transform_mat3x3` attributes.

```python
from crc_sdk.connectors import CurveFitIngestPolicy

plan = HazardDataset.from_zarr(
    "s3://bucket/flood.zarr", array="depth", storage_options={"anon": True},
    hazard_type="RiverineInundation", indicator_id="flood_depth",
    scenario="historical", year=2020, units="m",
    policy=CurveFitIngestPolicy(h3_resolution=8, family="gumbel_r", producer="me"),
).for_area((7.75, 49.75, 8.45, 50.25))
hazard = plan.cache("cache/byo").ensure_materialized()
```

## GeoTIFFs, one per return period

Requires `crc-sdk[raster,geometry]`. All rasters must share one grid.

```python
plan = HazardDataset.from_raster(
    {10: "rp10.tif", 50: "rp50.tif", 100: "rp100.tif", 500: "rp500.tif"},
    hazard_type="RiverineInundation", indicator_id="flood_depth",
    scenario="historical", year=2020, units="m",
    policy=CurveFitIngestPolicy(h3_resolution=8, family="gumbel_r", producer="me"),
)
hazard = plan.for_area((7.75, 49.75, 8.45, 50.25)).materialize("hazards/flood.parquet")
```

For sparse flood curves whose lower return periods are zero, prefer
`fit_method="censored_quantile"` with `family="genextreme"` and
`censor_location=0.0` ([why](fitting.md#censored-return-period-curves)).

## OS-Climate hazard indicators

`OSClimateProvider` (`crc-sdk[zarr]`) reads OS-Climate return-period Zarr
rasters; `canonicalize_os_climate` with an `OSClimateIngestPolicy` converts
them. The caller chooses the family and fit method. `LocalProvider` then queries
the persisted rows through `HazardQuery`.

## Low-level building blocks

`crc_sdk.connectors` exposes the pieces the plans use: format readers in
`crc_sdk.connectors.duckdb` (`GeoTiffRaster`, `NetCDFRaster`, `ZarrRaster`),
`canonicalize_curve_source`
for any `CurveSource`, and the canonical writers in
[core concepts](concepts.md#canonical-hazard-datasets).
`DuckDBRelationSource`, `ArrowBatchSource` and `DuckDBPipeline` form a lazy
filter/project/aggregate/write pipeline shared by SQL and chunked sources.
