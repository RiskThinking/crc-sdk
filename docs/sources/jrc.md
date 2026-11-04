# JRC flood and drought

[Docs index](../README.md) · [Project README](../../README.md)

Requires `crc-sdk[raster,geometry]` for flood and `crc-sdk[netcdf,geometry]`
for drought. Both sources are CC BY 4.0; cite JRC and the product version.

## River flood: EFAS and GloFAS

GloFAS 2.1.2 uses JRC's tiled global layout; EFAS 3.1.1 uses nine continental
return-period rasters (RP10–RP500). The plan hides tiles and filenames:

```python
from crc_sdk.workflows import HazardDataset, JRCFloodPolicy

plan = (
    HazardDataset.efas(version="latest")             # or .glofas(...)
    .for_area((7.75, 49.75, 8.45, 50.25))
    .cache("cache/efas", mode="reuse")
    .source_periods("all")                           # rasters used for fitting
    .canonicalize(policy=JRCFloodPolicy.curated(h3_resolution=10))
)
hazard = plan.materialize("hazards/efas-rhine.parquet")

result = (
    plan.for_assets(assets)
    .select(hazard_names=["RiverineInundation"], pathways=["historical"])
    .return_periods([50, 100, 250, 500])             # evaluation periods
    .write_parquet("outputs/flood-depth.parquet")
)
```

- `materialize()` resolves `latest` once and pins the version in the cache
  manifest and canonical provenance, caches AOI crops, then fits.
- Source periods (`source_periods`) choose the input rasters; evaluation
  periods (`return_periods`) are independent. With RP10–RP500 inputs, RP250 is
  interpolation and RP1000 is extrapolation (a warning is emitted).
- Fitted values at source return periods are curve evaluations, not
  bit-for-bit source pixels.

## Drought: EDO Soil Moisture Index

Complete years are reduced to compact annual-minimum crops before fitting:

```python
from crc_sdk.workflows import EDODroughtPolicy, HazardDataset

plan = (
    HazardDataset.smi(version="latest")
    .for_area((9.5, 50.5, 10.5, 51.5))
    .years("all_complete")
    .cache("cache/edo-smi", mode="reuse")
    .canonicalize(policy=EDODroughtPolicy.curated(h3_resolution=6))
)
hazard = plan.materialize("hazards/edo-smi.parquet")
```

The curated policy fits the lower tail and requires at least 20 complete
years. Metadata records the tail and the Gringorten support of the annual
record, so evaluation selects the lower tail automatically and warns on
extrapolation. Manifests pin the EDO version, years, bounds, URLs and checksums.

## Low-level API

`canonicalize_jrc_flood` / `JRCIngestPolicy` and `canonicalize_edo_drought` /
`EDOIngestPolicy` (`crc_sdk.connectors`) run the same conversions without a
plan. `JRCProvider` and `EDOProvider` (`crc_sdk.providers`) list releases and
resources.
