# CRC SDK documentation

Start with the [project README](../README.md) for installation and a runnable
quick start. Each page below answers one question; read only what you need.

## Use the SDK

| Page | Answers |
| --- | --- |
| [Core concepts](concepts.md) | How plans, cache modes, canonical datasets and return periods work |
| [Portfolio evaluation and impacts](portfolio.md) | How to evaluate assets and apply damage functions |
| [Parametric covers](parametric.md) | How to define parametric covers and payout functions, and evaluate their payouts |
| [Fitting](fitting.md) | Which fit method to use, and how validation and diagnostics work |
| [Bring your own data](bring-your-own-data.md) | How to canonicalize your own tables, Zarr arrays or GeoTIFFs |
| [Runtime](runtime.md) | Extras, package layout, DuckDB tuning, private buckets, PMTiles |

## Data sources

| Page | Sources |
| --- | --- |
| [CRC Open](sources/crc-open.md) | Public pre-fitted curves (63 hazards) |
| [JRC](sources/jrc.md) | EFAS/GloFAS river flood, EDO drought |
| [ERA5](sources/era5.md) | Reanalysis annual extremes |
| [CMIP6](sources/cmip6.md) | Native CMIP6 and NEX-GDDP-CMIP6 projections |
| [Stations](sources/stations.md) | NOAA ISD and GHCNh temperature, ECCC design rainfall |
| [Observations](sources/observations.md) | USGS, CHIRPS, IBTrACS, ECCC daily, HYDAT, NOAA water levels and sea level |
| [Agriculture](sources/agriculture.md) | USDA Cropland Data Layer, Fields of The World |

## Background

| Page | Covers |
| --- | --- |
| [Design decisions](design-decisions.md) | Ensemble identity, time windows, probability semantics, registry, spatial support |
| [Data licences](data-licences.md) | How source terms are recorded, per-source summary |
| [Development](development.md) | Setup, checks, CI, releases, contributor credits |

## Related projects

- [crc-framework](https://github.com/RiskThinking/crc-framework-rs/tree/main/docs):
  distributions, fitting, impact contracts and risk metrics used by this SDK.
- [crc-docs](https://github.com/RiskThinking/crc-docs): notebooks, pipelines
  and AI skills that run end-to-end assessments with the SDK.

API details live in docstrings; `help(HazardDataset.era5)` and similar are the
authoritative reference for parameters.
