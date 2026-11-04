# CMIP6 and NEX-GDDP-CMIP6 projections

[Docs index](../README.md) · [Project README](../../README.md)

Both use the four ERA5 recipes (`txx`, `tnn`, `rx1day`, `rx5day`) with the same
definitions, so a historical baseline and a scenario are directly comparable.

## CMIP6 (native model grids)

Daily model output is read anonymously from the Pangeo Zarr mirror on AWS Open
Data (`s3://cmip6-pds`), requiring `crc-sdk[zarr,geometry]`. Only per-year
annual extremes are cached.

```python
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

plan = (
    HazardDataset.cmip6("rx1day", model="GFDL-ESM4", scenario="ssp585", member="r1i1p1f1")
    .for_area((-125.0, 42.0, -116.0, 49.0))
    .years(2021, 2050)
    .cache("cache/cmip6", mode="reuse")
    .canonicalize(policy=BlockExtremaPolicy.curated(h3_resolution=5))
)
hazard = plan.materialize("hazards/cmip6-rx1day-ssp585.parquet")
```

- **Datasets are looked up.** `CMIP6Catalogue` resolves (model, experiment,
  member, variable) to exactly one Zarr store, preferring the native grid
  (`gn`, then `gr`, `gr1`, `gr2`) and the newest version. Ambiguity raises with
  the candidates listed. The 80 MB index CSV is read with DuckDB and its daily
  subset is cached. Pin with `.with_grid("gr")` and `.version("20180701")`.
- **Model calendars are respected.** `noleap`, `360_day`, `all_leap` and
  Gregorian axes are decoded with `CFCalendar`; samples are placed by date, so
  gaps stay gaps. The calendar is recorded in `temporal_window.calendar`.
  `julian` is refused.
- **Units are checked.** `pr` must be `kg m-2 s-1` (×86400 → mm/day) and
  temperatures kelvin; anything else is rejected.
- **The pathway is the experiment.** `scenario` is the CMIP6 `experiment_id`
  (`historical`, `ssp126`, …), checked against the `crc_sdk.pathways` crosswalk.

One model, one member, one pathway per dataset: build one file per model and
compare files rather than pooling. Because `horizon` is the window centre, a
scenario file and a historical file join on
`(hazard_name, horizon, pathway, cell_index)` for change maps.

### Licences

Licences are set per modelling centre and read from each store. NOAA-GFDL, for
example, publishes CC BY-SA 4.0. The adapter records the store's licence, warns
(`RestrictiveLicenceWarning`) on terms beyond attribution, and refuses them with
`strict_licence=True`. The check applies wherever a dataset is produced or
returned, including warm caches.

### Caveats

A climate model is a model world: use results as change signals, not design
values. One member of one model is one realisation. A ~1° cell smooths
orographic and convective extremes more than reanalysis.

## NEX-GDDP-CMIP6 (downscaled, bias-adjusted)

NASA's 0.25° product is a different dataset from native CMIP6. It streams
annual extremes for one model and member (`crc-sdk[netcdf]`):

```python
plan = HazardDataset.nex_gddp_cmip6(
    "rx1day", model="ACCESS-CM2", scenario="historical", version="1.0",
).for_area((-79.5, 43.5, -79.25, 43.75)).years(2014)
for batch in plan.annual_extremes():
    print(batch)            # Arrow batches, one bounded spatial strip at a time
```

- File revisions `1.0`, `1.1` and `1.2` are selected explicitly; a missing
  revision fails rather than falling back. Coverage varies by model and year.
- Batches carry the source URL and retrieval time but no checksum or
  transactional snapshot. This interface yields annual samples, not fitted
  curves or portfolio evaluation.
- The [AWS registry](https://registry.opendata.aws/nex-gddp-cmip6/) states the
  archive is CC0, but older files can retain model licences such as CC BY-SA.
  Both are kept; non-CC0 originals warn, and `strict_licence=True` refuses them.
  Cite NASA and the source model.
- Bias adjustment does not remove model or downscaling uncertainty; 0.25° is not
  independent local detail.
