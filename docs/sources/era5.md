# ERA5 historical baselines

[Docs index](../README.md) · [Project README](../../README.md)

ERA5 hourly reanalysis becomes an annual-extreme baseline. Data is read
anonymously from public Zarr copies (`crc-sdk[zarr,netcdf,geometry]`), reduced
to one small file per year, and only those annual extremes are cached.

```python
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

plan = (
    HazardDataset.era5("txx", store="wb2-1p5")
    .for_area((-10.0, 36.0, 22.0, 56.0), land_only=True)
    .years(1991, 2020)
    .cache("cache/era5", mode="reuse")
    .canonicalize(policy=BlockExtremaPolicy.curated(h3_resolution=5))
)
hazard = plan.materialize("hazards/era5-txx.parquet")
annual = plan.annual_extremes()        # the per-year samples behind every curve
```

## Recipes

| Recipe | Definition (UTC days) | Default fit |
| --- | --- | --- |
| `txx` | Annual maximum of daily maximum 2 m temperature | GEV, sample L-moments |
| `tnn` | Annual minimum of daily minimum 2 m temperature | GEV, sample L-moments |
| `rx1day` | Annual maximum 1-day precipitation | GEV, sample L-moments |
| `rx5day` | Annual maximum 5-day precipitation | GEV, sample L-moments |

Override the fit through `BlockExtremaPolicy` ([fitting](../fitting.md)).

## Stores

| `store` | Grid | Speed | Coverage |
| --- | --- | --- | --- |
| `arco-0p25` (default) | Native 0.25°, ARCO-ERA5 on GCS | Minutes per variable-year (one global hourly chunk per read) | 1940 → latest final month |
| `wb2-1p5` | 1.5° WeatherBench2 copy | ~30 s per variable-year | 1959–2022 |

Only final (non-ERA5T) data is used; the cache is keyed by the final record's
end date, so it stays valid while the store grows. Precipitation is an hourly
accumulation over the hour **ending** at its timestamp. Daily maximum
temperature is the largest of 24 instantaneous values and slightly
underestimates the true extreme. `land_only` uses `land_sea_mask >= 0.5`.

## Metadata and caveats

Files carry CC-BY-4.0 licence and Copernicus attribution, retrieval time,
`temporal_window`, `ensemble.pooling="single_member"` and
`probability_semantics="annual_exceedance"`.

- Reanalysis blends observations and a model; convective rainfall extremes are
  underestimated.
- A grid cell is an area average, not a point.
- ERA5-Land is not supported (no anonymous ARCO copy); `ERA5Store` is the
  extension point.
