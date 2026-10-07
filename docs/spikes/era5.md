# ERA5 access evaluation

Date: 2026-10-02. Question: what is the cheapest *machine-access* route to
hourly ERA5 for annual-extreme recipes, with no account and no GRIB tooling?

## Routes compared

| Route | Auth | Layout | Verdict |
|---|---|---|---|
| **ARCO-ERA5 on GCS** (`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`) | None (anonymous gcsfs) | Zarr **v2** format despite the name; native 0.25°, hourly, `(time, lat, lon)`, **chunk = 1 hour × global field**, 1900–2050 skeleton (data from 1940) | Works; used as the native store. Cost is chunk count: a year is ~8,760 global-field reads whatever the area, ~20 chunks/s measured, so **minutes per variable-year**. |
| `co/single-level-reanalysis.zarr-v2` (same bucket) | None | Reduced-Gaussian flat `values` axis, time-chunk 1 | Rejected: not a regular lat/lon grid. |
| `ar/1959-2022-6h-1440x721.zarr` | None | 6-hourly, chunk 1 | Rejected: 6-hourly samples miss the diurnal peak, so TXx is biased low. |
| **WeatherBench2 regridded copies** (`gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_…conservative.zarr`, also 360×181) | None | Hourly, 1.5° (1°), chunk **8 hours** × global field, dims `(time, longitude, latitude)`, lat ascending −90…90 | Works; used as the laptop store. ~30 s per variable-year (≈21 s effective with four years in flight). Stops at 2022. |
| CDS ARCO / CDS API | CDS token + licence acceptance, rate limited | Time-series layouts exist | Requires authenticated access; the adapter uses anonymous Zarr stores. |
| `s3://era5-pds` | — | — | Not reachable anonymously (403) at test time. |

## Findings that shaped the adapter

* **Axis order differs between stores** (`time, lat, lon` vs `time, lon, lat`),
  and longitudes run 0…360. The reader follows `_ARRAY_DIMENSIONS`, normalises
  longitudes to −180…180 and stitches the two column runs at the 0° seam.
* **ARCO is a growing store.** `valid_time_stop` (final) and
  `valid_time_stop_era5t` (preliminary, ~2 months behind real time) are group
  attributes. The adapter uses **final data only**, refuses years outside the
  complete final record, and versions a cache by `final-through-<date>`, so it
  stays valid while the store appends.
* **ERA5T** is excluded; the adapter accepts only final data.
* `total_precipitation` is an hourly accumulation in metres over the hour
  **ending** at the stamp, so the hour stamped 00:00 belongs to the day that
  just ended (`DailyAggregation.interval_end_stamps`). Tested.
* `2m_temperature` is instantaneous hourly: the "daily maximum" is the largest
  of 24 hourly values and slightly underestimates the true extreme.
* Daily boundary is **UTC** (`DailyAggregation.utc_offset_hours` exists, but
  recipes use UTC).
* `land_sea_mask` exists in both stores (a time-indexed copy in ARCO); a cell
  is land at ≥ 0.5.
* The Zarr copies are v2 format, readable by zarr 2.18 and 3.x; the loader
  falls back when `zarr_format` is not accepted.
* Licence: CC-BY-4.0 since July 2025 (see `licence-policy.md`); recorded as
  `licence`/`attribution`/`retrieved_at` on every canonical file. WeatherBench2
  copies are derived from the same reanalysis.

## Limitations

* **ERA5-Land**: no public anonymous ARCO store was found, so the second recipe
  family is not implemented. `ERA5Store` is the extension point (a store
  descriptor plus variable names).
* **0.25° quick runs**: the native store is too slow for a laptop demo; no
  cheaper public layout was found in this evaluation. Use small caches of
  annual extremes for repeated local runs.
* No live test runs in CI; unit tests use a synthetic ERA5-shaped Zarr.
