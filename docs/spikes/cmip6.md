# CMIP6 access evaluation

Date: 2026-10-15. Question: what is the cheapest *machine-access* route to
daily CMIP6 output for the annual-extreme recipes ERA5 already supports, with
no ESGF account and no per-file NetCDF wrangling?

## Routes compared

| Route | Auth | Layout | Verdict |
|---|---|---|---|
| **Pangeo Zarr mirror on AWS Open Data** (`s3://cmip6-pds`, index `pangeo-cmip6.csv`) | None (anonymous S3; the CSV also over plain HTTPS) | One consolidated Zarr v2 store per dataset, `(time, lat, lon)` with `_ARRAY_DIMENSIONS`, longitudes 0…360 | Used. One store per (model, experiment, member, variable, grid, version); the CSV is the only practical index. |
| Same mirror on GCS (`gs://cmip6`) | None | Identical layout | Equivalent; not wired up. A second `CMIP6Archive` is all it would take. |
| ESGF search API + OPeNDAP/HTTP NetCDF | Account for some nodes; federated, frequently down | One NetCDF per variable per time chunk | Rejected: availability and auth make it unusable as a default path. |
| NEX-GDDP-CMIP6 (`s3://nex-gddp-cmip6`) | None | Downscaled, bias-adjusted 0.25°, NetCDF per variable-year | Deliberately out of scope here: it is a *different product* (downscaled and bias-adjusted), and `ensemble.downscaling_method`/`bias_adjustment` would have to be populated. A natural second provider, not a store of this one. |

## Findings that shaped the adapter

* **There is no store, only a catalogue.** Institution, grid label and DRS
  version cannot be known in advance, so a dataset has to be looked up. The
  published index is an 80 MB CSV, which is why it is read with DuckDB (already
  a core dependency) rather than pulled into memory, and why a plan's cache
  directory keeps the `table_id='day'` subset as Parquet: one download, then
  local lookups. Grid preference is `gn, gr, gr1, gr2`; `latest` takes the
  newest version; anything still ambiguous is an error listing the candidates,
  because silently picking one would change the answer without saying so.
* **Model calendars are not the civil calendar.** CMIP6 `day` output runs
  `noleap` (365 days), `360_day` (twelve 30-day months), `all_leap`, or a
  Gregorian variant, declared on the `time` variable. Decoding a `360_day` axis
  as real dates drifts five days a year and mislabels blocks within three years,
  so time is decoded into the model's own calendar and blocks are cut on
  calendar day ordinals (`CFCalendar`). `TemporalWindow.calendar` records what
  was used; `all_leap` has no spelling there and is recorded as `unspecified`.
  `julian` is refused rather than approximated.
* **Stamps are day-centred.** Daily values are commonly stamped at 12:00 as
  `0.5, 1.5, …` days since the origin, so elapsed time is floored: a stamp
  anywhere inside a day belongs to that day. `hours`/`minutes`/`seconds since`
  are accepted too.
* **Samples are placed by date, not by position.** A missing step in the record
  must stay missing: filling it by sliding later days forward would silently
  corrupt a rolling k-day statistic (Rx5day) while leaving Rx1day looking fine.
  Tested both ways.
* **Units must be checked, not assumed.** `pr` is a flux in `kg m-2 s-1`
  (×86400 → mm/day) and `tasmax`/`tasmin` are kelvin. A store declaring
  anything else is rejected before any conversion runs, rather than producing
  plausible numbers off by 86400.
* **Chunks usually span the globe.** A small area still pays for whole chunks,
  exactly as with ARCO-ERA5, which is why the per-year annual-extreme cache is
  the same design: pay once, keep a few kilobytes per year.
* **Licence is per-source.** Most CMIP6 output is CC-BY-4.0, but the licence is
  a global attribute of each store, so it is read from the store and carried
  into the canonical file rather than hard-coded. The obligation attaches to
  the *derived file*, not to the act of reading the store, so the check runs on
  every path that produces or returns a dataset -- a warm year cache and an
  already-materialized canonical file both carry the recorded identifier and
  are enforced from it, with no network access.

## Caveats to carry into any use of the results

* A climate model is a model world. Use these for **change** signals — compare
  `historical` with a scenario, or decades within one — not as design values.
* One member of one model is one realisation. ADR-0001 keeps that explicit
  (`ensemble.pooling="single_member"`); the spread across models and members is
  itself the signal, and belongs in a comparison between canonical files.
* A ~1 degree grid cell is a large area average: orography is smoothed and
  convective rainfall extremes are damped more than in reanalysis.
* Short windows barely constrain a three-parameter GEV. `return_period_support`
  bounds what the fitted curve can honestly be read at; `BlockExtremaPolicy`
  defaults to requiring 20 years for a reason.
