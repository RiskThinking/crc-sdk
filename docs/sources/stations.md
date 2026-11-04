# Station sources: NOAA ISD, NOAA GHCNh, ECCC IDF

[Docs index](../README.md) · [Project README](../../README.md)

Station adapters produce canonical curves at **station points**. Curves keep
WKB points and stable station source IDs; H3 is only an index and implies no
coverage of surrounding land. No interpolation, nearest-station assignment or
homogenization is performed. Requires `crc-sdk[geometry]`.

## Selecting stations

All three adapters accept explicit IDs or WGS84 boxes:

```python
plan = (
    HazardDataset.ghcnh("temperature_max")
    .for_area((-87.94, 41.95, -87.92, 41.97))     # or .for_areas([box1, box2])
    .years(2000, 2024)
    .cache("data/ghcnh", mode="reuse")
    .canonicalize()
)
selected = plan.stations()          # station metadata only; no observation downloads
pinned = plan.resolve_stations()    # new plan with explicit station IDs
pinned.materialize("data/ghcnh-curves.parquet")
```

`.for_stations(ids)` replaces area selection and `.for_area(...)` replaces IDs.
Several boxes select a union without duplicates. Catalogues are cached as
checksummed Parquet; offline selection needs a cached catalogue. An empty
selection raises before any observation is fetched. Catalogue locations can
differ from historical observation coordinates, so inspect the annual audit.

## NOAA ISD and GHCNh hourly temperature

```python
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

plan = (
    HazardDataset.isd("temperature_max")          # or HazardDataset.ghcnh(...)
    .for_stations(["71624099999"])                # Toronto Pearson, USAF + WBAN
    .years(2000, 2024)
    .quality(minimum_hour_fraction=0.75, minimum_day_fraction=0.9)
    .execution(workers=4, memory_limit="256MB")   # memory limit per worker
    .cache("data/isd", mode="reuse")
    .canonicalize(policy=BlockExtremaPolicy.curated(
        minimum_years=20, diagnostics="data/isd-fit-diagnostics.parquet",
    ))
)
plan.prefetch()
plan.write_annual_extremes("data/isd-annual.parquet")   # QA table, bounded batches
stations = plan.materialize("data/isd-curves.parquet")
```

| | ISD | GHCNh |
| --- | --- | --- |
| Status | Archive; updates ended August 2025, so complete years stop at 2024 | Current successor, updated daily |
| Station IDs | 11-digit USAF + WBAN | 11-character GHCN IDs (some with hyphens); [station list](https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/doc/ghcnh-station-list.csv) |
| Access | Anonymous AWS CSV per station/year; selected columns streamed | Anonymous AWS Parquet per station/year; 8 projected columns via range reads |
| Values | `TMP` in tenths °C with QC code | °C with integrated and source-specific QC flags |
| Default QC | Codes 1 and 5 | Blank flags and source-specific good legacy codes (5 is rejected where it means "removed") |
| Source version | `isd-2025-08-24` (archive cutoff) | `ghcnh-v1` (product major version, not a frozen release) |
| Licence | NOAA open data; attribution, no endorsement, label modified data | CC0-1.0 per the AWS registry; attribution requested |
| Hazard names | `isd_temperature_max/min` | `ghcnh_temperature_max/min` |

Shared behaviour:

- `temperature_max` fits GEV L-moments; `temperature_min` fits lower-tail
  Gumbel L-moments. Both describe extremes of **sampled** hourly temperature,
  not instrument daily TXx/TNn.
- Days are UTC. A valid day needs 18 distinct valid hour bins (by default) and
  a valid year needs 90% valid days; duplicates cannot inflate coverage.
- Only daily summaries are cached (shared between recipes), never hourly files.
  Missing station/years (HTTP 404) and incomplete years become null annual
  values; other network errors raise.
- Manifests record request URLs, ETag, Last-Modified, raw or projected-column
  fingerprints and reduced-file SHA-256. A checksum detects upstream changes;
  it cannot recover an old upstream object.
- Station coordinate jumps above 0.05° are rejected; smaller moves still need
  review. There is no automatic ISD↔GHCNh crosswalk or fallback.

## ECCC design rainfall (IDF)

```python
plan = (
    HazardDataset.eccc_idf("1h", province="ON", version="3.40")
    .for_stations(["6158731"])
    .cache("idf-cache")
)
plan.prefetch()
published = plan.source_curves()          # central mm/h rates and 95% half-widths
curves = plan.materialize("rainfall-1h.parquet")
plan.duration("24h").cache("idf-cache", mode="offline").materialize("rainfall-24h.parquet")
```

- Durations: `5min`, `10min`, `15min`, `30min`, `1h`, `2h`, `6h`, `12h`, `24h`.
  Each becomes `RainfallIntensity/<duration>`.
- Release `3.40` stages the provincial ZIP once (about 149 MB for Ontario), then
  keeps only small station files. Archived `3.20` reads station members by HTTP
  range. Releases are never substituted automatically.
- Curves are Gumbel quantile refits of published central rates (default
  absolute residual limit 0.15 mm/h), **derived from** ECCC IDF, not official
  ECCC curves. Each record keeps its own midpoint horizon.
- A 24-hour moving maximum is not calendar-day Rx1day, and rainfall is not
  flood depth. Data are under the Open Government Licence – Canada.

Worked notebooks and pipelines for these sources are in
[crc-docs](https://github.com/RiskThinking/crc-docs#readme).
