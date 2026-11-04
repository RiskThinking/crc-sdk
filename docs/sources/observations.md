# Observation sources

[Docs index](../README.md) · [Project README](../../README.md)

These adapters return source observations as Arrow batches for inspection.
They do **not** build canonical hazard curves, interpolate, or apply a quality
or completeness policy that the source does not define. Worked notebooks and
pipelines for each are in [crc-docs](https://github.com/RiskThinking/crc-docs#readme).

## Common pattern

```python
plan = SomePlan.for_station("...").cache("cache")   # lazy; explain() needs no network
source = plan.prefetch()                             # fetch and checksum a snapshot
for batch in plan.batches():                         # bounded Arrow batches
    ...
replay = plan.cache("cache", mode="offline", sha256=source["sha256"])
```

Most of these services are mutable or paginated and have no transactional
snapshot. Keep the cache and manifest, and replay with `mode="offline"` plus the
recorded SHA-256 (a single value, or a year/name → digest mapping where noted).
`refresh` deliberately fetches a new snapshot.

| Plan | Source | Selector | Output | Checksum covers |
| --- | --- | --- | --- | --- |
| `USGSStreamflowPlan.for_station("USGS-01491000").between(start, end)` | USGS daily values (parameter 00060, statistic 00003) | Monitoring location | Daily mean discharge, approval status, qualifiers | Canonical feature JSON lines |
| `CHIRPSMonthlyPlan.monthly(2020, 1).for_area(bounds)` | CHIRPS v3 final monthly COG (needs `raster`) | Year, month, bounds | Pixel centres with monthly mm; `-9999` excluded | Full COG (`cache`); `stream` mode cannot verify |
| `IBTrACSPlan("NA", Path("cache")).for_storms([...])` | NOAA IBTrACS v04r01 basin CSV | Basin, storm IDs | Track points, source columns as strings | Full CSV (`expected_sha256`) |
| `ECCCDailyPlan.for_station("6158731").years(a, b)` | ECCC GeoMet daily climate | 7-character climate ID | Daily measurements and flags | Per-year canonical records |
| `HYDATDailyPlan.for_station("02HA014").years(a, b)` | ECCC GeoMet HYDAT | Station number | Daily mean flow (m³/s), level (m), data symbols | Per-year canonical records |
| `NOAAExtremeWaterLevelPlan.for_station("8638610")` | NOAA CO-OPS | Station ID | `source_info()`, `source_levels()`, `annual_extremes()` | Each of three responses |
| `NOAASeaLevelPlan.for_station("8638610").scenario("all")` | NOAA CO-OPS 2022 scenarios | Station ID | `projections()` in cm vs a 2005 baseline | The single response |

## Interpretation limits

- **USGS:** daily mean discharge is not inundation depth or instantaneous peak
  flow. Credit USGS [Water Data for the Nation](https://doi.org/10.5066/F7P55KJN).
- **CHIRPS:** 60°S–60°N from 1981; a grid estimate is not design rainfall or
  flood depth, and a version label does not pin bytes. CC BY 4.0,
  [Climate Hazards Center](https://doi.org/10.15780/G2JQ0P).
- **IBTrACS:** files are updated in place; basin files can overlap and agency
  wind definitions differ. Track points are not asset winds or return-period
  curves.
- **ECCC daily:** the climate ID is not the web station ID. Local-date labels do
  not imply a shared UTC window; records are not homogenized (AHCCD).
- **HYDAT:** the level datum is not supplied by this endpoint and is recorded as
  unspecified. Gauge level is not flood depth or elevation; daily mean flow is
  not peak flow.
- **NOAA water levels:** published annual exceedance probabilities and heights
  relative to NOAA's stated datum and epoch. 90% confidence limits are not
  extra probability knots. The annual series is detrended and cannot be shifted
  to another datum like an ordinary measurement.
- **NOAA sea level:** 2022 interagency scenarios at U.S. stations, relative to a
  **2005** baseline (some tools use 2000). Scenario names are not emissions
  pathways or exceedance probabilities. A paginated response is rejected rather
  than returned partially.

ECCC sources are under the Open Government Licence – Canada; NOAA and USGS
sources are U.S. government works with attribution requested.
