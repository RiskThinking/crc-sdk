# ECCC short-duration rainfall IDF

## Access

Use ECCC's anonymous HTTPS [engineering dataset archive](https://collaboration.cmc.ec.gc.ca/cmc/climate/Engineer_Climate/IDF/),
linked from its [engineering climate datasets page](https://climate.weather.gc.ca/prods_servs/engineering_e.html).
No account, API key or new dependency is needed. Select a province/territory and
an explicit release; never resolve an unpinned "latest" during execution.

- `3.40`: `idf_v3-40_2025-12-5.zip`, 697,816,512 bytes. Contains provincial
  ZIP members; Ontario's member is 149,139,781 uncompressed bytes. HTTP byte
  ranges read the outer directory and selected provincial member only. The
  compressed provincial member must be staged to a temporary seekable file;
  the archive cannot provide individual station range reads through that layer.
- `3.20`: archived `IDF_v-3.20_2021_03_26_{province}.zip` files. Range reads
  retrieve the directory and selected station TXT members directly. A Toronto
  station TXT member is 3,198 compressed bytes. This is an explicitly older
  baseline, useful for repeatable release comparisons, not a current substitute.

ZIP station files contain bilingual text, annual maxima, return-period amounts,
rates with confidence limits, and duration interpolation coefficients. Read only
TXT members; do not extract arbitrary archive paths. Match climate IDs as strings
(they may include letters), not NOAA station identifiers. Cache raw station bytes
and their SHA-256, member path, URL, release, retrieval time and HTTP validators.
Reuse/offline verifies checksums; refresh rejects changed bytes under an already
cached release. ECCC does not guarantee content-addressed archive retention.

## Scientific contract

Table 2b gives published central rainfall **rates in mm/h** separately from
95% confidence half-widths. Parse the return-period header, never infer labels
from array length. Initial durations are 5, 10, 15, 30 minutes and 1, 2, 6, 12,
24 hours. One canonical hazard per duration. Preserve source rates and confidence
limits in an inspection table; fit central rates as probability-labelled knots,
using the annual non-exceedance convention `p = 1 - 1/T`. These are not samples.
Use quantile least squares with Gumbel, matching the published family without
claiming to reproduce ECCC's method-of-moments estimator or official numbers.

Station coordinates are degree/minute hemisphere coordinates. Retain WKB points;
H3 is an index, with no interpolation or area of representativeness. Individual
records retain their start/end years and duration-specific sample counts. A
record's midpoint is its horizon; different records are not one common baseline.
24-hour moving-window amounts are not calendar-day Rx1day. Rainfall rates are
neither flood depths nor climate-adjusted IDF_CC projections. Release-to-release
changes include record and methodology changes, not solely climate trends.

## Terms

The [Open Government dataset record](https://open.canada.ca/data/en/dataset/15aa455c-9f9f-4405-8eee-4445d02fd35c)
identifies the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).
Carry ECCC credit and "Contains information licensed under the Open Government
Licence – Canada"; label fitted outputs as derived from ECCC IDF. The data licence
is independent of the SDK software licence.

Verified against the archive listings, HTTP range access and a station TXT file.
