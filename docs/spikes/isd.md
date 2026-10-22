# NOAA ISD historical station observations

## Access decision

Use the anonymous AWS NODD CSV archive:
`https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}.csv`.
Station IDs are the eleven-digit USAF + WBAN identifiers, including leading
zeroes. A live access check of Toronto Pearson (`71624099999`, 2020) returned
HTTP 200, 5,941,107 bytes, ETag `b22be37b6337d170c326ec37c45b95bd`.
CSV columns include DATE, STATION, LATITUDE, LONGITUDE, ELEVATION and TMP.
TMP contains signed tenths of degrees Celsius and a quality code; +9999
is missing. Codes 1 and 5 mean all quality checks passed.

ISD is a historical archive: NOAA states updates stopped on 2025-08-24 and
NCEI HTTPS/FTP service retirement was planned for 2026-07-31. AWS remains the
advertised route. Complete-year requests therefore stop at 2024.

## ISD versus GHCNh

GHCNh replaces ISD, incorporates more historical holdings, has a different
station identifier/element schema and is the appropriate source for current
observations. ISD's established CSV schema and AWS station/year objects suit
historical validation. This adapter does not silently substitute GHCNh or
claim that ISD is still updated. GHCNh needs its own decoder and access spike.
The AWS registry describes approximately 600 GB uncompressed and over 35,000
stations over the archive's history; availability varies by station/year.

## Reduction and reproducibility

Stream selected CSV columns through Arrow into DuckDB. Reduce quality-filtered
observations to UTC days, counting distinct valid hour bins rather than rows.
Cache daily min/max, valid hours and station metadata, never hourly files.
Exclude incomplete days and calendar years explicitly, including absent days
and leap days in the denominator. Missing station/year objects are errors.

Pin the archive identity as `isd-2025-08-24`; this is the documented archive
cutoff, not an immutable NOAA release. Record each request URL, raw SHA-256,
ETag, Last-Modified, retrieval time and reduction policy in the manifest.
Offline/reuse reads verify reduced-file checksums. Refresh retrieves a new
snapshot. A checksum detects changes; it cannot retrieve an old upstream object.

The initial recipes describe extremes of sampled air temperature. ISD TMP is
instantaneous temperature, so these are labelled `isd_temperature_max` and
`isd_temperature_min`, not ETCCDI TXx/TNn based on daily extrema instruments.
Precipitation accumulation periods overlap and gust reporting is intermittent;
those recipes require separate completeness rules. Station curves retain WKB
points and use H3 only for indexing; no spatial interpolation is implied.
Voronoi coverage, geographic station discovery and gridded comparisons remain
separate work.

## Terms

The AWS registry's NODD terms allow public use, request attribution, prohibit
implied NOAA endorsement and require modified data to be distinguished from
unaltered NOAA data. Outputs say "Derived from NOAA ISD". Do not relabel this
mixed-origin archive CC0. WMO Resolution 40 restrictions on some national
contributions are not resolved by this access check; consult originator terms
before redistributing those contributions.

## Primary references

- [NOAA service notice](https://www.nesdis.noaa.gov/news/service-location-change-integrated-surface-data-global-hourly)
- [AWS ISD registry and terms](https://registry.opendata.aws/noaa-isd/)
- [NOAA GHCNh product](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly)
- [ISD format and quality codes](https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf)

Verified 2026-10-22.
