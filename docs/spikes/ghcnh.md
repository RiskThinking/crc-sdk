# NOAA GHCNh station observations

## Access and format

NOAA's Global Historical Climatology Network hourly (GHCNh) replaces ISD.
Use anonymous station/year Parquet objects:
`https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/access/by-year/{year}/parquet/GHCNh_{station}_{year}.parquet`.
The matching PSV and whole-period station products are not needed for this
adapter. Parquet supports column projection and HTTP range reads.

A live check of Chicago O'Hare `USW00094846`, 2020 returned HTTP 200,
1,137,792 bytes, ETag `0d056d6911c7d407d9749b1fd640800e`. There are over
300 columns, including STATION, DATE, LATITUDE, LONGITUDE, ELEVATION,
temperature, temperature_Quality_Code and temperature_Source_Code.
Parquet temperature values are degrees Celsius (e.g. -2.2), not integer
ISD tenths. Elevation -999.9 is missing. Read only the eight needed columns
in Arrow batches and feed them into the common DuckDB daily reducer.

The archive is updated daily, including historical objects. Record the
product major version, exact station/year URL, ETag, Last-Modified, size,
retrieval time, projection fingerprint and reduced-file SHA-256. HTTP range
requests use If-Match to reject changes during a read. Offline/reuse verifies
the daily cache; refresh retrieves the current upstream snapshot. A product
version is not an immutable data release. Complete-year requests exclude the
current UTC year, without claiming that all older years are complete.

## Compatibility and quality

GHCN station IDs are eleven characters, beginning with country and network
codes; some include hyphens. Do not infer them by reformatting ISD's eleven
numeric USAF + WBAN IDs. NOAA's station list contains names, WMO and ICAO
identifiers, but station history/harmonization can make a crosswalk ambiguous.
Use explicit GHCNh IDs and retain distinct source IDs and hazard names.

The default temperature policy accepts blank flags (no failed integrated QC)
and source-specific legacy good codes: 1 for sources 220, 221, 222, 223, 313,
314, 315, 322, 335, 343, 344, 346, 347, 348; 5 only for sources 313, 314,
315, 322, 335, 343, 344, 346. For sources 220–223/347–348, 5 means removed
and must be rejected. Other flags, combined flags and unknown numeric/source
combinations are rejected. User overrides can narrow the accepted set.
The accepted flags are recorded in the cache identity and source request.

Use the same UTC distinct-hour/day completeness rules, daily-summary schema,
annual audit, station-point support and block-extrema fitter as ISD. Indicators
are `ghcnh_temperature_max` and `ghcnh_temperature_min`, describing extremes of
sampled temperature rather than instrument daily TXx/TNn. This compatibility
does not imply identical samples or fitted curves across products.

## Terms and primary evidence

The GHCNh AWS registry explicitly labels NODD data CC0-1.0, requests attribution,
prohibits implied endorsement and requires modified data to be distinguished.
Store that product-specific label and a derivation notice.

- [NOAA GHCNh overview](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly)
- [AWS registry and terms](https://registry.opendata.aws/noaa-ghcnh/)
- [NOAA format, units and source-specific QC](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf)
- [Station catalogue](https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/doc/ghcnh-station-list.csv)

Verified 2026-10-22.
