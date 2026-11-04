# Data licences

[Docs index](README.md) · [Project README](../README.md)

This page summarizes how the SDK records source terms. It is not legal advice:
check each publisher's current terms before redistributing derived data.

## Principles

- The SDK **code** is AGPL-3.0-or-later. That licence does not attach to data
  the code produces.
- Derived canonical files are adaptations of their sources and inherit each
  source's attribution and "indicate changes" duties.
- Every canonical file records `source.licence`, `source.attribution`,
  `retrieved_at` and `checksum` (schema 1.3), so outputs self-describe.
- Adapters that read a licence from the source (CMIP6, NEX-GDDP-CMIP6) warn
  with `RestrictiveLicenceWarning` on share-alike or non-commercial terms, and
  `strict_licence=True` refuses them before any data is fetched. Unrecognised
  licences are recorded as `unknown`, never assumed.
- Licences change over time (Copernicus in 2025, NASA NEX in 2022): the
  retrieval date and checksum are the audit trail.

## Sources

| Source | Licence recorded | Notes |
| --- | --- | --- |
| CRC Open dataset | CC-BY-NC-4.0 | Attribution; commercial use needs a separate licence |
| ERA5 (Copernicus) | CC-BY-4.0 | Attribution: "Contains modified Copernicus Climate Change Service information [year]" |
| CMIP6 (Pangeo mirror) | Read from each store | Per modelling centre; e.g. NOAA-GFDL is CC-BY-SA-4.0. CMIP6 terms of use also apply |
| NEX-GDDP-CMIP6 (NASA) | CC0 archive plus the file's original model licence | Cite NASA and the modelling group |
| JRC EFAS / GloFAS flood | CC-BY-4.0 | Cite JRC and the product version |
| JRC EDO drought | CC-BY-4.0 | Cite EDO and the product version |
| ECCC IDF, GeoMet daily, HYDAT | Open Government Licence – Canada | Keep the attribution statement; do not imply endorsement; label refits as derived |
| NOAA ISD | NOAA open data terms | Attribution; no implied endorsement; label modified data. Some national contributions may carry originator terms (WMO Resolution 40) |
| NOAA GHCNh | CC0-1.0 (AWS registry) | Attribution requested; label modified data |
| NOAA CO-OPS, IBTrACS; USGS | U.S. government works | Attribution requested; follow each service's citation guidance |
| CHIRPS v3 | CC-BY-4.0 | Cite the Climate Hazards Center |
| OS-Climate layers | Read from layer metadata | Per layer |

Share-alike (`-SA`) and non-commercial (`-NC`) sources must not be pooled into
outputs released under incompatible terms. Keep connector-derived files
per source and per model.
