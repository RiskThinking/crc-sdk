# CRC SDK release notes

[Docs index](README.md)

## 1.1.0

Faster reads and evaluation with less memory and disk, built-in impact models
that state their calibration facts, more data accepted from NEX-GDDP-CMIP6,
station and JRC sources, and support for DuckDB 2.x. This release requires
`crc-framework>=1.1,<2`. Dataset, evaluation and other artifact formats keep
their versions.

### Performance

Results, metadata and reports are the same as in 1.0.1, apart from retrieval
times.

- Evaluation reads only the rows its assets can match. CRC Open skips files
  whose extent misses every asset, and local Dataset files skip row groups
  whose statistics exclude the selection and decode only the selected rows.
  One site against a local 163k-row file takes 0.07 s instead of 5.6 s.
- CRC Open parses only the release index entries a selection can use. The
  first read in a process takes 0.68 s instead of 3.5 s.
- Evaluating many assets repeats less work. 10,000 sites take 92 s instead of
  147 s.
- Long streams check row keys out of core and compress replay batches. A full
  read of one CRC Open variable (4.25M rows) takes 14 s instead of 38 s and
  482 MB of temporary disk instead of 3.1 GB.
- `Runtime(workers=...)` now also downloads ahead in parallel: CRC Open files,
  GHCNh and ISD station-years, JRC return-period windows and Zarr chunks. One
  worker keeps the serial behaviour.
- Downloaded files move into the execution scope instead of being copied.
  Cached snapshots are copy-on-write clones on APFS, Btrfs and XFS.
- Arrow curves decode column by column through the framework's typed
  constructors.
- `crc-framework` 1.0.1 and later build and query numerical laws natively.

### Impact models

`crc-framework` 1.1 records the calibration facts of seven built-in impact
models, so they bind to data that declares the same measurement and refuse
data that does not.
[Framework 1.1.0 release notes](https://github.com/RiskThinking/crc-framework-rs/blob/main/docs/releases.md)

- Fire-weather damage applies only in the wildland-urban interface.
  `BuiltinImpact.FIRE_WEATHER_INDEX` requires
  `WildfireContext(wildland_urban_interface=...)`, usually bound to a boolean
  asset column, and assets outside the interface get zero damage.
  [Evaluation](evaluation.md)
- `BuiltinImpact.HOT_DAYS` is retired and fails data binding with
  `model_retired`.
- The evaluation schema references impact context format 1.1, which adds the
  wildfire context. Every existing evaluation artifact stays valid.

### Data accepted

- NEX-GDDP-CMIP6 reads models on their native 365-day and 360-day calendars.
  1.0 accepted only Gregorian calendars, which excluded 20 of 34 models.
- GHCNh and ISD recipes take `daily_coverage`, the share of hours a day needs.
  Both coverage rules still default to complete coverage, and the
  `incomplete_days` warning names the settings that relax them.
- JRC flood reads return every flooded pixel. A return period that a flooded
  pixel lacks is read as 0 m, so pixels flooded at only one return period are
  no longer dropped. Dry pixels are reported in their own note.

### Fixes

- A cache that keeps a superseded CRC Open release (`Cache.reuse`) names that
  release and the fix (`Cache.refresh`) instead of reporting only two digests.
- Closing a prefetching read early never waits for its download threads, so a
  garbage-collected read can no longer deadlock.
- Evaluation artifacts compare -0.0 and 0.0 match distances consistently
  however many assets they hold.
- Reopening an artifact that keeps a boolean context column in its output no
  longer reports a contradiction with the recorded context.

### DuckDB

`duckdb>=1.4,<3` is supported, and new installs resolve 2.x. DuckDB 2.0
changes some SQL and connection behaviour for your own queries.
[DuckDB versions](duckdb.md) lists each change and how to stay on 1.x.

## 1.0.1

Source reads now report what 1.0.0 left silent or misattributed. No interface
or format changes.

- NEX-GDDP-CMIP6 selections that reach past a scenario's published years add a
  `source_year_unavailable` warning naming those years.
- JRC flood reads whose area misses the product extent add an
  `area_outside_coverage` warning instead of an empty, complete report.
- NEX-GDDP-CMIP6 resources record the CMIP6 licence that NASA writes in each
  file's `cmip6_license` attribute.
- GHCNh and ISD station lists and hourly files are attributed to NOAA NCEI only.

## 1.0.0

This breaking release requires Python 3.11 or later and `crc-framework` 1.x.
Import data and workflow interfaces from `crc_sdk` and numerical, measurement,
and impact interfaces from `crc_framework`. To upgrade existing code, follow the
[1.0 migration guide](migration-v1.md).

- Use your own variable and scenario IDs. Declare scientific meaning using
  typed measurements and probability records. Leave scenario or period absent
  when the source does not establish them. Use a catalog when you want to
  share exact definitions. [Scientific declarations](scientific-data.md)
- Read observations before you choose a fit. A completed sample result carries
  the data and source identity needed for fitting. Query supplied quantiles
  directly. Result terminals run upstream work automatically, and an execution
  scope can share work across validation, reads, and saving. [Execution](execution.md)
- Compose independent datasets and query them at assets. Spatial matching uses
  the original data support by default. Each named impact response keeps its
  own measurement, units, and resolved model. [Collections](collections.md) ·
  [Evaluation](evaluation.md)

Dataset and evaluation artifacts use format `2.0`. Portable collection bundles,
native evidence, and taxonomy catalogs use format `1.0`. The package reads
files in these formats. [Version ledger](../crc_sdk/contracts/api/release.json)

### CRC Open availability

CRC Open is published in Dataset 2.0 format. `CRCOpen()` follows the public
release pointer, and every file is pinned by checksum in the release index.
Pass `CRCOpen(catalog=...)` to read a different release-index 1.1 document or
pointer. [CRC Open discovery](sources.md#discover-a-crc-open-release)
