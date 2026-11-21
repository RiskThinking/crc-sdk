# Sources and native evidence

[Docs index](README.md) · [Execution](execution.md) · [Source terms](data-licences.md)

Import providers directly from `crc_sdk`. Constructors and builders describe
work without acquiring data. `inspect()` discovers supported indicators, actual
series combinations, stations, releases, and any selections still required.
`explain()` describes work and requirements without I/O. A terminal executes
the entire selected recipe.

## Choose the data kind

| Interface | What `read()` returns | Further scientific operation |
| --- | --- | --- |
| `Samples` | Genuine sample observations, including named annual indicators | Explicit framework sample fit |
| `Observations` | Native time observations, with their actual cadence when known | Typed temporal aggregation |
| `Projections` | Source time or return-period coordinates, point estimates and/or confidence intervals | Read and inspect the published evidence |
| `Layer` | Numeric or categorical values, labels, or geometry-only boundaries | Read and inspect spatial evidence |
| `Dataset` | Directly usable supplied quantiles or an explicitly fitted numerical law | Explicit framework quantile fit when needed |

The evidence kinds accept `from_table`. Use `Dataset.from_quantiles` or
`Dataset.from_raster` for supplied curves. Readers share `select`, `read`,
`batches`, `validate`, `execute`, and `write`. Open a saved file using the
type returned by `write`. For example, saving an ERA5 sample recipe returns
`Samples`, so use `Samples.open(path)` to reopen it.
Scientific transforms such as `rename`, `declare`, and `convert_units` also
return the canonical data kind. Choose a native product view before transforming
its scientific output.

| Provider | Data and capabilities |
| --- | --- |
| `CRCOpen` | Current canonical release datasets and their original quantile intervals |
| `JRC` | EFAS or GloFAS published flood return-period maps as `Dataset`; original pixels and quantile knots |
| `EDO` | Soil moisture index as native dekadal `Observations`; irregular monthly intervals have no invented fixed cadence |
| `ERA5`, `CMIP6`, `NEX` | Typed annual indicators as `Samples`; `observations()` exposes the selected underlying hourly/daily observations |
| `GHCNh`, `ISD` | Typed annual temperature indicators as `Samples`; `observations()` exposes native irregular, quality-filtered reports |
| `USGS`, `HYDAT` | Native daily gauge observations; discharge retains actual source units |
| `ECCCDaily` | Daily Canadian climate observations |
| `NOAASeaLevel` | Published time/scenario sea-level `Projections` |
| `NOAAWaterLevels` | Detrended annual water-level `Observations`; `levels()` returns central published exceedance curves and `confidence_intervals()` returns separate interval evidence |
| `ECCCIDF` | Central published rainfall IDF curves for an explicit duration; `confidence_intervals()` exposes the published 95% intervals |
| `CHIRPS` | Monthly CHIRPS v3 precipitation amounts as `Observations`, preserving raster pixel support |
| `IBTrACS` | Native storm point observations for an explicit `Agency`; event groups and positions remain distinct |
| `USDACropland` | Published categorical raster `Layer`, retaining numeric class codes, official labels, and actual delivered pixel geometry |
| `FieldsOfTheWorld` | Published field boundary `Layer` and field attributes; defaults to the latest discovered publication year |

Confidence intervals describe uncertainty about estimates. Their endpoints and
annual exceedance quantiles have distinct meanings. Storm tracks describe storm
positions. Projection times must fall within the series' declared period and
follow its calendar when specified. Categorical layers have no fitting method.

## Read an annual indicator before fitting

Inspect a provider to see its typed indicators, required selections, and
source details. ERA5 inspection applies the same series filters as reading.
An explicit empty selection returns an empty inventory and typed schema without
acquiring source data. Ordinary ERA5 discovery and reads use the `zarr` extra.
The following example reads ERA5's remote store.

<!-- live-example -->
```python
from crc_framework import DistributionFamily, SampleFit
from crc_sdk import Bounds, Cache, ERA5, Period, Runtime

runtime = Runtime(cache=Cache.reuse(".crc-cache"))
for indicator in ERA5().inspect(runtime=runtime).indicators:
    print(indicator.label, indicator.measurement)

indicator = ERA5.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION
area = Bounds(west=23.9, south=59.9, east=24.1, north=60.1)
period = Period.window(1991, 2020)
rain = ERA5(indicator=indicator).select(area=area, time=period)
inventory = rain.inspect(runtime=runtime)
for requirement in inventory.requirements:
    print(requirement.message)
annual_observations = rain.read(runtime=runtime)
print(annual_observations.report)
fitted = annual_observations.fit(
    SampleFit(family=DistributionFamily.GUMBEL_RIGHT),
)
saved = fitted.write("annual-rainfall.parquet")
```

The recipe decodes the native accumulation convention, produces daily totals,
and takes the largest daily total in each source-calendar year. The result is
one annual precipitation **amount** per support and year. Fitting the collected
result reuses those values and their pinned source identity. The five-day recipe
forms totals from consecutive five-day windows within each calendar year. Its
coverage fraction counts valid windows out of all eligible windows for that
year. Temperature recipes reduce daily maxima or minima before selecting the
annual extreme. No fitting occurs until `fit(...)`.

The default is `Coverage.complete()`. Incomplete days and years remain missing
observations with coverage and a reason; they do not silently enter the fit.
`Coverage.minimum_fraction(0.95)` explicitly permits a relaxed fraction without
rescaling missing values. Reports retain requested, usable, and incomplete year
counts. A fit record describes the usable observations supplied to the fitter.

ERA5's public ARCO store uses global spatial chunks for each hour. A small area
still requires those compressed source chunks. Explicit time bounds are
required, and inspection reports the native chunk layout and acquisition
estimate. Cache a deliberate selection before repeating a long calculation.

CMIP6 and NEX expose published combinations as `SeriesReference` records.
References from the catalog list the indicator, model, scenario, and member.
After selecting a combination, inspection reads its calendar and actual period.
This example uses the area and time window from above.

<!-- live-example -->
```python
from crc_sdk import CMIP6

choices = CMIP6(
    indicator=CMIP6.Indicator.ANNUAL_MAXIMUM_DAILY_PRECIPITATION,
).inspect(runtime=runtime).series
for choice in choices:
    print(choice.model, choice.scenario, choice.member)

# Use one of the references from the inventory.
recipe = CMIP6().select(series=choices[0], area=area, time=period)
print(recipe.inspect(runtime=runtime).series[0].period)
```

Use the same workflow with `NEX`. Filtering inspection by an indicator or model
can shorten the list. Inspect saved samples for a reference that includes the
verified payload release as well as the model, scenario, member, and period.
An unpinned, dated series can be read over a smaller observation window with
`select(series=reference, time=...)`. The series retains its declared period
and the metadata records the narrower observation window.

ERA5, CMIP6, NEX, JRC, EDO, and USDA Cropland inspection return release
references for the published inventory. Pass an inspected reference directly to
`select(series=...)` to check that inventory while choosing an area or observation
window. The completed read records a separate release reference for the exact
payload acquired. Reuse the same area and time bounds when replaying that
reference. An exact payload pin and a series reference from inspection can be
used together. The SDK verifies both constraints. This example continues the
ERA5 workflow above.

<!-- live-example -->
```python
source = annual_observations.metadata.source
assert source is not None and source.release is not None
pinned = ERA5(indicator=indicator, release=source.release).select(area=area, time=period)
offline = Runtime(cache=Cache.offline(".crc-cache"))
replayed = pinned.read(runtime=offline)
assert replayed.table.equals(annual_observations.table, check_metadata=False)
```

Pinned reads check the selected source bytes before returning even a preview
or the first batch. If a native object or cached view has changed, the read
raises a release mismatch.

Fields of the World inspection without an area lists the publication years and
their catalogue release. Inspection with an area uses the same selectors and
latest-year default as a read. Published STAC checksums identify the selected
snapshot before downloading its Parquet files.

IBTrACS inspection lists the chosen agency's series. Its release reference and
archive checksum come from the bytes acquired during a read.

Access the underlying observations through the same configured provider. This
example continues the live workflow above.

<!-- live-example -->
```python
raw = rain.observations()
preview = raw.read(limit=24, runtime=runtime)
```

A preview is incomplete until the selected source is exhausted. It cannot be
silently fitted or aggregated as if it were the complete observation record.
An unpinned CHIRPS preview acquires monthly rasters as it needs them. A pinned
read verifies all selected months before returning its first batch.

## Supply and aggregate your own observations

```python
from crc_framework import Duration, LengthUnit, PrecipitationAmount, TimeUnit
from crc_sdk import (
    AnnualMaximum, Calendar, DailyTotal, Observations, Sampling,
    TimestampPosition, Variable,
)

table = {
    "support_id": ["gauge-a"] * 24,
    "longitude": [24.0] * 24,
    "latitude": [60.0] * 24,
    "time": [f"2001-01-01T{hour:02d}:00:00Z" for hour in range(24)],
    "value": [1.0] * 24,
}
observations = Observations.from_table(
    table,
    variable=Variable(
        "company:rainfall",
        measurement=PrecipitationAmount(
            unit=LengthUnit.MILLIMETRE,
            accumulation_duration=Duration(1, unit=TimeUnit.HOUR),
        ),
    ),
    sampling=Sampling(
        calendar=Calendar.GREGORIAN,
        step=Duration(1, unit=TimeUnit.HOUR),
        timestamp=TimestampPosition.START,
    ),
)
daily = observations.aggregate(DailyTotal())
assert daily.read().table["value"].to_pylist() == [24.0]
annual = daily.aggregate(AnnualMaximum())
assert annual.read().report.counts.incomplete_years == 1
```

`TimeSeriesColumns` maps custom sample or observation column names.
`ProjectionColumns` maps a time or return-period coordinate and any published
confidence interval. `LayerColumns` maps numeric values, labels, and point or
WKB geometry. Column names and user IDs stay ordinary exact strings; calendars,
units, durations, and reduction choices are typed values.

An annual maximum applied directly to hourly evidence is the largest hourly
observation. Apply `DailyTotal()` first to compute the wettest day.
`TimestampPosition.END` assigns an accumulation to the interval it actually
measured, including midnight totals. Sampling intervals must align with the
native calendar grid. For example, consecutive hourly precipitation amounts
beginning at 23:30 cannot be allocated to separate days from the source data.
Annual reductions retain missing years in an explicit time selection for each
identified support within the declared period. Rows outside a declared period
are invalid. A calendar-year sampling step allows at most one observation per
support and year. Saved metadata retains the calendar, sampling, completeness,
measurement changes, and reductions. An irregular series can be read without
`Sampling`. Aggregation requires a known cadence.
When an ERA5 selection has no hourly records, raw evidence retains its known
grid cells through collection and reopening. Annual reduction can then report
missing years for those cells.

## Discover stations and keep source identities

This example reads the live station inventory. Inspect before choosing a station
and pass its returned reference to `select`.

<!-- live-example -->
```python
from crc_sdk import ISD

inventory = ISD().inspect()
station = next(station for station in inventory.stations if station.period is not None)
annual_maximum = inventory.series[0]
station_data = ISD().select(series=annual_maximum, stations=station, time=station.period)
```

Station references carry the source ID, label, and known location/record window.
A station ID identifies the station; support IDs identify immutable observation
locations. A station move does not rewrite the geometry of earlier reports.
Unavailable station-years appear as `UnavailableStationYear` diagnostics with
the station ID, year, and required coverage fraction. The report counts them as
requested and incomplete years. Source rows with known support carry their
actual coverage and any failure directly.
Snapshot checks use the absence status of a missing station resource. A new
HTTP request identifier leaves a missing year pinned to the same absence.
Raw GHCNh/ISD reports remain irregular. Their named annual recipes require all
24 distinct hours for a complete day and then apply the declared annual rule.

An inspected `SeriesReference` selects the indicator and supplies the
observation years when it contains a dated window. Its inventory release
identifies the discovered publication. A complete read records a separate
release for the exact selection, including its station catalogue and payload
files. Inspecting saved samples exposes that exact release for replay with the
same selection. Unpinned GHCNh and ISD previews acquire station-years as they
are needed. An exact release pin verifies every selected station-year before
returning any rows. Reports retain SHA-256 hashes, licences, attribution, and
retrieval times. `Cache.offline(...)` replays verified cached bytes.

Published IDF durations and provinces are typed too:

```python
from crc_framework import Duration, TimeUnit
from crc_sdk import ECCCIDF

idf = ECCCIDF(
    duration=Duration(1, unit=TimeUnit.HOUR),
    province=ECCCIDF.Province.ONTARIO,
)
central_estimates = idf
confidence_intervals = idf.confidence_intervals()
```

Without a province, `ECCCIDF().inspect().indicators` lists the available durations.
With a province, inspection lists exact series and station periods from its
archive. Use those series references directly in `select(series=...)`.

## Discover a CRC Open release

Pass the local path or HTTPS URL of a release index or pointer to
`CRCOpen(catalog=...)`. The index must use release-index format 1.0 and
point to Dataset 2.0 files. Use the index supplied with your release.
Publication of a matching production CRC Open dataset is pending.
[Release notes](release-notes.md#crc-open-availability)

Inspection reads the index and returns actual series combinations, including
their periods and release identities. This example lists the available
combinations, reads the first one, and replays it from the verified cache.

```python
from crc_sdk import CRCOpen, Cache, Runtime

source = CRCOpen(catalog="release-index.json")
runtime = Runtime(cache=Cache.reuse(".release-cache"))
inventory = source.inspect(runtime=runtime)
for series in inventory.series:
    print(series.variable.id, series.scenario, series.period, series.release)

selected = source.select(series=inventory.series[0])
published = selected.read(runtime=runtime)
replayed = selected.read(runtime=Runtime(cache=Cache.offline(".release-cache")))
assert replayed.table.equals(published.table, check_metadata=False)
```

A read checks each participating file's checksum, Arrow schema, indexed
scientific metadata, and complete row counts before delivering rows. It
retains each verified file's source identities, diagnostics, and missing-data
counts. Produced and no-data counts describe the selected rows. The read also
retains the transformation history needed to replay fitted curves. Files
combined in one read must agree on the history of any shared variable or
scenario identity. Pass a returned series reference directly to `select` to
pin that exact choice. For a known published variable or scenario, use a
provider enum such as `CRCOpen.Variable.ANNUAL_MAXIMUM_DAILY_PRECIPITATION` or
`CRCOpen.Scenario.SSP585` in the corresponding selector. User-owned IDs in a
release can be selected by their exact values.

## Define asset inputs

```python
from crc_sdk import Assets, PointColumns

assets = Assets(
    {
        "site": ["warehouse-a"], "x": [24.0], "y": [60.0],
        "replacement_cost": [1_000_000.0],
    },
    id_column="site",
    location=PointColumns(longitude="x", latitude="y"),
    columns=["replacement_cost"],
)
cost = assets.column("replacement_cost")
```

Strings passed to `Assets(...)` are CSV or Parquet paths. SQL is explicit:
`Assets.sql("SELECT * FROM sites", location=PointColumns())`, with the caller's
connection supplied through `Runtime(connection=...)`. Standard longitude and
latitude, H3 cell, or WKB geometry columns are inferred only when unambiguous;
`CellColumn` and `GeometryColumn` provide explicit mappings. IDs must be unique
and non-null. Locations must be valid WGS84; conflicting declared CRS is
rejected. Retained columns always include the asset ID.

SQL assets record the query identity and checksum of the materialized input
columns. Offline execution disables external access for SDK-owned DuckDB
connections. For offline use with a caller-owned connection, pass a materialized
asset table or local file as `Assets(...)`. This lets the SDK verify the exact
asset input without running caller-defined SQL functions or extensions.
