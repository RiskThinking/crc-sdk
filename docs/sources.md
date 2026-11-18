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
positions. Categorical layers have no fitting method.

## Read an annual indicator before fitting

Inspect a provider to see its typed indicators, required selections, and
source details. ERA5 inspection and data access require network access and
the `zarr` extra. The following example reads ERA5's remote store.

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
first forms five-day totals. Temperature recipes reduce daily maxima or minima
before selecting the annual extreme. No fitting occurs until `fit(...)`.

The default is `Coverage.complete()`. Incomplete days and years remain missing
observations with coverage and a reason; they do not silently enter the fit.
`Coverage.minimum_fraction(0.95)` explicitly permits a relaxed fraction without
rescaling missing values. Reports retain requested, usable, and incomplete year
counts, and fitting records retain excluded evidence counts.

ERA5's public ARCO store uses global spatial chunks for each hour. A small area
still requires those compressed source chunks. Explicit time bounds are
required, and inspection reports the native chunk layout and acquisition
estimate. Cache a deliberate selection before repeating a long calculation.
CMIP6 and NEX require an actual model, scenario, and member; discovery returns
the available combinations rather than guessing these scientific choices.

ERA5, CMIP6, NEX, JRC, and EDO discovery provides the available science and
coverage. Their catalogues lack content checksums for every native object, so
the release reference becomes available from the completed read. The reference
identifies the payload acquired for the selected area and time. Reuse those
bounds when replaying the pin. This example continues the ERA5 workflow above.

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

Access the underlying observations through the same configured provider. This
example continues the live workflow above.

<!-- live-example -->
```python
raw = rain.observations()
preview = raw.read(limit=24, runtime=runtime)
```

A preview is incomplete until the selected source is exhausted. It cannot be
silently fitted or aggregated as if it were the complete observation record.

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
identified support. Saved metadata retains the calendar, sampling, completeness,
measurement changes, and reductions. An irregular series can be read without
`Sampling`. Aggregation requires a known cadence.

## Discover stations and keep source identities

This example reads the live station inventory. Inspect before choosing a station
and pass its returned reference to `select`.

<!-- live-example -->
```python
from crc_sdk import GHCNh

inventory = GHCNh().inspect()
station = next(station for station in inventory.stations if station.period is not None)
station_data = GHCNh(
    indicator=GHCNh.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE,
).select(stations=station, time=station.period)
```

Station references carry the source ID, label, and known location/record window.
A station ID identifies the station; support IDs identify immutable observation
locations. A station move does not rewrite the geometry of earlier reports.
Unavailable station-years appear as `UnavailableStationYear` diagnostics with
the station ID, year, and required coverage fraction. The report counts them as
requested and incomplete years. Source rows with known support carry their
actual coverage and any failure directly.
Raw GHCNh/ISD reports remain irregular. Their named annual recipes require all
24 distinct hours for a complete day and then apply the declared annual rule.

GHCNh and ISD also accept an inspected `SeriesReference` in `select(series=...)`.
The reference selects the indicator and supplies the observation years when it
contains a dated window. Inspecting saved samples also exposes the verified
release, so that reference can replay the same source selection. Mutable API
discovery can leave the release unresolved until execution pins the consumed
bytes. Reports retain actual resources, SHA-256 hashes, licences, attribution,
and retrieval times. `Cache.offline(...)` replays verified cached bytes.

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

A selected file must match both its checksum and the scientific metadata in
the index. Pass a returned series reference directly to `select` to pin that
exact choice. For a known published variable or scenario, use a provider enum
such as `CRCOpen.Variable.ANNUAL_MAXIMUM_DAILY_PRECIPITATION` or
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
