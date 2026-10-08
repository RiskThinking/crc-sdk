# CRC SDK

CRC SDK is the higher-level Python interface for Climate Risk Commons data
access, storage providers, geometry utilities, and analytical workflows.
Numerical distributions, curve fitting, impact transforms, and risk metrics are
provided by the versioned
[`crc-framework`](https://pypi.org/project/crc-framework/) dependency.

## Development

uv is the recommended tool for managing the development environment:

```shell
uv sync --all-extras

uv run pytest
uv run mypy
uv run ruff check .
```

Or simply:

```shell
python -m venv .venv
.venv/bin/python -m pip install -e ".[zarr,raster,geometry,test]"
.venv/bin/python -m pytest
.venv/bin/python -m mypy
.venv/bin/python -m ruff check .
```

## Dependencies

DuckDB, Arrow (`pyarrow`), `psutil` (resource detection), and remote-storage
transport (`fsspec`, `s3fs`, `gcsfs`) are baseline dependencies — every
connector and workflow in this SDK is built on that stack, so gating it
behind an extra would just move the same install onto every real caller.

Everything else is a specific data-format adapter or a pure-geometry
dependency, opted into only by the callers that need it:

| Extra | Adds | Used by |
|---|---|---|
| `zarr` | `zarr` | `OSClimateProvider`/`ZarrRaster` (OS-Climate Zarr raster ingest) |
| `agriculture` | `icechunk`, Zarr v3, `pyproj` (Python 3.12+) | USDA CDL Icechunk AOI/year/class scans; FTW GeoParquet itself uses baseline DuckDB |
| `raster` | `rasterio` | `GeoTiffRaster` (GeoTIFF/COG ingest, streamed via GDAL VSI) |
| `netcdf` | `h5netcdf`, `h5py` | `NetCDFRaster` (NetCDF/CF ingest) |
| `geometry` | `h3`, `h3ronpy`, `shapely` | `H3Indexer`, `intersecting_cells`, `cell_polygon`, other abstract H3/geometry math, vectorized batch H3 ops on Arrow data (`polyfill_wkb`, `expand_polygon_candidates`, raster-to-H3 sampling) |
| `test` | `mypy`, `pytest`, `ruff` | Development only |

DuckDB's community `duckdb_zarr` extension can scan ordinary remote Zarr v2/v3
stores, but it does not open Icechunk's versioned repository/session model.
USDA CDL therefore uses the official Icechunk client for version resolution and
chunk reads, then exposes bounded Arrow batches to DuckDB; it never materializes
the full raster or a full AOI in memory.

Every function that needs an extra-gated dependency imports it lazily and
raises a clear `ImportError` naming the extra to install if it's missing —
importing `crc_sdk` (or any of its subpackages) itself never requires more
than the baseline dependencies.

**OS-level dependency (not a pip extra):** `tippecanoe` and `tile-join`
(https://github.com/felt/tippecanoe) must be present on `PATH` for
`crc_sdk.geometry.pmtiles` — they're assumed to already be installed on the
runtime image, not `pip install`-able, so there's no extra for them. Verify
availability with `require_tippecanoe()`/`require_tile_join()`, which raise a
friendly, actionable error (with install instructions) if either is missing.

## Package boundaries

- `crc_sdk.core`, `crc_sdk.fitting`, and `crc_sdk.impacts` expose the stable
  public API of `crc_framework`.
- `crc_sdk.connectors` handles external formats and query engines: DuckDB
  connection helpers (`DuckDBConnection`, `RuntimeResources`, streaming
  Parquet writes), OS-Climate Zarr ingest (`zarr` extra), and GeoTIFF/COG
  ingest (`GeoTiffRaster`, `raster` extra) — the latter streams directly
  from local paths or `gs://`/`s3://`/`http(s)://` URIs via GDAL's own
  range-request support, with no local download by default.
  `DuckDBRelationSource`, `ArrowBatchSource`, and `DuckDBPipeline` form the
  common lazy process seam: native SQL/Parquet adapters return relations,
  while chunk stores yield bounded Arrow batches into the same immutable
  filter/project/aggregate/write pipeline. `AgriculturalLayer.usda_cdl()` uses
  that Arrow seam for the versioned Icechunk v2 CDL store;
  `AgriculturalLayer.ftw_fields()` stays native in DuckDB over remote
  GeoParquet, applying `bbox` pruning before exact spatial filtering.
- `crc_sdk.providers` describes storage and dataset discovery.
- `crc_sdk.geometry` contains geometry conversion, DuckDB-native H3 polyfill
  (`H3Indexer`), Arrow batch polyfill (`polyfill_wkb`, `geometry`
  extra for h3ronpy), raster-to-H3 sampling primitives
  (`pixel_grid_resolution`, `sample_grid_to_h3`), exploded coverage writers
  (`write_exploded_coverage`), optional nested lookup derivation
  (`LookupCatalog`, `write_lookup_contract`, `write_partitioned_lookup`), and
  PMTiles generation (`crc_sdk.geometry.pmtiles` — also reachable flattened
  as `crc_sdk.geometry.PMTilesBuild`, etc.): `PMTilesBuild` streams a
  GeoParquet source (a single file, or a Hive-partitioned dataset glob) into
  one `.pmtiles` archive in one tiling pass, building the GeoParquet ->
  GeoJSON bridge itself in DuckDB `spatial`-extension SQL
  (`ST_AsGeoJSON`/`ST_ReducePrecision`/`ST_Transform`) rather than shelling
  out to an external converter, streamed via the same Arrow-batched-reader
  pattern used elsewhere in this SDK. `tippecanoe_threads`/`duckdb_threads`
  default to every detected core (no conservative per-thread cap, unlike
  DuckDB's own GEOS-throttled default) since tippecanoe's tile-building has
  no documented per-thread memory ceiling. A pre-flight budget check raises
  a clear, actionable error if a source is estimated to exceed available
  scratch disk, rather than silently degrading into a slower multi-batch
  fallback — provisioning more disk or narrowing the run's scope is left to
  the caller.
- `crc_sdk.schema` defines columnar data contracts.
- `crc_sdk.types` contains SDK-owned Pydantic configuration and metadata.
- `crc_sdk.workflows` coordinates data access and computation.

DuckDB resource limits are detected when requested
(`RuntimeResources.detect` / `DuckDBConnection.for_analytics`) and relayed
through the connection `config` mapping. Thread count is
`min(cpus, usable_RAM / GiB_per_thread)` with usable RAM ≈ 60% of detected
memory and a default of ~2.5 GiB/thread (GEOS spatial work often slows when
over-threaded). `memory_limit` and `max_temp_directory_size` remain hard
process caps. Override with `CRC_DUCKDB_THREADS`, `CRC_DUCKDB_MEMORY`, and/or
`CRC_DUCKDB_BYTES_PER_THREAD_GIB`, or pass an explicit `config` dict. Set
`CRC_DUCKDB_PROFILE=1` to enable detailed query profiling around
enrich/coverage stages in h3geo.

Constructors with no natural caller-supplied directory of their own
(`OSClimateProvider`, `ZarrRaster`, `H3Indexer`) build a resource-tuned
connection by default — via `DuckDBConnection.for_analytics` — instead of a
bare, untuned one, so this scales out of the box with no configuration.
Passing an explicit `connection`/`con` always wins and skips this entirely.
Otherwise the spill/temp directory defaults to a stable location under the
system temp directory (`default_work_dir()`, not a fresh one per call), and
can be set per-call via each constructor's own `work_dir` parameter, or
globally via `CRC_DUCKDB_WORK_DIR`.

Private/authenticated remote sources (a non-public GCS/S3 bucket) are
configured the same idiomatic-DuckDB way as everything else here: raw
`CREATE OR REPLACE SECRET` SQL, passed as `setup_sql=(...)` to
`DuckDBConnection`/`DuckDBConnection.for_analytics` to have it run
automatically on `.connect()`, right after extensions load. There is
deliberately no secret-builder type in the SDK — DuckDB's own secret DDL
(https://duckdb.org/docs/configuration/secrets_manager) is already the
documented interface, and a caller's own connection-setup module is a more
natural home for its specific credentials than a generic wrapper trying to
track every provider/type DuckDB supports. `sql_quote`/`sql_identifier` are
exported for safely building that SQL; the one DuckDB quirk worth knowing is
that `PROVIDER` is a bare keyword (`config`, `credential_chain`, ...), not a
quoted string literal, unlike every other secret option:

```python
import os
from crc_sdk.connectors.duckdb import DuckDBConnection, sql_quote

setup_sql = []
key_id, secret = os.getenv("GCS_ACCESS_KEY"), os.getenv("GCS_ACCESS_SECRET")
if key_id and secret:
    setup_sql.append(
        f"CREATE OR REPLACE SECRET gcs (TYPE GCS, KEY_ID {sql_quote(key_id)}, "
        f"SECRET {sql_quote(secret)})"
    )
con = DuckDBConnection.for_analytics(work_dir, setup_sql=setup_sql).connect()
```

## Agricultural layers

Agricultural requests are immutable and bounded before they can scan. Builder
calls perform no network I/O; `relation()`, `to_arrow_reader()`, and
`write_parquet()` are execution points:

```python
from crc_sdk.workflows import AgriculturalLayer

crop_mix = (
    AgriculturalLayer.usda_cdl()
    .resolution("30m")
    .for_area((-93.46, 42.14, -93.45, 42.15))
    .years(2025)
    .classes([1, 5])  # corn and soybeans
    .scan()
    .pipeline()
    .aggregate("count(*) AS sampled_pixels", groups="year, crop_code, crop_name")
)
for batch in crop_mix.to_arrow_reader():
    process(batch)
```

For global predicted field units, replace the source while keeping the same
process surface:

```python
fields = (
    AgriculturalLayer.ftw_fields()
    .in_country("FR")
    .for_area((2.0, 47.5, 3.0, 48.5))
    .years(2024)
    .confidence_at_least(80)
    .scan()
    .pipeline()
)
fields.write_parquet("outputs/france-fields.parquet")
```

FTW fields are remote-sensing units, not cadastral parcels or evidence of
ownership. CDL crop pixels are land-cover observations, not acreage, yield, or
financial exposure.

## Canonical hazard datasets

The SDK internalizes fitted hazards as one versioned Arrow/Parquet contract.
Rows contain a canonical unsigned H3 `cell_index`, stable `source_id`, optional
source WKB, scenario dimensions, and the parameters needed to reconstruct a
`crc_framework.FittedDistribution`, `HurdleDistribution`,
`PointMassDistribution`, or compact `TabulatedDistribution`.
`curve_shape` is nullable because Gumbel families do not use a shape parameter;
atom probability and location are populated for hurdle and point-mass rows.
Schema 1.1 added `curve_kind="point_mass"` for a distribution that is
constant across the complete source probability support. It uses the same
physical columns, with
`curve_type="point_mass"`, zero scale, and probability one at
`curve_location`; downstream quantile calls remain identical to fitted and
hurdle curves.

Schema 1.2 adds two nullable list columns, `curve_probabilities` and
`curve_values`. They are populated only for `curve_kind="tabulated"`, with
`curve_type="linear_probability"`; scalar curve parameters are null. It also
supports `curve_kind="no_data"`, whose `curve_type` is an explicit scientific
reason code and whose parameter fields are all null. Batch quantile evaluation
returns nulls for those rows. This tagged-union layout avoids redundant status,
fit-stage, interpolation, and reason columns: dataset metadata and run
manifests carry ordered-family and aggregate treatment provenance once.

Schema 1.3 is metadata-only: the physical columns, row key and sort order are
those of 1.2, and every 1.0–1.2 file reads unchanged. It adds optional dataset
metadata (decisions in [`docs/adr/`](docs/adr/README.md)):

- `probability_semantics` (`annual_exceedance`, `annual_value_distribution`,
  `within_period_percentile`, `projection_uncertainty`, `estimate_confidence`)
  and `source_return_period_convention` (`one_minus_inverse` or `poisson`);
- `temporal_window` (inclusive start/end years, `horizon` as the centre year,
  baseline, calendar, minimum complete years, or a `time_invariant`
  reference year) and `ensemble` (`single_member`/`pooled`/`unknown`, models,
  members, scenario, downscaling, bias adjustment);
- source `licence`, `attribution`, `retrieved_at` and `checksum`;
- fit provenance `input_kind`, `sample_resampling`, `lower_bound`, `platform`
  and `crc_framework_version`, and the `sample_mle` method value.

Fields outside 1.3 reject these additions, so a 1.2 file can never claim them.
Readers older than this release reject `schema_version: "1.3"`.

### Fitting distributions that already exist

`fit_cdf_quantile_batches` canonicalizes Arrow rows of quantiles without
knowing anything about the hazard. A row's probability axis may be the shared
one passed in, a per-row list (`CDFColumnSchema(probabilities=...)`), or
labelled return periods (`CDFColumnSchema(return_periods=...)`, converted with
the policy's explicit `return_period_convention`). Raw samples
(`CDFColumnSchema(samples=...)`) are sorted and resampled to
`sample_quantile_count` probabilities. Probability-labelled inputs are always
fitted by quantile least squares. Notable `CDFCurveFitPolicy` options:

- `lower_bound` censors values below a physical bound before fitting and turns
  fitted mass below it into a hurdle atom (for example depth `>= 0`);
- `short_axis_action` (`tabulated` by default) handles rows with fewer than
  four interior knots inside the fitter; unlabelled or length-mismatched rows
  are rejected with a reason rather than dropped;
- `registry` / `minimum_informative_overrides` resolve eligibility per hazard
  (override > registry > policy default), and `no_data_rate_threshold` warns
  (raises under `strict=True`) when a hazard is mostly `no_data`;
- `diagnostics="path.parquet"` writes a row-level sidecar of outcomes, chosen
  family, residuals, fallbacks and rejection reasons;
- `creation_version` defaults to the installed crc-sdk version.

The logical row key is
`(hazard_name, horizon, pathway, cell_index, source_id)`. `cell_index` is the
spatial join key, not a globally unique identifier. Canonical files are sorted
by that row key for predicate pruning and merge joins by default.

`write_hazard_stream(..., ordered=True)` retains that default contract. DuckDB
materializes the canonical stream, rejects duplicate keys, globally sorts, and
may spill to the configured work directory after reaching its memory limit.
This keeps engine memory predictable rather than batch-constant: reserve
additional process headroom for Arrow/Python buffers and the final Parquet
write. Ordered output generally compresses better and can improve scans that
benefit from physical key clustering.

`ordered=False` is an explicit low-memory alternative for unusually large
partitions or tighter containers. It appends validated Arrow batches directly
to a local Parquet staging file, scans only projected row-key columns for
duplicates, and atomically publishes after validation. Canonical schema,
metadata, uniqueness, and downstream curve/percentile APIs are identical, but
physical rows retain input order rather than the global canonical sort.

Ordered-writer memory is governed mainly by the DuckDB connection's
`memory_limit`. On a synthetic 2,000,000-row stream, the ordered writer's peak
RSS was 1,192 MiB with DuckDB's default limit and 598 MiB with a 300 MB limit,
versus 360 MiB (416 MiB with the same limit) for `ordered=False`; wall time was
equal. Containers that set a large default limit should lower it for the write
connection before giving up the canonical sort.

On one 2,148,497-row schema-1.2 sample, direct streaming used 656 MB peak RSS
and produced a 45 MB file in 16.03 seconds; ordered writing used 1.69 GB and
produced a 35 MB file in 16.38 seconds. A 300,000-row sample used 282 versus
525 MB, produced 6.3 versus 4.9 MB, and took 3.01 versus 2.78 seconds. These
measure the persistence pass only on one machine: storage reduction was
consistent, while the small timing difference changed direction. Treat them as
tradeoff evidence, not universal throughput guarantees.

Dataset-wide facts are stored once as a complete JSON payload under the
`crc.hazard.metadata` Parquet key: schema version, one uncompacted H3
resolution, non-exceedance probability convention, source probability
support, value unit and semantics, WKB CRS, producer, source provenance,
curve-fit policy, and creation version.

Each dataset is one self-describing Parquet file, expanded by H3 cell for
spatial joins. The caller chooses its full destination path and filename.
Writes use DuckDB, and an optional configured DuckDB connection allows the same
API to use its local or cloud filesystems, extensions, secrets, and settings.
Source knots and fit diagnostics are transient ingest inputs, not a second
persisted data contract. Values at source return periods are therefore fitted
curve evaluations rather than guaranteed bit-for-bit reproductions of source
pixels.

External connectors remain source-format readers. Ingest adapters perform the
explicit conversion:

```text
external raster/table -> source curves and geometry -> selected family fit
  -> conservative intersecting H3 cells -> canonical Arrow -> Parquet
```

Boundary candidate generation uses H3 overlap coverage, not center polyfill.
This makes the integer join a conservative superset before an exact
`ST_Contains(source_geometry, asset_point)` refinement. Resolution estimates
report measured coverage error and expanded row count, while ingest policy
selects and records the dataset resolution.

OS-Climate return-period rasters can be canonicalized with
`OSClimateIngestPolicy` and `canonicalize_os_climate`. The caller must choose
the distribution family and, for zero-heavy hazards, provide an explicit
`HurdleFitPolicy`; the SDK does not infer an exact point mass from sparse
knots. Plain curves use `fit_quantiles`, while hurdle curves use
`fit_hurdle_quantiles`. `LocalProvider` queries persisted hazard rows through
`HazardQuery`.

### Ingesting JRC flood maps

JRC flood acquisition is available as an immutable, lazy workflow. GLOFAS
2.1.2 uses JRC's tiled global layout; EFAS 3.1.1 uses nine continental
return-period rasters. The dataset descriptions encode that difference so an
AOI workflow does not expose tiles or filenames:

```python
from crc_sdk.workflows import HazardDataset, JRCFloodPolicy

plan = (
    HazardDataset.efas(version="latest")
    .for_area((7.75, 49.75, 8.45, 50.25))
    .cache("cache/efas", mode="reuse")
    .source_periods("all")
    .canonicalize(
        policy=JRCFloodPolicy.curated(h3_resolution=10),
    )
)

print(plan.explain())
hazard = plan.materialize("hazards/efas-rhine.parquet")
```

Builder methods and `explain()` do not resolve releases or fetch rasters.
`materialize()` resolves `latest` once, records the pinned version in the
cache manifest and canonical provenance, caches AOI crops, fits the canonical
curves, and returns an ordinary `HazardDataset`. Use `prefetch()` to populate
the source cache separately, then switch the same plan to `mode="offline"`.
`refresh` explicitly re-resolves the release and replaces cached crops;
`stream` reads remotely without a persistent source cache.

Source periods choose the rasters used for fitting and default to every period
in the resolved release. They are distinct from evaluation periods:

```python
result = (
    plan.for_assets(assets)
    .select(hazard_names=["RiverineInundation"], pathways=["historical"])
    .return_periods([50, 100, 250, 500])
    .write_parquet("outputs/flood-depth.parquet")
)
```

The compact chain lazily materializes a deterministic canonical file inside
the configured cache, then delegates to the same local portfolio evaluator
used by `HazardDataset.local(...)`.

Canonical metadata records the source return-period support. Portfolio
evaluation warns when a requested period falls outside it: with EFAS source
rasters from RP10 through RP500, RP250 is interpolation while RP1000 is
extrapolation.

### Ingesting JRC/EDO drought data

EDO Soil Moisture Index data uses the same lazy workflow, with complete years
reduced to compact annual-minimum AOI cache objects before fitting:

```python
from crc_sdk.workflows import EDODroughtPolicy, HazardDataset

plan = (
    HazardDataset.smi(version="latest")
    .for_area((9.5, 50.5, 10.5, 51.5))
    .years("all_complete")
    .cache("cache/edo-smi", mode="reuse")
    .canonicalize(policy=EDODroughtPolicy.curated(h3_resolution=6))
)

hazard = plan.materialize("hazards/edo-smi.parquet")
```

The curated policy uses the lower return-period tail and requires at least 20
complete years. Metadata stores both that tail and the Gringorten support of
the selected annual record. Evaluation therefore selects the correct lower
tail automatically and warns when a requested period is extrapolated. Cache
manifests pin the resolved EDO version, years, bounds, source URLs, local
objects, and checksums; `prefetch()` followed by `mode="offline"` avoids later
network access.

### ERA5 historical baselines

ERA5 hourly reanalysis becomes an annual-extreme baseline through the same lazy
workflow. Supported recipes are `txx` (annual maximum of daily maximum 2 m
temperature), `tnn`, `rx1day` and `rx5day`; days are UTC days. Data is read
anonymously from public Zarr copies (`crc-sdk[zarr,netcdf,geometry]`), reduced
to one small file per year, and only those annual extremes are cached.

All ERA5 recipes and generic `BlockExtremaPolicy` fits default to GEV with L-moments:

| Recipe | Family | Estimator |
|---|---|---|
| `txx`, `tnn`, `rx1day`, `rx5day` | GEV (`genextreme`) | Sample L-moments (`sample_lmoments`) |

`BlockExtremaPolicy` inherits these choices when `family` or `fit_method` is
unspecified, including when you customize resolution, minimum years, or
diagnostics. Explicit choices override the corresponding recipe default.
For block extremes fitted with Gumbel and L-moments, use
`BlockExtremaPolicy.curated(family="gumbel_r")`. To reproduce a Gumbel
quantile-least-squares fit, specify both `family="gumbel_r"` and
`fit_method="quantile_least_squares"`. A `CurveFitIngestPolicy` is fully
explicit and does not inherit recipe defaults.
Quantile residual gates require `fit_method="quantile_least_squares"`;
incompatible resolved policies are rejected when the plan is constructed.

```python
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

plan = (
    HazardDataset.era5("txx", store="wb2-1p5")  # or "arco-0p25" (native, slow)
    .for_area((-10.0, 36.0, 22.0, 56.0), land_only=True)
    .years(1991, 2020)
    .cache("cache/era5", mode="reuse")
    .canonicalize(policy=BlockExtremaPolicy.curated(h3_resolution=5))
)
print(plan.explain())
hazard = plan.materialize("hazards/era5-txx.parquet")
annual = plan.annual_extremes()  # the per-year samples behind every curve
```

`wb2-1p5` is a 1.5 degree copy (about 30 seconds per variable-year);
`arco-0p25` is native 0.25 degrees but one hour per chunk, so expect minutes per
variable-year. Only final (non-ERA5T) years are used, and the cache is keyed by
the final record's end date. Files carry schema-1.3 metadata: CC-BY-4.0
licence and attribution, retrieval time, the fitted window
(`temporal_window`), `ensemble.pooling="single_member"` and
`probability_semantics="annual_exceedance"`. Pass
`BlockExtremaPolicy(diagnostics="diag.parquet")` for a row-level trace of every
fitted or skipped cell. See `docs/spikes/era5.md` for access details and caveats:
reanalysis is a blend of observations and a model, convective rainfall extremes
are underestimated, and a grid cell is an area average, not a point.

`BlockExtremaCurveSource` (`crc_sdk.connectors.blocks`) is the generic engine
behind this and the EDO drought workflow: any reader that yields per-block
statistics (a year, a season or a water year; max, min, k-day sum or mean, or a
threshold count) can be fitted with a plotting position and least squares, or
with `fit_method="sample_mle"` or `fit_method="sample_lmoments"`. Sample
estimators use the original annual values without interpolation or resampling.

### Bringing your own data

`HazardDataset.from_table`, `from_zarr` and `from_raster` onboard local data
through the same lazy plan (`BYOPlan`): `canonicalize(policy=...)`, `explain()`,
`materialize(output)`, `ensure_materialized()` and `for_assets(...)`. Nothing
is opened or fitted until `materialize()`; `ensure_materialized()` and
`for_assets(...)` also need `.cache(directory)`, which gives a deterministic
canonical path (`mode="refresh"` rewrites it). Tables use the fit policy
described above, so diagnostics, strictness, `lower_bound` and the 1.3
metadata flow through unchanged.

A table of distributions already keyed by H3 cell (Parquet, CSV or any
DuckDB-readable path, or an Arrow table):

```python
from crc_sdk.fitting import CDFColumnSchema, CDFCurveFitPolicy
from crc_sdk.types import SourceProvenance
from crc_sdk.workflows import HazardDataset

policy = CDFCurveFitPolicy(
    h3_resolution=5, family="gumbel_r", value_unit="mm",
    value_semantics="annual maximum 1-day rainfall", producer="me",
    source=SourceProvenance(provider="lab", dataset="rx1day", version="v1"),
    source_id="lab-rx1day", probability_semantics="annual_value_distribution",
)
plan = HazardDataset.from_table(
    "rx1day.parquet", columns=CDFColumnSchema(), policy=policy,
    probabilities=[i / 10 for i in range(11)],  # or return_periods=[...]
)
hazard = plan.materialize("hazards/rx1day.parquet")
```

A return-period Zarr array (needs `crc-sdk[zarr]` and `crc-sdk[geometry]`),
optionally windowed to WGS84 bounds:

```python
from crc_sdk.connectors import CurveFitIngestPolicy

plan = HazardDataset.from_zarr(
    "s3://bucket/flood.zarr", array="depth", storage_options={"anon": True},
    hazard_type="RiverineInundation", indicator_id="flood_depth",
    scenario="historical", year=2020, units="m",
    policy=CurveFitIngestPolicy(h3_resolution=8, family="gumbel_r", producer="me"),
).for_area((7.75, 49.75, 8.45, 50.25))
print(plan.explain())
hazard = plan.cache("cache/byo").ensure_materialized()
```

The Zarr array carries `index_name` (containing "return period"),
`index_values` and `transform_mat3x3` attributes. Same-grid GeoTIFFs, one per
explicit return period (needs `crc-sdk[raster]` and `crc-sdk[geometry]`):

```python
plan = HazardDataset.from_raster(
    {10: "rp10.tif", 50: "rp50.tif", 100: "rp100.tif", 500: "rp500.tif"},
    hazard_type="RiverineInundation", indicator_id="flood_depth",
    scenario="historical", year=2020, units="m",
    policy=CurveFitIngestPolicy(h3_resolution=8, family="gumbel_r", producer="me"),
)
hazard = plan.for_area((7.75, 49.75, 8.45, 50.25)).materialize("hazards/flood.parquet")
```

### Evaluating asset portfolios at return periods

Canonical curve parameters can be evaluated for a portfolio without returning
to the external source format or refitting the data. The workflow joins every
asset to its canonical curve and writes one row per asset, hazard, horizon, and
pathway, with one value column per requested return period:

```python
import pyarrow as pa

from crc_sdk.workflows import HazardDataset

assets = pa.table(
    {
        "asset_id": ["warehouse-a", "warehouse-b"],
        "longitude": [6.9603, 7.5010],
        "latitude": [50.9375, 51.0030],
        "sector": ["logistics", "manufacturing"],
    }
)

result = (
    HazardDataset.local("flood.parquet")
    .for_assets(assets)
    .select(horizons=[2050], pathways=["ssp585"])
    .return_periods([25, 50, 100, 250, 500, 1000])
    .write_parquet("portfolio-flood.parquet")
)
```

The resulting value columns are `value_rp25`, `value_rp50`, `value_rp100`,
`value_rp250`, `value_rp500`, and `value_rp1000`. For upper-tail hazards, each
return period `RP` is evaluated at non-exceedance probability `1 - 1/RP`.
Value unit, value semantics, and the complete return-period/probability/column
mapping are stored under `crc.hazard.evaluation` in Parquet metadata.

An impact function can replace the sampled hazard values with event-aligned
impact values before writing:

```python
import numpy as np

impact_result = (
    HazardDataset.local("flood.parquet")
    .for_assets(assets)
    .return_periods([25, 100, 250])
    .impact(
        lambda depth: np.clip(depth / 2.0, 0.0, 1.0),
        name="depth_damage_ratio",
        value_unit="fraction",
        value_semantics="damage ratio",
    )
    .write_parquet(
        "portfolio-impact.parquet",
        execution=ExecutionOptions(max_workers=1),
    )
)
```

The SDK first samples each hazard return period and then calls
`impact.evaluate(...)` on that row's value vector. Therefore
`value_rp100 = impact(hazard_rp100)`: the return period continues to identify
the source hazard event. This differs intentionally from transforming a full
distribution and then taking an impact quantile, which can reorder decreasing
or non-monotonic impacts. Use the distribution interface in `crc-framework`
for that risk-analysis interpretation.

Built-in and registry-backed framework impacts use the same fluent method:

```python
from crc_sdk.impacts import PiecewiseLinearImpact, impacts
from crc_sdk.workflows import ImpactContextColumns

damage_curve = PiecewiseLinearImpact(
    exposure=[0.0, 0.2, 1.0, 2.0],
    impact=[0.0, 0.0, 0.25, 1.0],
)

request = request.impact(
    damage_curve,
    name="flood_damage_ratio",
    value_unit="fraction",
    value_semantics="damage ratio",
)

registry_request = request.impact(
    impacts.for_factor("inundation"),
    context=ImpactContextColumns(
        country="country",
        continent="continent",
        building_type="building_type",
        historic_mean="historic_mean",
    ),
    name="inundation_impact",
    value_unit="fraction",
    value_semantics="damage ratio",
)
```

The generated H3 `cell_index` is always supplied to the framework impact
context. Configured context columns are read from each asset, including when
they are not retained as output passthrough columns. Stored registry context
provides fallback values for fields without an asset value. Impact metadata
records the event-aligned interpretation, source hazard units and semantics,
output units and semantics, function name/type, and context-column mapping.

Framework impact objects and top-level Python callables can run in the existing
process pool. Lambdas and closures are not picklable, so they run serially when
the worker count is implicit; explicitly requesting more than one worker for
one raises an error.

Point assets are converted to the H3 resolution recorded by the canonical
dataset. The H3 join is refined with `ST_Covers(source_geometry, asset_point)`
when source WKB is present; rows without WKB retain cell-level precision.
`source_id` and `spatial_match` (`exact_geometry` or `h3_cell`) remain in the
output. Multiple source curves for one asset/hazard/horizon/pathway raise
instead of being silently selected or aggregated, and missing asset/scenario
matches raise rather than being dropped from the output.

When assets already contain canonical H3 indexes, use
`cell_index_column="cell_index"` instead of longitude/latitude columns. This
avoids point conversion and exact source-geometry refinement:

```python
(
    HazardDataset.local("flood.parquet")
    .for_assets(assets_with_cells)
    .return_periods([25, 50, 100, 250, 500, 1000])
    .write_parquet("portfolio-flood.parquet")
)
```

Arrow tables can be registered directly as shown above. A `Path` reads an
asset Parquet file, while a string is treated as caller-supplied DuckDB SQL.
Output evaluation is streamed in bounded Arrow batches to compressed Parquet.
For already selected canonical rows, `distribution_from_hazard_row` remains
available as the low-level curve reconstruction utility.

The common column names `asset_id`, `longitude`/`latitude`, and `cell_index`
are inferred. Use `AssetPortfolio`, `PointColumns`, or `CellColumn` only for a
nonstandard asset schema. Worker, batch, and connection controls are grouped
under `ExecutionOptions` on `write_parquet`, keeping execution tuning out of
the normal workflow.

## License

CRC SDK is licensed under the GNU Affero General Public License, version 3 or
later.

### L-moments fitting for annual extremes

ERA5 recipes and generic block-extrema policies default to GEV with L-moments. To select the standalone GEV L-moment estimator
explicitly, use:

```python
from crc_sdk.workflows import BlockExtremaPolicy

policy = BlockExtremaPolicy(
    family="genextreme",
    fit_method="sample_lmoments",
    h3_resolution=4,
)
# Supply policy to the existing ERA5 plan's .canonicalize(policy=policy).
```

This method also supports `gumbel_r` and `gumbel_l`. It requires genuine
observations (annual extremes), rather than probability-labelled hazard knots;
quantile quality gates and hurdle fits are not supported. The canonical metadata
records `sample_lmoments` as the estimator, and diagnostics retain skipped cells.

L-moment fitting is available in crc-framework 0.3.0 and later.
Installing the SDK pulls it in automatically; a compatible prebuilt wheel
requires neither a local backend checkout nor a Rust toolchain. Python 3.12
is recommended for the notebook test workspace.

To use an editable SDK checkout in a notebook environment, run this from the
notebook workspace (adjust `../crc-sdk` to your checkout's location):

```bash
uv pip install --python .venv/bin/python --reinstall-package crc-framework --editable ../crc-sdk
```

This also replaces a previously linked local backend with the published package.
Restart the notebook kernel after installation.

Reuse the annual-extremes cache to compare policies without downloading hourly
data again. L-moment fitting alone does not quantify return-level uncertainty;
use resampling or independent validation to assess a proposed default.

Block sample fits validate GEV support and finite return levels at 2, 5, 10,
20, 50 and 100 years by default. Configure `validation_return_periods` for your
application; set `minimum_return_value=0` for nonnegative rainfall return levels,
or supply an application-specific `maximum_return_value`. These bounds apply
only at the configured periods and do not truncate or otherwise alter a fitted
curve. Negative temperatures remain valid. GEV observed-support
checks apply to sample estimators only. Return-level checks also apply to
quantile-least-squares fits, independently of quantile residual gates, and
evaluate the complete distribution for hurdle fits.

A Gumbel fallback is opt-in and requires a diagnostics sidecar:

```python
policy = BlockExtremaPolicy.curated(
    minimum_return_value=0,  # rainfall example
    fallback_family="gumbel_r",
    diagnostics="rx1day-fit-diagnostics.parquet",
)
```

Only a failed GEV L-moment fit or its failed quality checks triggers the fallback.
Short and constant records remain ineligible; the fallback must pass the same
checks. The sidecar records both attempted families, failures, the original
failure reason and the selected family. Dataset provenance lists both candidate
families with `selection_metric="first_acceptable"`; each canonical row records
its actual family. For a lower-tail experiment, `gumbel_l` is also available.
No shape clipping, bootstrap uncertainty threshold or automatic zero point-mass
treatment is applied.
