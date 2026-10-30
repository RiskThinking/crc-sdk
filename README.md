# CRC SDK

CRC SDK is the higher-level Python interface for Climate Risk Commons data
access, storage providers, geometry utilities, and analytical workflows.
Numerical distributions, curve fitting, impact transforms, and risk metrics are
provided by the versioned
[`crc-framework`](https://pypi.org/project/crc-framework/) dependency.

## Contributors

Thanks to the people who shape this project through code, reviews, research,
discussion, and ideas.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChrisHampu"><img src="https://avatars.githubusercontent.com/u/6600532?s=80" width="80" alt="@ChrisHampu" /><br /><sub><b>@ChrisHampu</b></sub></a><br /><a href="https://github.com/RiskThinking/crc-sdk/commits?author=ChrisHampu" title="Code">💻</a> <a href="https://github.com/RiskThinking/crc-sdk/pulls?q=is%3Apr+reviewed-by%3AChrisHampu" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/woozyking"><img src="https://avatars.githubusercontent.com/u/2837532?s=80" width="80" alt="@woozyking" /><br /><sub><b>@woozyking</b></sub></a><br /><a href="https://github.com/RiskThinking/crc-sdk/commits?author=woozyking" title="Code">💻</a> <a href="https://github.com/RiskThinking/crc-sdk/pulls?q=is%3Apr+reviewed-by%3Awoozyking" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/awiebe42"><img src="https://avatars.githubusercontent.com/u/65562157?s=80" width="80" alt="@awiebe42" /><br /><sub><b>@awiebe42</b></sub></a><br /><a href="https://github.com/RiskThinking/crc-sdk/commits?author=awiebe42" title="Code">💻</a> <a href="https://github.com/RiskThinking/crc-sdk/pulls?q=is%3Apr+reviewed-by%3Aawiebe42" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/awiebe42" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kylalangotsky"><img src="https://avatars.githubusercontent.com/u/91707255?s=80" width="80" alt="@kylalangotsky" /><br /><sub><b>@kylalangotsky</b></sub></a><br /><a href="https://github.com/kylalangotsky" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/reshmasoman"><img src="https://avatars.githubusercontent.com/u/102987481?s=80" width="80" alt="@reshmasoman" /><br /><sub><b>@reshmasoman</b></sub></a><br /><a href="https://github.com/reshmasoman" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rkodamana"><img src="https://avatars.githubusercontent.com/u/84455891?s=80" width="80" alt="@rkodamana" /><br /><sub><b>@rkodamana</b></sub></a><br /><a href="https://github.com/rkodamana" title="Research">🔬</a> <a href="https://github.com/RiskThinking/crc-sdk/pulls?q=is%3Apr+reviewed-by%3Arkodamana" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dilshank"><img src="https://avatars.githubusercontent.com/u/1534562?s=80" width="80" alt="@dilshank" /><br /><sub><b>@dilshank</b></sub></a><br /><a href="https://github.com/dilshank" title="Research">🔬</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

Contribution types: 💻 code · 👀 review · 🔬 research. We recognize contributions
both on and off GitHub using [All Contributors](https://allcontributors.org/).

<!--
Maintaining contributor credits (run from the repository root):
  npx --yes all-contributors-cli@6.26.1 add HANDLE code,review
  npx --yes all-contributors-cli@6.26.1 add HANDLE research
  npx --yes all-contributors-cli@6.26.1 generate
Use code,review as the default types for code contributors; specify other types
explicitly where appropriate (for example research or ideas).
For manual additions or corrections, edit .all-contributorsrc and regenerate.
Edit the roster rather than the generated table, which is replaced on generation.
-->

## Development

uv is the recommended tool for managing the development environment:

```shell
uv sync --all-extras

uv run pytest
uv run mypy
uv run ruff check .
```

CI checks that the lockfile is current, runs the test suite with the locked
dependencies on Python 3.11 through 3.15, exercises the minimum supported
crc-framework version, and verifies a clean installation of the built wheel.
Python 3.14 and 3.15 checks are non-blocking while native dependency
compatibility is being addressed; Python 3.11 through 3.13 gate releases.
The release workflow runs these checks before publishing.


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
| `geometry` | `h3`, `shapely` | `H3Indexer`, `intersecting_cells`, `cell_polygon`, other abstract H3/geometry math, vectorized batch H3 ops on Arrow data (`polyfill_wkb`, `expand_polygon_candidates`, raster-to-H3 sampling) |
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
  extra), raster-to-H3 sampling primitives
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
`sample_quantile_count` probabilities. Probability-labelled inputs default to
quantile least squares. Sparse return-period flood/inundation curves with
bound-valued lower knots should select `fit_method="censored_quantile"`,
`family="genextreme"` and their physical `censor_location` (normally zero).
That method fits all knots with a left-censored likelihood, then stores the
clamped result as the existing hurdle representation. Notable
`CDFCurveFitPolicy` options:

- `lower_bound` censors values below a physical bound before fitting and turns
  fitted mass below it into a hurdle atom (for example depth `>= 0`);
- `censor_location` and `residual_scale_floor_ratio` configure the hardened
  censored likelihood. The floor prevents sparse, exactly determined tails
  from collapsing residual variance to zero and is recorded in provenance;
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
the distribution family and fit method. Use `censored_quantile` for sparse
already-return-period-based flood or inundation curves whose lower values are
at a physical bound. Use an explicit `HurdleFitPolicy` only when the source
defines a known point-mass probability; the SDK does not infer that probability
from sparse knots. `LocalProvider` queries persisted hazard rows through
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

### CMIP6 projections

CMIP6 daily model output becomes an annual-extreme projection through the same
workflow, with the same four recipes as ERA5 (`txx`, `tnn`, `rx1day`,
`rx5day`) and the same definitions, so a historical baseline and a scenario run
are directly comparable. Data is read anonymously from the Zarr copies on the
AWS Open Data registry (`s3://cmip6-pds`, the Pangeo mirror); as for ERA5, only
the per-year annual extremes are cached, never the daily fields.

```python
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

plan = (
    HazardDataset.cmip6(
        "rx1day", model="GFDL-ESM4", scenario="ssp585", member="r1i1p1f1"
    )
    .for_area((-125.0, 42.0, -116.0, 49.0))
    .years(2021, 2050)
    .cache("cache/cmip6", mode="reuse")
    .canonicalize(policy=BlockExtremaPolicy.curated(h3_resolution=5))
)
print(plan.explain())
hazard = plan.materialize("hazards/cmip6-rx1day-ssp585.parquet")
```

Three things have no ERA5 counterpart:

- **The dataset is found, not configured.** CMIP6 is a catalogue of one Zarr
  per (model, experiment, member, variable, grid, version).
  `CMIP6Catalogue` resolves a selection to exactly one of them — preferring the
  model's native grid (`gn`) and the newest DRS version, and raising with the
  candidates listed rather than guessing when that is not enough. The 80 MB
  catalogue CSV is read with DuckDB, and its daily subset is cached in the plan's
  cache directory, so later lookups are local. Pin a grid with
  `.with_grid("gr")` and a release with `.version("20180701")`.
- **Calendars are the model's.** CMIP6 models run `noleap`, `360_day` or
  `all_leap` years as readily as Gregorian ones, so time is decoded into the
  model's own calendar (`CFCalendar`) and blocks are cut on calendar day
  ordinals. Samples are placed by date rather than position, so a gap in the
  record stays a gap. The resolved calendar is recorded in
  `temporal_window.calendar`.
- **The pathway is the experiment.** `scenario` is the CMIP6 `experiment_id`
  (`historical`, `ssp126`, `ssp245`, `ssp370`, `ssp585`, ...) and travels into
  the canonical rows as the source's own pathway label, checked against the
  `crc_sdk.pathways` crosswalk (ADR-0002). Files carry
  `ensemble.pooling="single_member"` with the model and member named, and the
  licence and attribution the store itself declares. Licences are per
  modelling centre and not all are CC-BY — NOAA-GFDL publishes share-alike —
  so the adapter records the store's own licence, warns on terms beyond
  attribution, and refuses them under `strict_licence=True`. The obligation
  belongs to the output, so that check applies wherever a dataset is produced
  *or handed back*, including from a warm cache or an already-fitted file.

One model, one member, one pathway per canonical dataset (ADR-0001): build one
file per model and compare or difference them, rather than pooling inside a
single file. Because `horizon` is the window's centre year and `pathway` the
experiment, an `ssp585` file and a `historical` file for the same area join on
`(hazard_name, horizon, pathway, cell_index)` and subtract to a change map.

See `docs/spikes/cmip6.md` for access details and caveats: a climate model is a
model world, so these are change signals rather than design values; one member
of one model is one realisation, not an ensemble; and a ~1 degree cell damps
orographic and convective extremes further than reanalysis does.

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

A reproducible definition is a separate object from that workflow binding.
Construct it with an explicit calibration, persist the canonical JSON, and
reload it through the native parser. A registry selection is pinned to the
installed registry and keeps its identity until it is resolved:

```python
from crc_sdk.impacts import (
    ImpactCalibration,
    ImpactExpression,
    RegistryImpactReference,
    check_exposure,
    exposure_declaration,
)

calibration = ImpactCalibration(
    exposure_variable="flood_depth",
    exposure_unit="m",
    impact_metric="damage_ratio",
    impact_unit="1",
)
impact = ImpactExpression.tabulated(
    [(0.0, 0.0), (0.3, 0.08), (1.0, 0.45), (2.0, 1.0)],
    calibration=calibration,
    clamp={"minimum": 0.0, "maximum": 1.0},
)
restored = ImpactExpression.from_json(impact.to_json())
assert restored.definition_hash == impact.definition_hash

pinned = RegistryImpactReference.pinned("rx1day")
check_exposure(pinned.calibration(), exposure_declaration("rx1day"))
```

`to_json()` is the canonical encoding. `definition_hash` is the SHA-256 of
those bytes. `exposure_declaration` reads the framework hazard catalog, so
`rx1day` is millimetres per day and `fd` denotes `frost_days`. A caller unit
of `days` matches the calibration unit `d` through the unit catalog. Hazards
the catalog does not list need an explicit declaration.
`PortfolioEvaluation.impact` still accepts the existing impact objects shown
above.

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
later. That licence covers the software. The CRC Open dataset is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(`CC-BY-NC-4.0`).

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

L-moment fitting requires crc-framework 0.3.0 or later, installed automatically
with the SDK.

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

## CRC Open canonical data

`HazardDataset.crc_open()` reads the public source at
`https://data.riskthinking.ai/crc-open`, with
`ssp585-global-2026-10-20-v1` pinned as its default release. Public HTTPS access
requires no credentials. Use `release="another-release"` to select a specific
published release.

The default plan reuses a persistent local cache at
`$XDG_CACHE_HOME/crc-sdk/crc-open` (or `~/.cache/crc-sdk/crc-open`). Repeated
requests for verified cached data do not contact the server. Use `.cache(path)`
to choose a different location. Temporary downloads require the explicit opt-out
`.cache(None, mode="stream")`; `mode="refresh"` explicitly downloads data again.

Discover hazards and partitions without downloading curve files:

```python
from crc_sdk.workflows import HazardDataset

plan = HazardDataset.crc_open().cache("work/crc-open", mode="reuse")
catalog = plan.catalog()
print(sorted(catalog.hazards))

selection = plan.hazards(["rx1day"]).resolve()
print(selection.total_bytes, selection.total_rows)
for partition, url in zip(selection.partitions, selection.urls):
    print(partition.h3_r0, partition.rows, url)
```

`catalog()` returns all release declarations. `resolve()` honors the plan's
hazard, area and horizon selection using available partition metadata. Its URLs
point to **whole files**: byte and row totals describe those complete files,
before row or column filtering. An available r0 partition does not imply coverage
of every finer cell inside it. If per-partition horizons or sampled cells are not
listed, discovery keeps the file rather than guessing that coverage is absent.

For direct DuckDB integration, pass the discovered URL list:

```python
import duckdb

selection = plan.hazards(["rx1day"]).horizons([2050]).resolve()
with duckdb.connect() as con:
    rows = con.execute(
        "SELECT count(*) FROM read_parquet(?, hive_partitioning=false) "
        "WHERE pathway = 'ssp585' AND horizon = 2050",
        [list(selection.urls)],
    ).fetchone()
    print(rows)
```

Direct URL consumers must apply their own row filters. DuckDB can use HTTP range
requests and Parquet projection/filter pushdown; discovery does not fetch the
Parquet files or verify their complete file checksums.

To create an SDK dataset with verified downloads and canonical metadata:

```python
plan = (
    HazardDataset.crc_open(release="ssp585-global-2026-10-20-v1")
    .for_area((-79.5, 43.5, -79.2, 43.8))
    .hazards(["rx1day"])
    .horizons([2050])
    .cache("work/crc-open", mode="reuse")
)
print(plan.explain())  # no I/O
plan.prefetch()
curves = plan.materialize("work/rx1day.parquet")
```

Each release supplies `_CATALOG.json`, its `_SUCCESS` SHA256, and relative
Parquet partition paths. Discovery rejects an absent or mismatched completion
marker. `reuse`, `offline` and `refresh` use verified persistent caches; resolving
metadata alone is sufficient for subsequent offline discovery. Offline data
materialization additionally requires the selected partition files in the cache.
`refresh` rejects changed catalogue bytes under an existing release ID.

Curve downloads require explicit `.hazards([name, ...])` selection. Unfiltered
`catalog()` and `resolve()` remain available for discovery, but `prefetch()` and
`materialize_all()` never implicitly download all hazards. Selecting one hazard
without `.for_area()` intentionally requests its complete published coverage.
Portfolio evaluation requires `.for_area(bounds)` to bound acquisition before
reading curve files, even when the asset table itself is small.

Downloads stream to disk with SHA256 and size verification before installation.
Materialization processes Arrow batches and uses the SDK's DuckDB streaming
writer for key validation and canonical ordering, spilling to disk as needed.
`stream` cache mode uses temporary downloads; it still requires local disk.
`materialize()` returns one ordinary `HazardDataset` for a selected hazard;
`materialize_all(directory)` returns a mapping preserving each hazard's units,
tail, native resolution and scientific provenance. No fitting or resampling occurs.

Use `source=` to select another HTTP(S) catalogue root or local directory
containing release directories. Sampled GitHub fixtures remain explicit:

```python
from crc_sdk.providers.crc_open import DOCS_FIXTURES, FIXTURE_RELEASE

fixture = HazardDataset.crc_open(fixtures=DOCS_FIXTURES, release=FIXTURE_RELEASE)
```

Fixtures emit `CRCOpenFixtureWarning`. Hazard interpretation notes emit
`CRCOpenWarning`, including for the public dataset. Coastal and river flood use
scenario-compatible SSP585-labelled curves rather than independently fitted
SSP585 projections. Cyclone labels reuse a time/scenario-independent pooled
distribution; inundation retains unspecified probability semantics. No historical
baseline is included, and pooled curves do not represent per-model ensemble spread.
The data licence is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(`CC-BY-NC-4.0`), separate from the SDK software licence.

See the [access contract](docs/crc-open.md). To run the opt-in public
integration checks against the completed release:

```bash
CRC_OPEN_LIVE=1 python -m pytest -q tests/test_crc_open_public.py
```

### NOAA ISD station temperature observations

```python
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

plan = (
    HazardDataset.isd("temperature_max")
    .for_stations(["71624099999"])  # Toronto Pearson, USAF + WBAN
    .years(2000, 2024)
    .quality(minimum_hour_fraction=0.75, minimum_day_fraction=0.9)
    .execution(workers=4, memory_limit="256MB")
    .cache("data/isd", mode="reuse")
    .canonicalize(policy=BlockExtremaPolicy.curated(
        minimum_years=20, diagnostics="data/isd-fit-diagnostics.parquet",
    ))
)
print(plan.explain())
plan.prefetch()
plan.write_annual_extremes("data/isd-annual.parquet")
stations = plan.materialize("data/isd-curves.parquet")
```

`temperature_min` fits cold extremes with lower-tail Gumbel L-moments; the
maximum recipe defaults to GEV L-moments. Both measure extremes of sampled
`TMP` observations, named `isd_temperature_max` / `isd_temperature_min`, not
instrument-based daily TXx/TNn. Defaults accept QC codes 1 and 5, require 18
valid UTC hour bins per day and 90% valid calendar days per year. Duplicate
reports cannot inflate coverage. Absent station/year objects (HTTP 404) and
incomplete years have null annual values; other network errors raise. The
annual QA table includes coordinates, elevation, valid days and report counts.

Arrow streams selected CSV columns into DuckDB with a memory limit **per
worker** and temporary spill storage. Only daily summaries are cached and
shared between recipes. Fitting reads one station's annual series at a time;
`write_annual_extremes` writes bounded Arrow batches. `annual_extremes()` is an
in-memory QA view. Cache modes are `reuse`, `offline`, `refresh` and `stream`;
offline verifies checksums without network access. The manifest records exact
station/year requests, QC/completeness policy, retrieval dates, raw SHA-256,
ETag, Last-Modified, licence and reduced-file checksums. The source version
`isd-2025-08-24` identifies the documented archive cutoff, not an immutable
upstream release. Source provenance retains the request and combined reduction
checksum. No hourly files are retained.

Curves carry WKB **station points**, stable station source IDs, and H3 lookup
indexes. They imply no coverage of surrounding land or assets. Coordinate
jumps above 0.05 degrees are rejected; smaller changes and instrumentation
still need review. No homogenization is performed. Voronoi coverage,
precipitation/gust recipes and station/grid validation
are outside this adapter's supported recipes.

[NOAA's retirement notice](https://www.nesdis.noaa.gov/news/service-location-change-integrated-surface-data-global-hourly)
states ISD updates ended in August 2025; complete-year requests stop at 2024.
The adapter uses the public [AWS ISD archive](https://registry.opendata.aws/noaa-isd/).
Use GHCNh for current observations. Derived outputs carry NOAA attribution and
a modification notice; originator terms may apply to national contributions.
See [access and scientific decisions](docs/spikes/isd.md) and the
[worked notebook and pipeline](https://github.com/RiskThinking/crc-docs#readme).

### NOAA GHCNh station temperature observations

[GHCNh](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly)
is NOAA's successor to ISD, with additional historical sources and daily
updates. Use its explicit adapter with the same workflow:

```python
from crc_sdk.workflows import BlockExtremaPolicy, HazardDataset

plan = (
    HazardDataset.ghcnh("temperature_max")
    .for_stations(["USW00094846"])  # Chicago O'Hare, GHCN identifier
    .years(2000, 2025)
    .quality(minimum_hour_fraction=0.75, minimum_day_fraction=0.9)
    .execution(workers=4, memory_limit="256MB")
    .cache("data/ghcnh", mode="reuse")
    .canonicalize(policy=BlockExtremaPolicy.curated(
        minimum_years=20, diagnostics="data/ghcnh-fit-diagnostics.parquet",
    ))
)
plan.write_annual_extremes("data/ghcnh-annual.parquet")
stations = plan.materialize("data/ghcnh-curves.parquet")
```

GHCNh uses eleven-character GHCN station IDs, including some with hyphens.
They are not interchangeable with numeric ISD IDs; use NOAA's
[station list](https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/doc/ghcnh-station-list.csv).
There is no automatic station crosswalk or source fallback. Complete years
exclude the current UTC year. Both products share the daily-summary schema,
completeness gates, annual audit and station fitter. GHCNh curves are named
`ghcnh_temperature_max` / `ghcnh_temperature_min`, with `noaa-ghcnh:` source
IDs and separate cache identities. Equivalent workflow syntax does not imply
identical station histories or fitted curves.

The GHCNh reader projects eight Parquet columns through HTTP range requests,
streams Arrow batches into DuckDB and caches daily summaries. Units are already
Celsius. Default QC accepts blank integrated flags and source-specific good
legacy codes. Code `5` is accepted for legacy ISD sources but rejected for
sources where it means "removed". `.quality(quality_codes=("",))` narrows the
policy to unflagged observations; overrides cannot reinterpret a bad code as
good. The manifest records the source URL, ETag, Last-Modified, object size,
projected columns and their batch fingerprint, and reduced-file SHA-256.
The projection fingerprint is not a checksum of the entire upstream object.
Conditional range reads reject upstream changes during acquisition.
`ghcnh-v1` identifies the product major version, not an immutable release;
refresh obtains the current snapshot, while offline reuses verified reductions.

The [GHCNh AWS registry](https://registry.opendata.aws/noaa-ghcnh/) explicitly
labels the data CC0-1.0. Outputs retain NOAA attribution and a derivation notice.
See [access, units and QC decisions](docs/spikes/ghcnh.md). This adapter uses the
same two sampled-temperature recipes and point support as ISD; it does not add
precipitation, gust or regional interpolation recipes.


### Select observation stations by area

Both station adapters accept the same WGS84 bounding-box workflow as gridded
adapters. Station discovery reads a small NOAA catalogue, not observation files:

```python
plan = (
    HazardDataset.ghcnh("temperature_max")
    .for_area((-87.94, 41.95, -87.92, 41.97))
    .years(2000, 2024)
    .cache("data/ghcnh", mode="reuse")
    .canonicalize()
)
selected = plan.stations()       # Arrow station metadata; no observation downloads
pinned = plan.resolve_stations() # A new plan with explicit selected_stations IDs
pinned.materialize("data/ghcnh-curves.parquet")
```

`plan.materialize(...)` can resolve the area directly. `.for_areas([box1, box2])`
selects their union without duplicate stations. `.for_stations(ids)` replaces
area selection; `.for_area(...)` replaces explicit IDs. Builders and `explain()`
remain lazy. `stations()` and `resolve_stations()` are metadata-only terminal
operations and can run before `years(...)`.

The global catalogue is cached as compact Parquet and reused across areas and
recipes. DuckDB filters its coordinates and, for ISD, known record dates against
the requested year window. The GHCNh location catalogue does not report record
dates or temperature availability; completeness is checked from observations.
Offline selection requires a checksum-verified catalogue. Refresh updates it;
`resolve_stations()` pins the selected IDs in the plan. Daily caches are keyed
by station/year and QC policy, so area and explicit selection reuse identical
reductions. Provenance records requested bounds, resolved station IDs and the
catalogue's raw checksum. An empty selection raises before fetching observations.

Selection uses catalogue station locations, without nearest-station assignment
or interpolation. Curves retain station-point support, and requested bounds do
not turn them into regional coverage. Catalogue locations and historical
observation coordinates can differ; inspect metadata and the annual audit.

## ECCC design rainfall

Read one published rainfall duration at ECCC station points:

```python
from crc_sdk.workflows import HazardDataset

plan = (
    HazardDataset.eccc_idf("1h", province="ON", version="3.40")
    .for_stations(["6158731"])
    .cache("idf-cache")
)
print(plan.explain())
plan.prefetch()
published = plan.source_curves()  # central mm/h rates and 95% confidence half-widths
curves = plan.materialize("rainfall-1h.parquet")
plan.duration("24h").cache("idf-cache", mode="offline").materialize("rainfall-24h.parquet")
```

The `geometry` extra is required for canonical point indexing. Supported durations
are `5min`, `10min`, `15min`, `30min`, `1h`, `2h`, `6h`, `12h`, `24h`.
Use `.for_area((west, south, east, north))` or `.for_areas([box1, box2])` to
select station locations within the chosen province. Multiple boxes select a
union; duplicates and overlaps never duplicate stations or archive requests.
Area selection replaces explicit IDs, and `.for_stations(...)` clears areas.
Each record retains its own midpoint horizon; H3 indexes
points without assigning them an area of representativeness. Persistent caches
support `reuse`, `offline`, `refresh`; `.cache(None, mode="stream")` uses temporary
storage. Changed station bytes under a cached release are rejected.

Release `3.40` requires staging the selected provincial ZIP on first acquisition
(about 149 MB for Ontario), then retains only small station TXT files. Archived
release `3.20` permits direct station-member range reads. Both releases are
explicitly selected; no automatic substitution occurs. See the [access contract](docs/spikes/eccc-idf.md).

Canonical `RainfallIntensity/<duration>` curves are Gumbel quantile refits of
published central rates, with a default absolute residual limit of 0.15 mm/h.
They are derived from ECCC IDF, not official ECCC curves. Confidence limits remain
separate inspection data. Individual record windows can differ. A 24-hour moving
rainfall maximum differs from calendar-day Rx1day; rainfall is not flood depth.
Source provenance carries the Open Government Licence – Canada and ECCC credit.
The [paired notebook](https://github.com/RiskThinking/crc-docs/blob/main/notebooks/eccc_rainfall_idf.ipynb)
and [pipeline](https://github.com/RiskThinking/crc-docs/blob/main/pipelines/eccc_idf_pipeline.py)
compare published rates, fitted values and release changes.

## IBTrACS tropical cyclone tracks

Inspect NOAA best-track observations without constructing a hazard curve:

```python
from pathlib import Path
from crc_sdk.workflows import IBTrACSPlan

plan = IBTrACSPlan("NA", Path("cache")).for_storms(["2024279N21265"])
print(plan.explain())  # no network access
provenance = plan.prefetch()
for batch in plan.batches():
    print(batch.num_rows)
```

NOAA updates v04r01 files in place. Save the cached source and manifest; specify
`expected_sha256` and `mode="offline"` to replay the same bytes. `refresh`
explicitly replaces the cached snapshot. Each source CSV is downloaded in full;
selection is applied while reading bounded batches. Values retain the source
columns and units, with blank fields represented as null strings. Track points
are not asset wind estimates or return-period curves. Basin files may overlap;
agency wind definitions and data quality differ. NOAA describes data access,
commercial-use guidance and citation on its [IBTrACS page](https://www.ncei.noaa.gov/products/international-best-track-archive).
The [notebook](https://github.com/RiskThinking/crc-docs/blob/main/notebooks/ibtracs_tracks.ipynb)
and [pipeline](https://github.com/RiskThinking/crc-docs/blob/main/pipelines/ibtracs_pipeline.py)
show an event inspection.

## USGS daily streamflow

Inspect daily mean streamflow at a specific USGS monitoring location:

```python
from crc_sdk.workflows import USGSStreamflowPlan

plan = (
    USGSStreamflowPlan.for_station("USGS-01491000")
    .between("2020-01-01", "2021-12-31")
    .cache("cache")
)
print(plan.explain())
source = plan.prefetch()
for batch in plan.batches():
    print(batch.num_rows)
```

Requests use the USGS daily-values API, parameter 00060 and statistic 00003.
The returned daily mean discharge retains source units, observation and series
identifiers, approval status and qualifiers. It is not inundation depth or
an instantaneous peak flow. The source is mutable and paginated requests are
not transactional; preserve the cache and manifest for replay. The SHA-256
covers canonicalized feature JSON lines, not the original HTTP response bytes.
Specify `mode="offline"` and `sha256=source["sha256"]` to verify a saved
snapshot. `refresh` deliberately fetches a new snapshot.

Credit: U.S. Geological Survey, [Water Data for the Nation](https://doi.org/10.5066/F7P55KJN);
see its [citation guidance](https://waterdata.usgs.gov/citation/) and
[data disclaimer](https://water.usgs.gov/data/disclaimer.html). The
[notebook](https://github.com/RiskThinking/crc-docs/blob/main/notebooks/usgs_streamflow.ipynb)
and [pipeline](https://github.com/RiskThinking/crc-docs/blob/main/pipelines/usgs_streamflow_pipeline.py)
show a daily-series inspection.

## CHIRPS v3 monthly precipitation

Read final monthly gridded precipitation from the Climate Hazards Center:

```python
from crc_sdk.workflows import CHIRPSMonthlyPlan

plan = (
    CHIRPSMonthlyPlan.monthly(2020, 1)
    .for_area((36.75, -1.35, 36.85, -1.25))
    .cache("cache")
)
print(plan.explain())
source = plan.prefetch()
for batch in plan.batches():
    print(batch.num_rows)
```

Each output row represents a source pixel centre and a monthly precipitation
total in millimetres. The scan excludes the source's `-9999` ocean/no-data
sentinel, including when it is absent from the COG metadata; zero is valid.
Bounds select a raster window. No spatial averaging, interpolation, daily
disaggregation or extreme-rainfall fitting is implied.

The default `stream` mode reads the requested window over HTTP and cannot
verify a full-file checksum. `cache(...)` downloads the complete monthly COG
in bounded chunks and records its SHA-256, source URL and retrieval metadata.
Use `mode="offline"` and `sha256=source["sha256"]` to verify and replay a
retained snapshot. The source may be revised; a version label does not pin its
bytes. CHIRPS v3 covers 60°S–60°N from 1981; a grid estimate is not design
rainfall or flood depth.

Source: Climate Hazards Center, UC Santa Barbara, [CHIRPS3 Data Repository](https://doi.org/10.15780/G2JQ0P),
CC BY 4.0. See the [notebook](https://github.com/RiskThinking/crc-docs/blob/main/notebooks/chirps_monthly.ipynb)
and [pipeline](https://github.com/RiskThinking/crc-docs/blob/main/pipelines/chirps_monthly_pipeline.py).

## ECCC GeoMet daily climate observations

Inspect selected Canadian daily station observations with their source flags:

```python
from crc_sdk.workflows import ECCCDailyPlan

plan = (
    ECCCDailyPlan.for_station("6158731")
    .years(2020, 2020)
    .cache("cache")
)
print(plan.explain())
sources = plan.prefetch()
for batch in plan.batches():
    print(batch.num_rows)
```

The selector is the seven-character ECCC climate ID, not the separate
Historical Climate Data web station ID. GeoMet contains a selected subset of
stations. Output includes the source's daily measurements and flags; nulls and
trace flags are retained, without applying a quality or completeness policy.
The source local-date label is not an assertion that all stations share one UTC
observation window. A station record is not an interpolated area or an AHCCD
homogenized series, and daily precipitation is not subdaily design rainfall.

Each year is fetched in bounded pages and retained as checksummed canonical
feature records. Manifests include raw HTTP page checksums and request URLs.
Pages are not an upstream transactional snapshot, and the service can revise
data. For replay, keep the cache and pass `mode="offline"` with a mapping of
years to the recorded `sha256` values.

Data source: Environment and Climate Change Canada. The [GeoMet daily
collection](https://open.canada.ca/data/en/dataset/5f963c2d-d4ed-5a79-8a31-c9c582ca5098)
is offered under the Open Government Licence – Canada and the [ECCC data
services licence](https://eccc-msc.github.io/open-data/licence/readme_en/).
The [notebook](https://github.com/RiskThinking/crc-docs/blob/main/notebooks/eccc_daily_observations.ipynb)
and [pipeline](https://github.com/RiskThinking/crc-docs/blob/main/pipelines/eccc_daily_pipeline.py)
show an observation and flag audit.

## HYDAT daily mean hydrometric observations

Inspect historical Canadian gauge observations from ECCC GeoMet:

```python
from crc_sdk.workflows import HYDATDailyPlan

plan = (
    HYDATDailyPlan.for_station("02HA014")
    .years(2020, 2020)
    .cache("cache")
)
print(plan.explain())
sources = plan.prefetch()
for batch in plan.batches():
    print(batch.num_rows)
```

Output retains daily mean discharge in m³/s, gauge level in metres, missing
values and the source's English and French data symbols. The level datum is
not supplied by this daily endpoint and is recorded as unspecified. Gauge
level is not asset flood depth or necessarily elevation above sea level; daily
mean flow is not instantaneous peak flow. No quality or completeness policy
is inferred. See [Water Survey data symbols](https://wateroffice.ec.gc.ca/contactus/faq_e.html)
and [datum guidance](https://wateroffice.ec.gc.ca/report/datum_faq_e.html).

Each year is fetched in bounded pages and stored as checksummed canonical
feature records; manifests retain page URLs and raw HTTP response checksums.
Pages are not an upstream transactional snapshot and records can be revised.
Keep the cache and pass `mode="offline"` with recorded year-to-`sha256`
values to replay captured records.

Data source: Environment and Climate Change Canada, [Historical Hydrometric
Data](https://open.canada.ca/data/en/dataset/1ee9e14d-0814-5201-a3be-705809d8ee0e),
Open Government Licence – Canada; see the [ECCC data services licence](https://eccc-msc.github.io/open-data/licence/readme_en/).
The [notebook](https://github.com/RiskThinking/crc-docs/blob/main/notebooks/hydat_daily_observations.ipynb)
and [pipeline](https://github.com/RiskThinking/crc-docs/blob/main/pipelines/hydat_daily_pipeline.py)
show an observation audit.

## NOAA coastal extreme water levels

Inspect NOAA CO-OPS published annual high-water exceedance estimates:

```python
from crc_sdk.workflows import NOAAExtremeWaterLevelPlan

plan = NOAAExtremeWaterLevelPlan.for_station("8638610").cache("cache")
sources = plan.prefetch()
pinned = plan.cache(
    "cache", mode="offline",
    sha256={name: metadata["sha256"] for name, metadata in sources.items()},
)
info = pinned.source_info()
levels = pinned.source_levels()
annual = pinned.annual_extremes()
```

The levels are published station-point annual exceedance probabilities and
heights in metres relative to NOAA's stated datum and epoch. The reported
90% confidence limits describe uncertainty in those levels; they are not
additional probability knots. NOAA's rounded probability and return-period
labels are preserved independently. The annual series is detrended; its
reference notice is retained on each row and it cannot be shifted to another
datum as if it were an ordinary measured water level. This source is neither
asset inundation depth nor a future sea-level projection.

The station metadata, exceedance levels and detrended annual series are
fetched separately and checksummed as raw responses; they are not an atomic
upstream release. Retain all three snapshots for replay. No canonical hazard
curve or spatial interpolation is inferred. Source: NOAA National Ocean
Service, CO-OPS. See [NOAA's product documentation](https://api.tidesandcurrents.noaa.gov/dpapi/prod/),
[notebook](https://github.com/RiskThinking/crc-docs/blob/main/notebooks/noaa_extreme_water_levels.ipynb)
and [pipeline](https://github.com/RiskThinking/crc-docs/blob/main/pipelines/noaa_extreme_water_levels_pipeline.py).
