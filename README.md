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
| `raster` | `rasterio` | `GeoTiffRaster` (GeoTIFF/COG ingest, streamed via GDAL VSI) |
| `geometry` | `h3`, `h3ronpy`, `shapely` | `H3Indexer`, `intersecting_cells`, `cell_polygon`, other abstract H3/geometry math, vectorized batch H3 ops on Arrow data (`polyfill_wkb`, `expand_polygon_candidates`, raster-to-H3 sampling) |
| `test` | `mypy`, `pytest`, `ruff` | Development only |

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

## Canonical hazard datasets

The SDK internalizes fitted hazards as one versioned Arrow/Parquet contract.
Rows contain a canonical unsigned H3 `cell_index`, stable `source_id`, optional
source WKB, scenario dimensions, and the parameters needed to reconstruct
either a `crc_framework.FittedDistribution` or `HurdleDistribution`.
`curve_shape` is nullable because Gumbel families do not use a shape parameter;
atom probability and location are present only when `curve_kind` is `hurdle`.

The logical row key is
`(hazard_name, horizon, pathway, cell_index, source_id)`. `cell_index` is the
spatial join key, not a globally unique identifier. Canonical files are sorted
by that row key for predicate pruning and merge joins.

Dataset-wide facts are stored once as a complete JSON payload under the
`crc.hazard.metadata` Parquet key: schema version, one uncompacted H3
resolution, non-exceedance probability convention, value unit and semantics,
WKB CRS, producer, source provenance, and creation version.

Each dataset is one self-describing Parquet file, expanded by H3 cell for
spatial joins. The caller chooses its full destination path and filename.
Writes use DuckDB, and an optional configured DuckDB connection allows the same
API to use its local or cloud filesystems, extensions, secrets, and settings.
Source knots and fit diagnostics are transient ingest inputs, not a second
persisted data contract.

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
