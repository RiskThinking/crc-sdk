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
  (`write_exploded_coverage`), and optional nested lookup derivation
  (`LookupCatalog`, `write_lookup_contract`, `write_partitioned_lookup`).
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
