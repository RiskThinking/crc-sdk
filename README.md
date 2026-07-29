# CRC SDK

CRC SDK is the higher-level Python interface for Climate Risk Commons data
access, storage providers, geometry utilities, and analytical workflows.
Numerical distributions, curve fitting, impact transforms, and risk metrics are
provided by the versioned
[`crc-framework`](https://pypi.org/project/crc-framework/) dependency.

## Development

uv is the recommended tool for managing the development environment:

```shell
uv sync \
  --extra connectors \
  --extra geometry \
  --extra test

uv run pytest
uv run mypy
uv run ruff check .
```

Or simply:

```shell
python -m venv .venv
.venv/bin/python -m pip install -e ".[connectors,geometry,test]"
.venv/bin/python -m pytest
.venv/bin/python -m mypy
.venv/bin/python -m ruff check .
```

## Package boundaries

- `crc_sdk.core`, `crc_sdk.fitting`, and `crc_sdk.impacts` expose the stable
  public API of `crc_framework`.
- `crc_sdk.connectors` handles external formats and query engines, including
  DuckDB connection helpers (`DuckDBConnection`, `RuntimeResources`,
  streaming Parquet writes) and OS-Climate Zarr ingest.
- `crc_sdk.providers` describes storage and dataset discovery.
- `crc_sdk.geometry` contains geometry conversion, DuckDB-native H3 polyfill
  (`H3Indexer`), Arrow batch polyfill (`polyfill_wkb`, optional
  `geometry-vector` extra for h3ronpy Covers), exploded coverage writers
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

### Sampling canonical datasets

Persisted curve parameters can be reconstructed and sampled without returning
to the external source format or refitting the data:

```python
from crc_sdk.providers import LocalProvider
from crc_sdk.workflows import sample_hazard_at_point

provider = LocalProvider("flood.parquet")
result = sample_hazard_at_point(
    provider,
    "flood",
    longitude=6.9603,
    latitude=50.9375,
    horizon=2050,
    pathway="ssp585",
    size=10_000,
    seed=42,
)

samples = result.samples
distribution = result.distribution
```

The point workflow reads the dataset H3 resolution from canonical metadata,
converts the WGS84 point to a `cell_index`, and applies that filter together
with the requested hazard and scenario dimensions. Because canonical H3 rows
are conservative overlap candidates, `source_geometry` is used for exact
point-in-source refinement when present. `result.spatial_match` is
`"exact_geometry"` in that case; when the selected row has no source WKB it is
`"h3_cell"` and carries only cell-level spatial precision. No match and
multiple matches both raise instead of silently selecting or aggregating a
curve.

Point lookup requires the `geometry` extra. For an already selected canonical
Arrow row or table, `distribution_from_hazard_row` and `sample_hazard_row` in
`crc_sdk.workflows` reconstruct or sample directly without spatial lookup.
Sampling defaults to 10,000 values and an unseeded generator; pass `seed` for
reproducible draws.

Call `sample_hazard_at_cell` when the H3 index is already known. It accepts the
same provider, hazard/scenario filters, sample size, and seed as the point
workflow, but queries the canonical `cell_index` directly and therefore does
not perform source-geometry refinement. Both location workflows require one
matching canonical curve and raise on missing or ambiguous rows.

## License

CRC SDK is licensed under the GNU Affero General Public License, version 3 or
later.
