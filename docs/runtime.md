# Runtime and dependencies

[Docs index](README.md) · [Project README](../README.md)

## Dependencies

DuckDB, PyArrow, `psutil` and remote-storage transport (`fsspec`, `s3fs`,
`gcsfs`) are always installed, because every workflow uses them. Format and
geometry libraries are extras:

| Extra | Adds | Used by |
| --- | --- | --- |
| `zarr` | `zarr` | `ZarrRaster`, OS-Climate, ERA5, CMIP6, `from_zarr` |
| `netcdf` | `h5netcdf`, `h5py` | `NetCDFRaster`, EDO, NEX-GDDP-CMIP6 |
| `raster` | `rasterio` | `GeoTiffRaster` (GeoTIFF/COG via GDAL range reads), JRC flood, CHIRPS, `from_raster` |
| `geometry` | `h3`, `shapely` | H3 helpers, `polyfill_wkb`, raster-to-H3 sampling, station indexing |
| `agriculture` | `icechunk`, Zarr v3, `pyproj` (Python 3.12+) | USDA CDL; FTW fields need only DuckDB |
| `parametric` | `highspy` (the HiGHS solver) | Designing parametric covers on the LP and QP routes: `crc_sdk.parametric.design`, `ParametricPlan.objective` |
| `test` | `mypy`, `pytest`, `ruff` | Development |

Extra-gated modules import lazily and raise an `ImportError` naming the extra.

## Package layout

| Module | Responsibility |
| --- | --- |
| `crc_sdk.workflows` | User-facing plans (`HazardDataset`, `*Plan`), portfolio evaluation |
| `crc_sdk.providers` | Source discovery, catalogues, licences |
| `crc_sdk.connectors` | Format readers, canonical read/write, DuckDB helpers |
| `crc_sdk.fitting` | Quantile-table fitting |
| `crc_sdk.impacts` | Impact functions and reproducible definitions (wraps crc-framework) |
| `crc_sdk.parametric` | Parametric insurance covers: payouts, outcome samples, design, evaluation, comparison and the parametric plan |
| `crc_sdk.geometry` | H3, coverage, admin lookups, PMTiles |
| `crc_sdk.types`, `crc_sdk.schema` | Pydantic metadata and the columnar contract |
| `crc_sdk.core` | Stable re-exports from `crc_framework` |

## DuckDB resources

`DuckDBConnection.for_analytics(work_dir)` (and `RuntimeResources.detect()`)
sizes a connection from the machine:

- threads = `min(cpus, usable RAM / GiB per thread)`, with usable RAM ≈ 60% of
  detected memory and ~2.5 GiB per thread (GEOS spatial work slows when
  over-threaded);
- `memory_limit` and `max_temp_directory_size` are hard caps;
- spill files go to `work_dir`, else `CRC_DUCKDB_WORK_DIR`, else a stable
  directory under the system temp directory (`default_work_dir()`).

| Variable | Overrides |
| --- | --- |
| `CRC_DUCKDB_THREADS` | Thread count |
| `CRC_DUCKDB_MEMORY` | Memory limit |
| `CRC_DUCKDB_BYTES_PER_THREAD_GIB` | Memory budget per thread |
| `CRC_DUCKDB_WORK_DIR` | Spill directory |
| `CRC_DUCKDB_PARTITIONED_WRITE_MAX_OPEN_FILES` | Open-file ceiling for partitioned writes |
| `CRC_TIPPECANOE_THREADS` | PMTiles tiling threads |
| `CRC_TIPPECANOE_SCRATCH_FRACTION` | Share of free disk the PMTiles budget may use (default 25%) |
| `CRC_TIPPECANOE_TEMP_TO_INPUT_FACTOR` | Estimated tippecanoe scratch bytes per input byte |

Constructors that build their own connection (`OSClimateProvider`,
`ZarrRaster`, `H3Indexer`) use `for_analytics`. Passing an explicit
`connection`/`con` always wins.

## Private buckets

Use DuckDB's own [secrets DDL](https://duckdb.org/docs/configuration/secrets_manager),
passed as `setup_sql` so it runs right after extensions load.
`sql_quote`/`sql_identifier` help build it safely. `PROVIDER` is a bare keyword
(`config`, `credential_chain`), not a quoted string.

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
con = DuckDBConnection.for_analytics("work", setup_sql=setup_sql).connect()
```

GDAL-based raster reads (`GeoTiffRaster`) authenticate through GDAL's own
environment, for example `GOOGLE_APPLICATION_CREDENTIALS`.

## PMTiles

`crc_sdk.geometry.pmtiles.PMTilesBuild` streams a GeoParquet file or
Hive-partitioned glob into one `.pmtiles` archive in a single tiling pass. The
GeoParquet → GeoJSON bridge runs in DuckDB spatial SQL. `tippecanoe` and
`tile-join` ([felt/tippecanoe](https://github.com/felt/tippecanoe)) must be on
`PATH`; `require_tippecanoe()` / `require_tile_join()` raise an actionable error
when they are missing. Threads default to every detected core. A pre-flight
budget check (`check_tiling_budget`) fails early when scratch disk is too
small.

## Geometry helpers

`crc_sdk.geometry` also provides DuckDB-native H3 polyfill (`H3Indexer`), Arrow
batch polyfill (`polyfill_wkb`), raster-to-H3 sampling
(`pixel_grid_resolution`, `sample_grid_to_h3`), coverage writers
(`write_exploded_coverage`, `write_hierarchical_coverage`), nested lookup
tables (`LookupCatalog`, `write_lookup_contract`, `write_partitioned_lookup`)
and continent/IPCC-region lookups from crc-framework.
