# CRC SDK

Python SDK for [Climate Risk Commons](https://github.com/RiskThinking/crc-docs):
fetch open climate-hazard data, fit it into one canonical curve format, and
evaluate asset portfolios at return periods.

| Project | Role |
| --- | --- |
| **crc-sdk** (this repository) | Data access, canonical hazard datasets, portfolio and impact workflows |
| [crc-framework](https://github.com/RiskThinking/crc-framework-rs) | Numerical core: distributions, fitting, impacts, risk metrics (installed automatically) |
| [crc-docs](https://github.com/RiskThinking/crc-docs) | Notebooks, pipelines and AI skills built on the SDK |

## Install

```sh
pip install crc-sdk                       # Python 3.11+
pip install "crc-sdk[geometry,raster]"    # add extras as needed
```

| Extra | Needed for |
| --- | --- |
| `geometry` | H3/Shapely helpers, raster and station ingest, bring-your-own rasters |
| `raster` | GeoTIFF/COG sources (JRC flood, CHIRPS, `from_raster`) |
| `netcdf` | NetCDF sources (ERA5, NEX-GDDP-CMIP6) |
| `zarr` | Zarr sources (ERA5, CMIP6, OS-Climate, `from_zarr`) |
| `agriculture` | USDA Cropland Data Layer via Icechunk (Python 3.12+) |

Missing extras raise an `ImportError` naming the extra; `import crc_sdk` never
needs them. See [runtime and dependencies](docs/runtime.md).

## Quick start

Evaluate two sites against the public CRC Open dataset. No account is needed;
downloads are verified and cached under `~/.cache/crc-sdk`.

```python
import pyarrow as pa
from crc_sdk.workflows import HazardDataset

assets = pa.table({
    "asset_id": ["office-a", "warehouse-b"],
    "longitude": [-79.38, -79.30],
    "latitude": [43.65, 43.70],
})

plan = (
    HazardDataset.crc_open()                  # public release, cached locally
    .for_area((-79.5, 43.5, -79.2, 43.8))     # west, south, east, north (WGS84)
    .hazards(["rx1day"])                      # annual maximum 1-day rainfall
    .horizons([2050])
)
print(plan.explain())                         # builders do no I/O

result = (
    plan.for_assets(assets)
    .return_periods([10, 100])
    .write_parquet("rx1day-portfolio.parquet")
)
print(result.value_columns)                   # ('value_rp10', 'value_rp100')
```

Every source follows the same pattern: build a lazy plan, inspect it with
`explain()`, then `materialize()` a canonical dataset or evaluate assets with
`for_assets(...)`. See [core concepts](docs/concepts.md).

## Data sources

| Source | Entry point | Produces | Guide |
| --- | --- | --- | --- |
| CRC Open (63 pre-fitted hazards, SSP5-8.5) | `HazardDataset.crc_open()` | Canonical curves | [CRC Open](docs/sources/crc-open.md) |
| JRC EFAS / GloFAS river flood | `HazardDataset.efas()`, `.glofas()` | Canonical curves | [JRC](docs/sources/jrc.md) |
| JRC EDO soil-moisture drought | `HazardDataset.smi()` | Canonical curves | [JRC](docs/sources/jrc.md) |
| ERA5 reanalysis extremes | `HazardDataset.era5(recipe)` | Canonical curves | [ERA5](docs/sources/era5.md) |
| CMIP6 and NEX-GDDP-CMIP6 projections | `HazardDataset.cmip6(...)`, `.nex_gddp_cmip6(...)` | Curves / annual extremes | [CMIP6](docs/sources/cmip6.md) |
| NOAA ISD / GHCNh, ECCC IDF stations | `HazardDataset.isd()`, `.ghcnh()`, `.eccc_idf(...)` | Station-point curves | [Stations](docs/sources/stations.md) |
| USGS, CHIRPS, IBTrACS, ECCC daily, HYDAT, NOAA water levels and sea level | `*Plan` classes | Source observations | [Observations](docs/sources/observations.md) |
| USDA Cropland Data Layer, Fields of The World | `AgriculturalLayer` | Crop pixels / field polygons | [Agriculture](docs/sources/agriculture.md) |
| Your own tables, Zarr, rasters | `HazardDataset.from_table/from_zarr/from_raster` | Canonical curves | [Bring your own data](docs/bring-your-own-data.md) |

## Documentation

| Read | When you need to |
| --- | --- |
| [Core concepts](docs/concepts.md) | Understand plans, cache modes, the canonical dataset and probability conventions |
| [Portfolio evaluation and impacts](docs/portfolio.md) | Evaluate assets at return periods or apply damage functions |
| [Fitting](docs/fitting.md) | Choose fit methods, censoring, L-moments, validation and diagnostics |
| [Runtime](docs/runtime.md) | Tune DuckDB, use private buckets, build PMTiles |
| [Design decisions](docs/design-decisions.md) · [Data licences](docs/data-licences.md) | Review scientific choices and source terms |
| [Development](docs/development.md) | Set up, test and release the SDK |

The [documentation index](docs/README.md) lists every page with a one-line summary.

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

## License

The SDK is licensed under [AGPL-3.0-or-later](LICENSE). Data have their own
terms: the CRC Open dataset is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
and every canonical file records its source licence and attribution. See
[data licences](docs/data-licences.md).
