# CRC SDK

Scientific data, modelling workflows, and canonical datasets for
[Climate Risk Commons](https://github.com/RiskThinking/crc-docs).

The SDK owns data identity, temporal and spatial declarations, provenance, and
columnar storage. [crc-framework](https://github.com/RiskThinking/crc-framework-rs)
owns numerical laws, quantile evidence, fitting specifications, measurements,
and mathematical impacts.

## Install

```sh
python -m pip install crc-sdk
```

Python 3.11 or later is required. H3 and geometry validation are part of the core
package. Source-format extras are `raster`, `netcdf`, and `zarr`.
Source readers import those dependencies only when needed.

## Query a dataset

Supply quantile values as columns in a Python mapping. This example has two
depth values in metres at the same point and queries the 10-year and 100-year
event levels at a colocated asset.

```python
from crc_framework import DepthAboveGround, LengthUnit, Tail
from crc_sdk import Assets, Dataset, Probability, Variable

depth = Dataset.from_quantiles(
    {
        "support_id": ["station-a", "station-a"],
        "longitude": [24.0, 24.0],
        "latitude": [60.0, 60.0],
        "probability": [0.90, 0.99],
        "value": [0.3, 0.85],
    },
    variable=Variable(
        "company:water-depth",
        measurement=DepthAboveGround(unit=LengthUnit.METRE),
    ),
    probability=Probability.annual(tail=Tail.UPPER),
)
assets = Assets({"asset_id": ["warehouse"], "longitude": [24.0], "latitude": [60.0]})

events = depth.evaluate(assets).return_periods([10, 100]).read()
assert events.table["value"].to_pylist() == [0.3, 0.85]
```

The final `read()` validates the inputs, matches the asset, and evaluates the
two supplied knots. Use your own variable, support, and asset IDs. Supply a
scenario or period when your source establishes those facts. [Evaluation guide](docs/evaluation.md)

## Documentation

- [Scientific declarations](docs/scientific-data.md): variables, periods, probability,
  measurements, and exact catalogs.
- [Canonical datasets](docs/datasets.md): current rows, numerical variants,
  provenance, spatial support, and storage validation.
- [Execution and persistence](docs/execution.md): automatic upstream execution,
  explicit fitting, previews, reusable executions, streaming, and atomic writes.
- [Sources and native evidence](docs/sources.md): typed providers, temporal
  aggregation, user evidence, and explicit asset inputs.
- [Collections](docs/collections.md): selected scientific members and portable bundles.
- [Evaluation](docs/evaluation.md): match assets, query event levels, and add
  calibrated named responses.
- [Python interface contract](docs/api-reference.md): canonical imports and
  installed-package enforcement.
- [Development](docs/development.md): setup, checks, generated contracts, and
  releases.
- [Release notes](docs/release-notes.md)

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
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shlyakh"><img src="https://avatars.githubusercontent.com/u/35546708?s=80" width="80" alt="@shlyakh" /><br /><sub><b>@shlyakh</b></sub></a><br /><a href="https://github.com/RiskThinking/crc-sdk/commits?author=shlyakh" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

Contribution types: 💻 code · 👀 review · 🔬 research. We recognize contributions
both on and off GitHub using [All Contributors](https://allcontributors.org/).

## License

The SDK is licensed under [AGPL-3.0-or-later](LICENSE). Source data have their own
terms. Resource records retain their actual license, attribution, retrieval
time, and content identity; unknown facts are not replaced with guessed terms.
