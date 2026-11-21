# CRC SDK release notes

[Docs index](README.md)

## 1.0.0

This breaking release requires Python 3.11 or later and `crc-framework` 1.x.
Import data and workflow interfaces from `crc_sdk` and numerical, measurement,
and impact interfaces from `crc_framework`.

- Use your own variable and scenario IDs. Declare scientific meaning using
  typed measurements and probability records. Leave scenario or period absent
  when the source does not establish them. Use a catalog when you want to
  share exact definitions. [Scientific declarations](scientific-data.md)
- Read observations before you choose a fit. A completed sample result carries
  the data and source identity needed for fitting. Query supplied quantiles
  directly. Result terminals run upstream work automatically, and an execution
  scope can share work across validation, reads, and saving. [Execution](execution.md)
- Compose independent datasets and query them at assets. Spatial matching uses
  the original data support by default. Each named impact response keeps its
  own measurement, units, and resolved model. [Collections](collections.md) ·
  [Evaluation](evaluation.md)
- Simulate annual insured losses from named damage responses and policy terms.
  Combine separately insured perils with explicit dependence, compare paired
  climate runs, and recompute metrics on saved samples. [Annual insured loss](insurance.md)

Dataset and evaluation artifacts use format `2.0`. Portable collection bundles,
native evidence, taxonomy catalogs, and annual loss artifacts use format
`1.0`. [Version ledger](../crc_sdk/contracts/api/release.json)

### CRC Open availability

Production publication of CRC Open Dataset 2.0 resources and their release
index is pending. Use `CRCOpen(catalog=...)` with a release-index 1.0 document
or pointer for a current-format release that you can access. To work with your
own curves without a CRC Open release, use `Dataset.from_quantiles`.
[CRC Open discovery](sources.md#discover-a-crc-open-release)
