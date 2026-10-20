# CRC Open release access

The default catalogue root is `https://data.riskthinking.ai/crc-open`. The SDK pins
`ssp585-global-2026-10-20-v1` as its default release. Pass `release=` to select
another immutable release or `source=` to select another catalogue root. Both
HTTP(S) roots and local directories contain release directories with:

- `_CATALOG.json`: release ID, canonical schema version, data licence,
  attribution, hazard interpretation fields, coverage and partition paths.
- `_SUCCESS`: SHA256 of the exact catalogue bytes.
- `{hazard}/h3_r0={cell}/part-00000.parquet`: canonical curves, with SHA256,
  byte size and row count recorded in the catalogue. Consumers use the paths
  listed in the catalogue rather than assuming one filename per r0 forever.

No release index, cloud credentials or bucket listing is required. A release is
available only when its catalogue agrees with `_SUCCESS`. Published release IDs
are immutable; requests for an unavailable release fail without fixture fallback.

## Discovery

`HazardDataset.crc_open().catalog()` returns the validated `CRCCatalog`, including
all hazards and their `CRCPartition` records. Each partition declares its r0,
relative path, SHA256, byte size and row count. Per-partition horizons, curve-kind
counts and sampled-cell coverage may also be present.

`plan.resolve()` returns a `CRCOpenSelection` containing the verified catalogue,
its SHA256, the selected partition records, their complete HTTP URLs or local
paths, and `total_bytes` / `total_rows` properties. It downloads no curve files,
does no per-partition HEAD requests, and honors persistent/offline catalogue
caches. `catalog()` ignores plan filters; `resolve()` applies them to file
selection. Construction and `explain()` remain free of I/O.

Area selection prunes by r0 and checks recorded sampled-cell coverage where
available. Horizon selection prunes files with known non-overlapping horizons.
Missing optional coverage metadata means unknown, not empty. An r0 record
indicates that the partition contains data, not that every finer cell is present.

Resolved resources are complete files, potentially containing rows outside the
requested area, pathway or horizons. The totals are full-object estimates.
Direct consumers such as DuckDB must apply row filters themselves. HTTP range
reads can reduce transfer, but they do not verify a whole-file SHA256 unless the
whole file is read. SDK downloads verify complete files instead.

## Cached materialization

Plans use a persistent local cache by default, under
`$XDG_CACHE_HOME/crc-sdk/crc-open` or `~/.cache/crc-sdk/crc-open`. `.cache(path)`
overrides its location; `.cache(None, mode="stream")` explicitly opts into
temporary downloads. Repeated use of verified cached data makes no network
requests. Explicit refresh and temporary modes can repeat downloads.

Curve acquisition requires named hazards. `prefetch()` and `materialize_all()`
reject an unspecified hazard selection before network I/O. Metadata-only
discovery can still inspect all hazards. To request every hazard intentionally,
pass `.hazards(list(plan.catalog().hazards))`. A named hazard without area bounds
requests all its published partitions; portfolio evaluation requires
`.for_area(bounds)` so a small asset collection cannot silently trigger global
acquisition. The asset table is not implicitly treated as the acquisition area.

Metadata discovery needs only the catalogue and marker. A persistent cache pins
the catalogue checksum, release identity, licence and attribution. `reuse` uses
verified cached metadata; `refresh` fetches metadata again and rejects changes
under the same release ID. Offline discovery needs only cached metadata, whereas
offline materialization also requires every selected partition to be cached and
checksum-valid.

Partition downloads use bounded reads and install files atomically after checksum
and size verification. Materialization validates partition metadata and row
counts, filters bounded Arrow batches, and uses the canonical streaming writer
for key validation and external sorting. Output is installed only after the
complete selection passes validation. Temporary download and spill space are
required, including with `cache(None, mode="stream")`.

Each materialized hazard keeps its native resolution, units, tail and fitting
provenance. Existing curves are never refitted. Schema 1.2 releases remain
readable, with catalogue declarations carried into schema 1.3 output metadata.

## Fixtures and interpretation

Use `fixtures=` and an explicit fixture release to read sampled releases from
`https://raw.githubusercontent.com/RiskThinking/crc-docs/main/fixtures/crc_open`.
Fixtures emit `CRCOpenFixtureWarning`. Public hazard interpretation notes use
`CRCOpenWarning` and remain visible for complete releases.

The public SSP585 release includes 59 climate indices plus inundation, cyclone,
coastal flood and river flood. Coastal and river flood use scenario-compatible
SSP585-labelled curves rather than independently fitted SSP585 projections.
Cyclone reuses a time/scenario-independent curve; inundation's probability
semantics are unspecified. No historical baseline is included. These caveats
remain in catalogue notes and canonical metadata.

Data terms are independent of the SDK software licence. The CRC Open dataset is
licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(`CC-BY-NC-4.0`): attribution, and no commercial use without a separate licence.
