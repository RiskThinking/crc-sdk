# CRC Open

[Docs index](../README.md) · [Project README](../../README.md)

CRC Open is a public release of pre-fitted canonical curves: 59 climate indices
plus inundation, cyclone, coastal flood and river flood under SSP5-8.5. No
credentials are needed and nothing is refitted.

| Item | Value |
| --- | --- |
| Entry point | `HazardDataset.crc_open(release=..., source=..., fixtures=...)` |
| Default root | `https://data.riskthinking.ai/crc-open` |
| Default release | `ssp585-global-2026-10-20-v1` (release IDs are immutable) |
| Default cache | `$XDG_CACHE_HOME/crc-sdk/crc-open` or `~/.cache/crc-sdk/crc-open` |
| Data licence | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): attribution, no commercial use without a separate licence |

## Discover without downloading curves

```python
from crc_sdk.workflows import HazardDataset

plan = HazardDataset.crc_open()
catalog = plan.catalog()                        # every hazard in the release
print(sorted(catalog.hazards))

selection = plan.hazards(["rx1day"]).horizons([2050]).resolve()
print(selection.total_bytes, selection.total_rows)
for partition, url in zip(selection.partitions, selection.urls):
    print(partition.h3_r0, partition.rows, url)
```

`catalog()` ignores plan filters; `resolve()` applies hazard, area and horizon
filters to file selection. Resolved URLs are **whole files** that may contain
rows outside the request, so direct consumers must filter rows themselves:

```python
import duckdb

with duckdb.connect() as con:
    con.execute(
        "SELECT count(*) FROM read_parquet(?, hive_partitioning=false) "
        "WHERE pathway = 'ssp585' AND horizon = 2050",
        [list(selection.urls)],
    ).fetchone()
```

Area selection uses conservative r0 bounds (H3 children can extend outside
their parent). An available r0 partition does not imply that every finer cell
is present. Missing optional coverage metadata means "unknown", so the file is
kept.

Resolved files are whole partitions: direct readers must apply their own row
filters, and HTTP range reads skip the whole-file SHA-256 check that SDK
downloads perform.

## Materialize verified data

```python
plan = (
    HazardDataset.crc_open()
    .for_area((-79.5, 43.5, -79.2, 43.8))
    .hazards(["rx1day"])
    .horizons([2050])
)
plan.prefetch()
curves = plan.materialize("work/rx1day.parquet")      # one hazard
by_hazard = plan.materialize_all("work/crc-open")     # several hazards
```

- Curve downloads need explicit `.hazards([...])`. To fetch everything on
  purpose, pass `.hazards(list(plan.catalog().hazards))`. A hazard without
  `.for_area()` requests its whole published coverage.
- Downloads stream to disk and are installed only after SHA-256 and size
  verification. Materialization filters bounded Arrow batches and writes through
  the canonical streaming writer, which spills to disk when needed.
- Output keeps each hazard's native resolution, units, tail and provenance.
  Schema 1.2 releases are carried into 1.3 output metadata.
- `offline` discovery needs only cached metadata; offline materialization also
  needs every selected partition. `refresh` rejects changed catalogue bytes
  under an existing release ID.

## Release layout

Each release directory, over HTTP(S) or on local disk (`source=`), contains:

- `_CATALOG.json`: release ID, schema version, licence, attribution, hazard
  notes, coverage and partition records (path, SHA-256, bytes, rows, optional
  horizons and sampled cells);
- `_SUCCESS`: SHA-256 of the exact catalogue bytes; a mismatch is rejected;
- `{hazard}/h3_r0={cell}/part-00000.parquet`: canonical curves. Catalogue
  paths are authoritative.

## Interpretation

`CRCOpenWarning` carries hazard notes:

- Coastal and river flood use scenario-compatible SSP585-labelled curves, not
  independently fitted SSP585 projections.
- Cyclone uses one time- and scenario-independent pooled distribution.
- Inundation's probability semantics are unspecified.
- No historical baseline is included, and pooled curves do not show per-model
  ensemble spread.

## Fixtures and tests

Small sampled releases for documentation live in
[crc-docs](https://github.com/RiskThinking/crc-docs/tree/main/fixtures/crc_open)
and emit `CRCOpenFixtureWarning`:

```python
from crc_sdk.providers.crc_open import DOCS_FIXTURES, FIXTURE_RELEASE

fixture = HazardDataset.crc_open(fixtures=DOCS_FIXTURES, release=FIXTURE_RELEASE)
```

Opt-in live checks: `CRC_OPEN_LIVE=1 python -m pytest -q tests/test_crc_open_public.py`.
