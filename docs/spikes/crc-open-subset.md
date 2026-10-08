# CRC open subset access spike — 2026-10-08

Phase 1C starts with an explicitly limited crc-docs fixture release. A dedicated
anonymous upstream does not exist yet; sponsorship and hosting are pending.

- Private extraction source: `gs://climate_indices/bias_correction_distributions/curve_fitted/`.
  Authenticated GCS access succeeded. No SDK dependency on this private bucket.
- Existing canonical Parquet is schema 1.2, native r5 for climate indices, with
  all scenarios mixed in each r0 partition. The Toronto rx1day partition is
  42,729,689 bytes / 2,260,713 rows. Extract only literal `ssp585` rows; no
  historic, warming-band, water or cyclone rows. Preserve curves without refitting.
- Temporary public contract: a pinned release directory in crc-docs/fixtures,
  `_CATALOG.json`, `_SUCCESS` (SHA256 of catalogue), and relative Hive-style
  Parquet paths with SHA256, size and row count. No remote directory listing.
- Access: ordinary HTTP GET (GitHub raw content after the fixtures are committed
  and pushed), or a local directory / HTTP server during development. No auth,
  requester-pays or cloud credentials for SDK consumers. HTTP transfers download
  only selected catalogue partitions; each fixture file stays below 10 MiB and
  the complete release below 40 MiB, safely below GitHub's ordinary file limits.
- Release IDs are immutable by contract; cached catalogue bytes and checksums pin
  a release. Prefer a commit-pinned HTTP base URL for reproducibility. There is
  deliberately no invented permanent upstream URL and no automatic live lookup.
- The fixture is a geographically varied demonstration sample, not global
  coverage or a statistically representative scientific validation dataset.
  Actual cells, hazard coverage, horizons and curve-kind counts are recorded.
- The release licence requires the maintainer's decision. Do not infer an open
  data licence from the SDK's software licence. Source attribution is retained.
- Fixture metadata is upgraded additively to 1.3 with pooled ensemble and annual
  value distribution semantics. Window lengths are unknown in these source
  files: do not invent them. Keep original fit provenance; publish only public
  interpretation fields in the catalogue, never the private registry.

The SDK accepts an explicit catalogue root via `source=...`; an opt-in
`fixtures=...` fallback emits an alpha coverage warning. The same contract can
move to dedicated hosting later without changing evaluation or curve fitting.
