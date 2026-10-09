# CRC canonical release access

CRC releases use a catalogue rather than remote directory listing. A source root
contains immutable release directories with:

- `_CATALOG.json`: release ID, canonical schema version, data licence and
  attribution, public hazard interpretation fields, coverage and partition paths.
- `_SUCCESS`: SHA256 of the exact catalogue bytes.
- `{hazard}/h3_r0={cell}/part-00000.parquet`: canonical curves, with SHA256,
  byte size and row count recorded in the catalogue.

SDK consumers use ordinary HTTP(S) GETs or a directory. The default fixture root
is `https://raw.githubusercontent.com/RiskThinking/crc-docs/main/fixtures/crc_open`.
`fixtures=` selects another fixture root; `source=` selects a catalogue root.
Neither route requires cloud credentials or bucket listing. Use a commit-pinned
HTTP root when the host supports it.

Cache manifests pin catalogue and partition checksums, request, licence and
attribution. Area selection prunes r0 partitions and checks recorded sparse-cell
coverage before download. Each materialized hazard keeps its native resolution,
units, tail and fitting provenance. Existing curves are never refitted.

The SSP585 fixture covers 59 climate indices plus inundation and cyclone.
Coastal and river flood are excluded because their canonical source has no
SSP585 pathway. Cyclone's upstream labels broadcast a time/scenario-independent
curve; inundation's probability semantics are unspecified. These caveats are
preserved in metadata and public catalogue notes. Fixture coverage is sampled,
and no historical baseline is included.

Fixture production uses authenticated access to
`gs://climate_indices/bias_correction_distributions/curve_fitted/`; this is a
producer concern, not an SDK source default. The crc-docs extraction tool records
source checksums and preserves curve values. Data terms are carried independently
of the SDK's software licence.
