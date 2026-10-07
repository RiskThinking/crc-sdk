# Schema compatibility smoke test (0.7.1 ↔ 0.8.0a1)

Run 2026-10-01 with `tools/compat/` (isolated venvs: published `crc-sdk==0.7.1`
vs `crc-sdk==0.8.0a1`, both on crc-framework 0.2.6). The harness compares the
published SDK with the local checkout; reproducing these results requires the
recorded versions:

```bash
tools/compat/setup.sh /tmp/crc-compat                  # builds old/ and new/ venvs
python tools/compat/matrix.py --venvs /tmp/crc-compat --data <dir> --work /tmp/crc-work
```

`--data` needs real source slices: `hot_days.parquet` (a production distribution sample),
`rflood_dhaka.parquet` and `cyclone_miami.parquet` (first rows of the production
`distributions_final` objects). The open dataset is a live JRC **EFAS** window
(7.9–8.1°E, 50.0–50.1°N, r8, 1,639 rows).

## Results

Writers: **old** = 0.7.1 (writes 1.2), **new** = 0.8.0a1 default (1.3),
**new12** = 0.8.0a1 with `CDFCurveFitPolicy(schema_version="1.2")`.
Readers probe: `read_hazard_metadata`, `read_hazard_dataset`, table-level
`curve_quantiles` (what table-level consumers such as tile builders do), DuckDB column scan,
`distribution_from_hazard_row`, `HazardDataset.local`.

| file → reader | schema | metadata | read_hazard_dataset | everything else |
|---|---|---|---|---|
| old → old | 1.2 | ok | ok | ok |
| **old → new** (backward) | 1.2 | ok | ok | ok |
| new12 → old | 1.2 | ok | ok | ok |
| new12 → new | 1.2 | ok | ok | ok |
| new → new | 1.3 | ok | ok | ok |
| **new → old** (forward) | 1.3 | **FAIL** | **FAIL** | ok |

All four datasets (hot_days, rflood, cyclone, EFAS) behave identically.
The only failure is `schema_version: "1.3"` being rejected by 0.7.1's
`Literal["1.0","1.1","1.2"]` check. Everything that reads columns directly
works, and evaluated quantiles are digest-identical between versions on every
file.

**Numerical equivalence** (curve columns, same inputs):

| comparison | rows | result |
|---|---|---|
| hot_days / rflood / cyclone / EFAS, old vs new vs new12 (harness) | 169 / 252 / 300 / 1,639 | identical (0 kind/type mismatches, 0.0 max parameter diff) |
| production fitting CLI, previous code + 0.7.1 vs current code + 0.8.0a1: cyclone | 300 | identical |
| same, rflood (9-RP rows) | 252 | identical |
| same, **Ontario `heat_wave_frequency`, production inputs** | 631,553 | identical |
| Ontario vs an earlier production output file | 631,553 | 865 kind and 37,906 family-label differences: that environment predates crc-framework 0.2.6 (fallback allocation drift); 0.7.1 on framework 0.2.6 reproduces the *new* numbers, so the SDK change is not the cause |

## Behaviour differences found

1. **Shortened flood arrays.** The previous CLI raised on any non-9-length `rflood`
   array even with `--on-fit-failure skip`. The current CLI skips them with reason
   `axis_length_mismatch`, counts them (`fit_summary.rejected_rows`) and can list
   them in the diagnostics sidecar. On the Dhaka slice: 48 of 300 rows. This is
   deliberate (the plan asked for rejection with a diagnostic) but it is a
   behaviour change for anyone relying on the hard failure.
2. **Manifest contract.** `canonical_schema_version` is `"1.3"`, so verified 1.2
   partitions are not reused by `--resume`.
3. **Metadata size.** 1.3 adds ~7 null keys to the JSON payload; negligible.
4. **Unknown future versions.** The new reader rejects `schema_version: "1.4"`
   the same way 0.7.1 rejects 1.3; unknown *extra keys* are ignored.

## Writer timing, Ontario `heat_wave_frequency` (631,553 rows, same machine)

Peak RSS comes from a 60 s heartbeat, so it is noisy: two ordered runs gave 0.54
and 1.22 GiB. Treat differences under ~0.5 GiB as noise at this size.

| mode | peak RSS (GiB) | fit+write (s) | output (MB) |
|---|---|---|---|
| ordered, default DuckDB limit | 1.22 | 215 | 6.7 |
| ordered, `--write-memory-limit-gib 0.5` | 0.97 | 212 | 6.7 |
| unordered | 0.81 | 216 | 9.2 (+37%) |

Wall time is unaffected. The earlier synthetic 2M-row benchmark (1,192 / 598 /
360 MiB) is the clearer memory signal; the global-partition transient (3.5–4 GiB)
needs a real r0 run to confirm.

## Migration recommendations

1. **Upgrade readers before writers.** Upgrade consumers to 0.8.x
   (downstream packages pinned `<0.8`, documentation pipelines, anything calling
   `read_hazard_dataset`) before producers write 1.3. For older consumers, producers
   can pass `schema_version="1.2"`: bit-identical curves and fully readable by
   0.7.x.
2. **Add a producer switch.** Expose the 1.2/1.3 choice in the production CLI
   (`--schema-version`, default `1.2` until consumers are upgraded, then flip to
   1.3). Note 1.3-only options (`lower_bound` provenance, ensemble, semantics)
   are rejected under 1.2, so the water hazards would lose their provenance
   fields until the flip.
3. **Make readers tolerant of future minors.** Accept unknown `1.N` versions
   with a warning instead of a validation error (physical columns are
   unchanged within a minor by contract). This is an optional compatibility
   policy change; the tested reader rejects unknown schema versions.
4. **Keep table-level readers (tile builders, explorers) on columns only.** They already
   survive 1.3; do not start reading metadata JSON there.
5. **Writer default:** keep `ordered` + a `memory_limit` sized to the container
   (e.g. 40–50% of the pod limit) rather than `unordered`: same wall time, same
   file size, no loss of key clustering. Revisit only if a global r0 partition
   still exceeds the pod limit.
6. **Keep the harness manual.** `tools/compat/` is excluded from ruff and CI.
   Automated runs require network access (PyPI, JRC) and production GCS slices.
