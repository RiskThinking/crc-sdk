# ADR-0002: Temporal window and pathway semantics

Status: Proposed

## Context

`horizon` and `pathway` are row-key columns whose meaning is implicit.
Internally the grid is a 2010 historical baseline then 2025–2100 in 5-year
steps. Pathways mix SSPs, warming-level bands, `SV`/`RT3`, and `historic`;
crc-framework `PATHWAYS` spells the baseline `historic`, the SDK's EDO
provider writes `historical`. Time-invariant datasets (cyclone, JRC RP maps)
use an ad-hoc `2025`/`pooled`.

## Decision

### Temporal window

Add an optional `temporal_window` object to dataset metadata:

- `kind`: `window` \| `time_invariant`.
- `start_year`, `end_year`: **inclusive** bounds (required for `window`;
  `end_year >= start_year`).
- `horizon` is the window's **centre year**: `(start_year + end_year) // 2`.
  Rows keep carrying `horizon` as an int32; the window records what it means.
- `baseline`: optional `(start_year, end_year)` the anomaly or change is
  relative to.
- `calendar`: `gregorian` \| `noleap` \| `360_day` \| `unspecified`.
- `minimum_complete_years`: years of valid data required before a window is
  fitted (open adapters; recorded for audit).
- `reference_year`: required for `time_invariant`; rows carry it as `horizon`.

Time-invariant datasets declare `kind="time_invariant"` and a `reference_year`
(the epoch of the data, e.g. the JRC release epoch) instead of inventing a
placeholder. The pathway column carries the source's own label (for example
`historic`); datasets with no scenario at all use `time_invariant`.

### Pathway crosswalk

A registry of canonical pathway ids maps every known label to crc-framework
`PATHWAYS`:

- SSPs (`ssp119` … `ssp585`);
- warming-level bands (`<2 degrees` …);
- `SV`/`RT3`/`Hot House`/`Paris`/`NDC`;
- `historic` and `historical` as **aliases of one canonical id**.

Writes keep the **source's label**. Unknown labels warn; `strict=True` rejects
them. Existing files are never rewritten. The crosswalk is code, introduced
with the first adapter that needs it (Phase 1B/2); this ADR fixes the rules.

## Alternatives rejected

- Force a single spelling (`historic` or `historical`) at write time: rewrites
  production semantics and breaks consumers keyed on either.
- Window as a row column: a physical schema change for information constant
  across the dataset.

## Application

- (a) The 2010 baseline plus 2025–2100 grid is declared once per dataset;
  cyclone becomes `time_invariant` with its true reference year. Pathways
  stay as the upstream labels.
- (b) Open adapters put horizons on the same grid, so open and proprietary
  curves join on `(hazard_name, horizon, pathway, cell_index)` after the
  crosswalk.
