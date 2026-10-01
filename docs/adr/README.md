# Architecture decision records

Short records of decisions that shape the canonical hazard-curve contract
(schema 1.3) and the open-data adapters built on it. Each states the decision,
the alternatives considered, and how it applies to

- **(a) pooled proprietary CDFs** — RiskThinking's bias-corrected distributions
  fitted by the production CDF pipeline, and
- **(b) open per-model data** — ERA5, NEX-GDDP-CMIP6, CanDCS, ECCC and similar.

| ADR | Decision |
|---|---|
| [0001](0001-ensemble-identity.md) | Optional `ensemble` descriptor; open adapters default to one member per dataset |
| [0002](0002-temporal-window-and-pathways.md) | `temporal_window` metadata, `horizon` = window centre, pathway crosswalk |
| [0003](0003-probability-semantics.md) | `probability_semantics` and the explicit T↔p convention |
| [0004](0004-hazard-registry.md) | Optional hazard registry; internal catalogue registers privately |
| [0005](0005-spatial-policy.md) | Spatial indexing is separate from scientific resolution |
| [0006](0006-fit-methods-reproducibility.md) | Fit methods, provenance, tolerance-based reproducibility |
| [0007](0007-diagnostics-sidecar.md) | Opt-in row-level diagnostics Parquet sidecar |

All records are **Proposed** until the maintainer accepts them. Schema 1.3 is
metadata-only: physical columns, the row key and the sort order are unchanged,
and every 1.0–1.2 file reads unchanged.
