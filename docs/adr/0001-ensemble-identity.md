# ADR-0001: Ensemble identity

Status: Proposed

## Context

Proprietary curves are pooled upstream: no model, member or scenario identity
survives into the canonical files. Open sources (NEX-GDDP-CMIP6, CanDCS, ERA5
ensemble) are per-model. Spread between models is not year-to-year
variability, so pooling them silently answers a different question.

## Decision

Add an optional `ensemble` object to dataset metadata (schema 1.3):

| field | values |
|---|---|
| `pooling` | `single_member` \| `pooled` \| `unknown` |
| `models` | list of names, or a count |
| `members` | list of member ids, or a count |
| `scenario`, `downscaling_method`, `bias_adjustment`, `pooling_method` | free text, optional |

Absent means `unknown`; nothing is inferred or warned about.

Defaults:

- Pooled proprietary CDFs are legitimate and are labelled `pooled`, not forbidden.
- Open adapters produce **one model / member / scenario / window per
  canonical dataset** (`single_member`). Changes across models are summarised
  separately. An open adapter may offer an explicitly labelled
  `pooled` output for comparison with our product; it is never the default.

## Alternatives rejected

- Encode model identity in `source_id`: opaque, unqueryable, breaks the row key
  meaning.
- A model dimension in the row key or a collection API: a physical schema
  change requiring a concrete use case beyond dataset-level identity.

## Application

- (a) The production CDF fitting pipeline sets `pooling="pooled"` and
  `pooling_method` as documented upstream. Existing files are not rewritten.
- (b) Each adapter writes `single_member` with `models=[<one>]`, `scenario`
  and `downscaling_method`.
