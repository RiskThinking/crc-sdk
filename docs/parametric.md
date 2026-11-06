# Parametric covers

[Docs index](README.md) · [Project README](../README.md)

`crc_sdk.parametric` defines parametric insurance covers on hazard indices: a
cover pays an amount fixed in advance whenever a measured index, such as a
year's maximum daily rainfall, reaches stated thresholds. This page covers the
parts available now: covers, payout functions and their parameters. Outcome
samples, evaluation metrics, comparison and design build on them and will be
documented here as they land.

Everything here works on numpy arrays and does no I/O.

## Define a cover and evaluate its payouts

```python
from crc_sdk.parametric import Cover, IndexDefinition, StepPayout

index = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
cover = Cover(
    index,
    StepPayout(breakpoints=[60.0, 80.0, 100.0], levels=[1e3, 5e3, 2e4]),
    payout_unit="EUR",
    asset_id="warehouse-a",
)
cover.payout_values([45.0, 60.0, 85.0, 140.0])   # array([0., 1000., 5000., 20000.])
```

- **Index:** `IndexDefinition(hazard_name, unit, *, tail="upper", location=AtAsset(), aggregation, source=None)`.
  `tail="upper"` means larger values are more severe (rainfall, wind);
  `tail="lower"` means smaller values are (a drought index), as a dataset's
  `return_period_tail` says. `location` is `AtAsset()`, `AtPoint(longitude,
  latitude)`, `AtStation(station_id)` or `AtCell(cell_index)` (an H3 cell).
- **Cover:** `Cover(index, payout, *, payout_unit, asset_id=None, label=None)`.
  The payout's `tail` must equal the index's. `label` is a display name only.

## Payout families

Each family pays nothing below its first threshold and more as the index moves
to the severe side. Thresholds are in index units and listed in order of
increasing severity: increasing for `tail="upper"`, decreasing for
`tail="lower"`. A value equal to a threshold belongs to the more severe side.
Amounts are in payout units and non-negative.

| Family | Parameters | Payout at severity `s` |
| --- | --- | --- |
| `LinearPayout(attachment, exhaustion, limit)` | thresholds a < e; amount `limit` | `limit · clip((s − a)/(e − a), 0, 1)` |
| `StepPayout(breakpoints, levels)` | thresholds b_0 < … < b_{n−1}; one level each | `levels[k]` for b_k ≤ s < b_{k+1}; the last level from b_{n−1} on |
| `PiecewiseLinearPayout(knots, levels)` | thresholds c_0 < … < c_{m−1}, m ≥ 2; one level each | the points (c_k, `levels[k]`) joined linearly; the last level from c_{m−1} on |
| `BinaryPayout(trigger, amount)` | threshold t; `amount` | `amount` for s ≥ t |

`StepPayout` and `PiecewiseLinearPayout` take `monotone=True` by default, which
requires non-decreasing levels; `monotone=False` is allowed. Every built-in
family is linear in its amounts (`LinearInAmounts`): `basis(index) @ amounts`
gives the payout, and `amount_constraints(limit)` states the amounts' bounds.

## Fixed and free parameters

Each parameter is a float (a fixed term) or `Free(lower=0.0, upper=None, *,
grid=None)`, a value a design chooses within bounds. A cover with a free
parameter is a template: it has a document and a hash, but
`payout_values` raises `UnresolvedParameterError` until it is resolved.

```python
from crc_sdk.parametric import Free

template = Cover(index, StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
                 payout_unit="EUR", asset_id="warehouse-a")
template.free_parameters()   # ('levels[0]', 'levels[1]', 'levels[2]')
resolved = template.with_values({"levels[0]": 1e3, "levels[1]": 5e3, "levels[2]": 2e4})
```

- Parameter names are flat: `attachment`, `exhaustion`, `limit`,
  `breakpoints[0]`, `levels[2]`, `knots[1]`, `trigger`, `amount`.
- A free amount needs `lower >= 0`; `upper=None` leaves it unbounded.
- A free threshold needs a finite `upper` and a `grid`: a count of evenly
  spaced values, or the values themselves. A count is stored as its values.
- Invalid parameters raise `CoverDefinitionError` (a `ValueError`); invalid
  index values raise `SampleError`.

## The cover document and its hash

`cover.to_json()` returns the cover document as canonical JSON: the index, the
payout with every parameter, `payout_unit`, `asset_id`, `schema_version` and
`semantics_version`. `cover.definition_hash` is `sha256:` plus the SHA-256 of
that text, in the canonical spelling crc-framework uses for
`ImpactExpression.definition_hash`. The label is not part of the document, so
two covers with the same terms have the same hash; `asset_id` is, so the same
terms on another asset are another contract. `Cover.from_json(text)` reads a
cover back, and covers compare, hash and pickle by their document.

## User-defined payouts

A payout family is any class that implements the `PayoutFunction` protocol
(`kind`, `tail`, `index_names`, `parameters()`, `with_values()`, `evaluate()`,
`to_document()` and `from_document()`). Register it with `@register_payout`
so `Cover.from_json` can read covers that use it; an unregistered `kind`
raises `UnknownPayoutKindError`. A family that is linear in its amounts can
also implement `LinearInAmounts`.
