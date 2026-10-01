# ADR-0004: Hazard registry

Status: Proposed

## Context

Hazard knowledge (units, tail direction, eligibility floors) is hand-copied
between repositories (with known copy errors), and per-hazard `minimum_informative_value` overrides were added only after a
shared floor turned four hazards into ~100% `no_data`. The SDK core must stay
hazard-agnostic.

## Decision

Add an **optional** registry (`crc_sdk.registry`): `hazard_name` →
`HazardSpec` with

- `unit`, `value_semantics`, `tail` (`upper`/`lower`);
- `block` definition (free text, e.g. "annual maximum of daily maximum");
- `aliases`;
- eligibility defaults: `minimum_informative_value` (a number, or `None`
  meaning *no floor*), `minimum_informative_knots`,
  `minimum_distinct_informative_values`;
- `preferred_families`.

Fitting consults the registry only when a registry is passed; nothing is
hard-coded in the fitter.

Open users get a **public seed set**: the ETCCDI indices (TXx, TNn, Rx1day,
Rx5day, CDD, …). The internal 59+ hazard catalogue registers as a **separate,
private catalogue** into the same mechanism, which ends hand copies. Open recipes **reuse internal names where definitions are
identical** (`rx1day`, `rx5day`, `fwi`, `hot_days`, …) so open baselines and
proprietary curves compare directly; names differ where definitions differ.
Duration and threshold are part of the name, e.g. `RainfallIntensity/1h`.

Resolution order for an eligibility parameter: explicit per-hazard override in
the fit policy > registry spec > policy default.

## Alternatives rejected

- Publish the internal catalogue inside crc-sdk: exposes proprietary indicator
  definitions (recommendation; the maintainer may reverse this).
- Hard-code hazard tables in the fitter: breaks hazard-agnosticism.

## Application

- (a) The production pipeline registers its catalogue (units, semantics,
  floors; signed or tiny-scaled hazards get no floor) privately.
- (b) Open recipes look up the public seed for unit, tail and block
  definition.
