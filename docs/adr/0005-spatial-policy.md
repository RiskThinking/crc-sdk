# ADR-0005: Spatial policy

Status: Proposed

## Context

Canonical rows are keyed by H3 cell. Expanding a ~31 km reanalysis pixel into
r7 cells adds no local information, and stations, coastal points and exposure
counts each need a different mapping to cells.

## Decision

**Spatial indexing is separate from scientific resolution.** Metadata and docs
state the native resolution; the H3 resolution is a delivery choice.
Defaults follow internal practice: **r5 for climate indices, r7 for water**.

| source support | policy |
|---|---|
| grid | conservative H3 coverage (existing behaviour) |
| station | Voronoi/Thiessen polygon clipped to land or admin areas, with a maximum-distance and representativeness rule; polygon carried in `source_geometry` (WKB) |
| coastal point | nearest coastal segment within a distance limit |
| exposure counts | **conservative (area-weighted) allocation**; never duplicated across cells |

## Alternatives rejected

- Nearest-station assignment with no distance cap: silently extends a station
  across regions it does not represent.
- Duplicating counts into every touched cell: double counts exposure.

## Application

- (a) Unchanged: r5 indices, r7 water, already keyed to the source grid.
- (b) Each adapter documents native resolution; station adapters
  implement the Voronoi rule and persist the geometry.
