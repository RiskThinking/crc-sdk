# Source-data licence policy (discussion draft)

Status: **open** (plan question 4). This is a decision aid, not legal advice:
the "verified" column says whether the licence text was checked against the
publisher's page on 2026-10-01 or is recalled and still needs a check before
we publish anything derived from it. Have counsel confirm before any
redistribution decision.

## Our own position

- **Code:** crc-sdk is AGPL-3.0-or-later. That governs the *code*; it does not
  attach to data the code produces, and no source licence below is "viral" onto
  our code.
- **Derived canonical files** (fitted parameters, H3 curves) are adaptations of
  the source data. They inherit each source's attribution and
  "indicate changes" duties. None of the sources below currently carry
  share-alike or non-commercial terms *as published today* — but see the CMIP6
  caveat.
- **Proprietary products** (the pooled bias-corrected CDFs, VELO/CDT) are
  separate works. Open baselines may be shown beside them; the open data does
  not make the proprietary data open, and the proprietary data does not make
  the open data closed.
- Schema 1.3 gives us the place to carry this: `source.licence`,
  `attribution`, `retrieved_at`, `checksum` (and `ensemble` for per-model
  identity). Recommendation: make `licence` + `attribution` **required by the
  adapters** (not the schema) so every open canonical file self-describes.

## Decision table

| Source | Published licence | Implication for derived canonical files | Collision / risk with our position | Recommended path | Verified |
|---|---|---|---|---|---|
| ERA5 / ERA5-Land (CDS) | CC-BY 4.0 since 2 Jul 2025 (previously "Licence to use Copernicus Products"); portal licence acceptance still gates API downloads | Commercial use and redistribution of derived data allowed with credit + link + "changes made" notice | Data fetched before July 2025 was under the older licence (similar terms, attribution wording differs). No share-alike. | Adapter writes `licence="CC-BY-4.0"`, attribution *"Contains modified Copernicus Climate Change Service information [year]"*, and the retrieval date so pre/post-switch downloads are distinguishable. Surface a clear error when the portal licence is not accepted (already in Phase 5). | Yes ([search summary of Copernicus CC-BY change](https://casrai.org/guides/copernicus-climate-data-store-c3s)); re-check wording on the CDS page |
| NEX-GDDP-CMIP6 (NASA) | Blanket CC0 since Sep 2022; upstream CMIP6 model terms of use still apply and models carry CC-BY-4.0 or CC0 labels | Redistribution allowed; credit the NASA dataset *and* the modelling groups as good practice | Pre-2022 vintages and some mirrors still show CC-BY-SA-4.0 on individual files — a share-alike claim could attach if we ingest a mirror that kept it. "Any derivatives are governed by the original terms of use" is ambiguous. | Do **not** hard-exclude models (the plan's default). Instead record the per-model licence label read from the file metadata in `source.licence`, **flag** anything not CC0/CC-BY, and fail `strict=True` on CC-BY-SA/NC. Cite the NASA licence page in docs. | Yes ([Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/NASA_GDDP-CMIP6), [AWS registry](https://registry.opendata.aws/nex-gddp-cmip6/)) |
| CanDCS-M6 / U6 (ECCC, PCIC) | Open Government Licence – Canada | Free commercial use; must keep attribution statement, link the licence, not imply endorsement; "individual model datasets and derived products are subject to the source organization's terms" | Same upstream-CMIP6 caveat as above. OGL-Canada's non-endorsement clause matters for marketing copy. | Attribution *"Contains information licensed under the Open Government Licence – Canada"* plus PCIC/ECCC credit; carry model labels; keep non-endorsement language out of sales material. | Yes ([open.canada.ca record](https://open.canada.ca/data/en/dataset/f73d6939-912a-4add-a291-c233fc5d1946)) |
| ECCC IDF, climate-daily, climate stations | Open Government Licence – Canada | As above | None beyond attribution / non-endorsement. IDF *values* are authority-published design numbers: displaying our refit beside them must say it is a refit, not the official curve. | Store the published knots and ours separately; label refits "derived from ECCC IDF". | Yes ([Engineering Climate Datasets](https://catalogue.ec.gc.ca/geonetwork/srv/api/records/2b9bc161-ca00-4a1e-9c75-58ed621ef4b1)) |
| JRC EDO (drought) | CC-BY 4.0 | Commercial reuse with credit | None | `licence="CC-BY-4.0"`, cite EDO + product version (already in the cache manifest). | Yes ([JRC dataset page](https://data.jrc.ec.europa.eu/dataset/afa8a5ee-5473-439a-b062-ffdaedc38b2d)) |
| JRC EFAS / GloFAS flood hazard | CC-BY 4.0 (GloFAS maps); "no restriction on use or distribution" per JRC | Commercial reuse with credit | Source-page wording differs between the JRC portal, the EFAS pages and mirrors; EFAS 3.1.1 should be confirmed on its own record | Use the JRC record's licence string verbatim; keep the confirmation in the spike note. | Partly ([JRC GloFAS record](https://data.jrc.ec.europa.eu/dataset/da4d7f64-a5c3-403f-bd2b-11a97176031e)); confirm EFAS 3.1.1 |
| CEMS fire danger, C3S water-level indicators | Copernicus CC-BY (same switch as ERA5) | As ERA5 | Per-dataset licence acceptance in the portal | As ERA5; Phase 5 adapter. | Yes (Copernicus-wide change); confirm per dataset |
| IPCC AR6 sea-level (Zenodo) | CC-BY 4.0 on the Zenodo record (recalled) | Credit authors + DOI | None expected | Cite DOI in `source.uri`. | **No** – check the Zenodo record |
| STORM tropical-cyclone winds | Research data portal licence (recalled: CC0/CC-BY) | Credit | Unknown; model output not observations | Resolve in the Phase 6 spike before building. | **No** |
| CHIRPS v3 | Recalled as open/public-domain style | Credit UCSB CHC | Unconfirmed; some derived inputs (IMERG, ERA5) carry their own terms | Resolve in the Phase 6 spike. | **No** |
| GHSL population / built-up | CC-BY 4.0 (recalled, EU JRC) | Credit | None expected | Spike confirms. | **No** |
| ECCC HYDAT | OGL-Canada (recalled) | As ECCC | None | Spike confirms. | **No** |
| NOAA IBTrACS, FEMA NFHL | US government works; NOAA asks for credit (recalled) | Credit | Third-party-sourced fields inside IBTrACS may have their own terms | Event-data utility only; spike confirms. | **No** |
| ISIMIP3b | Per-dataset (recalled CC-BY 4.0 / CC0) | Credit data + impact-model identities | Upstream climate inputs' terms | Phase 6 spike. | **No** |
| NRCan flood-map inventory | OGL-Canada (recalled) | As ECCC | Polygons are regulatory zones; legal misuse risk is about interpretation, not licence | Phase 6 spike; wording in docs. | **No** |
| NOAA ISD / GHCNh (NCEI) | NOAA Open Data Dissemination terms: open use; attribution requested; must not imply NOAA endorsement; modified data must not be presented as unaltered NOAA data | Station-derived annual extremes are modified data: label them as derived from NOAA ISD | NOAA notes GHCNh replaces ISD; WMO Resolution 40 position for some national contributions not confirmed | Spike confirms terms for the chosen product; attribution + "modified" wording in the adapter. | Partly ([NCEI NODD terms](https://www.ncei.noaa.gov/products/ncei-data-noaa-open-dissemination-program)); WMO Res. 40 **not** checked |
| **RiskThinking open subset** (our release) | We choose (suggested CC-BY-4.0) | Must carry attributions required by whatever upstream inputs the bias correction used (reanalysis/CMIP6 terms) | Derived-from-ERA5/CMIP6 inputs: confirm the CC-BY / CC0 obligations are satisfied by our attribution page; pooled proprietary curves published openly sit beside "open per-model only" advice below | Legal review of upstream inputs list; ship attribution in `_CATALOG.json`; reconsider cross-cutting item 3 (it assumed we would not publish pooled curves). | No |
| OS-Climate / WRI hazard layers | Mixed per layer | Per layer | Aqueduct 4.0 etc. are explicitly out of scope | Carry the layer's own licence from the OS-Climate metadata. | n/a |

## Cross-cutting issues worth deciding

1. **Required attribution string per source.** Put it in the adapter, emit it
   in `source.attribution`, and print it from `HazardDataset.explain()` and any
   report writer so it survives into screenshots and PDFs.
2. **Share-alike contamination.** None of the target sources is share-alike
   today, but older CMIP6 mirrors are. Rule: `strict=True` rejects any source
   whose recorded licence contains `-SA` or `-NC`; non-strict warns.
3. **Redistribution of derived canonical files** (publishing them, not just
   using them internally): permitted under every verified licence with
   attribution and a changes notice. Still open: whether we *want* to publish
   them, given the pooled proprietary curves share a schema. Update 2026-10-01: we are publishing the pooled SSP5-8.5 curves as a subset
   (see the open-subset row), so the earlier advice (per-model only) no longer
   holds for that release; keep per-model for connector-derived files and label
   the pooled release unmistakably as pooled.
4. **Open-vs-proprietary comparison results.** Publishing a validation that
   benchmarks our proprietary curves against open data is allowed by the open
   licences; the exposure is in what we say about the proprietary side.
   Recommendation: internal-only notebooks (as planned) until a customer need
   exists; external versions go through review.
5. **Licence drift.** Copernicus changed terms in July 2025 and NASA in 2022.
   `retrieved_at` + `checksum` + the licence string at retrieval time are the
   audit trail; re-verify licences at each adapter release.

## Next step

Before Phase 2 ships: finish the unverified rows that Phase 2–4 touch (none:
ERA5, NEX-GDDP and the Canadian sources are verified). Unverified rows block
only their own Phase 6 spikes.
