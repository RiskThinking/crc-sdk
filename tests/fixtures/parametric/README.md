# Parity fixtures for `crc_sdk.parametric`

The cases in `cases/` are golden fixtures for the metrics
(`tests/test_parametric_parity.py`) and the designs
(`tests/test_parametric_design.py`) of `crc_sdk.parametric`. Each case is one
JSON document: an outcome sample, a cover, a pricing and the values the module
must reproduce. The tests read them with plain numpy; nothing imports the code
that generated them.

## Origin

Copied byte for byte from
[marc272/parametric-insurance-design](https://github.com/marc272/parametric-insurance-design),
`parity/crc/cases/`, all at commit `101326a`. That repository's
`parity/crc/README.md` describes the format, the comparison tolerances and how
the cases were generated; its script `parity/scripts/export_crc_fixtures.py`
regenerates them. Do not edit the cases here: regenerate them there and copy
them again, every case from one commit, with `git show <commit>:<path>` so that
the copies stay byte-identical. A case's `generator.commit` names the commit
of the code that generated it, which can be older than the commit it was copied
at.

The 54 cases copied (about 600 kB):

- 17 evaluation cases of fixed covers (`*-fixed-*`): 13 linear, binary and
  piecewise-linear covers whose expected values are closed forms, computed in
  exact rational arithmetic, and 4 step covers computed by the reference
  implementation;
- 8 out-of-sample cases (`*-holdout`): step covers designed on the first 70 %
  of the years and evaluated on the held-out rest, with the design premium;
- 6 k-fold cases (`*-synthetic52-kfold5-fold<i>`): step covers designed on
  four of five contiguous blocks of the 52 years and evaluated on the fifth,
  with the design premium; the MSE design for every fold i, and the weighted
  CVaR design with cost-of-capital pricing for fold 0;
- 2 fresh-draw cases (`step-{mse-pure,wcvar-b0_9-coc}-individual-synthetic52-fresh`):
  step covers designed on the 52 years and evaluated, with the design
  premium, on 50 further draws of the same synthetic process, which the case
  gives;
- 16 in-sample design cases of one asset
  (`step-{mse,eu,wcvar-b0_9,wcvar-b0_99}-{pure,coc}-individual-{synthetic,thesis}`):
  step covers designed by the reference implementation for each objective and
  pricing, with a budget of 0.8 times the expected loss;
- 5 further design cases: binary designs with a fixed trigger
  (`binary-upper-{mse-pure,wcvar-b0_9-coc}-individual-synthetic52`), the grid
  route with a trigger free on a grid and the objective at every candidate in
  `expected.profile` (`binary-upper-grid-{eu-pure,mse-coc}-individual-synthetic52`),
  and a lower-tail step design
  (`step-lower-wcvar-b0_9-pure-individual-handmade`).

Every case carries the fields `metric_options` and `expected.profile`; they
are null except in the grid-route cases, whose `expected.profile` lists the
candidates. The source README documents both.

The design test reads every case of `kind` `"design"` and `scope`
`"individual"` in `cases/`, the out-of-sample cases included, and judges it by
its `compare_parameters` (`"exact"`, `"within_ranges"` or `"none"`), and a
grid-route case by its `expected.profile` too. An out-of-sample case is also
evaluated on its `evaluation_sample` with the design premium:

- a holdout or k-fold case joins its two blocks into the full sample, in
  period order, and splits it again with `holdout` or `year_block_kfold`,
  which must return the case's two blocks; its rows carry the split's fold;
- a fresh-draw case's draws come from another source, so the test gives them
  a lineage of their own. Their periods are draw numbers that overlap the
  design years; with the design sample's lineage they would raise
  `LeakageError`.

Further design cases are picked up the same way once they are copied here.
The source's joint cases (`*-joint-*`) are for the joint designs of a later
phase and are not copied. Six more of its cases are not copied yet: three
designs (a linear cover on the grid route and two piecewise-linear designs, on
`handmade`), and three evaluation cases (two with non-default
`metric_options`, `*-options`, and a fixed lower-tail step,
`step-lower-fixed-pure-individual-handmade`), which the metric test does not
handle yet: it expects exactly 17 evaluation cases and evaluates with the
default metric options.

## Licence

The cases are data files with their own terms, rather than the repository's
AGPL-3.0-or-later. A case's terms follow its data set, which its id ends
with (before a split suffix such as `-holdout`, `-kfold5-fold<i>` or
`-fresh`):

- **`synthetic`, `synthetic52` and `handmade`: CC0-1.0**
  ([public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/)).
  These are the cases whose ids end in `-synthetic` or `-synthetic-holdout`
  (18 cases), in `-synthetic52`, `-synthetic52-kfold5-fold<i>` or
  `-synthetic52-fresh` (12) and in `-handmade` (10). They hold no third-party
  data: draws of a seeded synthetic generator, the fresh draws of the
  `-fresh` cases included, and values chosen by hand.
- **`thesis`: CC-BY-4.0**
  ([Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)).
  These are the cases whose ids end in `-thesis` or `-thesis-holdout` (14
  cases: `step-fixed-*-thesis`, `step-*-individual-thesis` and
  `step-*-thesis-holdout`). Their draws are taken from the source
  repository's parity fixture `parity/tests/data/thesis_eventset/`, which is
  derived from the IRIS event set. A copy of them keeps this attribution:
  - the draws: Marc Prinzing, parametric-insurance-design,
    `parity/tests/data/thesis_eventset/`,
    <https://github.com/marc272/parametric-insurance-design>, CC BY 4.0.
    Changes: a subsample of 160 years and 3 assets, described in the source's
    `parity/crc/README.md` ("Data sets");
  - the hazard data: Sparks, N. and Toumi, R. (2024). The Imperial College
    Storm Model (IRIS) Dataset. Scientific Data 11, 424,
    <https://doi.org/10.1038/s41597-024-03250-y>; dataset
    <https://doi.org/10.6084/m9.figshare.c.6724251.v1> (CC BY 4.0).

The script that generated the cases, and the reference implementation it
calls, are BSD-3-Clause (the source's `parity/LICENSE`). Paths above are paths
in the source repository.
