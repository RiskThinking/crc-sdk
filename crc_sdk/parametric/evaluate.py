"""Evaluate covers on an outcome sample and compare them in one table.

- `evaluate(cover, sample, *, split, ...)` returns a one-row comparison
  table: the cover's metrics (`crc_sdk.parametric.metrics`) on the sample.
- `compare(covers, sample, *, split, ...)` returns one row per cover, with
  covers as rows and metrics as columns, the layout of crc-sdk's portfolio
  output.
- `write_comparison(table, path)` writes a comparison table to one Parquet
  file atomically, with its metadata under `crc.parametric.evaluation`.
- `to_long(table)` gives the tidy form, one row per cover and metric.

A cover is a resolved `Cover`, or a design (`DesignedCover`, what `design`
returns), which brings its design sample and its premium.

**In and out of sample.** In sample means the design sample. Out of sample
means a held-out block of years (`holdout`, `year_block_kfold`), fresh Monte
Carlo draws, or any other sample that shares no year with the design sample.
Every row says which, in `split` and `fold`, and the metadata says how the
split was made. An out-of-sample evaluation needs the design sample (a
design's own, or `design_sample=` for a cover given as is) and raises
`LeakageError` when the two share a period of the same source.

**The premium used** is the `premium=` argument if given (`premium_basis`
`"stated"`); otherwise, for a design, the design premium, the contract price
priced on the design sample (`"design"`), so out-of-sample rows use the price
agreed at design; otherwise the premium priced on the evaluation sample with
`pricing` (`"evaluation"`).

Everything except `write_comparison` works in memory and does no I/O.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, get_args, runtime_checkable
from uuid import uuid4

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from crc_sdk._version import framework_version, platform_tag, sdk_version
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import SampleError, UnresolvedParameterError
from crc_sdk.parametric.metadata import (
    EVALUATION_METADATA_KEY,
    IDENTITY_COLUMNS,
    KEY_COLUMNS,
    ColumnEntry,
    CoverEntry,
    EvaluationMetadata,
    EvaluationSetup,
)
from crc_sdk.parametric.metrics import MetricOptions, metric_columns, metric_values
from crc_sdk.parametric.pricing import CostOfCapitalLoad, Pricing, PurePremium
from crc_sdk.parametric.sample import OutcomeSample, _check_out_of_sample

__all__ = [
    "DesignedCover",
    "PremiumBasis",
    "Split",
    "compare",
    "evaluate",
    "to_long",
    "write_comparison",
]

Split = Literal["in_sample", "out_of_sample"]
"""Whether a row is on the design sample or on a sample held out from it."""

PremiumBasis = Literal["stated", "design", "evaluation"]
"""Which premium a row uses: the stated one, the design's, or one priced on it."""

_INT32_MAX = 2**31 - 1


@runtime_checkable
class DesignedCover(Protocol):
    """A designed cover with its record, as `design` returns it.

    `evaluate` and `compare` take one in place of a `Cover`:

    - `cover`: the resolved cover;
    - `design_id`: the design's id (`"sha256:<hex>"`);
    - `premium`: the design premium, the contract price, priced on the
      design sample;
    - `design_sample`: the sample the cover was designed on, for the
      in-sample check and the leakage check out of sample;
    - `unique`, `ties`, `degenerate_tail`: the design's non-uniqueness report
      and degenerate-tail flag, copied into every row;
    - `to_record()`: the design's record as plain JSON types, written to the
      metadata's `designs`; a `"solver"` entry with the solver's `name` and
      `version` is listed in its `solvers`.
    """

    @property
    def cover(self) -> Cover: ...

    @property
    def design_id(self) -> str: ...

    @property
    def premium(self) -> float: ...

    @property
    def design_sample(self) -> OutcomeSample: ...

    @property
    def unique(self) -> bool | None: ...

    @property
    def ties(self) -> int | None: ...

    @property
    def degenerate_tail(self) -> str | None: ...

    def to_record(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _Item:
    """One cover to evaluate, with its design if it has one."""

    cover: Cover
    design: DesignedCover | None
    record: dict[str, Any] | None


# -- argument checks ---------------------------------------------------------


def _split(split: Any) -> Split:
    if split not in get_args(Split):
        raise ValueError(f"split must be one of {get_args(Split)}, got {split!r}")
    checked: Split = split
    return checked


def _fold(fold: Any) -> int:
    if (
        isinstance(fold, bool)
        or not isinstance(fold, (int, np.integer))
        or not 0 <= int(fold) <= _INT32_MAX
    ):
        raise ValueError(f"fold must be a non-negative integer, got {fold!r}")
    return int(fold)


def _pricing(pricing: Any) -> Pricing:
    if not isinstance(pricing, (PurePremium, CostOfCapitalLoad)):
        raise TypeError(
            "pricing must be PurePremium() or CostOfCapitalLoad(rate, q), "
            f"got {type(pricing).__name__}"
        )
    return pricing


def _options(options: Any) -> MetricOptions:
    if not isinstance(options, MetricOptions):
        raise TypeError(f"options must be MetricOptions, got {type(options).__name__}")
    return options


def _sample(sample: Any, name: str) -> OutcomeSample:
    if not isinstance(sample, OutcomeSample):
        raise TypeError(f"{name} must be an OutcomeSample, got {type(sample).__name__}")
    return sample


def _premium(premium: Any) -> float | None:
    if premium is None:
        return None
    if isinstance(premium, bool) or not isinstance(premium, (int, float, np.number)):
        raise ValueError(f"premium must be a number, got {premium!r}")
    value = float(premium)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"premium must be finite and non-negative, got {premium!r}")
    return value + 0.0


def _design_record(design: DesignedCover) -> dict[str, Any]:
    design_id = design.design_id
    if not isinstance(design_id, str) or not design_id:
        raise ValueError(f"a design needs a design_id, got {design_id!r}")
    record = design.to_record()
    if not isinstance(record, Mapping):
        raise TypeError("a design's to_record() must return a mapping")
    record = dict(record)
    if record.setdefault("design_id", design_id) != design_id:
        raise ValueError(
            f"the design's record names the design_id {record['design_id']!r}, "
            f"not {design_id!r}"
        )
    premium = design.premium
    if isinstance(premium, bool) or not isinstance(premium, (int, float, np.number)):
        raise ValueError(f"the design premium must be a number, got {premium!r}")
    if not math.isfinite(float(premium)) or float(premium) < 0.0:
        raise ValueError(f"the design premium must be finite and >= 0, got {premium!r}")
    _sample(design.design_sample, "a design's design_sample")
    unique, ties, flag = design.unique, design.ties, design.degenerate_tail
    if unique is not None and not isinstance(unique, bool):
        raise ValueError(f"a design's unique must be a bool or None, got {unique!r}")
    if ties is not None and (
        isinstance(ties, bool) or not isinstance(ties, (int, np.integer)) or ties < 0
    ):
        raise ValueError(f"a design's ties must be an integer >= 0, got {ties!r}")
    if flag is not None and not isinstance(flag, str):
        raise ValueError(f"a design's degenerate_tail must be a string, got {flag!r}")
    return record


def _items(covers: Any) -> list[_Item]:
    if isinstance(covers, (str, bytes, Cover)) or not isinstance(covers, Sequence):
        raise TypeError("covers must be a sequence of covers or designs")
    if not covers:
        raise ValueError("compare needs at least one cover")
    items: list[_Item] = []
    for entry in covers:
        if isinstance(entry, Cover):
            items.append(_Item(entry, None, None))
        elif isinstance(entry, DesignedCover):
            cover = entry.cover
            if not isinstance(cover, Cover):
                raise TypeError(
                    f"a design's cover must be a Cover, got {type(cover).__name__}"
                )
            items.append(_Item(cover, entry, _design_record(entry)))
        else:
            raise TypeError(
                f"expected a Cover or a designed cover, got {type(entry).__name__}"
            )
    return items


def _check_cover(cover: Cover, sample: OutcomeSample) -> None:
    free = cover.free_parameters()
    if free:
        raise UnresolvedParameterError(
            f"the cover is a template; resolve its free parameters before "
            f"evaluating it: {', '.join(free)}"
        )
    if cover.payout_unit != sample.loss_unit:
        raise SampleError(
            f"the cover pays in {cover.payout_unit!r}, but the sample's loss is "
            f"in {sample.loss_unit!r}"
        )
    if cover.index.unit != sample.index_unit:
        raise SampleError(
            f"the cover's index is in {cover.index.unit!r}, but the sample's "
            f"index is in {sample.index_unit!r}"
        )
    asset = sample.asset_ids[0]
    if cover.asset_id is not None and cover.asset_id != asset:
        raise SampleError(
            f"the cover is written for asset {cover.asset_id!r}, but the sample "
            f"holds asset {asset!r}"
        )


# -- splits ------------------------------------------------------------------


def _split_step(sample: OutcomeSample) -> Mapping[str, Any] | None:
    """The last holdout or k-fold step that made `sample`, if any."""
    step: Mapping[str, Any]
    for step in reversed(sample.provenance.selection):
        if step.get("kind") in ("holdout", "kfold"):
            return step
    return None


def _check_split(
    item: _Item,
    sample: OutcomeSample,
    split: Split,
    design_sample: OutcomeSample | None,
) -> OutcomeSample | None:
    """Check the split for one cover; return its design sample, if known."""
    own = None if item.design is None else item.design.design_sample
    known = own if own is not None else design_sample
    if split == "in_sample":
        if known is not None and known.content_hash != sample.content_hash:
            raise SampleError(
                "in sample means on the design sample, but the sample differs "
                "from the cover's design sample; evaluate with "
                "split='out_of_sample'"
            )
        return known
    if known is None:
        raise SampleError(
            "an out-of-sample evaluation needs the design sample to check that "
            "the two share no period: pass a design, or design_sample= with "
            "the sample the cover's terms were chosen on"
        )
    _check_out_of_sample(known, sample)
    return known


def _split_document(
    sample: OutcomeSample,
    split: Split,
    fold: int,
    design_samples: Sequence[OutcomeSample],
) -> dict[str, Any]:
    """How the split was made, for the metadata's `setup.evaluations`."""
    step = _split_step(sample)
    if step is not None and "fold" in step and int(step["fold"]) != fold:
        raise ValueError(
            f"the sample is block {step.get('block')!r} of fold {step['fold']} of "
            f"a {step['kind']} split, but fold={fold} was given"
        )
    details: dict[str, Any] = {}
    if split == "in_sample":
        method = "in_sample"
    elif step is not None and step.get("block") == "test":
        method = str(step["kind"])
        details = {key: step[key] for key in ("train_fraction", "k") if key in step}
    elif all(design.lineage != sample.lineage for design in design_samples):
        both_drawn = sample.provenance.source_kind == "monte_carlo" and all(
            design.provenance.source_kind == "monte_carlo" for design in design_samples
        )
        method = "fresh_draws" if both_drawn else "separate_sample"
    else:
        method = "disjoint_periods"
    hashes = sorted({design.content_hash for design in design_samples})
    return {
        "design_samples": hashes,
        "details": details,
        "fold": fold,
        "leakage_check": "passed" if split == "out_of_sample" else "not_applicable",
        "method": method,
        "sample_hash": sample.content_hash,
        "split": split,
    }


# -- rows --------------------------------------------------------------------


def _label(cover: Cover) -> str:
    if cover.label is not None:
        return cover.label
    return cover.definition_hash.removeprefix("sha256:")[:12]


def _row(
    item: _Item,
    sample: OutcomeSample,
    *,
    split: Split,
    fold: int,
    pricing: Pricing,
    premium: float | None,
    options: MetricOptions,
) -> dict[str, Any]:
    probability = sample.probability
    loss = sample.loss[:, 0]
    payout = item.cover.payout_values(sample.index[:, 0])
    basis: PremiumBasis
    if premium is not None:
        basis, used = "stated", premium
    elif item.design is not None:
        basis, used = "design", float(item.design.premium) + 0.0
    else:
        basis, used = "evaluation", pricing.premium(payout, probability)
    design = item.design
    row: dict[str, Any] = {
        "cover_label": _label(item.cover),
        "cover_hash": item.cover.definition_hash,
        "design_id": None if design is None else design.design_id,
        "asset_id": sample.asset_ids[0],
        "level": "asset",
        "scenario": sample.scenario,
        "split": split,
        "fold": fold,
        "sample_hash": sample.content_hash,
        "premium_basis": basis,
        "unique": None if design is None else design.unique,
        "ties": None if design is None or design.ties is None else int(design.ties),
        "degenerate_tail": None if design is None else design.degenerate_tail,
    }
    row.update(metric_values(loss, payout, probability, premium=used, options=options))
    return row


def _schema(options: MetricOptions, metadata: EvaluationMetadata) -> Any:
    identity = [
        pa.field("cover_label", pa.string(), nullable=False),
        pa.field("cover_hash", pa.string(), nullable=False),
        pa.field("design_id", pa.string()),
        pa.field("asset_id", pa.string(), nullable=False),
        pa.field("level", pa.string(), nullable=False),
        pa.field("scenario", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("fold", pa.int32(), nullable=False),
        pa.field("sample_hash", pa.string(), nullable=False),
        pa.field("premium_basis", pa.string(), nullable=False),
        pa.field("unique", pa.bool_()),
        pa.field("ties", pa.int32()),
        pa.field("degenerate_tail", pa.string()),
    ]
    metrics = [
        pa.field(column.name, pa.float64()) for column in metric_columns(options)
    ]
    return pa.schema(
        identity + metrics,
        metadata={EVALUATION_METADATA_KEY: metadata.to_json()},
    )


# -- metadata ----------------------------------------------------------------


def _sample_entry(sample: OutcomeSample, roles: set[str]) -> dict[str, Any]:
    provenance = sample.provenance.to_document()
    return {
        "asset_ids": list(sample.asset_ids),
        "basis_risk": provenance["basis_risk"],
        "details": provenance["details"],
        "draws": len(sample),
        "index_unit": sample.index_unit,
        "lineage": sample.lineage,
        "loss_model": provenance["loss_model"],
        "loss_unit": sample.loss_unit,
        "notes": provenance["notes"],
        "parent": provenance["parent"],
        "periods": [int(sample.period[0]), int(sample.period[-1])],
        "probability_semantics": provenance["probability_semantics"],
        "roles": sorted(roles),
        "sample_hash": sample.content_hash,
        "scenario": sample.scenario,
        "selection": provenance["selection"],
        "semantics": sample.semantics,
        "semantics_source": provenance["semantics_source"],
        "source_kind": provenance["source_kind"],
    }


def _metadata(
    items: Sequence[_Item],
    sample: OutcomeSample,
    *,
    evaluation: dict[str, Any],
    design_samples: Sequence[OutcomeSample],
    options: MetricOptions,
) -> EvaluationMetadata:
    roles: dict[str, tuple[OutcomeSample, set[str]]] = {
        sample.content_hash: (sample, {"evaluation"})
    }
    for design_sample in design_samples:
        entry = roles.setdefault(design_sample.content_hash, (design_sample, set()))
        entry[1].add("design")
    samples = [_sample_entry(each, kinds) for each, kinds in roles.values()]
    sources: list[dict[str, Any]] = []
    licences: list[dict[str, Any]] = []
    rng: list[dict[str, Any]] = []
    for each, _ in roles.values():
        document = each.provenance.to_document()
        sources += [
            {**source, "sample_hash": each.content_hash}
            for source in document["sources"]
        ]
        licences += document["licences"]
        rng += [
            {**record, "sample_hash": each.content_hash} for record in document["rng"]
        ]
    covers: dict[tuple[str, str | None], CoverEntry] = {}
    for item in items:
        key = (item.cover.definition_hash, item.cover.label)
        covers.setdefault(
            key,
            CoverEntry(
                label=item.cover.label,
                hash=item.cover.definition_hash,
                document=item.cover.to_document(),
            ),
        )
    records: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.record is not None:
            records.setdefault(item.record["design_id"], item.record)
    solvers: list[dict[str, Any]] = []
    for record in records.values():
        solver = record.get("solver")
        if isinstance(solver, Mapping) and solver not in solvers:
            solvers.append(dict(solver))
    unique_licences: list[dict[str, Any]] = []
    for licence in licences:
        if licence not in unique_licences:
            unique_licences.append(licence)
    return EvaluationMetadata(
        creation_version=sdk_version(),
        crc_framework_version=framework_version(),
        numpy_version=np.__version__,
        platform=platform_tag(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        solvers=tuple(solvers),
        setup=EvaluationSetup(
            basis="annual",
            metric_options=options.to_document(),
            samples=tuple(samples),
            evaluations=(evaluation,),
        ),
        columns={
            column.name: ColumnEntry(**column.to_document(sample.loss_unit))
            for column in metric_columns(options)
        },
        covers=tuple(covers.values()),
        designs=tuple(records.values()),
        sources=tuple(sources),
        assets={
            "count": len(sample.asset_ids),
            "id_column": None,
            "values_sha256": None,
        },
        licences=tuple(unique_licences),
        rng=tuple(rng),
    )


def _table(
    items: Sequence[_Item],
    sample: OutcomeSample,
    *,
    split: Any,
    fold: Any,
    pricing: Any,
    premium: Any,
    options: Any,
    design_sample: Any,
) -> Any:
    sample = _sample(sample, "sample")
    checked_split = _split(split)
    checked_fold = _fold(fold)
    checked_pricing = _pricing(pricing)
    checked_premium = _premium(premium)
    checked_options = _options(options)
    given = None if design_sample is None else _sample(design_sample, "design_sample")
    design_samples: dict[str, OutcomeSample] = {}
    for item in items:
        _check_cover(item.cover, sample)
        if item.design is None:
            continue
        own = item.design.design_sample
        if own.asset_ids != sample.asset_ids:
            raise SampleError(
                "the design sample and the evaluation sample hold different assets"
            )
        if given is not None and own.content_hash != given.content_hash:
            raise ValueError(
                "design_sample= differs from the design's own design sample; "
                "leave it out for designed covers"
            )
    for item in items:
        known = _check_split(item, sample, checked_split, given)
        if known is not None:
            design_samples.setdefault(known.content_hash, known)
    evaluation = _split_document(
        sample, checked_split, checked_fold, tuple(design_samples.values())
    )
    evaluation["pricing"] = checked_pricing.to_document()
    rows = [
        _row(
            item,
            sample,
            split=checked_split,
            fold=checked_fold,
            pricing=checked_pricing,
            premium=checked_premium,
            options=checked_options,
        )
        for item in items
    ]
    keys = [tuple(row[name] for name in KEY_COLUMNS) for row in rows]
    repeated = sorted({key[0] for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(
            f"covers repeat a row key (the same cover and design twice): {repeated}"
        )
    metadata = _metadata(
        items,
        sample,
        evaluation=evaluation,
        design_samples=tuple(design_samples.values()),
        options=checked_options,
    )
    return pa.Table.from_pylist(rows, schema=_schema(checked_options, metadata))


# -- the public functions ----------------------------------------------------


def evaluate(
    cover: Cover | DesignedCover,
    sample: OutcomeSample,
    *,
    split: Split,
    fold: int = 0,
    pricing: Pricing = PurePremium(),
    premium: float | None = None,
    options: MetricOptions = MetricOptions(),
    design_sample: OutcomeSample | None = None,
) -> Any:
    """Evaluate one cover on a sample; return a one-row comparison table.

    `cover` is a resolved `Cover`, or a design (`DesignedCover`). `split` is
    `"in_sample"` (on the design sample) or `"out_of_sample"` (on a sample
    that shares no period with it), and `fold` the fold of a k-fold split (0
    for a holdout and for fresh draws). The premium used is `premium` if
    given, else a design's premium, else the premium priced on `sample` with
    `pricing`; the row's `premium_basis` says which. `options` states the
    metrics' levels and thresholds.

    `design_sample` is the sample a cover given as is was chosen on. Out of
    sample, a design sample is required (a design brings its own), and the
    evaluation raises `LeakageError` if the two samples share a period of the
    same source; in sample, a known design sample must be `sample` itself.

    The table has the identity columns (`cover_label`, `cover_hash`,
    `design_id`, `asset_id`, `level`, `scenario`, `split`, `fold`,
    `sample_hash`, `premium_basis`, `unique`, `ties`, `degenerate_tail`)
    and one float64 column per metric, null where the metric is undefined.
    Its schema metadata holds the `EvaluationMetadata` under
    `crc.parametric.evaluation`.

    Raises `UnresolvedParameterError` for a template, `SampleError` if the
    cover's units or asset differ from the sample's or a design sample is
    missing or wrong, `LeakageError` for a shared period, `ValueError` for
    invalid arguments and `TypeError` for arguments of the wrong type.
    """
    return _table(
        _items([cover]),
        sample,
        split=split,
        fold=fold,
        pricing=pricing,
        premium=premium,
        options=options,
        design_sample=design_sample,
    )


def compare(
    covers: Sequence[Cover | DesignedCover],
    sample: OutcomeSample,
    *,
    split: Split,
    fold: int = 0,
    pricing: Pricing = PurePremium(),
    options: MetricOptions = MetricOptions(),
    design_sample: OutcomeSample | None = None,
) -> Any:
    """Evaluate several covers on one sample; return one row per cover.

    The arguments are those of `evaluate`, without `premium`: a design's
    rows use its design premium, and a cover given as is the premium priced
    on `sample` with `pricing`. `design_sample` is for the covers given as
    is; each design brings its own, and a `design_sample` that differs from
    it raises `ValueError`.

    The table has covers as rows and metrics as columns, in the order of
    `covers`. Its key (`cover_hash`, `design_id`, `asset_id`, `level`,
    `scenario`, `split`, `fold`) is unique: the same cover twice raises
    `ValueError`. To put rows of several samples, splits or folds in one
    file, concatenate the tables with `pyarrow.concat_tables` and pass
    `EvaluationMetadata.merge` of their metadata to `write_comparison`.

    Raises as `evaluate` does.
    """
    return _table(
        _items(covers),
        sample,
        split=split,
        fold=fold,
        pricing=pricing,
        premium=None,
        options=options,
        design_sample=design_sample,
    )


def write_comparison(
    table: Any,
    path: str | os.PathLike[str],
    *,
    metadata: EvaluationMetadata | None = None,
) -> None:
    """Write a comparison table to one local Parquet file, atomically.

    `metadata` defaults to the table's own (`EvaluationMetadata.from_table`);
    pass `EvaluationMetadata.merge(...)` for a concatenation of several
    tables. It is checked against the table (`EvaluationMetadata.check_table`)
    and written under `crc.parametric.evaluation`. The file is written to a
    temporary sibling and moved into place with `os.replace`, so a failed
    write leaves no partial file and any previous file intact.

    Raises `ValueError` if the metadata does not describe the table or the
    path is a URL, and `TypeError` for a table that is not a `pyarrow.Table`.
    """
    if not isinstance(table, pa.Table):
        raise TypeError(f"table must be a pyarrow.Table, got {type(table).__name__}")
    chosen = EvaluationMetadata.from_table(table) if metadata is None else metadata
    if not isinstance(chosen, EvaluationMetadata):
        raise TypeError(
            f"metadata must be EvaluationMetadata, got {type(chosen).__name__}"
        )
    chosen.check_table(table)
    text = os.fspath(path)
    if not isinstance(text, str) or "://" in text:
        raise ValueError(f"write_comparison writes local files, got {path!r}")
    target = Path(text)
    existing = dict(table.schema.metadata or {})
    existing[EVALUATION_METADATA_KEY.encode("utf-8")] = chosen.to_json().encode("utf-8")
    output = table.replace_schema_metadata(existing)
    staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        pq.write_table(output, staged)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def to_long(table: Any) -> Any:
    """Return the tidy form of a comparison table: one row per metric.

    Each row holds the identity columns of the wide row, then `column` (the
    wide column's name), `metric`, `parameter` and `parameter_value` (for
    example `"beta"` and 0.99; null for metrics without one), `unit` and
    `value`. Rows are ordered by metric column, then by the wide rows'
    order. The table's metadata is kept.

    Raises `ValueError` for a table without comparison metadata, and
    `TypeError` for a table that is not a `pyarrow.Table`.
    """
    if not isinstance(table, pa.Table):
        raise TypeError(f"table must be a pyarrow.Table, got {type(table).__name__}")
    metadata = EvaluationMetadata.from_table(table)
    metadata.check_table(table)
    identity = table.select(list(IDENTITY_COLUMNS))
    n = table.num_rows
    parts = []
    for name in table.column_names:
        if name in IDENTITY_COLUMNS:
            continue
        entry = metadata.columns[name]
        parameter: str | None = None
        value: float | None = None
        if entry.parameter:
            parameter, value = next(iter(entry.parameter.items()))
        parts.append(
            identity.append_column("column", pa.array([name] * n, pa.string()))
            .append_column("metric", pa.array([entry.metric] * n, pa.string()))
            .append_column("parameter", pa.array([parameter] * n, pa.string()))
            .append_column("parameter_value", pa.array([value] * n, pa.float64()))
            .append_column("unit", pa.array([entry.unit] * n, pa.string()))
            .append_column("value", table.column(name).cast(pa.float64()))
        )
    long = pa.concat_tables(parts)
    return long.replace_schema_metadata(table.schema.metadata)
