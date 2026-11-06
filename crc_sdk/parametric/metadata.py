"""The metadata of a comparison table: the setup behind every number.

A comparison table (`compare`, `evaluate`) carries one JSON document under
the Parquet schema metadata key `crc.parametric.evaluation`
(`EVALUATION_METADATA_KEY`), validated by `EvaluationMetadata` and written
with sorted keys. It holds:

- the environment: `schema_version`, `producer`, `creation_version` (the
  crc-sdk version), `crc_framework_version`, `numpy_version`, `platform`,
  `created_at` (UTC), and `solvers`, the solvers of any design evaluated;
- `setup`, the setup a published number needs next to it: the basis (one
  draw is one year), the metric options, every sample (source kind, draws
  and periods, semantics and who stated them, the basis-risk and loss
  models, units, lineage and split steps) and every evaluation (sample,
  split, fold, how the split was made, the design samples checked for
  leakage, the pricing);
- `columns`: for every metric column, its metric, parameter, unit and a
  one-line definition;
- `covers`: every cover's label, hash and full document;
- `designs`: the record of every design evaluated, with the label of its
  cover as evaluated (`cover_label`);
- `sources`, `licences` and `rng`: the samples' source documents, their
  licences and attributions, and the generator state before every random
  step, each tagged with its sample's hash;
- `assets`: the number of assets; the id column and the hash of the
  (asset id, value) pairs when the caller knows them. Asset values are never
  written.

The module claims no licence of its own for a comparison: it is derived data
under its inputs' terms, which `licences` lists.

`setup_text(metadata)` states the setup in plain text, one statement a line,
for a notebook or a report to print next to the numbers. Invalid metadata
raises `EvaluationError`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import ValidationError, model_validator

from crc_sdk.impacts._types import ContentHash, StrictFrozenModel
from crc_sdk.parametric.errors import EvaluationError

__all__ = [
    "EVALUATION_METADATA_KEY",
    "EVALUATION_SCHEMA_VERSION",
    "ColumnEntry",
    "CoverEntry",
    "EvaluationMetadata",
    "EvaluationSetup",
    "setup_text",
]

EVALUATION_METADATA_KEY = "crc.parametric.evaluation"
"""The Parquet schema metadata key of a comparison table."""

EVALUATION_SCHEMA_VERSION = "1.0"
"""The `schema_version` of `EvaluationMetadata`."""

IDENTITY_COLUMNS = (
    "cover_label",
    "cover_hash",
    "design_id",
    "asset_id",
    "level",
    "scenario",
    "split",
    "fold",
    "sample_hash",
    "premium_basis",
    "unique",
    "ties",
    "degenerate_tail",
)
"""The columns of a comparison table before its metric columns."""

KEY_COLUMNS = (
    "cover_hash",
    "design_id",
    "asset_id",
    "level",
    "scenario",
    "split",
    "fold",
)
"""The columns whose values identify a row of a comparison table."""


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def _plain(name: str, value: Any) -> None:
    try:
        _json_text(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must hold plain, finite JSON values: {error}"
        ) from error


def _evaluation_error(error: ValidationError) -> EvaluationError:
    """Turn a pydantic validation error into an `EvaluationError`."""
    messages: list[str] = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        original = (item.get("ctx") or {}).get("error")
        message = str(original) if isinstance(original, Exception) else item["msg"]
        messages.append(f"{location}: {message}" if location else message)
    return EvaluationError(f"invalid {error.title}: {'; '.join(messages)}")


class _EvaluationModel(StrictFrozenModel):
    """Frozen model whose validation failures raise `EvaluationError`."""

    # Hidden from type checkers, so that they keep the fields' signature.
    if not TYPE_CHECKING:

        def __init__(self, **data: Any) -> None:
            try:
                super().__init__(**data)
            except ValidationError as error:
                raise _evaluation_error(error) from error


def _unique(entries: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The entries without repeats, in first-seen order."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for entry in entries:
        text = _json_text(entry)
        if text not in seen:
            seen.add(text)
            kept.append(dict(entry))
    return tuple(kept)


class ColumnEntry(_EvaluationModel):
    """One metric column: its metric, parameter, unit and definition."""

    metric: str
    parameter: dict[str, float] | None
    unit: str
    definition: str


class CoverEntry(_EvaluationModel):
    """One cover: its label, its `definition_hash` and its cover document."""

    label: str | None
    hash: ContentHash
    document: dict[str, Any]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _plain("a cover document", self.document)
        return self


class EvaluationSetup(_EvaluationModel):
    """The setup of a comparison: basis, metric options, samples, evaluations.

    `samples` holds one document per sample (each with its `sample_hash`
    and `roles`, `"evaluation"` and/or `"design"`), and `evaluations` one per
    evaluation run (each with its `sample_hash`, `split` and `fold`).
    Raises `EvaluationError` otherwise.
    """

    basis: Literal["annual"] = "annual"
    metric_options: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _plain("setup", self.model_dump(mode="json"))
        for sample in self.samples:
            if "sample_hash" not in sample or "roles" not in sample:
                raise ValueError("every setup sample needs a sample_hash and roles")
        for evaluation in self.evaluations:
            if not {"sample_hash", "split", "fold"} <= set(evaluation):
                raise ValueError(
                    "every setup evaluation needs a sample_hash, a split and a fold"
                )
        return self


class EvaluationMetadata(_EvaluationModel):
    """The metadata of a comparison table, under `crc.parametric.evaluation`.

    `compare` and `evaluate` build it and attach it to their table;
    `from_table` reads it back, `merge` combines the metadata of several
    tables, `write_comparison` checks it against the table before writing,
    and `setup_text` states its setup in words. Raises `EvaluationError` for
    invalid fields and values that are not plain, finite JSON.
    """

    schema_version: Literal["1.0"] = "1.0"
    producer: Literal["crc_sdk.parametric"] = "crc_sdk.parametric"
    creation_version: str
    crc_framework_version: str | None
    numpy_version: str
    platform: str
    created_at: str
    solvers: tuple[dict[str, Any], ...] = ()
    setup: EvaluationSetup
    columns: dict[str, ColumnEntry]
    covers: tuple[CoverEntry, ...]
    designs: tuple[dict[str, Any], ...] = ()
    sources: tuple[dict[str, Any], ...] = ()
    assets: dict[str, Any]
    licences: tuple[dict[str, Any], ...] = ()
    rng: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name in ("solvers", "designs", "sources", "assets", "licences", "rng"):
            _plain(name, getattr(self, name))
        ids = [design.get("design_id") for design in self.designs]
        if not all(isinstance(value, str) for value in ids):
            raise ValueError("every design record needs a design_id")
        if len(set(ids)) != len(ids):
            raise ValueError("design records repeat a design_id")
        return self

    def to_json(self) -> str:
        """Return the metadata as JSON text with sorted keys."""
        return _json_text(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, text: str | bytes) -> EvaluationMetadata:
        """Read the metadata from its JSON text. Raises `EvaluationError`."""
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"evaluation metadata is not valid JSON: {error}"
            ) from error
        if not isinstance(document, dict):
            raise EvaluationError(
                "evaluation metadata must be a JSON object, got "
                f"{type(document).__name__}"
            )
        return cls(**document)

    @classmethod
    def from_table(cls, table: Any) -> EvaluationMetadata:
        """Read the metadata a comparison table carries.

        Raises `EvaluationError` if the table's schema has no
        `crc.parametric.evaluation` key, or its value is invalid.
        """
        metadata = getattr(getattr(table, "schema", None), "metadata", None) or {}
        text = metadata.get(EVALUATION_METADATA_KEY.encode("utf-8"))
        if text is None:
            raise EvaluationError(
                f"the table has no {EVALUATION_METADATA_KEY!r} metadata; build it "
                "with compare or evaluate"
            )
        return cls.from_json(text)

    @classmethod
    def merge(cls, *items: EvaluationMetadata) -> EvaluationMetadata:
        """Combine the metadata of several comparison tables into one.

        Use it with `pyarrow.concat_tables` to write the rows of several
        `compare` calls (in and out of sample, several folds or assets) into
        one file. The environments, the basis and the metric options must
        agree, and a column or design id must mean the same in each; samples,
        evaluations, covers, designs, sources, licences and generator states
        are combined without repeats. `created_at` is the latest one.

        Raises `EvaluationError` if the items disagree, and `TypeError` for an
        item that is not `EvaluationMetadata`.
        """
        if not items:
            raise EvaluationError("merge needs at least one EvaluationMetadata")
        for item in items:
            if not isinstance(item, EvaluationMetadata):
                raise TypeError(
                    f"merge takes EvaluationMetadata, got {type(item).__name__}"
                )
        first = items[0]
        for field in (
            "creation_version",
            "crc_framework_version",
            "numpy_version",
            "platform",
        ):
            values = {getattr(item, field) for item in items}
            if len(values) > 1:
                raise EvaluationError(
                    f"the metadata differ in {field}: {sorted(map(str, values))}"
                )
        options = {_json_text(item.setup.metric_options) for item in items}
        if len(options) > 1:
            raise EvaluationError(
                "the metadata differ in their metric options, so their tables "
                "have different columns"
            )
        columns: dict[str, ColumnEntry] = {}
        for item in items:
            for name, entry in item.columns.items():
                if name in columns and columns[name] != entry:
                    raise EvaluationError(
                        f"column {name!r} differs between the metadata (its unit "
                        f"{columns[name].unit!r} or {entry.unit!r}); write one table "
                        "per payout unit"
                    )
                columns.setdefault(name, entry)
        designs: dict[str, dict[str, Any]] = {}
        for item in items:
            for record in item.designs:
                design_id = record["design_id"]
                if design_id in designs and _json_text(
                    designs[design_id]
                ) != _json_text(record):
                    raise EvaluationError(
                        f"design {design_id!r} has two different records"
                    )
                designs.setdefault(design_id, record)
        samples: dict[str, dict[str, Any]] = {}
        for item in items:
            for sample in item.setup.samples:
                identity = _json_text({k: v for k, v in sample.items() if k != "roles"})
                if identity in samples:
                    roles = set(samples[identity]["roles"]) | set(sample["roles"])
                    samples[identity] = {**samples[identity], "roles": sorted(roles)}
                else:
                    samples[identity] = dict(sample)
        asset_ids = {
            asset_id for sample in samples.values() for asset_id in sample["asset_ids"]
        }
        assets = {key: value for key, value in first.assets.items() if key != "count"}
        for item in items[1:]:
            other = {key: value for key, value in item.assets.items() if key != "count"}
            if _json_text(other) != _json_text(assets):
                raise EvaluationError("the metadata differ in their asset records")
        return cls(
            creation_version=first.creation_version,
            crc_framework_version=first.crc_framework_version,
            numpy_version=first.numpy_version,
            platform=first.platform,
            created_at=max(item.created_at for item in items),
            solvers=_unique(entry for item in items for entry in item.solvers),
            setup=EvaluationSetup(
                basis=first.setup.basis,
                metric_options=first.setup.metric_options,
                samples=tuple(samples.values()),
                evaluations=_unique(
                    entry for item in items for entry in item.setup.evaluations
                ),
            ),
            columns=columns,
            covers=tuple(
                {
                    _json_text(entry.model_dump(mode="json")): entry
                    for item in items
                    for entry in item.covers
                }.values()
            ),
            designs=tuple(designs.values()),
            sources=_unique(entry for item in items for entry in item.sources),
            assets={"count": len(asset_ids), **assets},
            licences=_unique(entry for item in items for entry in item.licences),
            rng=_unique(entry for item in items for entry in item.rng),
        )

    def check_table(self, table: Any) -> None:
        """Check that this metadata describes `table`, a comparison table.

        The table must have the identity columns and exactly the metric
        columns of `columns`; its key (`cover_hash`, `design_id`,
        `asset_id`, `level`, `scenario`, `split`, `fold`) must be unique;
        and every cover, design, sample and evaluation it names must be
        described here. Raises `EvaluationError` otherwise.
        """
        names = list(getattr(table, "column_names", ()))
        missing = [name for name in IDENTITY_COLUMNS if name not in names]
        if missing:
            raise EvaluationError(f"the comparison table lacks the columns {missing}")
        metrics = [name for name in names if name not in IDENTITY_COLUMNS]
        if set(metrics) != set(self.columns):
            unknown = sorted(set(metrics) - set(self.columns))
            absent = sorted(set(self.columns) - set(metrics))
            raise EvaluationError(
                "the table's metric columns differ from the metadata's: "
                f"undescribed {unknown}, absent {absent}"
            )
        rows = table.select(list(IDENTITY_COLUMNS)).to_pylist()
        keys = [tuple(row[name] for name in KEY_COLUMNS) for row in rows]
        if len(set(keys)) != len(keys):
            raise EvaluationError("the comparison table repeats a row key")
        covers = {entry.hash for entry in self.covers}
        designs = {record["design_id"] for record in self.designs}
        samples = {sample["sample_hash"] for sample in self.setup.samples}
        evaluations = {
            (entry["sample_hash"], entry["split"], entry["fold"])
            for entry in self.setup.evaluations
        }
        for row in rows:
            if row["cover_hash"] not in covers:
                raise EvaluationError(
                    f"cover {row['cover_hash']} is not in the metadata"
                )
            if row["design_id"] is not None and row["design_id"] not in designs:
                raise EvaluationError(
                    f"design {row['design_id']} is not in the metadata"
                )
            if row["sample_hash"] not in samples:
                raise EvaluationError(
                    f"sample {row['sample_hash']} is not in the metadata"
                )
            if (row["sample_hash"], row["split"], row["fold"]) not in evaluations:
                raise EvaluationError(
                    f"no evaluation of sample {row['sample_hash']} "
                    f"({row['split']}, fold {row['fold']}) is in the metadata"
                )


# -- the setup in words --------------------------------------------------------


def _number(value: Any, where: str) -> str:
    """A finite number in at most six significant digits, without an exponent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{where} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise EvaluationError(f"{where} must be finite, got {value!r}")
    text = format(Decimal(repr(float(f"{value:.6g}") + 0.0)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    sign, digits = ("-", text[1:]) if text.startswith("-") else ("", text)
    whole, point, fraction = digits.partition(".")
    return f"{sign}{int(whole):,}{point}{fraction}"


def _numbers(values: Any, where: str) -> str:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EvaluationError(f"{where} must be a list of numbers")
    if not values:
        return "none"
    return ", ".join(_number(value, f"{where}[{i}]") for i, value in enumerate(values))


def _get(
    entry: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str
) -> Any:
    """`entry[key]` if it has the type `kind`; `None` when absent or null."""
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
        raise EvaluationError(f"{where}.{key} has the wrong type: {value!r}")
    return value


def _list(entry: Mapping[str, Any], key: str, where: str) -> list[Any]:
    """`entry[key]` as a list; empty when absent or null."""
    values = entry.get(key)
    if values is None:
        return []
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise EvaluationError(f"{where}.{key} must be a list")
    return list(values)


def _objects(entry: Mapping[str, Any], key: str, where: str) -> list[Mapping[str, Any]]:
    values = _list(entry, key, where)
    if not all(isinstance(value, Mapping) for value in values):
        raise EvaluationError(f"{where}.{key} must be a list of objects")
    return values


def _join(words: Sequence[str]) -> str:
    """`a`, `a and b`, `a, b and c`."""
    if len(words) <= 2:
        return " and ".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def _short(value: Any) -> str:
    """The first 12 hex digits of a `sha256:` hash."""
    return str(value).removeprefix("sha256:")[:12]


def _first_seen(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _years(sample: Mapping[str, Any] | None, where: str) -> str:
    """How many years a sample holds, which ones, and which block it is."""
    if sample is None:
        return "a sample the setup does not describe"
    kind = _get(sample, "source_kind", str, where)
    draws = _get(sample, "draws", int, where)
    periods = _list(sample, "periods", where)
    if periods and (
        len(periods) != 2
        or any(isinstance(p, bool) or not isinstance(p, int) for p in periods)
    ):
        raise EvaluationError(f"{where}.periods must be two integers")
    count = "an unknown number of" if draws is None else str(draws)
    span = f"{periods[0]} to {periods[1]}" if periods else ""
    if kind == "monte_carlo":
        text = f"{count} simulated years" + (f" (draws {span})" if span else "")
    elif kind == "annual":
        text = f"{count} recorded years" + (f" ({span})" if span else "")
    else:
        text = f"{count} years" + (f" (periods {span})" if span else "")
    steps = _objects(sample, "selection", where)
    step = steps[-1] if steps else {}
    block = step.get("block")
    if step.get("kind") == "holdout" and block in ("train", "test"):
        text += ", the design block" if block == "train" else ", the held-out block"
    elif step.get("kind") == "kfold" and block in ("train", "test"):
        text += (
            ", every block but the held-out one"
            if block == "train"
            else ", the held-out block"
        )
    elif step.get("kind") == "select":
        text += ", a selection of periods"
    return text


def _source(sample: Mapping[str, Any], where: str) -> str:
    kind = _get(sample, "source_kind", str, where)
    if kind == "monte_carlo":
        semantics = _get(sample, "probability_semantics", str, where)
        stated = _get(sample, "semantics_source", str, where)
        who = {"dataset": "as the dataset declares", "caller": "as the caller states"}
        if semantics is None and stated == "dataset":
            # A plan's setup before execution: the dataset is not read yet.
            return (
                "Monte Carlo draws from a hazard curve, with the probability "
                "semantics the dataset declares, not known yet"
            )
        return (
            f"Monte Carlo draws from a hazard curve with {semantics or 'unstated'} "
            f"semantics, {who.get(stated or '', 'stated by no one recorded')}"
        )
    if kind == "annual":
        return "per-year values, one row per recorded year"
    if kind == "table":
        return "rows of a user table, each a year as the caller states"
    return "a source of unknown kind"


def _basis_risk(sample: Mapping[str, Any], where: str) -> str:
    if "basis_risk" not in sample:
        return "not stated"
    model = sample["basis_risk"]
    if model is None:
        return "not modelled; the index and the loss are used as given"
    if not isinstance(model, Mapping):
        raise EvaluationError(f"{where}.basis_risk must be an object")
    kind = model.get("kind")
    if kind == "no_basis_risk":
        return "no model"
    if kind == "impact_noise":
        sigma = _number(model.get("sigma"), f"{where}.basis_risk.sigma")
        text = f"impact noise, a mean-one lognormal factor with sigma {sigma}"
        if model.get("upper") is not None:
            upper = _number(model["upper"], f"{where}.basis_risk.upper")
            text += f", the noisy impact capped at {upper}"
        return text
    terms = [
        f"{key} {json.dumps(model[key], sort_keys=True)}"
        for key in sorted(model)
        if key != "kind"
    ]
    return str(kind) + (f" ({', '.join(terms)})" if terms else "")


def _loss(sample: Mapping[str, Any], where: str) -> str:
    unit = _get(sample, "loss_unit", str, where) or "an unstated unit"
    if "loss_model" not in sample:
        return f"model not stated, in {unit}"
    model = sample["loss_model"]
    if model is None:
        return f"from the table, in {unit}"
    if not isinstance(model, Mapping):
        raise EvaluationError(f"{where}.loss_model must be an object")
    impact = _get(model, "impact", Mapping, f"{where}.loss_model") or {}
    value = _get(model, "value", Mapping, f"{where}.loss_model") or {}
    kind = {
        "expression": "impact expression",
        "function": "impact function",
        "callable": "callable impact",
    }.get(str(impact.get("kind")), "impact")
    text = f"{kind} {impact.get('name')}"
    if impact.get("definition_hash"):
        text += f" ({_short(impact['definition_hash'])})"
    if impact.get("exposure_unit"):
        text += f" for hazard values in {impact['exposure_unit']}"
    if value.get("source") == "column":
        text += f" times the asset value in column {value.get('column')}"
    elif value.get("source") == "constant":
        text += " times a stated asset value"
    else:
        text += ", the damage ratio"
    return f"{text}, in {unit}"


def _sample_text(sample: Mapping[str, Any], where: str) -> str:
    return (
        f"{_source(sample, where)}. Basis risk: {_basis_risk(sample, where)}. "
        f"Loss: {_loss(sample, where)}."
    )


def _pricing(pricing: Any, where: str) -> str:
    if pricing is None:
        return "not stated"
    if not isinstance(pricing, Mapping):
        raise EvaluationError(f"{where}.pricing must be an object")
    if pricing.get("kind") == "pure":
        return "the pure premium, P = E[payout]"
    if pricing.get("kind") == "cost_of_capital":
        rate = _number(pricing.get("rate"), f"{where}.pricing.rate")
        q = _number(pricing.get("q"), f"{where}.pricing.q")
        return (
            f"a cost-of-capital load, P = (1 - {rate}) E[payout] + {rate} "
            f"CVaR_{q}(payout)"
        )
    return str(pricing.get("kind"))


def _split(
    evaluation: Mapping[str, Any],
    samples: Mapping[str, Mapping[str, Any]],
    where: str,
) -> str:
    split = _get(evaluation, "split", str, where)
    fold = _get(evaluation, "fold", int, where)
    method = _get(evaluation, "method", str, where)
    details = _get(evaluation, "details", Mapping, where) or {}
    folded = "fold not stated" if fold is None else f"fold {fold}"
    sample_hash = evaluation.get("sample_hash")
    evaluated = (
        "the evaluation sample, not known yet"
        if sample_hash is None
        else _years(samples.get(str(sample_hash)), where)
    )
    if split == "in_sample":
        return (
            f"Split: in sample, {folded}: evaluated on the design sample itself, "
            f"{evaluated}. In-sample numbers show the fit, not out-of-sample "
            "performance."
        )
    if split != "out_of_sample":
        return f"Split: {split or 'not stated'}, {folded}: evaluated on {evaluated}."
    if method == "holdout":
        fraction = details.get("train_fraction")
        how = f"{folded}, a holdout" + (
            ""
            if fraction is None
            else f" with train fraction {_number(fraction, f'{where}.details')}"
        )
    elif method == "kfold":
        k = _get(details, "k", int, f"{where}.details")
        how = f"{folded} of a {k or 'k'}-fold split over blocks of years"
    elif method == "fresh_draws":
        how = f"{folded}, on fresh Monte Carlo draws"
    elif method == "separate_sample":
        how = f"{folded}, on a separate sample from another source"
    elif method == "disjoint_periods":
        how = f"{folded}, on periods of the same source the design sample lacks"
    else:
        how = f"{folded}, split method not stated"
    designs = [str(value) for value in _list(evaluation, "design_samples", where)]
    designed = _join([_years(samples.get(value), where) for value in designs])
    if "design_samples" not in evaluation:
        designed = "the design sample, not known yet"
    checked = (
        "the leakage check passed: the two share no period of one source"
        if evaluation.get("leakage_check") == "passed"
        else "no leakage check is recorded"
    )
    return (
        f"Split: out of sample, {how}: evaluated on {evaluated}; designed on "
        f"{designed or 'no design sample'}; {checked}."
    )


def _objective(objective: Mapping[str, Any], where: str) -> str:
    kind = objective.get("kind")
    if kind is None:
        return "not stated"
    if kind == "mse":
        return "MSE, E[(L - payout)^2]"
    if kind == "expected_uncovered":
        return "E[U], the expected uncovered loss"
    if kind == "weighted_cvar":
        alpha = _number(objective.get("alpha"), f"{where}.objective.alpha")
        beta = _number(objective.get("beta"), f"{where}.objective.beta")
        return (
            "weighted CVaR, alpha CVaR_beta(U) + (1 - alpha) E[U] with "
            f"alpha {alpha} and beta {beta}"
        )
    if kind == "user":
        parameters = objective.get("parameters") or {}
        return f"the user objective {objective.get('name')}" + (
            f" ({json.dumps(parameters, sort_keys=True)})" if parameters else ""
        )
    return str(kind)


def _design_labels(
    designs: tuple[dict[str, Any], ...], covers: tuple[CoverEntry, ...]
) -> dict[str, str | None]:
    """Each design's label, by `design_id`.

    A design record holds the label of its cover as evaluated
    (`cover_label`). Two designs can resolve to the same cover, for example
    under a binding budget, so a label is never looked up by the cover hash
    alone: for a record without `cover_label`, the cover's label is used
    only when the metadata gives that hash one label.
    """
    by_hash: dict[str, set[str | None]] = {}
    for entry in covers:
        by_hash.setdefault(entry.hash, set()).add(entry.label)
    labels: dict[str, str | None] = {}
    for record in designs:
        design_id = str(record["design_id"])
        if "cover_label" in record:
            where = f"designs[{_short(design_id)}]"
            labels[design_id] = _get(record, "cover_label", str, where)
            continue
        found = by_hash.get(str(record.get("cover_hash")), set())
        labels[design_id] = next(iter(found)) if len(found) == 1 else None
    return labels


def _design(
    record: Mapping[str, Any], labels: Mapping[str, str | None], unit: str
) -> str:
    design_id = _short(record["design_id"])
    where = f"designs[{design_id}]"
    label = labels.get(str(record["design_id"]))
    objective = _get(record, "objective", Mapping, where) or {}
    parts = [
        f"objective: {_objective(objective, where)}",
        f"pricing: {_pricing(record.get('pricing'), where)}",
    ]
    budget = _get(record, "budget", Mapping, where) or {}
    if budget.get("ratio") is not None:
        ratio = _number(budget["ratio"], f"{where}.budget.ratio")
        parts.append(f"budget: premium at most {ratio} x E[L] of the design sample")
    elif budget.get("amount") is not None:
        amount = _number(budget["amount"], f"{where}.budget.amount")
        parts.append(f"budget: premium at most {amount} {unit}")
    else:
        parts.append("no budget")
    limit = _get(record, "limit", Mapping, where) or {}
    if limit.get("amount") is not None:
        parts.append(f"limit: {_number(limit['amount'], f'{where}.limit')} {unit}")
    else:
        parts.append("no limit")
    if record.get("premium") is not None:
        premium = _number(record["premium"], f"{where}.premium")
        parts.append(f"design premium: {premium} {unit}")
    if record.get("route") is not None:
        parts.append(f"route: {record['route']}")
    unique = _get(record, "unique", bool, where)
    answer = "not checked" if unique is None else "yes" if unique else "no"
    parts.append(f"unique optimum: {answer}")
    ties = _get(record, "ties", int, where)
    parts.append("ties: not counted" if ties is None else f"ties: {ties}")
    unidentified = [str(name) for name in _list(record, "unidentified", where)]
    if unidentified:
        parts.append(
            f"amounts no design year reaches: {_join(unidentified)}, set to the "
            f"{record.get('fill') or 'lowest'} value allowed"
        )
    flag = _get(record, "degenerate_tail", str, where)
    if flag == "expected_uncovered":
        parts.append(
            "degenerate tail: beta is at most the probability of a zero loss, so "
            "the objective is a multiple of E[U]"
        )
    elif flag == "single_outcome":
        parts.append("degenerate tail: the tail holds at most one draw")
    elif flag is not None:
        parts.append(f"degenerate tail: {flag}")
    elif objective.get("kind") == "weighted_cvar":
        parts.append("tail: not degenerate")
    name = f"{label} ({design_id})" if label else design_id
    return f"Design {name}: {'; '.join(parts)}."


def _metrics(options: Mapping[str, Any], unit: str) -> str:
    def stated(key: str, suffix: str) -> str:
        where = f"setup.metric_options.{key}"
        if options.get(key) is None:
            return "not stated"
        if key == "payout_threshold":
            return f"{_number(options[key], where)}{suffix}"
        return f"{_numbers(options[key], where)}{suffix}"

    return (
        f"Metrics: beta {stated('betas', '')}; return periods "
        f"{stated('return_periods', ' years')}; loss thresholds "
        f"{stated('loss_thresholds', f' {unit}')}; payout threshold "
        f"{stated('payout_threshold', f' {unit}')}."
    )


def setup_text(
    metadata: EvaluationMetadata | EvaluationSetup | Mapping[str, Any],
) -> str:
    """Return the setup behind a comparison's numbers, as plain text.

    Each line is one statement, for a notebook or a report to print next to
    the numbers it describes:

    - `Level`, `Scenario`, `Basis`: one row per asset, the assets, the
      scenario, and the annual basis (one draw is one year);
    - `Split`, one line per evaluation: in or out of sample, the fold, how
      the split was made (a holdout and its train fraction, k-fold and k,
      fresh draws, a separate sample), the evaluation and design samples
      (years, periods, blocks) and the leakage check;
    - `Sample`: the source kind, the probability semantics and who stated
      them, the basis-risk model and its parameters, and the loss model;
    - `Sample notes`: what the sources recorded, such as draws beyond the
      fitted support, a pooled ensemble, dropped years, an index read from
      another series, or a comonotone index and loss;
    - `Pricing`: the pricing of the covers given as is; a design keeps its
      design premium, and a row's `premium_basis` says which premium it uses;
    - `Design`, one line per design: the objective, pricing, budget and
      limit, the design premium, the route, the non-uniqueness report
      (`unique`, `ties`, amounts no design year reaches) and a degenerate
      tail;
    - `Metrics`: the levels beta, the return periods and the loss and payout
      thresholds.

    Statements that every evaluation or sample shares are written once.

    `metadata` is a comparison's `EvaluationMetadata`, for example
    `EvaluationMetadata.from_table(table)`, or a setup alone: an
    `EvaluationSetup`, or a mapping in its layout, such as a setup known
    before anything is drawn. A setup alone has no design records, and keys
    it does not know yet (sample hashes, counts) may be missing: the text
    then says what is not known. The text depends only on `metadata`, so the
    same metadata gives the same text; nothing is read or written.

    Raises `EvaluationError` for setup values of the wrong type, and
    `TypeError` for an argument of none of these types.
    """
    designs: tuple[dict[str, Any], ...] = ()
    labels: dict[str, str | None] = {}
    if isinstance(metadata, EvaluationMetadata):
        setup: Mapping[str, Any] = metadata.setup.model_dump(mode="json")
        designs = metadata.designs
        labels = _design_labels(designs, metadata.covers)
    elif isinstance(metadata, EvaluationSetup):
        setup = metadata.model_dump(mode="json")
    elif isinstance(metadata, Mapping):
        setup = metadata
    else:
        raise TypeError(
            "setup_text takes EvaluationMetadata, an EvaluationSetup or a setup "
            f"mapping, got {type(metadata).__name__}"
        )
    samples = _objects(setup, "samples", "setup")
    evaluations = _objects(setup, "evaluations", "setup")
    by_hash = {
        str(sample["sample_hash"]): sample
        for sample in samples
        if sample.get("sample_hash") is not None
    }
    assets: list[str] = []
    scenarios: list[str] = []
    units: list[str] = []
    for i, sample in enumerate(samples):
        where = f"setup.samples[{i}]"
        assets += [str(asset) for asset in _list(sample, "asset_ids", where)]
        scenarios += [_get(sample, "scenario", str, where) or "not stated"]
        units += [_get(sample, "loss_unit", str, where) or "an unstated unit"]
    assets, scenarios, units = map(_first_seen, (assets, scenarios, units))
    unit = units[0] if len(units) == 1 else "loss units"

    if not assets:
        level = "asset, the assets not stated"
    elif len(assets) == 1:
        level = f"asset ({assets[0]})"
    else:
        level = f"asset, one row per asset ({_join(assets)}); no portfolio total"
    basis = setup.get("basis")
    if basis == "annual":
        basis = "annual, one draw is one year"
    lines = [
        f"Level: {level}. Scenario: {_join(scenarios) or 'not stated'}. "
        f"Basis: {basis or 'not stated'}."
    ]

    pricings = [
        _pricing(evaluation.get("pricing"), f"setup.evaluations[{i}]")
        for i, evaluation in enumerate(evaluations)
    ]
    shared = len(set(pricings)) <= 1
    for i, evaluation in enumerate(evaluations):
        text = _split(evaluation, by_hash, f"setup.evaluations[{i}]")
        if not shared:
            text += f" Pricing: {pricings[i]}."
        lines.append(text)
    if not evaluations:
        lines.append("Split: not known.")

    # Every sample is checked, also those the text describes only once.
    years = [_years(sample, f"setup.samples[{i}]") for i, sample in enumerate(samples)]
    descriptions = [
        _sample_text(sample, f"setup.samples[{i}]") for i, sample in enumerate(samples)
    ]
    if not samples:
        lines.append("Sample: not known.")
    elif len(set(descriptions)) == 1:
        lines.append(f"Sample: {descriptions[0]}")
    else:
        lines += [
            f"Sample of {span}: {text}"
            for span, text in zip(years, descriptions, strict=True)
        ]

    notes: list[str] = []
    for i, sample in enumerate(samples):
        values = _list(sample, "notes", f"setup.samples[{i}]")
        if not all(isinstance(note, str) for note in values):
            raise EvaluationError(f"setup.samples[{i}].notes must be a list of text")
        notes += values
    if notes:
        lines.append(f"Sample notes: {' '.join(_first_seen(notes))}")

    if not pricings:
        pricing = "not stated"
    else:
        pricing = pricings[0] if shared else "as each split states"
    lines.append(
        f"Pricing: {pricing}, on the evaluation sample, for the covers given as "
        "is; a design keeps its design premium, and premium_basis says which "
        "premium a row uses."
    )
    lines += [_design(record, labels, unit) for record in designs]
    options = setup.get("metric_options")
    if isinstance(options, Mapping):
        lines.append(_metrics(options, unit))
    else:
        lines.append("Metrics: the options are not stated.")
    return "\n".join(lines)
