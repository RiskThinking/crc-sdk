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
- `designs`: the record of every design evaluated;
- `sources`, `licences` and `rng`: the samples' source documents, their
  licences and attributions, and the generator state before every random
  step, each tagged with its sample's hash;
- `assets`: the number of assets; the id column and the hash of the
  (asset id, value) pairs when the caller knows them. Asset values are never
  written.

The module claims no licence of its own for a comparison: it is derived data
under its inputs' terms, which `licences` lists.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Self

from pydantic import ValidationError, model_validator

from crc_sdk.impacts._types import ContentHash, StrictFrozenModel

__all__ = [
    "EVALUATION_METADATA_KEY",
    "EVALUATION_SCHEMA_VERSION",
    "ColumnEntry",
    "CoverEntry",
    "EvaluationMetadata",
    "EvaluationSetup",
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


class ColumnEntry(StrictFrozenModel):
    """One metric column: its metric, parameter, unit and definition."""

    metric: str
    parameter: dict[str, float] | None
    unit: str
    definition: str


class CoverEntry(StrictFrozenModel):
    """One cover: its label, its `definition_hash` and its cover document."""

    label: str | None
    hash: ContentHash
    document: dict[str, Any]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _plain("a cover document", self.document)
        return self


class EvaluationSetup(StrictFrozenModel):
    """The setup of a comparison: basis, metric options, samples, evaluations.

    `samples` holds one document per sample (each with its `sample_hash`
    and `roles`, `"evaluation"` and/or `"design"`), and `evaluations` one per
    evaluation run (each with its `sample_hash`, `split` and `fold`).
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


class EvaluationMetadata(StrictFrozenModel):
    """The metadata of a comparison table, under `crc.parametric.evaluation`.

    `compare` and `evaluate` build it and attach it to their table;
    `from_table` reads it back, `merge` combines the metadata of several
    tables, and `write_comparison` checks it against the table before
    writing. Raises `ValueError` for values that are not plain, finite JSON.
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
        """Read the metadata from its JSON text. Raises `ValueError`."""
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"evaluation metadata is not valid JSON: {error}"
            ) from error
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            raise ValueError(f"invalid evaluation metadata: {error}") from error

    @classmethod
    def from_table(cls, table: Any) -> EvaluationMetadata:
        """Read the metadata a comparison table carries.

        Raises `ValueError` if the table's schema has no
        `crc.parametric.evaluation` key, or its value is invalid.
        """
        metadata = getattr(getattr(table, "schema", None), "metadata", None) or {}
        text = metadata.get(EVALUATION_METADATA_KEY.encode("utf-8"))
        if text is None:
            raise ValueError(
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

        Raises `ValueError` if the items disagree, and `TypeError` for an
        item that is not `EvaluationMetadata`.
        """
        if not items:
            raise ValueError("merge needs at least one EvaluationMetadata")
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
                raise ValueError(
                    f"the metadata differ in {field}: {sorted(map(str, values))}"
                )
        options = {_json_text(item.setup.metric_options) for item in items}
        if len(options) > 1:
            raise ValueError(
                "the metadata differ in their metric options, so their tables "
                "have different columns"
            )
        columns: dict[str, ColumnEntry] = {}
        for item in items:
            for name, entry in item.columns.items():
                if name in columns and columns[name] != entry:
                    raise ValueError(
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
                    raise ValueError(f"design {design_id!r} has two different records")
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
                raise ValueError("the metadata differ in their asset records")
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
        described here. Raises `ValueError` otherwise.
        """
        names = list(getattr(table, "column_names", ()))
        missing = [name for name in IDENTITY_COLUMNS if name not in names]
        if missing:
            raise ValueError(f"the comparison table lacks the columns {missing}")
        metrics = [name for name in names if name not in IDENTITY_COLUMNS]
        if set(metrics) != set(self.columns):
            unknown = sorted(set(metrics) - set(self.columns))
            absent = sorted(set(self.columns) - set(metrics))
            raise ValueError(
                "the table's metric columns differ from the metadata's: "
                f"undescribed {unknown}, absent {absent}"
            )
        rows = table.select(list(IDENTITY_COLUMNS)).to_pylist()
        keys = [tuple(row[name] for name in KEY_COLUMNS) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("the comparison table repeats a row key")
        covers = {entry.hash for entry in self.covers}
        designs = {record["design_id"] for record in self.designs}
        samples = {sample["sample_hash"] for sample in self.setup.samples}
        evaluations = {
            (entry["sample_hash"], entry["split"], entry["fold"])
            for entry in self.setup.evaluations
        }
        for row in rows:
            if row["cover_hash"] not in covers:
                raise ValueError(f"cover {row['cover_hash']} is not in the metadata")
            if row["design_id"] is not None and row["design_id"] not in designs:
                raise ValueError(f"design {row['design_id']} is not in the metadata")
            if row["sample_hash"] not in samples:
                raise ValueError(f"sample {row['sample_hash']} is not in the metadata")
            if (row["sample_hash"], row["split"], row["fold"]) not in evaluations:
                raise ValueError(
                    f"no evaluation of sample {row['sample_hash']} "
                    f"({row['split']}, fold {row['fold']}) is in the metadata"
                )
