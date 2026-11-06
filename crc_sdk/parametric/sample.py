"""Outcome samples: years with a probability, an index value and a loss each.

Every design and evaluation runs on an `OutcomeSample`. One row (a draw) is
one year: a cover pays once per draw, on that draw's index, against that
draw's loss. The rows are simulated years (`sample_curve`), observed or
modelled years (`sample_annual`), or the rows of a user table
(`OutcomeSample.from_table`).

- `period` labels the rows: the calendar year of a per-year sample, the draw
  number of a Monte Carlo sample. It is strictly increasing.
- `probability` is each row's probability; it sums to 1.
- `index` and `loss` have one row per draw and one column per asset. A
  sample holds exactly one asset for now, so they are `(n, 1)`.
- `hazard` is the value the loss model read, kept when it differs from the
  index (a basis-risk model or a separate index series); otherwise `None`.
- `provenance` (`SampleProvenance`) records the source, the semantics, the
  loss model, the basis-risk model and the generator state.

`holdout` and `year_block_kfold` split a sample into contiguous blocks of
periods, so the held-out block of a per-year sample is its latest years.
Monte Carlo draws are independent, so a fresh sample from another generator
state is the plainer out-of-sample test.

Everything here works on numpy arrays and Arrow tables and does no I/O:
read a Parquet file with `pyarrow.parquet.read_table` and pass the table.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel

from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.errors import ComonotoneIndexWarning, LeakageError, SampleError

__all__ = [
    "SAMPLE_METADATA_KEY",
    "SAMPLE_SCHEMA_VERSION",
    "SAMPLE_SEMANTICS_VERSION",
    "OutcomeSample",
    "SampleProvenance",
    "SemanticsSource",
    "SourceKind",
    "holdout",
    "year_block_kfold",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SAMPLE_METADATA_KEY = "crc.parametric.sample"
"""The Parquet schema metadata key of `OutcomeSample.to_table()`."""

SAMPLE_SCHEMA_VERSION = "1.0"
"""The `schema_version` of a sample's hash document and table metadata."""

SAMPLE_SEMANTICS_VERSION = "crc-parametric-sample/1"
"""The `semantics_version` of a sample's hash document and table metadata."""

SourceKind = Literal["monte_carlo", "annual", "table"]
"""Where a sample's rows come from: curve draws, per-year values or a table."""

SemanticsSource = Literal["dataset", "caller", "data"]
"""Who stated what a row means: the dataset, the caller, or the data itself."""

_PROBABILITY_TOLERANCE = 1e-12
_HOLDOUT_TOLERANCE = 1e-9
_IDENTIFIER = re.compile(r"^\S(?:[\s\S]*\S)?$")

COMONOTONE_MESSAGE = (
    "the index and the loss are comonotone: without a basis-risk model the "
    "loss is a function of the index, so a cover can track the loss almost "
    "exactly. State a basis-risk model (ImpactNoise, or an index series read "
    "elsewhere than the loss) to measure basis risk"
)


# -- plain JSON documents ----------------------------------------------------


def _freeze(value: Any) -> Any:
    """Return an immutable copy: mappings as read-only proxies, lists as tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return plain dicts and lists, the inverse of `_freeze`."""
    if isinstance(value, BaseModel):
        return _thaw(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _check_json(name: str, value: Any) -> None:
    """Raise `SampleError` unless `value` is a document of plain JSON types."""
    try:
        json.dumps(_thaw(value), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise SampleError(f"{name} must hold plain JSON values: {error}") from error


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SampleError(
            f"{name} must be a non-empty string without surrounding whitespace, "
            f"got {value!r}"
        )
    return value


def _source_document(source: Any) -> dict[str, Any]:
    """Return a caller's source description as a plain JSON document.

    A string is a name, a pydantic model (`SourceProvenance`) is dumped, and a
    mapping (for example a plan's `explain(format="json")`) is copied.
    """
    if source is None:
        return {}
    if isinstance(source, str):
        return {"name": _identifier("source", source)}
    if isinstance(source, BaseModel):
        document: dict[str, Any] = source.model_dump(mode="json")
    elif isinstance(source, Mapping):
        document = _thaw(source)
    else:
        raise TypeError(
            "source must be a name, a SourceProvenance or a mapping, "
            f"got {type(source).__name__}"
        )
    _check_json("source", document)
    return document


def _licence_entry(role: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the licence and attribution a source document declares."""

    def text(key: str) -> str | None:
        value = document.get(key)
        return None if value is None else str(value)

    name = document.get("dataset") or document.get("name")
    return {
        "role": role,
        "source": None if name is None else str(name),
        "licence": text("licence"),
        "attribution": text("attribution"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _generator_record(rng: np.random.Generator, step: str) -> dict[str, Any]:
    """Return the bit generator's name and its state, before `step` draws."""
    bit_generator = rng.bit_generator
    return {
        "step": step,
        "bit_generator": type(bit_generator).__name__,
        "state": json.dumps(bit_generator.state, sort_keys=True, default=_json_default),
    }


def _require_generator(rng: Any) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    return rng


# -- Arrow tables ------------------------------------------------------------


def _arrow_table(data: Any, what: str) -> Any:
    """Return `data` as a `pyarrow.Table`, without reading any file."""
    if isinstance(data, (str, bytes, os.PathLike)):
        raise TypeError(
            f"{what} must be an in-memory table; this module does no I/O, so "
            "read the file first (for example pyarrow.parquet.read_table)"
        )
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    if isinstance(data, pa.RecordBatchReader):
        return data.read_all()
    if isinstance(data, Mapping) or hasattr(data, "__arrow_c_stream__"):
        try:
            return pa.table(data)
        except (TypeError, ValueError, pa.ArrowException) as error:
            raise TypeError(f"{what} cannot be read as an Arrow table") from error
    if isinstance(data, Iterable):
        batches = list(data)
        if not batches or not all(isinstance(b, pa.RecordBatch) for b in batches):
            raise TypeError(f"{what} must be a table or Arrow record batches")
        return pa.Table.from_batches(batches)
    raise TypeError(f"{what} must be an Arrow table, got {type(data).__name__}")


def _column(table: Any, name: str, what: str) -> Any:
    if not isinstance(name, str) or name not in table.column_names:
        raise SampleError(
            f"{what} has no column {name!r}; its columns are {table.column_names}"
        )
    return table.column(name)


def _integer_values(column: Any, name: str) -> IntArray:
    if not (pa.types.is_integer(column.type)):
        raise SampleError(f"column {name!r} must hold integers, not {column.type}")
    if column.null_count:
        raise SampleError(f"column {name!r} has {column.null_count} null values")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def _float_values(column: Any, name: str) -> FloatArray:
    if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
        raise SampleError(f"column {name!r} must be numeric, not {column.type}")
    if column.null_count:
        raise SampleError(f"column {name!r} has {column.null_count} null values")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def _string_values(column: Any, name: str) -> list[str]:
    if pa.types.is_integer(column.type):
        column = column.cast(pa.string())
    elif pa.types.is_dictionary(column.type):
        column = column.cast(pa.string())
    if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
        raise SampleError(f"column {name!r} must hold strings, not {column.type}")
    if column.null_count:
        raise SampleError(f"column {name!r} has {column.null_count} null values")
    return [str(value) for value in column.to_pylist()]


# -- array validation --------------------------------------------------------


def _periods(value: Any) -> IntArray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise SampleError(f"period must hold integers, got dtype {array.dtype}")
    if array.ndim != 1 or array.shape[0] == 0:
        raise SampleError(f"period must be 1-D and non-empty, got shape {array.shape}")
    if array.dtype.kind == "u" and array.size and int(array.max()) > 2**63 - 1:
        raise SampleError("period values must fit in a signed 64-bit integer")
    periods = np.array(array, dtype=np.int64)
    if periods.shape[0] > 1 and not (np.diff(periods) > 0).all():
        raise SampleError("period must be strictly increasing")
    return periods


def _float_array(name: str, value: Any) -> FloatArray:
    try:
        return np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SampleError(f"{name} must be numeric") from error


def _matrix(name: str, value: Any, n: int, n_assets: int) -> FloatArray:
    array = _float_array(name, value)
    if array.ndim == 1 and n_assets == 1:
        array = array.reshape(-1, 1)
    if array.shape != (n, n_assets):
        raise SampleError(
            f"{name} must have shape ({n}, {n_assets}), one row per period and "
            f"one column per asset; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise SampleError(f"{name} must be finite")
    return array


def _read_only(array: NDArray[Any]) -> NDArray[Any]:
    array.setflags(write=False)
    return array


def _hash_document(
    *,
    asset_ids: Sequence[str],
    period: IntArray,
    probability: FloatArray,
    index: FloatArray,
    loss: FloatArray,
    hazard: FloatArray | None,
    index_unit: str,
    loss_unit: str,
    semantics: str,
    scenario: str,
) -> dict[str, Any]:
    """The document behind `content_hash`: the long table, sorted, and its units."""
    order = sorted(range(len(asset_ids)), key=lambda j: asset_ids[j].encode("utf-8"))
    n_assets = len(order)

    def flat(array: FloatArray) -> list[float]:
        values: list[float] = array[:, order].reshape(-1).tolist()
        return values

    return {
        "asset_id": [asset_ids[j] for j in order] * period.shape[0],
        "hazard": None if hazard is None else flat(hazard),
        "index": flat(index),
        "index_unit": index_unit,
        "loss": flat(loss),
        "loss_unit": loss_unit,
        "period": np.repeat(period, n_assets).tolist(),
        "probability": np.repeat(probability, n_assets).tolist(),
        "scenario": scenario,
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "semantics": semantics,
        "semantics_version": SAMPLE_SEMANTICS_VERSION,
    }


def _is_monotone_function(index: FloatArray, loss: FloatArray) -> bool:
    """Whether the loss is a non-constant monotone function of the index."""
    if loss.max() == loss.min():
        return False
    order = np.lexsort((loss, index))
    x, y = index[order], loss[order]
    tied = np.diff(x) == 0
    steps = np.diff(y)
    if (steps[tied] != 0).any():
        return False
    steps = steps[~tied]
    return bool((steps >= 0).all() or (steps <= 0).all())


# -- the provenance ----------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SampleProvenance:
    """Where a sample's rows come from, and how they were made.

    - `source_kind`: `"monte_carlo"` (draws from a curve), `"annual"`
      (per-year values) or `"table"` (a user table).
    - `sources`: one document per input, each with a `role` (`"hazard"`,
      `"index"` or `"table"`): for a curve, its dataset metadata, the curve's
      parameters and the caller's details (horizon, pathway, file checksum);
      for a per-year series, its provenance; for a table, the caller's name
      and the table's content hash.
    - `semantics_source`: who stated that a row is a year: `"dataset"` (the
      curve's metadata), `"caller"` (`assume_semantics=`, or a table's
      `semantics="annual"`) or `"data"` (per-year values are years).
    - `probability_semantics`: the curve's probability semantics, for Monte
      Carlo samples; otherwise `None`.
    - `loss_model` and `basis_risk`: their documents; `None` for a table.
    - `rng`: for every random step, in draw order, the bit generator's name
      and its state before the step, as JSON text. `generator()` rebuilds it.
    - `licences`: the licence and attribution each input declares.
    - `details`: what the source counted: draws outside the fitted support,
      draws in a hurdle atom, dropped years, the matched grid cell, ...
    - `notes`: plain sentences for the setup text next to a number.
    - `lineage`: identifies the source of the rows. Two samples with the same
      lineage and a shared period overlap; out-of-sample evaluation on them
      raises `LeakageError`.
    - `parent` and `selection`: for a sample split or selected from another,
      the content hash of the sample it came from and the steps taken.

    Mappings and sequences are stored read-only; `to_document()` returns
    plain JSON types. Raises `SampleError` for unknown kinds or values that
    are not plain JSON.
    """

    source_kind: SourceKind
    sources: tuple[Mapping[str, Any], ...]
    semantics_source: SemanticsSource
    probability_semantics: str | None = None
    loss_model: Mapping[str, Any] | None = None
    basis_risk: Mapping[str, Any] | None = None
    rng: tuple[Mapping[str, Any], ...] = ()
    licences: tuple[Mapping[str, Any], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    lineage: str | None = None
    parent: str | None = None
    selection: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.source_kind not in get_args(SourceKind):
            raise SampleError(
                f"source_kind must be one of {get_args(SourceKind)}, "
                f"got {self.source_kind!r}"
            )
        if self.semantics_source not in get_args(SemanticsSource):
            raise SampleError(
                f"semantics_source must be one of {get_args(SemanticsSource)}, "
                f"got {self.semantics_source!r}"
            )
        for name in ("probability_semantics", "lineage", "parent"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise SampleError(f"{name} must be a string or None, got {value!r}")
        for name in ("sources", "rng", "licences", "selection"):
            value = getattr(self, name)
            if isinstance(value, (str, Mapping)) or not isinstance(value, Sequence):
                raise SampleError(f"{name} must be a sequence of mappings")
            if not all(isinstance(item, Mapping) for item in value):
                raise SampleError(f"{name} must be a sequence of mappings")
        for record in self.rng:
            if not {"bit_generator", "state", "step"} <= set(record):
                raise SampleError(
                    "each rng record needs a step, a bit_generator and a state"
                )
        for name in ("loss_model", "basis_risk"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise SampleError(f"{name} must be a mapping or None")
        if not isinstance(self.details, Mapping):
            raise SampleError("details must be a mapping")
        if isinstance(self.notes, str) or not all(
            isinstance(note, str) for note in self.notes
        ):
            raise SampleError("notes must be a sequence of strings")
        for name in (
            "sources",
            "loss_model",
            "basis_risk",
            "rng",
            "licences",
            "details",
            "selection",
        ):
            value = getattr(self, name)
            _check_json(name, value)
            object.__setattr__(self, name, _freeze(_thaw(value)))
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_document(self) -> dict[str, Any]:
        """Return every field as plain JSON types."""
        return {
            item.name: _thaw(getattr(self, item.name))
            for item in dataclasses.fields(self)
        }

    def generator(self, step: int = 0) -> np.random.Generator:
        """Return a generator in the recorded state before random step `step`.

        Step 0 is the state at entry of the call that built the sample, so
        calling the same function again with this generator rebuilds the
        sample exactly. Raises `SampleError` if nothing was drawn or the bit
        generator is not one of numpy's.
        """
        if not self.rng:
            raise SampleError("the sample was built without a random generator")
        record = self.rng[step]
        name = record["bit_generator"]
        bit_type = getattr(np.random, str(name), None)
        if not (
            isinstance(bit_type, type) and issubclass(bit_type, np.random.BitGenerator)
        ):
            raise SampleError(f"cannot rebuild the bit generator {name!r}")
        bit_generator = bit_type()
        bit_generator.state = json.loads(record["state"])
        return np.random.Generator(bit_generator)

    def _selected(
        self, *, parent: str, lineage: str, step: Mapping[str, Any]
    ) -> SampleProvenance:
        return dataclasses.replace(
            self,
            parent=parent,
            lineage=lineage,
            selection=(*self.selection, step),
        )


# -- the sample --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class OutcomeSample:
    """Years (or simulated years) with a probability, an index and a loss each.

    One draw is one year: a cover pays once per draw, on its index, against
    its loss. `period` is the year label or the draw number, strictly
    increasing. `probability` is positive and sums to 1 within 1e-12.
    `index` and `loss` have shape `(n, n_assets)`, one column per asset in
    `asset_ids` order; a 1-D array is read as one column. They are finite,
    and the loss is non-negative. `hazard` is the value the loss model read,
    when it differs from the index, else `None`. `index_unit` and
    `loss_unit` name the units; the loss unit must equal a cover's payout
    unit for designs and evaluations. `semantics` is `"annual"`.

    A sample holds exactly one asset for now; a second raises `SampleError`.
    The arrays are read-only copies. Samples compare by identity; compare
    `content_hash` for equal contents.

    Raises `SampleError` for any invalid field.
    """

    asset_ids: tuple[str, ...]
    period: IntArray
    probability: FloatArray
    index: FloatArray
    loss: FloatArray
    hazard: FloatArray | None
    index_unit: str
    loss_unit: str
    semantics: Literal["annual"]
    provenance: SampleProvenance
    scenario: str = "baseline"

    def __post_init__(self) -> None:
        ids = self.asset_ids
        if isinstance(ids, str) or not isinstance(ids, Sequence):
            raise SampleError("asset_ids must be a sequence of asset ids")
        asset_ids = tuple(_identifier("asset id", value) for value in ids)
        if len(asset_ids) != 1:
            raise SampleError(
                f"a sample holds exactly one asset, got {len(asset_ids)}; build "
                "one sample per asset"
            )
        period = _periods(self.period)
        n = period.shape[0]
        probability = _float_array("probability", self.probability)
        if probability.shape != (n,):
            raise SampleError(
                f"probability must have shape ({n},), got {probability.shape}"
            )
        if not np.isfinite(probability).all() or (probability <= 0).any():
            raise SampleError("probability must be finite and positive")
        total = math.fsum(probability.tolist())
        if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
            raise SampleError(
                f"probability must sum to 1 within {_PROBABILITY_TOLERANCE}, "
                f"got {total!r}"
            )
        index = _matrix("index", self.index, n, len(asset_ids))
        loss = _matrix("loss", self.loss, n, len(asset_ids))
        if (loss < 0).any():
            raise SampleError("loss must be non-negative")
        hazard = (
            None
            if self.hazard is None
            else _matrix("hazard", self.hazard, n, len(asset_ids))
        )
        _identifier("index_unit", self.index_unit)
        _identifier("loss_unit", self.loss_unit)
        if self.semantics != "annual":
            raise SampleError(
                f"semantics must be 'annual' (one draw is one year), got "
                f"{self.semantics!r}; 'event' is reserved"
            )
        if not isinstance(self.provenance, SampleProvenance):
            raise SampleError("provenance must be a SampleProvenance")
        _identifier("scenario", self.scenario)
        object.__setattr__(self, "asset_ids", asset_ids)
        object.__setattr__(self, "period", _read_only(period))
        object.__setattr__(self, "probability", _read_only(probability))
        object.__setattr__(self, "index", _read_only(index))
        object.__setattr__(self, "loss", _read_only(loss))
        object.__setattr__(
            self, "hazard", None if hazard is None else _read_only(hazard)
        )

    def __len__(self) -> int:
        return int(self.period.shape[0])

    @cached_property
    def content_hash(self) -> str:
        """`"sha256:<hex>"` of the sample's canonical long table and units.

        The document holds the long table (`to_table` without the scenario
        column), sorted by period and asset id, with the units, the
        semantics and the scenario, in the canonical JSON of cover documents.
        The provenance is not part of it.
        """
        return content_hash(
            _hash_document(
                asset_ids=self.asset_ids,
                period=self.period,
                probability=self.probability,
                index=self.index,
                loss=self.loss,
                hazard=self.hazard,
                index_unit=self.index_unit,
                loss_unit=self.loss_unit,
                semantics=self.semantics,
                scenario=self.scenario,
            )
        )

    @property
    def lineage(self) -> str:
        """The identity of the rows' source, for leakage checks.

        The provenance's `lineage`, else the content hash of the sample this
        one was selected from, else this sample's own content hash.
        """
        return self.provenance.lineage or self.provenance.parent or self.content_hash

    def select(self, periods: ArrayLike) -> OutcomeSample:
        """Return the sample of the given periods, in period order.

        The probabilities are rescaled to sum to 1 on the selection. Raises
        `SampleError` for an empty selection, repeated periods, or periods
        the sample does not hold.
        """
        wanted = np.asarray(periods)
        if wanted.dtype.kind not in "iu" or wanted.ndim != 1 or wanted.size == 0:
            raise SampleError("periods must be a non-empty 1-D sequence of integers")
        wanted = wanted.astype(np.int64)
        if np.unique(wanted).size != wanted.size:
            raise SampleError("periods must not repeat")
        unknown = np.setdiff1d(wanted, self.period)
        if unknown.size:
            raise SampleError(
                f"the sample holds no periods {unknown[:10].tolist()}"
                + (" ..." if unknown.size > 10 else "")
            )
        rows = np.isin(self.period, wanted)
        return self._subset(rows, {"kind": "select", "periods": int(rows.sum())})

    def _subset(
        self, rows: NDArray[np.bool_], step: Mapping[str, Any]
    ) -> OutcomeSample:
        probability = self.probability[rows]
        probability = probability / math.fsum(probability.tolist())
        provenance = self.provenance._selected(
            parent=self.provenance.parent or self.content_hash,
            lineage=self.lineage,
            step=step,
        )
        return OutcomeSample(
            asset_ids=self.asset_ids,
            period=self.period[rows],
            probability=probability,
            index=self.index[rows],
            loss=self.loss[rows],
            hazard=None if self.hazard is None else self.hazard[rows],
            index_unit=self.index_unit,
            loss_unit=self.loss_unit,
            semantics=self.semantics,
            provenance=provenance,
            scenario=self.scenario,
        )

    def to_table(self) -> Any:
        """Return the long table: one row per period and asset.

        Columns `scenario`, `period`, `asset_id`, `probability`, `index`,
        `loss` and `hazard` (null when the hazard is the index), sorted by
        period and asset id. The schema metadata holds, under
        `SAMPLE_METADATA_KEY` (`crc.parametric.sample`), a JSON document with
        the units, the semantics, the scenario, the asset ids, the content
        hash and the provenance, so `pyarrow.parquet.write_table` writes a
        self-describing file.
        """
        document = _hash_document(
            asset_ids=self.asset_ids,
            period=self.period,
            probability=self.probability,
            index=self.index,
            loss=self.loss,
            hazard=self.hazard,
            index_unit=self.index_unit,
            loss_unit=self.loss_unit,
            semantics=self.semantics,
            scenario=self.scenario,
        )
        rows = len(document["period"])
        metadata = {
            "asset_ids": list(self.asset_ids),
            "content_hash": self.content_hash,
            "index_unit": self.index_unit,
            "loss_unit": self.loss_unit,
            "provenance": self.provenance.to_document(),
            "scenario": self.scenario,
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "semantics": self.semantics,
            "semantics_version": SAMPLE_SEMANTICS_VERSION,
        }
        hazard = (
            pa.nulls(rows, pa.float64())
            if document["hazard"] is None
            else pa.array(document["hazard"], pa.float64())
        )
        schema = pa.schema(
            [
                ("scenario", pa.string()),
                ("period", pa.int64()),
                ("asset_id", pa.string()),
                ("probability", pa.float64()),
                ("index", pa.float64()),
                ("loss", pa.float64()),
                ("hazard", pa.float64()),
            ],
            metadata={
                SAMPLE_METADATA_KEY: json.dumps(
                    metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            },
        )
        return pa.Table.from_arrays(
            [
                pa.array([self.scenario] * rows, pa.string()),
                pa.array(document["period"], pa.int64()),
                pa.array(document["asset_id"], pa.string()),
                pa.array(document["probability"], pa.float64()),
                pa.array(document["index"], pa.float64()),
                pa.array(document["loss"], pa.float64()),
                hazard,
            ],
            schema=schema,
        )

    @classmethod
    def from_table(
        cls,
        table: Any,
        *,
        period: str,
        asset_id: str,
        index: str,
        loss: str,
        probability: str | None = None,
        hazard: str | None = None,
        semantics: Literal["annual"],
        index_unit: str,
        loss_unit: str,
        source: Any = None,
    ) -> OutcomeSample:
        """Build a sample from a user table, one row per period and asset.

        `period`, `asset_id`, `index` and `loss` name the required columns;
        `probability` and `hazard` name optional ones. Without a probability
        column every row gets `1/n`. `semantics="annual"` is required: the
        caller states that one row is one year. `table` is an Arrow table or
        record batch, or any object Arrow reads (a mapping of columns, a
        pandas or polars frame); to use a Parquet file, read it with
        `pyarrow.parquet.read_table`. `source` names the table for the
        provenance: a string, a `SourceProvenance` (with its licence) or a
        mapping.

        Rows are sorted by period. Every (period, asset) pair must appear
        once, and the table must hold exactly one asset. The table carries
        its own basis risk; if its loss is a monotone function of its index,
        this warns with `ComonotoneIndexWarning`.

        Raises `SampleError` for a missing column, nulls, non-integer
        periods, repeated periods, several assets, or invalid values, and
        `TypeError` for a file path or a non-table.
        """
        if semantics != "annual":
            raise SampleError(
                f"semantics must be 'annual' (one row is one year), got "
                f"{semantics!r}; 'event' is reserved"
            )
        data = _arrow_table(table, "a sample table")
        what = "the sample table"
        periods = _integer_values(_column(data, period, what), period)
        ids = _string_values(_column(data, asset_id, what), asset_id)
        index_values = _float_values(_column(data, index, what), index)
        loss_values = _float_values(_column(data, loss, what), loss)
        probability_values = (
            None
            if probability is None
            else _float_values(_column(data, probability, what), probability)
        )
        hazard_values = (
            None
            if hazard is None
            else _float_values(_column(data, hazard, what), hazard)
        )
        if not ids:
            raise SampleError("the sample table has no rows")
        distinct = sorted(set(ids))
        if len(distinct) != 1:
            raise SampleError(
                f"a sample holds exactly one asset, but the table holds "
                f"{len(distinct)} ({', '.join(distinct[:5])}"
                f"{', ...' if len(distinct) > 5 else ''}); filter it to one asset"
            )
        order = np.argsort(periods, kind="stable")
        periods = periods[order]
        repeated = np.unique(periods[1:][np.diff(periods) == 0])
        if repeated.size:
            raise SampleError(
                f"periods {repeated[:10].tolist()} appear more than once for "
                f"asset {distinct[0]!r}"
            )
        n = periods.shape[0]
        index_values = index_values[order]
        loss_values = loss_values[order]
        if probability_values is None:
            probability_values = np.full(n, 1.0 / n)
        else:
            probability_values = probability_values[order]
        if hazard_values is not None:
            hazard_values = hazard_values[order]
        source_document = _source_document(source)
        # Validate every field first; the provenance needs the content hash.
        bare = cls(
            asset_ids=(distinct[0],),
            period=periods,
            probability=probability_values,
            index=index_values,
            loss=loss_values,
            hazard=hazard_values,
            index_unit=index_unit,
            loss_unit=loss_unit,
            semantics=semantics,
            provenance=SampleProvenance(
                source_kind="table", sources=(), semantics_source="caller"
            ),
        )
        table_hash = bare.content_hash
        notes: list[str] = []
        if _is_monotone_function(index_values, loss_values):
            warnings.warn(COMONOTONE_MESSAGE, ComonotoneIndexWarning, stacklevel=2)
            notes.append(
                "The table's loss is a monotone function of its index: index "
                "and loss are comonotone."
            )
        identity: dict[str, Any] = {"kind": "table", "asset_id": distinct[0]}
        if source_document:
            identity["source"] = source_document
        else:
            identity["data"] = table_hash
        provenance = SampleProvenance(
            source_kind="table",
            sources=(
                {
                    "role": "table",
                    "source": source_document,
                    "columns": {
                        "period": period,
                        "asset_id": asset_id,
                        "index": index,
                        "loss": loss,
                        "probability": probability,
                        "hazard": hazard,
                    },
                    "content_hash": table_hash,
                },
            ),
            semantics_source="caller",
            licences=(_licence_entry("table", source_document),),
            details={
                "draws": n,
                "periods": [int(periods[0]), int(periods[-1])],
                "probability": "uniform" if probability is None else "column",
            },
            notes=tuple(notes),
            lineage=content_hash(identity),
        )
        return dataclasses.replace(bare, provenance=provenance)


# -- splits ------------------------------------------------------------------


def _require_sample(sample: Any) -> OutcomeSample:
    if not isinstance(sample, OutcomeSample):
        raise TypeError(f"expected an OutcomeSample, got {type(sample).__name__}")
    return sample


def holdout(
    sample: OutcomeSample, train_fraction: float = 0.7
) -> tuple[OutcomeSample, OutcomeSample]:
    """Split a sample into a design block and a held-out block.

    The design block is the first `floor(train_fraction * n + 1e-9)` periods
    in period order (the tolerance makes `0.7 * 10` give 7), the held-out
    block the rest. For per-year samples the held-out block is therefore the
    latest years. Each block's probabilities are rescaled to sum to 1. Both
    blocks share the sample's lineage, so evaluating a design on a block
    that overlaps its design block raises `LeakageError`.

    Raises `SampleError` unless `train_fraction` is a number in (0, 1) that
    leaves at least one period in each block.
    """
    sample = _require_sample(sample)
    if (
        isinstance(train_fraction, bool)
        or not isinstance(train_fraction, Real)
        or not math.isfinite(float(train_fraction))
        or not 0.0 < float(train_fraction) < 1.0
    ):
        raise SampleError(
            f"train_fraction must be a number in (0, 1), got {train_fraction!r}"
        )
    fraction = float(train_fraction)
    n = len(sample)
    n_train = math.floor(fraction * n + _HOLDOUT_TOLERANCE)
    if not 1 <= n_train <= n - 1:
        raise SampleError(
            f"train_fraction {fraction} of {n} periods gives {n_train} design "
            "periods; both blocks need at least one period"
        )
    design_rows = np.arange(n) < n_train
    step = {"kind": "holdout", "train_fraction": fraction, "fold": 0}
    return (
        sample._subset(design_rows, {**step, "block": "train"}),
        sample._subset(~design_rows, {**step, "block": "test"}),
    )


def year_block_kfold(
    sample: OutcomeSample, k: int
) -> tuple[tuple[OutcomeSample, OutcomeSample], ...]:
    """k-fold cross-validation over contiguous blocks of periods.

    The periods, in order, are split into `k` contiguous blocks with
    `numpy.array_split` (the first `n % k` blocks get one period more). Fold
    `i` holds out block `i` and designs on the others. Returns one
    `(design, held_out)` pair per fold, in fold order; each sample's
    probabilities are rescaled to sum to 1.

    Raises `SampleError` unless `k` is an integer with `2 <= k <= n`.
    """
    sample = _require_sample(sample)
    n = len(sample)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 2 <= k <= n:
        raise SampleError(f"k must be an integer in [2, {n}], got {k!r}")
    folds: list[tuple[OutcomeSample, OutcomeSample]] = []
    for fold, block in enumerate(np.array_split(np.arange(n), int(k))):
        held_out = np.zeros(n, dtype=bool)
        held_out[block] = True
        step = {"kind": "kfold", "k": int(k), "fold": fold}
        folds.append(
            (
                sample._subset(~held_out, {**step, "block": "train"}),
                sample._subset(held_out, {**step, "block": "test"}),
            )
        )
    return tuple(folds)


def _check_out_of_sample(design: OutcomeSample, evaluation: OutcomeSample) -> None:
    """Raise `LeakageError` if `evaluation` shares rows with `design`.

    Two samples share rows when they have the same lineage (the same source,
    or splits of one sample) and a common period.
    """
    design = _require_sample(design)
    evaluation = _require_sample(evaluation)
    if design.lineage != evaluation.lineage:
        return
    shared = np.intersect1d(design.period, evaluation.period)
    if shared.size:
        raise LeakageError(
            f"the evaluation sample shares {shared.size} periods with the design "
            f"sample (e.g. {shared[:5].tolist()}) from the same source; evaluate "
            "out of sample on a held-out block or on fresh draws"
        )
