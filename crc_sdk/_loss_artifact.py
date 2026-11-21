"""Insured annual loss results, paired comparisons, and Parquet storage."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import cast

import crc_framework
import numpy
import numpy.typing
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from crc_framework import AnnualLossSpec, AnnualLossSummary, CorrelationRepair

from ._dataset_schema import (
    ExecutionDiagnostic,
    ExecutionReport,
    Stage,
    _array,
    _digest,
    _dump,
    _identifier,
    _integer,
    _load,
    _number,
    _object,
    _report,
    _source,
    _transformation,
)
from ._declarations import (
    ProbabilityKind,
    ScientificValue,
    _freeze,
    _period,
    _plain,
    _probability,
    _scenario,
    _variable,
)
from ._errors import ValidationError
from ._inspection import SeriesReference
from ._loss_inputs import _LossInputs
from ._runtime import LocalPath, _local_path
from ._terminals import _publication_path

ANNUAL_LOSS_VERSION = "1.0"
_DOCUMENT = b"crc.annual_loss"
_DOCUMENT_DIGEST = b"crc.annual_loss.sha256"
_FOOTER_LIMIT = 8 * 1024 * 1024
_U64_MAX = (1 << 64) - 1
_STREAM_LAYOUT = "shared-primary-independent-secondary"
_TOTAL_FIELDS = (
    pa.field("trial_id", pa.int64(), nullable=False),
    pa.field("insured_loss", pa.float64(), nullable=False),
)
_COMPONENT_FIELDS = (
    pa.field(
        "component_ids", pa.list_(pa.field("element", pa.string(), nullable=False))
    ),
    pa.field(
        "component_losses",
        pa.list_(pa.field("element", pa.float64(), nullable=False)),
        nullable=False,
    ),
)
FloatArray = numpy.typing.NDArray[numpy.float64]


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _spec_document(spec: AnnualLossSpec) -> dict[str, object]:
    return {
        "trials": spec.trials,
        "seed": spec.seed,
        "peak_correlation": spec.peak_correlation,
        "decay_length_km": spec.decay_length_km,
        "damage_cv": spec.damage_cv,
        "repair_tolerance": spec.repair_tolerance,
        "calibration_source": spec.calibration_source,
    }


def _spec(value: object) -> AnnualLossSpec:
    document = _object(
        value,
        {
            "trials",
            "seed",
            "peak_correlation",
            "decay_length_km",
            "damage_cv",
            "repair_tolerance",
            "calibration_source",
        },
    )
    tolerance = document["repair_tolerance"]
    source = document["calibration_source"]
    return AnnualLossSpec(
        trials=_integer(document["trials"], "trials", 1),
        seed=_integer(document["seed"], "seed"),
        peak_correlation=_number(document["peak_correlation"], "peak_correlation"),
        decay_length_km=_number(document["decay_length_km"], "decay_length_km"),
        damage_cv=_number(document["damage_cv"], "damage_cv"),
        repair_tolerance=None
        if tolerance is None
        else _number(tolerance, "repair_tolerance"),
        calibration_source=None
        if source is None
        else _identifier(source, "calibration_source"),
    )


def _correlation_repair(value: object) -> CorrelationRepair | None:
    if value is None:
        return None
    document = _object(value, {"minimum_eigenvalue", "maximum_change"})
    change = _number(document["maximum_change"], "maximum_change")
    if change < 0:
        raise ValueError("correlation repair maximum_change must be nonnegative")
    return CorrelationRepair(
        minimum_eigenvalue=_number(
            document["minimum_eigenvalue"], "minimum_eigenvalue"
        ),
        maximum_change=change,
    )


def _series_reference(document: object, member_id: object) -> SeriesReference:
    series = _object(document, {"variable", "scenario", "period", "probability"})
    return SeriesReference(
        _variable(series["variable"]),
        scenario=None if series["scenario"] is None else _scenario(series["scenario"]),
        period=None if series["period"] is None else _period(series["period"]),
        probability=_probability(series["probability"]),
        member=None if member_id is None else _identifier(member_id, "member_id"),
    )


def _validate_selection(document: dict[str, object]) -> int:
    selection = _series_reference(document["series"], document["member_id"])
    if (
        selection.variable.id,
        None if selection.scenario is None else selection.scenario.id,
        None if selection.period is None else selection.period.id,
    ) != tuple(document[key] for key in ("variable_id", "scenario_id", "period_id")):
        raise ValueError(
            "Annual loss series identities do not match their declarations"
        )
    if (
        selection.probability is None
        or selection.probability.kind is not ProbabilityKind.ANNUAL
    ):
        raise ValueError("Annual loss evidence must declare annual probabilities")
    grid = [
        _number(q, "probability rank")
        for q in _array(document["probability_grid"], "probability_grid")
    ]
    if (
        len(grid) < 2
        or not 0 <= grid[0] < grid[-1] <= 1
        or any(a >= b for a, b in zip(grid, grid[1:], strict=False))
    ):
        raise ValueError("Annual loss evidence requires increasing ranks in [0, 1]")
    completion = document["tail_completion"]
    if completion is None:
        if (grid[0], grid[-1]) != (0, 1):
            raise ValueError(
                "Incomplete annual probability evidence requires tail completion"
            )
    else:
        endpoints = _object(completion, {"name", "lower", "upper"})
        lower, upper = (_number(endpoints[key], key) for key in ("lower", "upper"))
        if endpoints["name"] != "finite_endpoints" or not 0 <= lower <= upper <= 1:
            raise ValueError(
                "Annual loss tail completion requires bounded damage ratio endpoints"
            )
    return len(grid)


def _validated_evidence(
    raw: object, insured: int
) -> tuple[Mapping[str, ScientificValue], int]:
    if not isinstance(raw, Mapping):
        raise ValueError("evidence must be an object")
    evidence = cast(Mapping[str, ScientificValue], _freeze(raw))
    document = _object(
        _plain(evidence),
        {
            "damage_response",
            "response_model_sha256",
            "vulnerability_sha256",
            "member_id",
            "variable_id",
            "scenario_id",
            "period_id",
            "series",
            "probability_grid",
            "tail_completion",
            "matched_support_count",
            "matched_supports_sha256",
            "resolved_impacts",
            "input_resources",
        },
    )
    _identifier(document["damage_response"], "damage_response")
    for key in (
        "response_model_sha256",
        "vulnerability_sha256",
        "matched_supports_sha256",
    ):
        _digest(document[key])
    probability_count = _validate_selection(document)
    if (
        _integer(document["matched_support_count"], "matched_support_count", 1)
        > insured
    ):
        raise ValueError(
            "Annual loss metadata must account for the complete insured book"
        )
    assignments: dict[str, int] = {}
    for item in _array(document["resolved_impacts"], "resolved_impacts"):
        impact = _object(item, {"resolution_id", "impact_sha256", "asset_count"})
        key = _digest(impact["resolution_id"])
        _digest(impact["impact_sha256"])
        if key in assignments:
            raise ValueError("Annual loss resolved impacts must have unique identities")
        assignments[key] = _integer(impact["asset_count"], "asset_count", 1)
    if sum(assignments.values()) != insured:
        raise ValueError("Every insured interest must have a resolved impact")
    resources = _array(document["input_resources"], "input_resources")
    if not resources:
        raise ValueError("Annual loss metadata requires input resources")
    for item in resources:
        resource = _object(
            item, {"member_id", "resource_id", "source", "transformations"}
        )
        for key in ("member_id", "resource_id"):
            if resource[key] is not None:
                _identifier(resource[key], key)
        if resource["source"] is not None:
            _source(resource["source"])
        for transformation in _array(resource["transformations"], "transformations"):
            _transformation(transformation)
    return evidence, probability_count


@dataclass(frozen=True, slots=True, init=False)
class PerilLossEvidence:
    """Resolved hazard selection, vulnerability, coverage terms, and provenance."""

    terms_sha256: str
    input_sha256: str
    evidence: Mapping[str, ScientificValue]

    def __init__(self) -> None:
        raise TypeError("Read peril evidence from LossRunMetadata.perils")

    @property
    def series(self) -> SeriesReference:
        """The selected annual hazard variable, scenario, period and member."""
        return _series_reference(
            _plain(self.evidence["series"]), self.evidence["member_id"]
        )

    @property
    def damage_response(self) -> str:
        return cast(str, self.evidence["damage_response"])

    @property
    def vulnerability_sha256(self) -> str:
        return cast(str, self.evidence["vulnerability_sha256"])

    def _identity(self) -> dict[str, object]:
        return {
            "terms_sha256": self.terms_sha256,
            "input_sha256": self.input_sha256,
            "vulnerability_sha256": self.vulnerability_sha256,
        }

    def _document(self) -> dict[str, object]:
        return {
            "terms_sha256": self.terms_sha256,
            "input_sha256": self.input_sha256,
            "evidence": _plain(self.evidence),
        }


def _peril_evidence(document: object, insured: int) -> tuple[PerilLossEvidence, int]:
    record = _object(document, {"terms_sha256", "input_sha256", "evidence"})
    evidence, probability_count = _validated_evidence(record["evidence"], insured)
    value = object.__new__(PerilLossEvidence)
    object.__setattr__(value, "terms_sha256", _digest(record["terms_sha256"]))
    object.__setattr__(value, "input_sha256", _digest(record["input_sha256"]))
    object.__setattr__(value, "evidence", evidence)
    return value, probability_count


def _perils(
    value: object, report: ExecutionReport, component_count: int
) -> Mapping[str, PerilLossEvidence]:
    if not isinstance(value, dict) or not value:
        raise ValueError("Annual loss metadata requires at least one declared peril")
    coverage = report.counts
    insured = coverage.matched_assets
    perils = {}
    expected = 0
    for name, document in value.items():
        name = _identifier(name, "peril")
        evidence, q_count = _peril_evidence(document, insured)
        perils[name] = evidence
        expected += insured * q_count
    expected_components = insured if len(perils) == 1 else len(perils)
    if (
        coverage.unmatched_assets
        or coverage.no_data
        or coverage.requested != expected
        or coverage.produced != expected
        or component_count not in (0, expected_components)
    ):
        raise ValueError(
            "Annual loss metadata must account for the complete insured book"
        )
    return MappingProxyType(dict(sorted(perils.items())))


def _dependence(
    value: object, perils: Mapping[str, PerilLossEvidence]
) -> crc_framework.PerilDependence | None:
    if len(perils) == 1:
        if value is not None:
            raise ValueError("Cross-peril dependence requires multiple perils")
        return None
    if value is None:
        raise ValueError("Multi-peril annual loss requires explicit peril dependence")
    dependence = crc_framework.PerilDependence.from_json(_dump(value))
    if dependence.perils and dependence.perils != tuple(perils):
        raise ValueError("Peril dependence must cover exactly the simulated perils")
    return dependence


@dataclass(frozen=True, slots=True, init=False)
class LossRunMetadata:
    """Resolved provenance and storage checksums for an insured loss run."""

    run_id: str
    perils: Mapping[str, PerilLossEvidence]
    dependence: crc_framework.PerilDependence | None
    currency: str
    valuation_basis: str
    portfolio_sha256: str
    report: ExecutionReport
    spec: AnnualLossSpec
    secondary_stream: int
    rng_algorithm: str
    framework_version: str
    sdk_version: str
    total_insured_value: float
    correlation_repair: CorrelationRepair | None
    component_count: int
    totals_sha256: str
    components_sha256: str | None

    def __init__(self) -> None:
        raise TypeError("Read metadata from a LossResult or LossArtifact")

    @property
    def trial_count(self) -> int:
        return self.spec.trials

    def _identity(self) -> dict[str, object]:
        return {
            "version": ANNUAL_LOSS_VERSION,
            "loss_basis": "insured",
            "perils": {
                name: evidence._identity() for name, evidence in self.perils.items()
            },
            "dependence": None
            if self.dependence is None
            else _load(self.dependence.to_json()),
            "currency": self.currency,
            "valuation_basis": self.valuation_basis,
            "portfolio_sha256": self.portfolio_sha256,
            "spec": _spec_document(self.spec),
            "secondary_stream": self.secondary_stream,
            "stream_layout": _STREAM_LAYOUT,
            "rng_algorithm": self.rng_algorithm,
            "framework_version": self.framework_version,
            "sdk_version": self.sdk_version,
        }

    def _document(self) -> dict[str, object]:
        repair = self.correlation_repair
        return {
            **self._identity(),
            "run_id": self.run_id,
            "perils": {
                name: evidence._document() for name, evidence in self.perils.items()
            },
            "report": self.report._document(),
            "total_insured_value": self.total_insured_value,
            "correlation_repair": None
            if repair is None
            else {
                "minimum_eigenvalue": repair.minimum_eigenvalue,
                "maximum_change": repair.maximum_change,
            },
            "component_count": self.component_count,
            "totals_sha256": self.totals_sha256,
            "components_sha256": self.components_sha256,
        }

    def to_json(self) -> str:
        """Return the canonical JSON metadata stored in the artifact footer."""
        return _dump(self._document())


_METADATA_FIELDS = set(LossRunMetadata.__dataclass_fields__) | {
    "version",
    "loss_basis",
    "stream_layout",
}


def _metadata_from_document(document: object, *, new: bool = False) -> LossRunMetadata:
    fields = _METADATA_FIELDS - {"run_id"} if new else _METADATA_FIELDS
    v = _object(document, fields)
    if v["version"] != ANNUAL_LOSS_VERSION or v["loss_basis"] != "insured":
        raise ValueError("Expected the current insured annual loss format")
    if v["stream_layout"] != _STREAM_LAYOUT:
        raise ValueError("Unknown annual loss stream layout")
    report = _report(v["report"])
    if report.valid is not True:
        raise ValueError(
            "Annual loss metadata requires a complete valid execution report"
        )
    count = _integer(v["component_count"], "component_count")
    perils = _perils(v["perils"], report, count)
    components_sha256 = v["components_sha256"]
    if (components_sha256 is None) != (count == 0):
        raise ValueError("Component count and checksum must agree")
    if not 0 <= (tiv := _number(v["total_insured_value"], "total_insured_value")):
        raise ValueError("total_insured_value must be nonnegative")
    stream = _integer(v["secondary_stream"], "secondary_stream")
    if stream > _U64_MAX:
        raise ValueError("secondary_stream must be an unsigned 64-bit integer")
    metadata = object.__new__(LossRunMetadata)
    for key, value in {
        "perils": perils,
        "dependence": _dependence(v["dependence"], perils),
        "currency": _identifier(v["currency"], "currency"),
        "valuation_basis": _identifier(v["valuation_basis"], "valuation_basis"),
        "portfolio_sha256": _digest(v["portfolio_sha256"]),
        "report": report,
        "spec": _spec(v["spec"]),
        "secondary_stream": stream,
        "rng_algorithm": _identifier(v["rng_algorithm"], "rng_algorithm"),
        "framework_version": _identifier(v["framework_version"], "framework_version"),
        "sdk_version": _identifier(v["sdk_version"], "sdk_version"),
        "total_insured_value": tiv,
        "correlation_repair": _correlation_repair(v["correlation_repair"]),
        "component_count": count,
        "totals_sha256": _digest(v["totals_sha256"]),
        "components_sha256": None
        if components_sha256 is None
        else _digest(components_sha256),
    }.items():
        object.__setattr__(metadata, key, value)
    run_id = _sha256(_dump(metadata._identity()).encode())
    if not new and _digest(v["run_id"]) != run_id:
        raise ValueError("Annual loss run identity does not match its resolved inputs")
    object.__setattr__(metadata, "run_id", run_id)
    return metadata


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LossComparison:
    """Future minus baseline changes on aligned simulated years."""

    baseline_run_id: str
    future_run_id: str
    baseline_average_annual_loss: float
    future_average_annual_loss: float
    average_annual_loss_change: float
    trial_differences: numpy.typing.NDArray[numpy.float64]

    def __init__(self) -> None:
        raise TypeError("Use future.compare(baseline) to compare annual loss runs")

    @property
    def trial_ids(self) -> range:
        return range(len(self.trial_differences))


def _readonly(values: FloatArray) -> FloatArray:
    return numpy.frombuffer(values.tobytes(), dtype=numpy.float64)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LossResult:
    """Immutable annual totals and optional aligned interest or peril losses."""

    metadata: LossRunMetadata
    totals: numpy.typing.NDArray[numpy.float64]
    components: Mapping[str, numpy.typing.NDArray[numpy.float64]] | None

    def __init__(self) -> None:
        raise TypeError("Read a LossRun or LossArtifact to obtain a LossResult")

    @property
    def trial_ids(self) -> range:
        return range(self.metadata.trial_count)

    @property
    def report(self) -> ExecutionReport:
        return self.metadata.report

    def summary(
        self,
        *,
        levels: Sequence[float] = (0.95, 0.99, 0.995),
        return_periods: Sequence[float] = (10, 25, 50, 100, 200, 250),
    ) -> AnnualLossSummary:
        """Recompute empirical metrics and any available component allocations."""
        return crc_framework.summarize_annual_losses(
            self.totals,
            total_insured_value=self.metadata.total_insured_value,
            levels=levels,
            return_periods=return_periods,
            components=self.components,
        )

    def compare(self, baseline: LossResult) -> LossComparison:
        """Compare losses against a compatible baseline on shared primary draws."""
        if not isinstance(baseline, LossResult):
            raise TypeError("baseline must be a LossResult")
        before, after = baseline.metadata, self.metadata
        paired_fields = (
            "dependence",
            "currency",
            "valuation_basis",
            "portfolio_sha256",
            "spec",
            "rng_algorithm",
            "framework_version",
            "sdk_version",
            "total_insured_value",
            "correlation_repair",
        )
        differences = [
            field
            for field in paired_fields
            if getattr(before, field) != getattr(after, field)
        ]
        if before.perils.keys() != after.perils.keys():
            differences.append("perils")
        else:
            for name, peril in before.perils.items():
                other = after.perils[name]
                if peril.terms_sha256 != other.terms_sha256:
                    differences.append(f"terms_sha256 for {name!r}")
                if peril.vulnerability_sha256 != other.vulnerability_sha256:
                    differences.append(f"vulnerability_sha256 for {name!r}")
        if before.secondary_stream == after.secondary_stream:
            differences.append("secondary_stream must differ")
        if differences:
            raise ValueError("Cannot pair annual loss runs: " + ", ".join(differences))
        baseline_aal, future_aal = (
            crc_framework.summarize_annual_losses(
                result.totals,
                total_insured_value=result.metadata.total_insured_value,
                levels=(),
                return_periods=(),
            ).average_annual_loss
            for result in (baseline, self)
        )
        comparison = object.__new__(LossComparison)
        for name, value in {
            "baseline_run_id": before.run_id,
            "future_run_id": after.run_id,
            "baseline_average_annual_loss": baseline_aal,
            "future_average_annual_loss": future_aal,
            "average_annual_loss_change": future_aal - baseline_aal,
            "trial_differences": _readonly(self.totals - baseline.totals),
        }.items():
            object.__setattr__(comparison, name, value)
        return comparison


def _result(
    metadata: LossRunMetadata,
    totals: FloatArray,
    components: Mapping[str, FloatArray] | None,
) -> LossResult:
    value = object.__new__(LossResult)
    object.__setattr__(value, "metadata", metadata)
    object.__setattr__(value, "totals", totals)
    object.__setattr__(value, "components", components)
    return value


def _batch_size(component_count: int) -> int:
    """Keep reshaped component batches bounded as the portfolio grows."""
    return max(1, min(4096, 131_072 // max(1, component_count)))


def _ranges(size: int, component_count: int) -> Iterator[tuple[int, int]]:
    step = _batch_size(component_count)
    for start in range(0, size, step):
        yield start, min(start + step, size)


def _wire_bytes(values: FloatArray) -> bytes:
    return values.astype("<f8", copy=False).tobytes()


def _component_matrix(
    components: Mapping[str, FloatArray], first: int, stop: int
) -> FloatArray:
    return numpy.column_stack(
        tuple(losses[first:stop] for losses in components.values())
    )


def _component_checksum(components: Mapping[str, FloatArray], trials: int) -> str:
    digest = hashlib.sha256(_dump(list(components)).encode())
    for first, stop in _ranges(trials, len(components)):
        digest.update(_wire_bytes(_component_matrix(components, first, stop)))
    return "sha256:" + digest.hexdigest()


def make_loss_result(
    simulation: crc_framework.AnnualLossSimulation,
    *,
    peril_inputs: Mapping[str, _LossInputs],
    dependence: crc_framework.PerilDependence | None,
    currency: str,
    valuation_basis: str,
    portfolio_sha256: str,
    report: ExecutionReport,
) -> LossResult:
    if any(inputs.book_sha256 != portfolio_sha256 for inputs in peril_inputs.values()):
        raise ValueError("All simulated perils must describe the same insured book")
    if simulation.dependence != dependence or (
        dependence is not None and simulation.perils != tuple(sorted(peril_inputs))
    ):
        raise ValueError("Loss run provenance disagrees with the simulated perils")
    components = simulation.components
    if components is not None:
        components = MappingProxyType(dict(sorted(components.items())))
    count = 0 if components is None else len(components)
    repair = simulation.correlation_repair
    metadata = _metadata_from_document(
        {
            "version": ANNUAL_LOSS_VERSION,
            "loss_basis": "insured",
            "perils": {
                name: {
                    "terms_sha256": inputs.terms_sha256,
                    "input_sha256": inputs.input_sha256,
                    "evidence": inputs.evidence,
                }
                for name, inputs in peril_inputs.items()
            },
            "dependence": None if dependence is None else _load(dependence.to_json()),
            "currency": currency,
            "valuation_basis": valuation_basis,
            "portfolio_sha256": portfolio_sha256,
            "report": report._document(),
            "spec": _spec_document(simulation.spec),
            "secondary_stream": simulation.secondary_stream,
            "stream_layout": _STREAM_LAYOUT,
            "rng_algorithm": simulation.rng_algorithm,
            "framework_version": version("crc-framework"),
            "sdk_version": version("crc-sdk"),
            "total_insured_value": simulation.total_insured_value,
            "correlation_repair": None
            if repair is None
            else {
                "minimum_eigenvalue": repair.minimum_eigenvalue,
                "maximum_change": repair.maximum_change,
            },
            "component_count": count,
            "totals_sha256": _sha256(_wire_bytes(simulation.totals)),
            "components_sha256": None
            if components is None
            else _component_checksum(components, len(simulation.totals)),
        },
        new=True,
    )
    return _result(metadata, simulation.totals, components)


def _schema(component_count: int) -> pa.Schema:
    fields = _TOTAL_FIELDS + (_COMPONENT_FIELDS if component_count else ())
    return pa.schema(fields)


def _parquet_metadata(parquet: pq.ParquetFile) -> LossRunMetadata:
    footer = parquet.metadata.metadata or {}
    document = footer.get(_DOCUMENT)
    checksum = footer.get(_DOCUMENT_DIGEST)
    if document is None or checksum is None:
        raise ValueError(
            "Parquet footer does not contain annual loss metadata and checksum"
        )
    if len(document) > _FOOTER_LIMIT:
        raise ValueError("Annual loss metadata exceeds the 8 MiB footer limit")
    if _sha256(document).encode() != checksum:
        raise ValueError(
            "Annual loss metadata checksum does not match the Parquet footer"
        )
    metadata = _metadata_from_document(_load(document.decode()))
    if not parquet.schema_arrow.remove_metadata().equals(
        _schema(metadata.component_count)
    ):
        raise ValueError("Parquet fields do not match the annual loss format")
    if parquet.metadata.num_rows != metadata.trial_count:
        raise ValueError("Annual loss row count does not match metadata trial count")
    return metadata


def _read_component_ids(
    array: pa.Array, first: int, count: int
) -> tuple[str, ...] | None:
    if first:
        if array.null_count != len(array):
            raise ValueError("Component IDs must be stored on trial 0 only")
        return None
    if array.null_count != len(array) - 1 or array[0].as_py() is None:
        raise ValueError("Component IDs must be stored on trial 0 only")
    ids = tuple(_identifier(value, "component ID") for value in array[0].as_py())
    if len(ids) != count or tuple(sorted(set(ids))) != ids:
        raise ValueError(
            "Component IDs must be unique, sorted, and match component_count"
        )
    return ids


def _read_component_matrix(array: pa.Array, count: int) -> FloatArray:
    offsets = array.offsets.to_numpy()
    values = array.values.slice(int(offsets[0]), int(offsets[-1] - offsets[0]))
    if (
        array.null_count
        or values.null_count
        or not numpy.all(numpy.diff(offsets) == count)
    ):
        raise ValueError("Every trial must have a complete component loss vector")
    return cast(FloatArray, values.to_numpy()).reshape(-1, count)


def _losses_valid(values: FloatArray) -> bool:
    return bool(numpy.all(numpy.isfinite(values) & (values >= 0)))


def _scan_artifact(
    parquet: pq.ParquetFile, metadata: LossRunMetadata, *, load_values: bool
) -> LossResult | None:
    size, width = metadata.trial_count, metadata.component_count
    totals = numpy.empty(size, dtype=numpy.float64) if load_values else None
    components = (
        numpy.empty((width, size), dtype=numpy.float64)
        if width and load_values
        else None
    )
    total_digest = hashlib.sha256()
    component_digest = hashlib.sha256()
    ids: tuple[str, ...] | None = None
    insured_value = metadata.total_insured_value
    rounding_allowance = 1e-12 * insured_value
    peril_count = len(metadata.perils)
    first = 0
    for batch in parquet.iter_batches(batch_size=_batch_size(width)):
        stop = first + len(batch)
        trials, losses = batch.column(0), batch.column(1)
        if trials.null_count or not numpy.array_equal(
            trials.to_numpy(), numpy.arange(first, stop)
        ):
            raise ValueError("Annual loss trial IDs must be consecutive from zero")
        losses_array = cast(FloatArray, losses.to_numpy())
        if losses.null_count or not _losses_valid(losses_array):
            raise ValueError("Annual insured losses must be finite and nonnegative")
        if numpy.any(losses_array / peril_count - insured_value > rounding_allowance):
            raise ValueError(
                "Annual insured losses exceed total insured value across perils"
            )
        total_digest.update(_wire_bytes(losses_array))
        if totals is not None:
            totals[first:stop] = losses_array
        if width:
            first_ids = _read_component_ids(batch.column(2), first, width)
            if first_ids is not None:
                if peril_count > 1 and first_ids != tuple(metadata.perils):
                    raise ValueError("Component IDs must match the simulated perils")
                ids = first_ids
                component_digest.update(_dump(list(ids)).encode())
            matrix = _read_component_matrix(batch.column(3), width)
            if not _losses_valid(matrix):
                raise ValueError("Component losses must be finite and nonnegative")
            if numpy.any(matrix - insured_value > rounding_allowance):
                raise ValueError("Component losses exceed total insured value")
            with numpy.errstate(over="ignore", invalid="ignore"):
                sums = matrix.sum(axis=1)
                tolerance = 1e-12 * numpy.maximum(sums, losses_array)
                reconciled = numpy.isfinite(sums) & (
                    numpy.abs(sums - losses_array) <= tolerance
                )
            if not numpy.all(reconciled):
                raise ValueError(
                    "Component losses must reconcile with each trial total"
                )
            component_digest.update(_wire_bytes(matrix))
            if components is not None:
                components[:, first:stop] = matrix.T
        first = stop
    if first != size or "sha256:" + total_digest.hexdigest() != metadata.totals_sha256:
        raise ValueError("Annual totals checksum does not match the Parquet rows")
    if width and "sha256:" + component_digest.hexdigest() != metadata.components_sha256:
        raise ValueError("Annual component checksum does not match the Parquet rows")
    if totals is None:
        return None
    aligned = (
        MappingProxyType(
            {
                identifier: _readonly(values)
                for identifier, values in zip(ids, components, strict=True)
            }
        )
        if ids is not None and components is not None
        else None
    )
    return _result(metadata, _readonly(totals), aligned)


def _artifact_error(
    error: Exception, report: ExecutionReport | None
) -> ValidationError:
    diagnostic = ExecutionDiagnostic(
        "invalid_annual_loss_artifact",
        crc_framework.DiagnosticSeverity.ERROR,
        Stage.VALIDATION,
        f"Invalid annual loss artifact: {error}",
    )
    report = report or ExecutionReport(False, None)
    return ValidationError(
        report=replace(
            report,
            complete=True,
            valid=False,
            diagnostics=(*report.diagnostics, diagnostic),
        )
    )


@dataclass(frozen=True, slots=True, init=False)
class LossArtifact:
    """A local Parquet artifact containing an insured annual loss sample."""

    path: Path
    _expected: LossRunMetadata | None

    def __init__(self) -> None:
        raise TypeError("Use LossArtifact.open(path) or LossRun.write(path)")

    @classmethod
    def _create(
        cls, path: Path, expected: LossRunMetadata | None = None
    ) -> LossArtifact:
        value = object.__new__(cls)
        object.__setattr__(value, "path", path)
        object.__setattr__(value, "_expected", expected)
        return value

    @classmethod
    def open(cls, path: LocalPath) -> LossArtifact:
        """Reference a local artifact and defer file I/O to a terminal."""
        return cls._create(_local_path(path))

    def _load(
        self, *, scan: bool, load_values: bool
    ) -> tuple[LossRunMetadata, LossResult | None]:
        metadata = None
        try:
            with pq.ParquetFile(self.path) as parquet:
                metadata = _parquet_metadata(parquet)
                if self._expected is not None and metadata != self._expected:
                    raise ValueError("Annual loss artifact changed since publication")
                result = (
                    _scan_artifact(parquet, metadata, load_values=load_values)
                    if scan
                    else None
                )
        except (OSError, ValueError, TypeError, OverflowError) as error:
            raise _artifact_error(
                error, None if metadata is None else metadata.report
            ) from error
        return metadata, result

    @property
    def metadata(self) -> LossRunMetadata:
        if self._expected is not None:
            return self._expected
        return self._load(scan=False, load_values=False)[0]

    @property
    def report(self) -> ExecutionReport:
        return self.metadata.report

    def read(self) -> LossResult:
        """Read and verify every trial and any persisted component losses."""
        _, result = self._load(scan=True, load_values=True)
        assert result is not None
        return result

    def validate(self) -> ExecutionReport:
        """Validate footer identity, row alignment, and loss checksums."""
        try:
            return self._load(scan=True, load_values=False)[0].report
        except ValidationError as error:
            return error.report


def _record_batch(
    result: LossResult, schema: pa.Schema, first: int, stop: int
) -> pa.RecordBatch:
    columns = [
        pa.array(numpy.arange(first, stop, dtype=numpy.int64)),
        pa.array(result.totals[first:stop]),
    ]
    components = result.components
    if components is not None:
        row_count, width = stop - first, len(components)
        ids: list[list[str] | None] = [None] * row_count
        if not first:
            ids[0] = list(components)
        matrix = _component_matrix(components, first, stop)
        columns.extend(
            [
                pa.array(ids, type=schema.field("component_ids").type),
                pa.ListArray.from_arrays(
                    pa.array(numpy.arange(row_count + 1, dtype=numpy.int32) * width),
                    pa.array(matrix.reshape(-1)),
                    type=schema.field("component_losses").type,
                ),
            ]
        )
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def write_loss_artifact(
    path: LocalPath,
    result: LossResult,
    *,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> LossArtifact:
    """Stage, verify, and atomically publish one annual loss Parquet file."""
    destination = _publication_path(path, overwrite)
    metadata = result.metadata
    document = metadata.to_json().encode()
    if len(document) > _FOOTER_LIMIT:
        raise ValueError("Annual loss metadata exceeds the 8 MiB footer limit")
    schema = _schema(metadata.component_count)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".parquet", dir=destination.parent
    )
    os.close(descriptor)
    staging = Path(name)
    try:
        with pq.ParquetWriter(staging, schema) as writer:
            writer.add_key_value_metadata(
                {_DOCUMENT: document, _DOCUMENT_DIGEST: _sha256(document)}
            )
            for first, stop in _ranges(metadata.trial_count, metadata.component_count):
                writer.write_batch(_record_batch(result, schema, first, stop))
                if progress is not None:
                    progress(stop, metadata.trial_count)
        staged = LossArtifact._create(staging, metadata)
        staged._load(scan=True, load_values=False)
        with staging.open("rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(staging, destination)
        else:
            os.link(staging, destination)
        return LossArtifact._create(destination, metadata)
    finally:
        staging.unlink(missing_ok=True)
