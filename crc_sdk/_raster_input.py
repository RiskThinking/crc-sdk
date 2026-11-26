"""Lazy, verified quantile raster inputs on the canonical source graph."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]

from ._climate_io import RasterReadCounts, iter_raster_quantile_rows
from ._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
)
from ._declarations import DataSpec, Period, Probability, Scenario, SeriesSpec, Variable
from ._errors import CapabilityError
from ._graph import ExecutionScope, NodeStream
from ._inspection import Capability, Inspection, Requirement, SeriesReference
from ._recipes import _key
from ._runtime import LocalPath, _local_path, _memory_bytes
from ._schema import DatasetRow, dataset_arrow_schema, rows_to_table
from ._selection import SelectionState, selected_metadata
from ._source_io import SourceNode, fetch, native_data, release_snapshot


@dataclass(frozen=True, slots=True)
class _RasterNode(SourceNode[DatasetMetadata]):
    rasters: tuple[tuple[float, Path], ...]
    variable: Variable
    probability: Probability | None
    scenario: Scenario | None
    period: Period | None
    state: SelectionState = SelectionState()
    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _key(
                "user-quantile-rasters",
                [(q, str(path)) for q, path in self.rasters],
                self.variable._document(),
                None if self.probability is None else self.probability._document(),
                None if self.scenario is None else self.scenario._document(),
                None if self.period is None else self.period._document(),
                self.state.document(),
            ),
        )

    def selected(self, state: SelectionState) -> _RasterNode:
        return replace(self, state=state)

    def _variable_ids(self) -> frozenset[str]:
        return frozenset((self.variable.id,))

    def _metadata(self, source: SourceIdentity | None = None) -> DatasetMetadata:
        return DatasetMetadata(
            spec=DataSpec(
                [self.variable],
                series=[
                    SeriesSpec(
                        self.variable,
                        scenario=self.scenario,
                        period=self.period,
                        probability=self.probability,
                    )
                ],
            ),
            spatial=SpatialContract(
                supports=(SupportKind.POLYGON,), index_resolution=9
            ),
            report=ExecutionReport(
                False, None, source_identities=() if source is None else (source,)
            ),
            source=source,
        )

    def inspection(self, scope: ExecutionScope) -> Inspection:
        if self.state.time is not None:
            raise CapabilityError(
                "Supplied quantile rasters cannot be recomputed for "
                "observation-time windows; use periods= to choose "
                "complete supplied series."
            )
        metadata = selected_metadata(
            self._metadata(), self.state.resolve(self._metadata().spec)
        )
        return Inspection(
            schema=dataset_arrow_schema(metadata),
            series=tuple(
                SeriesReference(
                    s.variable,
                    scenario=s.scenario,
                    period=s.period,
                    probability=s.probability,
                )
                for s in metadata.spec.series
            ),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
                Capability.FIT,
            ),
            requirements=(
                Requirement(
                    "snapshot_resolved_on_read",
                    "Local raster bytes and their common grid are verified when read.",
                    "rasters",
                ),
            ),
        )

    def open(self, scope: ExecutionScope) -> NodeStream[DatasetMetadata]:
        metadata = self._metadata()
        unbound = replace(
            self.state,
            series=None
            if self.state.series is None
            else tuple(replace(r, release=None) for r in self.state.series),
        )
        initial = unbound.resolve(metadata.spec)
        if initial.empty:
            empty = selected_metadata(metadata, initial)
            return NodeStream(dataset_arrow_schema(empty), empty, iter(()))
        if self.state.time is not None:
            raise CapabilityError(
                "Supplied quantile rasters cannot be recomputed for "
                "observation-time windows; use periods= to choose "
                "complete supplied series."
            )
        paths, resources = {}, []
        for q, path in self.rasters:
            _, resource = fetch(scope, path.as_uri())
            pinned, _ = scope.pin_local(path, expected_sha256=resource.sha256)
            paths[q] = pinned
            resources.append(resource)
        source = SourceIdentity(
            "user",
            "quantile_rasters",
            release=release_snapshot(resources),
            resources=tuple({r.uri: r for r in resources}.values()),
        )
        metadata = self._metadata(source)
        resolved = self.state.resolve(metadata.spec, source=source)
        metadata = selected_metadata(metadata, resolved)
        counts = RasterReadCounts()
        produced = 0
        budget = _memory_bytes(scope.runtime.memory_limit)
        maximum = (
            scope.runtime.batch_rows
            if budget is None
            else min(scope.runtime.batch_rows, max(1, budget // (len(paths) * 24)))
        )

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal produced
            pending: list[DatasetRow] = []
            with native_data(source="quantile_rasters", scope=scope):
                rows = iter_raster_quantile_rows(
                    paths,
                    variable=self.variable,
                    scenario=self.scenario,
                    period=self.period,
                    area=resolved.area,
                    max_pixels=maximum,
                    counts=counts,
                )
                for row in rows:
                    if not resolved.matches(row):
                        continue
                    pending.append(row)
                    produced += 1
                    if len(pending) >= scope.runtime.batch_rows:
                        yield from rows_to_table(pending, metadata).to_batches(
                            max_chunksize=scope.runtime.batch_rows
                        )
                        pending.clear()
                if pending:
                    yield from rows_to_table(pending, metadata).to_batches(
                        max_chunksize=scope.runtime.batch_rows
                    )

        def report() -> ExecutionReport:
            diagnostics = (
                ()
                if not counts.skipped
                else (
                    ExecutionDiagnostic(
                        "source_no_data",
                        cf.DiagnosticSeverity.WARNING,
                        Stage.DECODING,
                        f"Skipped {counts.skipped} raster supports with fewer "
                        f"than two valid probability knots: {counts.dry} without "
                        f"a valid value and {counts.sparse} with one.",
                        source="quantile_rasters",
                    ),
                )
            )
            return ExecutionReport(
                False,
                None,
                diagnostics,
                ExecutionCounts(
                    requested=counts.requested,
                    produced=produced,
                    skipped=counts.skipped,
                ),
                (source,),
            )

        return NodeStream(
            dataset_arrow_schema(metadata), metadata, batches(), report=report
        )


def raster_node(
    rasters: Mapping[float, LocalPath],
    *,
    variable: Variable,
    probability: Probability | None,
    scenario: Scenario | None = None,
    period: Period | None = None,
) -> SourceNode[DatasetMetadata]:
    if not isinstance(rasters, Mapping) or len(rasters) < 2:
        raise ValueError(
            "rasters requires at least two probability-to-local-path entries"
        )
    if not isinstance(variable, Variable):
        raise TypeError("variable must be Variable")
    if probability is not None and not isinstance(probability, Probability):
        raise TypeError("probability must be Probability")
    if scenario is not None and not isinstance(scenario, Scenario):
        raise TypeError("scenario must be Scenario")
    if period is not None and not isinstance(period, Period):
        raise TypeError("period must be Period")
    entries = []
    for q, path in rasters.items():
        if (
            isinstance(q, bool)
            or not isinstance(q, (int, float))
            or not math.isfinite(q)
            or not 0 <= q <= 1
        ):
            raise ValueError("raster probabilities must be finite and within [0, 1]")
        entries.append((float(q), _local_path(path)))
    return _RasterNode(tuple(sorted(entries)), variable, probability, scenario, period)
