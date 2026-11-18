"""Predefined climate indicators built from explicit native daily evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cache
from typing import Any, Literal, cast
from urllib.parse import quote

import crc_framework as cf
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from ._aggregation import (
    Coverage,
    MeasurementChange,
    ReductionKind,
    Sampling,
    TemporalReduction,
    TimestampPosition,
)
from ._climate_grid import (
    GridCell,
    annual_records,
    calendar_name,
    cf_calendar,
    grid_cells,
    grid_values,
)
from ._climate_io import (
    netcdf_header_snapshot,
    netcdf_window_snapshot,
    open_zarr_snapshot,
    zarr_resources,
)
from ._dataset_schema import (
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    ObservationTimeSelection,
    ReleaseReference,
    ResourceIdentity,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
    UnitConversion,
)
from ._declarations import (
    Calendar,
    DataSpec,
    Period,
    PeriodKind,
    Probability,
    Scenario,
    SeriesSpec,
    Variable,
    scientific_identity,
)
from ._errors import CapabilityError
from ._evidence import (
    EvidenceKind,
    EvidenceMetadata,
    ObservationSupport,
    evidence_arrow_schema,
    evidence_table,
)
from ._evidence_recipes import Observations
from ._graph import ExecutionScope, Node, NodeStream
from ._inspection import (
    Capability,
    IndicatorReference,
    Inspection,
    Operation,
    Requirement,
    SeriesReference,
)
from ._providers import cmip6 as cmip
from ._providers import era5 as era
from ._recipes import Samples, _key
from ._runtime import _memory_bytes
from ._selection import SelectionState, selected_metadata
from ._source_io import (
    SourceNode,
    fetch,
    inventory_payload_pin,
    inventory_release,
    native_data,
    payload_selection,
    selected_series_period,
    selected_source_time,
    verify_payload,
)

ERA5_STORE = era.ARCO_0P25
CMIP6_CATALOGUE = cmip.CATALOGUE_URL
NEX_BUCKET = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com"
NEX_PREFIX = "NEX-GDDP-CMIP6/"
_OPERATIONS = {
    "annual_maximum_daily_maximum_temperature": "txx",
    "annual_minimum_daily_minimum_temperature": "tnn",
    "annual_maximum_daily_precipitation": "rx1day",
    "annual_maximum_five_day_precipitation": "rx5day",
}
_NATIVE_VARIABLES = {
    operation: cmip.INDICATOR_VARIABLES[native].variable
    for operation, native in _OPERATIONS.items()
}
_NEX_YEAR = re.compile(r"_(\d{4})(?:_v(\d+(?:\.\d+)*))?\.nc$")


def _initialize(value: object, node: _ClimateNode) -> None:
    for name, argument in (
        ("_base", node),
        ("_selection", SelectionState()),
        (
            "_operations",
            (
                Operation(
                    Stage.ACQUISITION,
                    f"Read {node.provider.upper()} source evidence "
                    "and the declared temporal indicator",
                ),
            ),
        ),
        ("_path", None),
        ("_metadata", None),
    ):
        object.__setattr__(value, name, argument)


def _coverage(value: Coverage | None) -> Coverage:
    if value is not None and not isinstance(value, Coverage):
        raise TypeError("coverage must be Coverage")
    return Coverage.complete() if value is None else value


def _indicator(
    provider: str, value: Enum | IndicatorReference | None, allowed: type[Enum]
) -> str | None:
    if value is None:
        return None
    if isinstance(value, allowed):
        return str(value.value)
    if not isinstance(value, IndicatorReference):
        raise TypeError(
            f"indicator must be {provider.upper()}.Indicator or IndicatorReference"
        )
    for operation in _OPERATIONS:
        if value.id == f"{provider}:{operation}":
            actual = _reference(provider, operation)
            if value.measurement is not None and scientific_identity(
                value.measurement
            ) != scientific_identity(cast(cf.Measurement, actual.measurement)):
                raise ValueError(
                    "indicator measurement contradicts the published recipe"
                )
            if (
                value.probability is not None
                and value.probability != actual.probability
            ):
                raise ValueError(
                    "indicator probability contradicts the published recipe"
                )
            return operation
    raise ValueError(f"{provider.upper()} has no indicator {value.id!r}")


class ERA5(Samples):
    __slots__ = ()

    class Indicator(Enum):
        ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE = (
            "annual_maximum_daily_maximum_temperature"
        )
        ANNUAL_MINIMUM_DAILY_MINIMUM_TEMPERATURE = (
            "annual_minimum_daily_minimum_temperature"
        )
        ANNUAL_MAXIMUM_DAILY_PRECIPITATION = "annual_maximum_daily_precipitation"
        ANNUAL_MAXIMUM_FIVE_DAY_PRECIPITATION = "annual_maximum_five_day_precipitation"

    def __init__(
        self,
        *,
        indicator: ERA5.Indicator | IndicatorReference | None = None,
        release: ReleaseReference | None = None,
        coverage: Coverage | None = None,
    ) -> None:
        _initialize(
            self,
            _ClimateNode(
                "era5",
                _indicator("era5", indicator, ERA5.Indicator),
                release,
                _coverage(coverage),
            ),
        )

    def observations(self) -> Observations:
        """Read native hourly evidence underlying this configured indicator."""
        return _observations(self)


class CMIP6(Samples):
    __slots__ = ()

    class Indicator(Enum):
        ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE = (
            "annual_maximum_daily_maximum_temperature"
        )
        ANNUAL_MINIMUM_DAILY_MINIMUM_TEMPERATURE = (
            "annual_minimum_daily_minimum_temperature"
        )
        ANNUAL_MAXIMUM_DAILY_PRECIPITATION = "annual_maximum_daily_precipitation"
        ANNUAL_MAXIMUM_FIVE_DAY_PRECIPITATION = "annual_maximum_five_day_precipitation"

    def __init__(
        self,
        *,
        indicator: CMIP6.Indicator | IndicatorReference | None = None,
        model: str | None = None,
        scenario: Scenario | str | None = None,
        member: str | None = None,
        release: ReleaseReference | None = None,
        coverage: Coverage | None = None,
    ) -> None:
        _initialize(
            self,
            _ClimateNode(
                "cmip6",
                _indicator("cmip6", indicator, CMIP6.Indicator),
                release,
                _coverage(coverage),
                model,
                scenario,
                member,
            ),
        )

    def observations(self) -> Observations:
        """Read native daily evidence underlying this configured indicator."""
        return _observations(self)


class NEX(Samples):
    __slots__ = ()

    class Indicator(Enum):
        ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE = (
            "annual_maximum_daily_maximum_temperature"
        )
        ANNUAL_MINIMUM_DAILY_MINIMUM_TEMPERATURE = (
            "annual_minimum_daily_minimum_temperature"
        )
        ANNUAL_MAXIMUM_DAILY_PRECIPITATION = "annual_maximum_daily_precipitation"
        ANNUAL_MAXIMUM_FIVE_DAY_PRECIPITATION = "annual_maximum_five_day_precipitation"

    def __init__(
        self,
        *,
        indicator: NEX.Indicator | IndicatorReference | None = None,
        model: str | None = None,
        scenario: Scenario | str | None = None,
        member: str | None = None,
        release: ReleaseReference | None = None,
        coverage: Coverage | None = None,
    ) -> None:
        _initialize(
            self,
            _ClimateNode(
                "nex",
                _indicator("nex", indicator, NEX.Indicator),
                release,
                _coverage(coverage),
                model,
                scenario,
                member,
            ),
        )

    def observations(self) -> Observations:
        """Read native daily evidence underlying this configured indicator."""
        return _observations(self)


def _native_variable(provider: str, operation: str) -> Variable:
    precipitation = _OPERATIONS[operation].startswith("rx")
    if provider == "era5":
        name = ERA5_STORE.variables["tp" if precipitation else "t2m"]
        measurement: cf.Measurement = (
            cf.PrecipitationAmount(
                unit=cf.LengthUnit.METRE,
                accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.HOUR),
            )
            if precipitation
            else cf.Temperature(unit=cf.TemperatureUnit.KELVIN)
        )
        label = (
            "Hourly precipitation amount" if precipitation else "Hourly 2 m temperature"
        )
    else:
        name = cmip.INDICATOR_VARIABLES[_OPERATIONS[operation]].variable
        measurement = (
            cf.PrecipitationRate(
                unit=cf.PrecipitationRateUnit.MILLIMETRES_PER_DAY,
                accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.DAY),
                aggregation=cf.Aggregation.MEAN,
            )
            if precipitation
            else cf.Temperature(unit=cf.TemperatureUnit.KELVIN)
        )
        label = (
            "Daily mean precipitation rate"
            if precipitation
            else "Daily minimum temperature"
            if name == "tasmin"
            else "Daily maximum temperature"
        )
    return Variable(f"{provider}:{name}", measurement=measurement, label=label)


def _observations(recipe: Samples) -> Observations:
    if not isinstance(recipe._base, _ClimateNode):
        raise CapabilityError(
            "Request native observations before applying scientific transformations"
        )
    source = recipe._base.selected(recipe._selection)
    operation = source._operation()
    if operation is None:
        raise CapabilityError(
            "Choose indicator= before observations(); "
            "inspect() returns typed indicator choices"
        )
    annual_variable = _variable(source.provider, operation)
    native = _native_variable(source.provider, operation)

    def variable(value: str | Variable) -> str | Variable:
        if (value.id if isinstance(value, Variable) else value) != annual_variable.id:
            return value
        if isinstance(value, Variable) and scientific_identity(
            value
        ) != scientific_identity(annual_variable):
            raise ValueError("selected indicator variable has conflicting science")
        return native if isinstance(value, Variable) else native.id

    references = None
    if source.state.series is not None:
        references = []
        for reference in source.state.series:
            actual = _reference(source.provider, operation)
            if (
                reference.probability is not None
                and reference.probability != actual.probability
            ):
                raise ValueError(
                    "selected indicator series has conflicting probability"
                )
            references.append(
                replace(
                    reference,
                    variable=cast(Variable, variable(reference.variable)),
                    probability=None,
                )
            )
    state = replace(
        source.state,
        variables=None
        if source.state.variables is None
        else tuple(variable(value) for value in source.state.variables),
        series=None if references is None else tuple(references),
    )
    node = replace(source, indicator=operation, raw=True, state=SelectionState())
    return Observations._create(
        cast(Node[EvidenceMetadata[Literal[EvidenceKind.OBSERVATIONS]]], node),
        selection=state,
        operations=(
            Operation(
                Stage.ACQUISITION,
                f"Read native {source.provider.upper()} observations "
                "without annual reduction",
            ),
        ),
    )


def _reference(provider: str, operation: str) -> IndicatorReference:
    native = _OPERATIONS[operation]
    if native in {"txx", "tnn"}:
        measurement: cf.Measurement = cf.Temperature(unit=cf.TemperatureUnit.CELSIUS)
    else:
        measurement = cf.PrecipitationAmount(
            unit=cf.LengthUnit.MILLIMETRE,
            accumulation_duration=cf.Duration(
                5 if native == "rx5day" else 1, unit=cf.TimeUnit.DAY
            ),
        )
    return IndicatorReference(
        f"{provider}:{operation}",
        operation.replace("_", " ").capitalize(),
        measurement,
        Probability.annual(tail=cf.Tail.LOWER if native == "tnn" else cf.Tail.UPPER),
    )


def _variable(provider: str, operation: str) -> Variable:
    reference = _reference(provider, operation)
    return Variable(
        reference.id, measurement=reference.measurement, label=reference.label
    )


@dataclass(frozen=True, slots=True)
class _ClimateNode(SourceNode[EvidenceMetadata]):
    provider: str
    indicator: str | None
    release: ReleaseReference | None
    coverage: Coverage
    model: str | None = None
    scenario: Scenario | str | None = None
    member: str | None = None
    state: SelectionState = SelectionState()
    raw: bool = False
    key: str = field(init=False)

    def __post_init__(self) -> None:
        references = self.state.series or ()
        if self.provider != "era5" and self.scenario is None and self.state.scenarios:
            if len(self.state.scenarios) == 1:
                object.__setattr__(self, "scenario", self.state.scenarios[0])
        if self.provider != "era5" and references:
            for name in ("model", "member"):
                selected = {
                    value
                    for ref in references
                    if (value := getattr(ref, name)) is not None
                }
                configured = getattr(self, name)
                if configured is not None:
                    selected.add(configured)
                if len(selected) > 1:
                    raise ValueError(
                        f"selected series has conflicting {name} identities"
                    )
                if selected:
                    object.__setattr__(self, name, selected.pop())
            scenarios = tuple(
                ref.scenario for ref in references if ref.scenario is not None
            )
            if scenarios:
                scenario = self.scenario if self.scenario is not None else scenarios[0]
                scenario_id = (
                    scenario.id if isinstance(scenario, Scenario) else scenario
                )
                if any(value.id != scenario_id for value in scenarios):
                    raise ValueError(
                        "selected series has conflicting scenario identities"
                    )
                object.__setattr__(self, "scenario", scenario)
        if self.release is not None and not isinstance(self.release, ReleaseReference):
            raise TypeError("release must be ReleaseReference")
        for name in ("model", "member"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or value.strip() != value
            ):
                raise ValueError(f"{name} must be a nonempty exact source identifier")
        if self.scenario is not None and not isinstance(self.scenario, (Scenario, str)):
            raise TypeError("scenario must be Scenario or a source experiment ID")
        object.__setattr__(
            self,
            "key",
            _key(
                "climate-indicator",
                self.provider,
                self.indicator,
                None if self.release is None else self.release._document(),
                self.coverage.fraction,
                self.model,
                self.scenario._document()
                if isinstance(self.scenario, Scenario)
                else self.scenario,
                self.member,
                self.state.document(),
                self.raw,
            ),
        )

    def selected(self, state: SelectionState) -> _ClimateNode:
        return replace(self, state=state)

    def _operation(self) -> str | None:
        if self.indicator is not None:
            return self.indicator
        if self.state.series is not None and len(self.state.series) == 1:
            requested = self.state.series[0].variable.id
        elif self.state.variables is not None and len(self.state.variables) == 1:
            value = self.state.variables[0]
            requested = value.id if isinstance(value, Variable) else value
        else:
            return None
        return next(
            (
                operation
                for operation in _OPERATIONS
                if requested == f"{self.provider}:{operation}"
            ),
            None,
        )

    def requirements(self) -> tuple[Requirement, ...]:
        return self._missing()

    def _missing(self) -> tuple[Requirement, ...]:
        result = []
        if self._operation() is None:
            result.append(
                Requirement(
                    "indicator_required",
                    "Choose a typed indicator reference returned by inspect().",
                    parameter="indicator",
                )
            )
        if self.state.area is None:
            result.append(
                Requirement(
                    "area_required",
                    "Select bounded coverage with .select(area=Bounds(...)).",
                    parameter="area",
                )
            )
        if self.provider == "era5" and selected_source_time(self.state) is None:
            result.append(
                Requirement(
                    "time_required",
                    "Select ERA5 time bounds with .select(time=Period.window(...)).",
                    parameter="time",
                )
            )
        if self.provider != "era5":
            for name in ("model", "scenario", "member"):
                if getattr(self, name) is None:
                    result.append(
                        Requirement(
                            name + "_required",
                            f"Choose the source {name} through inspect().series.",
                            parameter=name,
                        )
                    )
        return tuple(result)

    def _source(self, scope: ExecutionScope) -> _GridSource:
        with native_data(source=self.provider, scope=scope):
            operation = self._operation()
            if self.provider == "era5":
                return _era5_source(scope, self, operation)
            assert operation is not None
            if self.provider == "cmip6":
                _, release = payload_selection(self.state, self.release)
                return _cmip_source(scope, self, release=release)
            return _nex_source(scope, self, operation)

    def inspection(self, scope: ExecutionScope) -> Inspection:
        if self.provider != "era5" and (
            self._operation() is None
            or self.model is None
            or self.scenario is None
            or self.member is None
        ):
            with native_data(source=self.provider, scope=scope):
                return _discover_climate(scope, self)
        source = self._source(scope)
        assert source.identity.release is not None
        _, release = payload_selection(self.state, self.release)
        inventory_payload_pin(source.identity.release, release)
        operations = (
            tuple(_OPERATIONS)
            if self._operation() is None
            else (cast(str, self._operation()),)
        )
        if source.requirements:
            empty = EvidenceMetadata(
                kind=EvidenceKind.OBSERVATIONS if self.raw else EvidenceKind.SAMPLES,
                spec=DataSpec([], series=[]),
                spatial=SpatialContract(
                    supports=(SupportKind.POLYGON,), index_resolution=9
                ),
                report=ExecutionReport(
                    False, None, source_identities=(source.identity,)
                ),
                source=source.identity,
            )
            return Inspection(
                schema=evidence_arrow_schema(empty),
                indicators=tuple(
                    _reference(self.provider, operation) for operation in _OPERATIONS
                ),
                releases=(source.identity.release,),
                requirements=source.requirements + self._missing() + source.notes,
                source=source.identity,
                capabilities=(Capability.INSPECT,),
            )
        period = _period(
            selected_series_period(self.state) or selected_source_time(self.state),
            source.first_year,
            source.last_year,
            source.calendar,
        )
        references = tuple(
            SeriesReference(
                _native_variable(self.provider, operation)
                if self.raw
                else _variable(self.provider, operation),
                scenario=source.scenario,
                period=period,
                release=source.identity.release,
                probability=None
                if self.raw
                else _reference(self.provider, operation).probability,
                model=source.identity.model,
                member=source.identity.member,
            )
            for operation in operations
        )
        return Inspection(
            schema=evidence_arrow_schema(_metadata(self, source, operations, period)),
            series=references,
            releases=(source.identity.release,),
            indicators=tuple(
                _reference(self.provider, operation) for operation in _OPERATIONS
            ),
            capabilities=(
                Capability.INSPECT,
                Capability.READ,
                Capability.BATCHES,
                Capability.WRITE,
                Capability.AGGREGATE if self.raw else Capability.FIT,
            ),
            requirements=source.requirements + self._missing() + source.notes,
            source=source.identity,
        )

    def open(self, scope: ExecutionScope) -> NodeStream[EvidenceMetadata]:
        if any(
            value == []
            for key, value in self.state.document().items()
            if key in {"variables", "scenarios", "periods", "series", "stations"}
        ):
            return _empty_stream(self, scope)
        operation = self._operation()
        if operation is None:
            raise CapabilityError(
                f"{self.provider.upper()} requires indicator=; "
                "inspect() returns typed choices."
            )
        if self.state.area is None:
            raise CapabilityError(
                f"{self.provider.upper()} requires "
                ".select(area=Bounds(...)) before source acquisition."
            )
        time = selected_source_time(self.state)
        if self.provider == "era5" and time is None:
            raise CapabilityError(
                "ERA5 requires explicit .select(time=Period.window(...)) bounds."
            )
        if self.provider != "era5" and (
            self.model is None or self.scenario is None or self.member is None
        ):
            raise CapabilityError(
                f"{self.provider.upper()} requires a model, scenario, and member. "
                "Select a published combination from inspect().series."
            )
        source = self._source(scope)
        if source.requirements:
            raise CapabilityError(" ".join(r.message for r in source.requirements))
        source_records = source.records
        assert source_records is not None
        window = _period(time, source.first_year, source.last_year, source.calendar)
        period = _period(
            selected_series_period(self.state) or time,
            source.first_year,
            source.last_year,
            source.calendar,
        )
        metadata = _metadata(self, source, (operation,), period)
        state, release = payload_selection(self.state, self.release)
        assert source.identity.release is not None
        pin = inventory_payload_pin(source.identity.release, release)
        resolved = replace(
            state,
            time=time,
            series=None
            if state.series is None
            else tuple(
                replace(ref, period=period)
                if ref.period is None or ref.period.calendar is None
                else ref
                for ref in state.series
            ),
        ).resolve(metadata.spec, source=source.identity)
        metadata = selected_metadata(metadata, resolved)
        current = metadata
        observation_supports: tuple[ObservationSupport, ...] = ()

        def result_metadata(
            count: int, missing: int, *, consumed: bool = False
        ) -> EvidenceMetadata:
            actual_source = source.final_identity()
            if consumed:
                assert source.release_id is not None
                actual_source = verify_payload(
                    actual_source, id=source.release_id, expected=pin
                )
            diagnostics = (
                ()
                if not missing
                else (
                    ExecutionDiagnostic(
                        "source_no_data" if self.raw else "incomplete_year",
                        cf.DiagnosticSeverity.WARNING,
                        Stage.DECODING if self.raw else Stage.REDUCTION,
                        f"{missing} native source observations are missing."
                        if self.raw
                        else f"{missing} annual source observations do not meet "
                        "the explicit completeness policy.",
                        source=actual_source.product,
                    ),
                )
            )
            return replace(
                metadata,
                source=actual_source,
                _empty_supports=observation_supports,
                report=ExecutionReport(
                    False,
                    None,
                    diagnostics,
                    ExecutionCounts(
                        requested=count,
                        produced=count,
                        no_data=missing,
                        usable_years=0 if self.raw else count - missing,
                        incomplete_years=0 if self.raw else missing,
                    ),
                    (actual_source,),
                ),
            )

        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal current, observation_supports
            records: list[dict[str, object]] = []
            count = missing = 0
            if not resolved.empty:
                start, end = cast(int, window.start), cast(int, window.end)
                if self.raw:
                    start, end = (
                        max(start, source.first_year),
                        min(end, source.last_year),
                    )
                for year in range(start, end + 1):
                    scope.emit(
                        Stage.DECODING if self.raw else Stage.REDUCTION,
                        year - cast(int, window.start),
                    )

                    def decoded() -> Iterator[dict[str, object]]:
                        with native_data(source=self.provider, scope=scope):
                            yield from source_records(
                                year, period, operation, self.coverage.fraction
                            )

                    for record in decoded():
                        # Native grid/time acquisition already applies bounds,
                        # including end-labelled intervals at next-year midnight.
                        if (
                            resolved.stations is not None
                            and (record.get("station_id") or record["support_id"])
                            not in resolved.stations
                        ):
                            continue
                        records.append(record)
                        count += 1
                        missing += record["value"] is None
                        if len(records) >= scope.runtime.batch_rows:
                            current = result_metadata(count, missing)
                            with native_data(source=self.provider, scope=scope):
                                table = evidence_table(records, metadata)
                            yield from table.to_batches(
                                max_chunksize=scope.runtime.batch_rows
                            )
                            records.clear()
                    if records:
                        current = result_metadata(count, missing)
                        with native_data(source=self.provider, scope=scope):
                            table = evidence_table(records, metadata)
                        yield from table.to_batches(
                            max_chunksize=scope.runtime.batch_rows
                        )
                        records.clear()
            if records:
                current = result_metadata(count, missing)
                with native_data(source=self.provider, scope=scope):
                    table = evidence_table(records, metadata)
                yield from table.to_batches(max_chunksize=scope.runtime.batch_rows)
            if self.raw and not count and not resolved.empty and source.supports:
                with native_data(source=self.provider, scope=scope):
                    observation_supports = tuple(
                        support
                        for support in source.supports()
                        if resolved.stations is None or support.id in resolved.stations
                    )
            current = result_metadata(count, missing, consumed=True)

        return NodeStream(
            evidence_arrow_schema(metadata),
            metadata,
            batches(),
            report=lambda: current.report,
            final_metadata=lambda: current,
            verify_before_yield=pin is not None,
        )


def _period(
    requested: Period | None, first: int, last: int, calendar: Calendar
) -> Period:
    if calendar is Calendar.GREGORIAN and first < 1583:
        raise ValueError("mixed Gregorian source dates before 1583 are unsupported")
    if requested is None:
        return Period.window(first, last, calendar=calendar)
    if requested.kind is not PeriodKind.WINDOW:
        raise ValueError("source observation time requires Period.window(...)")
    if requested.calendar is not None and requested.calendar is not calendar:
        raise ValueError(
            "selected time calendar contradicts the native source calendar"
        )
    return Period.window(
        cast(int, requested.start), cast(int, requested.end), calendar=calendar
    )


def _metadata(
    node: _ClimateNode, source: _GridSource, operations: tuple[str, ...], period: Period
) -> EvidenceMetadata:
    variables = [
        _native_variable(node.provider, operation)
        if node.raw
        else _variable(node.provider, operation)
        for operation in operations
    ]
    spec = DataSpec(
        variables,
        series=[
            SeriesSpec(
                v,
                scenario=source.scenario,
                period=period,
                probability=None
                if node.raw
                else _reference(node.provider, operation).probability,
            )
            for v, operation in zip(variables, operations, strict=True)
        ],
    )
    provenance: list[Any] = []
    if (
        node.raw
        and len(variables) == 1
        and node.provider != "era5"
        and _OPERATIONS[operations[0]].startswith("rx")
    ):
        provenance.append(
            UnitConversion(
                variable_id=variables[0].id,
                source_unit="kg m-2 s-1",
                target_unit="mm/day",
                scale=86400.0,
            )
        )
    elif not node.raw and len(variables) == 1:
        provenance.extend(
            _provenance(
                variables[0],
                source.calendar,
                node.provider == "era5",
                _OPERATIONS[operations[0]],
                node.coverage,
            )
        )
    selected_time = selected_source_time(node.state)
    if selected_time is not None:
        provenance.append(
            ObservationTimeSelection(
                time=_period(
                    selected_time, source.first_year, source.last_year, source.calendar
                )
            )
        )
    return EvidenceMetadata(
        kind=EvidenceKind.OBSERVATIONS if node.raw else EvidenceKind.SAMPLES,
        spec=spec,
        spatial=SpatialContract(supports=(SupportKind.POLYGON,), index_resolution=9),
        report=ExecutionReport(False, None, source_identities=(source.identity,)),
        source=source.identity,
        provenance=tuple(provenance),
        sampling=Sampling(
            calendar=source.calendar,
            step=cf.Duration(
                1, unit=cf.TimeUnit.HOUR if node.provider == "era5" else cf.TimeUnit.DAY
            )
            if node.raw
            else cf.Duration(1, unit=cf.TimeUnit.YEAR),
            timestamp=TimestampPosition.END
            if node.raw
            and node.provider == "era5"
            and len(operations) == 1
            and _OPERATIONS[operations[0]].startswith("rx")
            else TimestampPosition.START,
        ),
    )


def _provenance(
    variable: Variable,
    calendar: Calendar,
    hourly: bool,
    native: str,
    coverage: Coverage,
) -> tuple[Any, ...]:
    source_sampling = Sampling(
        calendar=calendar,
        step=cf.Duration(1, unit=cf.TimeUnit.HOUR if hourly else cf.TimeUnit.DAY),
        timestamp=TimestampPosition.END
        if hourly and native.startswith("rx")
        else TimestampPosition.START,
    )
    daily_sampling = Sampling(
        calendar=calendar, step=cf.Duration(1, unit=cf.TimeUnit.DAY)
    )
    yearly_sampling = Sampling(
        calendar=calendar, step=cf.Duration(1, unit=cf.TimeUnit.YEAR)
    )
    records: list[Any] = []
    if native.startswith("rx"):
        if hourly:
            raw_measurement: cf.Measurement = cf.PrecipitationAmount(
                unit=cf.LengthUnit.METRE,
                accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.HOUR),
            )
        else:
            raw_measurement = cf.PrecipitationRate(
                unit=cf.PrecipitationRateUnit.MILLIMETRES_PER_DAY,
                accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.DAY),
                aggregation=cf.Aggregation.MEAN,
            )
            records.append(
                UnitConversion(
                    variable_id=variable.id,
                    source_unit="kg m-2 s-1",
                    target_unit="mm/day",
                    scale=86400.0,
                )
            )
        daily = cf.PrecipitationAmount(
            unit=cf.LengthUnit.MILLIMETRE,
            accumulation_duration=cf.Duration(1, unit=cf.TimeUnit.DAY),
        )
        records.append(
            TemporalReduction(
                operation=ReductionKind.DAILY_TOTAL,
                coverage=Coverage.complete(),
                input_sampling=source_sampling,
                output_sampling=daily_sampling,
                measurements=(
                    MeasurementChange(
                        variable_id=variable.id, input=raw_measurement, output=daily
                    ),
                ),
            )
        )
        if native == "rx5day":
            records.append(
                TemporalReduction(
                    operation=ReductionKind.ROLLING_TOTAL,
                    coverage=Coverage.complete(),
                    input_sampling=daily_sampling,
                    output_sampling=daily_sampling,
                    window=cf.Duration(5, unit=cf.TimeUnit.DAY),
                    measurements=(
                        MeasurementChange(
                            variable_id=variable.id,
                            input=daily,
                            output=cast(cf.Measurement, variable.measurement),
                        ),
                    ),
                )
            )
    else:
        records.append(
            UnitConversion(
                variable_id=variable.id,
                source_unit="K",
                target_unit="degC",
                scale=1.0,
                offset=-273.15,
            )
        )
        if hourly:
            records.append(
                TemporalReduction(
                    operation=ReductionKind.DAILY_MINIMUM
                    if native == "tnn"
                    else ReductionKind.DAILY_MAXIMUM,
                    coverage=Coverage.complete(),
                    input_sampling=source_sampling,
                    output_sampling=daily_sampling,
                )
            )
    records.append(
        TemporalReduction(
            operation=ReductionKind.ANNUAL_MINIMUM
            if native == "tnn"
            else ReductionKind.ANNUAL_MAXIMUM,
            coverage=coverage,
            input_sampling=daily_sampling,
            output_sampling=yearly_sampling,
        )
    )
    return tuple(records)


def _empty_stream(
    node: _ClimateNode, scope: ExecutionScope
) -> NodeStream[EvidenceMetadata]:
    metadata = EvidenceMetadata(
        kind=EvidenceKind.OBSERVATIONS if node.raw else EvidenceKind.SAMPLES,
        spec=DataSpec([], series=[]),
        spatial=SpatialContract(supports=(SupportKind.POLYGON,), index_resolution=9),
        report=ExecutionReport(False, None),
    )
    return NodeStream(evidence_arrow_schema(metadata), metadata, iter(()))


@dataclass(slots=True)
class _GridSource:
    identity: SourceIdentity
    calendar: Calendar
    first_year: int
    last_year: int
    scenario: Scenario | None
    requirements: tuple[Requirement, ...]
    records: Callable[[int, Period, str, float], Iterator[dict[str, object]]] | None
    final_identity: Callable[[], SourceIdentity]
    notes: tuple[Requirement, ...] = ()
    release_id: str | None = None
    supports: Callable[[], tuple[ObservationSupport, ...]] | None = None


def _raw_grid_records(
    scope: ExecutionScope,
    *,
    array: Any,
    time_name: str,
    lat_name: str,
    lon_name: str,
    first: int,
    stop: int,
    stamps: np.ndarray[Any, Any],
    cells: tuple[GridCell, ...],
    support_prefix: str,
    variable_id: str,
    scenario_id: str | None,
    period: Period,
    scale: float = 1.0,
    calendar: cmip.CFCalendar | None = None,
) -> Iterator[dict[str, object]]:
    if not cells:
        return
    budget = _memory_bytes(scope.runtime.memory_limit)
    rectangular_cells = (max(c.row for c in cells) - min(c.row for c in cells) + 1) * (
        max(c.column for c in cells) - min(c.column for c in cells) + 1
    )
    steps = max(1, scope.runtime.batch_rows // len(cells))
    if budget is not None:
        steps = min(steps, max(1, budget // (8 * rectangular_cells)))
    for start in range(first, stop, steps):
        end = min(stop, start + steps)
        values = grid_values(
            array,
            time_name=time_name,
            lat_name=lat_name,
            lon_name=lon_name,
            start=start,
            stop=end,
            cells=cells,
            memory_limit=budget,
        )
        block_stamps: Any = stamps[start:end]
        if calendar is not None:
            block_stamps = _daily_stamps(block_stamps, calendar)
        for offset, stamp in enumerate(block_stamps):
            for column, cell in enumerate(cells):
                value = float(values[offset, column]) * scale
                valid = math.isfinite(value)
                source_row = cell.row if cell.source_row is None else cell.source_row
                source_column = (
                    cell.column if cell.source_column is None else cell.source_column
                )
                yield {
                    "variable_id": variable_id,
                    "scenario_id": scenario_id,
                    "period_id": period.id,
                    "support_id": f"{support_prefix}:{source_row}:{source_column}",
                    "cell_id": cell.cell_id,
                    "geometry": cell.geometry,
                    "time": str(stamp),
                    "value": value if valid else None,
                    "station_id": None,
                    "group_id": None,
                    "coverage": None,
                    "failure": None
                    if valid
                    else {
                        "code": "source_no_data",
                        "message": "The native source observation is missing "
                        "or non-finite.",
                    },
                }


def _daily_stamps(ordinals: Any, calendar: cmip.CFCalendar) -> tuple[str, ...]:
    years, months, days = calendar.from_ordinal(ordinals)
    return tuple(
        f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        for year, month, day in zip(years, months, days, strict=True)
    )


def _era5_source(
    scope: ExecutionScope, node: _ClimateNode, operation: str | None
) -> _GridSource:
    from ._providers.era5 import _decode_time

    snapshot = open_zarr_snapshot(
        scope,
        ERA5_STORE.url,
        license=era.ERA5_LICENCE,
        attribution=era.ERA5_ATTRIBUTION,
    )
    group = snapshot.group
    calendar = calendar_name(group[ERA5_STORE.time_name].attrs.get("calendar"))
    if calendar not in {Calendar.GREGORIAN, Calendar.PROLEPTIC_GREGORIAN}:
        raise ValueError("ERA5 time requires its native Gregorian calendar")
    time = _decode_time(group[ERA5_STORE.time_name])
    if time.size == 0 or np.any(np.diff(time) <= np.timedelta64(0, "s")):
        raise ValueError("ERA5 source time must be nonempty, unique, and increasing")
    attrs = dict(group.attrs)
    first_date = str(attrs.get("valid_time_start", str(time[0])))[:10]
    last_date = str(attrs.get("valid_time_stop", str(time[-1])))[:10]
    first, last = int(first_date[:4]), int(last_date[:4])
    release_id = "final-through-" + last_date
    resources = zarr_resources(snapshot)
    identity = SourceIdentity(
        "era5",
        ERA5_STORE.name,
        release=inventory_release(resources, id=release_id),
        resources=resources,
    )

    def final_identity() -> SourceIdentity:
        return replace(identity, resources=zarr_resources(snapshot))

    @cache
    def cells() -> tuple[GridCell, ...]:
        assert node.state.area is not None
        return grid_cells(
            group,
            lat_name=ERA5_STORE.lat_name,
            lon_name=ERA5_STORE.lon_name,
            area=node.state.area,
            fallback_resolution=ERA5_STORE.resolution_degrees,
        )

    def supports() -> tuple[ObservationSupport, ...]:
        return tuple(
            ObservationSupport(
                f"era5:{ERA5_STORE.name}:{cell.row}:{cell.column}",
                cell.cell_id,
                cell.geometry,
            )
            for cell in cells()
        )

    def records(
        year: int, period: Period, selected: str, fraction: float
    ) -> Iterator[dict[str, object]]:
        assert node.state.area is not None
        native = _OPERATIONS[selected]
        name = ERA5_STORE.variables["tp" if native.startswith("rx") else "t2m"]
        array = group[name]
        expected_unit = "m" if native.startswith("rx") else "K"
        if array.attrs.get("units") != expected_unit:
            raise ValueError(
                f"ERA5 {name} must declare its actual source unit {expected_unit!r}"
            )
        grid = cells()
        start = np.datetime64(f"{year}-01-01T00:00:00", "s")
        stop = np.datetime64(f"{year + 1}-01-01T00:00:00", "s")
        interval_end = native.startswith("rx")
        mask = (
            (time > start) & (time <= stop)
            if interval_end
            else (time >= start) & (time < stop)
        )
        mask &= (time >= np.datetime64(first_date, "s")) & (
            time <= np.datetime64(last_date, "s") + np.timedelta64(1, "D")
        )
        indices = np.flatnonzero(mask)
        if node.raw and not indices.size:
            return
        if indices.size:
            t0, t1 = int(indices[0]), int(indices[-1]) + 1
            if np.any(
                time[t0:t1].astype("datetime64[h]").astype("datetime64[s]")
                != time[t0:t1]
            ):
                raise ValueError(
                    "ERA5 native source timestamps must identify hourly intervals"
                )
            if node.raw:
                yield from _raw_grid_records(
                    scope,
                    array=array,
                    time_name=ERA5_STORE.time_name,
                    lat_name=ERA5_STORE.lat_name,
                    lon_name=ERA5_STORE.lon_name,
                    first=t0,
                    stop=t1,
                    stamps=time,
                    cells=grid,
                    support_prefix=f"era5:{ERA5_STORE.name}",
                    variable_id=_native_variable("era5", selected).id,
                    scenario_id=None,
                    period=period,
                )
                return
            values = grid_values(
                array,
                time_name=ERA5_STORE.time_name,
                lat_name=ERA5_STORE.lat_name,
                lon_name=ERA5_STORE.lon_name,
                start=t0,
                stop=t1,
                cells=grid,
                memory_limit=_memory_bytes(scope.runtime.memory_limit),
            )
            values = values * 1000 if interval_end else values - 273.15
            hours = time[t0:t1].astype("datetime64[h]").astype(np.int64)
        else:
            values = np.empty((0, len(grid)), dtype=np.float64)
            hours = np.empty(0, dtype=np.int64)
        yield from annual_records(
            times=hours,
            values=values,
            cells=grid,
            calendar=calendar,
            selected=Period.window(year, year, calendar=calendar),
            native=native,
            hourly=True,
            interval_end=interval_end,
            support_prefix=f"era5:{ERA5_STORE.name}",
            variable_id=f"era5:{selected}",
            scenario_id=None,
            minimum_fraction=fraction,
        )

    def rewritten(
        year: int, period: Period, selected: str, fraction: float
    ) -> Iterator[dict[str, object]]:
        for record in records(year, period, selected, fraction):
            record["period_id"] = period.id
            yield record

    native = None if operation is None else _OPERATIONS[operation]
    selected_time = selected_source_time(node.state)
    notes: tuple[Requirement, ...] = ()
    if native is not None and selected_time is not None:
        variable_name = ERA5_STORE.variables["tp" if native.startswith("rx") else "t2m"]
        array = group[variable_name]
        start = np.datetime64(f"{selected_time.start}-01-01", "s")
        stop = np.datetime64(f"{cast(int, selected_time.end) + 1}-01-01", "s")
        hours = int(np.count_nonzero((time >= start) & (time < stop)))
        chunks = math.ceil(hours / int(array.chunks[0]))
        native_chunk_bytes = int(np.prod(array.chunks)) * array.dtype.itemsize
        notes = (
            Requirement(
                "source_chunk_layout",
                f"Published ERA5 chunks have shape {array.chunks!r}; "
                f"this time window can require {chunks} chunks "
                f"({chunks * native_chunk_bytes} uncompressed bytes) "
                "even for one pixel. Compressed transfer size depends "
                "on the source content.",
            ),
        )
    return _GridSource(
        identity,
        calendar,
        first,
        last,
        None,
        (),
        rewritten,
        final_identity,
        notes,
        release_id=release_id,
        supports=supports,
    )


def _cmip_datasets(
    scope: ExecutionScope, node: _ClimateNode
) -> tuple[list[cmip.CMIP6Dataset], ResourceIdentity]:
    path, catalogue = fetch(scope, CMIP6_CATALOGUE)
    variables = _discovery_variables(node)
    scenario = (
        node.scenario.id if isinstance(node.scenario, Scenario) else node.scenario
    )
    datasets = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("table_id") != "day" or row.get("variable_id") not in variables:
                continue
            if any(
                value is not None and row.get(name) != value
                for name, value in (
                    ("source_id", node.model),
                    ("experiment_id", scenario),
                    ("member_id", node.member),
                )
            ):
                continue
            datasets.append(
                cmip.CMIP6Dataset(*(row[name] for name in cmip.CATALOGUE_COLUMNS))
            )
    return datasets, catalogue


def _cmip_source(
    scope: ExecutionScope, node: _ClimateNode, *, release: ReleaseReference | None
) -> _GridSource:
    datasets, catalogue = _cmip_datasets(scope, node)
    if not datasets:
        raise LookupError(
            "No published CMIP6 dataset matches the exact source selection"
        )
    matches = [
        d
        for d in datasets
        if release is None or d.dataset_id == release.id.removeprefix("inventory:")
    ]
    if not matches:
        raise LookupError("The pinned CMIP6 release is not in the supported catalogue")
    dataset = _preferred_cmip_dataset(matches)
    if dataset is None:
        raise ValueError(
            "CMIP6 selection has no unique publication on a supported grid"
        )
    record = _cmip_record(dataset, catalogue)
    publication = inventory_release((record,), id=dataset.dataset_id)
    inventory_payload_pin(publication, release)
    snapshot = open_zarr_snapshot(scope, dataset.zstore)
    group = snapshot.group
    calendar = calendar_name(group["time"].attrs.get("calendar"))
    cal = cf_calendar(calendar)
    ordinals = cmip.decode_time_ordinals(
        np.asarray(group["time"][:]), str(group["time"].attrs.get("units", "")), cal
    )
    if ordinals.size == 0 or np.any(np.diff(ordinals) <= 0):
        raise ValueError("CMIP6 source days must be nonempty, unique, and increasing")
    years, _, _ = cal.from_ordinal(ordinals[[0, -1]])
    licence = group.attrs.get("license") or group.attrs.get("licence")
    attribution = (
        str(group.attrs.get("institution", dataset.institution_id))
        + "; CMIP6 modelling centre data published through the Pangeo catalogue"
    )

    def actual_resources() -> tuple[ResourceIdentity, ...]:
        return (
            catalogue,
            record,
            *(
                replace(
                    r,
                    license=None if licence is None else str(licence),
                    attribution=attribution,
                )
                for r in zarr_resources(snapshot)
            ),
        )

    identity = SourceIdentity(
        "cmip6",
        dataset.dataset_id,
        release=publication,
        resources=actual_resources(),
        member=dataset.member_id,
        model=dataset.source_id,
    )
    actual_scenario = (
        node.scenario
        if isinstance(node.scenario, Scenario)
        else Scenario(dataset.experiment_id)
    )

    def records(
        year: int, period: Period, selected: str, fraction: float
    ) -> Iterator[dict[str, object]]:
        assert node.state.area is not None
        native = _OPERATIONS[selected]
        recipe = cmip.INDICATOR_VARIABLES[native]
        array = group[recipe.variable]
        if array.attrs.get("units") != recipe.source_unit:
            raise ValueError(
                f"CMIP6 {recipe.variable} source units contradict the named indicator"
            )
        cells = grid_cells(group, lat_name="lat", lon_name="lon", area=node.state.area)
        first, stop = cal.to_ordinal(year, 1, 1), cal.to_ordinal(year + 1, 1, 1)
        t0, t1 = (int(x) for x in np.searchsorted(ordinals, [first, stop]))
        if node.raw:
            yield from _raw_grid_records(
                scope,
                array=array,
                time_name="time",
                lat_name="lat",
                lon_name="lon",
                first=t0,
                stop=t1,
                stamps=ordinals,
                calendar=cal,
                cells=cells,
                support_prefix=f"cmip6:{dataset.source_id}:{dataset.grid_label}",
                variable_id=_native_variable("cmip6", selected).id,
                scenario_id=actual_scenario.id,
                period=period,
                scale=86400.0 if native.startswith("rx") else 1.0,
            )
            return
        values = grid_values(
            array,
            time_name="time",
            lat_name="lat",
            lon_name="lon",
            start=t0,
            stop=t1,
            cells=cells,
            memory_limit=_memory_bytes(scope.runtime.memory_limit),
        )
        values = values * 86400 if native.startswith("rx") else values - 273.15
        for record in annual_records(
            times=ordinals[t0:t1],
            values=values,
            cells=cells,
            calendar=calendar,
            selected=Period.window(year, year, calendar=calendar),
            native=native,
            hourly=False,
            interval_end=False,
            support_prefix=f"cmip6:{dataset.source_id}:{dataset.grid_label}",
            variable_id=f"cmip6:{selected}",
            scenario_id=actual_scenario.id,
            minimum_fraction=fraction,
        ):
            record["period_id"] = period.id
            yield record

    return _GridSource(
        identity,
        calendar,
        int(years[0]),
        int(years[1]),
        actual_scenario,
        (),
        records,
        lambda: replace(identity, resources=actual_resources()),
        release_id=dataset.dataset_id,
    )


def _cmip_record(
    dataset: cmip.CMIP6Dataset, catalogue: ResourceIdentity
) -> ResourceIdentity:
    payload = json.dumps(
        {name: getattr(dataset, name) for name in dataset.__dataclass_fields__},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return ResourceIdentity(
        catalogue.uri + "#" + dataset.dataset_id,
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        retrieved_at=catalogue.retrieved_at,
    )


def _nex_listing(
    scope: ExecutionScope, prefix: str, *, recursive: bool = False
) -> tuple[dict[str, str | None], tuple[ResourceIdentity, ...]]:
    objects: dict[str, str | None] = {}
    resources = []
    token: str | None = None
    while True:
        uri = NEX_BUCKET + "/?list-type=2&prefix=" + quote(prefix, safe="")
        if not recursive:
            uri += "&delimiter=/"
        if token is not None:
            uri += "&continuation-token=" + quote(token, safe="")
        path, resource = fetch(scope, uri)
        resources.append(resource)
        tree = ET.fromstring(path.read_bytes())
        for content in tree.findall(".//{*}Contents"):
            key = content.findtext("{*}Key")
            if key is not None:
                etag = content.findtext("{*}ETag")
                if objects.setdefault(key, etag) != etag:
                    raise ValueError(
                        "NEX inventory describes conflicting object revisions"
                    )
        token = tree.findtext("{*}NextContinuationToken")
        if token is None:
            return objects, tuple(resources)


def _nex_files(objects: dict[str, str | None]) -> dict[int, tuple[str, str | None]]:
    versions: dict[int, tuple[int, ...]] = {}
    files: dict[int, tuple[str, str | None]] = {}
    for key, etag in objects.items():
        match = _NEX_YEAR.search(key)
        if match is None:
            continue
        year = int(match[1])
        version = tuple(map(int, (match[2] or "1.0").split(".")))
        if year not in versions or version > versions[year]:
            versions[year] = version
            files[year] = (key, etag)
        elif version == versions[year] and files[year][0] != key:
            raise ValueError("NEX inventory describes ambiguous annual publications")
    return files


def _nex_release(files: dict[int, tuple[str, str | None]], id: str) -> ReleaseReference:
    identity = ResourceIdentity(
        NEX_BUCKET + "/" + NEX_PREFIX + id + "/",
        "sha256:"
        + hashlib.sha256(
            json.dumps(sorted(files.items()), separators=(",", ":")).encode()
        ).hexdigest(),
    )
    return inventory_release((identity,), id=id)


def _discovery_variables(node: _ClimateNode) -> dict[str, list[str]]:
    operation = node._operation()
    by_variable: dict[str, list[str]] = {}
    for candidate, native in _NATIVE_VARIABLES.items():
        if operation is None or candidate == operation:
            by_variable.setdefault(native, []).append(candidate)
    return by_variable


def _preferred_cmip_dataset(
    datasets: list[cmip.CMIP6Dataset],
) -> cmip.CMIP6Dataset | None:
    latest = max(d.version for d in datasets)
    newest = {d for d in datasets if d.version == latest}
    for grid in cmip.GRID_PREFERENCE:
        matches = {d for d in newest if d.grid_label == grid}
        if matches:
            return matches.pop() if len(matches) == 1 else None
    return None


def _discover_climate(scope: ExecutionScope, node: _ClimateNode) -> Inspection:
    supported = _discovery_variables(node)
    _, requested = payload_selection(node.state, node.release)
    requested_id = (
        None if requested is None else requested.id.removeprefix("inventory:")
    )
    if node.provider == "cmip6":
        datasets, catalogue = _cmip_datasets(scope, node)
        by_series: dict[tuple[str, str, str, str], list[cmip.CMIP6Dataset]] = {}
        for d in datasets:
            if requested_id is not None and d.dataset_id != requested_id:
                continue
            key = (d.source_id, d.experiment_id, d.member_id, d.variable_id)
            by_series.setdefault(key, []).append(d)
        releases = {
            key: inventory_release(
                (_cmip_record(dataset, catalogue),), id=dataset.dataset_id
            )
            for key, group in by_series.items()
            if (dataset := _preferred_cmip_dataset(group)) is not None
        }
        source = SourceIdentity("cmip6", "catalogue", resources=(catalogue,))
    else:
        available, resources = _nex_catalogue(scope, node, supported)
        releases = {
            key: release
            for key, release in available.items()
            if requested_id is None or "/".join(key) == requested_id
        }
        source = SourceIdentity("nex", "NEX-GDDP-CMIP6", resources=resources)
    if requested is not None:
        if not releases:
            raise OSError(
                f"source release changed: {requested.id} "
                "is not in the published inventory"
            )
        for release in releases.values():
            inventory_payload_pin(release, requested)
    references = tuple(
        SeriesReference(
            _native_variable(node.provider, operation)
            if node.raw
            else _variable(node.provider, operation),
            scenario=node.scenario
            if isinstance(node.scenario, Scenario)
            else Scenario(scenario),
            probability=None
            if node.raw
            else _reference(node.provider, operation).probability,
            model=model,
            member=member,
            release=release,
        )
        for (model, scenario, member, variable), release in sorted(releases.items())
        for operation in supported[variable]
    )
    available_operations = {
        operation for *_, variable in releases for operation in supported[variable]
    }
    return Inspection(
        series=references,
        releases=tuple(dict.fromkeys(r for _, r in sorted(releases.items()))),
        indicators=tuple(
            _reference(node.provider, operation)
            for operation in _OPERATIONS
            if operation in available_operations
        ),
        capabilities=(Capability.INSPECT,),
        requirements=node.requirements(),
        source=source,
    )


def _nex_catalogue(
    scope: ExecutionScope, node: _ClimateNode, supported: dict[str, list[str]]
) -> tuple[
    dict[tuple[str, str, str, str], ReleaseReference], tuple[ResourceIdentity, ...]
]:
    scenario = (
        node.scenario.id if isinstance(node.scenario, Scenario) else node.scenario
    )
    selection = (node.model, scenario, node.member)
    prefix = NEX_PREFIX
    for component in selection:
        if component is None:
            break
        prefix += component + "/"
    objects, resources = _nex_listing(scope, prefix, recursive=True)
    entries: dict[tuple[str, str, str, str], dict[str, str | None]] = {}
    for key, etag in objects.items():
        if not key.startswith(NEX_PREFIX):
            continue
        parts = key.removeprefix(NEX_PREFIX).split("/")
        if len(parts) != 5:
            continue
        model, experiment, member, variable, filename = parts
        if variable not in supported or _NEX_YEAR.search(filename) is None:
            continue
        if all(
            selected is None or selected == actual
            for selected, actual in zip(selection, parts[:3], strict=True)
        ):
            entries.setdefault((model, experiment, member, variable), {})[key] = etag
    return {
        key: _nex_release(_nex_files(objects), "/".join(key))
        for key, objects in entries.items()
    }, resources


def _nex_source(
    scope: ExecutionScope, node: _ClimateNode, operation: str
) -> _GridSource:
    scenario = (
        node.scenario.id if isinstance(node.scenario, Scenario) else node.scenario
    )
    assert node.model is not None and scenario is not None and node.member is not None
    variable = _NATIVE_VARIABLES[operation]
    release_id = f"{node.model}/{scenario}/{node.member}/{variable}"
    prefix = f"{NEX_PREFIX}{release_id}/"
    objects, identities = _nex_listing(scope, prefix)
    resources = list(identities)
    files = _nex_files(
        {
            key: etag
            for key, etag in objects.items()
            if key.startswith(prefix) and "/" not in key.removeprefix(prefix)
        }
    )
    if not files:
        raise LookupError("The selected NEX source contains no annual NetCDF resources")
    publication = _nex_release(files, release_id)
    _, requested = payload_selection(node.state, node.release)
    inventory_payload_pin(publication, requested)
    header, header_resource = netcdf_header_snapshot(
        scope, NEX_BUCKET + "/" + files[min(files)][0], variable
    )
    resources.append(header_resource)
    calendar = calendar_name(header.get("calendar"))
    if calendar not in {Calendar.GREGORIAN, Calendar.PROLEPTIC_GREGORIAN}:
        raise ValueError("NEX source does not use its published Gregorian calendar")
    actual_scenario = (
        node.scenario if isinstance(node.scenario, Scenario) else Scenario(scenario)
    )
    identity = SourceIdentity(
        "nex",
        "NEX-GDDP-CMIP6",
        release=publication,
        resources=tuple(resources),
        member=node.member,
        model=node.model,
    )

    def records(
        year: int, period: Period, selected: str, fraction: float
    ) -> Iterator[dict[str, object]]:
        if node.raw and year not in files:
            return
        source_year = year if year in files else min(files, key=lambda y: abs(y - year))
        assert node.state.area is not None
        group, resource = netcdf_window_snapshot(
            scope,
            NEX_BUCKET + "/" + files[source_year][0],
            area=node.state.area,
            variable=variable,
            resolution=0.25,
            attribution="NASA Earth Exchange NEX-GDDP-CMIP6",
        )
        resources.append(resource)
        actual_calendar = calendar_name(group["time"].attrs.get("calendar"))
        if actual_calendar is not calendar:
            raise ValueError(
                "NEX publication calendar contradicts its declared Gregorian recipe"
            )
        native = _OPERATIONS[selected]
        recipe = cmip.INDICATOR_VARIABLES[native]
        array = group[variable]
        if array.attrs.get("units") != recipe.source_unit:
            raise ValueError("NEX source units contradict its named indicator")
        licence = group.attrs.get("license") or group.attrs.get("licence")
        if licence is not None:
            resources[-1] = replace(
                resource,
                license=str(licence),
                attribution="NASA Earth Exchange NEX-GDDP-CMIP6",
            )
        cal = cf_calendar(calendar)
        ordinals = cmip.decode_time_ordinals(
            np.asarray(group["time"][:]),
            str(group["time"].attrs.get("units", "")),
            cal,
        )
        if np.any(np.diff(ordinals) <= 0):
            raise ValueError("NEX source days must be unique and increasing")
        assert node.state.area is not None
        cells = grid_cells(
            group,
            lat_name="lat",
            lon_name="lon",
            area=node.state.area,
            fallback_resolution=0.25,
        )
        if node.raw:
            first, stop = cal.to_ordinal(year, 1, 1), cal.to_ordinal(year + 1, 1, 1)
            t0, t1 = (int(x) for x in np.searchsorted(ordinals, [first, stop]))
            yield from _raw_grid_records(
                scope,
                array=array,
                time_name="time",
                lat_name="lat",
                lon_name="lon",
                first=t0,
                stop=t1,
                stamps=ordinals,
                calendar=cal,
                cells=cells,
                support_prefix="nex:0p25",
                variable_id=_native_variable("nex", selected).id,
                scenario_id=actual_scenario.id,
                period=period,
                scale=86400.0 if native.startswith("rx") else 1.0,
            )
            return
        values = grid_values(
            array,
            time_name="time",
            lat_name="lat",
            lon_name="lon",
            start=0,
            stop=len(ordinals),
            cells=cells,
            memory_limit=_memory_bytes(scope.runtime.memory_limit),
        )
        values = values * 86400 if native.startswith("rx") else values - 273.15
        if year not in files:
            ordinals, values = ordinals[:0], values[:0]
        for record in annual_records(
            times=ordinals,
            values=values,
            cells=cells,
            calendar=calendar,
            selected=Period.window(year, year, calendar=calendar),
            native=native,
            hourly=False,
            interval_end=False,
            support_prefix="nex:0p25",
            variable_id=f"nex:{selected}",
            scenario_id=actual_scenario.id,
            minimum_fraction=fraction,
        ):
            record["period_id"] = period.id
            yield record

    return _GridSource(
        identity,
        calendar,
        min(files),
        max(files),
        actual_scenario,
        (),
        records,
        lambda: replace(
            identity, resources=tuple({r.uri: r for r in resources}.values())
        ),
        release_id=release_id,
    )
