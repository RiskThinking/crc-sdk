"""Typed inspection values shared by scientific recipes and execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, Union

import crc_framework
import pyarrow as pa  # type: ignore[import-untyped]

import crc_sdk

from ._dataset_schema import ReleaseReference, SourceIdentity, Stage, _identifier, _text
from ._declarations import Period, Probability, Scenario, Variable
from ._runtime import Cache

ArtifactMetadata: TypeAlias = Union[
    "crc_sdk.DatasetMetadata",
    "crc_sdk.EvidenceMetadata",
    "crc_sdk.CollectionMetadata",
    "crc_sdk.EvaluationMetadata",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Bounds:
    """Inclusive WGS84 bounds; west greater than east crosses the antimeridian."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        for name in ("west", "south", "east", "north"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if not -180 <= self.west <= 180 or not -180 <= self.east <= 180:
            raise ValueError("west and east must lie within [-180, 180]")
        if not -90 <= self.south <= self.north <= 90:
            raise ValueError("south and north must be ordered within [-90, 90]")


class Capability(Enum):
    INSPECT = "inspect"
    READ = "read"
    BATCHES = "batches"
    WRITE = "write"
    FIT = "fit"
    AGGREGATE = "aggregate"
    EVALUATE = "evaluate"


@dataclass(frozen=True, slots=True)
class Requirement:
    code: str
    message: str
    parameter: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.code, "requirement.code")
        _text(self.message, "requirement.message")
        if self.parameter is not None:
            _identifier(self.parameter, "requirement.parameter")


@dataclass(frozen=True, slots=True)
class Operation:
    stage: Stage
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage):
            raise TypeError("operation stage must be Stage")
        _text(self.description, "operation.description")


@dataclass(frozen=True, slots=True)
class SeriesReference:
    variable: Variable
    scenario: Scenario | None = None
    period: Period | None = None
    release: ReleaseReference | None = None
    probability: Probability | None = None
    model: str | None = None
    member: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variable, Variable):
            raise TypeError("series variable must be Variable")
        for name in ("model", "member"):
            value = getattr(self, name)
            if value is not None:
                _identifier(value, f"series {name}")
        for name, expected in (
            ("scenario", Scenario),
            ("period", Period),
            ("release", ReleaseReference),
            ("probability", Probability),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"series {name} must be {expected.__name__} or None")

    @property
    def measurement(self) -> crc_framework.Measurement | None:
        return self.variable.measurement


@dataclass(frozen=True, slots=True)
class IndicatorReference:
    id: str
    label: str
    measurement: crc_framework.Measurement | None = None
    probability: Probability | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "indicator.id")
        _text(self.label, "indicator.label")
        if self.measurement is not None and not isinstance(
            self.measurement, crc_framework.Measurement
        ):
            raise TypeError("indicator measurement must be Measurement or None")
        if self.probability is not None and not isinstance(
            self.probability, Probability
        ):
            raise TypeError("indicator probability must be Probability or None")


@dataclass(frozen=True, slots=True)
class StationReference:
    id: str
    label: str
    longitude: float | None = None
    latitude: float | None = None
    period: Period | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "station.id")
        _text(self.label, "station.label")
        if (self.longitude is None) != (self.latitude is None):
            raise ValueError("station longitude and latitude must be supplied together")
        if self.longitude is not None:
            assert self.latitude is not None
            Bounds(
                west=self.longitude,
                east=self.longitude,
                south=self.latitude,
                north=self.latitude,
            )
        if self.period is not None and not isinstance(self.period, Period):
            raise TypeError("station period must be Period or None")


@dataclass(frozen=True, slots=True)
class Inspection:
    schema: pa.Schema | None = None
    series: tuple[SeriesReference, ...] = ()
    indicators: tuple[IndicatorReference, ...] = ()
    releases: tuple[ReleaseReference, ...] = ()
    capabilities: tuple[Capability, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    source: SourceIdentity | None = None
    stations: tuple[StationReference, ...] = ()

    @property
    def metadata(self) -> ArtifactMetadata | None:
        """Typed science, sampling, and transformations declared by the source."""
        from ._dataset_schema import DatasetMetadata
        from ._evidence import EvidenceMetadata

        if self.schema is None:
            return None
        headers = self.schema.metadata or {}
        if b"crc.dataset" in headers:
            return DatasetMetadata.from_json(headers[b"crc.dataset"].decode())
        if b"crc.evidence" in headers:
            return EvidenceMetadata.from_json(headers[b"crc.evidence"].decode())
        if b"crc.collection" in headers:
            from ._collection_schema import CollectionMetadata

            return CollectionMetadata.from_json(headers[b"crc.collection"].decode())
        if b"crc.evaluation" in headers:
            from ._evaluation_schema import evaluation_metadata_from_schema

            return evaluation_metadata_from_schema(self.schema)
        return None

    def __post_init__(self) -> None:
        if self.schema is not None and not isinstance(self.schema, pa.Schema):
            raise TypeError("inspection schema must be an Arrow Schema or None")
        for name, expected in (
            ("series", SeriesReference),
            ("indicators", IndicatorReference),
            ("releases", ReleaseReference),
            ("capabilities", Capability),
            ("requirements", Requirement),
            ("stations", StationReference),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(
                isinstance(value, expected) for value in values
            ):
                raise TypeError(f"inspection {name} must be an immutable typed tuple")
        if self.source is not None and not isinstance(self.source, SourceIdentity):
            raise TypeError("inspection source must be SourceIdentity or None")


@dataclass(frozen=True, slots=True)
class Explanation:
    operations: tuple[Operation, ...]
    requirements: tuple[Requirement, ...]
    cache: Cache
    source: SourceIdentity | None = None
    workers: int = 1
    batch_rows: int = 65536
    memory_limit: str | int | None = None

    def __post_init__(self) -> None:
        from ._runtime import Cache

        for name, expected in (
            ("operations", Operation),
            ("requirements", Requirement),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(
                isinstance(value, expected) for value in values
            ):
                raise TypeError(f"explanation {name} must be an immutable typed tuple")
        if not isinstance(self.cache, Cache):
            raise TypeError("explanation cache must be Cache")
        if self.source is not None and not isinstance(self.source, SourceIdentity):
            raise TypeError("explanation source must be SourceIdentity or None")
        for name in ("workers", "batch_rows"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"explanation {name} must be a positive integer")
        if self.memory_limit is not None:
            if isinstance(self.memory_limit, bool) or not isinstance(
                self.memory_limit, (str, int)
            ):
                raise TypeError(
                    "explanation memory_limit must be a string, integer or None"
                )
            if isinstance(self.memory_limit, int) and self.memory_limit <= 0:
                raise ValueError("explanation memory_limit must be positive")
            if isinstance(self.memory_limit, str) and not self.memory_limit:
                raise ValueError("explanation memory_limit must be nonempty")
