"""Catalogue-driven access to canonical CRC releases, without bucket listing."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crc_sdk.types import ProbabilitySemantics

FIXTURE_RELEASE = "ssp585-fixture-2026-10-08-v2"
DOCS_FIXTURES = (
    "https://raw.githubusercontent.com/RiskThinking/crc-docs/main/fixtures/crc_open"
)


class CRCOpenFixtureWarning(UserWarning):
    """The selected fixture has sampled geographic coverage."""


class CRCPartition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    h3_r0: str
    sha256: str
    size_bytes: int = Field(gt=0)
    rows: int = Field(gt=0)
    cells: tuple[int, ...] | None = None
    horizons: tuple[int, ...] = ()
    curve_kinds: dict[str, int] = Field(default_factory=dict)
    source_generation: str | None = None
    source_sha256: str | None = None

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))
            or "\\" in value
            or not re.fullmatch(r"[A-Za-z0-9_./=\-]+", value)
            or not value.endswith(".parquet")
        ):
            raise ValueError("partition path must be a safe relative Parquet path")
        return value

    @field_validator("sha256", "source_sha256")
    @classmethod
    def valid_digest(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("digest must be a lowercase SHA256")
        return value

    @field_validator("h3_r0")
    @classmethod
    def valid_parent(cls, value: str) -> str:
        if not re.fullmatch(r"80[0-9a-f]{2}fffffffffff", value):
            raise ValueError("h3_r0 must be a resolution-zero H3 address")
        return value


class CRCHazard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: str = Field(min_length=1)
    value_semantics: str = Field(min_length=1)
    tail: Literal["upper", "lower"]
    h3_resolution: int = Field(ge=0, le=15)
    probability_semantics: ProbabilitySemantics | None
    notes: str | None = None
    pooling: Literal["single_member", "pooled", "unknown"]
    pathways: tuple[str, ...]
    horizons: tuple[int, ...]
    partitions: tuple[CRCPartition, ...]


class CRCCatalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_version: int = 1
    release: str
    schema_version: str
    fixture: bool = False
    pathways: tuple[str, ...]
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    retrieved_at: str
    sites: tuple[dict[str, Any], ...] = ()
    hazards: dict[str, CRCHazard]
    no_data_reasons: dict[str, str] = Field(default_factory=dict)
    excluded_hazards: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> CRCCatalog:
        if self.catalog_version != 1 or self.schema_version not in {"1.2", "1.3"}:
            raise ValueError("unsupported CRC catalogue or canonical schema version")
        if set(self.hazards) & set(self.excluded_hazards):
            raise ValueError("a hazard cannot be both published and excluded")
        paths = []
        for name, hazard in self.hazards.items():
            if not set(hazard.pathways).issubset(self.pathways):
                raise ValueError("hazard pathways must belong to the release")
            if not hazard.partitions or not hazard.horizons:
                raise ValueError("hazards must declare partitions and horizons")
            for partition in hazard.partitions:
                if not partition.path.startswith(f"{name}/h3_r0={partition.h3_r0}/"):
                    raise ValueError("partition path disagrees with hazard or r0")
                paths.append(partition.path)
        if not self.hazards or len(paths) != len(set(paths)):
            raise ValueError("catalogue must have hazards and unique partition paths")
        return self


class CRCOpenHazards:
    """Read a pinned release from an HTTP root or a local directory.

    ``source`` contains release directories; it is never a private GCS default.
    """

    def __init__(self, source: str | Path, release: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", release):
            raise ValueError("release must be a pinned, safe directory name")
        self.source = str(source).rstrip("/")
        self.release = release
        parsed = urlparse(self.source)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            raise ValueError("CRC source must be an HTTP(S) root or local directory")

    def fetch(self, relative: str) -> bytes:
        location = f"{self.source}/{self.release}/{relative}"
        if urlparse(self.source).scheme in {"http", "https"}:
            with urlopen(location, timeout=60) as response:
                return response.read()  # type: ignore[no-any-return]
        return Path(location).read_bytes()

    def catalog_bytes(self) -> bytes:
        payload = self.fetch("_CATALOG.json")
        expected = self.fetch("_SUCCESS").decode().strip()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("CRC catalogue _SUCCESS checksum mismatch")
        self.parse_catalog(payload)
        return payload

    def parse_catalog(self, payload: bytes) -> CRCCatalog:
        catalog = CRCCatalog.model_validate(json.loads(payload))
        if catalog.release != self.release:
            raise ValueError(
                "catalogue release does not match requested pinned release"
            )
        return catalog

    def partition_bytes(self, partition: CRCPartition) -> bytes:
        payload = self.fetch(partition.path)
        if (
            len(payload) != partition.size_bytes
            or hashlib.sha256(payload).hexdigest() != partition.sha256
        ):
            raise ValueError(f"CRC partition checksum/size mismatch: {partition.path}")
        return payload
