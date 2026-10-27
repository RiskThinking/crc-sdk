"""Lazy, point-supported ECCC rainfall intensity-duration-frequency curves."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from crc_sdk.connectors.adapters import (
    CurveFitIngestPolicy,
    CurveSourceInfo,
    canonicalize_curve_source,
)
from crc_sdk.connectors.duckdb.zarr import RasterCurve, RasterMetadata
from crc_sdk.connectors.parquet import write_hazard_stream
from crc_sdk.providers import eccc_idf as provider
from crc_sdk.types import TemporalWindow

from ._remote import (
    Bounds,
    CacheMode,
    MaterializationResult,
    PrefetchResult,
    ProgressCallback,
    RemotePortfolioEvaluation,
    file_checksum,
    read_manifest,
    validate_bounds,
    write_manifest,
)
from .portfolio import AssetPortfolio, HazardDataset


class _IDFCurveSource:
    axis_name = "return period (years)"

    def __init__(
        self,
        records: list[dict[str, Any]],
        duration: str,
        metadata: RasterMetadata,
        info: CurveSourceInfo,
    ) -> None:
        self.records = records
        self.duration = duration
        self.metadata = metadata
        self.info = info
        self.return_period_support = (
            max(record["return_periods"][0] for record in records),
            min(record["return_periods"][-1] for record in records),
        )

    def iter_curves(self, bounds: Bounds | None = None) -> Iterator[RasterCurve]:
        for index, record in enumerate(self.records):
            curve = next(
                c
                for c in record["curves"]
                if c["duration_minutes"] == provider.DURATIONS[self.duration]
            )
            point = (record["longitude"], record["latitude"])
            yield RasterCurve(
                row=index,
                column=0,
                boundary=(point,) * 4,
                axis_values=np.asarray(record["return_periods"], dtype=float),
                values=np.asarray(curve["intensity"], dtype=float),
                point=point,
                source_id=f"eccc-idf:{record['station']}",
                horizon=(record["start_year"] + record["end_year"]) // 2,
            )


@dataclass(frozen=True)
class ECCCIDFPlan:
    """One rainfall duration, with explicit release and provincial archive scope.

    Stations retain point support and their own record midpoints as horizons.
    ``source_curves()`` exposes official rates and confidence limits separately
    from the derived canonical fits. Builders and ``explain()`` perform no I/O.
    """

    duration_name: str
    province: str
    version: str = "3.40"
    selected_stations: tuple[str, ...] = ()
    requested_areas: tuple[Bounds, ...] = ()
    cache_dir: Path | None = None
    cache_mode: CacheMode = "stream"
    fit_policy: CurveFitIngestPolicy = CurveFitIngestPolicy(
        h3_resolution=5,
        family="gumbel_r",
        producer="crc-sdk",
        maximum_absolute_residual=0.15,
    )

    def __post_init__(self) -> None:
        provider.archive_url(self.version, self.province)
        if self.duration_name not in provider.DURATIONS:
            raise ValueError(f"duration must be one of {list(provider.DURATIONS)}")

    def duration(self, value: str) -> ECCCIDFPlan:
        """Select another published duration, reusing the same raw station cache."""
        return replace(self, duration_name=value)

    def for_stations(self, stations: Sequence[str]) -> ECCCIDFPlan:
        if isinstance(stations, str):
            raise TypeError("stations must be a sequence of ECCC climate IDs")
        ids = tuple(stations)
        if not ids or any(not re.fullmatch(r"[A-Z0-9]{7}", s) for s in ids):
            raise ValueError("stations must contain seven-character ECCC climate IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("station IDs must be unique")
        return replace(self, selected_stations=tuple(sorted(ids)), requested_areas=())

    def for_area(self, bounds: Sequence[float]) -> ECCCIDFPlan:
        """Select station locations within a WGS84 box in the chosen province."""
        return self.for_areas([bounds])

    def for_areas(self, bounds: Sequence[Sequence[float]]) -> ECCCIDFPlan:
        """Select the union of WGS84 boxes within the chosen province."""
        areas = tuple(sorted({validate_bounds(area) for area in bounds}))
        if not areas:
            raise ValueError("for_areas requires at least one bounding box")
        return replace(self, requested_areas=areas, selected_stations=())

    def cache(
        self, directory: str | Path | None, *, mode: CacheMode = "reuse"
    ) -> ECCCIDFPlan:
        if mode not in ("reuse", "offline", "refresh", "stream"):
            raise ValueError("invalid ECCC cache mode")
        if (mode == "stream") != (directory is None):
            raise ValueError("only stream mode requires directory=None")
        return replace(
            self, cache_dir=Path(directory) if directory else None, cache_mode=mode
        )

    def canonicalize(
        self, *, policy: Literal["curated"] | CurveFitIngestPolicy = "curated"
    ) -> ECCCIDFPlan:
        if policy == "curated":
            policy = ECCCIDFPlan(
                self.duration_name, self.province, self.version
            ).fit_policy
        if not isinstance(policy, CurveFitIngestPolicy):
            raise TypeError("policy must be 'curated' or CurveFitIngestPolicy")
        if policy.tail != "upper" or policy.fit_method != "quantile_least_squares":
            raise ValueError(
                "IDF central rates require upper-tail quantile least squares"
            )
        return replace(self, fit_policy=policy)

    def _request(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "province": self.province,
            "stations": self.selected_stations,
            "areas": self.requested_areas,
            "duration": self.duration_name,
        }

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        details = {
            "provider": "eccc-idf",
            **self._request(),
            "url": provider.archive_url(self.version, self.province),
            "hazard_name": f"RainfallIntensity/{self.duration_name}",
            "unit": "mm/h",
            "spatial_support": "station point; H3 index only; no interpolation",
            "horizon": "each station's inclusive record midpoint",
            "acquisition": "provincial bundle"
            if self.version == "3.40"
            else "station TXT byte ranges",
            "licence": provider.LICENCE,
            "attribution": provider.ATTRIBUTION,
            "cache": {
                "mode": self.cache_mode,
                "directory": str(self.cache_dir) if self.cache_dir else None,
            },
        }
        if format == "json":
            return details
        if format != "text":
            raise ValueError("format must be text or json")
        return json.dumps(details, indent=2)

    def _prepare(
        self, directory: Path, progress: ProgressCallback | None = None
    ) -> tuple[list[dict[str, Any]], int, int]:
        if not self.selected_stations and not self.requested_areas:
            raise ValueError(
                "configure for_area(...), for_areas(...) or for_stations(...)"
            )
        scope = f"eccc-idf-{self.version}-{self.province}"
        directory = directory / scope
        manifest = read_manifest(directory) or {}
        stored = dict(manifest.get("entries", {}))
        complete = manifest.get("complete", False)
        wanted = set(self.selected_stations or stored)
        valid = {
            s: e
            for s, e in stored.items()
            if s in wanted
            and (directory / f"{s}.txt").is_file()
            and file_checksum(directory / f"{s}.txt") == e["checksum"]
        }
        missing = wanted - valid.keys()
        fetch = (
            self.cache_mode == "refresh"
            or bool(missing)
            or (not self.selected_stations and not complete)
        )
        if fetch and self.cache_mode == "offline":
            raise FileNotFoundError(
                "ECCC station cache missing or corrupt; prefetch online"
            )
        hits = len(wanted & valid.keys()) if self.cache_mode != "refresh" else 0
        misses = 0
        if fetch:
            selection = (
                self.selected_stations
                if self.cache_mode == "refresh"
                else tuple(sorted(missing))
                if self.selected_stations
                else ()
            )
            if progress:
                progress("fetch", {"province": self.province, "version": self.version})
            directory.mkdir(parents=True, exist_ok=True)
            for station, raw, remote in provider.fetch_stations(
                self.version, self.province, selection
            ):
                provider.parse_station(raw, station, self.province)
                previous = stored.get(station)
                if previous and previous["checksum"] != remote["checksum"]:
                    raise RuntimeError(
                        "ECCC station bytes changed under a pinned release"
                    )
                entry = {
                    **remote,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "version": self.version,
                    "province": self.province,
                    "station": station,
                    "licence": provider.LICENCE,
                    "attribution": provider.ATTRIBUTION,
                }
                path = directory / f"{station}.txt"
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
                    tmp.write(raw)
                Path(tmp.name).replace(path)
                stored[station] = entry
                valid[station] = entry
                misses += 1
            write_manifest(
                directory,
                {"entries": stored, "complete": complete or not self.selected_stations},
            )
        records: list[dict[str, Any]] = []
        for station in self.selected_stations or tuple(sorted(valid)):
            record = provider.parse_station(
                (directory / f"{station}.txt").read_bytes(), station, self.province
            )
            if self.requested_areas and not any(
                west <= record["longitude"] <= east
                and south <= record["latitude"] <= north
                for west, south, east, north in self.requested_areas
            ):
                continue
            records.append({**record, "source": valid[station]})
        if not records:
            raise ValueError("no ECCC station locations match the selection")
        return records, hits, misses

    def prefetch(self, *, progress: ProgressCallback | None = None) -> PrefetchResult:
        if self.cache_dir is None:
            raise ValueError("prefetch requires a persistent cache directory")
        records, hits, misses = self._prepare(self.cache_dir, progress)
        return PrefetchResult(self.version, hits, misses, len(records))

    def source_curves(self, *, progress: ProgressCallback | None = None) -> Any:
        """Official rates and 95% confidence half-widths, without fitting."""
        with tempfile.TemporaryDirectory(prefix="crc-idf-") as scratch:
            records, _, _ = self._prepare(self.cache_dir or Path(scratch), progress)
            return pa.Table.from_pylist(
                [
                    {
                        **{
                            k: v
                            for k, v in record.items()
                            if k not in ("curves", "source")
                        },
                        **curve,
                        "source_version": self.version,
                        "source_checksum": record["source"]["checksum"],
                    }
                    for record in records
                    for curve in record["curves"]
                    if curve["duration_minutes"]
                    == provider.DURATIONS[self.duration_name]
                ]
            )

    def materialize(
        self, output: str | Path, *, progress: ProgressCallback | None = None
    ) -> HazardDataset:
        with tempfile.TemporaryDirectory(prefix="crc-idf-") as scratch:
            records, hits, misses = self._prepare(
                self.cache_dir or Path(scratch), progress
            )
            windows = {(r["start_year"], r["end_year"]) for r in records}
            window = (
                TemporalWindow(
                    start_year=records[0]["start_year"],
                    end_year=records[0]["end_year"],
                    calendar="gregorian",
                )
                if len(windows) == 1
                else None
            )
            request = json.dumps(
                {
                    **self._request(),
                    "records": [
                        {
                            "station": r["station"],
                            "start_year": r["start_year"],
                            "end_year": r["end_year"],
                        }
                        for r in records
                    ],
                },
                sort_keys=True,
            )
            url = provider.archive_url(self.version, self.province)
            name = f"RainfallIntensity/{self.duration_name}"
            source = _IDFCurveSource(
                records,
                self.duration_name,
                RasterMetadata(
                    hazard_type=name,
                    indicator_id=name,
                    scenario="historical",
                    year=0,
                    units="mm/h",
                    path=url,
                ),
                CurveSourceInfo(
                    uri=f"{url}#crc-request={quote(request)}",
                    licence=provider.LICENCE,
                    attribution=provider.ATTRIBUTION,
                    retrieved_at=max(r["source"]["retrieved_at"] for r in records),
                    checksum=hashlib.sha256(
                        "".join(r["source"]["checksum"] for r in records).encode()
                    ).hexdigest(),
                    temporal_window=window,
                    probability_semantics="annual_exceedance",
                    source_return_period_convention="one_minus_inverse",
                ),
            )
            policy = replace(
                self.fit_policy,
                source_version=self.version,
                value_semantics=(
                    f"Annual maximum {self.duration_name} rainfall intensity; "
                    "derived quantile fit of ECCC central IDF rates"
                ),
            )
            stream = canonicalize_curve_source(source, policy, provider="eccc-idf")
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_hazard_stream(stream, destination, overwrite=True, ordered=False)
        return HazardDataset.local(
            destination,
            materialization=MaterializationResult(
                destination,
                self.version,
                hits,
                misses,
                pq.ParquetFile(destination).metadata.num_rows,
            ),
        )

    def ensure_materialized(
        self, *, progress: ProgressCallback | None = None
    ) -> HazardDataset:
        """Fit verified cached station bytes to a deterministic local path."""
        if self.cache_dir is None:
            raise ValueError("one-chain evaluation requires cache(path)")
        identity = json.dumps([self._request(), repr(self.fit_policy)], sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return self.materialize(
            self.cache_dir / "canonical" / f"idf-{key}.parquet", progress=progress
        )

    def for_assets(self, assets: Any | AssetPortfolio) -> RemotePortfolioEvaluation:
        portfolio = (
            assets if isinstance(assets, AssetPortfolio) else AssetPortfolio(assets)
        )
        return RemotePortfolioEvaluation(plan=self, portfolio=portfolio)
