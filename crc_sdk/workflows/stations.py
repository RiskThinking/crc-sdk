"""Shared immutable NOAA station observation acquisition and fitting plans."""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import import_module
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeVar
from urllib.parse import quote

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from crc_sdk.connectors.adapters import CurveSourceInfo, canonicalize_curve_source
from crc_sdk.connectors.duckdb.zarr import RasterMetadata
from crc_sdk.connectors.parquet import write_hazard_stream
from crc_sdk.connectors.stations import StationCurveSource
from crc_sdk.providers.ghcnh import GHCNhCompleteness
from crc_sdk.providers.isd import ISDCompleteness, validate_recipe
from crc_sdk.providers.station_catalogues import fetch_catalogue
from crc_sdk.types import TemporalWindow

from ._remote import (
    Bounds,
    CacheMode,
    MaterializationResult,
    PrefetchResult,
    ProgressCallback,
    file_checksum,
    read_manifest,
    validate_bounds,
    write_manifest,
)
from .blocks import BlockExtremaPolicy
from .portfolio import HazardDataset

PlanT = TypeVar("PlanT", bound="StationObservationPlan")


@dataclass(frozen=True)
class StationObservationPlan:
    """Historical temperature extremes at stations selected by ID or area.

    Builders and explain perform no network I/O. Each worker streams one
    station/year, with its own DuckDB memory limit; only daily reductions are
    cached. ``annual_extremes`` includes excluded years with null values.
    """

    provider: Literal["isd", "ghcnh"] = "isd"
    recipe: str = "temperature_max"
    selected_stations: tuple[str, ...] = ()
    requested_areas: tuple[Bounds, ...] = ()
    selection_catalogue_sha256: str | None = None
    selected_years: tuple[int, ...] = ()
    completeness: ISDCompleteness | GHCNhCompleteness = ISDCompleteness()
    fit_policy: BlockExtremaPolicy = BlockExtremaPolicy()
    cache_dir: Path | None = None
    cache_mode: CacheMode = "stream"
    workers: int = 4
    memory_limit: str = "256MB"

    def __post_init__(self) -> None:
        validate_recipe(self.recipe)
        if self.provider not in ("isd", "ghcnh"):
            raise ValueError("station provider must be isd or ghcnh")

    @property
    def _provider(self) -> ModuleType:
        return import_module(f"crc_sdk.providers.{self.provider}")

    def for_stations(self: PlanT, stations: Sequence[str]) -> PlanT:
        if isinstance(stations, str):
            raise TypeError("stations must be a sequence of station IDs")
        ids = tuple(stations)
        if not ids or any(
            not isinstance(s, str)
            or not re.fullmatch(self._provider.STATION_PATTERN, s)
            for s in ids
        ):
            raise ValueError(
                f"stations must contain {self._provider.STATION_DESCRIPTION}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("station IDs must be unique")
        return replace(
            self,
            selected_stations=tuple(sorted(ids)),
            requested_areas=(),
            selection_catalogue_sha256=None,
        )

    def for_area(self: PlanT, bounds: Sequence[float]) -> PlanT:
        """Select catalogue station locations inside a WGS84 bounding box."""
        return self.for_areas([bounds])

    def for_areas(self: PlanT, bounds: Sequence[Sequence[float]]) -> PlanT:
        """Select the union of boxes; overlapping boxes never duplicate stations."""
        areas = tuple(sorted({validate_bounds(area) for area in bounds}))
        if not areas:
            raise ValueError("for_areas requires at least one bounding box")
        return replace(
            self,
            requested_areas=areas,
            selected_stations=(),
            selection_catalogue_sha256=None,
        )

    def _catalogue(self, directory: Path) -> tuple[Any, dict[str, Any]]:
        manifest = read_manifest(directory) or {}
        catalogues = dict(manifest.get("catalogues", {}))
        entry = catalogues.get(self.provider)
        if (
            entry
            and entry.get("normalization_version") == 1
            and self.cache_mode != "refresh"
        ):
            path = directory / entry["local"]
            if path.is_file() and file_checksum(path) == entry["checksum"]:
                return pq.ParquetFile(path).read(), entry
        if self.cache_mode == "offline":
            raise FileNotFoundError(
                f"{self.provider} station catalogue missing or corrupt; resolve online"
            )
        table, remote = fetch_catalogue(
            self.provider, self._provider.CATALOGUE_URL, self._provider.STATION_PATTERN
        )
        path = directory / "catalogues" / f"{self.provider}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet") as tmp:
            pq.write_table(table, tmp.name, compression="zstd")
            checksum = file_checksum(Path(tmp.name))
            Path(tmp.name).replace(path)
        entry = {
            **remote,
            "normalization_version": 1,
            "licence": self._provider.LICENCE,
            "attribution": self._provider.ATTRIBUTION,
            "checksum": checksum,
            "local": path.relative_to(directory).as_posix(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        catalogues[self.provider] = entry
        write_manifest(directory, {**manifest, "catalogues": catalogues})
        return table, entry

    def _station_selection(self, directory: Path) -> tuple[Any, dict[str, Any]]:
        if not self.selected_stations and not self.requested_areas:
            raise ValueError(
                "configure for_area(...), for_areas(...) or for_stations(...)"
            )
        catalogue, entry = self._catalogue(directory)
        with duckdb.connect() as con:
            con.execute("SET memory_limit = '256MB'")
            con.execute("SET threads = 1")
            con.register("catalogue", catalogue)
            predicates: list[str] = []
            parameters: list[Any] = []
            if self.selected_stations:
                predicates.append("station IN (SELECT unnest(?))")
                parameters.append(list(self.selected_stations))
            elif self.requested_areas:
                con.register(
                    "areas",
                    pa.Table.from_pylist(
                        [
                            dict(zip(("west", "south", "east", "north"), area))
                            for area in self.requested_areas
                        ]
                    ),
                )
                predicates.append("""EXISTS (
                    SELECT 1 FROM areas
                    WHERE longitude BETWEEN west AND east
                      AND latitude BETWEEN south AND north
                )""")
            if self.selected_years:
                predicates.extend(
                    [
                        "(start_year IS NULL OR start_year <= ?)",
                        "(end_year IS NULL OR end_year >= ?)",
                    ]
                )
                parameters.extend([self.selected_years[-1], self.selected_years[0]])
            query = con.execute(
                "SELECT * FROM catalogue WHERE "
                + " AND ".join(predicates)
                + " ORDER BY station",
                parameters,
            )
            table = (
                query.to_arrow_table()
                if hasattr(query, "to_arrow_table")
                else query.fetch_arrow_table()
            )
        if table.num_rows == 0:
            raise ValueError("no catalogue stations match the selection and year range")
        if self.selected_stations:
            missing = set(self.selected_stations) - set(
                table.column("station").to_pylist()
            )
            if missing:
                raise ValueError(
                    f"station catalogue has no matching records for {sorted(missing)}"
                )
        return table, entry

    def stations(self) -> Any:
        """Read selected station metadata only; never fetch observations."""
        with tempfile.TemporaryDirectory(prefix="crc-stations-") as scratch:
            table, _ = self._station_selection(self.cache_dir or Path(scratch))
            return table

    def _resolve_stations(self: PlanT, directory: Path) -> PlanT:
        table, entry = self._station_selection(directory)
        return replace(
            self,
            selected_stations=tuple(table.column("station").to_pylist()),
            selection_catalogue_sha256=entry["raw_sha256"],
        )

    def resolve_stations(self: PlanT) -> PlanT:
        """Pin catalogue selection as explicit IDs in a new immutable plan."""
        with tempfile.TemporaryDirectory(prefix="crc-stations-") as scratch:
            return self._resolve_stations(self.cache_dir or Path(scratch))

    def years(self: PlanT, start: int, end: int) -> PlanT:
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not self._provider.FIRST_YEAR <= start <= end <= self._provider.LAST_YEAR
        ):
            raise ValueError(
                f"{self.provider.upper()} complete-year range must be within "
                f"{self._provider.FIRST_YEAR}–{self._provider.LAST_YEAR}"
            )
        return replace(self, selected_years=tuple(range(start, end + 1)))

    def quality(
        self: PlanT,
        *,
        minimum_hour_fraction: float = 0.75,
        minimum_day_fraction: float = 0.9,
        quality_codes: tuple[str, ...] | None = None,
    ) -> PlanT:
        return replace(
            self,
            completeness=self._provider.Completeness(
                minimum_hour_fraction,
                minimum_day_fraction,
                self._provider.Completeness().quality_codes
                if quality_codes is None
                else quality_codes,
            ),
        )

    def execution(
        self: PlanT, *, workers: int = 4, memory_limit: str = "256MB"
    ) -> PlanT:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?(?:KB|MB|GB)", memory_limit):
            raise ValueError("memory_limit must be a positive size in KB, MB or GB")
        return replace(self, workers=workers, memory_limit=memory_limit)

    def cache(
        self: PlanT, directory: str | Path | None, *, mode: CacheMode = "reuse"
    ) -> PlanT:
        if mode not in ("reuse", "offline", "refresh", "stream"):
            raise ValueError("invalid station cache mode")
        if (mode == "stream") != (directory is None):
            raise ValueError("only stream mode requires directory=None")
        return replace(
            self, cache_dir=Path(directory) if directory else None, cache_mode=mode
        )

    def canonicalize(
        self: PlanT, *, policy: str | BlockExtremaPolicy = "curated"
    ) -> PlanT:
        if policy == "curated":
            policy = BlockExtremaPolicy.curated()
        if not isinstance(policy, BlockExtremaPolicy):
            raise TypeError("policy must be 'curated' or BlockExtremaPolicy")
        resolved = policy.resolve(
            family="genextreme" if self.recipe == "temperature_max" else "gumbel_l"
        )
        resolved.ingest_policy(
            tail=self._tail,
            value_semantics=self._semantics,
            source_version=self._provider.VERSION,
        )
        return replace(self, fit_policy=resolved)

    @property
    def _tail(self) -> Literal["upper", "lower"]:
        return "upper" if self.recipe == "temperature_max" else "lower"

    @property
    def _semantics(self) -> str:
        statistic = self.recipe.removeprefix("temperature_")
        return f"Annual {statistic} of QC-filtered sampled air temperature; UTC days"

    def _request(self) -> dict[str, Any]:
        return {
            "version": self._provider.VERSION,
            "provider": self.provider,
            "stations": self.selected_stations,
            "areas": self.requested_areas,
            "years": self.selected_years,
            "completeness": asdict(self.completeness),
            "recipe": self.recipe,
            "daily_boundary": "UTC",
            "reduction_version": 1,
        }

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        details = {
            **self._request(),
            "hazard_name": f"{self.provider}_{self.recipe}",
            "unit": "degC",
            "spatial_support": "station point; H3 index only; no area interpolation",
            "cache": {
                "mode": self.cache_mode,
                "directory": str(self.cache_dir) if self.cache_dir else None,
                "object": "station/year daily min/max and quality counts",
            },
            "execution": {
                "workers": self.workers,
                "memory_limit_per_worker": self.memory_limit,
            },
            "station_selection": (
                "catalogue locations within bounds; resolution deferred"
                if self.requested_areas and not self.selected_stations
                else "explicit IDs"
            ),
            "minimum_years": self.fit_policy.minimum_years,
            "licence": self._provider.LICENCE,
            "attribution": self._provider.ATTRIBUTION,
        }
        if format == "json":
            return details
        if format != "text":
            raise ValueError("format must be text or json")
        return json.dumps(details, indent=2)

    def _validate(self, *, fitting: bool = False) -> None:
        if (
            not self.selected_stations and not self.requested_areas
        ) or not self.selected_years:
            raise ValueError(
                "configure for_area(...) or for_stations(...), and years(start, end)"
            )
        if fitting:
            self.fit_policy.validate_years(self.selected_years)

    def _key(self, station: str, year: int) -> str:
        # Daily reductions serve either tail, but QC and decoder changes do not.
        identity = [self._provider.VERSION, 1, station, year, asdict(self.completeness)]
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def _prepare(
        self, directory: Path, progress: ProgressCallback | None = None
    ) -> tuple[list[dict[str, Any]], int, int]:
        self._validate()
        if self.requested_areas and not self.selected_stations:
            return self._resolve_stations(directory)._prepare(directory, progress)
        manifest = read_manifest(directory) or {}
        stored = dict(manifest.get("entries", {}))
        entries: dict[str, dict[str, Any]] = {}
        todo: list[tuple[str, int]] = []
        for station in self.selected_stations:
            for year in self.selected_years:
                key = self._key(station, year)
                entry = stored.get(key)
                if entry and self.cache_mode != "refresh":
                    path = directory / entry["local"]
                    if path.is_file() and file_checksum(path) == entry["checksum"]:
                        entries[key] = entry
                        continue
                todo.append((station, year))
        if todo and self.cache_mode == "offline":
            raise FileNotFoundError(
                f"{self.provider.upper()} cache missing or corrupt: {todo}; "
                "prefetch online"
            )
        hits = len(entries)
        directory.mkdir(parents=True, exist_ok=True)

        def fetch(job: tuple[str, int]) -> tuple[str, dict[str, Any]]:
            station, year = job
            key = self._key(station, year)
            if progress:
                progress("fetch", {"station": station, "year": year})
            with tempfile.TemporaryDirectory(prefix="crc-station-worker-") as scratch:
                table, remote = self._provider.fetch_daily(
                    station,
                    year,
                    self.completeness,
                    temporary_directory=scratch,
                    memory_limit=self.memory_limit,
                )
            path = directory / "daily" / f"{station}-{year}-{key[:16]}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet") as tmp:
                pq.write_table(table, tmp.name, compression="zstd")
                checksum = file_checksum(Path(tmp.name))
                Path(tmp.name).replace(path)
            return key, {
                **remote,
                "station": station,
                "year": year,
                "resolved_version": self._provider.VERSION,
                "request": asdict(self.completeness),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "licence": self._provider.LICENCE,
                "attribution": self._provider.ATTRIBUTION,
                "local": path.relative_to(directory).as_posix(),
                "checksum": checksum,
            }

        failure: Exception | None = None
        jobs = iter(todo)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = {pool.submit(fetch, job) for job in islice(jobs, self.workers)}
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    try:
                        key, entry = future.result()
                    except Exception as error:
                        if failure is None:
                            failure = error
                    else:
                        entries[key] = entry
                        stored[key] = entry
                if failure is None:
                    pending.update(
                        pool.submit(fetch, job) for job in islice(jobs, len(completed))
                    )
        # Persist partial successes on failure, without rewriting per object.
        if todo:
            write_manifest(
                directory,
                {
                    **(read_manifest(directory) or {}),
                    "provider": "noaa-stations",
                    "entries": stored,
                },
            )
        if failure is not None:
            raise failure
        ordered = [
            entries[self._key(s, y)]
            for s in self.selected_stations
            for y in self.selected_years
        ]
        return ordered, hits, len(todo)

    def prefetch(self, *, progress: ProgressCallback | None = None) -> PrefetchResult:
        if self.cache_dir is None:
            raise ValueError("prefetch requires a persistent cache directory")
        entries, hits, misses = self._prepare(self.cache_dir, progress)
        return PrefetchResult(self._provider.VERSION, hits, misses, len(entries))

    def _annual(
        self, directory: Path, entries: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        for entry in entries:
            daily = pq.ParquetFile(directory / entry["local"]).read().to_pylist()
            if not daily:
                yield {
                    "station": entry["station"],
                    "year": entry["year"],
                    "latitude": None,
                    "longitude": None,
                    "elevation": None,
                    "valid_days": 0,
                    "expected_days": 366 if calendar.isleap(entry["year"]) else 365,
                    "accepted_reports": 0,
                    "reports": 0,
                    "value": None,
                    "complete": False,
                }
                continue
            if any(
                row["latitude"] is None or row["longitude"] is None for row in daily
            ):
                raise ValueError(f"missing coordinates for {entry['station']}")
            lat = min(row["latitude"] for row in daily)
            lon = min(row["longitude"] for row in daily)
            elevation = min(
                (row["elevation"] for row in daily if row["elevation"] is not None),
                default=None,
            )
            if (
                not -90 <= lat <= 90
                or not -180 <= lon <= 180
                or max(row["latitude_max"] for row in daily) - lat > 0.05
                or max(row["longitude_max"] for row in daily) - lon > 0.05
            ):
                raise ValueError(
                    f"invalid or moving coordinates for {entry['station']}"
                )
            valid = [
                row
                for row in daily
                if row["valid_hours"]
                >= math.ceil(24 * self.completeness.minimum_hour_fraction)
            ]
            expected = 366 if calendar.isleap(entry["year"]) else 365
            complete = len(valid) >= math.ceil(
                expected * self.completeness.minimum_day_fraction
            )
            values = [row[self.recipe] for row in valid]
            value = (
                (max(values) if self._tail == "upper" else min(values))
                if complete
                else None
            )
            yield {
                "station": entry["station"],
                "year": entry["year"],
                "latitude": lat,
                "longitude": lon,
                "elevation": elevation,
                "valid_days": len(valid),
                "expected_days": expected,
                "accepted_reports": sum(row["accepted_reports"] for row in daily),
                "reports": sum(row["reports"] for row in daily),
                "value": value,
                "complete": complete,
            }

    def annual_extremes(self, *, progress: ProgressCallback | None = None) -> Any:
        """Arrow QA table, including incomplete years with null values."""
        with tempfile.TemporaryDirectory(prefix="crc-station-") as scratch:
            directory = self.cache_dir or Path(scratch)
            entries, _, _ = self._prepare(directory, progress)
            return pa.Table.from_pylist(
                list(self._annual(directory, entries)), schema=self._annual_schema()
            )

    @staticmethod
    def _annual_schema() -> Any:
        return pa.schema(
            [
                ("station", pa.string()),
                ("year", pa.int32()),
                ("latitude", pa.float64()),
                ("longitude", pa.float64()),
                ("elevation", pa.float64()),
                ("valid_days", pa.int32()),
                ("expected_days", pa.int32()),
                ("accepted_reports", pa.int64()),
                ("reports", pa.int64()),
                ("value", pa.float64()),
                ("complete", pa.bool_()),
            ]
        )

    def write_annual_extremes(
        self, output: str | Path, *, progress: ProgressCallback | None = None
    ) -> Path:
        """Write the annual QA series with memory bounded to one station/year."""
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="crc-station-") as scratch:
            directory = self.cache_dir or Path(scratch)
            entries, _, _ = self._prepare(directory, progress)
            with pq.ParquetWriter(destination, self._annual_schema()) as writer:
                batch: list[dict[str, Any]] = []
                for annual in self._annual(directory, entries):
                    batch.append(annual)
                    if len(batch) == 4096:
                        writer.write_table(
                            pa.Table.from_pylist(batch, schema=self._annual_schema())
                        )
                        batch.clear()
                if batch:
                    writer.write_table(
                        pa.Table.from_pylist(batch, schema=self._annual_schema())
                    )
        return destination

    def materialize(
        self, output: str | Path, *, progress: ProgressCallback | None = None
    ) -> HazardDataset:
        self._validate(fitting=True)
        resolved = self.canonicalize(policy=self.fit_policy)
        with tempfile.TemporaryDirectory(prefix="crc-station-") as scratch:
            directory = self.cache_dir or Path(scratch)
            entries, hits, misses = self._prepare(directory, progress)
            window = TemporalWindow(
                start_year=self.selected_years[0],
                end_year=self.selected_years[-1],
                calendar="gregorian",
                minimum_complete_years=self.fit_policy.minimum_years,
            )

            def groups() -> Iterator[list[dict[str, Any]]]:
                for index in range(0, len(entries), len(self.selected_years)):
                    yield list(
                        self._annual(
                            directory, entries[index : index + len(self.selected_years)]
                        )
                    )

            request_details = self._request()
            request_details["stations"] = tuple(
                sorted({entry["station"] for entry in entries})
            )
            if self.requested_areas:
                catalogue = (
                    (read_manifest(directory) or {})
                    .get("catalogues", {})
                    .get(self.provider, {})
                )
                request_details["catalogue"] = {
                    "url": self._provider.CATALOGUE_URL,
                    "raw_sha256": self.selection_catalogue_sha256
                    or catalogue.get("raw_sha256"),
                }
            request = json.dumps(request_details, sort_keys=True)
            source = StationCurveSource(
                groups,
                RasterMetadata(
                    hazard_type=f"{self.provider}_{self.recipe}",
                    indicator_id=f"{self.provider}_{self.recipe}",
                    scenario="historical",
                    year=window.horizon,
                    units="degC",
                    path=request,
                ),
                info=CurveSourceInfo(
                    uri=f"{self._provider.URL}/#crc-request={quote(request)}",
                    licence=self._provider.LICENCE,
                    attribution=self._provider.ATTRIBUTION,
                    retrieved_at=max(entry["retrieved_at"] for entry in entries),
                    checksum=hashlib.sha256(
                        "".join(entry["checksum"] for entry in entries).encode()
                    ).hexdigest(),
                    temporal_window=window,
                    probability_semantics="annual_exceedance",
                ),
                tail=self._tail,
                provider=f"noaa-{self.provider}",
                minimum_years=self.fit_policy.minimum_years,
            )
            stream = canonicalize_curve_source(
                source,
                resolved.fit_policy.ingest_policy(
                    tail=self._tail,
                    value_semantics=self._semantics,
                    source_version=self._provider.VERSION,
                ),
                provider=f"noaa-{self.provider}",
            )
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_hazard_stream(stream, destination, overwrite=True, ordered=False)
        rows = pq.ParquetFile(destination).metadata.num_rows
        return HazardDataset.local(
            destination,
            materialization=MaterializationResult(
                destination,
                self._provider.VERSION,
                hits,
                misses,
                rows,
            ),
        )
