"""Fluent, lazy bring-your-own-data canonicalization plans.

Three entry points (`HazardDataset.from_table`, `from_zarr`, `from_raster`)
share one immutable plan: nothing is opened, read or fitted until
`materialize()` / `ensure_materialized()` runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, Union

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from crc_sdk.connectors.adapters import (
    CanonicalHazardStream,
    CurveFitIngestPolicy,
    canonicalize_curve_source,
)
from crc_sdk.connectors.duckdb.zarr import RasterMetadata, ZarrRaster
from crc_sdk.connectors.parquet import write_hazard_stream
from crc_sdk.fitting.workflows import (
    CDFColumnSchema,
    CDFCurveFitPolicy,
    fit_cdf_quantile_batches,
)

from ._remote import (
    Bounds,
    CacheMode,
    MaterializationResult,
    ProgressCallback,
    RemotePortfolioEvaluation,
    validate_bounds,
)
from .distributions import return_periods_to_probabilities
from .portfolio import AssetPortfolio, HazardDataset

BYOPolicy = Union[CDFCurveFitPolicy, CurveFitIngestPolicy]

_BATCH_ROWS = 65_536
_PARQUET_SUFFIXES = (".parquet", ".pq")
_CSV_SUFFIXES = (".csv", ".tsv", ".csv.gz", ".tsv.gz")


class _BYOSource(Protocol):
    """One kind of user-supplied input; opening it is the only I/O."""

    @property
    def kind(self) -> str: ...

    def location(self) -> str: ...

    def identity(self) -> list[Any]: ...

    def check_policy(self, policy: BYOPolicy) -> None: ...

    def supports_bounds(self) -> bool: ...

    def open(
        self, policy: BYOPolicy, bounds: Bounds | None
    ) -> Any:  # context manager yielding CanonicalHazardStream
        ...


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _duckdb_batches(path: str, list_columns: Sequence[str]) -> Iterator[pa.RecordBatch]:
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - duckdb is a core dependency
        raise ImportError("DuckDB table sources require duckdb") from error
    lowered = path.lower()
    connection = duckdb.connect()
    try:
        relation = (
            f"read_csv({_sql_string(path)})"
            if lowered.endswith(_CSV_SUFFIXES)
            else _sql_string(path)
        )
        # CSV cannot type list cells, so `[1.0, 2.0]` text is cast to numbers.
        casts = ", ".join(
            f'CAST("{name}" AS DOUBLE[]) AS "{name}"' for name in list_columns
        )
        replace = f" REPLACE ({casts})" if casts else ""
        result = connection.execute(f"SELECT *{replace} FROM {relation}")
        # `fetch_record_batch` was renamed in DuckDB 1.4.
        open_reader = getattr(result, "to_arrow_reader", None)
        reader = (open_reader or result.fetch_record_batch)(_BATCH_ROWS)
        yield from reader
    finally:
        connection.close()


def _parquet_batches(path: str) -> Iterator[pa.RecordBatch]:
    yield from pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS)


def _table_identity(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _file_identity(path: str) -> list[Any]:
    # Local files are fingerprinted so a changed source invalidates the cache.
    target = Path(path)
    if "://" in path or not target.is_file():
        return [path]
    status = target.stat()
    return [str(target.resolve()), status.st_size, status.st_mtime_ns]


@dataclass(frozen=True)
class _TableSource:
    source: str | Path | pa.Table = field(compare=False)
    columns: CDFColumnSchema
    probabilities: tuple[float, ...] | None
    # Shared return-period axis, converted at fit time with the policy that is
    # actually in force (tail and convention), not the one known at plan time.
    return_periods: tuple[float, ...] | None = None
    kind: str = "table"

    def location(self) -> str:
        if isinstance(self.source, pa.Table):
            return f"in-memory Arrow table ({self.source.num_rows} rows)"
        return str(self.source)

    def identity(self) -> list[Any]:
        source = (
            [_table_identity(self.source)]
            if isinstance(self.source, pa.Table)
            else _file_identity(str(self.source))
        )
        return [
            self.kind,
            source,
            repr(self.columns),
            self.probabilities,
            self.return_periods,
        ]

    def check_policy(self, policy: BYOPolicy) -> None:
        if not isinstance(policy, CDFCurveFitPolicy):
            raise TypeError("table sources require a CDFCurveFitPolicy")

    def supports_bounds(self) -> bool:
        return False

    def _batches(self) -> Iterator[pa.RecordBatch]:
        if isinstance(self.source, pa.Table):
            return iter(self.source.to_batches(max_chunksize=_BATCH_ROWS))
        path = str(self.source)
        if path.lower().endswith(_PARQUET_SUFFIXES):
            return _parquet_batches(path)
        axis = (self.columns.probabilities, self.columns.return_periods)
        values = self.columns.samples or self.columns.quantiles
        return _duckdb_batches(
            path, [name for name in (values, *axis) if name is not None]
        )

    @contextmanager
    def open(
        self, policy: BYOPolicy, bounds: Bounds | None
    ) -> Iterator[CanonicalHazardStream]:
        assert isinstance(policy, CDFCurveFitPolicy)
        batches = self._batches()
        probabilities = self.probabilities
        if self.return_periods is not None:
            probabilities = return_periods_to_probabilities(
                self.return_periods,
                tail=policy.return_period_tail,
                convention=policy.return_period_convention,
            )
        try:
            yield fit_cdf_quantile_batches(
                batches,
                probabilities,
                policy,
                columns=self.columns,
            ).stream
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()


def _metadata_identity(metadata: RasterMetadata) -> list[Any]:
    return [
        metadata.hazard_type,
        metadata.indicator_id,
        metadata.scenario,
        metadata.year,
        metadata.units,
    ]


@dataclass(frozen=True)
class _ZarrSource:
    source: Any = field(compare=False)
    metadata: RasterMetadata
    array: str | None = None
    storage_options: Mapping[str, Any] | None = None
    kind: str = "zarr"

    def location(self) -> str:
        if isinstance(self.source, (str, Path)):
            base = str(self.source)
            return f"{base}#{self.array}" if self.array else base
        return self.metadata.path

    def identity(self) -> list[Any]:
        where = (
            _file_identity(str(self.source))
            if isinstance(self.source, (str, Path))
            else [self.metadata.path]
        )
        return [
            self.kind,
            where,
            self.array,
            _metadata_identity(self.metadata),
            sorted((self.storage_options or {}).keys()),
        ]

    def check_policy(self, policy: BYOPolicy) -> None:
        if not isinstance(policy, CurveFitIngestPolicy):
            raise TypeError("zarr sources require a CurveFitIngestPolicy")

    def supports_bounds(self) -> bool:
        return True

    def _array(self) -> Any:
        if not isinstance(self.source, (str, Path)):
            return self.source
        try:
            import zarr
        except ImportError as error:
            raise ImportError(
                "Zarr sources require `pip install crc-sdk[zarr]`"
            ) from error
        options = dict(self.storage_options) if self.storage_options else {}
        target = str(self.source)
        try:
            if self.array is None:
                return zarr.open_array(target, mode="r", **_storage(options))
            group = zarr.open_group(target, mode="r", **_storage(options))
        except TypeError:
            # zarr 2 takes a mapping, not `storage_options`.
            import fsspec  # type: ignore[import-untyped]

            store = fsspec.get_mapper(target, **options)
            if self.array is None:
                return zarr.open_array(store, mode="r")
            group = zarr.open_group(store, mode="r")
        return group[self.array]

    @contextmanager
    def open(
        self, policy: BYOPolicy, bounds: Bounds | None
    ) -> Iterator[CanonicalHazardStream]:
        assert isinstance(policy, CurveFitIngestPolicy)
        raster = ZarrRaster(self._array(), self.metadata)
        yield canonicalize_curve_source(
            raster, policy, provider="byo-zarr", bounds=bounds
        )


def _storage(options: dict[str, Any]) -> dict[str, Any]:
    return {"storage_options": options} if options else {}


@dataclass(frozen=True)
class _RasterSource:
    paths: tuple[tuple[int, str], ...]
    metadata: RasterMetadata
    band: int = 1
    assumed_crs: str | None = None
    kind: str = "raster"

    def location(self) -> str:
        return ", ".join(f"RP{period}={path}" for period, path in self.paths)

    def identity(self) -> list[Any]:
        return [
            self.kind,
            [[period, _file_identity(path)] for period, path in self.paths],
            _metadata_identity(self.metadata),
            self.band,
            self.assumed_crs,
        ]

    def check_policy(self, policy: BYOPolicy) -> None:
        if not isinstance(policy, CurveFitIngestPolicy):
            raise TypeError("raster sources require a CurveFitIngestPolicy")

    def supports_bounds(self) -> bool:
        return True

    @contextmanager
    def open(
        self, policy: BYOPolicy, bounds: Bounds | None
    ) -> Iterator[CanonicalHazardStream]:
        assert isinstance(policy, CurveFitIngestPolicy)
        try:
            import rasterio  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "GeoTIFF sources require `pip install crc-sdk[raster]`"
            ) from error
        from crc_sdk.connectors.duckdb.geotiff import JRCReturnPeriodRaster

        with JRCReturnPeriodRaster.open(
            dict(self.paths),
            self.metadata,
            band=self.band,
            assumed_crs=self.assumed_crs,
        ) as raster:
            native = raster.bounds_from_wgs84(bounds) if bounds is not None else None
            yield canonicalize_curve_source(
                raster, policy, provider="byo-raster", bounds=native
            )


def _source_version(policy: BYOPolicy) -> str:
    version = (
        policy.source.version
        if isinstance(policy, CDFCurveFitPolicy)
        else policy.source_version
    )
    return version or "unspecified"


@dataclass(frozen=True)
class BYOPlan:
    """Lazy plan canonicalizing user-supplied data into a hazard dataset."""

    source: _BYOSource
    policy: BYOPolicy | None = None
    bounds: Bounds | None = None
    cache_dir: Path | None = None
    cache_mode: Literal["reuse", "refresh"] = "reuse"

    def canonicalize(self, *, policy: BYOPolicy) -> BYOPlan:
        """Set the fit policy (`CDFCurveFitPolicy` for tables, else ingest)."""
        self.source.check_policy(policy)
        return replace(self, policy=policy)

    def for_area(self, bounds: Sequence[float]) -> BYOPlan:
        """Window a raster or Zarr read to WGS84 bounds."""
        if not self.source.supports_bounds():
            raise ValueError(f"{self.source.kind} sources have no spatial window")
        return replace(self, bounds=validate_bounds(bounds))

    def cache(
        self,
        directory: str | Path | None,
        *,
        mode: CacheMode = "reuse",
    ) -> BYOPlan:
        """Choose where `ensure_materialized()` keeps its canonical file."""
        if mode not in ("reuse", "refresh"):
            raise ValueError("BYO plans support only reuse or refresh cache modes")
        return replace(
            self,
            cache_dir=Path(directory) if directory is not None else None,
            cache_mode=mode,
        )

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        policy = self.policy
        details: dict[str, Any] = {
            "source": self.source.kind,
            "location": self.source.location(),
            "policy": type(policy).__name__ if policy is not None else None,
            "h3_resolution": policy.h3_resolution if policy is not None else None,
            "area": self.bounds,
            "cache": {
                "mode": self.cache_mode,
                "directory": str(self.cache_dir) if self.cache_dir else None,
            },
            "execution": "reading and fitting occur only at materialize/write",
        }
        if format == "json":
            return details
        if format != "text":
            raise ValueError("explain format must be 'text' or 'json'")
        area = ",".join(str(value) for value in self.bounds) if self.bounds else "all"
        return (
            f"Source: {details['source']} ({details['location']})\n"
            f"Policy: {details['policy'] or 'not set; call canonicalize(policy=...)'}\n"
            f"Area: {area}\n"
            f"Cache: {self.cache_mode}"
            f"{f' ({self.cache_dir})' if self.cache_dir else ''}\n"
            "Reading and fitting occur only at materialize/write."
        )

    def materialize(
        self,
        output: str | Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> HazardDataset:
        """Read, fit and write one canonical Parquet file at `output`."""
        policy = self.policy
        if policy is None:
            raise ValueError("a fit policy is required; call canonicalize(policy=...)")
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress("fit", {"source": self.source.location()})
        with self.source.open(policy, self.bounds) as stream:
            write_hazard_stream(stream, destination, overwrite=True)
        rows = int(pq.ParquetFile(destination).metadata.num_rows)
        result = MaterializationResult(
            output=destination,
            source_version=_source_version(policy),
            source_cache_hits=0,
            source_cache_misses=0,
            canonical_rows=rows,
        )
        return HazardDataset.local(destination, materialization=result)

    def _automatic_output(self) -> Path:
        if self.cache_dir is None:
            raise ValueError(
                "one-chain evaluation requires a persistent cache; call "
                ".cache(path) or materialize(...) explicitly"
            )
        identity = json.dumps(
            [self.source.identity(), self.bounds, repr(self.policy)],
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return self.cache_dir / "canonical" / f"hazard-{digest}.parquet"

    def ensure_materialized(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> HazardDataset:
        """Materialize to a deterministic path under the cache directory."""
        output = self._automatic_output()
        if output.is_file() and self.cache_mode != "refresh":
            return HazardDataset.local(output)
        return self.materialize(output, progress=progress)

    def for_assets(self, assets: Any | AssetPortfolio) -> RemotePortfolioEvaluation:
        portfolio = (
            assets if isinstance(assets, AssetPortfolio) else AssetPortfolio(assets)
        )
        return RemotePortfolioEvaluation(plan=self, portfolio=portfolio)


def table_plan(
    source: str | Path | pa.Table,
    *,
    columns: CDFColumnSchema,
    policy: CDFCurveFitPolicy | None,
    probabilities: Sequence[float] | None,
    return_periods: Sequence[float] | None,
) -> BYOPlan:
    if probabilities is not None and return_periods is not None:
        raise ValueError("pass at most one of probabilities and return_periods")
    if not isinstance(source, (str, Path, pa.Table)):
        raise TypeError("source must be a path or a pyarrow Table")
    shared = (
        tuple(float(value) for value in probabilities)
        if probabilities is not None
        else None
    )
    periods = (
        tuple(float(value) for value in return_periods)
        if return_periods is not None
        else None
    )
    if periods is not None:
        # Validate now; the tail-dependent conversion happens at fit time.
        return_periods_to_probabilities(periods)
    plan = BYOPlan(_TableSource(source, columns, shared, periods))
    return plan if policy is None else plan.canonicalize(policy=policy)


def zarr_plan(
    source: Any,
    *,
    hazard_type: str,
    indicator_id: str,
    scenario: str,
    year: int,
    units: str,
    array: str | None,
    policy: CurveFitIngestPolicy | None,
    bounds: Sequence[float] | None,
    storage_options: Mapping[str, Any] | None,
) -> BYOPlan:
    location = (
        str(source) if isinstance(source, (str, Path)) else getattr(source, "path", "")
    )
    metadata = RasterMetadata(
        hazard_type=hazard_type,
        indicator_id=indicator_id,
        scenario=scenario,
        year=year,
        units=units,
        path=f"{location}#{array}" if array else str(location or indicator_id),
    )
    plan = BYOPlan(_ZarrSource(source, metadata, array, storage_options))
    if policy is not None:
        plan = plan.canonicalize(policy=policy)
    return plan if bounds is None else plan.for_area(bounds)


def raster_plan(
    paths: Mapping[int, str | Path] | Sequence[str | Path],
    *,
    return_periods: Sequence[int] | None,
    hazard_type: str,
    indicator_id: str,
    scenario: str,
    year: int,
    units: str,
    policy: CurveFitIngestPolicy | None,
    bounds: Sequence[float] | None,
    band: int,
    assumed_crs: str | None,
) -> BYOPlan:
    if isinstance(paths, Mapping):
        if return_periods is not None:
            raise ValueError("return_periods applies only to a sequence of paths")
        pairs = list(paths.items())
    else:
        if return_periods is None or len(return_periods) != len(paths):
            raise ValueError("pass one return period per GeoTIFF path")
        pairs = list(zip(return_periods, paths))
    normalized: list[tuple[int, str]] = []
    for period, path in pairs:
        if float(period) != int(period) or period <= 0:
            raise ValueError(f"return period {period!r} must be a positive integer")
        normalized.append((int(period), str(path)))
    ordered = tuple(sorted(normalized))
    if len({period for period, _ in ordered}) != len(ordered):
        raise ValueError("source return periods must be unique")
    if len(ordered) < 4:
        raise ValueError("at least four source return periods are required for fitting")
    metadata = RasterMetadata(
        hazard_type=hazard_type,
        indicator_id=indicator_id,
        scenario=scenario,
        year=year,
        units=units,
        path=";".join(path for _, path in ordered),
    )
    plan = BYOPlan(_RasterSource(ordered, metadata, band, assumed_crs))
    if policy is not None:
        plan = plan.canonicalize(policy=policy)
    return plan if bounds is None else plan.for_area(bounds)
