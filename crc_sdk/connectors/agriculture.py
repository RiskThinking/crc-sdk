"""Lazy agricultural layers from Source Cooperative."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.connectors.duckdb import (
    ArrowBatchSource,
    Bounds,
    DuckDBConnection,
    DuckDBPipeline,
    default_work_dir,
    ensure_extensions,
    sql_quote,
)

USDA_CDL_REPOSITORY = "usda-cropland-data-layer/v0.1.0.icechunk"
USDA_CDL_URI = f"https://data.source.coop/chill/{USDA_CDL_REPOSITORY}"
FTW_VECTOR_BASE = (
    "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/"
    "predictions/vectors/alpha/results-by-admin-conf"
)

_COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")


def _validate_bounds(bounds: Sequence[float]) -> Bounds:
    if len(bounds) != 4:
        raise ValueError("bounds must contain min_lon, min_lat, max_lon, max_lat")
    result = tuple(float(value) for value in bounds)
    minimum_x, minimum_y, maximum_x, maximum_y = result
    if not (
        -180 <= minimum_x < maximum_x <= 180 and -90 <= minimum_y < maximum_y <= 90
    ):
        raise ValueError("bounds must be ordered WGS84 longitude/latitude values")
    return result


def _require_cdl_extra() -> tuple[Any, Any, Any]:
    if sys.version_info < (3, 12):
        raise ImportError("USDA CDL access requires Python 3.12 or newer")
    try:
        import icechunk
        import pyproj
        import zarr
    except ImportError as error:
        raise ImportError(
            "USDA CDL access requires `pip install crc-sdk[agriculture]`"
        ) from error
    return icechunk, pyproj, zarr


def _open_cdl_group(source: USDACropland) -> tuple[Any, Any, Any, Any, Any]:
    icechunk, pyproj, zarr = _require_cdl_extra()
    storage = icechunk.s3_storage(
        bucket="chill",
        prefix=source.repository,
        endpoint_url="https://data.source.coop",
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    repository = icechunk.Repository.open(storage)
    session = repository.readonly_session(source.branch)
    root = zarr.open_group(store=session.store, mode="r")
    group = root[source.group]
    return group, repository, session, pyproj, zarr


def _class_names(attrs: Any) -> dict[int, str]:
    raw_names = attrs.get("class_names")
    if isinstance(raw_names, dict):
        return {int(code): str(name) for code, name in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        flags = attrs.get("flag_values")
        codes = flags if isinstance(flags, (list, tuple)) else range(len(raw_names))
        return {int(code): str(name) for code, name in zip(codes, raw_names)}
    meanings = attrs.get("flag_meanings")
    flags = attrs.get("flag_values")
    if isinstance(meanings, str) and isinstance(flags, (list, tuple)):
        return {
            int(code): name.replace("_", " ")
            for code, name in zip(flags, meanings.split())
        }
    return {}


@dataclass(frozen=True)
class USDACropland:
    """Immutable request for the USDA NASS Cropland Data Layer."""

    repository: str = USDA_CDL_REPOSITORY
    branch: str = "main"
    group: str = "30m"
    bounds: Bounds | None = None
    selected_years: tuple[int, ...] | None = None
    crop_codes: tuple[int, ...] | None = None
    include_background: bool = False

    def resolution(self, value: str | int) -> USDACropland:
        normalized = f"{int(value)}m" if isinstance(value, int) else value
        if normalized not in ("10m", "30m"):
            raise ValueError("CDL resolution must be '10m' or '30m'")
        return replace(self, group=normalized)

    def for_area(self, bounds: Sequence[float]) -> USDACropland:
        return replace(self, bounds=_validate_bounds(bounds))

    def years(self, values: int | Sequence[int]) -> USDACropland:
        years = (values,) if isinstance(values, int) else tuple(int(v) for v in values)
        if not years or len(set(years)) != len(years):
            raise ValueError("years must be a non-empty unique selection")
        return replace(self, selected_years=tuple(sorted(years)))

    def classes(self, values: Sequence[int]) -> USDACropland:
        codes = tuple(sorted({int(value) for value in values}))
        if not codes or any(not 0 <= code <= 255 for code in codes):
            raise ValueError("crop class codes must be unique uint8 values")
        return replace(self, crop_codes=codes)

    def with_background(self, include: bool = True) -> USDACropland:
        return replace(self, include_background=include)

    def scan(self, *, batch_rows: int = 65_536) -> USDACroplandScan:
        if self.bounds is None:
            raise ValueError("for_area(...) is required before scanning USDA CDL")
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        return USDACroplandScan(self, batch_rows)


@dataclass(frozen=True)
class USDACroplandScan:
    request: USDACropland
    batch_rows: int

    @property
    def schema(self) -> Any:
        return pa.schema(
            [
                ("longitude", pa.float64()),
                ("latitude", pa.float64()),
                ("year", pa.int16()),
                ("crop_code", pa.uint8()),
                ("crop_name", pa.string()),
                ("source_path", pa.string()),
            ]
        )

    def relation(self, *, connection: DuckDBConnection | None = None) -> Any:
        source = ArrowBatchSource(self.schema, self._batches)
        return source.relation(connection=connection)

    def pipeline(self, *, connection: DuckDBConnection | None = None) -> DuckDBPipeline:
        return DuckDBPipeline(self, connection=connection)

    def _batches(self) -> Iterator[Any]:
        request = self.request
        group, repository, session, pyproj, _ = _open_cdl_group(request)
        # Keep repository/session alive for the generator's full lifetime.
        _keepalive = (repository, session)
        crop = group["crop_type"]
        years = np.asarray(group["year"][:], dtype=np.int64)
        xs = np.asarray(group["x"][:], dtype=np.float64)
        ys = np.asarray(group["y"][:], dtype=np.float64)

        requested_years = request.selected_years or tuple(int(v) for v in years)
        year_lookup = {int(value): index for index, value in enumerate(years)}
        missing = sorted(set(requested_years) - set(year_lookup))
        if missing:
            raise ValueError(
                f"years {missing} are unavailable in USDA CDL group {request.group}"
            )
        year_indices = tuple(year_lookup[value] for value in requested_years)

        assert request.bounds is not None
        to_native = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:5070", always_xy=True
        )
        native_bounds = to_native.transform_bounds(*request.bounds, densify_pts=21)
        minimum_x, minimum_y, maximum_x, maximum_y = native_bounds
        x_hits = np.flatnonzero((xs >= minimum_x) & (xs <= maximum_x))
        y_hits = np.flatnonzero((ys >= minimum_y) & (ys <= maximum_y))
        if not len(x_hits) or not len(y_hits):
            return
        column_start, column_stop = int(x_hits[0]), int(x_hits[-1]) + 1
        row_start, row_stop = int(y_hits[0]), int(y_hits[-1]) + 1
        width = column_stop - column_start
        target_pixels = max(1, self.batch_rows // len(year_indices))
        columns_per_batch = min(width, target_pixels)
        rows_per_batch = max(1, target_pixels // columns_per_batch)
        names = _class_names(crop.attrs)
        allowed: Any = (
            np.asarray(request.crop_codes, dtype=np.uint8)
            if request.crop_codes is not None
            else None
        )
        to_wgs84 = pyproj.Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

        for row in range(row_start, row_stop, rows_per_batch):
            row_end = min(row + rows_per_batch, row_stop)
            for column in range(column_start, column_stop, columns_per_batch):
                column_end = min(column + columns_per_batch, column_stop)
                values = np.stack(
                    [
                        np.asarray(
                            crop[index, row:row_end, column:column_end],
                            dtype=np.uint8,
                        )
                        for index in year_indices
                    ]
                )
                native_x: Any
                native_y: Any
                native_x, native_y = np.meshgrid(xs[column:column_end], ys[row:row_end])
                longitude, latitude = to_wgs84.transform(native_x, native_y)
                pixel_count = native_x.size
                flat_values = values.reshape(-1)
                mask = np.ones(flat_values.shape, dtype=bool)
                longitude_values = np.asarray(longitude).reshape(-1)
                latitude_values = np.asarray(latitude).reshape(-1)
                minimum_lon, minimum_lat, maximum_lon, maximum_lat = request.bounds
                inside = (
                    (longitude_values >= minimum_lon)
                    & (longitude_values <= maximum_lon)
                    & (latitude_values >= minimum_lat)
                    & (latitude_values <= maximum_lat)
                )
                mask &= np.tile(inside, len(year_indices))
                background_selected = allowed is not None and 0 in allowed
                if not request.include_background and not background_selected:
                    mask &= flat_values != 0
                if allowed is not None:
                    mask &= np.isin(flat_values, allowed)
                if not np.any(mask):
                    continue
                selected = flat_values[mask]
                longitudes = np.tile(longitude_values, len(year_indices))[mask]
                latitudes = np.tile(latitude_values, len(year_indices))[mask]
                selected_year_values = np.repeat(requested_years, pixel_count)[mask]
                selected_names = [
                    names.get(int(code), f"Class {int(code)}") for code in selected
                ]
                yield pa.record_batch(
                    [
                        pa.array(longitudes),
                        pa.array(latitudes),
                        pa.array(selected_year_values, type=pa.int16()),
                        pa.array(selected, type=pa.uint8()),
                        pa.array(selected_names),
                        pa.array([USDA_CDL_URI] * int(np.count_nonzero(mask))),
                    ],
                    schema=self.schema,
                )


@dataclass(frozen=True)
class FTWFields:
    """Immutable AOI request for FTW field-boundary GeoParquet."""

    base_uri: str = FTW_VECTOR_BASE
    country_code: str | None = None
    bounds: Bounds | None = None
    selected_years: tuple[int, ...] = (2025,)
    minimum_confidence: float = 0.0

    def in_country(self, code: str) -> FTWFields:
        normalized = code.upper()
        if not _COUNTRY_CODE.fullmatch(normalized):
            raise ValueError("country code must be a two-letter ISO 3166-1 code")
        return replace(self, country_code=normalized)

    def for_area(self, bounds: Sequence[float]) -> FTWFields:
        return replace(self, bounds=_validate_bounds(bounds))

    def years(self, values: int | Sequence[int]) -> FTWFields:
        years = (values,) if isinstance(values, int) else tuple(int(v) for v in values)
        if not years or any(year not in (2024, 2025) for year in years):
            raise ValueError("FTW field-boundary years must be 2024 and/or 2025")
        return replace(self, selected_years=tuple(sorted(set(years))))

    def confidence_at_least(self, value: float) -> FTWFields:
        confidence = float(value)
        if not 0 <= confidence <= 100:
            raise ValueError("confidence must be between zero and 100")
        return replace(self, minimum_confidence=confidence)

    def scan(self) -> FTWFieldScan:
        if self.country_code is None:
            raise ValueError("in_country(...) is required before scanning FTW fields")
        if self.bounds is None:
            raise ValueError("for_area(...) is required before scanning FTW fields")
        return FTWFieldScan(self)


@dataclass(frozen=True)
class FTWFieldScan:
    request: FTWFields

    @property
    def uri(self) -> str:
        assert self.request.country_code is not None
        return (
            f"{self.request.base_uri}/"
            f"admin:country_code={self.request.country_code}/*.parquet"
        )

    def relation(self, *, connection: DuckDBConnection | None = None) -> Any:
        active_config = connection or DuckDBConnection.for_analytics(
            default_work_dir(), extensions=("spatial", "httpfs")
        )
        active = active_config.connect()
        ensure_extensions(active, "spatial", "httpfs")
        scope = self.request.base_uri.rstrip("/")
        active.execute(
            f"""
            CREATE OR REPLACE TEMPORARY SECRET crc_source_cooperative_ftw (
                TYPE S3,
                PROVIDER config,
                REGION 'us-west-2',
                URL_STYLE 'path',
                KEY_ID '',
                SECRET '',
                SCOPE {sql_quote(scope)}
            )
            """
        )
        years = ", ".join(str(year) for year in self.request.selected_years)
        assert self.request.bounds is not None
        minimum_x, minimum_y, maximum_x, maximum_y = self.request.bounds
        query = f"""
            SELECT
                id,
                CAST(EXTRACT(year FROM "determination:datetime") AS INTEGER) AS year,
                CAST(confidence AS DOUBLE) AS confidence,
                CAST("metrics:area" AS DOUBLE) AS area_m2,
                CAST("metrics:perimeter" AS DOUBLE) AS perimeter_m,
                geometry,
                {sql_quote(self.request.country_code)} AS country_code,
                filename AS source_path
            FROM read_parquet(
                {sql_quote(self.uri)},
                filename = true,
                hive_partitioning = true
            )
            WHERE confidence >= {self.request.minimum_confidence}
              AND EXTRACT(year FROM "determination:datetime") IN ({years})
              AND bbox.xmax >= {minimum_x}
              AND bbox.xmin <= {maximum_x}
              AND bbox.ymax >= {minimum_y}
              AND bbox.ymin <= {maximum_y}
              AND ST_Intersects(
                    geometry,
                    ST_MakeEnvelope(
                        {minimum_x}, {minimum_y}, {maximum_x}, {maximum_y}
                    )
                  )
        """
        return active.sql(query)

    def pipeline(self, *, connection: DuckDBConnection | None = None) -> DuckDBPipeline:
        return DuckDBPipeline(self, connection=connection)
