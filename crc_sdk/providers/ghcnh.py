"""Projected, range-read NOAA GHCNh Parquet observations with source-aware QC."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import IncompleteRead
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import fsspec  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .stations import aggregate_daily, empty_daily

VERSION = "ghcnh-v1"
URL = "https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/access/by-year"
LICENCE = "CC0-1.0"
ATTRIBUTION = (
    "Derived from NOAA Global Historical Climatology Network hourly (GHCNh), "
    "accessed through NOAA Open Data Dissemination on AWS. Quality-filtered "
    "and aggregated by crc-sdk; modified data, no NOAA endorsement implied."
)
STATION_PATTERN = r"[A-Z]{2}[A-Z0-9-]{9}"
STATION_DESCRIPTION = "eleven-character GHCN station IDs"
FIRST_YEAR = 1768
LAST_YEAR = datetime.now(timezone.utc).year - 1
COLUMNS = (
    "STATION",
    "DATE",
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "temperature",
    "temperature_Quality_Code",
    "temperature_Source_Code",
)
ISD_SOURCES = ("313", "314", "315", "322", "335", "343", "344", "346")
GOOD_SOURCES = (*ISD_SOURCES, "220", "221", "222", "223", "347", "348")


@dataclass(frozen=True)
class GHCNhCompleteness:
    """GHCNh quality overrides narrow the source-aware accepted flag set."""

    minimum_hour_fraction: float = 0.75
    minimum_day_fraction: float = 0.9
    quality_codes: tuple[str, ...] = ("", "1", "5")

    def __post_init__(self) -> None:
        if any(
            not 0 < v <= 1
            for v in (
                self.minimum_hour_fraction,
                self.minimum_day_fraction,
            )
        ):
            raise ValueError("completeness fractions must be in (0, 1]")
        if not self.quality_codes or any(
            code not in ("", "1", "5") for code in self.quality_codes
        ):
            raise ValueError("GHCNh quality_codes must be a subset of ('', '1', '5')")


Completeness = GHCNhCompleteness


def reduce_daily(
    source: Any,
    station: str,
    year: int,
    completeness: GHCNhCompleteness,
    *,
    memory_limit: str = "256MB",
    temporary_directory: str,
) -> tuple[Any, str]:
    """Project eight columns and stream bounded Arrow batches into DuckDB."""
    parquet = pq.ParquetFile(source)
    schema = pa.schema([parquet.schema_arrow.field(name) for name in COLUMNS])
    digest = hashlib.sha256()

    def batches() -> Iterator[Any]:
        for batch in parquet.iter_batches(
            columns=list(COLUMNS),
            batch_size=8192,
            use_threads=False,
        ):
            digest.update(batch.serialize().to_pybytes())
            yield batch

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    table = aggregate_daily(
        reader,
        """
        SELECT STATION AS station, try_cast(DATE AS TIMESTAMP) AS time,
               LATITUDE AS latitude, LONGITUDE AS longitude,
               nullif(ELEVATION, -999.9) AS elevation,
               CASE WHEN coalesce(trim(temperature_Quality_Code), '')
                             IN (SELECT unnest(?))
                         AND (
                             coalesce(trim(temperature_Quality_Code), '') = ''
                             OR (trim(temperature_Quality_Code) = '1'
                                 AND trim(temperature_Source_Code)
                                     IN (SELECT unnest(?)))
                             OR (trim(temperature_Quality_Code) = '5'
                                 AND trim(temperature_Source_Code)
                                     IN (SELECT unnest(?)))
                         )
                         AND isfinite(temperature)
                         AND temperature BETWEEN -93.2 AND 61.8
                    THEN temperature END AS value
        FROM observations
        """,
        [list(completeness.quality_codes), list(GOOD_SOURCES), list(ISD_SOURCES)],
        station,
        year,
        memory_limit=memory_limit,
        temporary_directory=temporary_directory,
    )
    return table, digest.hexdigest()


def _fetch_daily(
    station: str,
    year: int,
    completeness: GHCNhCompleteness,
    *,
    temporary_directory: str,
    memory_limit: str,
) -> tuple[Any, dict[str, Any]]:
    url = f"{URL}/{year}/parquet/GHCNh_{station}_{year}.parquet"
    try:
        with urlopen(Request(url, method="HEAD"), timeout=120) as response:
            etag = response.headers.get("ETag")
            modified = response.headers.get("Last-Modified")
            size = int(response.headers["Content-Length"])
    except HTTPError as error:
        if error.code != 404:
            raise
        return empty_daily(), {
            "url": url,
            "missing": True,
            "etag": None,
            "last_modified": None,
            "projection_sha256": None,
        }
    if not etag:
        raise RuntimeError(f"GHCNh object has no ETag: {url}")
    # Conditional range requests prevent mixing versions of a changing object.
    with fsspec.open(
        url,
        "rb",
        block_size=64 * 1024,
        cache_type="readahead",
        headers={"If-Match": etag},
    ) as source:
        table, checksum = reduce_daily(
            source,
            station,
            year,
            completeness,
            temporary_directory=temporary_directory,
            memory_limit=memory_limit,
        )
    return table, {
        "url": url,
        "etag": etag,
        "last_modified": modified,
        "size_bytes": size,
        "projection_sha256": checksum,
        "projected_columns": COLUMNS,
    }


def fetch_daily(
    station: str,
    year: int,
    completeness: GHCNhCompleteness,
    *,
    temporary_directory: str,
    memory_limit: str,
) -> tuple[Any, dict[str, Any]]:
    """Retry transient errors; retain only reduced daily values in the cache."""
    for attempt in range(3):
        try:
            return _fetch_daily(
                station,
                year,
                completeness,
                temporary_directory=temporary_directory,
                memory_limit=memory_limit,
            )
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError, ConnectionError, IncompleteRead):
            if attempt == 2:
                raise
        time.sleep(0.5 * 2**attempt)
    raise AssertionError("unreachable")


CATALOGUE_URL = (
    "https://noaa-ghcnh-pds.s3.amazonaws.com/hourly/doc/ghcnh-station-list.csv"
)
