"""Streaming NOAA ISD CSV decoding and UTC daily temperature reduction."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from http.client import IncompleteRead
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as csv  # type: ignore[import-untyped]

from .stations import DigestReader, aggregate_daily, empty_daily

ISD_VERSION = "isd-2025-08-24"
ISD_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com"
ISD_LICENCE = "NOAA NODD open-use terms; originator terms may apply"
ISD_ATTRIBUTION = (
    "Derived from NOAA Integrated Surface Database (ISD), accessed through "
    "NOAA Open Data Dissemination on AWS. Quality-filtered and aggregated "
    "by crc-sdk; modified data, no NOAA endorsement implied."
)


@dataclass(frozen=True)
class ISDCompleteness:
    """Coverage gates count distinct UTC hour bins and all calendar days."""

    minimum_hour_fraction: float = 0.75
    minimum_day_fraction: float = 0.9
    quality_codes: tuple[str, ...] = ("1", "5")

    def __post_init__(self) -> None:
        for value in (self.minimum_hour_fraction, self.minimum_day_fraction):
            if not 0 < value <= 1:
                raise ValueError("completeness fractions must be in (0, 1]")
        if not self.quality_codes or any(
            code not in ("0", "1", "4", "5", "9", "A", "C", "I", "M", "P", "R", "U")
            for code in self.quality_codes
        ):
            raise ValueError(
                "quality_codes must contain accepted ISD temperature codes"
            )


def validate_recipe(recipe: str) -> str:
    if recipe not in ("temperature_max", "temperature_min"):
        raise ValueError("ISD recipe must be temperature_max or temperature_min")
    return recipe


def reduce_daily(
    source: BinaryIO,
    station: str,
    year: int,
    completeness: ISDCompleteness,
    *,
    memory_limit: str = "256MB",
    temporary_directory: str,
    expected_bytes: int | None = None,
) -> tuple[Any, str]:
    """Consume a CSV once, retaining only a station/year's daily summaries.

    Arrow reads 1 MiB blocks; DuckDB owns aggregation and can spill to the
    caller's scratch directory. No hourly CSV or hourly Arrow table is saved.
    """
    columns = ("STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "TMP")
    digest = DigestReader(source)
    with io.BufferedReader(digest) as buffered:
        reader = csv.open_csv(
            buffered,
            read_options=csv.ReadOptions(block_size=1024 * 1024, use_threads=False),
            convert_options=csv.ConvertOptions(
                include_columns=list(columns),
                column_types={name: pa.string() for name in columns},
            ),
        )
        result = aggregate_daily(
            reader,
            """
            SELECT STATION AS station, try_cast(DATE AS TIMESTAMP) AS time,
                   try_cast(LATITUDE AS DOUBLE) AS latitude,
                   try_cast(LONGITUDE AS DOUBLE) AS longitude,
                   nullif(try_cast(ELEVATION AS DOUBLE), 9999) AS elevation,
                   CASE WHEN split_part(TMP, ',', 2) IN (SELECT unnest(?))
                        AND try_cast(split_part(TMP, ',', 1) AS INTEGER)
                            BETWEEN -932 AND 618
                        THEN try_cast(split_part(TMP, ',', 1) AS DOUBLE) / 10.0
                   END AS value
            FROM observations
            """,
            [list(completeness.quality_codes)],
            station,
            year,
            memory_limit=memory_limit,
            temporary_directory=temporary_directory,
        )
    if expected_bytes is not None and digest.byte_count != expected_bytes:
        raise IncompleteRead(b"", expected_bytes - digest.byte_count)
    return result, digest.digest.hexdigest()


def _fetch_daily(
    station: str,
    year: int,
    completeness: ISDCompleteness,
    *,
    temporary_directory: str,
    memory_limit: str,
) -> tuple[Any, dict[str, Any]]:
    url = f"{ISD_URL}/{year}/{station}.csv"
    try:
        response = urlopen(url, timeout=120)
    except HTTPError as error:
        if error.code != 404:
            raise
        return empty_daily(), {
            "url": url,
            "raw_sha256": None,
            "missing": True,
            "etag": None,
            "last_modified": None,
        }
    with response:
        table, checksum = reduce_daily(
            response,
            station,
            year,
            completeness,
            temporary_directory=temporary_directory,
            memory_limit=memory_limit,
            expected_bytes=int(response.headers["Content-Length"])
            if response.headers.get("Content-Length")
            else None,
        )
        return table, {
            "url": url,
            "raw_sha256": checksum,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def fetch_daily(
    station: str,
    year: int,
    completeness: ISDCompleteness,
    *,
    temporary_directory: str,
    memory_limit: str,
) -> tuple[Any, dict[str, Any]]:
    """Retry transient transport failures before publishing a daily reduction."""
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


# Provider identity used by the shared station observation workflow.
VERSION = ISD_VERSION
URL = ISD_URL
LICENCE = ISD_LICENCE
ATTRIBUTION = ISD_ATTRIBUTION
STATION_PATTERN = r"[0-9]{11}"
STATION_DESCRIPTION = "eleven-digit USAF + WBAN IDs"
FIRST_YEAR = 1901
LAST_YEAR = 2024
Completeness = ISDCompleteness

CATALOGUE_URL = "https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv"
