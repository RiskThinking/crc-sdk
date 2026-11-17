"""Small NOAA station catalogues normalized with Arrow and DuckDB."""

from __future__ import annotations

import io
from http.client import IncompleteRead
from typing import Any
from urllib.request import urlopen

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as csv  # type: ignore[import-untyped]

from ._http import http_request
from .stations import DigestReader


def _elevation(expression: str) -> str:
    """SQL nulling station-elevation sentinels.

    NOAA catalogues mark unknown elevation as -999, -999.0 or -999.9 (ISD
    history, GHCNh) or +9999 (ISD); no station lies below the Dead Sea shore
    (about -430 m) or above 9000 m.
    """
    return (
        f"CASE WHEN {expression} <= -999 OR {expression} >= 9999 "
        f"THEN NULL ELSE {expression} END"
    )


def fetch_catalogue(
    provider: str, url: str, pattern: str
) -> tuple[Any, dict[str, Any]]:
    """Stream catalogue CSVs; preserve identifiers, coordinates and record dates."""
    if provider == "isd":
        columns: tuple[str, ...] = (
            "USAF",
            "WBAN",
            "STATION NAME",
            "LAT",
            "LON",
            "ELEV(M)",
            "BEGIN",
            "END",
        )
        elevation = _elevation('try_cast("ELEV(M)" AS DOUBLE)')
        projection = f"""
            SELECT USAF || WBAN AS station, "STATION NAME" AS name,
                   try_cast(LAT AS DOUBLE) AS latitude,
                   try_cast(LON AS DOUBLE) AS longitude,
                   {elevation} AS elevation,
                   try_cast(substr("BEGIN", 1, 4) AS INTEGER) AS start_year,
                   try_cast(substr("END", 1, 4) AS INTEGER) AS end_year
            FROM catalogue
        """
    elif provider == "ghcnh":
        columns = ("GHCN_ID", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION")
        elevation = _elevation("try_cast(ELEVATION AS DOUBLE)")
        projection = f"""
            SELECT GHCN_ID AS station, NAME AS name,
                   try_cast(LATITUDE AS DOUBLE) AS latitude,
                   try_cast(LONGITUDE AS DOUBLE) AS longitude,
                   {elevation} AS elevation,
                   NULL::INTEGER AS start_year, NULL::INTEGER AS end_year
            FROM catalogue
        """
    else:
        raise ValueError("station provider must be isd or ghcnh")
    with urlopen(http_request(url), timeout=120) as response:
        digest = DigestReader(response)
        with io.BufferedReader(digest) as buffered:
            reader = csv.open_csv(
                buffered,
                read_options=csv.ReadOptions(block_size=1024 * 1024, use_threads=False),
                convert_options=csv.ConvertOptions(
                    include_columns=list(columns),
                    column_types={name: pa.string() for name in columns},
                ),
            )
            with duckdb.connect() as con:
                con.execute("SET memory_limit = '256MB'")
                con.execute("SET threads = 1")
                con.register("catalogue", reader)
                query = con.execute(
                    f"""WITH normalized AS ({projection})
                    SELECT * FROM normalized
                    WHERE latitude BETWEEN -90 AND 90
                      AND longitude BETWEEN -180 AND 180
                      AND regexp_full_match(station, ?)
                    ORDER BY station""",
                    [pattern],
                )
                table = (
                    query.to_arrow_table()
                    if hasattr(query, "to_arrow_table")
                    else query.fetch_arrow_table()
                )
        size = response.headers.get("Content-Length")
        if size and digest.byte_count != int(size):
            raise IncompleteRead(b"", int(size) - digest.byte_count)
        ids = table.column("station").to_pylist()
        if len(set(ids)) != len(ids):
            raise ValueError(f"{provider} catalogue contains duplicate station IDs")
        return table, {
            "url": url,
            "raw_sha256": digest.digest.hexdigest(),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
