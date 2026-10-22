"""Shared bounded aggregation of normalized station observation batches."""

from __future__ import annotations

import hashlib
import io
from typing import Any, BinaryIO

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]


def aggregate_daily(
    observations: Any,
    projection: str,
    parameters: list[Any],
    station: str,
    year: int,
    *,
    memory_limit: str,
    temporary_directory: str,
) -> Any:
    """Reduce source-decoded station/time/value columns into UTC daily summaries."""
    with duckdb.connect() as con:
        con.execute("SET memory_limit = ?", [memory_limit])
        con.execute("SET temp_directory = ?", [temporary_directory])
        con.execute("SET threads = 1")
        con.register("observations", observations)
        query = con.execute(
            f"""
            WITH normalized AS ({projection}), selected AS (
                SELECT * FROM normalized WHERE station = ? AND year(time) = ?
            )
            SELECT station, cast(time AS DATE) AS day,
                   min(latitude) AS latitude, max(latitude) AS latitude_max,
                   min(longitude) AS longitude, max(longitude) AS longitude_max,
                   min(elevation) AS elevation, max(elevation) AS elevation_max,
                   count(*) AS reports, count(value) AS accepted_reports,
                   count(DISTINCT date_trunc('hour', time))
                       FILTER (WHERE value IS NOT NULL) AS valid_hours,
                   min(value) AS temperature_min, max(value) AS temperature_max
            FROM selected GROUP BY station, day ORDER BY day
            """,
            [*parameters, station, year],
        )
        result = (
            query.to_arrow_table()
            if hasattr(query, "to_arrow_table")
            else query.fetch_arrow_table()
        )
    if result.num_rows == 0:
        raise ValueError(f"{station}/{year} contains no matching observations")
    return result


def empty_daily() -> Any:
    """Typed empty reduction for an absent archive object (HTTP 404 only)."""
    fields = [("station", pa.string()), ("day", pa.date32())]
    fields += [
        (name, pa.float64())
        for name in (
            "latitude",
            "latitude_max",
            "longitude",
            "longitude_max",
            "elevation",
            "elevation_max",
        )
    ]
    fields += [
        (name, pa.int64())
        for name in (
            "reports",
            "accepted_reports",
            "valid_hours",
        )
    ]
    fields += [("temperature_min", pa.float64()), ("temperature_max", pa.float64())]
    return pa.Table.from_batches([], schema=pa.schema(fields))


class DigestReader(io.RawIOBase):
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, target: Any) -> int:
        data = self.source.read(len(target))
        target[: len(data)] = data
        self.digest.update(data)
        self.byte_count += len(data)
        return len(data)
