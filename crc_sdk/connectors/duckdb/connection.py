"""DuckDB connection and extension lifecycle."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional
import os

from duckdb import DuckDBPyConnection


@dataclass(frozen=True)
class DuckDBConnection:
    """Configuration for a lazily created DuckDB connection."""

    database: Optional[str] = None
    read_only: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)

    def connect(self):
        """Create a connection without imposing SDK-level resource limits."""
        try:
            import duckdb
        except ImportError as error:
            raise ImportError(
                "DuckDB support requires `pip install crc-sdk[connectors]`"
            ) from error

        return duckdb.connect(
            self.database or ":memory:",
            read_only=self.read_only,
            config=dict(self.config),
        )


def sql_quote(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


class DuckDBStreamEngine:
    """Configures DuckDB runtime for streaming large datasets with explicit memory bounds."""

    @staticmethod
    def create_streaming_connection(
        max_memory: str = "4GB",
        threads: Optional[int] = None,
        temp_dir: Optional[str] = None,
    ):
        con = DuckDBConnection().connect()

        # Prevent insertion ordering overhead for streaming queries
        con.execute("SET preserve_insertion_order = false;")
        con.execute(f"SET max_memory = {sql_quote(max_memory)};")

        if threads is not None:
            con.execute(f"SET threads = {int(threads)};")

        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            con.execute(f"SET temp_directory = {sql_quote(temp_dir)};")

        return con

    @staticmethod
    def execute_to_parquet_stream(
        con: DuckDBPyConnection,
        query_sql: str,
        output_parquet_path: str,
        compression: str = "SNAPPY",
    ) -> None:
        """Executes query directly to Parquet using streaming out-of-core write."""

        quoted_output = sql_quote(output_parquet_path)
        quoted_compression = sql_quote(compression)

        copy_sql = f"""
            COPY ({query_sql})
            TO {quoted_output}
            (FORMAT PARQUET, COMPRESSION {quoted_compression});
        """
        con.execute(copy_sql)
