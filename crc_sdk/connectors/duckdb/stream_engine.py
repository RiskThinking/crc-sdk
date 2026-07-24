"""Memory-bounded, streaming DuckDB execution wrapper."""

import os
from typing import Optional

import duckdb


class DuckDBStreamEngine:
    """Configures DuckDB runtime for streaming large datasets with explicit memory bounds."""

    @staticmethod
    def create_streaming_connection(
        max_memory: str = "4GB",
        threads: Optional[int] = None,
        temp_dir: Optional[str] = None,
    ) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()

        # Prevent insertion ordering overhead for streaming queries
        con.execute("SET preserve_insertion_order = false;")
        con.execute(f"SET max_memory = '{max_memory}';")

        if threads:
            con.execute(f"SET threads = {threads};")

        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            con.execute(f"SET temp_directory = '{temp_dir}';")

        return con

    @staticmethod
    def execute_to_parquet_stream(
        con: duckdb.DuckDBPyConnection,
        query_sql: str,
        output_parquet_path: str,
        compression: str = "SNAPPY",
    ) -> None:
        """Executes query directly to Parquet using streaming out-of-core write."""
        copy_sql = f"""
            COPY ({query_sql})
            TO '{output_parquet_path}'
            (FORMAT PARQUET, COMPRESSION '{compression}');
        """
        con.execute(copy_sql)
