from pathlib import Path

import duckdb

from crc_sdk.connectors.duckdb import DuckDBConnection, sql_identifier, sql_quote


def _secret_names(con: "duckdb.DuckDBPyConnection") -> set[str]:
    rows = con.execute("SELECT name FROM duckdb_secrets()").fetchall()
    return {row[0] for row in rows}


def test_sql_identifier_quotes_and_escapes_embedded_quotes() -> None:
    assert sql_identifier("plain") == '"plain"'
    assert sql_identifier('has"quote') == '"has""quote"'


def test_sql_quote_escapes_embedded_quotes() -> None:
    assert sql_quote("abc") == "'abc'"
    assert sql_quote("xy'z") == "'xy''z'"


def test_duckdb_connection_runs_setup_sql_on_connect() -> None:
    statement = (
        f"CREATE OR REPLACE SECRET {sql_identifier('anon_bucket')} "
        f"(TYPE S3, KEY_ID {sql_quote('')}, SECRET {sql_quote('')})"
    )
    con = DuckDBConnection(setup_sql=(statement,)).connect()
    assert "anon_bucket" in _secret_names(con)


def test_duckdb_connection_setup_sql_runs_after_extensions_load() -> None:
    # An "httpfs"-provided S3 secret type only parses if the extension is
    # already loaded by the time setup_sql runs -- proving connect()'s
    # ordering (extensions, then setup_sql), not just that setup_sql runs.
    statement = (
        f"CREATE OR REPLACE SECRET {sql_identifier('anon_bucket')} "
        f"(TYPE S3, KEY_ID {sql_quote('')}, SECRET {sql_quote('')})"
    )
    con = DuckDBConnection(extensions=("httpfs",), setup_sql=(statement,)).connect()
    assert "anon_bucket" in _secret_names(con)


def test_for_analytics_forwards_setup_sql(tmp_path: Path) -> None:
    statement = (
        f"CREATE OR REPLACE SECRET {sql_identifier('anon_bucket')} "
        f"(TYPE S3, KEY_ID {sql_quote('')}, SECRET {sql_quote('')})"
    )
    config = DuckDBConnection.for_analytics(
        tmp_path, extensions=(), setup_sql=(statement,)
    )
    assert config.setup_sql == (statement,)
    con = config.connect()
    assert "anon_bucket" in _secret_names(con)
