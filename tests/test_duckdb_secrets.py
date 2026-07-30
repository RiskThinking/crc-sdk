from pathlib import Path

import duckdb
import pytest

from crc_sdk.connectors.duckdb import (
    DuckDBConnection,
    DuckDBSecret,
    apply_secret,
    gcs_hmac_secret_from_env,
    secret_sql,
    sql_identifier,
)


def _secret_names(con: "duckdb.DuckDBPyConnection") -> set[str]:
    rows = con.execute("SELECT name FROM duckdb_secrets()").fetchall()
    return {row[0] for row in rows}


def test_sql_identifier_quotes_and_escapes_embedded_quotes() -> None:
    assert sql_identifier("plain") == '"plain"'
    assert sql_identifier('has"quote') == '"has""quote"'


def test_duckdb_secret_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        DuckDBSecret(name="", type="GCS")


def test_duckdb_secret_rejects_empty_type() -> None:
    with pytest.raises(ValueError):
        DuckDBSecret(name="my_secret", type="")


def test_apply_secret_declares_a_usable_duckdb_secret() -> None:
    con = duckdb.connect()
    secret = DuckDBSecret(
        name="test_secret",
        type="S3",
        params={"KEY_ID": "abc", "SECRET": "xyz", "ENDPOINT": "example.com"},
    )
    apply_secret(con, secret)
    assert "test_secret" in _secret_names(con)


def test_secret_sql_renders_provider_as_a_bare_keyword_not_a_string_literal() -> None:
    # PROVIDER is unlike every other secret option in DuckDB's grammar -- a
    # bare identifier, not a quoted string -- and providers like
    # `credential_chain` validate against live cloud credentials at CREATE
    # SECRET time, so this stays a pure string-building assertion rather
    # than an executed one.
    secret = DuckDBSecret(name="cred_chain", type="S3", provider="credential_chain")
    statement = secret_sql(secret)
    assert "PROVIDER credential_chain" in statement
    assert "'credential_chain'" not in statement


def test_secret_sql_quotes_every_param_value_as_a_string_literal() -> None:
    secret = DuckDBSecret(
        name="my_secret", type="GCS", params={"key_id": "abc", "secret": "xy'z"}
    )
    statement = secret_sql(secret)
    assert "KEY_ID 'abc'" in statement
    assert "SECRET 'xy''z'" in statement


def test_gcs_hmac_secret_from_env_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GCS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("GCS_ACCESS_SECRET", raising=False)
    assert gcs_hmac_secret_from_env() is None


def test_gcs_hmac_secret_from_env_returns_none_when_only_one_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCS_ACCESS_KEY", "key-only")
    monkeypatch.delenv("GCS_ACCESS_SECRET", raising=False)
    assert gcs_hmac_secret_from_env() is None


def test_gcs_hmac_secret_from_env_builds_secret_when_both_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCS_ACCESS_KEY", "my-key")
    monkeypatch.setenv("GCS_ACCESS_SECRET", "my-secret")
    secret = gcs_hmac_secret_from_env()
    assert secret is not None
    assert secret.type == "GCS"
    assert secret.params == {"KEY_ID": "my-key", "SECRET": "my-secret"}


def test_duckdb_connection_applies_secrets_on_connect() -> None:
    secret = DuckDBSecret(
        name="anon_bucket", type="S3", params={"KEY_ID": "", "SECRET": ""}
    )
    con = DuckDBConnection(secrets=(secret,)).connect()
    assert "anon_bucket" in _secret_names(con)


def test_for_analytics_forwards_secrets(tmp_path: Path) -> None:
    secret = DuckDBSecret(
        name="anon_bucket", type="S3", params={"KEY_ID": "", "SECRET": ""}
    )
    config = DuckDBConnection.for_analytics(tmp_path, extensions=(), secrets=(secret,))
    assert config.secrets == (secret,)
    con = config.connect()
    assert "anon_bucket" in _secret_names(con)
