"""Real byte identity and cache policies at the provider acquisition boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from crc_sdk import Cache, Runtime
from crc_sdk._dataset_schema import ResourceIdentity
from crc_sdk._graph import ExecutionScope
from crc_sdk._source_io import (
    _json,
    check_release,
    fetch,
    fetch_custom,
    fetch_json,
    fetch_optional,
    release_snapshot,
)


def test_uri_cache_reuse_refresh_offline_and_scope_ownership(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"value": 1}')
    uri = source.as_uri()
    cache = tmp_path / "cache"
    with ExecutionScope(Runtime(cache=Cache.reuse(cache))) as scope:
        old, first = fetch_json(scope, uri, license="CC0", attribution="Fixture")
        assert old == {"value": 1}
        owned, again = fetch(scope, uri)
        assert first == again
        assert first.retrieved_at is not None and first.license == "CC0"
        source.write_text('{"value": 2}')
        assert fetch_json(scope, uri)[0] == old
        assert owned.exists()
    assert not owned.exists()
    for policy in (Cache.reuse(cache), Cache.offline(cache)):
        with ExecutionScope(Runtime(cache=policy)) as scope:
            assert fetch_json(scope, uri)[0] == old
    with ExecutionScope(Runtime(cache=Cache.refresh(cache))) as scope:
        assert fetch_json(scope, uri)[0] == {"value": 2}
    source.unlink()
    with ExecutionScope(Runtime(cache=Cache.offline(cache))) as scope:
        assert fetch_json(scope, uri)[0] == {"value": 2}
        # Explicit immutable revisions coexist with the current URI snapshot.
        assert fetch_json(scope, uri, expected_sha256=first.sha256)[0] == old


def test_ephemeral_does_not_populate_other_execution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"first")
    with ExecutionScope() as scope:
        assert fetch(scope, source.as_uri())[0].read_bytes() == b"first"
    source.write_bytes(b"second")
    with ExecutionScope() as scope:
        assert fetch(scope, source.as_uri())[0].read_bytes() == b"second"
    with ExecutionScope(Runtime(cache=Cache.offline(tmp_path / "empty"))) as scope:
        with pytest.raises(FileNotFoundError, match="offline"):
            fetch(scope, source.as_uri())


def test_corrupt_cache_never_becomes_provider_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"scientific data")
    cache = tmp_path / "cache"
    with ExecutionScope(Runtime(cache=Cache.reuse(cache))) as scope:
        _, resource = fetch(scope, source.as_uri())
    cached = cache / "source-objects-1.0" / resource.sha256.removeprefix("sha256:")
    cached.write_bytes(b"corrupt")
    with ExecutionScope(Runtime(cache=Cache.offline(cache))) as scope:
        with pytest.raises(OSError, match="content identity mismatch"):
            fetch(scope, source.as_uri())


def test_negative_native_responses_are_verified_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def missing(uri: str, target: Path) -> int:
        nonlocal calls
        calls += 1
        target.write_bytes(b"native not found response")
        return 404

    monkeypatch.setattr("crc_sdk._source_io._download", missing)
    cache = tmp_path / "cache"
    with ExecutionScope(Runtime(cache=Cache.reuse(cache))) as scope:
        path, original = fetch_optional(scope, "https://fixture/station/year")
        assert path is None
        assert fetch_optional(scope, original.uri)[0] is None
        assert calls == 1
    with ExecutionScope(Runtime(cache=Cache.offline(cache))) as scope:
        assert fetch_optional(scope, original.uri) == (None, original)
        assert calls == 1


def test_custom_bounded_view_is_pinned_before_decode_and_cleans_failure(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    calls = 0

    def acquire(target: Path) -> None:
        nonlocal calls
        calls += 1
        target.write_bytes(b"bounded native grid view")

    with ExecutionScope(Runtime(cache=Cache.reuse(cache))) as scope:
        path, identity = fetch_custom(scope, "https://fixture/#window=1", acquire)
        assert path.read_bytes() == b"bounded native grid view"
        assert (
            identity.sha256 == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        assert fetch_custom(scope, identity.uri, acquire)[1] == identity
    with ExecutionScope(Runtime(cache=Cache.offline(cache))) as scope:
        assert fetch_custom(scope, identity.uri, acquire)[0].read_bytes()
    assert calls == 1

    def fail(target: Path) -> None:
        target.write_bytes(b"partial")
        raise ConnectionError("fixture disconnected")

    with ExecutionScope(Runtime(cache=Cache.reuse(cache))) as scope:
        with pytest.raises(ConnectionError, match="disconnected"):
            fetch_custom(scope, "https://fixture/#window=2", fail)
        assert not list(scope.directory.glob("source-download-*"))
    with ExecutionScope(Runtime(cache=Cache.offline(cache))) as scope:
        with pytest.raises(FileNotFoundError):
            fetch_custom(scope, "https://fixture/#window=2", acquire)


@pytest.mark.parametrize("text", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'])
def test_native_json_rejects_ambiguous_or_nonfinite_values(text: str) -> None:
    with pytest.raises(ValueError):
        _json(text)


def test_snapshot_identity_uses_consumed_resource_revisions() -> None:
    first = ResourceIdentity("https://fixture/data", "sha256:" + "a" * 64)
    second = ResourceIdentity("https://fixture/catalog", "sha256:" + "b" * 64)
    snapshot = release_snapshot([first, second])
    assert snapshot == release_snapshot([second, first, first])
    check_release(snapshot, snapshot)
    with pytest.raises(OSError, match="release changed"):
        check_release(snapshot, release_snapshot([first]))
    with pytest.raises(ValueError, match="actual source"):
        release_snapshot([])
    with pytest.raises(ValueError, match="two revisions"):
        release_snapshot([first, ResourceIdentity(first.uri, second.sha256)])
