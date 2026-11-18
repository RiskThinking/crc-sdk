"""Scope-owned source acquisition and content-verified URI cache indexes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from typing import TypeVar, cast

import pyarrow as pa  # type: ignore[import-untyped]

from ._dataset_schema import (
    ExecutionReport,
    ReleaseReference,
    ResourceIdentity,
    SourceIdentity,
    Stage,
    _resource,
)
from ._declarations import Period, PeriodKind, ScientificValue
from ._graph import ExecutionScope, Node, NodeStream
from ._inspection import Inspection, Requirement
from ._runtime import CacheMode
from ._selection import SelectionState

MetadataT = TypeVar("MetadataT")


class SourceNode(Node[MetadataT]):
    """An explicitly selectable source, never inferred from method names."""

    state: SelectionState

    @abstractmethod
    def selected(self, state: SelectionState) -> SourceNode[MetadataT]:
        raise NotImplementedError

    @abstractmethod
    def inspection(self, scope: ExecutionScope) -> Inspection:
        raise NotImplementedError

    def requirements(self) -> tuple[Requirement, ...]:
        return ()

    def checked(
        self, scope: ExecutionScope, stream: NodeStream[MetadataT]
    ) -> NodeStream[MetadataT]:
        """Install one global row check without replacing source execution facts."""
        from ._dataset_schema import DatasetMetadata
        from ._evidence import EvidenceMetadata
        from ._recipes import _Checked, _report_view

        if not isinstance(stream.metadata, (DatasetMetadata, EvidenceMetadata)):
            raise TypeError("Scientific sources require dataset or evidence metadata")
        state = None if stream.checked else _Checked(scope, stream.metadata)
        source = (
            None if stream.metadata.source is None else stream.metadata.source.product
        )

        def batches() -> Iterator[pa.RecordBatch]:
            with native_data(scope=scope, source=source):
                for batch in stream.batches:
                    yield batch if state is None else state.check(batch)

        def metadata() -> MetadataT:
            with native_data(scope=scope, source=source):
                return (
                    stream.metadata
                    if stream.final_metadata is None
                    else stream.final_metadata()
                )

        def report() -> ExecutionReport:
            with native_data(scope=scope, source=source):
                current = (
                    stream.metadata
                    if stream.final_metadata is None
                    else stream.final_metadata()
                )
                if not isinstance(current, (DatasetMetadata, EvidenceMetadata)):
                    raise TypeError(
                        "Scientific sources require dataset or evidence metadata"
                    )
                original = current.report if stream.report is None else stream.report()
                if original.valid is False:
                    original = _report_view(original, baseline=original)
                if state is None:
                    return original
                counts = replace(
                    original.counts, produced=state.produced, no_data=state.no_data
                )
                if stream.report is None:
                    counts = replace(
                        counts, requested=max(counts.requested, state.produced)
                    )
                return replace(original, counts=counts)

        def close() -> None:
            first_error: BaseException | None = None
            try:
                close_batches = getattr(stream.batches, "close", None)
                if close_batches is not None:
                    close_batches()
            except BaseException as error:
                first_error = error
            try:
                if stream.close is not None:
                    stream.close()
            except BaseException as error:
                first_error = first_error or error
            if first_error is not None:
                raise first_error

        return NodeStream(
            stream.schema,
            stream.metadata,
            batches(),
            report=report,
            final_metadata=metadata,
            close=close,
            checked=True,
            verify_before_yield=stream.verify_before_yield,
        )


@contextmanager
def native_data(
    *, scope: ExecutionScope | None = None, source: str | None = None
) -> Iterator[None]:
    """Translate malformed native science, retaining operational exception types."""
    from csv import Error as CSVError
    from xml.etree.ElementTree import ParseError
    from zipfile import BadZipFile

    from crc_framework import DiagnosticSeverity

    from ._dataset_schema import ExecutionDiagnostic
    from ._errors import CapabilityError
    from ._graph import DataValidationError

    try:
        yield
    except CapabilityError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ParseError,
        BadZipFile,
        CSVError,
    ) as error:
        if scope is not None and error is scope._callback_error:
            raise
        raise DataValidationError(
            (
                ExecutionDiagnostic(
                    "invalid_source_data",
                    DiagnosticSeverity.ERROR,
                    Stage.DECODING,
                    str(error),
                    source=source,
                ),
            )
        ) from error


def _manifest_path(
    scope: ExecutionScope, uri: str, digest: str | None = None
) -> Path | None:
    directory = scope.runtime.cache.directory
    if directory is None:
        return None
    key = hashlib.sha256(uri.encode()).hexdigest()
    if digest is not None:
        return (
            directory
            / "source-index-1.0"
            / key
            / (digest.removeprefix("sha256:") + ".json")
        )
    return directory / "source-index-1.0" / (key + ".json")


def _read_index(path: Path, uri: str) -> tuple[ResourceIdentity, int]:
    try:
        document = _json(path.read_text())
        if (
            not isinstance(document, dict)
            or set(document) != {"schema_version", "resource", "status"}
            or document["schema_version"] != "1.0"
        ):
            raise ValueError("unsupported source cache index")
        identity = _resource(document["resource"])
        if identity.uri != uri:
            raise ValueError("source cache URI differs from its key")
        if document["status"] not in (200, 404) or isinstance(document["status"], bool):
            raise ValueError("unsupported source response status")
        return identity, int(document["status"])
    except (ValueError, TypeError, UnicodeError) as error:
        raise OSError(f"Invalid source cache index: {path}") from error


def _write_index(path: Path, resource: ResourceIdentity, status: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".source-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "schema_version": "1.0",
                    "resource": resource._document(),
                    "status": status,
                },
                output,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _download(uri: str, path: Path) -> int:
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("file source URI must refer to this machine")
        shutil.copyfile(urllib.request.url2pathname(parsed.path), path)
        return 200
    if not parsed.scheme:
        shutil.copyfile(uri, path)
        return 200
    if parsed.scheme == "s3":
        uri = f"https://{parsed.netloc}.s3.amazonaws.com{parsed.path}"
    elif parsed.scheme in {"gs", "gcs"}:
        uri = f"https://storage.googleapis.com/{parsed.netloc}{parsed.path}"
    elif parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported public source URI scheme: {parsed.scheme}")
    request = urllib.request.Request(uri, headers={"User-Agent": "crc-sdk/1.0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        return 200
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        with error, path.open("wb") as output:
            shutil.copyfileobj(error, output, length=1024 * 1024)
        return 404


class SourceNotFound(urllib.error.HTTPError):
    def __init__(self, resource: ResourceIdentity) -> None:
        self.resource = resource
        super().__init__(
            resource.uri, 404, "source resource is unavailable", Message(), None
        )


def _fetch(
    scope: ExecutionScope,
    uri: str,
    *,
    expected_sha256: str | None = None,
    license: str | None = None,
    attribution: str | None = None,
    acquire_callback: Callable[[Path], None] | None = None,
) -> tuple[Path, ResourceIdentity]:
    """Return verified owned bytes; offline never invokes acquisition."""
    if not isinstance(uri, str) or not uri or uri.strip() != uri:
        raise ValueError("source URI must be a nonempty exact string")
    key = (uri, expected_sha256)
    if key in scope._source_memo:
        path, identity, status = scope._source_memo[key]
        identity = replace(
            identity,
            license=license if license is not None else identity.license,
            attribution=attribution
            if attribution is not None
            else identity.attribution,
        )
        if status == 404:
            raise SourceNotFound(identity)
        return path, identity
    mode = scope.runtime.cache.mode
    index = _manifest_path(scope, uri)
    candidate_index = (
        _manifest_path(scope, uri, expected_sha256)
        if expected_sha256 is not None
        else index
    )
    cached: ResourceIdentity | None = None
    status = 200
    if (
        candidate_index is not None
        and candidate_index.exists()
        and mode is not CacheMode.REFRESH
    ):
        cached, status = _read_index(candidate_index, uri)
        if expected_sha256 is not None and cached.sha256 != expected_sha256:
            raise OSError("source cache revision differs from its checksum key")
    digest = expected_sha256 or (None if cached is None else cached.sha256)
    retrieved: datetime | None = None
    if digest is not None:
        identity = ResourceIdentity(
            uri,
            digest,
            license=license
            if license is not None
            else None
            if cached is None
            else cached.license,
            attribution=attribution
            if attribution is not None
            else None
            if cached is None
            else cached.attribution,
            retrieved_at=None if cached is None else cached.retrieved_at,
        )

        def acquire(path: Path) -> None:
            nonlocal retrieved, status
            status = (
                (_download(uri, path) or 200)
                if acquire_callback is None
                else (acquire_callback(path) or 200)
            )
            retrieved = datetime.now(UTC)

        path = scope.pin(identity, acquire)
        if retrieved is not None:
            identity = replace(identity, retrieved_at=retrieved)
    else:
        if mode is CacheMode.OFFLINE:
            raise FileNotFoundError(
                f"Source is unavailable in the verified offline cache: {uri}"
            )
        fd, name = tempfile.mkstemp(prefix="source-download-", dir=scope.directory)
        os.close(fd)
        temporary = Path(name)
        try:
            scope.emit(Stage.ACQUISITION, 0, 1)
            status = (
                (_download(uri, temporary) or 200)
                if acquire_callback is None
                else (acquire_callback(temporary) or 200)
            )
            retrieved = datetime.now(UTC)
            with temporary.open("rb") as reader:
                path, found = scope.pin_handle(reader, uri=uri)
            identity = ResourceIdentity(
                uri,
                found.sha256,
                license=license,
                attribution=attribution,
                retrieved_at=retrieved,
            )
            scope.emit(Stage.ACQUISITION, 1, 1)
        finally:
            temporary.unlink(missing_ok=True)
    if index is not None and mode is not CacheMode.OFFLINE:
        revision = _manifest_path(scope, uri, identity.sha256)
        assert revision is not None
        _write_index(revision, identity, status)
        if expected_sha256 is None or retrieved is not None:
            _write_index(index, identity, status)
    scope._source_memo[key] = (path, identity, status)
    if status == 404:
        raise SourceNotFound(identity)
    return path, identity


def fetch(
    scope: ExecutionScope,
    uri: str,
    *,
    expected_sha256: str | None = None,
    license: str | None = None,
    attribution: str | None = None,
) -> tuple[Path, ResourceIdentity]:
    return _fetch(
        scope,
        uri,
        expected_sha256=expected_sha256,
        license=license,
        attribution=attribution,
    )


def fetch_custom(
    scope: ExecutionScope,
    uri: str,
    acquire: Callable[[Path], None],
    *,
    expected_sha256: str | None = None,
    license: str | None = None,
    attribution: str | None = None,
) -> tuple[Path, ResourceIdentity]:
    """Cache a verified source view whose content is produced by a scoped decoder."""
    if not callable(acquire):
        raise TypeError("acquire must be callable")
    return _fetch(
        scope,
        uri,
        expected_sha256=expected_sha256,
        license=license,
        attribution=attribution,
        acquire_callback=acquire,
    )


def fetch_optional(
    scope: ExecutionScope,
    uri: str,
    *,
    expected_sha256: str | None = None,
    license: str | None = None,
    attribution: str | None = None,
) -> tuple[Path | None, ResourceIdentity]:
    """Retain a verified missing-object response where absence is native data."""
    try:
        return fetch(
            scope,
            uri,
            expected_sha256=expected_sha256,
            license=license,
            attribution=attribution,
        )
    except SourceNotFound as error:
        return None, error.resource


def _json(text: str) -> ScientificValue:
    def pairs(items: list[tuple[str, ScientificValue]]) -> dict[str, ScientificValue]:
        result: dict[str, ScientificValue] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def number(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("source JSON contains a non-finite number")
        return value

    return cast(
        ScientificValue,
        json.loads(
            text, object_pairs_hook=pairs, parse_float=number, parse_constant=number
        ),
    )


def fetch_json(
    scope: ExecutionScope,
    uri: str,
    *,
    expected_sha256: str | None = None,
    license: str | None = None,
    attribution: str | None = None,
) -> tuple[ScientificValue, ResourceIdentity]:
    path, identity = fetch(
        scope,
        uri,
        expected_sha256=expected_sha256,
        license=license,
        attribution=attribution,
    )
    return _json(path.read_text(encoding="utf-8-sig")), identity


def release_snapshot(
    resources: Sequence[ResourceIdentity], *, id: str | None = None
) -> ReleaseReference:
    if not resources:
        raise ValueError("a release snapshot requires actual source resources")
    identities: dict[str, str] = {}
    for resource in resources:
        if not isinstance(resource, ResourceIdentity):
            raise TypeError("release resources must be ResourceIdentity records")
        if resource.uri in identities and identities[resource.uri] != resource.sha256:
            raise ValueError(
                "one release cannot contain two revisions of the same source URI"
            )
        identities[resource.uri] = resource.sha256
    data = json.dumps(
        sorted(identities.items()), ensure_ascii=False, separators=(",", ":")
    ).encode()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    return ReleaseReference(digest if id is None else id, digest)


def check_release(actual: ReleaseReference, requested: ReleaseReference | None) -> None:
    if requested is not None and (
        actual.id != requested.id or actual.sha256 != requested.sha256
    ):
        raise OSError(
            f"source release changed: expected {requested.id} ({requested.sha256}), "
            f"received {actual.id} ({actual.sha256})"
        )


def payload_selection(
    state: SelectionState, release: ReleaseReference | None
) -> tuple[SelectionState, ReleaseReference | None]:
    """Resolve science selectors independently of the pending payload check."""
    references = state.series
    if references is None:
        return state, release
    for reference in references:
        pin = reference.release
        if pin is not None:
            check_release(pin, release)
            release = pin
    return replace(
        state,
        series=tuple(replace(reference, release=None) for reference in references),
    ), release


def selected_source_time(state: SelectionState) -> Period | None:
    """Use an exact series window when a separate time selector is absent."""
    if state.time is not None:
        return state.time
    if state.series is not None and len(state.series) == 1:
        period = state.series[0].period
        if period is not None and period.kind is PeriodKind.WINDOW:
            return period
    return None


def verify_payload(
    source: SourceIdentity, *, id: str, expected: ReleaseReference | None
) -> SourceIdentity:
    release = release_snapshot(source.resources, id=id)
    check_release(release, expected)
    return replace(source, release=release)
