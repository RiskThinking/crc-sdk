"""Atomic publication and admission of owned collection directories."""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from ._collection_schema import CollectionMetadata
from ._dataset_schema import _combine_reports, _digest
from ._errors import CapabilityError
from ._runtime import LocalPath, _local_path

MANIFEST = "manifest.json"
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ExistingBundle:
    identity: tuple[int, int]
    metadata: CollectionMetadata
    files: tuple[tuple[str, tuple[int, int, int, int, int]], ...]


def _directory_identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino) if stat.S_ISDIR(info.st_mode) else None


def read_manifest(path: Path) -> CollectionMetadata:
    manifest = path / MANIFEST
    if path.is_symlink() or manifest.is_symlink():
        raise ValueError("collection bundles and manifests must be owned local files")
    if not stat.S_ISREG(manifest.stat().st_mode):
        raise ValueError("the collection manifest must be an ordinary local file")
    result = CollectionMetadata.from_json(manifest.read_text(encoding="utf-8"))
    if result.report.valid is not True or not result.report.complete:
        raise ValueError("the collection manifest must record a complete valid bundle")
    for resource in result.resources:
        if resource.path is None or resource.sha256 is None:
            raise ValueError(
                "each published collection resource requires a verified file"
            )
        candidate = path / resource.path
        if candidate.is_symlink() or candidate.parent.is_symlink():
            raise ValueError("collection resources must be owned local files")
        if not stat.S_ISREG(candidate.stat().st_mode):
            raise ValueError("collection resources must be ordinary local files")
        _digest(resource.sha256)
        if (
            not resource.metadata.report.complete
            or resource.metadata.report.valid is not True
        ):
            raise ValueError(
                "published collection resources must be complete and valid"
            )
    resources = {resource.id: resource for resource in result.resources}
    combined = _combine_reports(
        (
            resources[r].metadata.report
            for member in result.members
            for r in member.resource_ids
        ),
        complete=True,
    )
    for name in ("counts", "diagnostics", "source_identities"):
        if getattr(combined, name) != getattr(result.report, name):
            raise ValueError(f"collection report {name} do not match member resources")
    return result


def bundle_destination(
    path: LocalPath, overwrite: bool
) -> tuple[Path, _ExistingBundle | None]:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    destination = _local_path(path)
    existing = None
    if os.path.lexists(destination):
        if not overwrite:
            raise FileExistsError(destination)
        if sys.platform not in {"linux", "darwin"}:
            raise CapabilityError(
                "Atomic replacement of an existing collection is unavailable "
                "on this platform. Choose a new destination directory."
            )
        existing = _existing_bundle(destination)
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    return destination, existing


def _existing_bundle(path: Path) -> _ExistingBundle:
    identity = _directory_identity(path)
    if identity is None:
        raise NotADirectoryError(path)
    try:
        manifest = read_manifest(path)
    except (ValueError, TypeError, FileNotFoundError) as error:
        raise ValueError(
            "overwrite requires an existing complete collection bundle"
        ) from error
    files = _owned_files(path, manifest)
    if _directory_identity(path) != identity:
        raise OSError(f"collection destination changed during ownership check: {path}")
    return _ExistingBundle(identity, manifest, files)


@contextmanager
def stage_bundle(destination: Path) -> Iterator[Path]:
    # Atomic exchange can move an unexpected directory to the staging path.
    # Keep that path inside a private workspace and only clean up our scratch.
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    staged = workspace / "bundle"
    identity = None
    try:
        staged.mkdir()
        identity = _directory_identity(staged)
        yield staged
    finally:
        current = _directory_identity(staged)
        if current is not None and current == identity:
            shutil.rmtree(staged)
        try:
            workspace.rmdir()
        except OSError as error:
            if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise


def _owned_files(
    path: Path, metadata: CollectionMetadata
) -> tuple[tuple[str, tuple[int, int, int, int, int]], ...]:
    """Protect files that the bundle does not own from replacement."""
    actual = {p.name for p in path.iterdir()}
    permitted = {MANIFEST, "resources"}
    if actual - permitted:
        raise ValueError("the destination contains files outside its collection bundle")
    resource_dir = path / "resources"
    if resource_dir.is_symlink():
        raise ValueError("the collection resource directory cannot be a symlink")
    expected = {Path(r.path).name for r in metadata.resources if r.path is not None}
    files: tuple[Path, ...] = ()
    if resource_dir.exists():
        files = tuple(sorted(resource_dir.iterdir()))
        if {p.name for p in files} != expected:
            raise ValueError(
                "collection directory files do not match its resource inventory"
            )
    elif expected:
        raise ValueError("collection resource directory is missing")

    identities = []
    for file in (path / MANIFEST, *files):
        info = file.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("the collection files must be ordinary local files")
        identities.append((str(file.relative_to(path)), _file_stamp(info)))
    return tuple(identities)


def _file_stamp(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _retire_bundle(path: Path, bundle: _ExistingBundle) -> None:
    """Remove verified files and leave any other contents in the displaced tree."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as cleanup:
        directory = os.open(path, flags)
        cleanup.callback(os.close, directory)
        info = os.fstat(directory)
        if (info.st_dev, info.st_ino) != bundle.identity:
            raise OSError("displaced collection identity changed before cleanup")
        resources = None
        try:
            resources = os.open("resources", flags, dir_fd=directory)
        except FileNotFoundError:
            pass
        else:
            cleanup.callback(os.close, resources)
        for relative, stamp in bundle.files:
            location = Path(relative)
            parent = directory if location.parent == Path(".") else resources
            if parent is None:
                continue
            try:
                info = os.stat(location.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode) and _file_stamp(info) == stamp:
                os.unlink(location.name, dir_fd=parent)
        if resources is not None:
            os.rmdir("resources", dir_fd=directory)
        path.rmdir()


def publish_bundle(
    staging: Path, destination: Path, *, existing: _ExistingBundle | None
) -> None:
    """Publish a complete directory and retire only the verified old bundle."""
    if existing is None:
        _rename_bundle(staging, destination, exchange=False)
        return

    # Cooperating writers serialize the final check and exchange. Other
    # filesystem operations can still race, so verify the displaced inode.
    import fcntl

    parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(parent, fcntl.LOCK_EX)
        if _existing_bundle(destination) != existing:
            raise OSError(
                f"collection destination changed before publication: {destination}"
            )
        new_identity = _directory_identity(staging)
        _rename_bundle(staging, destination, exchange=True)
        try:
            if _existing_bundle(staging) != existing:
                raise ValueError(
                    "displaced collection differs from the verified bundle"
                )
        except (OSError, ValueError) as error:
            if _directory_identity(destination) == new_identity:
                try:
                    _rename_bundle(staging, destination, exchange=True)
                except OSError as rollback_error:
                    raise OSError(
                        "collection rollback failed. "
                        f"Displaced directory retained at {staging}"
                    ) from rollback_error
                if _directory_identity(staging) == new_identity:
                    raise OSError(
                        "collection destination changed during publication: "
                        f"{destination}"
                    ) from error
            raise OSError(
                "collection destination changed. "
                f"Displaced directory retained at {staging}"
            ) from error
        try:
            _retire_bundle(staging, existing)
        except OSError as error:
            _LOG.warning(
                "Collection published at %s. Displaced data retained at %s: %s",
                destination,
                staging,
                error,
            )
    finally:
        os.close(parent)


def _rename_bundle(staging: Path, destination: Path, *, exchange: bool) -> None:
    if sys.platform in {"linux", "darwin"}:
        libc = ctypes.CDLL(None, use_errno=True)
        before, after = os.fsencode(staging), os.fsencode(destination)
        if sys.platform == "linux":
            rename = getattr(libc, "renameat2", None)
            if rename is None:
                raise CapabilityError(
                    "Atomic collection publication requires renameat2"
                )
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            result = rename(-100, before, -100, after, 2 if exchange else 1)
        else:
            rename = libc.renamex_np
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(before, after, 2 if exchange else 4)
        if result:
            code = ctypes.get_errno()
            if code in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
                raise CapabilityError(
                    "The destination filesystem does not support "
                    "atomic collection publication"
                )
            raise OSError(code, os.strerror(code), destination)
        return
    if os.name == "nt" and not exchange:
        os.rename(staging, destination)
        return
    raise CapabilityError(
        "Atomic collection publication is unavailable on this platform"
    )
