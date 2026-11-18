"""Atomic publication and admission of owned collection directories."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

from ._collection_schema import CollectionMetadata
from ._dataset_schema import _digest
from ._errors import CapabilityError
from ._runtime import LocalPath, _local_path

MANIFEST = "manifest.json"


def read_manifest(path: Path) -> CollectionMetadata:
    manifest = path / MANIFEST
    if path.is_symlink() or manifest.is_symlink():
        raise ValueError("collection bundles and manifests must be owned local files")
    result = CollectionMetadata.from_json(manifest.read_text(encoding="utf-8"))
    if result.report.valid is not True or not result.report.complete:
        raise ValueError("the collection manifest must record a complete valid bundle")
    for resource in result.resources:
        if resource.path is None or resource.sha256 is None:
            raise ValueError(
                "each published collection resource requires a verified file"
            )
        _digest(resource.sha256)
        if (
            not resource.metadata.report.complete
            or resource.metadata.report.valid is not True
        ):
            raise ValueError(
                "published collection resources must be complete and valid"
            )
    resources = {resource.id: resource for resource in result.resources}
    for name in ("produced", "no_data"):
        actual = sum(
            getattr(resources[r].metadata.report.counts, name)
            for member in result.members
            for r in member.resource_ids
        )
        if actual != getattr(result.report.counts, name):
            raise ValueError(
                f"collection report {name} does not match member resources"
            )
    return result


def bundle_destination(path: LocalPath, overwrite: bool) -> tuple[Path, bool]:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    destination = _local_path(path)
    existing = os.path.lexists(destination)
    if existing:
        if not overwrite:
            raise FileExistsError(destination)
        if destination.is_symlink() or not destination.is_dir():
            raise NotADirectoryError(destination)
        if sys.platform not in {"linux", "darwin"}:
            raise CapabilityError(
                "Atomic replacement of an existing collection is unavailable "
                "on this platform. Choose a new destination directory."
            )
        try:
            manifest = read_manifest(destination)
        except (ValueError, TypeError, FileNotFoundError) as error:
            raise ValueError(
                "overwrite requires an existing complete collection bundle"
            ) from error
        _check_owned_tree(destination, manifest)
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    return destination, existing


def _check_owned_tree(path: Path, metadata: CollectionMetadata) -> None:
    """Protect files that the bundle does not own from replacement."""
    actual = {p.name for p in path.iterdir()}
    permitted = {MANIFEST, "resources"}
    if actual - permitted:
        raise ValueError("the destination contains files outside its collection bundle")
    resource_dir = path / "resources"
    if resource_dir.is_symlink():
        raise ValueError("the collection resource directory cannot be a symlink")
    expected = {Path(r.path).name for r in metadata.resources if r.path is not None}
    if resource_dir.exists():
        files = tuple(resource_dir.iterdir())
        if any(p.is_symlink() or not p.is_file() for p in files):
            raise ValueError("the collection resources must be ordinary local files")
        if {p.name for p in files} != expected:
            raise ValueError(
                "collection directory files do not match its resource inventory"
            )
    elif expected:
        raise ValueError("collection resource directory is missing")


def publish_bundle(staging: Path, destination: Path, *, existing: bool) -> None:
    """Commit a complete directory without replacing files at an unowned path."""
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
            result = rename(-100, before, -100, after, 2 if existing else 1)
        else:
            rename = libc.renamex_np
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(before, after, 2 if existing else 4)
        if result:
            code = ctypes.get_errno()
            if code in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
                raise CapabilityError(
                    "The destination filesystem does not support "
                    "atomic collection publication"
                )
            raise OSError(code, os.strerror(code), destination)
        return
    if os.name == "nt" and not existing:
        os.rename(staging, destination)
        return
    raise CapabilityError(
        "Atomic collection publication is unavailable on this platform"
    )
