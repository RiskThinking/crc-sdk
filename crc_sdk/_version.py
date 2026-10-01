"""Installed-package identity helpers."""

from __future__ import annotations

import platform
from importlib import metadata


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sdk_version() -> str:
    """The installed crc-sdk version, used as the default ``creation_version``."""
    return _distribution_version("crc-sdk") or "0+unknown"


def framework_version() -> str | None:
    """The installed crc-framework version recorded in fit provenance."""
    return _distribution_version("crc-framework")


def platform_tag() -> str:
    """Operating system and CPU architecture, e.g. ``Linux-aarch64``."""
    return f"{platform.system()}-{platform.machine()}"
