"""Errors for impact configuration and calibration checks.

Document parsing reports ``crc_framework.ImpactError`` with the contract
``code`` and ``path``. ``ImpactModelError`` means the SDK model rejected a
document the framework had already accepted.
"""

from __future__ import annotations

from crc_framework import ImpactError


class ImpactCalibrationError(Exception):
    """A dataset declaration does not match an impact calibration."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ImpactModelError(RuntimeError):
    """The SDK model does not accept the framework's canonical document."""


def impact_error(code: str, path: str, message: str) -> ImpactError:
    """Build an ``ImpactError`` carrying the ``code`` and ``path`` attributes."""
    error = ImpactError(f"{code} at {path or '/'}: {message}")
    error.code = code
    error.path = path
    return error
