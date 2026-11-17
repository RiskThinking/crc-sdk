"""OS-binary presence guards for ``tippecanoe``/``tile-join``.

``tippecanoe`` and `tile-join` are not pip-installable -- they're assumed
present on the runtime image (a documented OS-level prerequisite, not a
``crc-sdk`` extra). These guards give a friendly, actionable error instead of
a raw "file not found" if that assumption doesn't hold, mirroring the
``pip install crc-sdk[extra]`` convention used for genuinely optional
Python dependencies elsewhere in the SDK.
"""

from __future__ import annotations

import shutil

_INSTALL_HINT = (
    "macOS: `brew install tippecanoe`; Ubuntu: `apt install tippecanoe`; "
    "source: https://github.com/felt/tippecanoe#installation"
)


def require_tippecanoe() -> str:
    """Return the resolved ``tippecanoe`` executable path, or raise."""
    tippecanoe = shutil.which("tippecanoe")
    if not tippecanoe:
        raise RuntimeError(f"tippecanoe not found on PATH. {_INSTALL_HINT}")
    return tippecanoe


def require_tile_join() -> str:
    """Return the resolved ``tile-join`` executable path, or raise.

    ``tile-join`` ships alongside ``tippecanoe`` from the same project, so
    the install hint is identical -- installing one installs the other.
    """
    tile_join = shutil.which("tile-join")
    if not tile_join:
        raise RuntimeError(f"tile-join not found on PATH. {_INSTALL_HINT}")
    return tile_join
