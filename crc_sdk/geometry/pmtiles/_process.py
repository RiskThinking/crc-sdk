"""Subprocess management for a single tippecanoe tiling pass.

Ports the idle-timeout watchdog and background stderr draining from
gen_pmtiles_v2's ``process.py::StreamingConsumer``, simplified to "one
process, one combined query, one write loop, then wait" -- the SQL-based
bridge in ``_geojson_sql.py`` already combines every layer into one query, so
there's exactly one producer here, not the many short-lived gpio feeds the
original had to coordinate.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

_DEFAULT_IDLE_TIMEOUT_SECONDS = 5_400.0
_DEFAULT_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class SubprocessPipeOptions:
    """Idle-timeout watchdog tuning for a supervised subprocess."""

    idle_timeout_seconds: float = _DEFAULT_IDLE_TIMEOUT_SECONDS
    poll_seconds: float = _DEFAULT_POLL_SECONDS

    def __post_init__(self) -> None:
        if self.idle_timeout_seconds <= 0 or self.poll_seconds <= 0:
            raise ValueError(
                "idle_timeout_seconds and poll_seconds must both be positive"
            )


class _Activity:
    """Thread-safe last-touched timestamp, used to detect a stalled process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = time.monotonic()

    def touch(self) -> None:
        with self._lock:
            self._last = time.monotonic()

    def quiet_for(self) -> float:
        with self._lock:
            return time.monotonic() - self._last


def _drain_stderr(
    name: str, stream: IO[bytes] | None, activity: _Activity
) -> threading.Thread:
    def drain() -> None:
        if stream is None:
            return
        while True:
            chunk = stream.readline()
            if not chunk:
                break
            activity.touch()
            text = chunk.decode(errors="replace").rstrip()
            if text:
                logger.info("%s: %s", name, text)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread


def _wait_with_idle_timeout(
    process: subprocess.Popen[bytes],
    activity: _Activity,
    options: SubprocessPipeOptions,
    label: str,
) -> None:
    while True:
        remaining = options.idle_timeout_seconds - activity.quiet_for()
        if remaining <= 0:
            process.kill()
            raise RuntimeError(
                f"{label} produced no stderr output for "
                f"{options.idle_timeout_seconds:.0f}s -- treating as stalled"
            )
        try:
            process.wait(timeout=min(options.poll_seconds, remaining))
            return
        except subprocess.TimeoutExpired:
            continue


def run_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    """Run a short subprocess to completion, raising on a nonzero exit."""
    env = os.environ.copy()
    if environment:
        env.update(environment)
    logger.info("run: %s", " ".join(command))
    subprocess.run(list(command), check=True, env=env, cwd=str(cwd) if cwd else None)


class TippecanoeProcess:
    """One tippecanoe process, fed by a single Arrow-batched write loop.

    The caller (``_build.py``) pulls Arrow batches from one combined query
    and calls :meth:`write` once per batch; there is exactly one producer, so
    this only needs to manage the process's lifecycle plus the idle-timeout
    watchdog and stderr draining -- a "spawn, write, close, wait" shape
    rather than a stateful multi-feed consumer.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        options: SubprocessPipeOptions = SubprocessPipeOptions(),
    ) -> None:
        env = os.environ.copy()
        if environment:
            env.update(environment)
        self._options = options
        self._activity = _Activity()
        self._process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if self._process.stdin is None:
            raise RuntimeError("cannot open tippecanoe stdin")
        self._stderr_thread = _drain_stderr(
            "tippecanoe", self._process.stderr, self._activity
        )
        self._closed = False

    def write(self, data: bytes) -> None:
        """Write one batch's worth of already-serialized GeoJSONSeq bytes."""
        if self._closed:
            raise RuntimeError("tippecanoe process is already closed")
        if self._process.poll() is not None:
            raise RuntimeError("tippecanoe exited before all input was written")
        assert self._process.stdin is not None
        self._process.stdin.write(data)
        self._activity.touch()

    def finish(self) -> None:
        """Close stdin and wait for tippecanoe to exit, raising on failure."""
        if self._closed:
            return
        self._closed = True
        assert self._process.stdin is not None
        self._process.stdin.close()
        try:
            _wait_with_idle_timeout(
                self._process, self._activity, self._options, "tippecanoe"
            )
            code = self._process.wait()
        except Exception:
            self._process.kill()
            raise
        finally:
            self._stderr_thread.join(timeout=60)
        if code != 0:
            raise RuntimeError(f"tippecanoe failed with exit code {code}")

    def abort(self) -> None:
        """Kill the process immediately, e.g. after an upstream failure."""
        if not self._closed:
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
        self._process.kill()
        self._stderr_thread.join(timeout=30)

    def __enter__(self) -> TippecanoeProcess:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()
