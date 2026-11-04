from __future__ import annotations

import sys

import pytest

from crc_sdk.geometry.pmtiles._process import TippecanoeProcess

EXIT_EARLY = [sys.executable, "-c", "import sys; sys.stdin.read(1); sys.exit(3)"]


def test_write_reports_the_exit_code_not_a_broken_pipe() -> None:
    process = TippecanoeProcess(EXIT_EARLY)
    with pytest.raises(RuntimeError, match="code 3"):
        for _ in range(1000):  # far beyond the pipe buffer
            process.write(b"x" * 65536)
    process.abort()


def test_finish_raises_the_exit_code() -> None:
    with pytest.raises(RuntimeError, match="exit code 3"):
        with TippecanoeProcess(EXIT_EARLY) as process:
            process.write(b"x")


def test_abort_after_a_broken_pipe_still_reaps_the_process() -> None:
    process = TippecanoeProcess(EXIT_EARLY)
    with pytest.raises(RuntimeError):
        for _ in range(1000):
            process.write(b"x" * 65536)
    process._closed = False  # a pending close whose flush hits the pipe
    process.abort()
    assert process._process.returncode is not None
    assert not process._stderr_thread.is_alive()
