"""Type checkers expose only the scientific operations a result supports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_collected_evidence_capabilities_are_checked_before_execution(
    tmp_path: Path,
) -> None:
    snippet = tmp_path / "capabilities.py"
    snippet.write_text("""
from crc_framework import DistributionFamily, SampleFit
from crc_sdk import (
    AnnualMaximum, DailyTotal, Dataset, Layer, Observations, Projections, Samples,
)

def accepted(samples: Samples, observations: Observations, dataset: Dataset) -> None:
    spec = SampleFit(family=DistributionFamily.GUMBEL_RIGHT)
    samples.read().fit(spec)
    observations.read().aggregate(AnnualMaximum())
    observations.read().aggregate(DailyTotal()).aggregate(AnnualMaximum())
    dataset.inspect().metadata

def rejected(
    samples: Samples, observations: Observations,
    layer: Layer, projections: Projections,
) -> None:
    spec = SampleFit(family=DistributionFamily.GUMBEL_RIGHT)
    observations.read().fit(spec)
    layer.read().fit(spec)
    projections.read().fit(spec)
    samples.read().aggregate(AnnualMaximum())
""")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--follow-imports=silent",
            str(snippet),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.count("[misc]") == 4, result.stdout
    assert "Found 4 errors in 1 file" in result.stdout
