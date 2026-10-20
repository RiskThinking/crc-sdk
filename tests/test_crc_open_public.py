"""Opt-in integration checks against the pinned public CRC Open release."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.providers.crc_open import (
    DEFAULT_RELEASE,
    PUBLIC_SOURCE,
    CRCOpenFixtureWarning,
    CRCOpenHazards,
)
from crc_sdk.workflows import HazardDataset

pytestmark = pytest.mark.skipif(
    os.environ.get("CRC_OPEN_LIVE") != "1",
    reason="set CRC_OPEN_LIVE=1 to test the public release",
)


def test_public_release_discovery(tmp_path: Path) -> None:
    plan = HazardDataset.crc_open().cache(tmp_path / "cache")
    catalog = plan.catalog()
    assert catalog.release == DEFAULT_RELEASE
    assert not catalog.fixture
    assert len(catalog.hazards) == 63
    assert catalog.pathways == ("ssp585",)
    assert catalog.licence == "CC-BY-NC-4.0"
    with warnings.catch_warnings(record=True) as recorded:
        resolved = plan.resolve()
    assert not any(issubclass(w.category, CRCOpenFixtureWarning) for w in recorded)
    assert len(resolved.partitions) == sum(
        len(h.partitions) for h in catalog.hazards.values()
    )
    assert resolved.total_bytes > 0 and resolved.total_rows > 0
    assert len(set(resolved.urls)) == len(resolved.partitions)
    assert all(
        url.startswith(f"{PUBLIC_SOURCE}/{DEFAULT_RELEASE}/") for url in resolved.urls
    )
    assert not list((tmp_path / "cache").rglob("*.parquet"))
    single = plan.hazards(["rx1day"]).resolve()
    assert len(single.partitions) == len(catalog.hazards["rx1day"].partitions)


def test_public_subset_materialization_and_offline_reuse(tmp_path: Path) -> None:
    import h3  # type: ignore[import-untyped]

    plan = HazardDataset.crc_open().hazards(["rx1day"]).cache(tmp_path / "cache")
    resolved = plan.resolve()
    smallest = min(resolved.partitions, key=lambda p: p.size_bytes)
    sample = CRCOpenHazards(PUBLIC_SOURCE, DEFAULT_RELEASE).download_partition(
        smallest, tmp_path / "sample.parquet"
    )
    cell = pq.ParquetFile(sample).read(columns=["cell_index"])["cell_index"][0].as_py()
    lat, lon = h3.cell_to_latlng(h3.int_to_str(cell))
    horizon = min(smallest.horizons)
    selected = plan.for_area(
        (lon - 0.00001, lat - 0.00001, lon + 0.00001, lat + 0.00001)
    ).horizons([horizon])
    result = selected.materialize(tmp_path / "rx1day.parquet")
    table = pq.ParquetFile(result.provider.source).read()
    assert table.num_rows > 0
    assert set(table["hazard_name"].to_pylist()) == {"rx1day"}
    assert set(table["pathway"].to_pylist()) == {"ssp585"}
    assert set(table["horizon"].to_pylist()) == {horizon}
    assert result.metadata().value_unit == "mm/day"
    assert result.provenance().checksum == resolved.catalog_sha256
    assert result.provenance().licence == "CC-BY-NC-4.0"
    with patch.object(CRCOpenHazards, "fetch", side_effect=AssertionError("network")):
        with patch.object(
            CRCOpenHazards, "download_partition", side_effect=AssertionError("network")
        ):
            offline = selected.cache(tmp_path / "cache", mode="offline")
            again = offline.materialize(tmp_path / "rx1day-offline.parquet")
    assert pq.ParquetFile(again.provider.source).metadata.num_rows == table.num_rows
