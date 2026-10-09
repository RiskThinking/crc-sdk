from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.connectors.parquet import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.geometry import point_to_cell
from crc_sdk.providers.crc_open import (
    CRCCatalog,
    CRCOpenFixtureWarning,
    CRCOpenHazards,
    CRCPartition,
)
from crc_sdk.types import EnsembleDescriptor, HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import HazardDataset, HorizonExtrapolationWarning

RELEASE = "test-v1"


def _publish_catalog(root: Path, catalog: dict[str, Any]) -> None:
    payload = json.dumps(catalog, sort_keys=True).encode()
    (root / "_CATALOG.json").write_bytes(payload)
    (root / "_SUCCESS").write_text(hashlib.sha256(payload).hexdigest() + "\n")


def _release(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    import h3  # type: ignore[import-untyped]

    source = tmp_path / "source"
    root = source / RELEASE
    root.mkdir(parents=True)
    catalog: dict[str, Any] = dict(
        catalog_version=1,
        release=RELEASE,
        schema_version="1.3",
        fixture=True,
        pathways=["ssp585"],
        licence="CC-BY-4.0",
        attribution="test source",
        retrieved_at="2026-10-08T00:00:00Z",
        hazards={},
    )
    for name, resolution, unit in (("rx1day", 5, "mm/day"), ("heat", 7, "degC")):
        metadata = HazardDatasetMetadata(
            h3_resolution=resolution,
            value_unit=unit,
            value_semantics=name,
            producer="test",
            creation_version="test",
            source=SourceProvenance(provider="fixture", dataset=name),
            probability_semantics="annual_value_distribution",
            ensemble=EnsembleDescriptor(pooling="pooled", scenario="ssp585"),
        )
        entries = []
        for lat, lon in ((43.65, -79.38), (-1.29, 36.82)):
            cell = point_to_cell(lon, lat, resolution)
            parent = h3.cell_to_parent(h3.int_to_str(cell), 0)
            path = f"{name}/h3_r0={parent}/part-00000.parquet"
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                dict(
                    cell_index=cell,
                    source_id="test",
                    source_geometry=None,
                    hazard_name=name,
                    horizon=horizon,
                    pathway="ssp585",
                    curve_kind="fitted",
                    curve_type="gumbel_r",
                    curve_shape=None,
                    curve_location=20.0,
                    curve_scale=5.0,
                    curve_atom_probability=None,
                    curve_atom_location=None,
                    curve_probabilities=None,
                    curve_values=None,
                )
                for horizon in (2050, 2090)
            ]
            table = pa.Table.from_pylist(rows, schema=hazard_arrow_schema(metadata))
            write_hazard_dataset(table, destination, metadata)
            entries.append(
                dict(
                    path=path,
                    h3_r0=parent,
                    sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    size_bytes=destination.stat().st_size,
                    rows=2,
                    cells=[cell],
                    horizons=[2050, 2090],
                )
            )
        catalog["hazards"][name] = dict(
            unit=unit,
            value_semantics=name,
            tail="upper",
            h3_resolution=resolution,
            probability_semantics="annual_value_distribution",
            pooling="pooled",
            pathways=["ssp585"],
            horizons=[2050, 2090],
            partitions=entries,
        )
    _publish_catalog(root, catalog)
    return source, catalog


def test_lazy_construction_and_explicit_source(tmp_path: Path) -> None:
    with patch.object(CRCOpenHazards, "fetch", side_effect=AssertionError("I/O")):
        plan = HazardDataset.crc_open().hazards(["rx1day"]).for_area([-80, 43, -79, 44])
        details = plan.explain(format="json")
        assert isinstance(details, dict)
        assert details["fixture_fallback"] is True
        assert HazardDataset.crc_open(source=tmp_path).fixture_fallback is False
    with pytest.raises(ValueError, match="choose source"):
        HazardDataset.crc_open(source=tmp_path, fixtures=tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        HazardDataset.crc_open(release="../private")


def test_pruned_materialize_cache_and_offline(tmp_path: Path) -> None:
    source, catalog = _release(tmp_path)
    plan = (
        HazardDataset.crc_open(release=RELEASE, fixtures=source)
        .hazards(["rx1day"])
        .for_area([-79.4, 43.6, -79.3, 43.7])
        .horizons([2050])
        .cache(tmp_path / "cache")
    )
    with pytest.warns(CRCOpenFixtureWarning):
        result = plan.prefetch()
    assert result.resources == result.cache_misses == 1
    with pytest.warns(CRCOpenFixtureWarning):
        dataset = plan.materialize(tmp_path / "result.parquet")
    table = pq.read_table(dataset.provider.source)
    assert table.num_rows == 1
    assert table["curve_location"].to_pylist() == [20.0]
    assert (
        dataset.provenance().checksum
        == hashlib.sha256((source / RELEASE / "_CATALOG.json").read_bytes()).hexdigest()
    )
    assert dataset.provenance().licence == catalog["licence"]
    with patch.object(CRCOpenHazards, "fetch", side_effect=AssertionError("network")):
        with pytest.warns(CRCOpenFixtureWarning):
            offline = plan.cache(tmp_path / "cache", mode="offline").prefetch()
    assert offline.cache_hits == 1
    cached_file = next((tmp_path / "cache").rglob("*.parquet"))
    cached_file.write_bytes(b"corrupt")
    with pytest.warns(CRCOpenFixtureWarning):
        with pytest.raises(FileNotFoundError, match="corrupt"):
            plan.cache(tmp_path / "cache", mode="offline").prefetch()
    with pytest.warns(CRCOpenFixtureWarning):
        assert plan.prefetch().cache_misses == 1


def test_scope_guards_and_sparse_coverage(tmp_path: Path) -> None:
    source, _ = _release(tmp_path)
    base = HazardDataset.crc_open(release=RELEASE, source=source).hazards(["rx1day"])
    with pytest.warns(CRCOpenFixtureWarning):
        with pytest.raises(ValueError, match="not in the open subset"):
            HazardDataset.crc_open(
                release=RELEASE, source=source, pathway="historic"
            ).materialize_all(tmp_path / "bad")
    with pytest.warns(CRCOpenFixtureWarning):
        with pytest.raises(LookupError, match="no coverage"):
            base.for_area([130, 30, 131, 31]).materialize(tmp_path / "empty.parquet")
    with pytest.warns((CRCOpenFixtureWarning, HorizonExtrapolationWarning)):
        with pytest.raises(ValueError, match="horizons"):
            base.horizons([2100]).materialize(tmp_path / "future.parquet")
    with pytest.warns(CRCOpenFixtureWarning):
        with pytest.raises(ValueError, match="hazards"):
            base.hazards(["private"]).materialize(tmp_path / "private.parquet")


def test_mixed_hazards_keep_metadata_separate(tmp_path: Path) -> None:
    source, _ = _release(tmp_path)
    plan = HazardDataset.crc_open(release=RELEASE, fixtures=source)
    with pytest.raises(ValueError, match="materialize_all"):
        plan.materialize(tmp_path / "mixed.parquet")
    with pytest.warns(CRCOpenFixtureWarning):
        datasets = plan.materialize_all(tmp_path / "outputs")
    assert datasets["rx1day"].metadata().h3_resolution == 5
    assert datasets["heat"].metadata().h3_resolution == 7
    assert datasets["heat"].metadata().value_unit == "degC"


def test_catalogue_and_partition_integrity(tmp_path: Path) -> None:
    source, catalog = _release(tmp_path)
    provider = CRCOpenHazards(source, RELEASE)
    (source / RELEASE / "_SUCCESS").write_text("0" * 64)
    with pytest.raises(ValueError, match="_SUCCESS"):
        provider.catalog_bytes()
    _publish_catalog(source / RELEASE, catalog)
    partition = CRCPartition.model_validate(
        catalog["hazards"]["rx1day"]["partitions"][0]
    )
    (source / RELEASE / partition.path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum/size"):
        provider.partition_bytes(partition)
    with pytest.raises(ValueError, match="safe relative"):
        CRCPartition.model_validate(
            dict(partition.model_dump(), path="../secret.parquet")
        )
    catalog["hazards"]["rx1day"]["private_eligibility_override"] = 0.01
    with pytest.raises(ValueError, match="Extra inputs"):
        CRCCatalog.model_validate(catalog)


def test_refresh_and_immutable_release(tmp_path: Path) -> None:
    source, catalog = _release(tmp_path)
    plan = (
        HazardDataset.crc_open(release=RELEASE, fixtures=source)
        .hazards(["rx1day"])
        .cache(tmp_path / "cache")
    )
    with pytest.warns(CRCOpenFixtureWarning):
        assert plan.prefetch().cache_misses == 2
    with pytest.warns(CRCOpenFixtureWarning):
        assert (
            plan.cache(tmp_path / "cache", mode="refresh").prefetch().cache_misses == 2
        )
    catalog["attribution"] = "changed"
    _publish_catalog(source / RELEASE, catalog)
    with pytest.raises(ValueError, match="pinned CRC release changed"):
        plan.cache(tmp_path / "cache", mode="refresh").prefetch()


def test_http_uses_catalogue_relative_paths(tmp_path: Path) -> None:
    source, catalog = _release(tmp_path)
    root = source / RELEASE
    urls = []

    def get(url: str, timeout: int) -> io.BytesIO:
        urls.append(url)
        assert timeout == 60
        return io.BytesIO((root / url.split(f"/{RELEASE}/", 1)[1]).read_bytes())

    with patch("crc_sdk.providers.crc_open.urlopen", side_effect=get):
        provider = CRCOpenHazards("https://fixtures.example/crc_open", RELEASE)
        provider.catalog_bytes()
        partition = CRCPartition.model_validate(
            catalog["hazards"]["rx1day"]["partitions"][0]
        )
        provider.partition_bytes(partition)
    assert urls == [
        f"https://fixtures.example/crc_open/{RELEASE}/{path}"
        for path in ("_CATALOG.json", "_SUCCESS", partition.path)
    ]


def test_portfolio_selection_scope_and_evaluation(tmp_path: Path) -> None:
    source, _ = _release(tmp_path)
    assets = pa.table(
        {"asset_id": ["Toronto"], "longitude": [-79.38], "latitude": [43.65]}
    )
    plan = HazardDataset.crc_open(release=RELEASE, fixtures=source).for_area(
        [-80, 43, -79, 44]
    )
    request = (
        plan.for_assets(assets)
        .select(hazard_names=["rx1day"], horizons=[2050])
        .return_periods([10])
    )
    with pytest.warns(CRCOpenFixtureWarning):
        result = request.write_parquet(tmp_path / "portfolio.parquet")
    assert result.row_count == 1
    assert pq.read_table(result.output)[result.value_columns[0]].to_pylist()[0] > 20
    with pytest.raises(ValueError, match="pathway"):
        request.select(pathways=["ssp245"]).write_parquet(tmp_path / "other.parquet")


def test_metadata_disagreement_and_empty_offline_cache(tmp_path: Path) -> None:
    source, catalog = _release(tmp_path)
    plan = HazardDataset.crc_open(release=RELEASE, fixtures=source).hazards(["rx1day"])
    with pytest.raises(FileNotFoundError, match="catalogue"):
        plan.cache(tmp_path / "empty-cache", mode="offline").prefetch()
    catalog["hazards"]["rx1day"]["unit"] = "incorrect"
    _publish_catalog(source / RELEASE, catalog)
    with pytest.warns(CRCOpenFixtureWarning):
        with pytest.raises(ValueError, match="metadata disagrees"):
            plan.materialize(tmp_path / "bad-unit.parquet")
    with pytest.raises(ValueError, match="persistent cache"):
        plan.prefetch()


def test_exclusions_and_unknown_semantics_are_preserved(tmp_path: Path) -> None:
    source, catalog = _release(tmp_path)
    catalog["excluded_hazards"] = {"cflood": "Source has no SSP585 pathway"}
    hazard = catalog["hazards"]["heat"]
    hazard["probability_semantics"] = None
    hazard["notes"] = "Source probability semantics are unspecified"
    for entry in hazard["partitions"]:
        path = source / RELEASE / entry["path"]
        table = pq.ParquetFile(path).read()
        metadata = HazardDatasetMetadata.from_parquet_metadata(table.schema.metadata)
        write_hazard_dataset(
            table, path, metadata.model_copy(update={"probability_semantics": None})
        )
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["size_bytes"] = path.stat().st_size
    _publish_catalog(source / RELEASE, catalog)
    plan = HazardDataset.crc_open(release=RELEASE, fixtures=source)
    with pytest.warns(CRCOpenFixtureWarning) as recorded:
        dataset = plan.hazards(["heat"]).materialize(tmp_path / "unknown.parquet")
    assert any("semantics are unspecified" in str(w.message) for w in recorded)
    assert dataset.metadata().probability_semantics is None
    with pytest.warns(CRCOpenFixtureWarning):
        with pytest.raises(ValueError, match="Source has no SSP585"):
            plan.hazards(["cflood"]).materialize(tmp_path / "excluded.parquet")
