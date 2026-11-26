"""Selecting from a local Dataset file reads and decodes only what it needs."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import crc_framework as cf
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely.geometry import Point  # type: ignore[import-untyped]

from crc_sdk import Bounds, Dataset, Runtime
from crc_sdk._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionReport,
    SourceUnavailable,
    SpatialContract,
    SupportKind,
)
from crc_sdk._declarations import DataSpec, SeriesSpec, Variable
from crc_sdk._evaluation_spatial import candidate_cells
from crc_sdk._recipes import _group_no_data, _ParquetSelectionNode
from crc_sdk._schema.dataset import DatasetRow, rows_to_table

_CENTRES = ((10.0, 20.0), (-30.0, 140.0), (45.0, -75.0))


def _rows() -> list[DatasetRow]:
    rows = []
    for variable in ("company:a", "company:b"):
        for latitude, longitude in _CENTRES:
            centre = h3.latlng_to_cell(latitude, longitude, 5)
            for index, cell in enumerate(sorted(h3.grid_disk(centre, 1))):
                rows.append(
                    DatasetRow(
                        variable_id=variable,
                        support_id=f"h3:{cell:x}",
                        cell_id=cell,
                        curve=None
                        if index % 3 == 0
                        else cf.QuantileCurve([0.2, 0.8], [1.0, 3.0 + index]),
                        failure=SourceUnavailable("no_source", "no source value")
                        if index % 3 == 0
                        else None,
                    )
                )
    return sorted(rows, key=lambda row: (row.variable_id, row.cell_id))


def _metadata(rows: list[DatasetRow], *, no_data: int | None = None) -> DatasetMetadata:
    failures = sum(row.failure is not None for row in rows)
    return DatasetMetadata(
        spec=DataSpec(
            [Variable("company:a"), Variable("company:b")],
            series=[
                SeriesSpec(Variable("company:a")),
                SeriesSpec(Variable("company:b")),
            ],
        ),
        spatial=SpatialContract(supports=(SupportKind.CELL,), index_resolution=5),
        report=ExecutionReport(
            complete=True,
            valid=True,
            counts=ExecutionCounts(
                requested=len(rows),
                produced=len(rows),
                no_data=failures if no_data is None else no_data,
            ),
        ),
    )


def _write(path: Path, table: pa.Table) -> Path:
    # Small row groups, sorted by variable and cell as releases are.
    pq.write_table(table, path, row_group_size=7)
    return path


@pytest.fixture
def stored(tmp_path: Path) -> tuple[Path, pa.Table, DatasetMetadata]:
    rows = _rows()
    metadata = _metadata(rows)
    table = rows_to_table(rows, metadata)
    return _write(tmp_path / "rows.parquet", table), table, metadata


def _restricted(dataset: Dataset, latitude: float, longitude: float) -> Dataset:
    """The cells an asset at this point can match, as evaluation restricts."""
    return dataset._restricted(candidate_cells([Point(longitude, latitude).wkb]))


def test_selected_local_reads_equal_selecting_the_whole_table(
    stored: tuple[Path, pa.Table, DatasetMetadata],
) -> None:
    path, table, metadata = stored
    file, memory = Dataset.open(path), Dataset._collected(table, metadata)
    assert isinstance(file.select(variables="company:b")._node, _ParquetSelectionNode)
    latitude, longitude = _CENTRES[1]
    cases = (
        lambda d: d.select(variables="company:b"),
        lambda d: d.select(area=Bounds(west=130, south=-40, east=150, north=-20)),
        lambda d: _restricted(d, latitude, longitude),
        lambda d: _restricted(d.select(variables="company:a"), latitude, longitude),
        lambda d: d.select(variables=[]),
    )
    for case in cases:
        for runtime in (Runtime(), Runtime(batch_rows=2)):
            actual, expected = (
                case(file).read(runtime=runtime),
                case(memory).read(runtime=runtime),
            )
            assert actual.table.equals(expected.table)
            assert actual.report.valid is expected.report.valid is True
            assert actual.report.counts == expected.report.counts


def test_row_groups_the_statistics_exclude_are_not_read(
    stored: tuple[Path, pa.Table, DatasetMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = stored
    read: list[list[int]] = []
    original = pq.ParquetFile.iter_batches

    def spy(self: pq.ParquetFile, *args: object, **kwargs: object) -> object:
        groups = cast(list[int] | None, kwargs.get("row_groups"))
        read.append(list(range(self.num_row_groups)) if groups is None else groups)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    groups = pq.ParquetFile(path).num_row_groups
    latitude, longitude = _CENTRES[0]
    result = _restricted(
        Dataset.open(path).select(variables="company:a"), latitude, longitude
    ).read()
    assert result.report.valid is True and result.table.num_rows
    assert read and 0 < len(read[0]) < groups


def test_statistics_count_unread_failures_exactly(
    stored: tuple[Path, pa.Table, DatasetMetadata],
) -> None:
    path, _, _ = stored
    reader = pq.ParquetFile(path)
    for index in range(reader.num_row_groups):
        failures = reader.read_row_group(index, columns=["failure"]).column("failure")
        assert _group_no_data(reader, index) == len(failures) - failures.null_count


def test_stored_counts_of_unread_row_groups_are_still_checked(
    tmp_path: Path,
) -> None:
    rows = _rows()
    table = rows_to_table(rows, _metadata(rows))
    # The footer claims no failures; the row count still agrees with it.
    wrong = _metadata(rows, no_data=0)
    path = _write(
        tmp_path / "wrong.parquet",
        table.replace_schema_metadata({b"crc.dataset": wrong.to_json().encode()}),
    )
    latitude, longitude = _CENTRES[0]
    for dataset in (
        Dataset.open(path),
        Dataset.open(path).select(variables="company:b"),
        Dataset.open(path).select(variables=[]),
        _restricted(Dataset.open(path), latitude, longitude),
    ):
        report = dataset.validate()
        assert report.valid is False
        assert "stored_count_mismatch" in {d.code for d in report.diagnostics}


def test_curves_outside_the_selection_are_not_decoded(
    stored: tuple[Path, pa.Table, DatasetMetadata], tmp_path: Path
) -> None:
    _, table, _ = stored
    records = table.to_pylist()
    broken = next(r for r in records if r["variable_id"] == "company:b" and r["curve"])
    broken["curve"]["quantile_curve"]["values"] = [3.0, 1.0]
    path = _write(
        tmp_path / "broken.parquet", pa.Table.from_pylist(records, schema=table.schema)
    )
    assert Dataset.open(path).validate().valid is False
    assert Dataset.open(path).select(variables="company:b").validate().valid is False
    assert Dataset.open(path).select(variables="company:a").validate().valid is True
