from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from crc_sdk.connectors.agriculture import FTWFields, USDACropland
from crc_sdk.connectors.duckdb import DuckDBConnection


class _Array:
    def __init__(self, values: Any, attrs: dict[str, Any] | None = None) -> None:
        self.values = np.asarray(values)
        self.attrs = attrs or {}

    def __getitem__(self, key: Any) -> Any:
        return self.values[key]


class _IdentityTransformer:
    def transform_bounds(self, *bounds: float, **_: Any) -> tuple[float, ...]:
        return bounds

    def transform(self, x: Any, y: Any) -> tuple[Any, Any]:
        return x, y


class _Pyproj:
    class Transformer:
        @staticmethod
        def from_crs(*_: Any, **__: Any) -> _IdentityTransformer:
            return _IdentityTransformer()


def test_usda_cdl_reads_only_selected_aoi_years_in_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    crop = _Array(
        [
            [[0, 1, 5], [1, 5, 0], [0, 1, 5]],
            [[0, 5, 1], [5, 1, 0], [0, 5, 1]],
        ],
        attrs={
            "flag_values": [0, 1, 5],
            "class_names": ["Background", "Corn", "Soybeans"],
        },
    )
    group = {
        "crop_type": crop,
        "year": _Array([2024, 2025]),
        "x": _Array([0.5, 1.5, 2.5]),
        "y": _Array([2.5, 1.5, 0.5]),
    }
    monkeypatch.setattr(
        "crc_sdk.connectors.agriculture._open_cdl_group",
        lambda _: (group, object(), object(), _Pyproj(), object()),
    )
    scan = (
        USDACropland()
        .for_area((0.0, 1.0, 2.0, 2.0))
        .years(2025)
        .classes([1, 5])
        .scan(batch_rows=1)
    )

    assert all(batch.num_rows <= 1 for batch in scan._batches())
    relation = scan.relation(
        connection=DuckDBConnection.for_analytics(tmp_path, extensions=())
    )
    rows = relation.order("latitude DESC, longitude").fetchall()
    assert [(row[2], row[3], row[4]) for row in rows] == [
        (2025, 5, "Soybeans"),
        (2025, 1, "Corn"),
    ]


@pytest.mark.parametrize(
    ("selected", "expected"),
    [([0], [0, 0, 0]), ([0, 1, 5], [0, 0, 0, 1, 1, 1, 5, 5, 5])],
)
def test_usda_explicit_background_class_is_retained(
    monkeypatch: pytest.MonkeyPatch,
    selected: list[int],
    expected: list[int],
) -> None:
    crop = _Array(
        [[[0, 5, 1], [5, 1, 0], [0, 5, 1]]],
        attrs={
            "flag_values": [0, 1, 5],
            "class_names": ["Background", "Corn", "Soybeans"],
        },
    )
    group = {
        "crop_type": crop,
        "year": _Array([2025]),
        "x": _Array([0.5, 1.5, 2.5]),
        "y": _Array([2.5, 1.5, 0.5]),
    }
    monkeypatch.setattr(
        "crc_sdk.connectors.agriculture._open_cdl_group",
        lambda _: (group, object(), object(), _Pyproj(), object()),
    )
    scan = (
        USDACropland()
        .for_area((0.0, 0.0, 3.0, 3.0))
        .years(2025)
        .classes(selected)
        .scan()
    )

    codes = sorted(
        code.as_py()
        for batch in scan._batches()
        for code in batch.column("crop_code")
    )

    assert codes == expected


def test_usda_and_ftw_require_bounded_aoi() -> None:
    with pytest.raises(ValueError, match="for_area"):
        USDACropland().years(2025).scan()
    with pytest.raises(ValueError, match="for_area"):
        FTWFields().in_country("FR").scan()


class _SQLConnection:
    def __init__(self) -> None:
        self.query = ""
        self.executed = ""

    def sql(self, query: str) -> str:
        self.query = query
        return query

    def execute(self, query: str) -> _SQLConnection:
        self.executed = query
        return self


class _ConnectionConfig:
    def __init__(self, connection: _SQLConnection) -> None:
        self.connection = connection

    def connect(self) -> _SQLConnection:
        return self.connection


def test_ftw_builds_remote_predicate_pushdown_query() -> None:
    active = _SQLConnection()
    request = (
        FTWFields()
        .in_country("fr")
        .for_area((2.0, 47.5, 3.0, 48.5))
        .years([2024, 2025])
        .confidence_at_least(80)
        .scan()
    )
    result = request.relation(connection=_ConnectionConfig(active))  # type: ignore[arg-type]

    assert result == active.query
    assert "admin:country_code=FR/*.parquet" in active.query
    assert "confidence >= 80.0" in active.query
    assert "ST_MakeEnvelope" in active.query
    assert "IN (2024, 2025)" in active.query
    assert "CREATE OR REPLACE TEMPORARY SECRET" in active.executed
    assert f"SCOPE '{FTWFields().base_uri}'" in active.executed
