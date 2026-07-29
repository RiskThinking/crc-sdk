"""Internal SQL execution for portfolio evaluations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from crc_sdk.connectors.duckdb import (
    DuckDBConnection,
    ensure_extensions,
    sql_quote,
)
from crc_sdk.connectors.parquet import read_hazard_metadata
from crc_sdk.providers.local import LocalProvider

from .distributions import (
    CURVE_COLUMNS,
    return_period_value_columns,
    return_periods_to_probabilities,
    stream_curve_quantiles_wide_to_parquet,
)
from .portfolio import PORTFOLIO_METADATA_KEY, PortfolioEvaluationResult


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_values(values: Sequence[Any]) -> str:
    rendered = []
    for value in values:
        if isinstance(value, str):
            rendered.append(sql_quote(value))
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("scenario filters must contain strings or numbers")
        else:
            rendered.append(str(value))
    return ", ".join(rendered)


def _filter_sql(column: str, values: Sequence[Any] | None) -> str:
    if values is None:
        return ""
    normalized = tuple(values)
    if not normalized:
        return " AND FALSE"
    return f" AND h.{_sql_identifier(column)} IN ({_sql_values(normalized)})"


def _assets_source(
    con: Any,
    assets: Any,
    relation_name: str,
) -> tuple[str, bool]:
    if isinstance(assets, Path):
        return f"read_parquet({sql_quote(assets)})", False
    if isinstance(assets, str):
        return f"({assets})", False
    con.register(relation_name, assets)
    return _sql_identifier(relation_name), True


def _validate_asset_columns(
    con: Any,
    source_sql: str,
    required: Sequence[str],
) -> None:
    schema = con.execute(f"SELECT * FROM {source_sql} LIMIT 0").description
    available = {column[0] for column in schema}
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"asset source is missing columns {missing!r}")


def _portfolio_metadata(
    *,
    metadata: Any,
    return_periods: tuple[float, ...],
    probabilities: tuple[float, ...],
    value_columns: tuple[str, ...],
) -> Mapping[bytes, bytes]:
    payload = {
        "probability_convention": metadata.probability_convention,
        "value_unit": metadata.value_unit,
        "value_semantics": metadata.value_semantics,
        "return_periods": [
            {
                "return_period": period,
                "probability": probability,
                "column": column,
            }
            for period, probability, column in zip(
                return_periods,
                probabilities,
                value_columns,
            )
        ],
    }
    return {
        PORTFOLIO_METADATA_KEY.encode("utf-8"): json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    }


def evaluate_hazard_portfolio(
    provider: LocalProvider,
    assets: Any,
    output: str | Path,
    return_periods: Sequence[float],
    *,
    asset_id_column: str = "asset_id",
    longitude_column: str | None = None,
    latitude_column: str | None = None,
    cell_index_column: str | None = None,
    hazard_names: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    pathways: Sequence[str] | None = None,
    passthrough_columns: Sequence[str] = (),
    connection: Any | None = None,
    batch_rows: int = 50_000,
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> PortfolioEvaluationResult:
    """Join assets to canonical curves and stream wide RP values to Parquet."""
    if not isinstance(provider, LocalProvider):
        raise TypeError("provider must be a LocalProvider")
    point_input = longitude_column is not None or latitude_column is not None
    if point_input:
        if longitude_column is None or latitude_column is None:
            raise ValueError("longitude and latitude columns must be provided together")
        if cell_index_column is not None:
            raise ValueError("choose longitude/latitude or cell_index, not both")
    elif cell_index_column is None:
        raise ValueError("provide longitude/latitude or a cell_index column")

    probabilities = return_periods_to_probabilities(return_periods)
    periods = tuple(float(period) for period in return_periods)
    value_columns = return_period_value_columns(periods)
    metadata = read_hazard_metadata(provider.source)
    output_path = Path(output)
    if output_path.resolve() == Path(provider.source).resolve():
        raise ValueError("portfolio output must not overwrite the hazard source")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    owned = connection is None
    extensions = ("spatial", "h3") if point_input else ()
    con = connection or DuckDBConnection.for_analytics(
        output_path.parent,
        extensions=extensions,
    ).connect()
    if connection is not None and extensions:
        ensure_extensions(con, *extensions)

    relation_name = f"_crc_assets_{uuid4().hex}"
    registered = False
    try:
        assets_sql, registered = _assets_source(con, assets, relation_name)
        identity_columns = [asset_id_column, *passthrough_columns]
        if point_input:
            assert longitude_column is not None
            assert latitude_column is not None
            for name in (longitude_column, latitude_column):
                if name not in identity_columns:
                    identity_columns.append(name)
        else:
            assert cell_index_column is not None
            if cell_index_column not in identity_columns:
                identity_columns.append(cell_index_column)
        _validate_asset_columns(con, assets_sql, identity_columns)
        if len(set(identity_columns)) != len(identity_columns):
            raise ValueError("asset identity and passthrough columns must be unique")
        reserved = {
            "cell_index",
            "hazard_name",
            "horizon",
            "pathway",
            "source_id",
            "spatial_match",
            *value_columns,
        }
        collision_candidates = [asset_id_column, *passthrough_columns]
        if point_input:
            assert longitude_column is not None
            assert latitude_column is not None
            collision_candidates.extend([longitude_column, latitude_column])
        collisions = reserved & set(collision_candidates)
        if collisions:
            raise ValueError(
                f"asset output columns collide with reserved names: "
                f"{sorted(collisions)!r}"
            )

        asset_id = _sql_identifier(asset_id_column)
        longitude = (
            _sql_identifier(longitude_column)
            if longitude_column is not None
            else None
        )
        latitude = (
            _sql_identifier(latitude_column)
            if latitude_column is not None
            else None
        )
        output_asset_columns = [asset_id_column, *passthrough_columns]
        if point_input:
            assert longitude_column is not None
            assert latitude_column is not None
            for name in (longitude_column, latitude_column):
                if name not in output_asset_columns:
                    output_asset_columns.append(name)
        passthrough = [
            f"a.{_sql_identifier(name)} AS {_sql_identifier(name)}"
            for name in output_asset_columns
            if name != asset_id_column
        ]
        if point_input:
            assert longitude is not None
            assert latitude is not None
            cell_expression = (
                f"h3_latlng_to_cell(a.{latitude}, a.{longitude}, "
                f"{metadata.h3_resolution})"
            )
            spatial_predicate = (
                " AND (h.source_geometry IS NULL OR "
                "ST_Covers(ST_GeomFromWKB(h.source_geometry), "
                f"ST_Point(a.{longitude}, a.{latitude})))"
            )
            spatial_match = (
                "CASE WHEN h.source_geometry IS NULL THEN 'h3_cell' "
                "ELSE 'exact_geometry' END"
            )
        else:
            assert cell_index_column is not None
            cell_expression = (
                f"CAST(a.{_sql_identifier(cell_index_column)} AS UBIGINT)"
            )
            spatial_predicate = ""
            spatial_match = "'h3_cell'"

        filters = (
            _filter_sql("hazard_name", hazard_names)
            + _filter_sql("horizon", horizons)
            + _filter_sql("pathway", pathways)
        )
        selected = [
            f"a.{asset_id} AS {asset_id}",
            *passthrough,
            f"CAST({cell_expression} AS UBIGINT) AS cell_index",
            "h.hazard_name",
            "h.horizon",
            "h.pathway",
            "h.source_id",
            f"{spatial_match} AS spatial_match",
            *(f"h.{name}" for name in CURVE_COLUMNS),
        ]
        matched_sql = f"""
            SELECT {", ".join(selected)}
            FROM {assets_sql} AS a
            JOIN read_parquet({sql_quote(provider.source)}) AS h
              ON h.cell_index = CAST({cell_expression} AS UBIGINT)
            WHERE TRUE{filters}{spatial_predicate}
        """

        duplicate_assets = con.execute(
            f"""
            SELECT {asset_id}, COUNT(*) AS count
            FROM {assets_sql}
            GROUP BY {asset_id}
            HAVING COUNT(*) > 1
            LIMIT 10
            """
        ).fetchall()
        if duplicate_assets:
            raise ValueError(
                f"asset_id values must be unique; duplicates: {duplicate_assets!r}"
            )
        null_asset_id = con.execute(
            f"SELECT 1 FROM {assets_sql} WHERE {asset_id} IS NULL LIMIT 1"
        ).fetchone()
        if null_asset_id is not None:
            raise ValueError("asset_id values must not be null")
        ambiguous = con.execute(
            f"""
            SELECT {asset_id}, hazard_name, horizon, pathway,
                   COUNT(*) AS source_count
            FROM ({matched_sql})
            GROUP BY {asset_id}, hazard_name, horizon, pathway
            HAVING COUNT(*) > 1
            LIMIT 10
            """
        ).fetchall()
        if ambiguous:
            raise LookupError(
                "portfolio assets match multiple source curves: "
                f"{ambiguous!r}"
            )
        missing = con.execute(
            f"""
            WITH scenarios AS (
                SELECT DISTINCT h.hazard_name, h.horizon, h.pathway
                FROM read_parquet({sql_quote(provider.source)}) AS h
                WHERE TRUE{filters}
            ),
            matched AS (
                SELECT DISTINCT {asset_id}, hazard_name, horizon, pathway
                FROM ({matched_sql})
            )
            SELECT a.{asset_id}, s.hazard_name, s.horizon, s.pathway
            FROM {assets_sql} AS a
            CROSS JOIN scenarios AS s
            LEFT JOIN matched AS m
              ON m.{asset_id} = a.{asset_id}
             AND m.hazard_name = s.hazard_name
             AND m.horizon = s.horizon
             AND m.pathway = s.pathway
            WHERE m.{asset_id} IS NULL
            LIMIT 10
            """
        ).fetchall()
        if missing:
            raise LookupError(
                "portfolio assets are missing hazard curves: "
                f"{missing!r}"
            )

        output_columns = [
            *output_asset_columns,
            "cell_index",
            "hazard_name",
            "horizon",
            "pathway",
            "source_id",
            "spatial_match",
        ]
        row_count = stream_curve_quantiles_wide_to_parquet(
            con,
            matched_sql,
            probabilities,
            output_path,
            passthrough_columns=output_columns,
            value_columns=value_columns,
            parquet_metadata=_portfolio_metadata(
                metadata=metadata,
                return_periods=periods,
                probabilities=probabilities,
                value_columns=value_columns,
            ),
            batch_rows=batch_rows,
            max_workers=max_workers,
            chunk_rows=chunk_rows,
        )
    finally:
        if registered:
            con.unregister(relation_name)
        if owned:
            con.close()

    return PortfolioEvaluationResult(
        output=output,
        row_count=row_count,
        return_periods=periods,
        value_columns=value_columns,
    )
