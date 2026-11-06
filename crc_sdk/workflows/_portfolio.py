"""Internal SQL execution for portfolio evaluations."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from crc_sdk.connectors.duckdb import (
    DuckDBConnection,
    ensure_extensions,
    sql_quote,
)
from crc_sdk.connectors.parquet import read_hazard_metadata
from crc_sdk.providers.local import LocalProvider
from crc_sdk.schema import hazard_fields_for_version

from .distributions import (
    CURVE_COLUMNS,
    check_return_period_semantics,
    return_period_value_columns,
    return_periods_to_probabilities,
    stream_curve_quantiles_wide_to_parquet,
    warn_if_extrapolated,
    warn_if_outside_window,
)
from .portfolio import (
    _PORTFOLIO_RESERVED_COLUMNS,
    PORTFOLIO_METADATA_KEY,
    ImpactSpec,
    PortfolioEvaluationResult,
    _reserved_portfolio_columns,
)

if TYPE_CHECKING:
    import pyarrow as pa


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


def _aggregate_jrc_cells(
    con: Any,
    source: Path,
    output: Path,
    output_columns: Sequence[str],
    value_columns: Sequence[str],
    parquet_metadata: Mapping[bytes, bytes],
    aggregate: str,
) -> int:
    grouped = [
        column
        for column in output_columns
        if column not in ("source_id", "spatial_match")
    ]
    selected = [
        (
            "'jrc-cell-aggregate' AS source_id"
            if column == "source_id"
            else "'h3_cell_aggregate' AS spatial_match"
            if column == "spatial_match"
            else _sql_identifier(column)
        )
        for column in output_columns
    ]
    selected.extend(
        f"{aggregate}({_sql_identifier(column)}) AS {_sql_identifier(column)}"
        for column in value_columns
    )
    query = (
        f"SELECT {', '.join(selected)} FROM read_parquet({sql_quote(source)}) "
        f"GROUP BY {', '.join(_sql_identifier(column) for column in grouped)}"
    )
    reader = con.execute(query).to_arrow_reader(50_000)
    schema = reader.schema.with_metadata(dict(parquet_metadata))
    rows = 0
    staged = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    try:
        with pq.ParquetWriter(staged, schema, compression="zstd") as writer:
            for batch in reader:
                writer.write_batch(batch)
                rows += batch.num_rows
        os.replace(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    return rows


def _portfolio_metadata(
    *,
    metadata: Any,
    return_periods: tuple[float, ...],
    probabilities: tuple[float, ...],
    value_columns: tuple[str, ...],
    impact: ImpactSpec | None,
) -> Mapping[bytes, bytes]:
    payload = {
        "probability_convention": metadata.probability_convention,
        "return_period_tail": metadata.return_period_tail,
        "return_period_support": metadata.return_period_support,
        "value_unit": impact.value_unit if impact is not None else metadata.value_unit,
        "value_semantics": (
            impact.value_semantics if impact is not None else metadata.value_semantics
        ),
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
    if impact is not None:
        payload.update(
            {
                "interpretation": "event_aligned",
                "source_value_unit": metadata.value_unit,
                "source_value_semantics": metadata.value_semantics,
                "impact": {
                    "name": impact.name,
                    "type": impact.function_type,
                    "context_columns": dict(impact.context_columns.items()),
                },
            }
        )
        if impact.definition_hash is not None:
            payload["impact"]["definition_hash"] = impact.definition_hash
    return {
        PORTFOLIO_METADATA_KEY.encode("utf-8"): json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    }


def _point_input(
    longitude_column: str | None,
    latitude_column: str | None,
    cell_index_column: str | None,
) -> bool:
    """Whether assets are located by points; validate the location columns."""
    point_input = longitude_column is not None or latitude_column is not None
    if point_input:
        if longitude_column is None or latitude_column is None:
            raise ValueError("longitude and latitude columns must be provided together")
        if cell_index_column is not None:
            raise ValueError("choose longitude/latitude or cell_index, not both")
    elif cell_index_column is None:
        raise ValueError("provide longitude/latitude or a cell_index column")
    return point_input


def _curve_columns(metadata: Any) -> tuple[str, ...]:
    """The curve columns of a canonical dataset's schema version."""
    version_columns = {
        field.name for field in hazard_fields_for_version(metadata.schema_version)
    }
    return tuple(name for name in CURVE_COLUMNS if name in version_columns)


@dataclass(frozen=True)
class _CurveJoin:
    """The checked join of assets to their canonical curves."""

    matched_sql: str
    output_columns: tuple[str, ...]
    context_record_columns: Mapping[str, str]
    aggregate_jrc_cells: bool


def _join_curves(
    con: Any,
    provider: LocalProvider,
    assets_sql: str,
    metadata: Any,
    *,
    asset_id_column: str,
    longitude_column: str | None,
    latitude_column: str | None,
    cell_index_column: str | None,
    hazard_names: Sequence[str] | None,
    horizons: Sequence[int] | None,
    pathways: Sequence[str] | None,
    passthrough_columns: Sequence[str],
    impact: ImpactSpec | None,
    reserved: frozenset[str],
    aggregate_jrc: bool,
) -> _CurveJoin:
    """Join registered assets to canonical curves and run the join's checks.

    Returns the SQL of the matched rows, one per asset, hazard, horizon and
    pathway, and its output columns. Points are matched by H3 cell and
    refined by exact geometry; cells by H3 cell. Raises ``ValueError`` for
    invalid asset columns and ``LookupError`` for an asset with several or no
    curves for a selected scenario. With ``aggregate_jrc``, cell inputs of a
    JRC dataset may match several curves, for the caller to aggregate.
    """
    point_input = longitude_column is not None
    curve_columns = _curve_columns(metadata)
    identity_columns = [asset_id_column, *passthrough_columns]
    context_source_columns = (
        tuple(column for _, column in impact.context_columns.items())
        if impact is not None
        else ()
    )
    for name in context_source_columns:
        if name not in identity_columns:
            identity_columns.append(name)
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
    collision_candidates = [asset_id_column, *passthrough_columns]
    if point_input:
        assert longitude_column is not None
        assert latitude_column is not None
        collision_candidates.extend([longitude_column, latitude_column])
    collisions = reserved & set(collision_candidates)
    if collisions:
        raise ValueError(
            f"asset output columns collide with reserved names: {sorted(collisions)!r}"
        )

    asset_id = _sql_identifier(asset_id_column)
    longitude = (
        _sql_identifier(longitude_column) if longitude_column is not None else None
    )
    latitude = _sql_identifier(latitude_column) if latitude_column is not None else None
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
    context_record_columns = (
        {
            field_name: f"_crc_impact_context_{field_name}"
            for field_name, _ in impact.context_columns.items()
        }
        if impact is not None
        else {}
    )
    internal_context = (
        [
            f"a.{_sql_identifier(source_name)} AS "
            f"{_sql_identifier(context_record_columns[field_name])}"
            for field_name, source_name in impact.context_columns.items()
        ]
        if impact is not None
        else []
    )
    if point_input:
        assert longitude is not None
        assert latitude is not None
        cell_expression = (
            f"h3_latlng_to_cell(a.{latitude}, a.{longitude}, {metadata.h3_resolution})"
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
        cell_expression = f"CAST(a.{_sql_identifier(cell_index_column)} AS UBIGINT)"
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
        *internal_context,
        f"CAST({cell_expression} AS UBIGINT) AS cell_index",
        "h.hazard_name",
        "h.horizon",
        "h.pathway",
        "h.source_id",
        f"{spatial_match} AS spatial_match",
        *(f"h.{name}" for name in curve_columns),
    ]
    candidate_extras = []
    if point_input:
        assert longitude is not None
        assert latitude is not None
        point = f"ST_Point(a.{longitude}, a.{latitude})"
        geometry = "ST_GeomFromWKB(h.source_geometry)"
        candidate_extras = [
            "CASE WHEN h.source_geometry IS NULL THEN NULL "
            f"ELSE ST_Contains({geometry}, {point}) END AS _crc_contains",
            "CASE WHEN h.source_geometry IS NULL THEN NULL "
            f"ELSE ST_Distance({point}, ST_Centroid({geometry})) "
            "END AS _crc_centroid_distance",
        ]
    candidates_sql = f"""
        SELECT {", ".join([*selected, *candidate_extras])}
        FROM {assets_sql} AS a
        JOIN read_parquet({sql_quote(provider.source)}) AS h
          ON h.cell_index = CAST({cell_expression} AS UBIGINT)
        WHERE TRUE{filters}{spatial_predicate}
    """
    matched_sql = (
        f"""
        SELECT * EXCLUDE (_crc_contains, _crc_centroid_distance)
        FROM ({candidates_sql})
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY {asset_id}, hazard_name, horizon, pathway
            ORDER BY _crc_contains DESC NULLS LAST,
                     _crc_centroid_distance ASC NULLS LAST,
                     source_id
        ) = 1
        """
        if point_input
        else candidates_sql
    )

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
    aggregate_jrc_cells = (
        aggregate_jrc and not point_input and metadata.source.provider == "jrc"
    )
    ambiguity_source = candidates_sql if point_input else matched_sql
    ambiguity_condition = (
        "HAVING COUNT(*) > 1 AND ("
        "COUNT_IF(_crc_contains) > 1 OR "
        "COUNT_IF(_crc_contains IS NULL) > 0)"
        if point_input
        else "HAVING COUNT(*) > 1"
    )
    ambiguous = (
        []
        if aggregate_jrc_cells
        else con.execute(
            f"""
            SELECT {asset_id}, hazard_name, horizon, pathway,
                   COUNT(*) AS source_count
            FROM ({ambiguity_source})
            GROUP BY {asset_id}, hazard_name, horizon, pathway
            {ambiguity_condition}
            LIMIT 10
            """
        ).fetchall()
    )
    if ambiguous:
        raise LookupError(
            f"portfolio assets match multiple source curves: {ambiguous!r}"
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
        raise LookupError(f"portfolio assets are missing hazard curves: {missing!r}")

    output_columns = (
        *output_asset_columns,
        "cell_index",
        "hazard_name",
        "horizon",
        "pathway",
        "source_id",
        "spatial_match",
    )
    return _CurveJoin(
        matched_sql=matched_sql,
        output_columns=output_columns,
        context_record_columns=context_record_columns,
        aggregate_jrc_cells=aggregate_jrc_cells,
    )


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
    impact: ImpactSpec | None = None,
    connection: Any | None = None,
    batch_rows: int = 50_000,
    max_workers: int | None = None,
    chunk_rows: int = 20_000,
) -> PortfolioEvaluationResult:
    """Join assets to canonical curves and stream wide RP values to Parquet."""
    if not isinstance(provider, LocalProvider):
        raise TypeError("provider must be a LocalProvider")
    point_input = _point_input(longitude_column, latitude_column, cell_index_column)

    periods = tuple(float(period) for period in return_periods)
    value_columns = return_period_value_columns(periods)
    metadata = read_hazard_metadata(provider.source)
    check_return_period_semantics(metadata, stacklevel=3)
    for horizon in horizons or ():
        warn_if_outside_window(horizon, metadata.temporal_window, stacklevel=3)
    warn_if_extrapolated(periods, metadata.return_period_support, stacklevel=3)
    probabilities = return_periods_to_probabilities(
        return_periods,
        tail=metadata.return_period_tail,
    )
    output_path = Path(output)
    if output_path.resolve() == Path(provider.source).resolve():
        raise ValueError("portfolio output must not overwrite the hazard source")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    owned = connection is None
    extensions = ("spatial", "h3") if point_input else ()
    con = (
        connection
        or DuckDBConnection.for_analytics(
            output_path.parent,
            extensions=extensions,
        ).connect()
    )
    if connection is not None and extensions:
        ensure_extensions(con, *extensions)

    relation_name = f"_crc_assets_{uuid4().hex}"
    registered = False
    try:
        assets_sql, registered = _assets_source(con, assets, relation_name)
        join = _join_curves(
            con,
            provider,
            assets_sql,
            metadata,
            asset_id_column=asset_id_column,
            longitude_column=longitude_column,
            latitude_column=latitude_column,
            cell_index_column=cell_index_column,
            hazard_names=hazard_names,
            horizons=horizons,
            pathways=pathways,
            passthrough_columns=passthrough_columns,
            impact=impact,
            reserved=_reserved_portfolio_columns(periods),
            aggregate_jrc=True,
        )
        evaluation_metadata = _portfolio_metadata(
            metadata=metadata,
            return_periods=periods,
            probabilities=probabilities,
            value_columns=value_columns,
            impact=impact,
        )
        evaluation_output = (
            output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp.parquet")
            if join.aggregate_jrc_cells
            else output_path
        )
        row_count = stream_curve_quantiles_wide_to_parquet(
            con,
            join.matched_sql,
            probabilities,
            evaluation_output,
            passthrough_columns=join.output_columns,
            value_columns=value_columns,
            parquet_metadata=evaluation_metadata,
            impact=impact,
            context_columns=join.context_record_columns,
            batch_rows=batch_rows,
            max_workers=max_workers,
            chunk_rows=chunk_rows,
        )
        if join.aggregate_jrc_cells:
            try:
                row_count = _aggregate_jrc_cells(
                    con,
                    evaluation_output,
                    output_path,
                    join.output_columns,
                    value_columns,
                    evaluation_metadata,
                    "MAX" if metadata.return_period_tail == "upper" else "MIN",
                )
            finally:
                evaluation_output.unlink(missing_ok=True)
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


def hazard_portfolio_curves(
    provider: LocalProvider,
    assets: Any,
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
) -> pa.Table:
    """Join assets to canonical curves and return the matched curve rows.

    The join and its checks are those of ``evaluate_hazard_portfolio``,
    except that cell inputs of a JRC dataset that match several curves raise
    instead of being aggregated: a set of curves has no single curve row.
    Rows are sorted by asset id, hazard, horizon and pathway.
    """
    if not isinstance(provider, LocalProvider):
        raise TypeError("provider must be a LocalProvider")
    point_input = _point_input(longitude_column, latitude_column, cell_index_column)
    metadata = read_hazard_metadata(provider.source)
    for horizon in horizons or ():
        warn_if_outside_window(horizon, metadata.temporal_window, stacklevel=4)
    curve_columns = _curve_columns(metadata)
    extensions = ("spatial", "h3") if point_input else ()
    relation_name = f"_crc_assets_{uuid4().hex}"
    with ExitStack() as stack:
        if connection is None:
            work_dir = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="crc-asset-curves-")
            )
            con = DuckDBConnection.for_analytics(
                work_dir, extensions=extensions
            ).connect()
            stack.callback(con.close)
        else:
            con = connection
            if extensions:
                ensure_extensions(con, *extensions)
        assets_sql, registered = _assets_source(con, assets, relation_name)
        if registered:
            stack.callback(con.unregister, relation_name)
        join = _join_curves(
            con,
            provider,
            assets_sql,
            metadata,
            asset_id_column=asset_id_column,
            longitude_column=longitude_column,
            latitude_column=latitude_column,
            cell_index_column=cell_index_column,
            hazard_names=hazard_names,
            horizons=horizons,
            pathways=pathways,
            passthrough_columns=passthrough_columns,
            impact=None,
            reserved=_PORTFOLIO_RESERVED_COLUMNS,
            aggregate_jrc=False,
        )
        columns = (*join.output_columns, *curve_columns)
        order = ", ".join(
            _sql_identifier(name)
            for name in (asset_id_column, "hazard_name", "horizon", "pathway")
        )
        query = (
            f"SELECT {', '.join(_sql_identifier(name) for name in columns)} "
            f"FROM ({join.matched_sql}) ORDER BY {order}"
        )
        result = con.execute(query)
        if hasattr(result, "to_arrow_table"):
            return result.to_arrow_table()
        return result.fetch_arrow_table()
