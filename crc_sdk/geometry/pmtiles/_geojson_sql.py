"""DuckDB spatial-SQL Feature-JSON query builder.

This is the direct GeoParquet -> GeoJSONSeq bridge that replaces shelling out
to `gpio convert geojson`: the whole conversion is one SQL query per layer
(`ST_AsGeoJSON(ST_ReducePrecision(ST_Transform(...)))`, CRS auto-detected from
the source's own GeoParquet metadata, same as gpio's own approach), each
producing a single `feature: VARCHAR` column with any `tippecanoe` tag
already spliced in via string concatenation. N layers combine via a plain
`UNION ALL` -- one query, one Arrow batch reader, one write loop into
tippecanoe's stdin (see `_process.py`/`_build.py`), instead of N subprocess
feeds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from crc_sdk.connectors.duckdb import sql_quote

DEFAULT_PRECISION = 6
_WGS84 = "EPSG:4326"
_WGS84_ALIASES = frozenset({"EPSG:4326", "OGC:CRS84", "CRS84", "WGS84"})
STANDARD_GEOMETRY_NAMES = ("geometry", "geom", "wkb_geometry", "the_geom")


@dataclass(frozen=True)
class LayerSource:
    """One layer's resolved inputs for building its feature-selection query."""

    source: str
    layer: str
    minzoom: int
    maxzoom: int
    precision: int = DEFAULT_PRECISION


def geo_metadata(con: Any, source: str) -> dict[str, Any] | None:
    """Return a GeoParquet source's parsed `geo` metadata sidecar, if present.

    Reads only the schema of the first file matched by `source` (a glob or a
    single path) -- sufficient for a Hive-partitioned dataset written with a
    uniform schema, which is the only shape this SDK's own writers produce.
    """
    import fsspec  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    filesystem, path = fsspec.core.url_to_fs(source)
    is_glob = any(character in path for character in "*?[")
    first = sorted(filesystem.glob(path))[0] if is_glob else path
    with filesystem.open(first, "rb") as handle:
        metadata = pq.read_schema(handle).metadata or {}
    raw = metadata.get(b"geo")
    if raw is None:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _relation_column_types(con: Any, source_ref: str) -> dict[str, str]:
    result = con.execute(f"SELECT * FROM {source_ref} LIMIT 0")
    return {row[0]: str(row[1]) for row in result.description}


def _find_geometry_column(
    geo_meta: Mapping[str, Any] | None, columns: Sequence[str]
) -> str:
    if geo_meta:
        primary = geo_meta.get("primary_column")
        if isinstance(primary, str) and primary in columns:
            return primary
    lowered = {name.lower(): name for name in columns}
    for candidate in STANDARD_GEOMETRY_NAMES:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        f"could not find a geometry column among {list(columns)!r} -- expected "
        f"one of {STANDARD_GEOMETRY_NAMES!r}, or GeoParquet 'geo' metadata "
        "naming a primary_column"
    )


def _source_crs(geo_meta: Mapping[str, Any] | None, geometry_column: str) -> str | None:
    if not geo_meta:
        return None
    columns = geo_meta.get("columns") or {}
    column_meta = columns.get(geometry_column) or {}
    crs = column_meta.get("crs")
    if crs is None:
        return None
    if isinstance(crs, Mapping):
        identifier = crs.get("id") or {}
        authority, code = identifier.get("authority"), identifier.get("code")
        return f"{authority}:{code}" if authority and code else None
    return str(crs)


def _needs_reprojection(source_crs: str | None) -> bool:
    if source_crs is None:
        return False
    return source_crs.upper().replace(" ", "") not in _WGS84_ALIASES


def build_layer_query(con: Any, layer_source: LayerSource) -> str:
    """Build one layer's ``SELECT feature FROM ...`` sub-query.

    ``con`` must already have the ``spatial`` extension loaded (native
    GeoParquet geometry decoding plus
    ``ST_AsGeoJSON``/``ST_ReducePrecision``/``ST_Transform``).
    """
    source_ref = f"read_parquet({sql_quote(layer_source.source)})"
    column_types = _relation_column_types(con, source_ref)
    columns = list(column_types)
    geo_meta = geo_metadata(con, layer_source.source)
    geometry_column = _find_geometry_column(geo_meta, columns)
    source_crs = _source_crs(geo_meta, geometry_column)
    quoted_geom = _quote_ident(geometry_column)

    # DuckDB's `spatial` extension decodes a GeoParquet geometry column into
    # a native GEOMETRY type on read -- except for a genuinely empty (0 row
    # group) file, where it falls back to the raw BLOB/WKB storage type. An
    # explicit `ST_GeomFromWKB` cast handles both without needing to special
    # case "empty".
    is_native_geometry = column_types[geometry_column].upper().startswith("GEOMETRY")
    geom_expr = quoted_geom if is_native_geometry else f"ST_GeomFromWKB({quoted_geom})"
    if _needs_reprojection(source_crs):
        assert source_crs is not None
        geom_expr = (
            f"ST_Transform({quoted_geom}, {sql_quote(source_crs)}, {sql_quote(_WGS84)})"
        )
    geom_with_precision = (
        f"ST_ReducePrecision({geom_expr}, power(10, -{int(layer_source.precision)}))"
    )
    geom_json_expr = f"ST_AsGeoJSON({geom_with_precision})"

    excluded_names = {geometry_column.lower(), "bbox"}
    property_columns = [name for name in columns if name.lower() not in excluded_names]
    if property_columns:
        pairs = ", ".join(
            f"{_quote_ident(name)} := {_quote_ident(name)}" for name in property_columns
        )
        properties_expr = f"to_json(struct_pack({pairs}))"
    else:
        properties_expr = "'{}'"

    tag_literal = (
        '"tippecanoe":'
        + json.dumps(
            {
                "layer": layer_source.layer,
                "minzoom": layer_source.minzoom,
                "maxzoom": layer_source.maxzoom,
            },
            separators=(",", ":"),
        )
        + ","
    )

    return (
        "SELECT "
        "'{\"type\":\"Feature\",' || "
        f"{sql_quote(tag_literal)} || "
        "'\"geometry\":' || "
        f"COALESCE({geom_json_expr}, 'null') || "
        "',\"properties\":' || "
        f"{properties_expr} || "
        "'}' AS feature "
        f"FROM {source_ref} "
        f"WHERE {quoted_geom} IS NOT NULL"
    )


def build_combined_query(con: Any, layer_sources: Sequence[LayerSource]) -> str:
    """Combine one or more layer queries into a single ``feature`` relation.

    Multiple layers become a plain ``UNION ALL`` -- one Arrow-batched scan
    still produces every feature for every layer, so the caller feeds exactly
    one long-lived tippecanoe process regardless of layer count.
    """
    if not layer_sources:
        raise ValueError("at least one layer is required")
    queries = [build_layer_query(con, layer_source) for layer_source in layer_sources]
    if len(queries) == 1:
        return queries[0]
    unioned = " UNION ALL ".join(f"({query})" for query in queries)
    return f"SELECT feature FROM ({unioned})"
