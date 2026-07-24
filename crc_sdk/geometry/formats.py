"""Multi-format GIS ingestion and DuckDB spatial expression builder."""

from enum import Enum

import duckdb


class GeoFormat(str, Enum):
    WKB = "wkb"
    WKT = "wkt"
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    PARQUET = "parquet"
    GEOPARQUET = "geoparquet"


class FormatAdapter:
    """Builds DuckDB spatial query snippets for various input formats."""

    @staticmethod
    def build_read_relation(
        con: duckdb.DuckDBPyConnection,
        file_path: str,
        fmt: GeoFormat,
        geometry_column: str = "geometry",
        preserve_source_geom: bool = True,
        source_geom_col_name: str = "source_geometry",
    ) -> str:
        """Generates a DuckDB SQL subquery expression yielding standard geometry and source_geometry."""

        con.execute("INSTALL spatial; LOAD spatial;")

        if fmt in (GeoFormat.SHAPEFILE, GeoFormat.GEOJSON):
            # ST_Read handles SHP and GeoJSON
            base_select = f"SELECT * FROM ST_Read('{file_path}')"
            geom_expr = f"geom AS {geometry_column}"
            source_expr = (
                f"ST_AsWKB(geom) AS {source_geom_col_name}"
                if preserve_source_geom
                else ""
            )
        elif fmt == GeoFormat.PARQUET or fmt == GeoFormat.GEOPARQUET:
            base_select = f"SELECT * FROM read_parquet('{file_path}')"
            geom_expr = (
                f"ST_GeomFromWKB({geometry_column}) AS {geometry_column}"
                if geometry_column != "geom"
                else f"{geometry_column}"
            )
            source_expr = (
                f"ST_AsWKB({geometry_column}) AS {source_geom_col_name}"
                if preserve_source_geom
                else ""
            )
        elif fmt == GeoFormat.WKT:
            base_select = f"SELECT * FROM read_csv_auto('{file_path}')"
            geom_expr = f"ST_GeomFromText({geometry_column}) AS {geometry_column}"
            source_expr = (
                f"{geometry_column} AS {source_geom_col_name}"
                if preserve_source_geom
                else ""
            )
        elif fmt == GeoFormat.WKB:
            base_select = f"SELECT * FROM read_csv_auto('{file_path}')"
            geom_expr = f"ST_GeomFromWKB({geometry_column}) AS {geometry_column}"
            source_expr = (
                f"{geometry_column} AS {source_geom_col_name}"
                if preserve_source_geom
                else ""
            )
        else:
            raise ValueError(f"Unsupported geometry format: {fmt}")

        select_cols = [f"* EXCLUDE({geometry_column})", geom_expr]
        if preserve_source_geom and source_expr:
            select_cols.append(source_expr)

        return f"(SELECT {', '.join(select_cols)} FROM {base_select})"
