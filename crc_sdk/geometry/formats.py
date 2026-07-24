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
        """Yields standard GEOMETRY and source_geometry."""
        con.execute("INSTALL spatial; LOAD spatial;")

        fmt = GeoFormat(fmt.lower()) if isinstance(fmt, str) else fmt

        if fmt in (GeoFormat.SHAPEFILE, GeoFormat.GEOJSON):
            raw_geom_col = "geom"
            from_clause = f"ST_Read('{file_path}') AS _src"

            select_cols = [
                f"_src.* EXCLUDE({raw_geom_col})",
                f"_src.{raw_geom_col} AS {geometry_column}",
            ]
            if preserve_source_geom:
                select_cols.append(
                    f"ST_AsWKB(_src.{raw_geom_col}) AS {source_geom_col_name}"
                )

            return f"(SELECT {', '.join(select_cols)} FROM {from_clause})"

        elif fmt == GeoFormat.GEOPARQUET:
            from_clause = f"read_parquet('{file_path}') AS _src"
            select_cols = ["_src.*"]
            if preserve_source_geom:
                select_cols.append(
                    f"ST_AsWKB(_src.{geometry_column}) AS {source_geom_col_name}"
                )

            return f"(SELECT {', '.join(select_cols)} FROM {from_clause})"

        elif fmt == GeoFormat.PARQUET:
            from_clause = f"read_parquet('{file_path}') AS _src"
            select_cols = [f"_src.* EXCLUDE({geometry_column})"]
            if preserve_source_geom:
                # Direct passthrough explicitly bound to raw table column
                select_cols.append(f"_src.{geometry_column} AS {source_geom_col_name}")
            select_cols.append(
                f"ST_GeomFromWKB(_src.{geometry_column}) AS {geometry_column}"
            )

            return f"(SELECT {', '.join(select_cols)} FROM {from_clause})"

        elif fmt == GeoFormat.WKT:
            from_clause = f"read_csv_auto('{file_path}') AS _src"
            select_cols = [f"_src.* EXCLUDE({geometry_column})"]
            if preserve_source_geom:
                select_cols.append(f"_src.{geometry_column} AS {source_geom_col_name}")
            select_cols.append(
                f"ST_GeomFromText(_src.{geometry_column}) AS {geometry_column}"
            )

            return f"(SELECT {', '.join(select_cols)} FROM {from_clause})"

        elif fmt == GeoFormat.WKB:
            from_clause = f"read_csv_auto('{file_path}') AS _src"
            select_cols = [f"_src.* EXCLUDE({geometry_column})"]
            if preserve_source_geom:
                select_cols.append(f"_src.{geometry_column} AS {source_geom_col_name}")
            select_cols.append(
                f"ST_GeomFromWKB(_src.{geometry_column}) AS {geometry_column}"
            )

            return f"(SELECT {', '.join(select_cols)} FROM {from_clause})"

        else:
            raise ValueError(f"Unsupported geometry format: {fmt}")
