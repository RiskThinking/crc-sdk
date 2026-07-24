"""Administrative H3 lookup contracts and consumer catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from crc_sdk.connectors.duckdb.connection import sql_quote

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def enrich_adm2_with_adm1_sql(
    adm1_source: str,
    adm2_source: str,
    *,
    adm1_id_expr: str = "shapeID::VARCHAR",
    adm1_name_expr: str = "shapeName",
    adm1_group_expr: str = "shapeGroup",
    adm2_id_expr: str = "shapeID::VARCHAR",
    adm2_name_expr: str = "shapeName",
    adm2_group_expr: str = "shapeGroup",
    geom_col: str = "geom",
) -> str:
    """Max-overlap join of child ADM2 polygons onto parent ADM1 attributes."""
    adm1 = sql_quote(adm1_source)
    adm2 = sql_quote(adm2_source)
    return f"""
        WITH adm1_data AS (
            SELECT row_number() OVER () AS adm1_rid, *
            FROM ST_Read({adm1})
        ),
        adm2_data AS (
            SELECT row_number() OVER () AS adm2_rid, *
            FROM ST_Read({adm2})
        ),
        adm2_adm1_join AS (
            SELECT
                a2.adm2_rid,
                a2.{adm2_id_expr} AS adm2_id,
                a2.{adm2_name_expr} AS adm2_name,
                a2.{adm2_group_expr} AS adm0_iso,
                a1.{adm1_id_expr} AS adm1_id,
                a1.{adm1_name_expr} AS adm1_name,
                a2.{geom_col} AS geom,
                ST_Area(ST_Intersection(a2.{geom_col}, a1.{geom_col})) AS overlap_area
            FROM adm2_data AS a2
            LEFT JOIN adm1_data AS a1
              ON ST_Intersects(a2.{geom_col}, a1.{geom_col})
             AND a2.{adm2_group_expr} = a1.{adm1_group_expr}
        )
        SELECT
            adm2_rid,
            adm2_id,
            adm2_name,
            adm0_iso,
            arg_max(adm1_id, overlap_area) AS adm1_id,
            arg_max(adm1_name, overlap_area) AS adm1_name,
            ST_MakeValid(ANY_VALUE(geom)) AS geom
        FROM adm2_adm1_join
        GROUP BY adm2_rid, adm2_id, adm2_name, adm0_iso
    """


def write_lookup_contract(
    con: DuckDBPyConnection,
    exploded_path: str | Path,
    output_path: str | Path,
    *,
    resolution: int,
    include_coverage: bool,
    interior_threshold: float = 0.999999,
) -> None:
    """Derive a flat, coherent ADM assignment contract from exact cell coverage."""
    escaped_input = sql_quote(exploded_path)
    escaped_output = sql_quote(output_path)
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({escaped_input})"
        ).fetchall()
    }
    def _optional(column: str) -> str:
        return column if column in columns else f"NULL::VARCHAR AS {column}"

    adm1_id = _optional("adm1_id")
    adm1_name = _optional("adm1_name")
    adm2_id = _optional("adm2_id")
    adm2_name = _optional("adm2_name")
    parents = []
    for parent_resolution in range(3, 8):
        expression = (
            "h3_h3_to_string(h3_cell_to_parent("
            f"h3_string_to_h3(hex_id), {parent_resolution}))"
            if parent_resolution <= resolution
            else "NULL::VARCHAR"
        )
        parents.append(f"{expression} AS h3_r{parent_resolution}")
    parent_sql = ",\n                   ".join(parents)
    coverage_sql = (
        """,
               list(struct_pack(
                   adm0_iso := adm0_iso,
                   adm1_id := adm1_id,
                   adm1_name := adm1_name,
                   adm2_id := adm2_id,
                   adm2_name := adm2_name,
                   pct := pct
               ) ORDER BY pct DESC, adm0_iso, adm1_id, adm2_id) AS coverage"""
        if include_coverage
        else ""
    )
    con.execute(
        f"""
        COPY (
            WITH source AS (
                SELECT hex_id, adm0_iso,
                       {adm1_id}, {adm1_name},
                       {adm2_id}, {adm2_name},
                       greatest(0.0, least(1.0, pct))::DOUBLE AS pct
                FROM read_parquet({escaped_input})
                WHERE pct > 0
            ), paths AS (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY hex_id
                           ORDER BY pct DESC, adm0_iso,
                                    adm1_id NULLS LAST, adm1_name,
                                    adm2_id NULLS LAST, adm2_name
                       ) AS path_rank
                FROM source
            ), best_path AS (
                SELECT * EXCLUDE (path_rank)
                FROM paths
                WHERE path_rank = 1
            ), adm0_rollup AS (
                SELECT hex_id, adm0_iso, least(1.0, sum(pct)) AS pct
                FROM source GROUP BY hex_id, adm0_iso
            ), adm1_rollup AS (
                SELECT hex_id, adm0_iso, adm1_id, adm1_name,
                       least(1.0, sum(pct)) AS pct
                FROM source
                GROUP BY hex_id, adm0_iso, adm1_id, adm1_name
            ), counts AS (
                SELECT hex_id,
                       count(DISTINCT adm0_iso)::INTEGER AS adm0_candidate_count,
                       count(DISTINCT struct_pack(
                           adm0_iso := adm0_iso,
                           adm1 := coalesce(adm1_id, adm1_name)
                       ))::INTEGER AS adm1_candidate_count,
                       count(*)::INTEGER AS adm2_candidate_count
                       {coverage_sql}
                FROM source
                GROUP BY hex_id
            )
            SELECT b.hex_id,
                   {parent_sql},
                   b.adm0_iso AS best_adm0,
                   b.adm1_id AS best_adm1_id,
                   b.adm1_name AS best_adm1,
                   b.adm2_id AS best_adm2_id,
                   b.adm2_name AS best_adm2,
                   a0.pct::DOUBLE AS best_adm0_pct,
                   a1.pct::DOUBLE AS best_adm1_pct,
                   b.pct::DOUBLE AS best_adm2_pct,
                   b.pct::DOUBLE AS best_pct,
                   c.adm0_candidate_count,
                   c.adm1_candidate_count,
                   c.adm2_candidate_count,
                   c.adm2_candidate_count AS candidate_count,
                   c.adm0_candidate_count = 1
                       AND a0.pct >= {interior_threshold} AS is_adm0_interior,
                   c.adm1_candidate_count = 1
                       AND a1.pct >= {interior_threshold} AS is_adm1_interior,
                   c.adm2_candidate_count = 1
                       AND b.pct >= {interior_threshold} AS is_adm2_interior
                   {", c.coverage" if include_coverage else ""}
            FROM best_path AS b
            JOIN counts AS c USING (hex_id)
            JOIN adm0_rollup AS a0
              ON b.hex_id = a0.hex_id AND b.adm0_iso = a0.adm0_iso
            JOIN adm1_rollup AS a1
              ON b.hex_id = a1.hex_id
             AND b.adm0_iso = a1.adm0_iso
             AND b.adm1_name IS NOT DISTINCT FROM a1.adm1_name
             AND b.adm1_id IS NOT DISTINCT FROM a1.adm1_id
            ORDER BY h3_r3, hex_id
        ) TO {escaped_output}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
        """
    )


def write_partitioned_lookup(
    con: DuckDBPyConnection,
    input_path: str | Path,
    output_path: str | Path,
    *,
    resolution: int,
    partition_resolution: int = 0,
    row_group_size: int = 122_880,
) -> None:
    """Repartition a nested lookup without materializing exploded coverage."""
    if not 0 <= partition_resolution < resolution <= 15:
        raise ValueError("require 0 <= partition_resolution < resolution <= 15")
    input_sql = sql_quote(input_path)
    output_sql = sql_quote(output_path)
    parent = f"h3_r{partition_resolution}"
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({input_sql})"
        ).fetchall()
    }
    parent_expression = (
        parent
        if parent in columns
        else (
            "h3_h3_to_string(h3_cell_to_parent("
            f"h3_string_to_h3(hex_id), {partition_resolution}))"
        )
    )
    projection = (
        "*" if parent in columns else f"*, {parent_expression}::VARCHAR AS {parent}"
    )
    con.execute(
        f"""
        COPY (
            SELECT {projection}
            FROM read_parquet({input_sql})
        ) TO {output_sql}
        (FORMAT PARQUET, COMPRESSION ZSTD,
         ROW_GROUP_SIZE {int(row_group_size)},
         PARTITION_BY ({parent}))
        """
    )


@dataclass(frozen=True)
class LookupCatalog:
    """Stable URI and SQL contract for published H3 admin lookup artifacts."""

    root: str

    def lookup_uri(self, resolution: int) -> str:
        return f"{self.root.rstrip('/')}/r{resolution}.parquet"

    def exploded_uri(self, resolution: int) -> str:
        """Optional analytical coverage artifact; OBF does not require it."""
        return f"{self.root.rstrip('/')}/r{resolution}_exploded.parquet"

    def partitioned_lookup_root(self, resolution: int) -> str:
        """Hive directory root for a partitioned lookup (no partition filter)."""
        return f"{self.root.rstrip('/')}/r{resolution}"

    def partitioned_lookup_uri(
        self,
        resolution: int,
        partition_resolution: int,
        parent: str | None = None,
    ) -> str:
        root = self.partitioned_lookup_root(resolution)
        value = parent or "*"
        return f"{root}/h3_r{partition_resolution}={value}/*.parquet"

    @property
    def version_uri(self) -> str:
        return f"{self.root.rstrip('/')}/.version"

    def country_cells_sql(
        self,
        adm0_iso: str,
        resolution: int = 3,
        *,
        coverage_adm0_field: str = "adm0_iso",
        has_coverage: bool | None = None,
    ) -> str:
        """SQL for distinct hexes touching an ADM0.

        Lookup files only expose a nested ``coverage`` list when they were
        written with ``include_coverage=True`` (h3geo default for r<8). When
        that column is absent — r8+, or an explicit ``include_coverage=False``
        write — pass ``has_coverage=False`` (or rely on the r>=8 default) so
        the query uses the exploded artifact instead.
        """
        iso3 = adm0_iso.upper()
        if not coverage_adm0_field.replace("_", "").isalnum():
            raise ValueError("unsafe coverage ADM0 field")

        use_coverage = (resolution < 8) if has_coverage is None else has_coverage
        if not use_coverage:
            uri = sql_quote(self.exploded_uri(resolution))
            return (
                "SELECT DISTINCT hex_id "
                f"FROM read_parquet({uri}) "
                f"WHERE {coverage_adm0_field}={sql_quote(iso3)} "
                "AND pct > 0 ORDER BY hex_id"
            )

        return (
            "SELECT DISTINCT hex_id "
            f"FROM read_parquet({sql_quote(self.lookup_uri(resolution))}) "
            f"WHERE best_adm0={sql_quote(iso3)} "
            "OR list_contains(list_transform(coverage, x -> "
            f"x.{coverage_adm0_field}), {sql_quote(iso3)}) ORDER BY hex_id"
        )

    def assignment_sql(
        self,
        resolution: int = 7,
        *,
        partition_resolution: int | None = None,
        parent: str | None = None,
    ) -> str:
        uri = (
            self.lookup_uri(resolution)
            if partition_resolution is None
            else self.partitioned_lookup_uri(resolution, partition_resolution, parent)
        )
        return f"SELECT * FROM read_parquet({sql_quote(uri)}, hive_partitioning=true)"
