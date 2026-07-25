"""Arrow-oriented batch H3 polyfill for vector geometries."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


class VectorContainment(str, Enum):
    """Containment modes for batch polygon→cell expansion.

    Both modes use h3ronpy's Arrow vector path (no per-geometry Python loop).
    ``COVERS`` is required for CGAZ admin-lookup parity. ``OVERLAP`` maps to
    h3ronpy ``IntersectsBoundary``, the closest batch analogue of H3 overlap.
    """

    COVERS = "covers"
    OVERLAP = "overlap"


_H3RONPY_MODE = {
    VectorContainment.COVERS: "Covers",
    VectorContainment.OVERLAP: "IntersectsBoundary",
}


def polyfill_wkb(
    wkb_values: Any,
    resolution: int,
    *,
    containment: VectorContainment | str = VectorContainment.COVERS,
    flatten: bool = False,
) -> Any:
    """Polyfill WKB polygons into per-row H3 cell lists (Arrow list-array).

    Requires ``pip install crc-sdk[geometry-vector]`` (h3ronpy + pyarrow).
    """
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    mode = VectorContainment(containment)
    array = _as_binary_array(wkb_values)

    try:
        import h3ronpy.vector
    except ImportError as error:
        raise ImportError(
            "Batch polyfill requires `pip install crc-sdk[geometry-vector]`"
        ) from error

    containment_mode = getattr(h3ronpy.vector.ContainmentMode, _H3RONPY_MODE[mode])
    try:
        return h3ronpy.vector.wkb_to_cells(
            array,
            resolution,
            containment_mode=containment_mode,
            flatten=flatten,
        )
    except TypeError:
        return h3ronpy.vector.wkb_to_cells(
            array,
            resolution,
            containment_mode=containment_mode,
        )


def expand_polygon_candidates(
    con: DuckDBPyConnection,
    polygon_sql: str,
    resolution: int,
    *,
    containment: VectorContainment | str = VectorContainment.COVERS,
    id_col: str = "poly_rid",
    wkb_col: str = "wkb",
    hex_col: str = "hex_id",
    candidates_table: str = "candidates",
    batch_rows: int | None = None,
) -> int:
    """Expand polygon WKB from DuckDB into string H3 cell candidates.

    ``batch_rows=None`` (default) uses one-shot polyfill + a single DuckDB
    ingest — fastest when the Arrow WKB+cell payload fits in memory (CGAZ
    global at modest resolutions). Pass a positive ``batch_rows`` to polyfill
    and insert in chunks; when batching, input is streamed via record batches
    so not all WKB must live in Arrow at once.
    """
    try:
        import pyarrow as pa
    except ImportError as error:
        raise ImportError(
            "Batch polyfill requires `pip install crc-sdk[geometry-vector]`"
        ) from error

    mode = VectorContainment(containment)
    con.execute(f"DROP TABLE IF EXISTS {candidates_table}")
    con.execute(f"CREATE TABLE {candidates_table} ({hex_col} VARCHAR, {id_col} BIGINT)")

    if batch_rows is None or batch_rows <= 0:
        return _expand_one_shot(
            con,
            polygon_sql,
            resolution,
            mode=mode,
            id_col=id_col,
            wkb_col=wkb_col,
            hex_col=hex_col,
            candidates_table=candidates_table,
            pa=pa,
        )

    return _expand_batched(
        con,
        polygon_sql,
        resolution,
        mode=mode,
        id_col=id_col,
        wkb_col=wkb_col,
        hex_col=hex_col,
        candidates_table=candidates_table,
        batch_rows=int(batch_rows),
        pa=pa,
    )


def _expand_one_shot(
    con: DuckDBPyConnection,
    polygon_sql: str,
    resolution: int,
    *,
    mode: VectorContainment,
    id_col: str,
    wkb_col: str,
    hex_col: str,
    candidates_table: str,
    pa: Any,
) -> int:
    table = _fetch_arrow_table(con, polygon_sql)
    if table.num_rows == 0:
        return 0
    cells = polyfill_wkb(
        table.column(wkb_col),
        resolution,
        containment=mode,
        flatten=False,
    )
    batch_table = pa.Table.from_arrays(
        [table.column(id_col), _to_pyarrow_array(cells, pa)],
        names=["_poly_id", "_cells"],
    )
    register_name = "_crc_polyfill_all"
    con.register(register_name, batch_table)
    try:
        con.execute(
            f"""
            INSERT INTO {candidates_table}
            SELECT
                h3_h3_to_string(unnest(_cells)) AS {hex_col},
                _poly_id AS {id_col}
            FROM {register_name}
            """
        )
    finally:
        con.unregister(register_name)
    return int(con.execute(f"SELECT COUNT(*) FROM {candidates_table}").fetchone()[0])


def _expand_batched(
    con: DuckDBPyConnection,
    polygon_sql: str,
    resolution: int,
    *,
    mode: VectorContainment,
    id_col: str,
    wkb_col: str,
    hex_col: str,
    candidates_table: str,
    batch_rows: int,
    pa: Any,
) -> int:
    result = con.execute(polygon_sql)
    reader = _arrow_reader(result, pa)
    batch_index = 0
    for batch in reader:
        if batch.num_rows == 0:
            continue
        # Chunk large record batches to the requested size.
        for start in range(0, batch.num_rows, batch_rows):
            chunk = batch.slice(start, min(batch_rows, batch.num_rows - start))
            cells = polyfill_wkb(
                chunk.column(wkb_col),
                resolution,
                containment=mode,
                flatten=False,
            )
            batch_table = pa.Table.from_arrays(
                [chunk.column(id_col), _to_pyarrow_array(cells, pa)],
                names=["_poly_id", "_cells"],
            )
            register_name = f"_crc_polyfill_batch_{batch_index}"
            batch_index += 1
            con.register(register_name, batch_table)
            try:
                con.execute(
                    f"""
                    INSERT INTO {candidates_table}
                    SELECT
                        h3_h3_to_string(unnest(_cells)) AS {hex_col},
                        _poly_id AS {id_col}
                    FROM {register_name}
                    """
                )
            finally:
                con.unregister(register_name)

    return int(con.execute(f"SELECT COUNT(*) FROM {candidates_table}").fetchone()[0])


def _fetch_arrow_table(con: DuckDBPyConnection, polygon_sql: str) -> Any:
    result = con.execute(polygon_sql)
    if hasattr(result, "to_arrow_table"):
        return result.to_arrow_table()
    if hasattr(result, "fetch_arrow_table"):
        return result.fetch_arrow_table()
    arrow_obj = result.arrow()
    return arrow_obj.read_all() if hasattr(arrow_obj, "read_all") else arrow_obj


def _arrow_reader(result: Any, pa: Any) -> Any:
    if hasattr(result, "to_arrow_reader"):
        try:
            reader = result.to_arrow_reader()
            if reader is not None:
                return reader
        except Exception:
            pass
    if hasattr(result, "fetch_record_batch"):
        try:
            reader = result.fetch_record_batch()
            if reader is not None:
                return reader
        except Exception:
            pass
    if hasattr(result, "arrow"):
        arrow_obj = result.arrow()
        if hasattr(arrow_obj, "__iter__") and not hasattr(arrow_obj, "num_rows"):
            return arrow_obj
        table = arrow_obj.read_all() if hasattr(arrow_obj, "read_all") else arrow_obj
        return table.to_batches()
    table = (
        result.to_arrow_table()
        if hasattr(result, "to_arrow_table")
        else result.fetch_arrow_table()
    )
    return table.to_batches()


def _as_binary_array(wkb_values: Any) -> Any:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise ImportError(
            "Batch polyfill requires `pip install crc-sdk[geometry-vector]`"
        ) from error

    if hasattr(wkb_values, "num_chunks"):
        return (
            wkb_values.combine_chunks()
            if wkb_values.num_chunks > 1
            else wkb_values.chunk(0)
        )
    if isinstance(wkb_values, pa.Array):
        return wkb_values
    return pa.array(list(wkb_values), type=pa.binary())


def _to_pyarrow_array(values: Any, pa: Any) -> Any:
    if isinstance(values, pa.Array):
        return values
    # Prefer Arrow C data / capsule bridges (e.g. arro3) over to_pylist().
    try:
        converted = pa.array(values)
        if isinstance(converted, pa.Array):
            return converted
    except (TypeError, ValueError):
        pass
    if hasattr(values, "to_pylist"):
        return pa.array(values.to_pylist())
    return pa.array(list(values))
