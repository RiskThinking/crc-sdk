"""Arrow-oriented batch H3 polyfill for vector geometries."""

from __future__ import annotations

from enum import Enum
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


class VectorContainment(str, Enum):
    """Containment modes for batch polygon→cell expansion.

    Both modes select every cell whose boundary intersects the polygon,
    including the single cell enclosing a polygon smaller than one cell (H3
    ``overlap`` containment). ``COVERS`` is required for CGAZ admin-lookup
    parity; ``OVERLAP`` is kept as an alias for callers that already pass it.
    """

    COVERS = "covers"
    OVERLAP = "overlap"


_EMPTY_CELLS = np.empty(0, dtype=np.uint64)


def polyfill_wkb(
    wkb_values: Any,
    resolution: int,
    *,
    containment: VectorContainment | str = VectorContainment.COVERS,
    flatten: bool = False,
) -> Any:
    """Polyfill WKB geometries into per-row H3 cell lists.

    Returns a ``list<uint64>`` Arrow array with one sorted, de-duplicated cell
    list per input row (null WKB yields a null row, empty geometry an empty
    list), or with ``flatten`` the concatenated ``uint64`` cells. Polygons use
    overlap containment; points map to their cell and lines to the grid path
    between consecutive vertex cells.

    Requires ``pip install crc-sdk[geometry]``.
    """
    if not 0 <= resolution <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    VectorContainment(containment)
    array = _as_binary_array(wkb_values)

    try:
        import shapely  # type: ignore[import-untyped]
        from h3.api import basic_int  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "Batch polyfill requires `pip install crc-sdk[geometry]`"
        ) from error

    geometries = shapely.force_2d(
        shapely.from_wkb(np.asarray(array.to_numpy(zero_copy_only=False)))
    )
    rows = [
        None
        if geometry is None
        else _unique_cells(_geometry_cells(geometry, resolution, basic_int, shapely))
        for geometry in geometries
    ]
    values = np.concatenate([row for row in rows if row is not None] or [_EMPTY_CELLS])
    flat = pa.array(values, type=pa.uint64())
    if flatten:
        return flat
    lengths = np.array([0 if row is None else len(row) for row in rows], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]).astype(np.int32)
    mask = pa.array([row is None for row in rows], type=pa.bool_())
    return pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()), flat, mask=mask if mask.true_count else None
    )


def _unique_cells(cells: list[int]) -> np.ndarray:
    if not cells:
        return _EMPTY_CELLS
    return np.unique(np.asarray(cells, dtype=np.uint64))


def _geometry_cells(geometry: Any, resolution: int, h3: Any, shapely: Any) -> list[int]:
    if geometry.is_empty:
        return []
    kind = geometry.geom_type
    if kind in ("Polygon", "MultiPolygon"):
        shape = h3.geo_to_h3shape(geometry.__geo_interface__)
        return list(
            h3.h3shape_to_cells_experimental(shape, resolution, contain="overlap")
        )
    if kind in ("Point", "MultiPoint"):
        coords = shapely.get_coordinates(geometry)
        return [h3.latlng_to_cell(lat, lng, resolution) for lng, lat in coords]
    if kind in ("LineString", "LinearRing", "MultiLineString"):
        cells: list[int] = []
        for line in shapely.get_parts(geometry):
            vertex_cells = [
                h3.latlng_to_cell(lat, lng, resolution)
                for lng, lat in shapely.get_coordinates(line)
            ]
            for start, end in pairwise(vertex_cells):
                cells.extend(h3.grid_path_cells(start, end))
        return cells
    if kind == "GeometryCollection":
        return [
            cell
            for part in shapely.get_parts(geometry)
            for cell in _geometry_cells(part, resolution, h3, shapely)
        ]
    raise TypeError(f"unsupported geometry type {kind!r}")


_AUTO_BATCH_THRESHOLD_ROWS = 500_000
_AUTO_BATCH_ROWS = 200_000


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

    ``batch_rows=None`` (default) auto-detects: a cheap ``COUNT(*)`` decides
    between one-shot polyfill (fastest when the Arrow WKB+cell payload fits
    in memory — the common notebook-scale case) and chunked ingest once the
    input is large enough (CGAZ-global scale) that one-shot risks not
    fitting. Pass an explicit positive ``batch_rows`` to force a specific
    chunk size regardless of input size, or ``0``/a negative value to force
    one-shot regardless of size. When batching, input is streamed via record
    batches so not all WKB must live in Arrow at once.
    """
    mode = VectorContainment(containment)
    con.execute(f"DROP TABLE IF EXISTS {candidates_table}")
    con.execute(f"CREATE TABLE {candidates_table} ({hex_col} VARCHAR, {id_col} BIGINT)")

    resolved_batch_rows = batch_rows
    if resolved_batch_rows is None:
        row_count = con.execute(
            f"SELECT COUNT(*) FROM ({polygon_sql}) AS _crc_polygon_count"
        ).fetchone()[0]
        if row_count > _AUTO_BATCH_THRESHOLD_ROWS:
            resolved_batch_rows = _AUTO_BATCH_ROWS

    if resolved_batch_rows is None or resolved_batch_rows <= 0:
        return _expand_one_shot(
            con,
            polygon_sql,
            resolution,
            mode=mode,
            id_col=id_col,
            wkb_col=wkb_col,
            hex_col=hex_col,
            candidates_table=candidates_table,
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
        batch_rows=int(resolved_batch_rows),
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
        [table.column(id_col), _to_pyarrow_array(cells)],
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
) -> int:
    result = con.execute(polygon_sql)
    reader = _arrow_reader(result)
    batch_index = 0
    for batch in reader:
        if batch.num_rows == 0:
            continue
        for start in range(0, batch.num_rows, batch_rows):
            chunk = batch.slice(start, min(batch_rows, batch.num_rows - start))
            cells = polyfill_wkb(
                chunk.column(wkb_col),
                resolution,
                containment=mode,
                flatten=False,
            )
            batch_table = pa.Table.from_arrays(
                [chunk.column(id_col), _to_pyarrow_array(cells)],
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


def _arrow_reader(result: Any) -> Any:
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
    if hasattr(wkb_values, "num_chunks"):
        return (
            wkb_values.combine_chunks()
            if wkb_values.num_chunks > 1
            else wkb_values.chunk(0)
        )
    if isinstance(wkb_values, pa.Array):
        return wkb_values
    return pa.array(list(wkb_values), type=pa.binary())


def _to_pyarrow_array(values: Any) -> Any:
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
