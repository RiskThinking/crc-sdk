"""Scoped native grid decoding shared by climate source recipes."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.error import HTTPError

import numpy as np
from crc_framework import QuantileCurve
from h3.api import basic_int as h3  # type: ignore[import-untyped]
from shapely.geometry import Polygon, box  # type: ignore[import-untyped]

from ._dataset_schema import ResourceIdentity
from ._declarations import Period, Scenario, Variable
from ._graph import ExecutionScope
from ._inspection import Bounds
from ._schema import DatasetRow


def _parts(area: Bounds | None) -> tuple[Any, ...]:
    if area is None:
        return ()
    if area.west <= area.east:
        return (box(area.west, area.south, area.east, area.north),)
    return (
        box(area.west, area.south, 180, area.north),
        box(-180, area.south, area.east, area.north),
    )


@dataclass(slots=True)
class RasterReadCounts:
    requested: int = 0
    skipped: int = 0


def iter_raster_quantile_rows(
    paths: Mapping[float, Path],
    *,
    variable: Variable,
    scenario: Scenario | None = None,
    period: Period | None = None,
    area: Bounds | None = None,
    support_prefix: str | None = None,
    max_pixels: int = 65536,
    counts: RasterReadCounts | None = None,
) -> Iterator[DatasetRow]:
    """Stream source pixels and their actual valid q knots in bounded windows."""
    import hashlib
    import json
    from contextlib import ExitStack

    try:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.warp import transform as project  # type: ignore[import-untyped]
        from rasterio.warp import transform_bounds
        from rasterio.windows import Window  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "Raster inputs require `pip install crc-sdk[raster]`"
        ) from error
    if len(paths) < 2:
        raise ValueError("quantile rasters require at least two probability values")
    probabilities = sorted(paths)
    if any(
        isinstance(q, bool) or not math.isfinite(q) or not 0 <= q <= 1
        for q in probabilities
    ):
        raise ValueError("raster probabilities must be finite and within [0, 1]")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    observed = RasterReadCounts() if counts is None else counts
    pieces = _parts(area)
    seen: set[tuple[int, int]] = set()
    with ExitStack() as stack:
        rasters = [stack.enter_context(rasterio.open(paths[q])) for q in probabilities]
        first = rasters[0]
        if first.crs is None or any(r.count != 1 for r in rasters):
            raise ValueError("quantile rasters require one band and an explicit CRS")
        layout = (first.width, first.height, first.transform, first.crs)
        if any((r.width, r.height, r.transform, r.crs) != layout for r in rasters[1:]):
            raise ValueError("quantile rasters must use the same source grid and CRS")
        tags = first.tags()
        row_offset, column_offset = (
            int(tags.get("crc_source_row", 0)),
            int(tags.get("crc_source_column", 0)),
        )
        source_transform = (
            first.transform
            if "crc_source_transform" not in tags
            else rasterio.Affine(*json.loads(tags["crc_source_transform"]))
        )
        ranges = []
        if not pieces:
            ranges.append((0, first.height, 0, first.width))
        else:
            inverse = ~first.transform
            for piece in pieces:
                west, south, east, north = transform_bounds(
                    "EPSG:4326", first.crs, *piece.bounds, densify_pts=21
                )
                pixels = [
                    inverse @ p
                    for p in (
                        (west, south),
                        (west, north),
                        (east, south),
                        (east, north),
                    )
                ]
                r0, r1 = (
                    max(0, math.floor(min(p[1] for p in pixels))),
                    min(first.height, math.ceil(max(p[1] for p in pixels))),
                )
                c0, c1 = (
                    max(0, math.floor(min(p[0] for p in pixels))),
                    min(first.width, math.ceil(max(p[0] for p in pixels))),
                )
                if r1 > r0 and c1 > c0:
                    ranges.append((r0, r1, c0, c1))
        for r0, r1, c0, c1 in ranges:
            columns = min(c1 - c0, max_pixels)
            rows = max(1, max_pixels // columns)
            for r in range(r0, r1, rows):
                for c in range(c0, c1, columns):
                    window = Window(c, r, min(columns, c1 - c), min(rows, r1 - r))
                    arrays = [
                        source.read(1, window=window, masked=True) for source in rasters
                    ]
                    for dy in range(int(window.height)):
                        for dx in range(int(window.width)):
                            local_row, local_column = r + dy, c + dx
                            if (local_row, local_column) in seen:
                                continue
                            # Only split antimeridian windows can overlap.
                            if len(ranges) > 1:
                                seen.add((local_row, local_column))
                            row, column = (
                                local_row + row_offset,
                                local_column + column_offset,
                            )
                            corners = [
                                source_transform @ p
                                for p in (
                                    (column, row),
                                    (column + 1, row),
                                    (column + 1, row + 1),
                                    (column, row + 1),
                                )
                            ]
                            x, y = project(
                                first.crs,
                                "EPSG:4326",
                                [p[0] for p in corners],
                                [p[1] for p in corners],
                            )
                            geometry = Polygon(zip(x, y, strict=True))
                            if pieces and not any(
                                p.intersects(geometry) for p in pieces
                            ):
                                continue
                            observed.requested += 1
                            evidence = [
                                (q, float(a[dy, dx]))
                                for q, a in zip(probabilities, arrays, strict=True)
                                if not np.ma.is_masked(a[dy, dx])
                                and math.isfinite(float(a[dy, dx]))
                            ]
                            if len(evidence) < 2:
                                observed.skipped += 1
                                continue
                            curve = QuantileCurve(
                                [q for q, _ in evidence], [v for _, v in evidence]
                            )
                            centre = source_transform @ (column + 0.5, row + 0.5)
                            lon, lat = project(
                                first.crs, "EPSG:4326", [centre[0]], [centre[1]]
                            )
                            support = (
                                f"{support_prefix}:{row}:{column}"
                                if support_prefix is not None
                                else "user:raster:"
                                + hashlib.sha256(geometry.wkb).hexdigest()
                            )
                            yield DatasetRow(
                                variable_id=variable.id,
                                scenario_id=None if scenario is None else scenario.id,
                                period_id=None if period is None else period.id,
                                support_id=support,
                                cell_id=int(
                                    h3.latlng_to_cell(float(lat[0]), float(lon[0]), 9)
                                ),
                                geometry=geometry.wkb,
                                curve=curve,
                            )


def raster_quantile_rows(
    paths: Mapping[float, Path],
    *,
    variable: Variable,
    scenario: Scenario | None = None,
    period: Period | None = None,
    area: Bounds | None = None,
    support_prefix: str,
) -> tuple[tuple[DatasetRow, ...], int]:
    counts = RasterReadCounts()
    rows = tuple(
        iter_raster_quantile_rows(
            paths,
            variable=variable,
            scenario=scenario,
            period=period,
            area=area,
            support_prefix=support_prefix,
            counts=counts,
        )
    )
    return rows, counts.skipped


@dataclass(slots=True)
class ZarrSnapshot:
    group: Any
    _resource_map: dict[str, ResourceIdentity]


def open_zarr_snapshot(
    scope: ExecutionScope,
    uri: str,
    *,
    license: str | None = None,
    attribution: str | None = None,
) -> ZarrSnapshot:
    """Read published Zarr 2/3 objects through verified, scope-owned snapshots."""
    try:
        import zarr
        from zarr.abc.store import (
            OffsetByteRequest,
            RangeByteRequest,
            Store,
            SuffixByteRequest,
        )
    except ImportError as error:
        raise ImportError(
            "Climate Zarr sources require `pip install crc-sdk[zarr]`"
        ) from error
    from ._source_io import fetch

    resources: dict[str, ResourceIdentity] = {}
    last_io_error: FileNotFoundError | None = None
    format_version: Literal[2, 3] = 2
    failures: list[OSError] = []
    for version, name in ((2, ".zmetadata"), (3, "zarr.json")):
        try:
            _, identity = fetch(
                scope,
                uri.rstrip("/") + "/" + name,
                license=license,
                attribution=attribution,
            )
        except FileNotFoundError as error:
            failures.append(error)
            continue
        except HTTPError as error:
            if error.code != 404:
                raise
            failures.append(error)
            continue
        resources[identity.uri] = identity
        format_version = 2 if version == 2 else 3
        break
    else:
        raise failures[0]

    class SnapshotStore(Store):
        supports_writes = False
        supports_deletes = False
        supports_listing = False

        def __init__(self) -> None:
            super().__init__(read_only=True)

        def __eq__(self, other: object) -> bool:
            return self is other

        async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
            nonlocal last_io_error
            path = PurePosixPath(key)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("invalid Zarr resource key")
            try:
                local, resource = fetch(
                    scope,
                    uri.rstrip("/") + "/" + key,
                    license=license,
                    attribution=attribution,
                )
            except FileNotFoundError as error:
                last_io_error = error
                raise
            except HTTPError as error:
                if error.code == 404:
                    identity = getattr(error, "resource", None)
                    if isinstance(identity, ResourceIdentity):
                        missing_uri = identity.uri + "#crc-http-status=404"
                        resources[missing_uri] = replace(identity, uri=missing_uri)
                    return None
                raise
            resources[resource.uri] = resource
            value = local.read_bytes()
            if isinstance(byte_range, RangeByteRequest):
                value = value[byte_range.start : byte_range.end]
            elif isinstance(byte_range, OffsetByteRequest):
                value = value[byte_range.offset :]
            elif isinstance(byte_range, SuffixByteRequest):
                value = value[-byte_range.suffix :]
            return prototype.buffer.from_bytes(value)

        async def exists(self, key: str) -> bool:
            from zarr.core.buffer import default_buffer_prototype

            return await self.get(key, default_buffer_prototype()) is not None

        async def get_partial_values(
            self, prototype: Any, key_ranges: Iterable[tuple[str, Any]]
        ) -> list[Any]:
            return [await self.get(k, prototype, request) for k, request in key_ranges]

        async def set(self, key: str, value: Any) -> None:
            raise PermissionError("source snapshots are read-only")

        async def delete(self, key: str) -> None:
            raise PermissionError("source snapshots are read-only")

        async def list(self) -> AsyncIterator[str]:
            raise NotImplementedError(
                "climate sources require consolidated Zarr metadata"
            )
            yield ""

        async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
            raise NotImplementedError(
                "climate sources require consolidated Zarr metadata"
            )
            yield ""

        async def list_dir(self, prefix: str) -> AsyncIterator[str]:
            raise NotImplementedError(
                "climate sources require consolidated Zarr metadata"
            )
            yield ""

    store = SnapshotStore()
    scope.callback(store.close)
    try:
        group = zarr.open_group(
            store=store, mode="r", use_consolidated=True, zarr_format=format_version
        )
    except (FileNotFoundError, ValueError):
        if last_io_error is not None:
            raise last_io_error
        raise
    # zarr metadata and chunks are acquired lazily. The same owned list is
    # refreshed when consumers request a current source identity.
    return ZarrSnapshot(group, resources)


def zarr_resources(snapshot: ZarrSnapshot) -> tuple[ResourceIdentity, ...]:
    return tuple(snapshot._resource_map.values())


def raster_window_snapshot(
    scope: ExecutionScope,
    uri: str,
    *,
    area: Bounds,
    license: str | None = None,
    attribution: str | None = None,
) -> tuple[Path, ResourceIdentity]:
    """Own only the requested raster pixels; checksum refers to this view."""
    from ._source_io import fetch_custom

    def acquire(destination: Path) -> None:
        import json

        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import Window, from_bounds

        before = _validator(uri)
        options = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
        if before.get("etag"):
            options["GDAL_HTTP_HEADERS"] = "If-Match: " + before["etag"]
        with rasterio.Env(**options), rasterio.open(uri) as source:
            if source.crs is None or source.count != 1:
                raise ValueError("source raster requires one band and an explicit CRS")
            windows = []
            for piece in _parts(area):
                bounds = transform_bounds("EPSG:4326", source.crs, *piece.bounds)
                candidate = from_bounds(*bounds, transform=source.transform)
                r0, c0 = (
                    max(0, math.floor(candidate.row_off)),
                    max(0, math.floor(candidate.col_off)),
                )
                r1, c1 = (
                    min(source.height, math.ceil(candidate.row_off + candidate.height)),
                    min(source.width, math.ceil(candidate.col_off + candidate.width)),
                )
                if r1 > r0 and c1 > c0:
                    windows.append(Window(c0, r0, c1 - c0, r1 - r0))
            # A crossing request is represented by its two bounded pixel views.
            # JRC products use -180..180; no unrelated longitude slab is read.
            contents: dict[str, bytes] = {}
            for index, window in enumerate(windows):
                data = source.read(1, window=window, masked=True)
                profile = source.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=int(window.height),
                    width=int(window.width),
                    count=1,
                    transform=source.window_transform(window),
                    tiled=False,
                )
                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)
                with rasterio.io.MemoryFile() as memory:
                    with memory.open(**profile) as output:
                        output.write(
                            data.filled(
                                source.nodata if source.nodata is not None else np.nan
                            ),
                            1,
                        )
                        output.write_mask(~np.ma.getmaskarray(data))
                        output.update_tags(
                            crc_source_row=int(window.row_off),
                            crc_source_column=int(window.col_off),
                            crc_source_transform=json.dumps(list(source.transform)[:6]),
                        )
                    contents[f"view-{index}.tif"] = memory.read()
        if before != _validator(uri):
            raise OSError("source raster changed while acquiring the selected window")
        contents["metadata.json"] = _json_bytes(
            {
                "source_uri": uri,
                "validator": before,
                "area": [area.west, area.south, area.east, area.north],
            }
        )
        _bundle(destination, contents)

    view_uri = (
        uri
        + "#crc-window="
        + ",".join(
            format(v, ".17g") for v in (area.west, area.south, area.east, area.north)
        )
    )
    return fetch_custom(
        scope, view_uri, acquire, license=license, attribution=attribution
    )


def raster_views(scope: ExecutionScope, bundle: Path) -> tuple[Path, ...]:
    import zipfile

    result = []
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            if not name.endswith(".tif"):
                continue
            path = scope.directory / (
                "raster-view-" + __import__("uuid").uuid4().hex + ".tif"
            )
            path.write_bytes(archive.read(name))
            result.append(path)
    return tuple(result)


def _json_bytes(value: object) -> bytes:
    import json

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _bundle(path: Path, entries: Mapping[str, bytes]) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, entries[name])


def _validator(uri: str) -> dict[str, str]:
    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme in {"", "file"}:
        path = Path(urllib.request.url2pathname(parsed.path))
        stat = path.stat()
        return {"size": str(stat.st_size), "mtime_ns": str(stat.st_mtime_ns)}
    with urllib.request.urlopen(
        urllib.request.Request(uri, method="HEAD"), timeout=60
    ) as response:
        result = {"size": response.headers.get("Content-Length", "")}
        for source, target in (("ETag", "etag"), ("Last-Modified", "last_modified")):
            if response.headers.get(source):
                result[target] = response.headers[source]
    if not result["size"] or not (result.get("etag") or result.get("last_modified")):
        raise OSError(
            "bounded source access requires Content-Length and an HTTP validator"
        )
    return result


class _Array:
    def __init__(
        self,
        values: np.ndarray[Any, Any],
        attrs: dict[str, Any],
        dimensions: tuple[str, ...] = (),
    ) -> None:
        self.values, self.attrs, self.dimensions = values, attrs, dimensions

    def __getitem__(self, selection: Any) -> Any:
        return self.values[selection]


class GridSnapshot:
    def __init__(
        self, arrays: dict[str, _Array], attrs: dict[str, Any], offsets: tuple[int, int]
    ) -> None:
        self.arrays, self.attrs, self.offsets = arrays, attrs, offsets
        self.source_rows: list[int] = []
        self.source_columns: list[int] = []

    def __getitem__(self, name: str) -> _Array:
        return self.arrays[name]

    def __contains__(self, name: str) -> bool:
        return name in self.arrays


class _RangeFile:
    """Seekable strict HTTP ranges; never accepts a full-file range fallback."""

    def __init__(self, uri: str, validator: dict[str, str]) -> None:
        self.uri, self.validator = uri, validator
        self.position = 0
        self.size = int(validator["size"])
        self.blocks: dict[int, bytes] = {}

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        import urllib.request

        end = self.size if size < 0 else min(self.size, self.position + size)
        pieces = []
        block_size = 1024 * 1024
        while self.position < end:
            index = self.position // block_size
            if index not in self.blocks:
                start, stop = (
                    index * block_size,
                    min(self.size, (index + 1) * block_size),
                )
                headers = {"Range": f"bytes={start}-{stop - 1}"}
                if self.validator.get("etag"):
                    headers["If-Match"] = self.validator["etag"]
                elif self.validator.get("last_modified"):
                    headers["If-Unmodified-Since"] = self.validator["last_modified"]
                with urllib.request.urlopen(
                    urllib.request.Request(self.uri, headers=headers), timeout=60
                ) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{stop - 1}/{self.size}"
                    ):
                        raise OSError(
                            "source must support exact HTTP byte ranges "
                            "for bounded NetCDF access"
                        )
                    data = response.read(stop - start + 1)
                    if len(data) != stop - start:
                        raise OSError("source returned an incomplete byte range")
                    self.blocks[index] = data
            data = self.blocks[index]
            begin = self.position - index * block_size
            count = min(len(data) - begin, end - self.position)
            pieces.append(data[begin : begin + count])
            self.position += count
        return b"".join(pieces)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def netcdf_window_snapshot(
    scope: ExecutionScope,
    uri: str,
    *,
    area: Bounds,
    variable: str,
    latitude: str = "lat",
    longitude: str = "lon",
    time: str = "time",
    resolution: float | None = None,
    license: str | None = None,
    attribution: str | None = None,
) -> tuple[GridSnapshot, ResourceIdentity]:
    """Pin a bounded native NetCDF view, including CF coordinates and attributes."""
    import io
    import json
    import urllib.parse
    import urllib.request
    import zipfile
    from contextlib import ExitStack

    from ._climate_grid import _coordinate_bounds, grid_cells
    from ._source_io import fetch_custom

    def acquire(destination: Path) -> None:
        try:
            import h5netcdf  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "NetCDF sources require `pip install crc-sdk[netcdf]`"
            ) from error
        before = _validator(uri)
        parsed = urllib.parse.urlsplit(uri)
        raw: Any = (
            Path(urllib.request.url2pathname(parsed.path))
            if parsed.scheme in {"", "file"}
            else _RangeFile(uri, before)
        )
        with ExitStack() as stack:
            dataset = stack.enter_context(h5netcdf.File(raw, "r"))
            group = _NetCDFView(dataset)
            cells = grid_cells(
                group,
                lat_name=latitude,
                lon_name=longitude,
                area=area,
                fallback_resolution=resolution,
            )
            # Separate longitude intervals avoid turning an antimeridian window
            # into a download of the entire longitude axis.
            rows = sorted({c.row for c in cells})
            columns = sorted({c.column for c in cells})
            from ._runtime import _memory_bytes

            memory_limit = _memory_bytes(scope.runtime.memory_limit)
            required = (
                len(rows) * len(columns) * len(group[time]) * 8
                if hasattr(group[time], "__len__")
                else len(rows) * len(columns) * group[time].shape[0] * 8
            )
            if memory_limit is not None and required > memory_limit:
                raise MemoryError(
                    f"Source NetCDF view needs at least {required} bytes; "
                    "select a smaller area or increase Runtime(memory_limit=...)."
                )
            entries: dict[str, bytes] = {}
            arrays: dict[str, Any] = {}
            names = [time, latitude, longitude, variable]
            for coordinate in (latitude, longitude):
                bound = group[coordinate].attrs.get("bounds")
                if bound is not None:
                    names.append(str(bound))
            for name in names:
                array = group[name]
                dims = tuple(array.dimensions)
                selection: list[Any] = [slice(None)] * len(dims)
                # h5py allows one fancy index at a time; bounded rows are contiguous.
                if latitude in dims:
                    selection[dims.index(latitude)] = (
                        slice(min(rows), max(rows) + 1) if rows else slice(0, 0)
                    )
                if longitude in dims:
                    selection[dims.index(longitude)] = columns
                data = np.asarray(array[tuple(selection)])
                output = io.BytesIO()
                np.lib.format.write_array(output, data, allow_pickle=False)
                entries[name + ".npy"] = output.getvalue()
                attrs = {str(k): _plain(v) for k, v in array.attrs.items()}
                arrays[name] = {"attrs": attrs, "dimensions": dims}
            for name, indexes in ((latitude, rows), (longitude, columns)):
                _, bounds = _coordinate_bounds(group, name, fallback=resolution)
                bound_name = "crc_" + name + "_bounds"
                output = io.BytesIO()
                np.lib.format.write_array(output, bounds[indexes], allow_pickle=False)
                entries[bound_name + ".npy"] = output.getvalue()
                arrays[bound_name] = {"attrs": {}, "dimensions": (name, "bounds")}
                arrays[name]["attrs"]["bounds"] = bound_name
            attrs = {str(k): _plain(v) for k, v in dataset.attrs.items()}
            metadata = {
                "source_uri": uri,
                "validator": before,
                "attrs": attrs,
                "arrays": arrays,
                "rows": rows,
                "columns": columns,
            }
            entries["metadata.json"] = _json_bytes(metadata)
        if before != _validator(uri):
            raise OSError("source NetCDF changed while acquiring the selected window")
        _bundle(destination, entries)

    view_uri = (
        uri
        + "#crc-variable="
        + variable
        + "&window="
        + ",".join(
            format(v, ".17g") for v in (area.west, area.south, area.east, area.north)
        )
    )
    path, identity = fetch_custom(
        scope, view_uri, acquire, license=license, attribution=attribution
    )
    with zipfile.ZipFile(path) as archive:
        document = json.loads(archive.read("metadata.json"))
        arrays = {
            name: _Array(
                np.lib.format.read_array(
                    io.BytesIO(archive.read(name + ".npy")), allow_pickle=False
                ),
                _restore(value["attrs"]),
                tuple(value["dimensions"]),
            )
            for name, value in document["arrays"].items()
        }
    rows, columns = document["rows"], document["columns"]
    group = GridSnapshot(
        arrays,
        _restore(document["attrs"]),
        (rows[0] if rows else 0, columns[0] if columns else 0),
    )
    # Source indexes are retained explicitly; disjoint selected longitudes are
    # never mistaken for a new regular native axis.
    group.source_rows = rows
    group.source_columns = columns
    return group, identity


class _NetCDFView:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __getitem__(self, name: str) -> Any:
        return self.dataset.variables[name]

    def __contains__(self, name: str) -> bool:
        return name in self.dataset.variables


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "crc_nonfinite_float": "nan"
            if math.isnan(value)
            else "inf"
            if value > 0
            else "-inf"
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _restore(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"crc_nonfinite_float"}:
            return float(value["crc_nonfinite_float"])
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return value


def netcdf_header_snapshot(
    scope: ExecutionScope, uri: str, variable: str
) -> tuple[dict[str, Any], ResourceIdentity]:
    """Inspect native scientific headers through pinned byte ranges."""
    import json
    import urllib.parse
    import urllib.request

    from ._source_io import fetch_custom

    def acquire(destination: Path) -> None:
        try:
            import h5netcdf
        except ImportError as error:
            raise ImportError(
                "NetCDF requires `pip install crc-sdk[netcdf]`"
            ) from error
        before = _validator(uri)
        parsed = urllib.parse.urlsplit(uri)
        raw: Any = (
            Path(urllib.request.url2pathname(parsed.path))
            if parsed.scheme in {"", "file"}
            else _RangeFile(uri, before)
        )
        with h5netcdf.File(raw, "r") as dataset:
            value = {
                "source_uri": uri,
                "validator": before,
                "calendar": _plain(dataset.variables["time"].attrs.get("calendar")),
                "variable": variable,
                "units": _plain(dataset.variables[variable].attrs.get("units")),
                "attrs": {str(k): _plain(v) for k, v in dataset.attrs.items()},
            }
        if before != _validator(uri):
            raise OSError("source changed while reading the native header")
        destination.write_bytes(_json_bytes(value))

    path, resource = fetch_custom(scope, uri + "#crc-header=" + variable, acquire)
    return _restore(json.loads(path.read_bytes())), resource
