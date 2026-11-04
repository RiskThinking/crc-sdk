"""Compact per-year annual-extreme files shared by the ERA5 and CMIP6 providers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from crc_sdk.connectors.blocks import (
    BlockExtremaCurveSource,
    BlockSpec,
    BlockStatistic,
    _strip_row_count,
)
from crc_sdk.connectors.duckdb.netcdf import NetCDFBlockReader, NetCDFRaster
from crc_sdk.connectors.duckdb.zarr import RasterMetadata

EXTREME_VARIABLE = "extreme"


def require_h5netcdf(label: str) -> None:
    try:
        import h5netcdf  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as error:
        raise ImportError(
            f"{label} caching requires `pip install crc-sdk[netcdf]`"
        ) from error


def write_annual_extreme(
    reader: Any,
    year: int,
    destination: str | Path,
    *,
    strip_bytes: int,
    attributes: Mapping[str, Any],
) -> Path:
    """Reduce one year from a block reader and persist it as a NetCDF file."""
    import h5netcdf

    width = reader.lon.size
    strip_rows = _strip_row_count(
        strip_bytes, width, 8, 1, leading_axis=reader.leading_axis_samples
    )
    parts = [
        reader.reduce_block(
            year,
            slice(start, min(start + strip_rows, reader.lat.size)),
            slice(0, width),
        )
        for start in range(0, reader.lat.size, strip_rows)
    ]
    extreme = np.concatenate(parts, axis=0)

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.nc")
    with h5netcdf.File(temporary, "w") as output:
        output.dimensions = {"time": 1, "lat": reader.lat.size, "lon": width}
        output.create_variable("lat", ("lat",), dtype=np.float64, data=reader.lat)
        output.create_variable("lon", ("lon",), dtype=np.float64, data=reader.lon)
        variable = output.create_variable(
            EXTREME_VARIABLE,
            ("time", "lat", "lon"),
            dtype=np.float32,
            data=extreme[np.newaxis].astype(np.float32),
        )
        for key, value in attributes.items():
            variable.attrs[key] = value
    temporary.replace(target)
    return target


def open_annual_extremes(
    resources: Mapping[int, str | Path],
    metadata: RasterMetadata,
    *,
    tail: Literal["upper", "lower"],
    strip_bytes: int,
    minimum_blocks: int,
    info: Any,
) -> BlockExtremaCurveSource:
    """Open cached annual-extreme files as one block-extrema curve source."""
    if not resources:
        raise ValueError("at least one year is required")
    rasters: dict[int, NetCDFRaster] = {}
    try:
        for year, path in resources.items():
            rasters[int(year)] = NetCDFRaster.open(path, variable=EXTREME_VARIABLE)
    except Exception:
        for raster in rasters.values():
            raster.close()
        raise
    # A cached one-step file reduces to itself under any statistic. (Not the
    # recipe's statistic: a rolling k-day statistic needs a daily series,
    # which a cached annual-extreme file no longer has.)
    identity = BlockStatistic.minimum() if tail == "lower" else BlockStatistic.maximum()
    return BlockExtremaCurveSource(
        reader=NetCDFBlockReader(rasters, spec=BlockSpec(statistic=identity)),
        metadata=metadata,
        tail=tail,
        strip_bytes=strip_bytes,
        minimum_blocks=minimum_blocks,
        info=info,
    )
