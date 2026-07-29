"""Reconstruct and sample distributions from canonical hazard rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Union, cast

from crc_framework.distributions import FittedDistribution, HurdleDistribution

from crc_sdk.geometry.h3 import point_to_cell
from crc_sdk.providers.protocol import Provider
from crc_sdk.types import CurveParameters, HazardQuery

CURVE_COLUMNS = (
    "curve_kind",
    "curve_type",
    "curve_shape",
    "curve_location",
    "curve_scale",
    "curve_atom_probability",
    "curve_atom_location",
)

CurveDistribution = Union[FittedDistribution, HurdleDistribution]
SpatialMatch = Literal["exact_geometry", "h3_cell"]


@dataclass(frozen=True)
class CurveSample:
    """Samples and reconstructed distribution for one canonical curve."""

    parameters: CurveParameters
    distribution: CurveDistribution
    samples: Any


@dataclass(frozen=True)
class HazardPointSample:
    """A uniquely resolved canonical hazard curve sampled at one point."""

    hazard_name: str
    horizon: int
    pathway: str
    cell_index: int
    source_id: str
    longitude: float
    latitude: float
    spatial_match: SpatialMatch
    value_unit: str
    value_semantics: str
    parameters: CurveParameters
    distribution: CurveDistribution
    samples: Any


def _python_value(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def curve_parameters_from_row(row: Mapping[str, Any]) -> CurveParameters:
    """Reconstruct validated curve parameters from one canonical row mapping."""
    try:
        values = {name: _python_value(row[name]) for name in CURVE_COLUMNS}
    except KeyError as error:
        raise ValueError(
            f"canonical row is missing curve column {error.args[0]!r}"
        ) from error
    return CurveParameters.model_validate(values)


def _row_at(value: Any, row_index: int) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        if row_index != 0:
            raise IndexError("row_index must be 0 when sampling a row mapping")
        return value
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise TypeError("row_index must be an integer")
    if row_index < 0:
        raise IndexError("row_index must not be negative")
    if not hasattr(value, "num_rows") or not hasattr(value, "slice"):
        raise TypeError("hazard row must be a mapping or Arrow table-like value")
    if row_index >= value.num_rows:
        raise IndexError(
            f"row_index {row_index} is outside a table with {value.num_rows} rows"
        )
    rows = value.slice(row_index, 1).to_pylist()
    return cast(Mapping[str, Any], rows[0])


def distribution_from_hazard_row(
    value: Any, *, row_index: int = 0
) -> CurveDistribution:
    """Reconstruct one framework distribution from a row or Arrow table."""
    return curve_parameters_from_row(_row_at(value, row_index)).to_distribution()


def sample_hazard_row(
    value: Any,
    *,
    row_index: int = 0,
    size: int = 10_000,
    seed: int | None = None,
) -> CurveSample:
    """Sample one canonical row or one indexed row of an Arrow table."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 1:
        raise ValueError("size must be positive")
    parameters = curve_parameters_from_row(_row_at(value, row_index))
    distribution = parameters.to_distribution()
    return CurveSample(
        parameters=parameters,
        distribution=distribution,
        samples=distribution.sample(size, seed=seed),
    )


def _validate_point(longitude: float, latitude: float) -> None:
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be finite and between -180 and 180")
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be finite and between -90 and 90")


def _point_candidates(
    table: Any, longitude: float, latitude: float
) -> list[tuple[Mapping[str, Any], SpatialMatch]]:
    if not hasattr(table, "to_pylist"):
        raise TypeError("provider.read() must return an Arrow table-like value")
    rows = table.to_pylist()
    if not any(row["source_geometry"] is not None for row in rows):
        return [(row, "h3_cell") for row in rows]

    try:
        from shapely import wkb  # type: ignore[import-untyped]
        from shapely.geometry import Point  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "Exact point matching requires `pip install crc-sdk[geometry]`"
        ) from error

    point = Point(longitude, latitude)
    candidates: list[tuple[Mapping[str, Any], SpatialMatch]] = []
    for row in rows:
        encoded = row["source_geometry"]
        if encoded is None:
            candidates.append((row, "h3_cell"))
            continue
        try:
            geometry = wkb.loads(encoded)
        except Exception as error:
            raise ValueError(
                f"source_geometry for source {row['source_id']!r} is invalid WKB"
            ) from error
        if geometry.covers(point):
            candidates.append((row, "exact_geometry"))
    return candidates


def sample_hazard_at_point(
    provider: Provider,
    hazard_name: str,
    longitude: float,
    latitude: float,
    *,
    horizon: int | None = None,
    pathway: str | None = None,
    size: int = 10_000,
    seed: int | None = None,
) -> HazardPointSample:
    """Resolve exactly one canonical curve at a WGS84 point and sample it."""
    _validate_point(longitude, latitude)
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 1:
        raise ValueError("size must be positive")

    metadata = provider.metadata(hazard_name)
    cell_index = point_to_cell(longitude, latitude, metadata.h3_resolution)
    query = HazardQuery(
        hazard_name=hazard_name,
        horizon=horizon,
        pathway=pathway,
        cell_index=cell_index,
    )
    candidates = _point_candidates(
        provider.read(query),
        longitude,
        latitude,
    )
    if not candidates:
        raise LookupError(
            f"no {hazard_name!r} curve matches point "
            f"({longitude}, {latitude}) and the requested scenario"
        )
    if len(candidates) > 1:
        identities = [
            (
                row["horizon"],
                row["pathway"],
                row["source_id"],
                spatial_match,
            )
            for row, spatial_match in candidates
        ]
        raise LookupError(
            f"point ({longitude}, {latitude}) matches multiple "
            f"{hazard_name!r} curves: {identities!r}; specify horizon/pathway "
            "or disambiguate the source data"
        )

    row, spatial_match = candidates[0]
    sampled = sample_hazard_row(row, size=size, seed=seed)
    return HazardPointSample(
        hazard_name=row["hazard_name"],
        horizon=row["horizon"],
        pathway=row["pathway"],
        cell_index=row["cell_index"],
        source_id=row["source_id"],
        longitude=longitude,
        latitude=latitude,
        spatial_match=spatial_match,
        value_unit=metadata.value_unit,
        value_semantics=metadata.value_semantics,
        parameters=sampled.parameters,
        distribution=sampled.distribution,
        samples=sampled.samples,
    )
