"""Point-supported block-extrema curves from station annual summaries."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np

from crc_sdk.connectors.adapters import CurveSourceInfo
from crc_sdk.connectors.blocks import plotting_position_curve
from crc_sdk.connectors.duckdb.zarr import Bounds, RasterCurve, RasterMetadata


class StationCurveSource:
    """Stream one station's annual series at a time through the SDK fitter.

    Each group contains annual summary dictionaries with value, station,
    latitude and longitude. H3 indexes the point; it is not an area of
    representativeness. Duplicate H3 cells retain distinct station source IDs.
    """

    values_are_samples = True

    def __init__(
        self,
        groups: Callable[[], Iterator[list[dict[str, Any]]]],
        metadata: RasterMetadata,
        *,
        info: CurveSourceInfo,
        tail: Literal["upper", "lower"],
        minimum_years: int,
        provider: str = "noaa-isd",
    ) -> None:
        self.groups = groups
        self.metadata = metadata
        self.info = info
        self.tail = tail
        self.minimum_years = minimum_years
        self.provider = provider
        self.skip_sink: Callable[[int, int, str, str], None] | None = None

    @property
    def axis_name(self) -> str:
        return "return period (empirical, Gringorten plotting position)"

    def iter_curves(self, bounds: Bounds | None = None) -> Iterator[RasterCurve]:
        for index, annual in enumerate(self.groups()):
            first = next((row for row in annual if row["latitude"] is not None), None)
            if first is None:
                if self.skip_sink:
                    self.skip_sink(index, 0, "no_data", "no station observations")
                continue
            lat, lon = first["latitude"], first["longitude"]
            if bounds is not None and not (
                bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]
            ):
                continue
            # Coordinate jumps are not silently pooled into a single point.
            if any(
                abs(row["latitude"] - lat) > 0.05 or abs(row["longitude"] - lon) > 0.05
                for row in annual
                if row["latitude"] is not None
            ):
                raise ValueError(f"station {first['station']} moved; split its record")
            periods, values = plotting_position_curve(
                np.asarray([row["value"] for row in annual], dtype=float),
                self.tail,
                minimum_blocks=self.minimum_years,
            )
            if not values.size:
                if self.skip_sink:
                    self.skip_sink(
                        index,
                        0,
                        "too_few_blocks",
                        f"station {first['station']}: need {self.minimum_years} years",
                    )
                continue
            point = (lon, lat)
            yield RasterCurve(
                row=index,
                column=0,
                boundary=(point, point, point, point),
                axis_values=periods,
                values=values,
                point=point,
                source_id=f"{self.provider}:{first['station']}",
            )
