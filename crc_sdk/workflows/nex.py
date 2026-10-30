"""Stream annual extremes from NASA's NEX-GDDP-CMIP6 archive.

NEX is bias-adjusted and statistically downscaled model output, not native
CMIP6. Each result is one model/member, not an ensemble. The AWS archive's
current CC0 terms and the original per-file model licence are both retained.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.connectors.blocks import STRIP_BYTES, _strip_row_count
from crc_sdk.connectors.duckdb.netcdf import _open_backend
from crc_sdk.providers.cmip6 import (
    CMIP6BlockReader,
    CMIP6Dataset,
    CMIP6Provider,
    RestrictiveLicenceWarning,
    cmip6_recipe,
    normalize_licence,
)
from crc_sdk.workflows._remote import Bounds, validate_bounds
from crc_sdk.workflows.cmip6 import _normalize_years

ARCHIVE = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/NEX-GDDP-CMIP6"
LICENCE_URL = "https://registry.opendata.aws/nex-gddp-cmip6/"
ATTRIBUTION = "NASA NEX-GDDP-CMIP6, https://doi.org/10.7917/OFSG3345"


class _Array:
    """Expose NetCDF dimensions through the existing CMIP6 block reader."""

    def __init__(self, variable: Any) -> None:
        self.variable = variable
        self.attrs = dict(variable.attrs)
        self.attrs["_ARRAY_DIMENSIONS"] = tuple(variable.dimensions)

    def __getitem__(self, item: Any) -> Any:
        return self.variable[item]


class _Group:
    def __init__(self, file: Any) -> None:
        self.file = file
        self.attrs = file.attrs

    def __getitem__(self, name: str) -> _Array:
        return _Array(self.file.variables[name])

    def __contains__(self, name: str) -> bool:
        return name in self.file.variables


def _component(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or ".." in value:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


@dataclass(frozen=True)
class NEXSourcePlan:
    recipe: str
    model: str
    scenario: str
    member: str = "r1i1p1f1"
    grid_label: str = "gn"
    version: str = "1.0"
    strict_licence: bool = False

    def __post_init__(self) -> None:
        cmip6_recipe(self.recipe)
        for name in ("model", "scenario", "member", "grid_label", "version"):
            _component(getattr(self, name), name)
        if self.scenario not in ("historical", "ssp126", "ssp245", "ssp370", "ssp585"):
            raise ValueError("unsupported NEX experiment")
        if self.version not in ("1.0", "1.1", "1.2"):
            raise ValueError("NEX file version must be 1.0, 1.1 or 1.2")

    def for_area(self, bounds: Sequence[float]) -> NEXAreaPlan:
        return NEXAreaPlan(self, validate_bounds(bounds))

    def _url(self, year: int) -> str:
        variable = cmip6_recipe(self.recipe).variable
        prefix = (
            f"{variable}_day_{self.model}_{self.scenario}_"
            f"{self.member}_{self.grid_label}_{year}"
        )
        suffix = "" if self.version == "1.0" else f"_v{self.version}"
        return (
            f"{ARCHIVE}/{self.model}/{self.scenario}/{self.member}/"
            f"{variable}/{prefix}{suffix}.nc"
        )


@dataclass(frozen=True)
class NEXAreaPlan:
    source: NEXSourcePlan
    bounds: Bounds

    def years(self, start: int | Sequence[int], end: int | None = None) -> NEXYearPlan:
        years = _normalize_years(start, end)
        lower, upper = (
            (1950, 2014) if self.source.scenario == "historical" else (2015, 2100)
        )
        if any(not lower <= year <= upper for year in years):
            raise ValueError(
                f"{self.source.scenario} years must be in {lower}..{upper}"
            )
        return NEXYearPlan(self, years)


@dataclass(frozen=True)
class NEXYearPlan:
    area: NEXAreaPlan
    selected_years: tuple[int, ...]

    def annual_extremes(
        self, *, strip_bytes: int = STRIP_BYTES
    ) -> Iterator[pa.RecordBatch]:
        """Yield bounded Arrow batches; network and computation begin on iteration.

        Rows carry their actual source URL and original licence metadata. Version
        is explicit: a missing revision fails instead of silently substituting a
        different one. Consume one batch at a time for out-of-core processing.
        """
        if strip_bytes <= 0:
            raise ValueError("strip_bytes must be positive")
        source = self.area.source
        recipe = cmip6_recipe(source.recipe)
        for year in self.selected_years:
            url = source._url(year)
            retrieved_at = datetime.now(UTC).isoformat()
            with _open_backend(url, None) as file:
                attrs = file.attrs
                for key, expected in (
                    ("cmip6_source_id", source.model),
                    ("scenario", source.scenario),
                    ("variant_label", source.member),
                    ("version", source.version),
                ):
                    actual = str(attrs.get(key, ""))
                    if actual != expected:
                        raise ValueError(
                            f"NEX {url}: {key} is {actual!r}, expected {expected!r}"
                        )
                licence = str(attrs.get("cmip6_license", ""))
                identifier = normalize_licence(licence)
                if identifier != "CC0-1.0":
                    message = (
                        f"NEX original model licence metadata is "
                        f"{licence or 'missing'!r}; "
                        f"NASA states the archive is now CC0. Review {LICENCE_URL} "
                        "and the original model terms before redistribution."
                    )
                    if source.strict_licence:
                        raise ValueError(message)
                    warnings.warn(message, RestrictiveLicenceWarning, stacklevel=2)
                dataset = CMIP6Dataset(
                    activity_id="NEX-GDDP-CMIP6",
                    institution_id=str(attrs.get("cmip6_institution_id", "unknown")),
                    source_id=source.model,
                    experiment_id=source.scenario,
                    member_id=source.member,
                    table_id="day",
                    variable_id=recipe.variable,
                    grid_label=source.grid_label,
                    zstore=url,
                    version=source.version,
                )
                provider = CMIP6Provider(dataset, group=_Group(file))
                provider.check_variable(recipe)
                if not str(
                    file.variables[recipe.variable].attrs.get("units", "")
                ).strip():
                    raise ValueError(f"NEX {url}: source units are missing")
                ordinals = provider.ordinals
                if ordinals.size == 0 or np.any(np.diff(ordinals) <= 0):
                    raise ValueError(
                        f"NEX {url}: daily time axis must be strictly increasing"
                    )
                start = provider.calendar.to_ordinal(year, 1, 1)
                stop = provider.calendar.to_ordinal(year + 1, 1, 1)
                if ordinals[0] < start or ordinals[-1] >= stop:
                    raise ValueError(
                        f"NEX {url}: time axis contains days outside {year}"
                    )
                reader = CMIP6BlockReader(
                    provider, recipe, self.area.bounds, years=[year]
                )
                width = len(reader.lon)
                rows = _strip_row_count(
                    strip_bytes, width, 8, 1, leading_axis=reader.leading_axis_samples
                )
                for first in range(0, len(reader.lat), rows):
                    last = min(first + rows, len(reader.lat))
                    values = reader.reduce_block(
                        year, slice(first, last), slice(0, width)
                    )
                    r, c = np.nonzero(np.isfinite(values))
                    count = len(r)
                    if not count:
                        continue
                    yield pa.record_batch(
                        {
                            "year": pa.array([year] * count, type=pa.int32()),
                            "latitude": pa.array(reader.lat[first:last][r]),
                            "longitude": pa.array(reader.lon[c]),
                            "value": pa.array(values[r, c]),
                            "unit": pa.array([recipe.unit] * count),
                            "recipe": pa.array([recipe.name] * count),
                            "model": pa.array([source.model] * count),
                            "scenario": pa.array([source.scenario] * count),
                            "member": pa.array([source.member] * count),
                            "grid_label": pa.array([source.grid_label] * count),
                            "version": pa.array([source.version] * count),
                            "calendar": pa.array([provider.calendar.name] * count),
                            "source_url": pa.array([url] * count),
                            "retrieved_at": pa.array([retrieved_at] * count),
                            "archive_licence": pa.array(["CC0-1.0"] * count),
                            "archive_licence_url": pa.array([LICENCE_URL] * count),
                            "original_model_licence": pa.array([licence] * count),
                            "attribution": pa.array([ATTRIBUTION] * count),
                        }
                    )
