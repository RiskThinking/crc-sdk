"""Composable asset portfolio evaluation interface."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from crc_framework import CallableImpact, ImpactFunction

from crc_sdk.providers.local import LocalProvider
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

from .distributions import (
    CURVE_COLUMNS,
    return_period_value_columns,
    return_periods_to_probabilities,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from crc_sdk.connectors.adapters import CurveFitIngestPolicy
    from crc_sdk.fitting.workflows import CDFColumnSchema, CDFCurveFitPolicy

    from ._remote import MaterializationResult
    from .byo import BYOPlan
    from .edo import EDOSourcePlan
    from .era5 import ERA5SourcePlan
    from .jrc import JRCSourcePlan

PORTFOLIO_METADATA_KEY = "crc.hazard.evaluation"
_PORTFOLIO_RESERVED_COLUMNS = frozenset(
    {
        "cell_index",
        "hazard_name",
        "horizon",
        "pathway",
        "source_id",
        "spatial_match",
        *CURVE_COLUMNS,
    }
)


def _reserved_portfolio_columns(
    return_periods: Sequence[float],
) -> frozenset[str]:
    return _PORTFOLIO_RESERVED_COLUMNS | frozenset(
        return_period_value_columns(return_periods)
    )


@dataclass(frozen=True)
class PortfolioEvaluationResult:
    """Summary of one streamed portfolio evaluation."""

    output: str | Path
    row_count: int
    return_periods: tuple[float, ...]
    value_columns: tuple[str, ...]


@dataclass(frozen=True)
class PointColumns:
    """Longitude/latitude columns used to locate portfolio assets."""

    longitude: str = "longitude"
    latitude: str = "latitude"


@dataclass(frozen=True)
class CellColumn:
    """Canonical unsigned H3 column used to locate portfolio assets."""

    name: str = "cell_index"


AssetLocation = Union[PointColumns, CellColumn]


def _source_columns(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, Path):
        return tuple(pq.read_schema(value).names)
    if isinstance(value, str):
        return None
    columns = getattr(value, "column_names", None)
    if columns is None:
        columns = getattr(value, "columns", None)
    if columns is None:
        schema = getattr(value, "schema", None)
        columns = getattr(schema, "names", None)
    return tuple(columns) if columns is not None else None


@dataclass(frozen=True)
class AssetPortfolio:
    """Asset source plus its identifier, location, and retained columns."""

    data: Any
    id_column: str = "asset_id"
    location: AssetLocation | None = None
    passthrough_columns: tuple[str, ...] | None = None
    _passthrough_inferred: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.id_column:
            raise ValueError("asset id column must not be empty")
        columns = _source_columns(self.data)
        location = self.location
        if location is None:
            if columns is None:
                raise ValueError(
                    "SQL asset sources require an explicit PointColumns "
                    "or CellColumn location"
                )
            if {"longitude", "latitude"}.issubset(columns):
                location = PointColumns()
            elif "cell_index" in columns:
                location = CellColumn()
            else:
                raise ValueError(
                    "assets must contain longitude/latitude or cell_index; "
                    "provide an explicit location mapping for other names"
                )
            object.__setattr__(self, "location", location)
        elif not isinstance(location, (PointColumns, CellColumn)):
            raise TypeError("location must be PointColumns or CellColumn")

        required = [self.id_column]
        if isinstance(location, PointColumns):
            if not location.longitude or not location.latitude:
                raise ValueError("point column names must not be empty")
            required.extend([location.longitude, location.latitude])
        else:
            if not location.name:
                raise ValueError("cell column name must not be empty")
            required.append(location.name)
        if columns is not None:
            missing = [name for name in required if name not in columns]
            if missing:
                raise ValueError(f"asset source is missing columns {missing!r}")

        passthrough = self.passthrough_columns
        inferred_passthrough = passthrough is None
        if passthrough is None:
            passthrough = (
                tuple(
                    name
                    for name in columns
                    if name not in required and name not in _PORTFOLIO_RESERVED_COLUMNS
                )
                if columns is not None
                else ()
            )
        else:
            passthrough = tuple(passthrough)
        object.__setattr__(self, "passthrough_columns", passthrough)
        object.__setattr__(self, "_passthrough_inferred", inferred_passthrough)


@dataclass(frozen=True)
class HazardSelection:
    """Optional canonical scenario filters."""

    hazard_names: tuple[str, ...] | None = None
    horizons: tuple[int, ...] | None = None
    pathways: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ImpactContextColumns:
    """Asset columns used to build per-row framework impact context."""

    country: str | None = None
    continent: str | None = None
    building_type: str | None = None
    historic_mean: str | None = None

    def __post_init__(self) -> None:
        for field_name, column_name in self.items():
            if not column_name:
                raise ValueError(f"{field_name} context column must not be empty")

    def items(self) -> tuple[tuple[str, str], ...]:
        """Return configured framework-context field and asset-column pairs."""
        return tuple(
            (field_name, column_name)
            for field_name, column_name in (
                ("country", self.country),
                ("continent", self.continent),
                ("building_type", self.building_type),
                ("historic_mean", self.historic_mean),
            )
            if column_name is not None
        )


@dataclass(frozen=True)
class ImpactSpec:
    """Internal event-aligned impact evaluation configuration."""

    function: ImpactFunction
    name: str
    value_unit: str
    value_semantics: str
    context_columns: ImpactContextColumns = ImpactContextColumns()

    @property
    def function_type(self) -> str:
        return type(self.function).__name__

    def is_picklable(self) -> bool:
        try:
            pickle.dumps(self)
        except (AttributeError, pickle.PickleError, TypeError):
            return False
        return True


@dataclass(frozen=True)
class ExecutionOptions:
    """Advanced streaming and worker controls."""

    connection: Any | None = None
    batch_rows: int = 50_000
    max_workers: int | None = None
    chunk_rows: int = 20_000

    def __post_init__(self) -> None:
        if self.batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be positive")


@dataclass(frozen=True)
class HazardDataset:
    """Composable facade over one canonical local hazard dataset."""

    provider: LocalProvider
    materialization: MaterializationResult | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def local(
        cls,
        source: str | Path,
        *,
        materialization: MaterializationResult | None = None,
    ) -> HazardDataset:
        """Open a canonical local Parquet hazard dataset."""
        return cls(LocalProvider(source), materialization=materialization)

    @classmethod
    def jrc(
        cls,
        dataset: str,
        *,
        version: str = "latest",
    ) -> JRCSourcePlan:
        """Plan lazy ingestion of a named JRC flood dataset."""
        from crc_sdk.providers.jrc import jrc_dataset

        from .jrc import JRCSourcePlan

        jrc_dataset(dataset)
        return JRCSourcePlan(dataset=dataset.lower(), requested_version=version)

    @classmethod
    def efas(cls, *, version: str = "latest") -> JRCSourcePlan:
        """Plan lazy ingestion of CEMS-EFAS flood maps."""
        return cls.jrc("efas", version=version)

    @classmethod
    def glofas(cls, *, version: str = "latest") -> JRCSourcePlan:
        """Plan lazy ingestion of CEMS-GLOFAS flood maps."""
        return cls.jrc("glofas", version=version)

    @classmethod
    def edo(
        cls,
        dataset: str,
        *,
        version: str = "latest",
    ) -> EDOSourcePlan:
        """Plan lazy ingestion of a named JRC/EDO drought dataset."""
        from crc_sdk.providers.jrc_edo import edo_dataset

        from .edo import EDOSourcePlan

        edo_dataset(dataset)
        return EDOSourcePlan(dataset=dataset.lower(), requested_version=version)

    @classmethod
    def smi(cls, *, version: str = "latest") -> EDOSourcePlan:
        """Plan lazy ingestion of EDO Soil Moisture Index drought curves."""
        return cls.edo("smi", version=version)

    @classmethod
    def from_table(
        cls,
        source: str | Path | pa.Table,
        *,
        columns: CDFColumnSchema | None = None,
        policy: CDFCurveFitPolicy | None = None,
        probabilities: Sequence[float] | None = None,
        return_periods: Sequence[float] | None = None,
    ) -> BYOPlan:
        """Plan canonicalization of one-row-per-distribution tabular data.

        `source` is a Parquet path, a DuckDB-readable path (CSV, JSON, ...) or
        a pyarrow Table whose rows are already keyed by H3 cell.
        """
        from crc_sdk.fitting.workflows import CDFColumnSchema as Columns

        from .byo import table_plan

        return table_plan(
            source,
            columns=columns or Columns(),
            policy=policy,
            probabilities=probabilities,
            return_periods=return_periods,
        )

    @classmethod
    def from_zarr(
        cls,
        source: str | Path | Any,
        *,
        hazard_type: str,
        indicator_id: str,
        scenario: str,
        year: int,
        units: str,
        array: str | None = None,
        policy: CurveFitIngestPolicy | None = None,
        bounds: Sequence[float] | None = None,
        storage_options: Mapping[str, Any] | None = None,
    ) -> BYOPlan:
        """Plan canonicalization of a return-period Zarr raster.

        `source` is a Zarr URL/path (opened lazily; `array` names the array
        inside a group) or an already-open array.
        """
        from .byo import zarr_plan

        return zarr_plan(
            source,
            hazard_type=hazard_type,
            indicator_id=indicator_id,
            scenario=scenario,
            year=year,
            units=units,
            array=array,
            policy=policy,
            bounds=bounds,
            storage_options=storage_options,
        )

    @classmethod
    def from_raster(
        cls,
        paths: Mapping[int, str | Path] | Sequence[str | Path],
        *,
        return_periods: Sequence[int] | None = None,
        hazard_type: str,
        indicator_id: str,
        scenario: str,
        year: int,
        units: str,
        policy: CurveFitIngestPolicy | None = None,
        bounds: Sequence[float] | None = None,
        band: int = 1,
        assumed_crs: str | None = None,
    ) -> BYOPlan:
        """Plan canonicalization of same-grid GeoTIFFs, one per return period.

        Pass `{return_period: path}`, or a path sequence with `return_periods`.
        """
        from .byo import raster_plan

        return raster_plan(
            paths,
            return_periods=return_periods,
            hazard_type=hazard_type,
            indicator_id=indicator_id,
            scenario=scenario,
            year=year,
            units=units,
            policy=policy,
            bounds=bounds,
            band=band,
            assumed_crs=assumed_crs,
        )

    @classmethod
    def era5(
        cls,
        recipe: str,
        *,
        store: str = "arco-0p25",
        version: str = "latest",
    ) -> ERA5SourcePlan:
        """Plan a lazy ERA5 historical baseline for one supported recipe.

        ``recipe`` is ``txx``, ``tnn``, ``rx1day`` or ``rx5day``; ``store`` is
        ``arco-0p25`` (native 0.25 degrees, minutes per year) or ``wb2-1p5``
        (1.5 degree copy, ~30 seconds per year).
        """
        from crc_sdk.providers.era5 import era5_recipe, era5_store

        from .era5 import ERA5SourcePlan

        era5_recipe(recipe)
        era5_store(store)
        return ERA5SourcePlan(
            recipe=recipe.lower(), store=store.lower(), requested_version=version
        )

    def metadata(self) -> HazardDatasetMetadata:
        """Return this canonical dataset's embedded metadata."""
        from crc_sdk.connectors.parquet import read_hazard_metadata

        return read_hazard_metadata(self.provider.source)

    def provenance(self) -> SourceProvenance:
        """Return the embedded source provenance."""
        return self.metadata().source

    def for_assets(
        self,
        assets: Any | AssetPortfolio,
    ) -> PortfolioEvaluation:
        """Start an evaluation for an inferred or explicit asset portfolio."""
        portfolio = (
            assets if isinstance(assets, AssetPortfolio) else AssetPortfolio(assets)
        )
        return PortfolioEvaluation(dataset=self, portfolio=portfolio)


@dataclass(frozen=True)
class PortfolioEvaluation:
    """Immutable, composable portfolio evaluation request."""

    dataset: HazardDataset
    portfolio: AssetPortfolio
    selection: HazardSelection = HazardSelection()
    periods: tuple[float, ...] | None = None
    impact_spec: ImpactSpec | None = None

    def select(
        self,
        *,
        hazard_names: Sequence[str] | None = None,
        horizons: Sequence[int] | None = None,
        pathways: Sequence[str] | None = None,
    ) -> PortfolioEvaluation:
        """Return a request with canonical hazard/scenario filters."""
        current = self.selection
        return replace(
            self,
            selection=HazardSelection(
                hazard_names=(
                    tuple(hazard_names)
                    if hazard_names is not None
                    else current.hazard_names
                ),
                horizons=(
                    tuple(horizons) if horizons is not None else current.horizons
                ),
                pathways=(
                    tuple(pathways) if pathways is not None else current.pathways
                ),
            ),
        )

    def return_periods(
        self,
        values: Sequence[float],
    ) -> PortfolioEvaluation:
        """Return a request evaluated with the dataset's persisted tail."""
        periods = tuple(float(value) for value in values)
        return_periods_to_probabilities(periods)
        return replace(self, periods=periods)

    def impact(
        self,
        function: ImpactFunction | Callable[[Any], Any],
        *,
        name: str,
        value_unit: str,
        value_semantics: str,
        context: ImpactContextColumns | None = None,
    ) -> PortfolioEvaluation:
        """Return a request applying an event-aligned impact function."""
        for field_name, value in (
            ("impact name", name),
            ("impact value_unit", value_unit),
            ("impact value_semantics", value_semantics),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if isinstance(function, ImpactFunction):
            impact_function = function
        elif callable(function):
            impact_function = CallableImpact(function)
        else:
            raise TypeError("impact function must define evaluate(...) or be callable")
        if context is not None and not isinstance(context, ImpactContextColumns):
            raise TypeError("impact context must be ImpactContextColumns")
        return replace(
            self,
            impact_spec=ImpactSpec(
                function=impact_function,
                name=name,
                value_unit=value_unit,
                value_semantics=value_semantics,
                context_columns=context or ImpactContextColumns(),
            ),
        )

    def write_parquet(
        self,
        output: str | Path,
        *,
        execution: ExecutionOptions | None = None,
    ) -> PortfolioEvaluationResult:
        """Stream this evaluation to compressed Parquet."""
        from ._portfolio import evaluate_hazard_portfolio

        if self.periods is None:
            raise ValueError("return_periods(...) must be configured before writing")
        options = execution or ExecutionOptions()
        max_workers = options.max_workers
        if self.impact_spec is not None and not self.impact_spec.is_picklable():
            if max_workers is not None and max_workers > 1:
                raise ValueError(
                    "impact function is not picklable; use max_workers=1 "
                    "or a top-level callable"
                )
            max_workers = 1
        location = self.portfolio.location
        if isinstance(location, PointColumns):
            longitude_column = location.longitude
            latitude_column = location.latitude
            cell_index_column = None
        else:
            assert isinstance(location, CellColumn)
            longitude_column = None
            latitude_column = None
            cell_index_column = location.name
        passthrough_columns = self.portfolio.passthrough_columns or ()
        if self.portfolio._passthrough_inferred:
            reserved = _reserved_portfolio_columns(self.periods)
            passthrough_columns = tuple(
                name for name in passthrough_columns if name not in reserved
            )
        return evaluate_hazard_portfolio(
            provider=self.dataset.provider,
            assets=self.portfolio.data,
            output=output,
            return_periods=self.periods,
            asset_id_column=self.portfolio.id_column,
            longitude_column=longitude_column,
            latitude_column=latitude_column,
            cell_index_column=cell_index_column,
            hazard_names=self.selection.hazard_names,
            horizons=self.selection.horizons,
            pathways=self.selection.pathways,
            passthrough_columns=passthrough_columns,
            impact=self.impact_spec,
            connection=options.connection,
            batch_rows=options.batch_rows,
            max_workers=max_workers,
            chunk_rows=options.chunk_rows,
        )
