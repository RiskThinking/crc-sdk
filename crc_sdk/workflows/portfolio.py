"""Composable asset portfolio evaluation interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Union

from crc_sdk.providers.local import LocalProvider

from .distributions import return_periods_to_probabilities

PORTFOLIO_METADATA_KEY = "crc.hazard.evaluation"


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
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "Asset Parquet inspection requires "
                "`pip install crc-sdk[connectors]`"
            ) from error
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
        if passthrough is None:
            passthrough = (
                tuple(name for name in columns if name not in required)
                if columns is not None
                else ()
            )
        else:
            passthrough = tuple(passthrough)
        object.__setattr__(self, "passthrough_columns", passthrough)


@dataclass(frozen=True)
class HazardSelection:
    """Optional canonical scenario filters."""

    hazard_names: tuple[str, ...] | None = None
    horizons: tuple[int, ...] | None = None
    pathways: tuple[str, ...] | None = None


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

    @classmethod
    def local(cls, source: str | Path) -> HazardDataset:
        """Open a canonical local Parquet hazard dataset."""
        return cls(LocalProvider(source))

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
        """Return a request evaluating the supplied upper-tail periods."""
        periods = tuple(float(value) for value in values)
        return_periods_to_probabilities(periods)
        return replace(self, periods=periods)

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
            passthrough_columns=self.portfolio.passthrough_columns or (),
            connection=options.connection,
            batch_rows=options.batch_rows,
            max_workers=options.max_workers,
            chunk_rows=options.chunk_rows,
        )
