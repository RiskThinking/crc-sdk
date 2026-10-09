"""Cross-module analytical workflows."""

from crc_sdk.providers.crc_open import CRCOpenFixtureWarning

from ._remote import MaterializationResult, PrefetchResult
from .agriculture import AgriculturalLayer
from .blocks import BlockExtremaPolicy
from .byo import BYOPlan
from .crc_open import CRCOpenPlan
from .distributions import (
    CURVE_COLUMNS,
    HorizonExtrapolationWarning,
    ProbabilitySemanticsWarning,
    ReturnPeriodExtrapolationWarning,
    check_return_period_semantics,
    curve_parameters_from_row,
    curve_quantiles,
    distribution_from_hazard_row,
    return_period_value_columns,
    return_periods_to_probabilities,
    stream_curve_quantiles_wide_to_parquet,
    warn_if_extrapolated,
    warn_if_outside_window,
)
from .edo import (
    EDOAreaPlan,
    EDOCanonicalizationPlan,
    EDODroughtPolicy,
    EDOSourcePlan,
    EDOYearPlan,
)
from .era5 import (
    ERA5AreaPlan,
    ERA5CanonicalizationPlan,
    ERA5SourcePlan,
    ERA5YearPlan,
)
from .jrc import (
    JRCAreaPlan,
    JRCCanonicalizationPlan,
    JRCFloodPolicy,
    JRCPortfolioEvaluation,
    JRCSourcePlan,
)
from .portfolio import (
    PORTFOLIO_METADATA_KEY,
    AssetPortfolio,
    CellColumn,
    ExecutionOptions,
    HazardDataset,
    HazardSelection,
    ImpactContextColumns,
    PointColumns,
    PortfolioEvaluation,
    PortfolioEvaluationResult,
)
from .tiling import (
    OSClimateSelectionSpec,
    curve_quantiles_at,
    run_tiled_canonicalization,
    stream_curve_quantiles_to_parquet,
    tile_bounds,
)

__all__ = [
    "CRCOpenPlan",
    "CRCOpenFixtureWarning",
    "BlockExtremaPolicy",
    "ERA5AreaPlan",
    "ERA5CanonicalizationPlan",
    "ERA5SourcePlan",
    "ERA5YearPlan",
    "BYOPlan",
    "AgriculturalLayer",
    "AssetPortfolio",
    "CellColumn",
    "CURVE_COLUMNS",
    "ExecutionOptions",
    "EDOAreaPlan",
    "EDOCanonicalizationPlan",
    "EDODroughtPolicy",
    "EDOSourcePlan",
    "EDOYearPlan",
    "HazardDataset",
    "HazardSelection",
    "ImpactContextColumns",
    "JRCAreaPlan",
    "JRCCanonicalizationPlan",
    "JRCFloodPolicy",
    "JRCPortfolioEvaluation",
    "JRCSourcePlan",
    "MaterializationResult",
    "OSClimateSelectionSpec",
    "PORTFOLIO_METADATA_KEY",
    "PointColumns",
    "PrefetchResult",
    "PortfolioEvaluation",
    "PortfolioEvaluationResult",
    "HorizonExtrapolationWarning",
    "ProbabilitySemanticsWarning",
    "ReturnPeriodExtrapolationWarning",
    "check_return_period_semantics",
    "warn_if_outside_window",
    "curve_parameters_from_row",
    "curve_quantiles",
    "curve_quantiles_at",
    "distribution_from_hazard_row",
    "return_period_value_columns",
    "return_periods_to_probabilities",
    "run_tiled_canonicalization",
    "stream_curve_quantiles_to_parquet",
    "stream_curve_quantiles_wide_to_parquet",
    "tile_bounds",
    "warn_if_extrapolated",
]
