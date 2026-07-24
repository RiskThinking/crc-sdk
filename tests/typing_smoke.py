from pathlib import Path

from crc_sdk import (
    FittedDistribution,
    HurdleDistribution,
    HurdleQuantileFitResult,
    LocalProvider,
    QuantileFitResult,
    TabulatedDistribution,
    fit_distribution,
    fit_hurdle_quantiles,
    fit_quantiles,
)
from crc_sdk.types import HazardQuery

provider = LocalProvider(Path("/tmp/hazards"))
query = HazardQuery(hazard_name="cflood")
fit: FittedDistribution = fit_distribution([0.1, 0.2, 0.3]).distribution
knots = TabulatedDistribution([0.5, 0.8, 0.9, 0.99], [0.0, 0.2, 0.5, 1.0])
quantile_fit: QuantileFitResult = fit_quantiles(knots)
hurdle_fit: HurdleQuantileFitResult = fit_hurdle_quantiles(
    knots,
    atom_probability=0.5,
)
hurdle: HurdleDistribution = hurdle_fit.distribution
