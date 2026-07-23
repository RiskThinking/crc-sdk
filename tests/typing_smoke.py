from pathlib import Path

from crc_sdk import FittedDistribution, LocalProvider, fit_distribution
from crc_sdk.types import HazardQuery

provider = LocalProvider(Path("/tmp/hazards"))
query = HazardQuery(hazard_name="cflood")
fit: FittedDistribution = fit_distribution([0.1, 0.2, 0.3]).distribution

