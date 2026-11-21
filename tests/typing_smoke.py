from datetime import date
from typing import assert_type, reveal_type

import numpy as np
from crc_framework import (
    DepthAboveGround,
    DistributionFamily,
    Duration,
    LengthUnit,
    SampleFit,
    Tail,
    TimeUnit,
)

from crc_sdk import (
    AnnualMaximum,
    Calendar,
    Catalog,
    Dataset,
    DataSpec,
    GHCNh,
    IBTrACS,
    Observations,
    Period,
    Probability,
    Runtime,
    Samples,
    Sampling,
    Scenario,
    SeriesSpec,
    TimeSeriesColumns,
    Variable,
    scientific_identity,
)

variable = Variable("user:depth", measurement=DepthAboveGround(unit=LengthUnit.METRE))
scenario = Scenario("user:scenario", definition={"warming": 2.0})
period = Period.window(2030, 2050, calendar=Calendar.NO_LEAP)
series = SeriesSpec(
    variable,
    scenario=scenario,
    period=period,
    probability=Probability.annual(tail=Tail.UPPER),
)
spec = DataSpec([variable], series=[series])
reveal_type(spec.variables[0].measurement)
reveal_type(spec.series[0].probability)
reveal_type(scientific_identity(variable))
catalog = Catalog(
    namespace="user:catalog", revision="1", variables=[variable], scenarios=[scenario]
)
reveal_type(catalog.variable(variable.id))
reveal_type(SampleFit(family=DistributionFamily.GUMBEL_RIGHT))

quantile_data = {
    "support_id": ["a", "a"],
    "longitude": [0.0, 0.0],
    "latitude": [0.0, 0.0],
    "probability": [0.1, 0.9],
    "value": [1.0, 2.0],
}
dataset = Dataset.from_quantiles(quantile_data, variable=variable)
reveal_type(dataset.read().metadata)
reveal_type(dataset.batches().metadata)
with dataset.execute(runtime=Runtime()) as execution:
    reveal_type(execution.read().metadata)
    assert_type(execution.write("depth.parquet"), Dataset)

sample_data = {
    "support_id": ["a"] * 3,
    "longitude": [0.0] * 3,
    "latitude": [0.0] * 3,
    "time": [2000, 2001, 2002],
    "value": [1.0, 2.0, 4.0],
}
samples = Samples.from_table(sample_data, variable=variable)
with samples.execute() as sample_execution:
    assert_type(sample_execution.write("samples.parquet"), Samples)
reveal_type(samples.read().metadata)
reveal_type(samples.read().fit(SampleFit(family=DistributionFamily.GUMBEL_RIGHT)))

array_samples = Samples.from_table(
    {
        "support_id": np.array(["a", "a", "a"]),
        "longitude": np.zeros(3),
        "latitude": np.zeros(3),
        "time": [date(2000, 1, 1), date(2001, 1, 1), date(2002, 1, 1)],
        "value": np.array([1.0, 2.0, 4.0]),
    },
    variable=variable,
)
reveal_type(array_samples.read().metadata)

# Native evidence carries its capability and scientific kind through terminals.

station = GHCNh(indicator=GHCNh.Indicator.ANNUAL_MAXIMUM_DAILY_MAXIMUM_TEMPERATURE)
reveal_type(station.read().metadata.kind)
reveal_type(station.observations().read().metadata.kind)
reveal_type(IBTrACS(agency=IBTrACS.Agency.USA).read().metadata.kind)
observations = Observations.from_table(
    sample_data,
    variable=variable,
    columns=TimeSeriesColumns(),
    sampling=Sampling(calendar=Calendar.GREGORIAN, step=Duration(1, unit=TimeUnit.DAY)),
)
reveal_type(observations.aggregate(AnnualMaximum()).read().metadata.kind)
reveal_type(observations.read().aggregate(AnnualMaximum()))
