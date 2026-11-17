from typing import reveal_type

from crc_framework import (
    DepthAboveGround,
    DistributionFamily,
    LengthUnit,
    SampleFit,
    Tail,
)

from crc_sdk import (
    Calendar,
    Catalog,
    DataSpec,
    Period,
    Probability,
    Scenario,
    SeriesSpec,
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
