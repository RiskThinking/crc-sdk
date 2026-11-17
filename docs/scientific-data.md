# Scientific declarations and exact catalogs

[Documentation index](README.md) · [Python interface](api-reference.md)

A variable identity is user data. Neither a built-in name nor registration in a
catalog is required. Scenario and period are optional dimensions; missing facts
remain absent.

```python
from crc_sdk import DataSpec, SeriesSpec, Variable

variable = Variable("customer:temperature change")
specification = DataSpec([variable], series=[SeriesSpec(variable)])
assert specification.scenarios == ()
assert specification.periods == ()
```

Measurements belong to the framework. Use its named records when their
scientific meaning applies, and an open `Measurement` for custom quantities and
units. Reading data does not require an invented measurement declaration.

```python
from crc_framework import DepthAboveGround, LengthUnit, Tail
from crc_sdk import DataSpec, Probability, SeriesSpec, Variable

depth = Variable("depth", measurement=DepthAboveGround(unit=LengthUnit.METRE))
specification = DataSpec(
    [depth],
    series=[SeriesSpec(depth, probability=Probability.annual(tail=Tail.UPPER))],
)
```

Named measurements make built-in scientific facts discoverable. Depth above
local ground and elevation with a vertical datum have different records.
Model-specific durations, aggregations, thresholds, and references are named
parameters. Inspect a built-in impact's `requirements` for its actual context
constructor and accepted measurement fields; a variable name never chooses a
model. Unit conversions change numerical values and record their scale and
offset. Changing a label cannot repair a datum or aggregation mismatch.

The annual probability declaration gives return periods their scientific
meaning. A scenario name or a year window does not establish an annual
frequency. Distribution ranks, confidence probabilities, and projection
probabilities have separate typed declarations.

Year-window bounds are inclusive. Omitting a calendar permits a source to
resolve its native calendar; it does not assert Gregorian dates. An explicit
calendar is a scientific fact and must agree with the source.

```python
from crc_sdk import Calendar, DataSpec, Period, Scenario, SeriesSpec

scenario = Scenario("customer:planning")
period = Period.window(2030, 2040, calendar=Calendar.NO_LEAP)
specification = DataSpec(
    [depth], series=[SeriesSpec(depth, scenario=scenario, period=period)]
)
assert specification.periods == (period,)
```

`DataSpec` gathers the referenced scenario and period definitions. Users do not
construct descriptor-reference maps or period IDs. IDs are derived from actual
period definitions. `Period.unknown()` explicitly records missing temporal
knowledge; `Period.timeless()` declares time invariance. Omitting a period is
also valid and is preserved as a null row reference.

Scenario scientific definitions can contain user-owned fields. Supplying a
schema is optional. When present, the complete JSON Schema draft 2020-12
constraint is validated against the definition without acquiring remote schema
resources. Unknown fields in a built-in record are rejected.

`scientific_identity` compares scientific meaning: complete measurement fields,
scenario definitions and their custom schemas, and period boundaries and
calendars. Descriptor IDs, display labels, and catalog provenance do not alter
that meaning. Composition separately checks exact descriptor identities.

Catalogs are immutable and optional. Lookup uses exact nonempty IDs without
boundary whitespace. Case, internal whitespace, and Unicode are preserved;
there is no alias table, implicit fitting preset, or process-global registry.

```python
from crc_sdk import Catalog

catalog = Catalog(namespace="customer", revision="v1", variables=[depth])
assert catalog.variable("depth") is depth
assert catalog.variables == (depth,)
```

`Catalog.crc()` provides local convenience entries for documented CRC variable
and scenario identities. It does not infer science from a name, register user
data, acquire provider releases, or select a model. Provider release catalogs
can contain additional exact identities.

Catalog presets contain the framework's fitting specifications directly.
`merged` rejects conflicting scientific definitions or fitting specifications
for the same exact ID. Display labels are not scientific conflicts; the left
catalog's label is retained. The resulting namespace and revision identify the
composed catalogs, while `sha256` identifies their complete serialized content.

Catalog JSON uses taxonomy format `1.0`, and dataset metadata uses dataset
format `2.0`. Historical hazard/pathway/horizon fields and alias registries are
rejected. No migration envelope or compatibility flag is required.
