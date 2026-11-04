# Agricultural layers

[Docs index](../README.md) · [Project README](../../README.md)

`AgriculturalLayer` requests are immutable and must be bounded before they can
scan. Builder calls do no network I/O; `relation()`, `to_arrow_reader()` and
`write_parquet()` execute.

## USDA Cropland Data Layer

Requires `crc-sdk[agriculture]` (Python 3.12+). The versioned Icechunk store is
read with the official client and exposed to DuckDB as bounded Arrow batches;
neither the full raster nor the full AOI is held in memory.

```python
from crc_sdk.workflows import AgriculturalLayer

crop_mix = (
    AgriculturalLayer.usda_cdl()
    .resolution("30m")
    .for_area((-93.46, 42.14, -93.45, 42.15))
    .years(2025)
    .classes([1, 5])                       # corn and soybeans
    .scan()
    .pipeline()
    .aggregate("count(*) AS sampled_pixels", groups="year, crop_code, crop_name")
)
for batch in crop_mix.to_arrow_reader():
    process(batch)
```

## Fields of The World (FTW)

Global predicted field polygons, read natively by DuckDB over remote
GeoParquet with `bbox` pruning before exact spatial filtering (no extra needed):

```python
fields = (
    AgriculturalLayer.ftw_fields()
    .in_country("FR")
    .for_area((2.0, 47.5, 3.0, 48.5))
    .years(2024)
    .confidence_at_least(80)
    .scan()
    .pipeline()
)
fields.write_parquet("outputs/france-fields.parquet")
```

## Interpretation

FTW fields are remote-sensing units, not cadastral parcels or evidence of
ownership. CDL pixels are land-cover observations, not acreage, yield or
financial exposure.
