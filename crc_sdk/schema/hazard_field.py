"""Physical field contract for fitted hazard curves."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HazardField:
    """Dependency-neutral description of one columnar field."""

    name: str
    data_type: str
    nullable: bool = False


HAZARD_FIELDS: tuple[HazardField, ...] = (
    HazardField("cell_index", "uint64"),
    HazardField("source_geometry", "binary", nullable=True),
    HazardField("hazard_name", "string"),
    HazardField("horizon", "int32"),
    HazardField("pathway", "string"),
    HazardField("curve_type", "string"),
    HazardField("curve_shape", "float64", nullable=True),
    HazardField("curve_location", "float64"),
    HazardField("curve_scale", "float64"),
)

