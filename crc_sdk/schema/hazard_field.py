"""Physical field contract for fitted hazard curves."""

from dataclasses import dataclass

CANONICAL_HAZARD_SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class HazardField:
    """Dependency-neutral description of one columnar field."""

    name: str
    data_type: str
    nullable: bool = False


HAZARD_FIELDS: tuple[HazardField, ...] = (
    HazardField("cell_index", "uint64"),
    HazardField("source_id", "string"),
    HazardField("source_geometry", "binary", nullable=True),
    HazardField("hazard_name", "string"),
    HazardField("horizon", "int32"),
    HazardField("pathway", "string"),
    HazardField("curve_kind", "string"),
    HazardField("curve_type", "string"),
    HazardField("curve_shape", "float64", nullable=True),
    HazardField("curve_location", "float64"),
    HazardField("curve_scale", "float64"),
    HazardField("curve_atom_probability", "float64", nullable=True),
    HazardField("curve_atom_location", "float64", nullable=True),
)

HAZARD_ROW_KEY = (
    "hazard_name",
    "horizon",
    "pathway",
    "cell_index",
    "source_id",
)

HAZARD_SORT_ORDER = HAZARD_ROW_KEY
