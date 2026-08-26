"""Physical field contract for fitted hazard curves."""

from dataclasses import dataclass

CANONICAL_HAZARD_SCHEMA_VERSION = "1.2"


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
    HazardField("curve_location", "float64", nullable=True),
    HazardField("curve_scale", "float64", nullable=True),
    HazardField("curve_atom_probability", "float64", nullable=True),
    HazardField("curve_atom_location", "float64", nullable=True),
    HazardField("curve_probabilities", "list_float64", nullable=True),
    HazardField("curve_values", "list_float64", nullable=True),
)


def hazard_fields_for_version(schema_version: str) -> tuple[HazardField, ...]:
    """Return the physical fields used by a canonical schema version."""
    if schema_version not in {"1.0", "1.1", "1.2"}:
        raise ValueError(f"unsupported canonical schema version: {schema_version}")
    if schema_version == "1.2":
        return HAZARD_FIELDS
    return tuple(
        HazardField(
            field.name,
            field.data_type,
            nullable=(
                False
                if field.name in {"curve_location", "curve_scale"}
                else field.nullable
            ),
        )
        for field in HAZARD_FIELDS
        if field.name not in {"curve_probabilities", "curve_values"}
    )

HAZARD_ROW_KEY = (
    "hazard_name",
    "horizon",
    "pathway",
    "cell_index",
    "source_id",
)

HAZARD_SORT_ORDER = HAZARD_ROW_KEY
