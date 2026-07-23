from crc_sdk.schema import HAZARD_FIELDS


def test_hazard_schema_matches_initial_contract() -> None:
    assert [field.name for field in HAZARD_FIELDS] == [
        "cell_index",
        "source_geometry",
        "hazard_name",
        "horizon",
        "pathway",
        "curve_type",
        "curve_shape",
        "curve_location",
        "curve_scale",
    ]
    assert HAZARD_FIELDS[0].data_type == "uint64"
    assert HAZARD_FIELDS[1].nullable
    assert HAZARD_FIELDS[6].nullable
