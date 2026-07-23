import pytest
from crc_sdk.types import CurveParameters, HazardDatasetMetadata
from pydantic import ValidationError


def test_curve_shape_is_optional() -> None:
    parameters = CurveParameters(
        curve_type="gumbel_r",
        curve_location=1.0,
        curve_scale=2.0,
    )

    assert parameters.curve_shape is None


def test_h3_resolution_is_validated() -> None:
    with pytest.raises(ValidationError):
        HazardDatasetMetadata(
            schema_version="1",
            h3_resolution=16,
            value_unit="metres",
        )

