from typing import Optional

import pytest
from crc_framework import FittedDistribution, HurdleDistribution
from pydantic import ValidationError

from crc_sdk.types import (
    CurveParameters,
    HazardDatasetMetadata,
    SourceProvenance,
)


def test_point_mass_curve_parameters_reconstruct_exact_distribution() -> None:
    parameters = CurveParameters(
        curve_kind="point_mass",
        curve_type="point_mass",
        curve_shape=None,
        curve_location=0.0,
        curve_scale=0.0,
        curve_atom_probability=1.0,
        curve_atom_location=0.0,
    )

    distribution = parameters.to_distribution()
    assert distribution.quantiles([0.0, 0.5, 1.0]).tolist() == [0.0, 0.0, 0.0]


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
            h3_resolution=16,
            value_unit="metres",
            value_semantics="flood depth",
            producer="test",
            source=SourceProvenance(provider="test", dataset="flood"),
            creation_version="1.0",
        )


@pytest.mark.parametrize(
    ("family", "shape"),
    [
        ("genextreme", 0.1),
        ("weibull_min", 1.1),
        ("weibull_max", 1.1),
        ("skewnorm", 0.2),
        ("gumbel_r", None),
        ("gumbel_l", None),
        ("genpareto", 0.1),
    ],
)
def test_supported_curve_families_construct_public_distributions(
    family: str, shape: Optional[float]
) -> None:
    parameters = CurveParameters.model_validate(
        {
            "curve_type": family,
            "curve_shape": shape,
            "curve_location": 1.0,
            "curve_scale": 2.0,
        }
    )

    distribution = parameters.to_distribution()
    assert isinstance(distribution, FittedDistribution)
    assert distribution.family == family


def test_curve_parameters_delegate_validation_to_core() -> None:
    with pytest.raises(ValidationError):
        CurveParameters(
            curve_type="genextreme",
            curve_location=1.0,
            curve_scale=2.0,
        )
    with pytest.raises(ValidationError):
        CurveParameters(
            curve_type="gumbel_r",
            curve_location=1.0,
            curve_scale=0.0,
        )


def test_gumbel_shape_is_normalized_to_null() -> None:
    parameters = CurveParameters(
        curve_type="gumbel_r",
        curve_shape=3.0,
        curve_location=1.0,
        curve_scale=2.0,
    )

    assert parameters.curve_shape is None


def test_hurdle_curve_round_trip_uses_public_framework_api() -> None:
    parameters = CurveParameters(
        curve_kind="hurdle",
        curve_type="gumbel_r",
        curve_location=0.5,
        curve_scale=1.25,
        curve_atom_probability=0.6,
        curve_atom_location=0.0,
    )

    distribution = parameters.to_distribution()
    assert isinstance(distribution, HurdleDistribution)
    assert distribution.atom_probability == 0.6
    assert distribution.base.family == "gumbel_r"
    assert distribution.ppf(0.5) == 0.0


def test_curve_kind_controls_atom_fields() -> None:
    with pytest.raises(ValidationError, match="must not define"):
        CurveParameters(
            curve_kind="fitted",
            curve_type="gumbel_r",
            curve_location=1.0,
            curve_scale=2.0,
            curve_atom_probability=0.5,
            curve_atom_location=0.0,
        )
    with pytest.raises(ValidationError, match="require atom"):
        CurveParameters(
            curve_kind="hurdle",
            curve_type="gumbel_r",
            curve_location=1.0,
            curve_scale=2.0,
        )


def test_dataset_metadata_round_trip() -> None:
    metadata = HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="metres",
        value_semantics="flood depth",
        producer="crc-sdk",
        source=SourceProvenance(provider="test", dataset="flood"),
        creation_version="0.1.0",
    )

    assert HazardDatasetMetadata.from_json_bytes(metadata.to_json_bytes()) == metadata
    assert (
        HazardDatasetMetadata.from_parquet_metadata(metadata.to_parquet_metadata())
        == metadata
    )
