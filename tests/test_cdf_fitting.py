from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution

from crc_sdk.connectors import (
    read_hazard_dataset,
    read_hazard_metadata,
    write_hazard_stream,
)
from crc_sdk.fitting import (
    CDFColumnSchema,
    CDFCurveFitPolicy,
    fit_cdf_quantile_batches,
)
from crc_sdk.types import SourceProvenance
from crc_sdk.workflows import distribution_from_hazard_row


def _policy() -> CDFCurveFitPolicy:
    return CDFCurveFitPolicy(
        h3_resolution=5,
        family="gumbel_r",
        value_unit="days",
        value_semantics="annual hot days",
        producer="tests",
        creation_version="1",
        source=SourceProvenance(provider="fixture", dataset="hot_days"),
        source_id="fixture-hot-days",
    )


def test_cdf_batch_fitting_preserves_point_mass_and_fits_continuous_row(
    tmp_path: Path,
) -> None:
    probabilities = np.linspace(0.0, 1.0, 11)
    continuous = np.array([0.0, 0.0, 0.0, 0.5, 0.9, 1.4, 2.0, 2.8, 4.0, 6.0, 9.0])
    source = pa.table(
        {
            "hex_id": pa.array([599024279241097215] * 2, type=pa.uint64()),
            "index_name": ["hot_days", "hot_days"],
            "year": pa.array([2010, 2025], type=pa.int32()),
            "pathway": ["historic", "ssp245"],
            "cdf_quantiles": pa.array(
                [np.zeros(11).tolist(), continuous.tolist()],
                type=pa.large_list(pa.float64()),
            ),
        }
    )

    result = fit_cdf_quantile_batches(source, probabilities.tolist(), _policy())
    destination = tmp_path / "canonical.parquet"
    write_hazard_stream(result.stream, destination, max_workers=1)
    canonical = read_hazard_dataset(destination)

    assert result.summary.source_rows == 2
    assert result.summary.point_mass_rows == 1
    assert result.summary.hurdle_rows == 1
    assert canonical["curve_kind"].to_pylist() == ["point_mass", "hurdle"]
    point_mass = distribution_from_hazard_row(canonical)
    assert point_mass.quantiles([0.0, 0.37, 1.0]).tolist() == [0.0, 0.0, 0.0]
    assert canonical.schema.metadata is not None
    metadata = read_hazard_metadata(destination)
    assert metadata.schema_version == "1.1"
    assert metadata.source_probability_support == (0.1, 0.9)
    assert metadata.fitting is not None
    assert metadata.fitting.constant_policy == "point_mass"


def test_cdf_batch_fitting_can_skip_invalid_rows() -> None:
    probabilities = np.linspace(0.0, 1.0, 6)
    source = pa.table(
        {
            "hex_id": pa.array([599024279241097215], type=pa.uint64()),
            "index_name": ["hot_days"],
            "year": pa.array([2010], type=pa.int32()),
            "pathway": ["historic"],
            "cdf_quantiles": pa.array(
                [[0.0, 2.0, 1.0, 3.0, 4.0, 5.0]],
                type=pa.large_list(pa.float64()),
            ),
        }
    )
    policy = CDFCurveFitPolicy(
        **{
            **_policy().__dict__,
            "on_fit_failure": "skip",
        }
    )

    result = fit_cdf_quantile_batches(source, probabilities.tolist(), policy)
    assert result.stream.read_all().num_rows == 0
    assert result.summary.skipped_rows == 1


def test_configured_source_id_column_is_required() -> None:
    source = pa.table(
        {
            "hex_id": pa.array([599024279241097215], type=pa.uint64()),
            "index_name": ["hot_days"],
            "year": pa.array([2010], type=pa.int32()),
            "pathway": ["historic"],
            "cdf_quantiles": pa.array(
                [[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]],
                type=pa.large_list(pa.float64()),
            ),
        }
    )
    result = fit_cdf_quantile_batches(
        source,
        np.linspace(0.0, 1.0, 6),
        _policy(),
        columns=CDFColumnSchema(source_id="missing_source_id"),
    )

    with pytest.raises(ValueError, match="missing_source_id"):
        result.stream.read_all()


def test_unconverged_fit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pa.table(
        {
            "hex_id": pa.array([599024279241097215], type=pa.uint64()),
            "index_name": ["hot_days"],
            "year": pa.array([2010], type=pa.int32()),
            "pathway": ["historic"],
            "cdf_quantiles": pa.array(
                [[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]],
                type=pa.large_list(pa.float64()),
            ),
        }
    )
    unconverged = SimpleNamespace(
        distribution=FittedDistribution.from_parameters(
            "gumbel_r", location=1.0, scale=1.0
        ),
        diagnostics=SimpleNamespace(
            converged=False,
            iterations=2_000,
            normalized_rmse=0.0,
            maximum_absolute_residual=0.0,
        ),
    )
    monkeypatch.setattr(
        "crc_sdk.fitting.workflows.fit_quantiles",
        lambda *args, **kwargs: unconverged,
    )
    result = fit_cdf_quantile_batches(
        source,
        np.linspace(0.0, 1.0, 6),
        _policy(),
    )

    with pytest.raises(ValueError, match="did not converge after 2000 iterations"):
        result.stream.read_all()
