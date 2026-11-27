"""Quantile fits stream batch by batch and match for every worker count."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from crc_framework import (
    DistributionFamily,
    QuantileFit,
    Tail,
    Temperature,
    TemperatureUnit,
)

from crc_sdk import (
    Dataset,
    FitFailure,
    Probability,
    QuantileColumns,
    Runtime,
    ValidationError,
    Variable,
)

PROBABILITIES = np.linspace(0.0, 1.0, 21)[1:-1]
FLAT = {3, 11}


def quantiles(count: int, flat: set[int] = FLAT) -> Dataset:
    """Gumbel quantiles per site; the sites in `flat` cannot be fitted."""
    rng = np.random.default_rng(7)
    location = rng.normal(30.0, 5.0, count)
    scale = rng.uniform(1.0, 3.0, count)
    values = location[:, None] - scale[:, None] * np.log(-np.log(PROBABILITIES))
    for site in flat:
        values[site] = 25.0
    knots = len(PROBABILITIES)
    return Dataset.from_quantiles(
        {
            "site": np.repeat([f"site-{i:02d}" for i in range(count)], knots),
            "lon": np.repeat(np.linspace(-120.0, 120.0, count), knots),
            "lat": np.repeat(np.linspace(-40.0, 40.0, count), knots),
            "q": np.tile(PROBABILITIES, count),
            "value": values.ravel(),
        },
        variable=Variable(
            "lab:annual-maximum-temperature",
            measurement=Temperature(unit=TemperatureUnit.CELSIUS),
        ),
        probability=Probability.annual(tail=Tail.UPPER),
        columns=QuantileColumns(
            support_id="site", longitude="lon", latitude="lat", probability="q"
        ),
    )


GUMBEL = QuantileFit(family=DistributionFamily.GUMBEL_RIGHT)


@pytest.mark.parametrize("failure", [FitFailure.KEEP_NO_DATA, FitFailure.SKIP])
def test_worker_processes_reproduce_the_in_process_fit(failure: FitFailure) -> None:
    fitted = quantiles(16).fit(GUMBEL, on_failure=failure)
    serial = fitted.read(runtime=Runtime(batch_rows=5))
    parallel = fitted.read(runtime=Runtime(batch_rows=5, workers=3))

    assert parallel.table.equals(serial.table, check_metadata=True)
    assert parallel.report == serial.report
    counts = serial.report.counts
    assert counts.requested == 16
    assert (counts.skipped, counts.no_data) == (
        (2, 0) if failure is FitFailure.SKIP else (0, 2)
    )
    failed = [d.support_id for d in serial.report.diagnostics]
    assert failed == ["site-03", "site-11"]
    assert serial.table.num_rows == (14 if failure is FitFailure.SKIP else 16)


def test_the_first_failed_row_in_input_order_raises() -> None:
    fitted = quantiles(16).fit(GUMBEL)
    for runtime in (Runtime(batch_rows=5), Runtime(batch_rows=5, workers=3)):
        with pytest.raises(ValidationError) as raised:
            fitted.read(runtime=runtime)
        (diagnostic,) = raised.value.report.diagnostics
        assert diagnostic.support_id == "site-03"


def test_quantile_fitting_streams_without_a_collection_budget(
    tmp_path: Path,
) -> None:
    fitted = quantiles(16, flat=set()).fit(GUMBEL)
    saved = fitted.write(
        tmp_path / "fitted.parquet", runtime=Runtime(batch_rows=4, memory_limit=1)
    )

    assert saved.report is not None and saved.report.valid
    assert Dataset.open(tmp_path / "fitted.parquet").read().table.num_rows == 16


def test_an_empty_selection_fits_to_an_empty_valid_dataset() -> None:
    fitted = quantiles(4, flat=set()).select(variables=[]).fit(GUMBEL)
    result = fitted.read(runtime=Runtime(workers=2))

    assert result.report.valid
    assert result.table.num_rows == 0
    assert result.report.counts.requested == 0
