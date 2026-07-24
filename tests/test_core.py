import crc_framework

from crc_sdk import core


def test_core_reexports_preserve_identity() -> None:
    assert core.FittedDistribution is crc_framework.FittedDistribution
    assert core.HurdleDistribution is crc_framework.HurdleDistribution
    assert core.fit_distribution is crc_framework.fit_distribution
    assert core.fit_quantiles is crc_framework.fit_quantiles
    assert core.fit_hurdle_quantiles is crc_framework.fit_hurdle_quantiles
