import crc_framework
from crc_sdk import core


def test_core_reexports_preserve_identity() -> None:
    assert core.FittedDistribution is crc_framework.FittedDistribution
    assert core.fit_distribution is crc_framework.fit_distribution
