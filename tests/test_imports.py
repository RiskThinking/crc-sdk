import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "crc_sdk",
        "crc_sdk.connectors",
        "crc_sdk.connectors.duckdb",
        "crc_sdk.core",
        "crc_sdk.fitting",
        "crc_sdk.geometry",
        "crc_sdk.impacts",
        "crc_sdk.providers",
        "crc_sdk.schema",
        "crc_sdk.types",
        "crc_sdk.workflows",
    ],
)
def test_public_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name)

