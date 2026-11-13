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
        "crc_sdk.geometry.pmtiles",
        "crc_sdk.impacts",
        "crc_sdk.parametric",
        "crc_sdk.providers",
        "crc_sdk.schema",
        "crc_sdk.types",
        "crc_sdk.workflows",
    ],
)
def test_public_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name)


def test_importing_the_sdk_does_not_import_s3fs() -> None:
    # s3fs (and its aiobotocore stack) is only needed for Zarr 2 OS-Climate
    # reads; importing it eagerly cost every `import crc_sdk` ~180 ms.
    import subprocess
    import sys

    code = "import sys, crc_sdk, crc_sdk.providers; print('s3fs' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
