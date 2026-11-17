"""The root exposes canonical science, while removed public paths stay absent."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "crc_sdk",
        "crc_sdk._schema",
        "crc_sdk._connectors.duckdb",
        "crc_sdk._providers.cmip6",
        "crc_sdk._geometry.pmtiles",
    ],
)
def test_canonical_and_internal_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name",
    [
        "crc_sdk.core",
        "crc_sdk.types",
        "crc_sdk.fitting",
        "crc_sdk.impacts",
        "crc_sdk.pathways",
        "crc_sdk.registry",
        "crc_sdk.workflows",
        "crc_sdk.schema",
        "crc_sdk.connectors",
        "crc_sdk.providers",
        "crc_sdk.geometry",
    ],
)
def test_removed_import_paths_are_absent(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_importing_the_sdk_does_not_import_remote_storage_clients() -> None:
    code = "import sys, crc_sdk; print('s3fs' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
