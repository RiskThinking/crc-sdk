"""Validate declared interfaces without freezing superseded implementation."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Generic, Literal, TypeVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
ResultT = TypeVar("ResultT")


class _GenericRecipe(Generic[ResultT]):
    def read(self) -> ResultT:
        raise NotImplementedError


class _StringRecipe(_GenericRecipe[str]):
    pass


class _EvidenceKind(Enum):
    SAMPLES = "samples"
    OBSERVATIONS = "observations"


KindT = TypeVar("KindT")
ItemT = TypeVar("ItemT")


class _TypedEvidence(Generic[KindT]):
    pass


class _LiteralRecipe(_GenericRecipe[_TypedEvidence[Literal[_EvidenceKind.SAMPLES]]]):
    pass


class _NestedRecipe(_GenericRecipe[tuple[ItemT, ...]], Generic[ItemT]):
    pass


class _NestedStringRecipe(_NestedRecipe[str]):
    pass


class _Provider:
    class Product(Enum):
        OBSERVATIONS = "observations"

    def __init__(self, *, product: _Provider.Product = Product.OBSERVATIONS) -> None:
        self.product = product


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "sdk_interface_checker", ROOT / "tools" / "api_contracts" / "check.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def declarations(checker: ModuleType) -> tuple[dict[str, Any], dict[str, Any]]:
    contract, release = checker.load_contracts()
    return copy.deepcopy(contract), copy.deepcopy(release)


def test_release_contract_is_consistent(
    checker: ModuleType, declarations: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, release = declarations
    checker.validate_contract(contract, release)
    assert "sources" not in contract["exports"]
    assert "Dataset" in contract["exports"]
    assert "Samples" in contract["exports"]
    assert "Measurement" not in contract["exports"]


def test_inherited_generic_contracts_check_the_specialized_type(
    checker: ModuleType,
) -> None:
    bindings = checker._generic_bindings(_StringRecipe)
    assert (
        checker._check_runtime_signature(
            _StringRecipe.read,
            {"parameters": [], "returns": "str"},
            "StringRecipe.read",
            bindings=bindings,
        )
        == []
    )
    assert "return annotation differs" in " ".join(
        checker._check_runtime_signature(
            _StringRecipe.read,
            {"parameters": [], "returns": "int"},
            "StringRecipe.read",
            bindings=bindings,
        )
    )


def test_declaration_validation_does_not_import_the_sdk(
    checker: ModuleType,
    declarations: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_import(name: str) -> None:
        raise AssertionError(f"static validation imported {name}")

    monkeypatch.setattr(checker.importlib, "import_module", reject_import)
    checker.validate_contract(*declarations)


@pytest.mark.parametrize(
    "annotation", ["Any", "object", "UndefinedType", "crc_framework.UndefinedType"]
)
def test_untyped_or_unknown_annotations_are_rejected(
    checker: ModuleType,
    declarations: tuple[dict[str, Any], dict[str, Any]],
    annotation: str,
) -> None:
    contract, release = declarations
    contract["exports"]["Dataset"]["methods"]["read"]["returns"] = annotation
    with pytest.raises(checker.ContractError, match="untyped|unknown"):
        checker.validate_contract(contract, release)


def test_enum_aliases_are_rejected(
    checker: ModuleType, declarations: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, release = declarations
    contract["exports"]["FitFailure"]["members"]["ERROR"] = "raise"
    with pytest.raises(checker.ContractError, match="enum aliases"):
        checker.validate_contract(contract, release)


def test_unknown_enum_defaults_are_rejected(
    checker: ModuleType, declarations: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, release = declarations
    contract["exports"]["Samples"]["methods"]["fit"]["parameters"][1]["default"] = {
        "enum": "FitFailure.UNDECLARED"
    }
    with pytest.raises(checker.ContractError, match="unknown enum default"):
        checker.validate_contract(contract, release)


def test_duplicate_member_names_are_rejected(
    checker: ModuleType, declarations: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, release = declarations
    contract["exports"]["Dataset"]["properties"]["read"] = "str"
    with pytest.raises(checker.ContractError, match="name collision"):
        checker.validate_contract(contract, release)


def test_inheritance_cycles_are_rejected(
    checker: ModuleType, declarations: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, release = declarations
    contract["exports"]["Samples"]["inherits"] = ["ERA5"]
    with pytest.raises(checker.ContractError, match="inheritance cycle"):
        checker.validate_contract(contract, release)


def test_removed_methods_cannot_reenter_the_allowlist(
    checker: ModuleType, declarations: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, release = declarations
    contract["exports"]["Dataset"]["methods"]["preserve"] = {
        "parameters": [],
        "returns": "Dataset",
    }
    with pytest.raises(checker.ContractError, match="removed method"):
        checker.validate_contract(contract, release)


def test_framework_types_cannot_be_reexported_by_sdk(
    checker: ModuleType, declarations: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, release = declarations
    contract["exports"]["Measurement"] = {"kind": "class"}
    contract["exports"] = dict(sorted(contract["exports"].items()))
    with pytest.raises(checker.ContractError, match="must not re-export"):
        checker.validate_contract(contract, release)


@pytest.mark.parametrize(
    ("format_name", "unnecessary_version"),
    [
        ("evaluation", "3.0"),
        ("collection", "2.0"),
        ("taxonomy", "2.0"),
        ("dataset", "3.0"),
        ("annual-loss", "2.0"),
    ],
)
def test_release_ledger_must_match_owned_formats(
    checker: ModuleType,
    declarations: tuple[dict[str, Any], dict[str, Any]],
    format_name: str,
    unnecessary_version: str,
) -> None:
    _, release = declarations
    release["formats"][format_name] = unnecessary_version
    with pytest.raises(
        checker.ContractError, match=f"{format_name} schema does not match"
    ):
        checker.check_schema_versions(release)


def test_duplicate_json_fields_are_rejected(
    checker: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "public-api.json").write_text('{"exports": {}, "exports": {}}')
    (tmp_path / "release.json").write_text("{}")
    with pytest.raises(checker.ContractError, match="duplicate JSON field"):
        checker.load_contracts(tmp_path)


def test_framework_version_floor_comes_from_the_ledger(
    checker: ModuleType,
    declarations: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    _, release = declarations
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nversion = "1.0.0"\ndependencies = ["crc-framework>=1.0.0,<2"]\n'
    )
    checker.check_versions(release, tmp_path)
    project.write_text(project.read_text().replace(">=1.0.0,<2", ">=0.6.0,<0.7"))
    with pytest.raises(checker.ContractError, match="framework dependency"):
        checker.check_versions(release, tmp_path)


def test_runtime_signatures_check_defaults_kinds_and_return_types(
    checker: ModuleType,
) -> None:
    def function(value: str, *, overwrite: bool = False) -> str:
        return value if not overwrite else value.upper()

    signature: dict[str, Any] = {
        "parameters": [
            {"name": "value", "kind": "positional_or_keyword", "annotation": "str"},
            {
                "name": "overwrite",
                "kind": "keyword_only",
                "annotation": "bool",
                "default": False,
            },
        ],
        "returns": "str",
    }
    assert checker._check_runtime_signature(function, signature, "function") == []
    signature["parameters"][1]["default"] = True
    assert "default differs" in " ".join(
        checker._check_runtime_signature(function, signature, "function")
    )
    signature["parameters"][1]["default"] = False
    signature["parameters"][1]["kind"] = "positional_or_keyword"
    assert "kind differs" in " ".join(
        checker._check_runtime_signature(function, signature, "function")
    )
    signature["parameters"][1]["kind"] = "keyword_only"
    signature["returns"] = "int"
    assert "return annotation differs" in " ".join(
        checker._check_runtime_signature(function, signature, "function")
    )


def test_omitted_selection_default_is_a_singleton(checker: ModuleType) -> None:
    expected = {"sentinel": "omitted"}
    assert checker._default_matches(Ellipsis, expected)
    assert not checker._default_matches(object(), expected)
    assert not checker._default_matches(None, expected)


def test_tuple_defaults_preserve_values_and_types(
    checker: ModuleType, declarations: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, release = declarations
    level = contract["exports"]["LossResult"]["methods"]["summary"]["parameters"][0]
    assert checker._default_matches((0.95, 0.99, 0.995), level["default"])
    assert not checker._default_matches((0.95, 0.99), level["default"])
    assert not checker._default_matches((10.0,), {"tuple": [10]})
    level["default"]["tuple"] = [0.95, [0.99]]
    with pytest.raises(checker.ContractError, match="tuple default"):
        checker.validate_contract(contract, release)


@pytest.mark.parametrize("extra", ["root_alias", "method_alias"])
def test_runtime_rejects_undeclared_aliases_outside_all(
    checker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: str,
) -> None:
    class Dataset:
        def read(self) -> str:
            return "rows"

    Dataset.__module__ = "crc_sdk._dataset"
    package = ModuleType("crc_sdk")
    package.__file__ = str(tmp_path / "__init__.py")
    setattr(package, "__all__", ["Dataset"])
    setattr(package, "Dataset", Dataset)
    if extra == "root_alias":
        setattr(package, "AlternateDataset", Dataset)
    else:
        Dataset.collect = Dataset.read  # type: ignore[attr-defined]
    _, release = checker.load_contracts()
    contract = {
        "schema_version": "1.0",
        "package": "crc-sdk",
        "canonical_module": "crc_sdk",
        "exports": {
            "Dataset": {
                "kind": "class",
                "methods": {"read": {"parameters": [], "returns": "str"}},
            }
        },
        "external_symbols": {"crc_framework": []},
        "removals": {
            "symbols": ["crc_sdk.removed.FormerDataset"],
            "modules": ["crc_sdk.removed"],
            "methods": {"*": ["preserve"]},
            "schema_fields": ["hazard_name"],
            "wire_formats": ["dataset:1.3"],
        },
        "naming": {"one_name_per_concept": True},
    }
    checker.validate_contract(contract, release)
    resources = tmp_path / "contracts" / "api"
    resources.mkdir(parents=True)
    (resources / "public-api.json").write_text(json.dumps(contract))
    (resources / "release.json").write_text(json.dumps(release))
    monkeypatch.setattr(checker, "import_runtime", lambda path: package)
    monkeypatch.setattr(checker, "files", lambda owner: tmp_path)
    monkeypatch.setattr(checker, "packaged_errors", lambda path: [])

    def missing(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        checker,
        "importlib",
        SimpleNamespace(
            import_module=missing, metadata=SimpleNamespace(version=lambda _: "1.0.0")
        ),
    )
    with pytest.raises(checker.ContractError, match="undeclared public"):
        checker.check_runtime(contract, release)


def test_contract_files_are_plain_packaged_json() -> None:
    directory = ROOT / "crc_sdk" / "contracts" / "api"
    for path in directory.glob("*.json"):
        parsed = json.loads(path.read_text())
        assert parsed["schema_version"] == "1.0"
        assert "0x" not in path.read_text()
    assert not (directory / "legacy-baseline.json").exists()


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("schema/hazard_field.py", "removed public namespace remains packaged"),
        ("connectors/parquet.py", "removed public namespace remains packaged"),
        ("pathways.py", "removed public namespace remains packaged"),
        ("contracts/schemas/dataset-1.3.schema.json", "undeclared schema artifact"),
        ("contracts/api/legacy-baseline.json", "superseded contract resource"),
    ],
)
def test_unimportable_removed_artifacts_are_rejected_in_installed_packages(
    checker: ModuleType,
    declarations: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
    relative: str,
    message: str,
) -> None:
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("raise ModuleNotFoundError('removed dependency')")
    errors = checker._packaged_removal_errors(tmp_path, *declarations)
    assert any(message in error for error in errors), errors


def test_cli_checks_declarations_without_importing_current_exports() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "api_contracts" / "check.py")],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "declarations only" in result.stdout


def test_function_exports_validate_their_signature(
    checker: ModuleType,
    declarations: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def identity(value: str) -> str:
        return value

    _, release = declarations
    contract: dict[str, Any] = {
        "schema_version": "1.0",
        "package": "crc-sdk",
        "canonical_module": "crc_sdk",
        "exports": {
            "identity": {
                "kind": "function",
                "parameters": [
                    {
                        "name": "value",
                        "kind": "positional_or_keyword",
                        "annotation": "str",
                    }
                ],
                "returns": "str",
            }
        },
        "external_symbols": {"crc_framework": []},
        "removals": {
            "symbols": ["crc_sdk.removed.Old"],
            "modules": ["crc_sdk.removed"],
            "methods": {"*": ["preserve"]},
            "schema_fields": ["hazard_name"],
            "wire_formats": ["dataset:1.3"],
        },
        "naming": {"one_name_per_concept": True},
    }
    checker.validate_contract(contract, release)
    package = ModuleType("crc_sdk")
    package.__file__ = str(tmp_path / "__init__.py")
    setattr(package, "__all__", ["identity"])
    setattr(package, "identity", identity)
    resources = tmp_path / "contracts" / "api"
    resources.mkdir(parents=True)
    (resources / "public-api.json").write_text(json.dumps(contract))
    (resources / "release.json").write_text(json.dumps(release))
    monkeypatch.setattr(checker, "import_runtime", lambda path: package)
    monkeypatch.setattr(checker, "files", lambda owner: tmp_path)
    monkeypatch.setattr(checker, "packaged_errors", lambda path: [])

    def missing(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        checker,
        "importlib",
        SimpleNamespace(
            import_module=missing, metadata=SimpleNamespace(version=lambda _: "1.0.0")
        ),
    )
    checker.check_runtime(contract, release)
    delattr(package, "identity")
    with pytest.raises(
        checker.ContractError, match="crc_sdk.identity: declared export is missing"
    ):
        checker.check_runtime(contract, release)
    setattr(package, "identity", identity)
    contract["exports"]["identity"]["returns"] = "int"
    (resources / "public-api.json").write_text(json.dumps(contract))
    with pytest.raises(checker.ContractError, match="return annotation differs"):
        checker.check_runtime(contract, release)


def test_sequence_defaults_are_immutable_exact_tuples(checker: ModuleType) -> None:
    assert checker._default_matches((), {"tuple": []})
    assert not checker._default_matches([], {"tuple": []})
    assert not checker._default_matches(None, {"tuple": []})


def test_nested_literal_generic_parameters_keep_the_evidence_kind(
    checker: ModuleType,
) -> None:
    bindings = checker._generic_bindings(_LiteralRecipe)
    assert bindings["ResultT"] == "_TypedEvidence[Literal[_EvidenceKind.SAMPLES]]"
    assert (
        checker._check_runtime_signature(
            _LiteralRecipe.read,
            {
                "parameters": [],
                "returns": "_TypedEvidence[Literal[_EvidenceKind.SAMPLES]]",
            },
            "LiteralRecipe.read",
            bindings=bindings,
        )
        == []
    )
    assert checker._check_runtime_signature(
        _LiteralRecipe.read,
        {
            "parameters": [],
            "returns": "_TypedEvidence[Literal[_EvidenceKind.OBSERVATIONS]]",
        },
        "LiteralRecipe.read",
        bindings=bindings,
    ) == ["LiteralRecipe.read: return annotation differs"]


def test_transitive_generic_bindings_keep_nested_parameters_and_ellipsis(
    checker: ModuleType,
) -> None:
    bindings = checker._generic_bindings(_NestedStringRecipe)
    assert bindings["ItemT"] == "str"
    assert bindings["ResultT"] == "tuple[str,...]"
    assert (
        checker._check_runtime_signature(
            _NestedStringRecipe.read,
            {"parameters": [], "returns": "tuple[str, ...]"},
            "NestedStringRecipe.read",
            bindings=bindings,
        )
        == []
    )


def test_nested_enum_defaults_require_the_canonical_owner(checker: ModuleType) -> None:
    assert checker._default_matches(
        _Provider.Product.OBSERVATIONS, {"enum": "_Provider.Product.OBSERVATIONS"}
    )
    assert not checker._default_matches(
        _Provider.Product.OBSERVATIONS, {"enum": "Other.Product.OBSERVATIONS"}
    )
    assert (
        checker._check_runtime_signature(
            _Provider,
            {
                "parameters": [
                    {
                        "name": "product",
                        "kind": "keyword_only",
                        "annotation": "_Provider.Product",
                        "default": {"enum": "_Provider.Product.OBSERVATIONS"},
                    }
                ],
                "returns": "Self",
            },
            "Provider",
        )
        == []
    )


def test_current_evidence_interfaces_declare_specific_kinds_and_canonical_artifacts(
    checker: ModuleType,
    declarations: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    contract, release = declarations
    checker.validate_contract(contract, release)
    exports = contract["exports"]
    assert "SampleColumns" not in exports
    assert "TimeSeriesColumns" in exports
    for name, kind in [
        ("Samples", "SAMPLES"),
        ("Observations", "OBSERVATIONS"),
        ("Projections", "PROJECTIONS"),
        ("Layer", "LAYER"),
    ]:
        assert (
            exports[name]["methods"]["read"]["returns"]
            == f"ReadResult[EvidenceMetadata[Literal[EvidenceKind.{kind}]]]"
        )
        assert exports[name]["methods"]["open"]["returns"] == name
        assert exports[name]["methods"]["write"]["returns"] == name
    assert (
        exports["Inspection"]["properties"]["stations"]
        == "tuple[StationReference, ...]"
    )
    assert (
        exports["SeriesReference"]["properties"]["probability"] == "Probability | None"
    )
    assert exports["IBTrACS"]["constructor"]["parameters"][0]["default"] == {
        "enum": "IBTrACS.Agency.USA"
    }
    assert exports["JRC"]["nested_enums"]["Product"]["members"] == {
        "EFAS": "efas",
        "GLOFAS": "glofas",
    }
    assert exports["EDO"]["inherits"] == ["Observations"]


def test_literal_annotations_reference_typed_enum_choices(checker: ModuleType) -> None:
    checker._annotation(
        "EvidenceMetadata[Literal[EvidenceKind.SAMPLES]]",
        {"EvidenceMetadata", "Literal", "EvidenceKind", "EvidenceKind.SAMPLES"},
        "evidence",
    )
    with pytest.raises(checker.ContractError, match="typed choices"):
        checker._annotation("Literal['samples']", {"Literal"}, "evidence")
