"""Validate the declared Python interface and installed implementation.

Static validation reads only contract files and package metadata. Runtime
validation is a release requirement and checks the complete declared surface.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import json
import re
import sys
import tomllib
from collections.abc import Mapping
from enum import Enum
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "crc_sdk" / "contracts" / "api"
BUILTINS = {
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "None",
    "tuple",
    "list",
    "dict",
    "type",
    "BaseException",
    "Self",
    "Mapping",
    "Sequence",
    "Iterator",
    "Iterable",
    "Callable",
    "AbstractContextManager",
    "Enum",
    "Path",
}
PARAMETER_KINDS = {
    "positional_only": inspect.Parameter.POSITIONAL_ONLY,
    "positional_or_keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "keyword_only": inspect.Parameter.KEYWORD_ONLY,
}
EXPORT_KINDS = {"class", "enum", "exception", "protocol", "function"}


class ContractError(ValueError):
    """A declared or implemented interface differs from the release contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_contracts(
    directory: Path | Traversable = CONTRACTS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result

    values = [
        json.loads(
            directory.joinpath(name).read_text(), object_pairs_hook=unique_object
        )
        for name in ("public-api.json", "release.json")
    ]
    return values[0], values[1]


def _annotation(expression: object, known: set[str], where: str) -> None:
    require(isinstance(expression, str) and bool(expression), f"{where}: missing type")
    assert isinstance(expression, str)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ContractError(f"{where}: invalid annotation {expression!r}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            require(node.id not in {"Any", "object"}, f"{where}: untyped interface")
            require(node.id in known, f"{where}: unknown type {node.id}")
        elif isinstance(node, ast.Attribute):
            require(
                ast.unparse(node) in known,
                f"{where}: unknown qualified type {ast.unparse(node)}",
            )
        elif isinstance(node, ast.Constant):
            require(
                node.value is None or node.value is Ellipsis,
                f"{where}: annotations must reference typed choices",
            )
        else:
            require(
                isinstance(
                    node,
                    (
                        ast.Expression,
                        ast.Attribute,
                        ast.Subscript,
                        ast.BinOp,
                        ast.BitOr,
                        ast.Tuple,
                        ast.List,
                        ast.Load,
                    ),
                ),
                f"{where}: unsupported annotation expression",
            )


def _signature(value: Mapping[str, Any], known: set[str], where: str) -> None:
    require(
        set(value) <= {"parameters", "returns", "binding", "available_for"},
        f"{where}: unknown signature field",
    )
    require(
        value.get("binding", "instance") in {"instance", "class", "static"},
        f"{where}: invalid binding",
    )
    parameters = value.get("parameters")
    require(isinstance(parameters, list), f"{where}: parameters must be declared")
    assert isinstance(parameters, list)
    names: set[str] = set()
    keyword_seen = False
    positional_default = False
    for parameter in parameters:
        require(isinstance(parameter, dict), f"{where}: invalid parameter")
        require(
            set(parameter) <= {"name", "kind", "annotation", "default"},
            f"{where}: unknown parameter field",
        )
        name = parameter.get("name")
        require(isinstance(name, str) and name.isidentifier(), f"{where}: invalid name")
        require(
            name not in names and name not in {"self", "cls"},
            f"{where}: duplicate or implicit receiver {name}",
        )
        names.add(name)
        kind = parameter.get("kind")
        require(kind in PARAMETER_KINDS, f"{where}.{name}: invalid parameter kind")
        keyword_seen = keyword_seen or kind == "keyword_only"
        require(
            not keyword_seen or kind == "keyword_only",
            f"{where}: positional argument after keyword-only arguments",
        )
        if kind != "keyword_only":
            require(
                not positional_default or "default" in parameter,
                f"{where}: required positional argument after default",
            )
            positional_default = positional_default or "default" in parameter
        _annotation(parameter.get("annotation"), known, f"{where}.{name}")
        if "default" in parameter:
            default = parameter["default"]
            if isinstance(default, dict):
                require(
                    set(default) in ({"enum"}, {"sentinel"}),
                    f"{where}.{name}: invalid structured default",
                )
                if "enum" in default:
                    reference = default["enum"]
                    require(
                        isinstance(reference, str) and "." in reference,
                        f"{where}.{name}: invalid enum default",
                    )
                    require(reference in known, f"{where}.{name}: unknown enum default")
                else:
                    require(
                        default["sentinel"] == "omitted",
                        f"{where}.{name}: unknown omission sentinel",
                    )
            else:
                require(
                    default is None
                    or isinstance(default, (str, int, float, bool, list)),
                    f"{where}.{name}: unsupported default",
                )
    _annotation(value.get("returns"), known, f"{where}.return")


def validate_contract(contract: dict[str, Any], release: dict[str, Any]) -> None:
    require(
        set(contract)
        <= {
            "schema_version",
            "package",
            "canonical_module",
            "exports",
            "type_aliases",
            "external_symbols",
            "external_modules",
            "external_types",
            "input_protocols",
            "generic_parameters",
            "opaque_native_types",
            "naming",
            "semantics",
            "removals",
            "provider_indicators",
            "catalog_reference_rules",
        },
        "unknown API contract field",
    )
    require(contract.get("schema_version") == "1.0", "unsupported API contract schema")
    require(contract.get("package") == "crc-sdk", "wrong package owner")
    require(contract.get("canonical_module") == "crc_sdk", "wrong canonical import")
    require(release.get("schema_version") == "1.0", "unsupported release ledger schema")
    require(
        release.get("packages")
        == {
            "framework": "1.0.0",
            "sdk": "1.0.0",
            "framework_requirement": "crc-framework>=1.0.0,<2",
        },
        "package versions differ from the single release ledger",
    )
    require(
        release.get("formats")
        == {
            "dataset": "2.0",
            "evaluation": "2.0",
            "collection": "1.0",
            "taxonomy": "1.0",
            "numerical": "1.0",
            "impact": "2.0",
            "evidence": "1.0",
        },
        "format versions differ from the single release ledger",
    )
    exports = contract.get("exports")
    require(isinstance(exports, dict) and bool(exports), "public allowlist is empty")
    assert isinstance(exports, dict)
    require(list(exports) == sorted(exports), "public allowlist must be sorted")
    require(
        all(name.isidentifier() and not name.startswith("_") for name in exports),
        "invalid canonical public name",
    )
    aliases = contract.get("type_aliases", {})
    external = contract.get("external_symbols", {})
    require(
        set(external) == {"crc_framework"}, "numerical types must be framework-owned"
    )
    modules = contract.get("external_modules", [])
    known = (
        BUILTINS
        | set(exports)
        | set(aliases)
        | set(external)
        | set(modules)
        | set(contract.get("generic_parameters", []))
        | set(contract.get("input_protocols", {}))
        | set(contract.get("opaque_native_types", []))
    )
    known.update(contract.get("external_types", []))
    known.update(
        f"{module}.{name}" for module, names in external.items() for name in names
    )
    for name, value in exports.items():
        known.update(f"{name}.{member}" for member in value.get("members", {}))
        for nested_name, nested in value.get("nested_enums", {}).items():
            known.add(f"{name}.{nested_name}")
            known.update(
                f"{name}.{nested_name}.{member}" for member in nested.get("members", {})
            )
    for name, expression in aliases.items():
        require(
            name.isidentifier() and name not in exports,
            f"{name}: internal alias must not create another public export",
        )
        _annotation(expression, known, f"type_aliases.{name}")
    for name, protocol in contract.get("input_protocols", {}).items():
        for method_name, signature in protocol.get("methods", {}).items():
            _signature(signature, known, f"{name}.{method_name}")
        for field_name, expression in protocol.get("properties", {}).items():
            _annotation(expression, known, f"{name}.{field_name}")
    for name, value in exports.items():
        require(isinstance(value, dict), f"{name}: declaration must be an object")
        require(value.get("kind") in EXPORT_KINDS, f"{name}: invalid declaration kind")
        require(
            set(value)
            <= {
                "kind",
                "constructor",
                "methods",
                "properties",
                "inherits",
                "members",
                "nested_enums",
                "generic_parameters",
            },
            f"{name}: unknown declaration field",
        )
        if value["kind"] == "enum":
            members = value.get("members")
            require(isinstance(members, dict) and bool(members), f"{name}: empty enum")
            assert isinstance(members, dict)
            require(
                all(key.isidentifier() and key == key.upper() for key in members),
                f"{name}: enum names must be uppercase identifiers",
            )
            encoded = [json.dumps(item, sort_keys=True) for item in members.values()]
            require(
                len(encoded) == len(set(encoded)), f"{name}: enum aliases are forbidden"
            )
        for parent in value.get("inherits", []):
            require(
                parent in exports and parent != name,
                f"{name}: unknown or recursive base",
            )
        if "constructor" in value:
            _signature(value["constructor"], known, f"{name}.__init__")
        methods = value.get("methods", {})
        properties = value.get("properties", {})
        require(
            not set(methods) & set(properties),
            f"{name}: method/property name collision",
        )
        for method_name, signature in methods.items():
            require(method_name.isidentifier(), f"{name}: invalid method name")
            _signature(signature, known, f"{name}.{method_name}")
        for property_name, annotation in properties.items():
            require(property_name.isidentifier(), f"{name}: invalid property name")
            _annotation(annotation, known, f"{name}.{property_name}")
        for nested_name, nested in value.get("nested_enums", {}).items():
            require(
                nested_name.isidentifier() and nested.get("kind") == "enum",
                f"{name}: invalid nested enum",
            )
            members = nested.get("members", {})
            require(
                bool(members) and len(members) == len(set(members.values())),
                f"{name}.{nested_name}: empty enum or aliases",
            )
    # Resolve inheritance now so cycles cannot hide a missing contract.
    for name in exports:
        expanded_members(exports, name)
    removed = contract.get("removals", {})
    for field in ("symbols", "modules", "schema_fields", "wire_formats"):
        values = removed.get(field)
        require(
            isinstance(values, list) and bool(values),
            f"missing removal inventory: {field}",
        )
        assert isinstance(values, list)
        require(len(values) == len(set(values)), f"duplicate removal: {field}")
    require(
        not {f"crc_sdk.{name}" for name in exports} & set(removed["symbols"]),
        "a removed symbol appears in the public allowlist",
    )
    for name, value in exports.items():
        methods, _ = expanded_members(exports, name)
        require(
            not set(methods) & set(removed_methods(contract, name)),
            f"{name}: removed method remains declared",
        )
        require(
            name not in external.get("crc_framework", []),
            f"{name}: SDK must not re-export framework-owned types",
        )
    require(
        contract.get("naming", {}).get("one_name_per_concept") is True,
        "canonical naming rule is required",
    )
    for provider, inventory in contract.get("provider_indicators", {}).items():
        members = exports[provider]["nested_enums"]["Indicator"]["members"]
        require(
            set(members) == set(inventory["members"]),
            f"{provider}: scientific indicator inventory differs from enum",
        )
        require(
            len({item["native_indicator"] for item in inventory["members"].values()})
            == len(members),
            f"{provider}: duplicate native indicator mapping",
        )


def removed_methods(contract: Mapping[str, Any], name: str) -> set[str]:
    removals = contract["removals"]["methods"]
    require(isinstance(removals, dict), "removed methods must be scoped by owner")
    result = set(removals.get("*", ()))
    result.update(removals.get(name, ()))
    return result


def expanded_members(
    exports: Mapping[str, Any],
    name: str,
    ancestry: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    require(
        name not in ancestry, f"inheritance cycle: {' -> '.join((*ancestry, name))}"
    )
    methods: dict[str, Any] = {}
    properties: dict[str, str] = {}
    declaration = exports[name]
    for parent in declaration.get("inherits", []):
        inherited_methods, inherited_properties = expanded_members(
            exports, parent, (*ancestry, name)
        )
        methods.update(inherited_methods)
        properties.update(inherited_properties)
    methods.update(declaration.get("methods", {}))
    properties.update(declaration.get("properties", {}))
    require(
        not set(methods) & set(properties), f"{name}: inherited member name collision"
    )
    return methods, properties


def check_versions(release: Mapping[str, Any], root: Path = ROOT) -> None:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    require(
        project["version"] == release["packages"]["sdk"],
        "pyproject.toml version differs from release ledger",
    )
    requirements = [
        item for item in project["dependencies"] if item.startswith("crc-framework")
    ]
    require(
        requirements == [release["packages"]["framework_requirement"]],
        "framework dependency differs from release ledger",
    )


def _normalized_annotation(value: Any) -> str:
    if value is inspect.Signature.empty:
        return "<missing>"
    if isinstance(value, str):
        return (
            re.sub(r"\s+", "", value)
            .replace("typing.", "")
            .replace("crc_framework.", "")
            .replace("crc_sdk.", "")
            .replace("pa.", "pyarrow.")
        )
    if value is None:
        return "None"
    return re.sub(r"\s+", "", inspect.formatannotation(value)).replace("typing.", "")


def _default_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "enum" in expected:
            return (
                isinstance(actual, Enum)
                and f"{type(actual).__name__}.{actual.name}" == expected["enum"]
            )
        return actual is Ellipsis
    if isinstance(expected, list):
        return bool(actual == expected or actual == tuple(expected))
    return bool(type(actual) is type(expected) and actual == expected)


def _check_runtime_signature(
    value: Any, expected: Mapping[str, Any], where: str
) -> list[str]:
    errors = []
    try:
        actual = inspect.signature(value)
    except (TypeError, ValueError) as error:
        return [f"{where}: signature unavailable: {error}"]
    parameters = list(actual.parameters.values())
    if parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]
    expected_parameters = expected["parameters"]
    if [item.name for item in parameters] != [
        item["name"] for item in expected_parameters
    ]:
        errors.append(f"{where}: parameter names differ")
        return errors
    for parameter, declaration in zip(parameters, expected_parameters, strict=True):
        if parameter.kind != PARAMETER_KINDS[declaration["kind"]]:
            errors.append(f"{where}.{parameter.name}: parameter kind differs")
        if _normalized_annotation(parameter.annotation) != _normalized_annotation(
            declaration["annotation"]
        ):
            errors.append(f"{where}.{parameter.name}: annotation differs")
        if "default" in declaration:
            if not _default_matches(parameter.default, declaration["default"]):
                errors.append(f"{where}.{parameter.name}: default differs")
        elif parameter.default is not inspect.Parameter.empty:
            errors.append(f"{where}.{parameter.name}: unexpected default")
    constructor_return = inspect.isclass(value) and actual.return_annotation in {
        None,
        "None",
    }
    if not constructor_return and _normalized_annotation(
        actual.return_annotation
    ) != _normalized_annotation(expected["returns"]):
        errors.append(f"{where}: return annotation differs")
    return errors


def import_runtime(package_root: Path | None = None) -> Any:
    if package_root is not None:
        package_root = package_root.resolve()
        # The checker itself lives outside the package import path. Explicitly
        # choose the installation under review instead of a source checkout.
        sys.path[:] = [
            str(package_root),
            *[
                entry
                for entry in sys.path
                if Path(entry or ".").resolve() != ROOT
                and Path(entry or ".").resolve() != package_root
            ],
        ]
    return importlib.import_module("crc_sdk")


def check_runtime(
    contract: Mapping[str, Any],
    release: Mapping[str, Any],
    package_root: Path | None = None,
) -> None:
    if package_root is not None:
        package_root = package_root.resolve()
    package = import_runtime(package_root)
    installed_contract, installed_release = load_contracts(
        files(package).joinpath("contracts", "api")
    )
    validate_contract(installed_contract, installed_release)
    require(
        installed_contract == contract and installed_release == release,
        "installed contract files differ from the source release declarations",
    )
    errors: list[str] = []
    if package_root is not None and not Path(
        package.__file__ or ""
    ).resolve().is_relative_to(package_root):
        errors.append("crc_sdk: import did not come from --package-root")
    if importlib.metadata.version("crc-sdk") != release["packages"]["sdk"]:
        errors.append("crc-sdk: installed package version differs")
    framework_version = importlib.metadata.version("crc-framework")
    if framework_version != release["packages"]["framework"]:
        errors.append(
            "crc-framework: installed package version differs from paired release"
        )
    expected_exports = set(contract["exports"])
    actual_exports = getattr(package, "__all__", ())
    if len(actual_exports) != len(set(actual_exports)):
        errors.append("crc_sdk: duplicate exports")
    if set(actual_exports) != expected_exports:
        missing = sorted(expected_exports - set(actual_exports))
        unexpected = sorted(set(actual_exports) - expected_exports)
        errors.append(
            f"crc_sdk: exports differ (missing={missing}, unexpected={unexpected})"
        )
    undeclared_attributes = {
        name
        for name in vars(package)
        if not name.startswith("_") and name not in expected_exports
    }
    if undeclared_attributes:
        errors.append(
            f"crc_sdk: undeclared public attributes {sorted(undeclared_attributes)}"
        )
    for qualified in contract["removals"]["symbols"]:
        module_name, _, name = qualified.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, name):
            errors.append(f"{qualified}: removed symbol remains accessible")
    for module_name in contract["removals"]["modules"]:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        errors.append(f"{module_name}: removed public module remains importable")
    for name, declaration in contract["exports"].items():
        value = getattr(package, name, None)
        if value is None:
            continue  # Exact export errors above already identify missing values.
        where = f"crc_sdk.{name}"
        if declaration["kind"] == "enum":
            if not inspect.isclass(value) or not issubclass(value, Enum):
                errors.append(f"{where}: enum implementation required")
            elif {
                key: item.value for key, item in value.__members__.items()
            } != declaration["members"]:
                errors.append(f"{where}: enum members or values differ")
            continue
        if not inspect.isclass(value):
            errors.append(f"{where}: class implementation required")
            continue
        if declaration["kind"] == "protocol" and not getattr(
            value, "_is_protocol", False
        ):
            errors.append(f"{where}: protocol implementation required")
        if declaration["kind"] == "exception" and not issubclass(value, Exception):
            errors.append(f"{where}: exception implementation required")
        if "constructor" in declaration:
            errors.extend(
                _check_runtime_signature(value, declaration["constructor"], where)
            )
        methods, properties = expanded_members(contract["exports"], name)
        for method_name, signature in methods.items():
            method_value = getattr(value, method_name, None)
            if not callable(method_value):
                errors.append(f"{where}.{method_name}: method missing")
                continue
            raw = inspect.getattr_static(value, method_name)
            binding = (
                "class"
                if isinstance(raw, classmethod)
                else "static"
                if isinstance(raw, staticmethod)
                else "instance"
            )
            if binding != signature.get("binding", "instance"):
                errors.append(f"{where}.{method_name}: binding differs")
            errors.extend(
                _check_runtime_signature(
                    method_value, signature, f"{where}.{method_name}"
                )
            )
        annotations: dict[str, Any] = {}
        for base in reversed(value.__mro__):
            annotations.update(getattr(base, "__annotations__", {}))
        for property_name, annotation in properties.items():
            descriptor = inspect.getattr_static(value, property_name, None)
            actual_type = annotations.get(property_name, inspect.Signature.empty)
            if isinstance(descriptor, property) and descriptor.fget is not None:
                actual_type = inspect.signature(descriptor.fget).return_annotation
            if _normalized_annotation(actual_type) != _normalized_annotation(
                annotation
            ):
                errors.append(f"{where}.{property_name}: property/field type differs")
        allowed_members = (
            set(methods) | set(properties) | set(declaration.get("nested_enums", {}))
        )
        for base in value.__mro__:
            if not base.__module__.startswith("crc_sdk"):
                continue
            own_members = set(vars(base)) | set(getattr(base, "__annotations__", {}))
            unexpected_members = {
                member
                for member in own_members
                if not member.startswith("_") and member not in allowed_members
            }
            if unexpected_members:
                errors.append(
                    f"{where}: undeclared public members {sorted(unexpected_members)}"
                )
        for forbidden in removed_methods(contract, name):
            if hasattr(value, forbidden):
                errors.append(f"{where}.{forbidden}: removed method remains accessible")
        for enum_name, enum_spec in declaration.get("nested_enums", {}).items():
            enum_value = getattr(value, enum_name, None)
            if not inspect.isclass(enum_value) or not issubclass(enum_value, Enum):
                errors.append(f"{where}.{enum_name}: nested enum missing")
            elif {
                key: item.value for key, item in enum_value.__members__.items()
            } != enum_spec["members"]:
                errors.append(f"{where}.{enum_name}: nested enum members differ")
    require(
        not errors,
        "runtime interface is not release-ready:\n"
        + "\n".join(f"- {item}" for item in errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="require installed implementation conformance",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        help="site-packages directory to inspect without source shadowing",
    )
    arguments = parser.parse_args()
    try:
        contract, release = load_contracts()
        validate_contract(contract, release)
        check_versions(release)
        if arguments.package_root is not None and not arguments.runtime:
            raise ContractError("--package-root requires --runtime")
        if arguments.runtime:
            check_runtime(contract, release, arguments.package_root)
    except (
        ContractError,
        ImportError,
        OSError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        print(error, file=sys.stderr)
        return 1
    print(
        "Python interface and release contract verified"
        + (" against runtime" if arguments.runtime else " (declarations only)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
