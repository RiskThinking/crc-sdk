"""Optional exact catalogs retain science without controlling user data."""

from __future__ import annotations

import json
from typing import cast

import crc_framework
import pytest

from crc_sdk import Catalog, Scenario, Variable, scientific_identity


def test_catalog_is_optional_and_exact_without_aliases_or_global_mutation() -> None:
    variable = Variable("My Variable")
    catalog = Catalog(namespace="user", revision="v1", variables=[variable])
    assert catalog.variable("My Variable") is variable
    assert catalog.variables == (variable,)
    assert catalog.scenarios == ()
    assert catalog.presets == {}
    for id in ("my variable", "MY VARIABLE", "unknown"):
        with pytest.raises(KeyError):
            catalog.variable(id)
    with pytest.raises(ValueError):
        catalog.variable(" My Variable")
    with pytest.raises(KeyError):
        Catalog(namespace="other", revision="v1").variable(variable.id)
    assert Variable("not-in-any-catalog").id == "not-in-any-catalog"


def test_catalog_crc_is_a_local_exact_convenience_without_implicit_policies() -> None:
    catalog = Catalog.crc()
    assert catalog.variable("crc:rx1day").measurement is None
    assert catalog.scenario("ssp585").id == "ssp585"
    assert catalog.presets == {}
    assert len(catalog.variables) == 17
    with pytest.raises(KeyError):
        catalog.variable("rx1day")
    with pytest.raises(KeyError):
        catalog.scenario("SSP585")
    with pytest.raises(KeyError):
        catalog.preset("crc:rx1day")


def test_catalog_presets_are_exact_framework_specs_and_round_trip() -> None:
    specification = crc_framework.SampleFit(
        family=crc_framework.DistributionFamily.GUMBEL_RIGHT,
        estimator=crc_framework.SampleEstimator.L_MOMENTS,
    )
    catalog = Catalog(
        namespace="user", revision="v1", presets={"observations": specification}
    )
    assert catalog.preset("observations") is specification
    reopened = Catalog.from_json(catalog.to_json())
    assert isinstance(reopened.preset("observations"), crc_framework.SampleFit)
    assert reopened.preset("observations").to_json() == specification.to_json()
    assert reopened.sha256 == catalog.sha256
    with pytest.raises(TypeError, match="framework"):
        Catalog(
            namespace="user",
            revision="v1",
            presets={"wrong": cast(crc_framework.FitSpec, {"family": "gumbel_r"})},
        )


def test_catalog_round_trips_open_custom_science_and_labels() -> None:
    measurement = crc_framework.Measurement(
        quantity="custom", unit="custom-unit", fields={"qualifier": {"duration": 2}}
    )
    variable = Variable("用户:指标", measurement=measurement, label="A user variable")
    scenario = Scenario("user-scenario", definition={"delta": 1.5}, label="Display")
    catalog = Catalog(
        namespace="user", revision="one", variables=[variable], scenarios=[scenario]
    )
    reopened = Catalog.from_json(catalog.to_json())
    assert scientific_identity(reopened.variable(variable.id)) == scientific_identity(
        variable
    )
    assert reopened.variable(variable.id).label == variable.label
    assert reopened.to_json() == catalog.to_json()


def test_composition_ignores_display_labels_but_rejects_scientific_conflicts() -> None:
    measurement = crc_framework.Measurement(quantity="depth", unit="metre")
    first = Catalog(
        namespace="a",
        revision="v1",
        variables=[Variable("depth", measurement=measurement, label="Left")],
    )
    same_science = Catalog(
        namespace="b",
        revision="v2",
        variables=[
            Variable("depth", measurement=measurement, label="Right"),
            Variable("other"),
        ],
    )
    merged = first.merged(same_science)
    assert merged.variable("depth").label == "Left"
    assert merged.variable("other").id == "other"
    assert first.variables == (first.variable("depth"),)
    conflicting = Catalog(
        namespace="b",
        revision="v2",
        variables=[
            Variable(
                "depth",
                measurement=crc_framework.Measurement(quantity="depth", unit="feet"),
            )
        ],
    )
    with pytest.raises(ValueError, match="conflicting scientific"):
        first.merged(conflicting)
    assert first.merged(first).to_json() == first.to_json()


def test_scenario_science_and_fit_presets_conflicts_are_not_silently_overwritten() -> (
    None
):
    first = Catalog(
        namespace="a", revision="v1", scenarios=[Scenario("s", definition={"delta": 1})]
    )
    second = Catalog(
        namespace="b", revision="v2", scenarios=[Scenario("s", definition={"delta": 2})]
    )
    with pytest.raises(ValueError, match="conflicting scientific"):
        first.merged(second)
    a = Catalog(
        namespace="a",
        revision="v1",
        presets={
            "fit": crc_framework.SampleFit(
                family=crc_framework.DistributionFamily.GUMBEL_RIGHT
            )
        },
    )
    b = Catalog(
        namespace="b",
        revision="v1",
        presets={
            "fit": crc_framework.SampleFit(
                family=crc_framework.DistributionFamily.GUMBEL_LEFT
            )
        },
    )
    with pytest.raises(ValueError, match="conflicting fitting"):
        a.merged(b)


def test_catalog_and_its_collections_are_immutable_snapshots() -> None:
    variables = [Variable("v")]
    catalog = Catalog(namespace="user", revision="v1", variables=variables)
    variables.append(Variable("later"))
    assert len(catalog.variables) == 1
    with pytest.raises(AttributeError, match="immutable"):
        catalog.namespace = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        cast(dict[str, crc_framework.FitSpec], catalog.presets)["new"] = (
            crc_framework.SampleFit(
                family=crc_framework.DistributionFamily.GUMBEL_RIGHT
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        Catalog(
            namespace="user", revision="v1", variables=[Variable("v"), Variable("v")]
        )


def test_catalog_rejects_aliases_unknown_fields_and_unsupported_versions() -> None:
    payload = json.loads(Catalog(namespace="user", revision="v1").to_json())
    for field in ("aliases", "preferred_families", "legacy_unknown"):
        with pytest.raises(ValueError, match="invalid fields"):
            Catalog.from_json(json.dumps({**payload, field: {}}))
    with pytest.raises(ValueError, match="taxonomy schema"):
        Catalog.from_json(json.dumps({**payload, "schema_version": "2.0"}))
    with pytest.raises(ValueError, match="duplicate"):
        Catalog.from_json('{"namespace":"a","namespace":"b"}')


def test_packaged_taxonomy_schema_and_fixtures_match_the_consumer() -> None:
    from importlib.resources import files

    import jsonschema
    from referencing import Registry, Resource

    from crc_sdk._catalog import _schema_document

    resources = files("crc_sdk").joinpath("contracts", "taxonomy")
    schema = json.loads(resources.joinpath("taxonomy-1.0.schema.json").read_text())
    assert schema == _schema_document()
    jsonschema.Draft202012Validator.check_schema(schema)
    framework = files("crc_framework").joinpath("contracts")
    registry: Registry = Registry()
    for directory in ("scientific", "numerical"):
        for path in framework.joinpath(directory).iterdir():
            if path.name.endswith(".schema.json"):
                value = json.loads(path.read_text())
                registry = registry.with_resource(
                    value["$id"], Resource.from_contents(value)
                )
    validator = jsonschema.Draft202012Validator(schema, registry=registry)
    fixtures = json.loads(resources.joinpath("fixtures.json").read_text())
    for case in fixtures["accepted"]:
        validator.validate(case["document"])
        Catalog.from_json(json.dumps(case["document"]))
    for case in fixtures["rejected"]:
        assert not validator.is_valid(case["document"]), case["name"]
        with pytest.raises((TypeError, ValueError)):
            Catalog.from_json(json.dumps(case["document"]))
