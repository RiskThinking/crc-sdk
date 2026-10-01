"""Schema 1.3 metadata additions stay backward compatible."""

import json

import pytest
from pydantic import ValidationError

from crc_sdk.registry import (
    EligibilityDefaults,
    HazardRegistry,
    HazardSpec,
    public_registry,
)
from crc_sdk.schema import CANONICAL_HAZARD_SCHEMA_VERSION, hazard_fields_for_version
from crc_sdk.types import (
    EnsembleDescriptor,
    HazardDatasetMetadata,
    SourceProvenance,
    TemporalWindow,
)


def _metadata(**overrides: object) -> HazardDatasetMetadata:
    values: dict[str, object] = {
        "h3_resolution": 5,
        "value_unit": "mm",
        "value_semantics": "annual maximum",
        "producer": "tests",
        "source": SourceProvenance(provider="p", dataset="d"),
        "creation_version": "1",
    }
    values.update(overrides)
    return HazardDatasetMetadata(**values)  # type: ignore[arg-type]


def test_1_3_has_the_1_2_physical_columns() -> None:
    assert CANONICAL_HAZARD_SCHEMA_VERSION == "1.3"
    assert hazard_fields_for_version("1.3") == hazard_fields_for_version("1.2")


def test_1_3_round_trips_every_new_field() -> None:
    metadata = _metadata(
        probability_semantics="annual_exceedance",
        source_return_period_convention="poisson",
        temporal_window=TemporalWindow(
            start_year=2041,
            end_year=2060,
            baseline=(1995, 2014),
            calendar="noleap",
            minimum_complete_years=18,
        ),
        ensemble=EnsembleDescriptor(
            pooling="single_member", models=("ACCESS-CM2",), scenario="ssp245"
        ),
        source=SourceProvenance(
            provider="p",
            dataset="d",
            licence="CC-BY-4.0",
            attribution="Someone",
            retrieved_at="2026-10-01T12:00:00Z",
            checksum="abc",
        ),
    )
    restored = HazardDatasetMetadata.from_json_bytes(metadata.to_json_bytes())
    assert restored == metadata
    assert (
        restored.temporal_window is not None
        and restored.temporal_window.horizon == 2050
    )


def test_existing_1_2_metadata_reads_unchanged() -> None:
    legacy = json.loads(_metadata(schema_version="1.2").to_json_bytes())
    for key in (
        "probability_semantics",
        "source_return_period_convention",
        "temporal_window",
        "ensemble",
    ):
        del legacy[key]
    for key in ("licence", "attribution", "retrieved_at", "checksum"):
        del legacy["source"][key]
    restored = HazardDatasetMetadata.model_validate_json(json.dumps(legacy))
    assert restored.schema_version == "1.2"
    assert restored.temporal_window is None and restored.ensemble is None


def test_new_fields_require_schema_1_3() -> None:
    with pytest.raises(ValidationError, match="require schema 1.3"):
        _metadata(schema_version="1.2", probability_semantics="annual_exceedance")
    with pytest.raises(ValidationError, match="require schema 1.3"):
        _metadata(
            schema_version="1.2",
            source=SourceProvenance(provider="p", dataset="d", licence="MIT"),
        )


def test_temporal_window_validation() -> None:
    with pytest.raises(ValidationError, match="require start_year"):
        TemporalWindow()
    with pytest.raises(ValidationError, match="precede"):
        TemporalWindow(start_year=2060, end_year=2041)
    with pytest.raises(ValidationError, match="window length"):
        TemporalWindow(start_year=2041, end_year=2045, minimum_complete_years=6)
    with pytest.raises(ValidationError, match="reference_year"):
        TemporalWindow(kind="time_invariant")
    constant = TemporalWindow(kind="time_invariant", reference_year=2025)
    assert constant.horizon == 2025
    assert TemporalWindow(start_year=2010, end_year=2010).horizon == 2010


def test_ensemble_validation() -> None:
    with pytest.raises(ValidationError, match="single_member"):
        EnsembleDescriptor(pooling="single_member", models=("a", "b"))
    with pytest.raises(ValidationError, match="positive"):
        EnsembleDescriptor(pooling="pooled", members=0)
    assert EnsembleDescriptor(pooling="pooled", models=33).models == 33
    assert EnsembleDescriptor().pooling == "unknown"


def test_retrieved_at_must_be_iso() -> None:
    with pytest.raises(ValidationError, match="ISO 8601"):
        SourceProvenance(provider="p", dataset="d", retrieved_at="yesterday")


def test_registry_aliases_and_clashes() -> None:
    registry = public_registry()
    assert registry.resolve("Rx1day").name == "rx1day"
    assert "TXx" in registry and "nope" not in registry
    assert registry.eligibility("txx") == EligibilityDefaults()
    assert registry.eligibility("rx1day") is None
    with pytest.raises(ValueError, match="already registered"):
        registry.register(HazardSpec("rx1day", "mm", "x"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(HazardSpec("other", "mm", "x", aliases=("Rx1day",)))
    registry.register(HazardSpec("rx1day", "mm/d", "x"), replace=True)
    assert registry.resolve("rx1day").unit == "mm/d"
    assert "Rx1day" not in registry
    private = HazardRegistry()
    private.register(HazardSpec("fwi", "dimensionless", "annual maximum FWI"))
    merged = public_registry().merged(private)
    assert "fwi" in merged and "txx" in merged
    with pytest.raises(KeyError):
        merged.resolve("unknown")


def test_1_3_fit_provenance_requires_schema_1_3() -> None:
    from crc_sdk.types import CurveFitProvenance

    def fit(**extra: object) -> CurveFitProvenance:
        return CurveFitProvenance(
            families=("gumbel_r",),
            atom_policy="none",
            on_fit_failure="raise",
            **extra,  # type: ignore[arg-type]
        )

    for extra in (
        {"input_kind": "samples"},
        {"lower_bound": 0.0},
        {"platform": "Linux-x86_64"},
        {"crc_framework_version": "0.2.6"},
        {"sample_resampling": 11},
        {"initialization": "lmoments"},
        {"method": "sample_mle"},
    ):
        with pytest.raises(ValidationError, match="require schema 1.3"):
            _metadata(schema_version="1.2", fitting=fit(**extra))
    assert _metadata(schema_version="1.2", fitting=fit()).fitting is not None
    assert _metadata(fitting=fit(input_kind="samples")).schema_version == "1.3"


def test_registry_replace_never_takes_another_specs_labels() -> None:
    registry = public_registry()
    with pytest.raises(ValueError, match="already registered for 'rx1day'"):
        registry.register(
            HazardSpec("other", "mm", "x", aliases=("Rx1day",)), replace=True
        )
    with pytest.raises(ValueError, match="already registered for 'rx1day'"):
        registry.register(HazardSpec("Rx1day", "mm", "x"), replace=True)
    # Nothing was changed by the refused registrations.
    assert registry.resolve("Rx1day").name == "rx1day"
    assert "other" not in registry
    registry.unregister("rx1day")
    assert "Rx1day" not in registry and "rx5day" in registry
    # Replacing a spec re-points only its own aliases.
    registry.register(HazardSpec("a", "m", "x", aliases=("A",)))
    registry.register(HazardSpec("a", "m", "y", aliases=("A2",)), replace=True)
    assert "A" not in registry and registry.resolve("A2").value_semantics == "y"
