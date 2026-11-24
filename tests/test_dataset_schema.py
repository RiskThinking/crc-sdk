"""Canonical scientific rows agree across Python, Arrow, JSON, and Parquet."""

from __future__ import annotations

import json
import struct
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    DiagnosticSeverity,
    Distribution,
    DistributionFamily,
    FitError,
    FitRecord,
    Impact,
    Measurement,
    QuantileCurve,
    Ratio,
    SampleEstimator,
    SampleFit,
    Tail,
    TailCompletion,
    fit_samples,
)

from crc_sdk._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionDiagnostic,
    ExecutionReport,
    IdentityKind,
    IdentityRename,
    ReleaseReference,
    ResourceIdentity,
    RetainedFailure,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
    UnitConversion,
)
from crc_sdk._declarations import (
    DataSpec,
    Period,
    Probability,
    Scenario,
    SeriesSpec,
    Variable,
)
from crc_sdk._schema import (
    DatasetRow,
    dataset_arrow_schema,
    dataset_from_json,
    dataset_to_json,
    metadata_from_table,
    read_parquet,
    rows_to_table,
    table_to_rows,
    validate_rows,
    write_parquet,
)
from crc_sdk._schema.contract import dataset_json_schema

CELL = 594475228122316799
ROOT = Path(__file__).resolve().parents[1]


def metadata(
    variable: Variable | None = None,
    *,
    period: Period | None = None,
    scenario: Scenario | None = None,
    probability: Probability | None = None,
    supports: tuple[SupportKind, ...] = (SupportKind.CELL,),
) -> DatasetMetadata:
    variable = variable or Variable("user:depth")
    spec = DataSpec(
        [variable],
        series=[
            SeriesSpec(
                variable, period=period, scenario=scenario, probability=probability
            )
        ],
    )
    return DatasetMetadata(
        spec=spec,
        spatial=SpatialContract(supports=supports, index_resolution=4),
        report=ExecutionReport(
            complete=True, valid=True, counts=ExecutionCounts(requested=1, produced=1)
        ),
    )


def row(
    curve: Distribution | QuantileCurve | None = None,
    *,
    scenario_id: str | None = None,
    period_id: str | None = None,
    geometry: bytes | None = None,
    fit: FitRecord | None = None,
) -> DatasetRow:
    return DatasetRow(
        "user:depth",
        "support:station",
        CELL,
        curve=curve or Distribution.point_mass(1),
        scenario_id=scenario_id,
        period_id=period_id,
        geometry=geometry,
        fit=fit,
    )


def point(lon: float = 10, lat: float = 45, *, big_endian: bool = False) -> bytes:
    return struct.pack(
        ">BIdd" if big_endian else "<BIdd", 0 if big_endian else 1, 1, lon, lat
    )


def numerical_cases() -> list[Distribution | QuantileCurve]:
    basic = Distribution.parametric(
        DistributionFamily.GUMBEL_RIGHT, location=0, scale=1
    )
    measurement = Measurement(quantity="index", unit="1")
    response = Impact.sigmoid(
        midpoint=0, steepness=1, input=measurement, output=Ratio()
    ).pushforward(basic)
    return [
        basic,
        Distribution.empirical([0, 0, 1, 3]),
        Distribution.discrete([0, 1], [0.3, 0.7]),
        Distribution.histogram([0, 1, 3], [0.4, 0.6]),
        Distribution.point_mass(2),
        Distribution.hurdle(basic, atom_probability=0.3),
        QuantileCurve([0.9, 0.99], [1, 2]),
        QuantileCurve([0.9, 0.99], [1, 2]).complete(TailCompletion(lower=0, upper=3)),
        response,
    ]


@pytest.mark.parametrize("curve", numerical_cases())
def test_numerical_variants_round_trip_without_losing_scientific_choice(
    curve: Distribution | QuantileCurve,
) -> None:
    info = metadata()
    original = row(curve)
    table = rows_to_table([original], info)
    decoded = table_to_rows(table)
    assert decoded[0].curve is not None
    assert decoded[0].curve.to_json() == curve.to_json()
    assert metadata_from_table(table).to_json() == info.to_json()
    encoded = dataset_to_json([original], info)
    loaded_info, loaded_rows = dataset_from_json(encoded)
    assert loaded_info.to_json() == info.to_json()
    assert loaded_rows[0].curve is not None
    assert loaded_rows[0].curve.to_json() == curve.to_json()
    arrow_curve = table["curve"].to_pylist()[0]
    active = arrow_curve["kind"]
    assert all(
        value is None
        for name, value in arrow_curve.items()
        if name not in ("kind", active, "evidence")
    )


def test_only_current_row_and_metadata_names_are_written() -> None:
    document = json.loads(dataset_to_json([row()], metadata()))
    assert set(document) == {"metadata", "rows"}
    assert document["metadata"]["schema_version"] == "2.0"
    assert document["rows"][0]["scenario_id"] is None
    assert document["rows"][0]["period_id"] is None
    for name in (
        "horizon",
        "pathway",
        "hazard_name",
        "source_id",
        "value_unit",
        "row_key",
        "sort_order",
        "fitting",
        "source_probability_support",
    ):
        assert name not in document["metadata"]
        assert name not in document["rows"][0]
    assert document["rows"][0]["curve"] == {
        "schema_version": "1.0",
        "object": "distribution",
        "kind": "point_mass",
        "value": 1.0,
    }


def test_optional_dimensions_require_no_fabricated_scenario_or_period() -> None:
    info = metadata()
    assert info.spec.scenarios == () and info.spec.periods == ()
    assert validate_rows([row()], info).counts.produced == 1


def test_inclusive_period_and_annual_probability_round_trip() -> None:
    period = Period.window(2030, 2050)
    scenario = Scenario("user:planning", definition={"warming": 2.0})
    info = metadata(
        period=period,
        scenario=scenario,
        probability=Probability.annual(tail=Tail.UPPER),
    )
    original = row(
        QuantileCurve([0.9, 0.99], [1, 2]), period_id=period.id, scenario_id=scenario.id
    )
    parsed_info, parsed_rows = dataset_from_json(dataset_to_json([original], info))
    assert parsed_info.spec.periods[0].start == 2030
    assert parsed_info.spec.periods[0].end == 2050
    assert parsed_info.spec.periods[0].calendar is None
    assert parsed_rows[0].period_id == period.id


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2", "1.3", "3.0"])
def test_historical_or_unknown_dataset_versions_are_rejected(version: str) -> None:
    document = json.loads(dataset_to_json([row()], metadata()))
    document["metadata"]["schema_version"] = version
    with pytest.raises(ValueError, match="2.0"):
        dataset_from_json(json.dumps(document))


@pytest.mark.parametrize("location", ["metadata", "row", "spatial", "curve"])
def test_unknown_nested_fields_are_rejected(location: str) -> None:
    document = json.loads(dataset_to_json([row()], metadata()))
    target = {
        "metadata": document["metadata"],
        "row": document["rows"][0],
        "spatial": document["metadata"]["spatial"],
        "curve": document["rows"][0]["curve"],
    }[location]
    target["misspelled"] = "not accepted"
    with pytest.raises(ValueError):
        dataset_from_json(json.dumps(document))


def test_unknown_descriptor_references_and_duplicate_null_keys_are_rejected() -> None:
    info = metadata()
    with pytest.raises(ValueError, match="undeclared"):
        validate_rows([replace(row(), scenario_id="missing")], info)
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows([row(), row()], info)
    document = json.loads(dataset_to_json([row()], info))
    document["metadata"]["spec"]["series"][0]["variable_id"] = "missing"
    with pytest.raises(ValueError, match="dangling"):
        dataset_from_json(json.dumps(document))


@pytest.mark.parametrize("cell", [0, -1, 1, True, 2**64])
def test_invalid_cells_cannot_be_canonical_rows(cell: int) -> None:
    with pytest.raises(ValueError):
        replace(row(), cell_id=cell)


def test_declared_index_resolution_is_checked_without_changing_support() -> None:
    info = replace(
        metadata(),
        spatial=SpatialContract(supports=(SupportKind.CELL,), index_resolution=5),
    )
    with pytest.raises(ValueError, match="resolution"):
        rows_to_table([row()], info)


def test_source_geometry_is_required_and_support_identity_is_consistent() -> None:
    info = metadata(supports=(SupportKind.POINT,))
    with pytest.raises(ValueError, match="support contract"):
        validate_rows([row()], info)
    variable = Variable("other")
    spec = DataSpec(
        [Variable("user:depth"), variable],
        series=[SeriesSpec(Variable("user:depth")), SeriesSpec(variable)],
    )
    info = replace(
        info,
        spec=spec,
        report=ExecutionReport(
            complete=True, valid=True, counts=ExecutionCounts(requested=2, produced=2)
        ),
    )
    first = row(geometry=point())
    same = DatasetRow(
        "other",
        first.support_id,
        CELL,
        geometry=point(big_endian=True),
        curve=Distribution.point_mass(2),
    )
    validate_rows([first, same], info)
    with pytest.raises(ValueError, match="different source supports"):
        validate_rows([first, replace(same, geometry=point(lon=11))], info)
    with pytest.raises(ValueError):
        replace(first, geometry=point() + b"trailing")


def test_arrow_inactive_variants_and_irrelevant_gumbel_shape_are_rejected() -> None:
    table = rows_to_table([row(numerical_cases()[0])], metadata())
    values = table.to_pylist()
    values[0]["curve"]["point_mass"] = {"value": 9}
    invalid = pa.Table.from_pylist(values, schema=table.schema)
    with pytest.raises(ValueError, match="active Arrow"):
        table_to_rows(invalid)
    values[0]["curve"]["point_mass"] = None
    values[0]["curve"]["parametric"]["shape"] = 0
    invalid = pa.Table.from_pylist(values, schema=table.schema)
    with pytest.raises(ValueError):
        table_to_rows(invalid)


def test_arrow_exact_fields_are_required_and_metadata_cannot_be_replaced() -> None:
    table = rows_to_table([row()], metadata())
    with pytest.raises(ValueError, match="metadata is missing"):
        table_to_rows(table.replace_schema_metadata(None))
    with pytest.raises(ValueError, match="fields"):
        table_to_rows(table.append_column("horizon", pa.array([2050])))
    other = metadata(variable=Variable("other"))
    with pytest.raises(ValueError, match="disagree"):
        table_to_rows(table, other)


def test_successful_fit_and_retained_failure_keep_authoritative_records() -> None:
    fit = fit_samples(
        [1, 2, 3, 4, 5, 6, 7, 8],
        SampleFit(
            family=DistributionFamily.GUMBEL_RIGHT, estimator=SampleEstimator.L_MOMENTS
        ),
    )
    original = row(fit.distribution, fit=fit.record)
    decoded = table_to_rows(rows_to_table([original], metadata()))[0]
    assert decoded.fit is not None
    assert decoded.fit.to_json() == fit.record.to_json()
    with pytest.raises(FitError) as error:
        fit_samples([1] * 8, SampleFit(family=DistributionFamily.GUMBEL_RIGHT))
    failure = RetainedFailure.from_error(error.value)
    failed_row = DatasetRow("user:depth", "failed", CELL, failure=failure)
    failed_info = replace(
        metadata(),
        report=ExecutionReport(
            complete=True,
            valid=True,
            counts=ExecutionCounts(requested=1, produced=1, no_data=1),
        ),
    )
    report = validate_rows([failed_row], failed_info)
    assert report.counts.no_data == 1
    decoded_failure = table_to_rows(rows_to_table([failed_row], failed_info))[0].failure
    assert isinstance(decoded_failure, RetainedFailure)
    assert decoded_failure.record.to_json() == failure.record.to_json()
    assert decoded_failure.code == error.value.code
    with pytest.raises(ValueError, match="exactly one"):
        replace(failed_row, curve=Distribution.point_mass(1))
    with pytest.raises(ValueError, match="successfully fitted"):
        replace(original, curve=QuantileCurve([0.1, 0.9], [1, 2]))
    wrong_family = replace(
        original,
        curve=Distribution.parametric(
            DistributionFamily.GUMBEL_LEFT, location=0, scale=1
        ),
    )
    with pytest.raises(ValueError, match="representation contradicts"):
        validate_rows([wrong_family], metadata())


def test_source_provenance_and_report_round_trip_without_envelope_checksum() -> None:
    resource = ResourceIdentity(
        "s3://custom/data.parquet",
        "sha256:" + "a" * 64,
        license="CC-BY-4.0",
        attribution="Example scientists",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    source = SourceIdentity(
        "user",
        "observations",
        ReleaseReference("2026", "sha256:" + "b" * 64),
        (resource,),
        member="model:member:1",
    )
    report = ExecutionReport(
        True,
        None,
        diagnostics=(
            ExecutionDiagnostic(
                "incomplete_year",
                DiagnosticSeverity.WARNING,
                Stage.REDUCTION,
                "One incomplete year",
                source=resource.uri,
            ),
        ),
        counts=ExecutionCounts(requested=2, produced=1, incomplete_years=1),
        source_identities=(source,),
    )
    info = replace(metadata(), source=source, report=report)
    decoded, _ = dataset_from_json(dataset_to_json([row()], info))
    assert decoded.source is not None
    assert decoded.source.resources[0] == resource
    assert decoded.source.member == source.member
    assert decoded.report == report
    assert "checksum" not in json.loads(info.to_json())
    with pytest.raises(ValueError):
        ResourceIdentity("x", "not-a-content-hash")
    with pytest.raises(ValueError, match="timezone-aware"):
        ResourceIdentity("x", "sha256:" + "a" * 64, retrieved_at=datetime(2026, 1, 1))


def test_typed_transformations_are_immutable_and_match_final_measurement() -> None:
    variable = Variable(
        "user:depth", measurement=Measurement(quantity="depth", unit="m")
    )
    conversion = UnitConversion(
        variable_id=variable.id, source_unit="cm", target_unit="m", scale=0.01
    )
    rename = IdentityRename(
        dimension=IdentityKind.VARIABLE, mapping={"depth_old": variable.id}
    )
    info = replace(metadata(variable), provenance=(rename, conversion))
    decoded, _ = dataset_from_json(dataset_to_json([row()], info))
    assert decoded.provenance[1] == conversion
    with pytest.raises(TypeError):
        rename.mapping["new"] = "another"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        conversion.scale = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="target unit"):
        replace(info, provenance=(replace(conversion, target_unit="ft"),))


def test_parquet_round_trip_is_validated_and_does_not_overwrite_by_default(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.parquet"
    original = row(QuantileCurve([0.9, 0.99], [1, 2]))
    table = rows_to_table([original], metadata())
    assert write_parquet(path, table) == path
    info, rows = read_parquet(path)
    assert info.to_json() == metadata().to_json()
    assert rows[0].curve is not None and original.curve is not None
    assert rows[0].curve.to_json() == original.curve.to_json()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_parquet(path, table)
    assert path.read_bytes() == before
    for invalid_overwrite in (1, "yes", None):
        with pytest.raises(ValueError, match="overwrite must be boolean"):
            write_parquet(path, table, overwrite=invalid_overwrite)  # type: ignore[arg-type]
        assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.parquet"))
    write_parquet(path, table, overwrite=True)
    with pytest.raises(ValueError, match="local filesystem"):
        write_parquet("s3://bucket/key", table)


def test_parquet_write_requires_an_identical_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = rows_to_table([row(QuantileCurve([0.9, 0.99], [1, 2]))], metadata())
    altered = table.set_column(
        table.column_names.index("support_id"),
        "support_id",
        pa.array(["other"], pa.string()),
    )
    monkeypatch.setattr("crc_sdk._schema.dataset.pq.read_table", lambda path: altered)
    path = tmp_path / "dataset.parquet"
    with pytest.raises(OSError, match="did not read back"):
        write_parquet(path, table)
    assert not path.exists() and not list(tmp_path.glob(".*.parquet"))


def test_stored_fit_records_are_validated_against_their_curves() -> None:
    def fitted(family: DistributionFamily) -> DatasetRow:
        fit = fit_samples(
            [1, 2, 3, 4, 5, 6, 7, 8],
            SampleFit(family=family, estimator=SampleEstimator.L_MOMENTS),
        )
        return row(fit.distribution, fit=fit.record)

    table = rows_to_table([fitted(DistributionFamily.GUMBEL_RIGHT)], metadata())
    other = rows_to_table([fitted(DistributionFamily.GUMBEL_LEFT)], metadata())
    # The stored record text is what gets checked against the stored curve.
    swapped = table.set_column(
        table.column_names.index("fit"), "fit", other.column("fit")
    )
    assert table_to_rows(table)
    with pytest.raises(ValueError, match="contradicts the selected fit"):
        table_to_rows(swapped)


def test_empty_canonical_data_keeps_schema_and_complete_typed_report() -> None:
    info = replace(metadata(), report=ExecutionReport(complete=True, valid=True))
    table = rows_to_table([], info)
    assert table.num_rows == 0 and table.schema.equals(dataset_arrow_schema(info))
    assert table_to_rows(table) == ()
    assert validate_rows([], info).complete


def test_generated_schema_is_current() -> None:
    path = ROOT / "crc_sdk/contracts/schemas/dataset-2.0.schema.json"
    assert json.loads(path.read_text()) == dataset_json_schema()


def test_installed_pair_shares_identical_dataset_and_taxonomy_contracts() -> None:
    sdk = files("crc_sdk").joinpath("contracts")
    framework = files("crc_framework").joinpath("contracts")
    for sdk_path, framework_path in (
        ("schemas/dataset-2.0.schema.json", "dataset/dataset-2.0.schema.json"),
        ("fixtures/dataset-2.0.json", "dataset/fixtures.json"),
        ("taxonomy/taxonomy-1.0.schema.json", "taxonomy/taxonomy-1.0.schema.json"),
        ("taxonomy/fixtures.json", "taxonomy/fixtures.json"),
    ):
        assert (
            sdk.joinpath(sdk_path).read_bytes()
            == framework.joinpath(framework_path).read_bytes()
        ), f"paired schema or fixture differs: {sdk_path}"


def test_shared_dataset_fixtures_agree_across_schema_python_and_arrow() -> None:
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource

    resources = []
    framework_contracts = files("crc_framework").joinpath("contracts")
    for directory in (
        framework_contracts.joinpath("scientific"),
        framework_contracts.joinpath("numerical"),
        files("crc_sdk").joinpath("contracts", "schemas"),
    ):
        for path in directory.iterdir():
            if path.name.endswith(".schema.json"):
                schema = json.loads(path.read_text(encoding="utf-8"))
                resources.append((schema["$id"], Resource.from_contents(schema)))
    validator = Draft202012Validator(
        dataset_json_schema(),
        registry=Registry().with_resources(resources),
        format_checker=FormatChecker(),
    )
    cases = json.loads(
        (ROOT / "crc_sdk/contracts/fixtures/dataset-2.0.json").read_text()
    )
    for case in cases["accepted"]:
        document = case["document"]
        errors = list(validator.iter_errors(document))
        assert not errors, (case["name"], errors)
        info, rows = dataset_from_json(json.dumps(document))
        assert (
            dataset_from_json(
                dataset_to_json(table_to_rows(rows_to_table(rows, info)), info)
            )[0].to_json()
            == info.to_json()
        )
    for case in cases["rejected"]:
        if case["structural"]:
            assert not validator.is_valid(case["document"]), case["name"]
        with pytest.raises((ValueError, TypeError)):
            dataset_from_json(json.dumps(case["document"]))


def test_complete_reports_cannot_contradict_actual_row_counts() -> None:
    info = replace(
        metadata(),
        report=ExecutionReport(True, True, counts=ExecutionCounts(produced=2)),
    )
    with pytest.raises(ValueError, match="produced count"):
        validate_rows([row()], info)
    info = replace(
        metadata(),
        report=ExecutionReport(
            True, True, counts=ExecutionCounts(produced=1, no_data=1)
        ),
    )
    with pytest.raises(ValueError, match="no_data count"):
        validate_rows([row()], info)
    preview = replace(
        metadata(),
        report=ExecutionReport(False, None, counts=ExecutionCounts(produced=99)),
    )
    validate_rows([row()], preview)


def test_prose_is_preserved_without_identifier_whitespace_rules() -> None:
    from crc_framework import DiagnosticSeverity

    resource = ResourceIdentity(
        "s3://example/data",
        "sha256:" + "a" * 64,
        license="  License text\n",
        attribution="\nOriginal source\n",
    )
    diagnostic = ExecutionDiagnostic(
        "example",
        DiagnosticSeverity.WARNING,
        Stage.ACQUISITION,
        "\n source message  ",
        remedy="  inspect the source\n",
    )
    report = ExecutionReport(
        True, None, diagnostics=(diagnostic,), counts=ExecutionCounts(produced=1)
    )
    info = replace(
        metadata(),
        source=SourceIdentity("example", "product", resources=(resource,)),
        report=report,
    )
    restored, _ = dataset_from_json(dataset_to_json([row()], info))
    assert restored.source is not None
    assert restored.source.resources[0].license == resource.license
    assert restored.source.resources[0].attribution == resource.attribution
    assert restored.report.diagnostics[0].message == diagnostic.message
    assert restored.report.diagnostics[0].remedy == diagnostic.remedy
    with pytest.raises(ValueError):
        replace(resource, uri=" leading-space")
    with pytest.raises(ValueError):
        replace(diagnostic, code=" leading-space")


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-01-01 00:00:00+00:00",
        "2026-01-01T00:00+00:00",
        "20260101T000000+0000",
        "2026-W01-4T00:00:00+00:00",
        "2026-01-01T00:00:60Z",
        "0000-01-01T00:00:00Z",
    ],
)
def test_json_retrieval_time_uses_rfc3339_not_permissive_iso(timestamp: str) -> None:
    source = SourceIdentity(
        "example",
        "product",
        resources=(ResourceIdentity("s3://example/data", "sha256:" + "a" * 64),),
    )
    document = json.loads(dataset_to_json([row()], replace(metadata(), source=source)))
    document["metadata"]["source"]["resources"][0]["retrieved_at"] = timestamp
    with pytest.raises(ValueError, match="RFC3339"):
        dataset_from_json(json.dumps(document))


def test_rfc3339_time_hydrates_and_serializes_canonically() -> None:
    source = SourceIdentity(
        "example",
        "product",
        resources=(ResourceIdentity("s3://example/data", "sha256:" + "a" * 64),),
    )
    document = json.loads(dataset_to_json([row()], replace(metadata(), source=source)))
    document["metadata"]["source"]["resources"][0]["retrieved_at"] = (
        "2026-01-01t00:00:00z"
    )
    info, _ = dataset_from_json(json.dumps(document))
    assert info.source is not None
    assert info.source.resources[0].retrieved_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        json.loads(info.to_json())["source"]["resources"][0]["retrieved_at"]
        == "2026-01-01T00:00:00+00:00"
    )


def test_typed_retrieval_times_preserve_the_instant_in_rfc3339_utc() -> None:
    timestamp = datetime(
        2026,
        1,
        1,
        tzinfo=timezone(timedelta(seconds=30, microseconds=250000)),
    )
    source = SourceIdentity(
        "example",
        "product",
        release=ReleaseReference(
            "release", "sha256:" + "a" * 64, retrieved_at=timestamp
        ),
        resources=(
            ResourceIdentity(
                "s3://example/data", "sha256:" + "b" * 64, retrieved_at=timestamp
            ),
        ),
    )
    text = dataset_to_json([row()], replace(metadata(), source=source))
    document = json.loads(text)
    expected = "2025-12-31T23:59:29.750000+00:00"
    assert document["metadata"]["source"]["release"]["retrieved_at"] == expected
    assert document["metadata"]["source"]["resources"][0]["retrieved_at"] == expected
    info, _ = dataset_from_json(text)
    assert info.source is not None and info.source.release is not None
    assert info.source.release.retrieved_at == timestamp
    assert info.source.resources[0].retrieved_at == timestamp


def test_decode_curves_returns_each_rows_framework_object() -> None:
    from crc_sdk import decode_curves

    curves = numerical_cases()
    counts = ExecutionCounts(requested=len(curves), produced=len(curves))
    table = rows_to_table(
        [
            DatasetRow("user:depth", f"support:{index}", CELL, curve=curve)
            for index, curve in enumerate(curves)
        ],
        replace(
            metadata(), report=ExecutionReport(complete=True, valid=True, counts=counts)
        ),
    )
    decoded = decode_curves(table)
    assert [value.to_json() for value in decoded if value is not None] == [
        curve.to_json() for curve in curves
    ]
    assert decode_curves(table.to_batches()[0])[0] is not None
    with pytest.raises(FitError) as error:
        fit_samples([1] * 8, SampleFit(family=DistributionFamily.GUMBEL_RIGHT))
    failed = DatasetRow(
        "user:depth", "failed", CELL, failure=RetainedFailure.from_error(error.value)
    )
    no_data = ExecutionCounts(requested=1, produced=1, no_data=1)
    failed_info = replace(
        metadata(), report=ExecutionReport(complete=True, valid=True, counts=no_data)
    )
    assert decode_curves(rows_to_table([failed], failed_info)) == (None,)
    with pytest.raises(ValueError, match="no curve column"):
        decode_curves(table.drop_columns(["curve"]))
    with pytest.raises(TypeError):
        decode_curves(table.to_pylist())
