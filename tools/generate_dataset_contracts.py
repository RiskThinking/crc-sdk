"""Generate canonical dataset schema and shared conformance fixtures."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from crc_framework import (
    Aggregation,
    DiagnosticSeverity,
    Distribution,
    DistributionFamily,
    Duration,
    FitConstraints,
    FitError,
    HurdleQuantileFit,
    LengthUnit,
    PrecipitationAmount,
    PrecipitationRate,
    PrecipitationRateUnit,
    QuantileCurve,
    QuantileFit,
    SampleEstimator,
    SampleFit,
    Tail,
    TailCompletion,
    TimeUnit,
    fit_quantiles,
    fit_samples,
)

from crc_sdk._aggregation import (
    Coverage,
    MeasurementChange,
    ReductionKind,
    Sampling,
    TemporalReduction,
)
from crc_sdk._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionReport,
    FittingDiagnostic,
    ObservationTimeSelection,
    ResourceIdentity,
    RetainedFailure,
    SourceIdentity,
    SpatialContract,
    Stage,
    SupportKind,
    UnavailableStationYear,
    UnitConversion,
)
from crc_sdk._declarations import (
    Calendar,
    DataSpec,
    Period,
    Probability,
    Scenario,
    SeriesSpec,
    Variable,
)
from crc_sdk._schema import DatasetRow, dataset_to_json
from crc_sdk._schema.contract import dataset_json_schema

ROOT = Path(__file__).resolve().parents[1]


def fixture_cases() -> dict[str, object]:
    variable = Variable("user:depth")
    metadata = DatasetMetadata(
        spec=DataSpec([variable], series=[SeriesSpec(variable)]),
        spatial=SpatialContract(supports=(SupportKind.CELL,), index_resolution=4),
        report=ExecutionReport(
            True, True, counts=ExecutionCounts(requested=1, produced=1)
        ),
    )
    row = DatasetRow(
        variable.id,
        "user:support",
        594475228122316799,
        curve=Distribution.point_mass(1),
    )
    minimal = json.loads(dataset_to_json([row], metadata))
    accepted = [{"name": "optional-dimensions", "document": minimal}]
    window = Period.window(2030, 2050)
    scenario = Scenario("user:scenario", definition={"warming": 2})
    series = SeriesSpec(
        variable,
        scenario=scenario,
        period=window,
        probability=Probability.annual(tail=Tail.UPPER),
    )
    declared = replace(metadata, spec=DataSpec([variable], series=[series]))
    declared_row = replace(
        row,
        scenario_id=scenario.id,
        period_id=window.id,
        curve=QuantileCurve([0.9, 0.99], [0.5, 2]),
    )
    annual = json.loads(dataset_to_json([declared_row], declared))
    accepted.append({"name": "annual-window", "document": annual})
    curve = QuantileCurve([0.9, 0.99], [0.5, 2]).complete(
        TailCompletion(lower=0, upper=4)
    )
    accepted.append(
        {
            "name": "explicit-completion",
            "document": json.loads(
                dataset_to_json([replace(row, curve=curve)], metadata)
            ),
        }
    )
    law = Distribution.parametric(DistributionFamily.GUMBEL_RIGHT, location=0, scale=1)
    accepted.append(
        {
            "name": "hurdle",
            "document": json.loads(
                dataset_to_json(
                    [
                        replace(
                            row, curve=Distribution.hurdle(law, atom_probability=0.25)
                        )
                    ],
                    metadata,
                )
            ),
        }
    )
    fit = fit_samples(
        [1, 2, 3, 4, 5, 6, 7, 8],
        SampleFit(
            family=DistributionFamily.GUMBEL_RIGHT, estimator=SampleEstimator.L_MOMENTS
        ),
    )
    fitted = json.loads(
        dataset_to_json(
            [replace(row, curve=fit.distribution, fit=fit.record)], metadata
        )
    )
    accepted.append({"name": "authoritative-fit-record", "document": fitted})
    bounded_quantile_fit = fit_quantiles(
        QuantileCurve([0.1, 0.25, 0.5, 0.75, 0.9], [1, 2, 3, 4, 5]),
        QuantileFit(
            family=DistributionFamily.WEIBULL_MINIMUM,
            constraints=FitConstraints(require_sample_support=True),
        ),
    )
    fitted_quantiles = json.loads(
        dataset_to_json(
            [
                replace(
                    row,
                    curve=bounded_quantile_fit.distribution,
                    fit=bounded_quantile_fit.record,
                )
            ],
            metadata,
        )
    )
    accepted.append(
        {"name": "fitted-quantile-value-interval", "document": fitted_quantiles}
    )
    hurdle = Distribution.hurdle(law, atom_probability=0.25)
    probabilities = [0.1, 0.2, 0.25, 0.4, 0.6, 0.8, 0.9, 0.99]
    hurdle_fit = fit_quantiles(
        QuantileCurve(probabilities, hurdle.quantile(probabilities)),
        HurdleQuantileFit(
            family=DistributionFamily.GUMBEL_RIGHT,
            atom_probability=0.25,
            constraints=FitConstraints(probability=0.9, maximum_value=10),
        ),
    )
    fitted_hurdle = json.loads(
        dataset_to_json(
            [replace(row, curve=hurdle_fit.distribution, fit=hurdle_fit.record)],
            metadata,
        )
    )
    accepted.append({"name": "fitted-hurdle", "document": fitted_hurdle})
    try:
        fit_samples([1] * 8, SampleFit(family=DistributionFamily.GUMBEL_RIGHT))
    except FitError as error:
        failure = RetainedFailure.from_error(error)
    else:
        raise AssertionError("constant MLE must fail explicitly")
    failed_info = replace(
        metadata,
        report=ExecutionReport(
            True, True, counts=ExecutionCounts(requested=1, produced=1, no_data=1)
        ),
    )
    failed = json.loads(
        dataset_to_json([replace(row, curve=None, failure=failure)], failed_info)
    )
    accepted.append({"name": "retained-fit-failure", "document": failed})
    diagnostic_info = replace(
        failed_info,
        report=replace(
            failed_info.report,
            diagnostics=(
                FittingDiagnostic(
                    code=failure.code,
                    severity=DiagnosticSeverity.WARNING,
                    stage=Stage.FITTING,
                    message=failure.message,
                    support_id=row.support_id,
                    record=failure.record,
                ),
            ),
        ),
    )
    fitting_diagnostic = json.loads(
        dataset_to_json([replace(row, curve=None, failure=failure)], diagnostic_info)
    )
    accepted.append(
        {"name": "fitting-failure-diagnostic", "document": fitting_diagnostic}
    )
    station_year_report = ExecutionReport(
        True,
        True,
        diagnostics=(
            UnavailableStationYear(
                Stage.ACQUISITION,
                "No station-year object exists. Historical support is unknown.",
                station_id="test:station",
                year=2020,
                minimum_fraction=1.0,
            ),
        ),
        counts=ExecutionCounts(requested=1, incomplete_years=1),
    )
    station_year = json.loads(
        dataset_to_json([], replace(metadata, report=station_year_report))
    )
    accepted.append({"name": "unavailable-station-year", "document": station_year})
    source_info = replace(
        metadata,
        source=SourceIdentity(
            "provider:user",
            "product:user",
            resources=(
                ResourceIdentity("s3://example/observations", "sha256:" + "a" * 64),
            ),
            member="member:1",
            model="model:1",
        ),
    )
    accepted.append(
        {
            "name": "typed-source-resource",
            "document": json.loads(dataset_to_json([row], source_info)),
        }
    )
    empty_info = replace(metadata, report=ExecutionReport(True, True))
    accepted.append(
        {
            "name": "empty-dataset",
            "document": json.loads(dataset_to_json([], empty_info)),
        }
    )
    time_info = replace(
        metadata, provenance=(ObservationTimeSelection(time=Period.window(1980, 2010)),)
    )
    accepted.append(
        {
            "name": "observation-time-selection",
            "document": json.loads(dataset_to_json([row], time_info)),
        }
    )
    rate = PrecipitationRate(
        unit=PrecipitationRateUnit.MILLIMETRES_PER_DAY,
        accumulation_duration=Duration(1, unit=TimeUnit.DAY),
        aggregation=Aggregation.MEAN,
    )
    daily_amount = PrecipitationAmount(
        unit=LengthUnit.MILLIMETRE,
        accumulation_duration=Duration(1, unit=TimeUnit.DAY),
    )
    five_day_amount = PrecipitationAmount(
        unit=LengthUnit.MILLIMETRE,
        accumulation_duration=Duration(5, unit=TimeUnit.DAY),
    )
    sampling = Sampling(calendar=Calendar.DAYS_360, step=Duration(1, unit=TimeUnit.DAY))
    reduced_variable = Variable(variable.id, measurement=five_day_amount)
    reduced_info = replace(
        metadata,
        spec=DataSpec([reduced_variable], series=[SeriesSpec(reduced_variable)]),
        provenance=(
            UnitConversion(
                variable_id=variable.id,
                source_unit="kg m-2 s-1",
                target_unit="mm/day",
                scale=86400,
            ),
            TemporalReduction(
                operation=ReductionKind.DAILY_TOTAL,
                coverage=Coverage.complete(),
                input_sampling=sampling,
                output_sampling=sampling,
                measurements=(
                    MeasurementChange(
                        variable_id=variable.id, input=rate, output=daily_amount
                    ),
                ),
            ),
            TemporalReduction(
                operation=ReductionKind.ROLLING_TOTAL,
                coverage=Coverage.complete(),
                input_sampling=sampling,
                output_sampling=sampling,
                measurements=(
                    MeasurementChange(
                        variable_id=variable.id,
                        input=daily_amount,
                        output=five_day_amount,
                    ),
                ),
                window=Duration(5, unit=TimeUnit.DAY),
            ),
        ),
    )
    reduced = json.loads(dataset_to_json([row], reduced_info))
    accepted.append(
        {"name": "temporal-precipitation-measurement-lineage", "document": reduced}
    )
    rejected = []

    def reject(name, document, structural):
        rejected.append({"name": name, "structural": structural, "document": document})

    value = deepcopy(reduced)
    value["metadata"]["provenance"][-1]["measurements"][0]["unknown"] = True
    reject("unknown-temporal-measurement-change-field", value, True)
    value = deepcopy(reduced)
    value["metadata"]["provenance"][-1]["window"]["unknown"] = True
    reject("unknown-temporal-window-field", value, True)
    value = deepcopy(reduced)
    value["metadata"]["provenance"][-1]["measurements"][0]["output"][
        "accumulation_duration"
    ]["value"] = 1
    reject("temporal-output-measurement-mismatch", value, False)
    value = deepcopy(reduced)
    value["metadata"]["provenance"][1]["input_sampling"]["step"] = {
        "value": 1,
        "unit": "yr",
    }
    reject("temporal-input-cadence", value, False)
    value = deepcopy(reduced)
    value["metadata"]["provenance"][-1]["output_sampling"]["calendar"] = "gregorian"
    reject("temporal-calendar-change", value, False)
    value = deepcopy(reduced)
    value["metadata"]["provenance"][-1].pop("window")
    reject("temporal-invalid-window", value, False)
    value = deepcopy(reduced)
    change = value["metadata"]["provenance"][-1]["measurements"][0]
    value["metadata"]["provenance"][-1]["measurements"].append(deepcopy(change))
    reject("temporal-duplicate-measurement-variable", value, False)
    value = deepcopy(minimal)
    value["metadata"]["schema_version"] = "1.3"
    reject("historical-dataset-version", value, True)
    value = deepcopy(minimal)
    value["metadata"]["value_unit"] = "m"
    reject("removed-global-unit", value, True)
    value = deepcopy(minimal)
    value["rows"][0]["hazard_name"] = "depth"
    reject("removed-row-alias", value, True)
    value = deepcopy(minimal)
    value["rows"][0]["curve"]["scale"] = 0
    reject("inactive-point-mass-field", value, True)
    value = deepcopy(minimal)
    value["rows"][0]["geometry"] = None
    reject("null-optional-geometry", value, True)
    value = deepcopy(minimal)
    value["rows"][0]["failure"] = failed["rows"][0]["failure"]
    reject("curve-and-failure", value, True)
    value = deepcopy(minimal)
    value["rows"][0]["cell_id"] = 0
    reject("invalid-h3-cell", value, False)
    value = deepcopy(minimal)
    value["rows"].append(deepcopy(value["rows"][0]))
    value["metadata"]["report"]["counts"]["produced"] = 2
    reject("duplicate-null-row-key", value, False)
    value = deepcopy(minimal)
    value["rows"][0]["scenario_id"] = "not-declared"
    reject("dangling-row-scenario", value, False)
    value = deepcopy(minimal)
    value["metadata"]["spec"]["series"][0]["variable_id"] = "not-declared"
    reject("dangling-series-variable", value, False)
    value = deepcopy(annual)
    value["metadata"]["spec"]["periods"][0]["start"] = 2020
    reject("period-content-identity-mismatch", value, False)
    value = deepcopy(minimal)
    value["metadata"]["report"]["counts"]["produced"] = 2
    reject("incorrect-produced-count", value, False)
    value = deepcopy(minimal)
    value["metadata"]["report"]["counts"]["no_data"] = 1
    reject("incorrect-no-data-count", value, False)
    value = deepcopy(minimal)
    value["metadata"]["report"]["complete"] = False
    reject("incomplete-valid-report", value, True)
    value = json.loads(dataset_to_json([row], time_info))
    value["metadata"]["provenance"][0]["time"]["start"] = 1981
    reject("time-selection-content-identity-mismatch", value, False)
    value = deepcopy(fitted)
    value["rows"][0]["curve"]["family"] = "gumbel_l"
    reject("fit-family-mismatch", value, False)
    value = deepcopy(fitted)
    value["rows"][0]["fit"].update(used_count=3, excluded_count=5)
    reject("fit-sample-invented-exclusions", value, False)
    value = deepcopy(fitted)
    value["rows"][0]["fit"]["value_interval"] = None
    reject("fit-sample-missing-observed-support", value, False)
    value = deepcopy(fitted_hurdle)
    base = value["rows"][0]["curve"]["base"]
    value["rows"][0]["curve"] = dict(base, schema_version="1.0", object="distribution")
    reject("fit-hurdle-removed", value, False)
    value = deepcopy(fitted_hurdle)
    value["rows"][0]["curve"]["atom_probability"] = 0.9
    reject("fit-hurdle-atom-changed", value, False)
    value = deepcopy(fitted_hurdle)
    value["rows"][0]["fit"]["probability_interval"] = [0, 1]
    reject("fit-active-probability-endpoints", value, False)
    value = deepcopy(fitted_quantiles)
    value["rows"][0]["curve"]["location"] = 100
    reject("fit-excludes-observed-quantile-values", value, False)
    value = deepcopy(fitted_hurdle)
    record = value["rows"][0]["fit"]
    for specification in (
        record["specification"],
        record["attempts"][0]["specification"],
    ):
        specification["constraints"]["maximum_value"] = 0.1
    reject("fit-constraint-violated", value, False)
    value = deepcopy(fitting_diagnostic)
    value["metadata"]["report"]["diagnostics"][0]["stage"] = "decoding"
    reject("fitting-diagnostic-wrong-stage", value, True)
    value = deepcopy(fitting_diagnostic)
    value["metadata"]["report"]["diagnostics"][0]["record"]["evidence_count"] += 1
    reject("fitting-diagnostic-invalid-trace", value, False)
    value = deepcopy(station_year)
    value["metadata"]["report"]["diagnostics"][0]["support_id"] = "inferred-point"
    reject("unavailable-station-year-with-support", value, True)
    value = json.loads(dataset_to_json([row], source_info))
    value["metadata"]["source"]["resources"][0]["sha256"] = "not-a-hash"
    reject("invalid-resource-digest", value, True)
    value = json.loads(dataset_to_json([row], source_info))
    value["metadata"]["source"]["model"] = " padded-model"
    reject("invalid-source-model-id", value, True)
    for field in ("source", "report"):
        value = json.loads(dataset_to_json([row], source_info))
        identity = value["metadata"]["source"]
        identity["resources"].append(
            {**identity["resources"][0], "sha256": "sha256:" + "b" * 64}
        )
        if field == "report":
            del value["metadata"]["source"]
            value["metadata"]["report"]["source_identities"] = [identity]
        reject(f"duplicate-resource-uri-in-{field}", value, False)
    value = deepcopy(annual)
    other_scenario = "user:other-scenario"
    spec = value["metadata"]["spec"]
    spec["scenarios"].append({"id": other_scenario})
    spec["series"].append({**spec["series"][0], "scenario_id": other_scenario})
    value["rows"].append({**value["rows"][0], "scenario_id": other_scenario})
    value["metadata"]["report"]["counts"].update(requested=2, produced=2)
    value["metadata"]["provenance"] = [
        {
            "kind": "rename",
            "dimension": "scenario",
            "mapping": {scenario.id: other_scenario},
        }
    ]
    reject("scenario-rename-collides-with-unchanged-identity", value, False)
    geometry_cases = json.loads(
        (ROOT / "crc_sdk/contracts/fixtures/geometry-cases.json").read_text()
    )
    for case in geometry_cases:
        document = deepcopy(minimal)
        document["metadata"]["spatial"]["supports"] = case["supports"]
        document["rows"] = []
        count = len(case["geometries"])
        document["metadata"]["report"]["counts"].update(requested=count, produced=count)
        for index, geometry in enumerate(case["geometries"]):
            value = deepcopy(minimal["rows"][0])
            if index:
                value["variable_id"] = f"second:{index}"
                document["metadata"]["spec"]["variables"].append(
                    {"id": value["variable_id"]}
                )
                document["metadata"]["spec"]["series"].append(
                    {
                        "variable_id": value["variable_id"],
                        "scenario_id": None,
                        "period_id": None,
                    }
                )
            value["geometry"] = geometry
            document["rows"].append(value)
        item = {"name": case["name"], "document": document}
        if case["valid"]:
            accepted.append(item)
        else:
            item["structural"] = False
            rejected.append(item)
    value = deepcopy(minimal)
    value["metadata"]["report"]["counts"]["produced"] = 1.0
    value["metadata"]["spatial"]["index_resolution"] = 4.0
    accepted.append({"name": "integral-json-number-spelling", "document": value})
    source_document = next(
        item["document"] for item in accepted if item["name"] == "typed-source-resource"
    )
    prose = deepcopy(source_document)
    resource = prose["metadata"]["source"]["resources"][0]
    resource["license"] = "  Source license text\n"
    resource["attribution"] = "\nOriginal scientists\n"
    prose["metadata"]["report"]["diagnostics"].append(
        {
            "code": "source.note",
            "severity": "info",
            "stage": "acquisition",
            "message": "\nsource message  ",
            "remedy": "  inspect source\n",
        }
    )
    accepted.append({"name": "literal-source-and-diagnostic-prose", "document": prose})
    timed = deepcopy(source_document)
    timed["metadata"]["source"]["resources"][0]["retrieved_at"] = "2026-01-01t00:00:00z"
    accepted.append({"name": "rfc3339-retrieval-time", "document": timed})
    for name, timestamp in (
        ("retrieval-space-separator", "2026-01-01 00:00:00+00:00"),
        ("retrieval-missing-seconds", "2026-01-01T00:00+00:00"),
        ("retrieval-week-date", "2026-W01-4T00:00:00+00:00"),
    ):
        value = deepcopy(source_document)
        value["metadata"]["source"]["resources"][0]["retrieved_at"] = timestamp
        reject(name, value, True)
    return {"accepted": accepted, "rejected": rejected}


def build() -> dict[str, bytes]:
    values = {
        "schemas/dataset-2.0.schema.json": dataset_json_schema(),
        "fixtures/dataset-2.0.json": fixture_cases(),
    }
    return {
        name: (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        for name, value in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for relative, content in build().items():
        path = ROOT / "crc_sdk/contracts" / relative
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit(f"stale generated dataset contract: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(
        "Dataset schema and shared fixtures verified"
        if args.check
        else "Dataset schema and shared fixtures generated"
    )


if __name__ == "__main__":
    main()
