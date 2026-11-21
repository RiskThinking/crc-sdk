"""Admit complete, source-backed annual damage curves and policy terms."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal
from numbers import Real
from typing import cast

import crc_framework as cf
from shapely import from_wkb  # type: ignore[import-untyped]

from ._assets import CellColumn
from ._dataset_schema import (
    ExecutionDiagnostic,
    ExecutionReport,
    SourceIdentity,
    Stage,
    _dump,
    _identifier,
    _load,
)
from ._declarations import ProbabilityKind, ScientificValue
from ._errors import CapabilityError, ValidationError
from ._evaluation import Evaluation, _EvaluationNode
from ._evaluation_schema import AssetId, EvaluationMetadata, QueryKind
from ._graph import DataValidationError, ExecutionScope


def _policy_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise TypeError(f"{label} must be a real number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return 0.0 if number == 0 else number


def _fraction(value: float, label: str) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"{label} must lie in [0, 1]")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyColumns:
    """Map book columns or constants to insured values and primary terms."""

    total_insured_value: str | float
    currency: str
    valuation_basis: str
    deductible: str | float = 0.0
    limit: str | float | None = None
    share: str | float = 1.0
    deductible_fraction: str | float | None = None

    def __post_init__(self) -> None:
        _identifier(self.currency, "currency")
        _identifier(self.valuation_basis, "valuation_basis")
        for name in (
            "total_insured_value",
            "deductible",
            "limit",
            "share",
            "deductible_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, str):
                _identifier(value, name + " column")
            elif value is not None or name not in ("limit", "deductible_fraction"):
                object.__setattr__(self, name, _policy_number(value, name))
        if self.deductible_fraction is not None and self.deductible != 0:
            raise ValueError("Choose either deductible or deductible_fraction")
        if isinstance(self.deductible_fraction, float):
            _fraction(self.deductible_fraction, "deductible_fraction")
        if isinstance(self.total_insured_value, float) and self.total_insured_value < 0:
            raise ValueError("total_insured_value must be nonnegative")
        cf.PolicyTerms(
            deductible=0.0 if isinstance(self.deductible, str) else self.deductible,
            limit=None if isinstance(self.limit, str) else self.limit,
            share=1.0 if isinstance(self.share, str) else self.share,
        )

    def _columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                value
                for value in (
                    self.total_insured_value,
                    self.deductible,
                    self.limit,
                    self.share,
                    self.deductible_fraction,
                )
                if isinstance(value, str)
            )
        )

    def _resolve(self, row: dict[str, object]) -> tuple[float, cf.PolicyTerms]:
        def value(binding: str | float, label: str) -> float:
            if isinstance(binding, str):
                return _policy_number(row[binding], f"{label} (column {binding!r})")
            return binding

        tiv = value(self.total_insured_value, "total_insured_value")
        deductible = (
            value(self.deductible, "deductible")
            if self.deductible_fraction is None
            else tiv
            * _fraction(
                value(self.deductible_fraction, "deductible_fraction"),
                "deductible_fraction",
            )
        )
        return tiv, cf.PolicyTerms(
            deductible=deductible,
            limit=None if self.limit is None else value(self.limit, "limit"),
            share=value(self.share, "share"),
        )


@dataclass(frozen=True, slots=True)
class _LossInputs:
    interests: tuple[cf.InsuredInterest, ...]
    book_sha256: str
    terms_sha256: str
    input_sha256: str
    evidence: dict[str, ScientificValue]
    report: ExecutionReport


def precheck_loss_inputs(
    evaluation: Evaluation,
    *,
    damage_response: str,
    tail_completion: cf.TailCompletion | None,
) -> Evaluation:
    """Narrow the recipe and check declarations without resolving any resource."""
    if not isinstance(evaluation, Evaluation):
        raise TypeError("evaluation must be an Evaluation recipe")
    _identifier(damage_response, "damage_response")
    plan = evaluation._plan
    responses = {response.name: response for response in plan._selected_responses()}
    if damage_response not in responses:
        raise ValueError(
            f"Named damage response {damage_response!r} is absent "
            "from the evaluation selection"
        )
    model = responses[damage_response].model
    output = model.output if isinstance(model, cf.Impact) else model.requirements.output
    if output != cf.Ratio(unit=cf.RatioUnit.FRACTION):
        raise CapabilityError(
            f"Damage response {damage_response!r} must output "
            "Ratio(unit=RatioUnit.FRACTION)"
        )
    if isinstance(plan.assets.location, CellColumn):
        raise CapabilityError(
            "Annual loss requires WGS84 asset points. "
            "Supply PointColumns or WKB points."
        )
    query = plan.query
    if query is None or query.kind is not QueryKind.PROBABILITY:
        raise CapabilityError(
            "Annual loss requires evaluation.probabilities([...]) "
            "in non-exceedance ranks"
        )
    probabilities = query.values
    if len(probabilities) < 2 or any(
        b <= a for a, b in zip(probabilities, probabilities[1:], strict=False)
    ):
        raise ValueError(
            "Annual loss requires at least two strictly increasing probability knots"
        )
    if tail_completion is not None and not isinstance(
        tail_completion, cf.TailCompletion
    ):
        raise TypeError("tail_completion must be a framework TailCompletion")
    if tail_completion is not None:
        _fraction(tail_completion.lower, "tail_completion.lower")
        _fraction(tail_completion.upper, "tail_completion.upper")
    elif not (probabilities[0] == 0 and probabilities[-1] == 1):
        raise ValueError(
            "Annual loss requires probability support from 0 through 1. "
            "Pass tail_completion=TailCompletion(lower=..., upper=...) to complete "
            "the damage ratios using explicit endpoints."
        )
    selected = evaluation.select(responses=damage_response)
    # Exercise static selectors without inspecting sources.
    selected._plan._selected_source()
    return selected


def _diagnostic(
    code: str,
    message: str,
    *,
    support_id: str | None = None,
    query_value: float | None = None,
) -> ExecutionDiagnostic:
    return ExecutionDiagnostic(
        code,
        cf.DiagnosticSeverity.ERROR,
        Stage.VALIDATION,
        message,
        support_id=support_id,
        query_value=query_value,
    )


def _fail(
    report: ExecutionReport, diagnostics: tuple[ExecutionDiagnostic, ...]
) -> ValidationError:
    return ValidationError(
        report=replace(
            report, valid=False, diagnostics=(*report.diagnostics, *diagnostics)
        )
    )


@dataclass(frozen=True, slots=True)
class _DamageKnot:
    q: float
    value: float | None
    exposure: float | None
    support_id: str | None
    resource_id: str | None
    resolution_id: str | None
    failure_code: str | None
    failure_message: str | None


def _update(digest: hashlib._Hash, value: object) -> None:
    digest.update(_dump(value).encode())
    digest.update(b"\n")


def _hex(digest: hashlib._Hash) -> str:
    return "sha256:" + digest.hexdigest()


def _stable_source(source: SourceIdentity) -> dict[str, object]:
    return replace(
        source,
        release=None
        if source.release is None
        else replace(source.release, retrieved_at=None),
        resources=tuple(
            replace(resource, retrieved_at=None)
            for resource in sorted(source.resources, key=lambda resource: resource.uri)
        ),
    )._document()


def _selection(
    metadata: EvaluationMetadata, report: ExecutionReport
) -> tuple[tuple[str | None, str, str | None, str | None], dict[str, ScientificValue]]:
    if len(metadata._series) != 1:
        found = sorted((_dump((member, *key)) for member, key in metadata._series))
        examples = ", ".join(found[:8]) or "none"
        raise _fail(
            report,
            (
                _diagnostic(
                    "loss_series_selection",
                    "Annual loss requires exactly one selected member and series. "
                    "Use evaluation.select(...) to narrow the selection. "
                    f"Found {len(found)} combinations: {examples}.",
                ),
            ),
        )
    (member, key), series = next(iter(metadata._series.items()))
    if (
        series.probability is None
        or series.probability.kind is not ProbabilityKind.ANNUAL
    ):
        raise _fail(
            report,
            (
                _diagnostic(
                    "loss_probability_not_annual",
                    f"Selected variable {series.variable.id!r} requires declared "
                    "annual probabilities. Supply Probability.annual(tail=...) only "
                    "when the source represents annual outcomes.",
                ),
            ),
        )
    selection: dict[str, ScientificValue] = {
        "member_id": member,
        "variable_id": key[0],
        "scenario_id": key[1],
        "period_id": key[2],
        "series": {
            "variable": series.variable._document(),
            "scenario": None
            if series.scenario is None
            else series.scenario._document(),
            "period": None if series.period is None else series.period._document(),
            "probability": series.probability._document(),
        },
    }
    return (member, *key), selection


def _damage_knots(
    scope: ExecutionScope,
    node: _EvaluationNode,
    selection: tuple[str | None, str, str | None, str | None],
    damage_response: str,
    rows: dict[AssetId, list[_DamageKnot]],
) -> None:
    for batch in scope.iter_batches(node):
        columns = batch.select(
            [
                "asset_id",
                "member_id",
                "resource_id",
                "variable_id",
                "scenario_id",
                "period_id",
                "support_id",
                "response_id",
                "resolution_id",
                "query_value",
                "value",
                "failure",
                "exposure",
            ]
        )
        for row in columns.to_pylist():
            if (
                row["member_id"],
                row["variable_id"],
                row["scenario_id"],
                row["period_id"],
            ) != selection or row["response_id"] != damage_response:
                raise DataValidationError(
                    (
                        _diagnostic(
                            "loss_row_identity",
                            "A damage row contradicts the selected "
                            "member, series or response.",
                        ),
                    )
                )
            failure = row["failure"]
            rows[row["asset_id"]].append(
                _DamageKnot(
                    row["query_value"],
                    row["value"],
                    row["exposure"]["value"],
                    row["support_id"],
                    row["resource_id"],
                    row["resolution_id"],
                    None if failure is None else failure["code"],
                    None if failure is None else failure["message"],
                )
            )


def _curve(
    asset_id: AssetId,
    knots: list[_DamageKnot],
    probabilities: tuple[float, ...],
    tail_completion: cf.TailCompletion | None,
) -> cf.QuantileCurve | cf.Distribution | tuple[ExecutionDiagnostic, ...]:
    if tuple(knot.q for knot in knots) != probabilities:
        return (
            _diagnostic(
                "loss_incomplete_grid",
                f"Asset {asset_id!r} requires the complete ordered probability grid of "
                f"{len(probabilities)} knots. Received {len(knots)} rows.",
            ),
        )
    failures: dict[str, list[_DamageKnot]] = defaultdict(list)
    for knot in knots:
        if knot.failure_code is not None:
            failures[knot.failure_code].append(knot)
    if failures:
        return tuple(
            _diagnostic(
                code,
                f"Asset {asset_id!r} failed {len(affected)} "
                f"of {len(knots)} damage queries. "
                f"First failure at q={affected[0].q:g}. {affected[0].failure_message}",
                support_id=affected[0].support_id,
                query_value=affected[0].q,
            )
            for code, affected in failures.items()
        )
    values = tuple(knot.value for knot in knots)
    if any(value is None or not (0 <= value <= 1) for value in values):
        return (
            _diagnostic(
                "loss_damage_ratio_bounds",
                f"Asset {asset_id!r} requires finite damage fractions in [0, 1].",
                support_id=knots[0].support_id,
            ),
        )
    try:
        curve = cf.QuantileCurve(probabilities, cast(tuple[float, ...], values))
        return curve if tail_completion is None else curve.complete(tail_completion)
    except ValueError as error:
        return (
            _diagnostic(
                "loss_damage_curve",
                f"Asset {asset_id!r}: {error}",
                support_id=knots[0].support_id,
            ),
        )


def prepare_loss_inputs(
    evaluation: Evaluation,
    *,
    damage_response: str,
    policy: PolicyColumns,
    tail_completion: cf.TailCompletion | None,
    scope: ExecutionScope,
    required_asset_columns: tuple[str, ...] = (),
) -> _LossInputs:
    """Resolve the selected source graph and require full book coverage."""
    node = _EvaluationNode(
        evaluation._plan,
        required_asset_columns=(*required_asset_columns, *policy._columns()),
        collect_context_failures=True,
    )
    try:
        metadata = scope.metadata(node)
        series, selection = _selection(metadata, scope.report(node))
        assets = node._asset_data(scope, node.responses)
        rows: dict[AssetId, list[_DamageKnot]] = defaultdict(list)
        expected = assets.table.num_rows * len(metadata.query.values)
        try:
            _damage_knots(scope, node, series, damage_response, rows)
        except DataValidationError:
            # The source can finish with model-context errors after producing
            # all rows. Keep those rows to report any other coverage failures.
            if not expected or sum(map(len, rows.values())) != expected:
                raise
        metadata = scope.metadata(node)
    except DataValidationError as error:
        report = scope.report(node)
        new_diagnostics = tuple(
            value for value in error.diagnostics if value not in report.diagnostics
        )
        raise _fail(report, new_diagnostics) from error
    report = scope.report(node)
    ids = assets.table[assets.id_column].to_pylist()
    if not ids:
        raise _fail(
            report,
            (
                _diagnostic(
                    "loss_empty_portfolio",
                    "Annual loss requires at least one insured asset.",
                ),
            ),
        )
    response = node.responses[0]
    probability_grid = metadata.query.values
    book_digest = hashlib.sha256()
    terms_digest = hashlib.sha256()
    supports_digest = hashlib.sha256()
    curves_digest = hashlib.sha256()
    vulnerability_digest = hashlib.sha256()
    _update(
        vulnerability_digest,
        (
            response.model.sha256,
            None if response.context is None else _load(response.context.to_json()),
        ),
    )
    impacts = {value.id: value.impact.sha256 for value in metadata.resolutions}
    counts: Counter[str] = Counter()
    supports: set[tuple[str | None, str | None]] = set()
    diagnostics: list[ExecutionDiagnostic] = []
    resources: list[ScientificValue] = []
    stable_resource_ids: dict[str | None, str] = {}
    for item in metadata.inputs:
        stable_resource: dict[str, ScientificValue] = {
            "member_id": item.member_id,
            "source": None
            if item.metadata.source is None
            else cast(ScientificValue, _stable_source(item.metadata.source)),
            "transformations": cast(
                ScientificValue, [step._document() for step in item.metadata.provenance]
            ),
        }
        stable_resource_ids[item.resource_id] = _hex(
            hashlib.sha256(_dump(stable_resource).encode())
        )
        resources.append({**stable_resource, "resource_id": item.resource_id})
    interests = []
    policy_rows = assets.context.select(policy._columns()).to_pylist()
    # An Arrow table with zero columns has no rows after some projections.
    if not policy._columns():
        policy_rows = [{}] * len(ids)
    entries = zip(ids, assets.geometries, policy_rows, strict=True)
    for asset_id, wkb, policy_row in sorted(entries, key=lambda item: str(item[0])):
        point = from_wkb(wkb)
        if point.geom_type != "Point":
            diagnostics.append(
                _diagnostic(
                    "loss_location_not_point",
                    f"Asset {asset_id!r} requires an actual WGS84 point "
                    "for spatial dependence. "
                    f"Received {point.geom_type}.",
                )
            )
            continue
        knots = rows.get(asset_id, [])
        curve = _curve(asset_id, knots, probability_grid, tail_completion)
        if isinstance(curve, tuple):
            diagnostics.extend(curve)
            continue
        try:
            tiv, terms = policy._resolve(policy_row)
            interest = cf.InsuredInterest(
                str(asset_id),
                longitude=point.x,
                latitude=point.y,
                total_insured_value=tiv,
                damage_curve=curve,
                terms=terms,
            )
        except (TypeError, ValueError) as error:
            diagnostics.append(
                _diagnostic("loss_policy_terms", f"Asset {asset_id!r}: {error}")
            )
            continue
        interests.append(interest)
        first = knots[0]
        support_key = (first.resource_id, first.support_id)
        supports.add(support_key)
        resolution_id = first.resolution_id or response.model.sha256
        counts[resolution_id] += 1
        _update(
            book_digest,
            (asset_id, interest.longitude, interest.latitude, tiv),
        )
        _update(
            terms_digest,
            (asset_id, terms.deductible, terms.limit, terms.share),
        )
        _update(
            supports_digest,
            (
                asset_id,
                stable_resource_ids[first.resource_id],
                first.support_id,
                resolution_id,
            ),
        )
        _update(vulnerability_digest, (asset_id, resolution_id))
        _update(
            curves_digest,
            (asset_id, [(knot.q, knot.exposure, knot.value) for knot in knots]),
        )
    if diagnostics:
        locations = sum(d.code == "loss_location_not_point" for d in diagnostics)
        invalid_terms = sum(d.code == "loss_policy_terms" for d in diagnostics)
        rejected = len(ids) - len(interests)
        raise _fail(
            report,
            (
                _diagnostic(
                    "loss_incomplete_portfolio",
                    f"Annual loss rejected {rejected} of {len(ids)} assets. "
                    f"Invalid locations: {locations}. Invalid terms: {invalid_terms}. "
                    f"Unusable damage curves: {rejected - locations - invalid_terms}.",
                ),
                *diagnostics,
            ),
        )
    if report.valid is not True:
        raise _fail(report, ())
    source_documents = sorted(
        {_dump(_stable_source(source)) for source in report.source_identities}
    )
    evidence: dict[str, ScientificValue] = {
        **selection,
        "damage_response": damage_response,
        "response_model_sha256": response.model.sha256,
        "vulnerability_sha256": _hex(vulnerability_digest),
        "probability_grid": probability_grid,
        "tail_completion": None
        if tail_completion is None
        else cast(ScientificValue, _load(tail_completion.to_json())),
        "matched_support_count": len(supports),
        "matched_supports_sha256": _hex(supports_digest),
        "resolved_impacts": [
            {
                "resolution_id": resolution_id,
                "impact_sha256": impacts.get(resolution_id, response.model.sha256),
                "asset_count": counts[resolution_id],
            }
            for resolution_id in sorted(counts)
        ],
        "input_resources": sorted(resources, key=_dump),
    }
    input_digest = hashlib.sha256()
    # Canonical readers have scope-local resource IDs. The audit record shows
    # them as observed; the run key uses resolved science and content instead.
    stable_evidence = {k: v for k, v in evidence.items() if k != "input_resources"}
    _update(
        input_digest,
        (
            stable_evidence,
            sorted(stable_resource_ids.values()),
            metadata.spatial._document(),
            source_documents,
            _hex(curves_digest),
        ),
    )
    return _LossInputs(
        tuple(interests),
        _hex(book_digest),
        _hex(terms_digest),
        _hex(input_digest),
        evidence,
        report,
    )
