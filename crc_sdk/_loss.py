"""Run annual insurance simulation on selected hazard evaluations."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Self

import crc_framework as cf
import numpy as np

from ._dataset_schema import (
    ExecutionDiagnostic,
    ExecutionReport,
    Stage,
    _combine_reports,
    _identifier,
)
from ._errors import ValidationError
from ._evaluation import Evaluation
from ._graph import ExecutionScope
from ._inspection import Explanation, Operation, Requirement
from ._loss_artifact import (
    LossArtifact,
    LossResult,
    make_loss_result,
    write_loss_artifact,
)
from ._loss_inputs import (
    PolicyColumns,
    _LossInputs,
    precheck_loss_inputs,
    prepare_loss_inputs,
)
from ._runtime import LocalPath, Runtime, _memory_bytes
from ._terminals import _publication_path


@dataclass(frozen=True, slots=True)
class _PerilRun:
    peril: str
    evaluation: Evaluation = field(repr=False)
    damage_response: str
    policy: PolicyColumns
    tail_completion: cf.TailCompletion | None

    def explain(self, runtime: Runtime | None) -> Explanation:
        evaluation = self.evaluation.explain(runtime=runtime)
        missing = self.evaluation._plan.assets._missing_bound_columns(
            self.policy._columns()
        )
        if not missing:
            return evaluation
        return replace(
            evaluation,
            requirements=(
                *evaluation.requirements,
                Requirement(
                    "asset_policy_columns_required",
                    f"Add the policy columns {list(missing)!r} for {self.peril!r} "
                    "to the assets or set constant values in PolicyColumns.",
                    "policy",
                ),
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class LossRun:
    """A lazy recipe for annual insured losses on one fixed book."""

    _perils: tuple[_PerilRun, ...]
    _spec: cf.AnnualLossSpec
    _secondary_stream: int
    _include_components: bool
    _dependence: cf.PerilDependence | None

    def __init__(
        self,
        evaluation: Evaluation,
        *,
        damage_response: str,
        peril: str,
        policy: PolicyColumns,
        spec: cf.AnnualLossSpec | None = None,
        tail_completion: cf.TailCompletion | None = None,
        secondary_stream: int = 0,
        include_components: bool = False,
    ) -> None:
        if not isinstance(policy, PolicyColumns):
            raise TypeError("policy must be PolicyColumns")
        if spec is not None and not isinstance(spec, cf.AnnualLossSpec):
            raise TypeError("spec must be AnnualLossSpec or None")
        if type(include_components) is not bool:
            raise TypeError("include_components must be bool")
        if isinstance(secondary_stream, (bool, np.bool_)):
            raise TypeError("secondary_stream must be an integer")
        stream = operator.index(secondary_stream)
        if not 0 <= stream < 2**64:
            raise ValueError("secondary_stream must be an unsigned 64-bit integer")
        _identifier(peril, "peril")
        selected = precheck_loss_inputs(
            evaluation, damage_response=damage_response, tail_completion=tail_completion
        )
        peril_run = _PerilRun(peril, selected, damage_response, policy, tail_completion)
        object.__setattr__(self, "_perils", (peril_run,))
        object.__setattr__(
            self, "_spec", cf.AnnualLossSpec.reference() if spec is None else spec
        )
        object.__setattr__(self, "_secondary_stream", stream)
        object.__setattr__(self, "_include_components", include_components)
        object.__setattr__(self, "_dependence", None)

    @classmethod
    def separate_perils(
        cls, runs: Sequence[LossRun], *, dependence: cf.PerilDependence
    ) -> LossRun:
        """Combine a fixed book whose policy terms apply to each peril separately."""
        if not isinstance(dependence, cf.PerilDependence):
            raise TypeError("dependence must be PerilDependence")
        selected = tuple(runs)
        if len(selected) < 2:
            raise ValueError("separate_perils requires at least two single-peril runs")
        if any(
            not isinstance(run, cls) or run._dependence is not None for run in selected
        ):
            raise TypeError("runs must contain single-peril LossRun recipes")
        perils = tuple(
            sorted((run._perils[0] for run in selected), key=lambda p: p.peril)
        )
        names = tuple(peril.peril for peril in perils)
        if len(set(names)) != len(names):
            raise ValueError("separate_perils requires unique peril names")
        if dependence.correlation is not None and dependence.perils != names:
            raise ValueError(
                "calibrated dependence must cover exactly the run's perils"
            )
        first = selected[0]
        policy = first._perils[0].policy
        for run in selected[1:]:
            other = run._perils[0].policy
            if (other.currency, other.valuation_basis) != (
                policy.currency,
                policy.valuation_basis,
            ):
                raise ValueError("All perils must use one currency and valuation basis")
            if (run._spec, run._secondary_stream, run._include_components) != (
                first._spec,
                first._secondary_stream,
                first._include_components,
            ):
                raise ValueError(
                    "All perils must use the same spec, secondary_stream and "
                    "include_components setting"
                )
        result = object.__new__(cls)
        object.__setattr__(result, "_perils", perils)
        object.__setattr__(result, "_spec", first._spec)
        object.__setattr__(result, "_secondary_stream", first._secondary_stream)
        object.__setattr__(result, "_include_components", first._include_components)
        object.__setattr__(result, "_dependence", dependence)
        return result

    def explain(self, *, runtime: Runtime | None = None) -> Explanation:
        explanations = tuple(peril.explain(runtime) for peril in self._perils)
        operations = []
        for peril, explanation in zip(self._perils, explanations, strict=True):
            if self._dependence is not None:
                operations.append(
                    Operation(Stage.SELECTION, f"Resolve {peril.peril!r}")
                )
            operations.extend(explanation.operations)
        scope = ", ".join(repr(peril.peril) for peril in self._perils)
        return replace(
            explanations[0],
            source=explanations[0].source if self._dependence is None else None,
            requirements=tuple(
                requirement
                for value in explanations
                for requirement in value.requirements
            ),
            operations=(
                *operations,
                Operation(
                    Stage.VALIDATION,
                    "Resolve portfolio terms and complete damage curves",
                ),
                Operation(
                    Stage.EVALUATION,
                    f"Simulate {self._spec.trials:,} annual insured losses for {scope}",
                ),
            ),
        )

    def execute(self, *, runtime: Runtime | None = None) -> LossExecution:
        return LossExecution(self, runtime=runtime)

    def validate(self, *, runtime: Runtime | None = None) -> ExecutionReport:
        with self.execute(runtime=runtime) as execution:
            return execution.validate()

    def read(self, *, runtime: Runtime | None = None) -> LossResult:
        with self.execute(runtime=runtime) as execution:
            return execution.read()

    def write(
        self,
        path: LocalPath,
        *,
        overwrite: bool = False,
        runtime: Runtime | None = None,
    ) -> LossArtifact:
        with self.execute(runtime=runtime) as execution:
            return execution.write(path, overwrite=overwrite)


class LossExecution:
    """Share one source snapshot and simulation across an execution's terminals."""

    def __init__(self, recipe: LossRun, *, runtime: Runtime | None = None) -> None:
        if not isinstance(recipe, LossRun):
            raise TypeError("recipe must be LossRun")
        self._recipe = recipe
        self._scope = ExecutionScope(runtime)
        self._entered = False
        self._closed = False
        self._running = False
        self._report = ExecutionReport(False, None)
        self._result: LossResult | None = None

    def __enter__(self) -> Self:
        if self._entered or self._closed:
            raise RuntimeError("an execution can only be entered once")
        self._scope.__enter__()
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._closed = True
        self._scope.__exit__(exc_type, exc, traceback)

    def _active(self) -> None:
        if not self._entered or self._closed:
            raise RuntimeError("use execution terminals inside 'with recipe.execute()'")

    @property
    def report(self) -> ExecutionReport:
        return self._report

    def _resolve_inputs(self) -> tuple[dict[str, _LossInputs], ExecutionReport]:
        recipes = self._recipe._perils
        columns: dict[str, set[str]] = {}
        for recipe in recipes:
            plan = recipe.evaluation._plan
            names = columns.setdefault(plan.assets._key, set())
            names.update(recipe.policy._columns())
            for response in plan._selected_responses():
                names.update(response.columns)

        inputs: dict[str, _LossInputs] = {}
        reports: list[ExecutionReport] = []
        for recipe in recipes:
            try:
                resolved = prepare_loss_inputs(
                    recipe.evaluation,
                    damage_response=recipe.damage_response,
                    policy=recipe.policy,
                    tail_completion=recipe.tail_completion,
                    scope=self._scope,
                    required_asset_columns=tuple(
                        sorted(columns[recipe.evaluation._plan.assets._key])
                    ),
                )
            except ValidationError as error:
                if len(recipes) == 1 or error is self._scope._callback_error:
                    raise
                failed = error.report
                report = _combine_reports((*reports, failed))
                diagnostic = ExecutionDiagnostic(
                    "invalid_peril_inputs",
                    cf.DiagnosticSeverity.ERROR,
                    Stage.VALIDATION,
                    f"Peril {recipe.peril!r} has invalid insurance inputs.",
                )
                raise ValidationError(
                    report=replace(
                        report,
                        valid=False,
                        diagnostics=(diagnostic, *report.diagnostics),
                        counts=replace(
                            report.counts,
                            matched_assets=failed.counts.matched_assets,
                            unmatched_assets=failed.counts.unmatched_assets,
                        ),
                    )
                ) from error
            inputs[recipe.peril] = resolved
            reports.append(resolved.report)

        first = inputs[recipes[0].peril]
        report = _combine_reports(reports, complete=True)
        report = replace(
            report, counts=replace(report.counts, matched_assets=len(first.interests))
        )
        for peril, resolved in inputs.items():
            if resolved.book_sha256 != first.book_sha256:
                diagnostic = ExecutionDiagnostic(
                    "peril_book_mismatch",
                    cf.DiagnosticSeverity.ERROR,
                    Stage.VALIDATION,
                    f"Perils {recipes[0].peril!r} and {peril!r} must have identical "
                    "asset IDs, WGS84 coordinates and total insured values.",
                )
                raise ValidationError(
                    report=replace(
                        report,
                        valid=False,
                        diagnostics=(*report.diagnostics, diagnostic),
                    )
                )
        return inputs, report

    def _simulate(self) -> LossResult:
        recipe = self._recipe
        inputs, report = self._resolve_inputs()
        first = next(iter(inputs.values()))
        self._report = replace(report, complete=False, valid=None)
        trials = recipe._spec.trials
        self._scope.emit(Stage.EVALUATION, 0, trials)
        memory_limit = _memory_bytes(self._scope.runtime.memory_limit)
        limits = {} if memory_limit is None else {"memory_limit": memory_limit}
        spec = recipe._spec
        stream = recipe._secondary_stream
        components = recipe._include_components
        try:
            if recipe._dependence is None:
                simulation = cf.simulate_annual_losses(
                    first.interests,
                    spec=spec,
                    secondary_stream=stream,
                    include_components=components,
                    **limits,
                )
            else:
                simulation = cf.simulate_separate_peril_losses(
                    {peril: resolved.interests for peril, resolved in inputs.items()},
                    dependence=recipe._dependence,
                    spec=spec,
                    secondary_stream=stream,
                    include_components=components,
                    **limits,
                )
        except ValueError as error:
            diagnostic = ExecutionDiagnostic(
                "annual_loss_simulation_failed",
                cf.DiagnosticSeverity.ERROR,
                Stage.EVALUATION,
                str(error),
            )
            raise ValidationError(
                report=replace(
                    report,
                    valid=False,
                    diagnostics=(*report.diagnostics, diagnostic),
                )
            ) from error
        result = make_loss_result(
            simulation,
            peril_inputs=inputs,
            dependence=recipe._dependence,
            currency=recipe._perils[0].policy.currency,
            valuation_basis=recipe._perils[0].policy.valuation_basis,
            portfolio_sha256=first.book_sha256,
            report=report,
        )
        self._scope.emit(Stage.EVALUATION, trials, trials)
        self._report = result.report
        return result

    def read(self) -> LossResult:
        self._active()
        if self._running:
            raise RuntimeError("an annual loss run is already executing in this scope")
        if self._report.valid is False:
            raise ValidationError(report=self._report)
        if self._result is None:
            self._running = True
            try:
                self._result = self._simulate()
            except ValidationError as error:
                if error is not self._scope._callback_error:
                    self._report = error.report
                raise
            finally:
                self._running = False
        return self._result

    def validate(self) -> ExecutionReport:
        self._active()
        try:
            self.read()
        except ValidationError as error:
            if error is self._scope._callback_error:
                raise
        return self._report

    def write(self, path: LocalPath, *, overwrite: bool = False) -> LossArtifact:
        self._active()
        destination = _publication_path(path, overwrite)

        def progress(completed: int, total: int) -> None:
            self._scope.emit(Stage.WRITING, completed, total)

        return write_loss_artifact(
            destination, self.read(), overwrite=overwrite, progress=progress
        )
