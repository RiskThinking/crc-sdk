"""Run one-peril insurance simulation on a selected hazard evaluation."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Self

import crc_framework as cf
import numpy as np

from ._dataset_schema import ExecutionDiagnostic, ExecutionReport, Stage, _identifier
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
from ._loss_inputs import PolicyColumns, precheck_loss_inputs, prepare_loss_inputs
from ._runtime import LocalPath, Runtime, _memory_bytes
from ._terminals import _publication_path


@dataclass(frozen=True, slots=True, init=False)
class LossRun:
    """A lazy recipe for annual insured losses for one peril and book."""

    _evaluation: Evaluation = field(repr=False)
    _damage_response: str
    _peril: str
    _policy: PolicyColumns
    _spec: cf.AnnualLossSpec
    _tail_completion: cf.TailCompletion | None
    _secondary_stream: int
    _include_components: bool

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
        object.__setattr__(self, "_evaluation", selected)
        object.__setattr__(self, "_damage_response", damage_response)
        object.__setattr__(self, "_peril", peril)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(
            self, "_spec", cf.AnnualLossSpec.reference() if spec is None else spec
        )
        object.__setattr__(self, "_tail_completion", tail_completion)
        object.__setattr__(self, "_secondary_stream", stream)
        object.__setattr__(self, "_include_components", include_components)

    def explain(self, *, runtime: Runtime | None = None) -> Explanation:
        evaluation = self._evaluation.explain(runtime=runtime)
        requirements = evaluation.requirements
        missing = self._evaluation._plan.assets._missing_bound_columns(
            self._policy._columns()
        )
        if missing:
            requirements = (
                *requirements,
                Requirement(
                    "asset_policy_columns_required",
                    f"Add the policy columns {list(missing)!r} to the assets or "
                    "supply constant values in PolicyColumns.",
                    "policy",
                ),
            )
        return replace(
            evaluation,
            requirements=requirements,
            operations=(
                *evaluation.operations,
                Operation(
                    Stage.VALIDATION,
                    "Resolve portfolio terms and complete damage curves",
                ),
                Operation(
                    Stage.EVALUATION,
                    f"Simulate {self._spec.trials:,} annual insured losses "
                    f"for {self._peril!r}",
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

    def _simulate(self) -> LossResult:
        recipe = self._recipe
        inputs = prepare_loss_inputs(
            recipe._evaluation,
            damage_response=recipe._damage_response,
            policy=recipe._policy,
            tail_completion=recipe._tail_completion,
            scope=self._scope,
        )
        self._report = replace(inputs.report, complete=False, valid=None)
        trials = recipe._spec.trials
        self._scope.emit(Stage.EVALUATION, 0, trials)
        memory_limit = _memory_bytes(self._scope.runtime.memory_limit)
        limits = {} if memory_limit is None else {"memory_limit": memory_limit}
        try:
            simulation = cf.simulate_annual_losses(
                inputs.interests,
                spec=recipe._spec,
                secondary_stream=recipe._secondary_stream,
                include_components=recipe._include_components,
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
                    inputs.report,
                    valid=False,
                    diagnostics=(*inputs.report.diagnostics, diagnostic),
                )
            ) from error
        result = make_loss_result(
            simulation,
            peril=recipe._peril,
            currency=recipe._policy.currency,
            valuation_basis=recipe._policy.valuation_basis,
            portfolio_sha256=inputs.portfolio_sha256,
            input_sha256=inputs.input_sha256,
            evidence=inputs.evidence,
            report=inputs.report,
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
