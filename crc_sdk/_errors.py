"""Public errors shared by recipes, execution, and scientific validation."""

from __future__ import annotations

from ._dataset_schema import ExecutionReport


class CapabilityError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ValidationError(ValueError):
    def __init__(self, *, report: ExecutionReport) -> None:
        self._report = report
        errors = [d.message for d in report.diagnostics if d.severity.value == "error"]
        super().__init__("; ".join(errors) or "Scientific data validation failed")

    @property
    def report(self) -> ExecutionReport:
        return self._report
