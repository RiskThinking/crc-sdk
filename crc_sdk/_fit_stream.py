"""Batch fitting of quantile evidence, in-process or in worker processes.

Each batch is decoded, fitted row by row, and re-encoded independently, so a
fit consumes its evidence one batch at a time and only up to
`Runtime(workers=...)` batches are in flight. Results keep the input order.
"""

from __future__ import annotations

import multiprocessing
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import TypeAlias

import crc_framework
import pyarrow as pa  # type: ignore[import-untyped]

from ._dataset_schema import DatasetMetadata, RetainedFailure
from ._errors import CapabilityError
from ._evidence import FitFailure
from ._schema.dataset import dataset_arrow_schema, rows_to_table, table_to_rows

QuantileFitSpec: TypeAlias = (
    crc_framework.QuantileFit
    | crc_framework.CensoredQuantileFit
    | crc_framework.HurdleQuantileFit
    | crc_framework.ModelSelection
)
QUANTILE_FIT_SPECS = (
    crc_framework.QuantileFit,
    crc_framework.CensoredQuantileFit,
    crc_framework.HurdleQuantileFit,
    crc_framework.ModelSelection,
)


@dataclass(frozen=True, slots=True)
class FailedFit:
    """A row whose fit failed, in a form that crosses process boundaries."""

    code: str
    message: str
    support_id: str
    variable_id: str
    record: str

    @classmethod
    def from_error(
        cls, error: crc_framework.FitError, *, support_id: str, variable_id: str
    ) -> FailedFit:
        return cls(
            error.code, error.message, support_id, variable_id, error.record.to_json()
        )

    def fit_record(self) -> crc_framework.FitRecord:
        return crc_framework.FitRecord.from_json(self.record)


@dataclass(frozen=True, slots=True)
class FittedBatch:
    """One fitted batch: its output rows and the rows that failed."""

    table: pa.Table
    requested: int
    failures: tuple[FailedFit, ...]
    raised: FailedFit | None = None


def fit_batch(
    batch: pa.RecordBatch,
    metadata: str,
    spec: QuantileFitSpec,
    failure: str,
) -> FittedBatch:
    """Fit one batch of quantile evidence.

    `metadata` is the evidence's DatasetMetadata JSON with an incomplete
    report, so each batch validates on its own. `failure` is a FitFailure
    value. Under "raise" the first failed row stops the batch and is returned
    as `raised`; the caller raises it in input order.
    """
    declared = DatasetMetadata.from_json(metadata)
    schema = dataset_arrow_schema(declared)
    rows = table_to_rows(
        pa.Table.from_batches([batch], schema=schema).replace_schema_metadata(
            schema.metadata
        )
    )
    mode = FitFailure(failure)
    result = []
    failures: list[FailedFit] = []
    for row in rows:
        if not isinstance(row.curve, crc_framework.QuantileCurve):
            raise CapabilityError(
                "fit() requires supplied quantile evidence. "
                "Existing distributions are already models."
            )
        try:
            fitted = crc_framework.fit_quantiles(row.curve, spec)
        except crc_framework.FitError as error:
            failed = FailedFit.from_error(
                error, support_id=row.support_id, variable_id=row.variable_id
            )
            if mode is FitFailure.RAISE:
                return FittedBatch(
                    pa.Table.from_batches([], schema=schema), len(rows), (), failed
                )
            failures.append(failed)
            if mode is FitFailure.KEEP_NO_DATA:
                result.append(
                    replace(
                        row,
                        curve=None,
                        fit=None,
                        failure=RetainedFailure.from_error(error),
                    )
                )
            continue
        result.append(replace(row, curve=fitted.distribution, fit=fitted.record))
    return FittedBatch(rows_to_table(result, declared), len(rows), tuple(failures))


def fitted_batches(
    batches: Iterator[pa.RecordBatch],
    metadata: str,
    spec: QuantileFitSpec,
    failure: str,
    *,
    workers: int,
    on_close: Callable[[Callable[[], None]], None],
) -> Iterator[FittedBatch]:
    """`fit_batch` over `batches` in order, on up to `workers` processes.

    A single batch, or `workers=1`, is fitted in this process. Otherwise worker
    processes start once a second batch arrives, and at most `workers` batches
    beyond the one being consumed are in flight. Closing the generator cancels
    waiting batches; `on_close` registers the pool's shutdown with the owner,
    which waits for running batches when it closes.
    """
    first = next(batches, None)
    if first is None:
        return
    second = None if workers <= 1 else next(batches, None)
    if second is None:
        yield fit_batch(first, metadata, spec, failure)
        for batch in batches:
            yield fit_batch(batch, metadata, spec, failure)
        return
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    on_close(pool.shutdown)
    pending: deque[Future[FittedBatch]] = deque(
        pool.submit(fit_batch, batch, metadata, spec, failure)
        for batch in (first, second)
    )
    try:
        for batch in batches:
            while len(pending) > workers:
                yield pending.popleft().result()
            pending.append(pool.submit(fit_batch, batch, metadata, spec, failure))
        while pending:
            yield pending.popleft().result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
