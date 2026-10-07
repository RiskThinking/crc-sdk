"""Dataset-level metadata models."""

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crc_sdk.schema import (
    HAZARD_ROW_KEY,
    HAZARD_SORT_ORDER,
)

PARQUET_METADATA_KEY = "crc.hazard.metadata"

ProbabilitySemantics = Literal[
    "annual_exceedance",
    "annual_value_distribution",
    "within_period_percentile",
    "projection_uncertainty",
    "estimate_confidence",
]
ReturnPeriodConvention = Literal["one_minus_inverse", "poisson"]


class SourceProvenance(BaseModel):
    """Stable identification of the external dataset used during ingest."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    uri: Optional[str] = None
    version: Optional[str] = None
    licence: Optional[str] = None
    attribution: Optional[str] = None
    retrieved_at: Optional[str] = None
    checksum: Optional[str] = None

    @field_validator("retrieved_at")
    @classmethod
    def validate_retrieved_at(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(
                    "retrieved_at must be an ISO 8601 timestamp"
                ) from error
        return value


class EnsembleDescriptor(BaseModel):
    """Which models and members one canonical dataset represents (ADR-0001)."""

    model_config = ConfigDict(frozen=True)

    pooling: Literal["single_member", "pooled", "unknown"] = "unknown"
    models: Optional[Union[int, tuple[str, ...]]] = None
    members: Optional[Union[int, tuple[str, ...]]] = None
    scenario: Optional[str] = None
    downscaling_method: Optional[str] = None
    bias_adjustment: Optional[str] = None
    pooling_method: Optional[str] = None

    @model_validator(mode="after")
    def validate_counts(self) -> "EnsembleDescriptor":
        for name in ("models", "members"):
            value = getattr(self, name)
            if isinstance(value, bool) or (isinstance(value, int) and value < 1):
                raise ValueError(f"{name} count must be a positive integer")
        if self.pooling == "single_member":
            for name in ("models", "members"):
                value = getattr(self, name)
                size = value if isinstance(value, int) else len(value or ())
                if size > 1:
                    raise ValueError(
                        f"single_member datasets cannot list several {name}"
                    )
        return self


class TemporalWindow(BaseModel):
    """The period one dataset's ``horizon`` stands for (ADR-0002).

    ``horizon`` is the centre year of an inclusive ``[start_year, end_year]``
    window; time-invariant datasets declare the ``reference_year`` carried in
    the ``horizon`` column instead of a placeholder.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["window", "time_invariant"] = "window"
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    reference_year: Optional[int] = None
    baseline: Optional[tuple[int, int]] = None
    calendar: Literal["gregorian", "noleap", "360_day", "unspecified"] = "unspecified"
    minimum_complete_years: Optional[int] = Field(default=None, ge=1)

    @property
    def horizon(self) -> int:
        """Centre year of the window, or the reference year when time-invariant."""
        if self.kind == "time_invariant":
            assert self.reference_year is not None
            return self.reference_year
        assert self.start_year is not None and self.end_year is not None
        return (self.start_year + self.end_year) // 2

    @model_validator(mode="after")
    def validate_window(self) -> "TemporalWindow":
        if self.kind == "window":
            if self.start_year is None or self.end_year is None:
                raise ValueError("window datasets require start_year and end_year")
            if self.end_year < self.start_year:
                raise ValueError("end_year must not precede start_year")
            if self.reference_year is not None:
                raise ValueError("reference_year is only for time_invariant datasets")
            length = self.end_year - self.start_year + 1
            if (
                self.minimum_complete_years is not None
                and self.minimum_complete_years > length
            ):
                raise ValueError("minimum_complete_years exceeds the window length")
        else:
            if self.reference_year is None:
                raise ValueError("time_invariant datasets require reference_year")
            if self.start_year is not None or self.end_year is not None:
                raise ValueError("time_invariant datasets have no start/end years")
        if self.baseline is not None and self.baseline[1] < self.baseline[0]:
            raise ValueError("baseline end must not precede its start")
        return self


class CurveFitProvenance(BaseModel):
    """Dataset-wide scientific and failure policy used for curve fitting."""

    model_config = ConfigDict(frozen=True)

    method: Literal["quantile_least_squares", "sample_mle", "sample_lmoments"] = (
        "quantile_least_squares"
    )
    initialization: Optional[str] = None
    input_kind: Optional[Literal["probability_labelled", "samples"]] = None
    sample_resampling: Optional[int] = Field(default=None, ge=2)
    lower_bound: Optional[float] = None
    platform: Optional[str] = None
    crc_framework_version: Optional[str] = None
    families: tuple[str, ...]
    selection_metric: Literal["fixed_family", "first_acceptable"] = "fixed_family"
    weighting: Literal["uniform"] = "uniform"
    endpoint_policy: Literal["exclude_zero_and_one"] = "exclude_zero_and_one"
    atom_policy: Literal["none", "infer_min_plateau"]
    constant_policy: Literal["point_mass"] = "point_mass"
    minimum_informative_value: Optional[float] = None
    minimum_informative_knots: int = Field(default=0, ge=0)
    minimum_distinct_informative_values: int = Field(default=0, ge=0)
    parametric_failure_action: Literal["raise", "skip", "tabulated"] = "raise"
    maximum_normalized_rmse: Optional[float] = None
    maximum_absolute_residual: Optional[float] = None
    on_fit_failure: Literal["raise", "skip"]


class HazardDatasetMetadata(BaseModel):
    """Metadata shared by every row in one canonical hazard dataset."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0", "1.1", "1.2", "1.3"] = "1.3"
    h3_resolution: int = Field(ge=0, le=15)
    probability_convention: Literal["non_exceedance"] = "non_exceedance"
    return_period_tail: Literal["upper", "lower"] = "upper"
    return_period_support: Optional[tuple[float, float]] = None
    source_probability_support: Optional[tuple[float, float]] = None
    value_unit: str = Field(min_length=1)
    value_semantics: str = Field(min_length=1)
    geometry_encoding: Literal["WKB"] = "WKB"
    geometry_crs: str = "EPSG:4326"
    producer: str = Field(min_length=1)
    source: SourceProvenance
    fitting: Optional[CurveFitProvenance] = None
    probability_semantics: Optional[ProbabilitySemantics] = None
    source_return_period_convention: Optional[ReturnPeriodConvention] = None
    temporal_window: Optional[TemporalWindow] = None
    ensemble: Optional[EnsembleDescriptor] = None
    creation_version: str = Field(min_length=1)
    row_key: tuple[str, ...] = HAZARD_ROW_KEY
    sort_order: tuple[str, ...] = HAZARD_SORT_ORDER

    @model_validator(mode="after")
    def validate_contract_constants(self) -> "HazardDatasetMetadata":
        if self.row_key != HAZARD_ROW_KEY:
            raise ValueError(f"row_key must be {HAZARD_ROW_KEY!r}")
        if self.sort_order != HAZARD_SORT_ORDER:
            raise ValueError(f"sort_order must be {HAZARD_SORT_ORDER!r}")
        if self.schema_version == "1.0" and (
            self.source_probability_support is not None or self.fitting is not None
        ):
            raise ValueError(
                "source probability support and fitting provenance require schema 1.1"
            )
        if self.schema_version != "1.3":
            added = [
                name
                for name in (
                    "probability_semantics",
                    "source_return_period_convention",
                    "temporal_window",
                    "ensemble",
                )
                if getattr(self, name) is not None
            ]
            if (
                self.source.licence
                or self.source.attribution
                or self.source.retrieved_at
            ):
                added.append("source licence/attribution/retrieved_at")
            if self.source.checksum:
                added.append("source checksum")
            fit = self.fitting
            if fit is not None:
                added += [
                    f"fitting.{name}"
                    for name in (
                        "initialization",
                        "input_kind",
                        "sample_resampling",
                        "lower_bound",
                        "platform",
                        "crc_framework_version",
                    )
                    if getattr(fit, name) is not None
                ]
                if fit.method != "quantile_least_squares":
                    added.append(f"fitting.method={fit.method}")
            if added:
                raise ValueError(f"{', '.join(added)} require schema 1.3")
        if self.return_period_support is not None:
            lower, upper = self.return_period_support
            if lower <= 1.0 or upper <= lower:
                raise ValueError(
                    "return_period_support must be increasing and greater than one"
                )
        if self.source_probability_support is not None:
            lower, upper = self.source_probability_support
            if not 0.0 <= lower < upper <= 1.0:
                raise ValueError(
                    "source_probability_support must be increasing within [0, 1]"
                )
        return self

    def to_json_bytes(self) -> bytes:
        """Serialize the complete Parquet metadata payload."""
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_json_bytes(cls, value: bytes) -> "HazardDatasetMetadata":
        return cls.model_validate_json(value)

    def to_parquet_metadata(
        self, existing: Optional[Mapping[bytes, bytes]] = None
    ) -> dict[bytes, bytes]:
        result = dict(existing or {})
        result[PARQUET_METADATA_KEY.encode("utf-8")] = self.to_json_bytes()
        return result

    @classmethod
    def from_parquet_metadata(
        cls, metadata: Optional[Mapping[bytes, bytes]]
    ) -> "HazardDatasetMetadata":
        key = PARQUET_METADATA_KEY.encode("utf-8")
        if metadata is None or key not in metadata:
            raise ValueError("Parquet schema is missing canonical hazard metadata")
        return cls.from_json_bytes(metadata[key])
