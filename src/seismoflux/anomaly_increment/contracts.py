"""Immutable, target-blind contracts for the stage-4 scoring mathematics.

The objects in this module deliberately contain only already-assembled numerical
terms.  They do not know how to locate a catalogue, a target table, or a locked
test partition.  Every NumPy input is copied into an owned, read-only float64
array so a frozen fit cannot be changed through an alias held by its caller.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from seismoflux.data.common import canonical_json_bytes

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
TransformName: TypeAlias = Literal[
    "identity_finite",
    "identity_binary",
    "log1p_nonnegative",
    "asinh_signed",
]
ScaleBranch: TypeAlias = Literal[
    "none_binary",
    "1.4826_times_training_MAD",
    "training_IQR_divided_by_1.349",
    "training_population_standard_deviation",
    "fixed_zero_training_constant",
    "fixed_zero_no_finite_values",
]
RateHeadStatus: TypeAlias = Literal[
    "active",
    "inactive_zero_training_events",
    "primary_evidence_insufficient_zero_training_events",
]

STAGE4_MATHEMATICS_PROTOCOL_VERSION = "0.4.1"
STAGE4_MAGNITUDE_BIN_IDS = ("M5_6", "M6_plus")


def readonly_float_vector(name: str, value: object, *, allow_empty: bool = False) -> FloatArray:
    """Return an owned read-only finite float64 vector."""

    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def readonly_float_matrix(
    name: str,
    value: object,
    *,
    allow_empty_rows: bool = False,
    allow_empty_columns: bool = False,
) -> FloatArray:
    """Return an owned read-only finite float64 matrix."""

    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not allow_empty_rows and result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not allow_empty_columns and result.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def readonly_bool_vector(name: str, value: object, *, allow_empty: bool = False) -> BoolArray:
    """Return an owned read-only boolean vector, rejecting numeric coercion."""

    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if raw.dtype.kind != "b":
        raise TypeError(f"{name} must contain booleans")
    result = np.array(raw, dtype=np.bool_, copy=True, order="C")
    result.setflags(write=False)
    return result


def canonical_mapping_sha256(value: object) -> str:
    """Hash a JSON-safe mapping using the repository canonical encoder."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _nonempty_identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


@dataclass(frozen=True, slots=True)
class FeatureColumnContract:
    """One logical value and its mandatory original-null indicator."""

    source_column: str
    logical_feature: str
    value_output_column: str
    missing_output_column: str
    transform: TransformName
    penalty_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "source_column",
            "logical_feature",
            "value_output_column",
            "missing_output_column",
        ):
            object.__setattr__(self, name, _nonempty_identifier(name, getattr(self, name)))
        if self.value_output_column == self.missing_output_column:
            raise ValueError("value and missing-indicator output columns must differ")
        if self.transform not in (
            "identity_finite",
            "identity_binary",
            "log1p_nonnegative",
            "asinh_signed",
        ):
            raise ValueError("unsupported frozen feature transform")
        factor = float(self.penalty_factor)
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError("penalty_factor must be finite and non-negative")
        object.__setattr__(self, "penalty_factor", factor)

    @property
    def is_binary(self) -> bool:
        return self.transform == "identity_binary"

    def as_mapping(self) -> dict[str, object]:
        return {
            "logical_feature": self.logical_feature,
            "missing_output_column": self.missing_output_column,
            "penalty_factor": self.penalty_factor,
            "source_column": self.source_column,
            "transform": self.transform,
            "value_output_column": self.value_output_column,
        }


@dataclass(frozen=True, slots=True)
class DesignMatrix:
    """A frozen model matrix plus coefficient constraints and ridge factors."""

    values: FloatArray
    column_names: tuple[str, ...]
    penalty_factors: FloatArray
    active_coefficients: BoolArray

    def __post_init__(self) -> None:
        names = tuple(self.column_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("design column names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("design column names must be unique")
        values = readonly_float_matrix(
            "values",
            self.values,
            allow_empty_rows=True,
            allow_empty_columns=False,
        )
        penalties = readonly_float_vector("penalty_factors", self.penalty_factors)
        active = readonly_bool_vector("active_coefficients", self.active_coefficients)
        if values.shape[1] != len(names):
            raise ValueError("design matrix width must equal the column-name count")
        if penalties.size != len(names) or active.size != len(names):
            raise ValueError("design metadata must have one entry per column")
        if np.any(penalties < 0.0):
            raise ValueError("design penalty factors must be non-negative")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "column_names", names)
        object.__setattr__(self, "penalty_factors", penalties)
        object.__setattr__(self, "active_coefficients", active)

    @property
    def row_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def column_count(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class MagnitudeBinRateHead:
    """One frozen background-only Poisson rate multiplier."""

    magnitude_bin_id: str
    training_event_count: int
    background_exposure: float
    rate_multiplier: float
    log_rate_multiplier: float | None
    status: RateHeadStatus

    def __post_init__(self) -> None:
        bin_id = _nonempty_identifier("magnitude_bin_id", self.magnitude_bin_id)
        if bin_id not in STAGE4_MAGNITUDE_BIN_IDS:
            raise ValueError("magnitude bin is not part of the frozen stage-4 joint fit")
        count = self.training_event_count
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("training_event_count must be a non-negative integer")
        exposure = float(self.background_exposure)
        rate = float(self.rate_multiplier)
        if not math.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("background_exposure must be finite and positive")
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("rate_multiplier must be finite and non-negative")
        if self.status == "active":
            if count == 0 or rate <= 0.0 or self.log_rate_multiplier is None:
                raise ValueError("an active rate head requires events and a positive finite rate")
            log_rate = float(self.log_rate_multiplier)
            if not math.isfinite(log_rate) or not math.isclose(
                log_rate,
                math.log(rate),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("log_rate_multiplier must be the natural log of the rate")
            expected = count / exposure
            if not math.isclose(rate, expected, rel_tol=1.0e-15, abs_tol=0.0):
                raise ValueError("rate head must be the background-only Poisson MLE")
        else:
            if count != 0 or rate != 0.0 or self.log_rate_multiplier is not None:
                raise ValueError(
                    "a zero-event rate head must use exact zero and no numeric log rate"
                )
            if bin_id == "M5_6" and self.status != (
                "primary_evidence_insufficient_zero_training_events"
            ):
                raise ValueError("zero-event M5_6 must stop as primary evidence insufficient")
            if bin_id == "M6_plus" and self.status != "inactive_zero_training_events":
                raise ValueError("zero-event M6_plus must be inactive")
        object.__setattr__(self, "magnitude_bin_id", bin_id)
        object.__setattr__(self, "background_exposure", exposure)
        object.__setattr__(self, "rate_multiplier", rate)

    @property
    def active(self) -> bool:
        return self.status == "active"

    def as_mapping(self) -> dict[str, object]:
        return {
            "active": self.active,
            "background_exposure": self.background_exposure,
            "log_rate_multiplier": self.log_rate_multiplier,
            "magnitude_bin_id": self.magnitude_bin_id,
            "pseudocount": 0,
            "rate_multiplier": self.rate_multiplier,
            "status": self.status,
            "training_event_count": self.training_event_count,
        }


@dataclass(frozen=True, slots=True)
class FrozenTargetRateHead:
    """The two bin-specific heads shared unchanged by every anomaly variant."""

    bins: tuple[MagnitudeBinRateHead, ...]
    protocol_version: str = STAGE4_MATHEMATICS_PROTOCOL_VERSION
    fit_family: str = "background_only_poisson_mle_no_pseudocount"

    def __post_init__(self) -> None:
        bins = tuple(self.bins)
        if tuple(item.magnitude_bin_id for item in bins) != STAGE4_MAGNITUDE_BIN_IDS:
            raise ValueError("rate heads must use the frozen M5_6, M6_plus order")
        if self.protocol_version != STAGE4_MATHEMATICS_PROTOCOL_VERSION:
            raise ValueError("rate head protocol version must be 0.4.1")
        if self.fit_family != "background_only_poisson_mle_no_pseudocount":
            raise ValueError("rate head family must forbid pseudocounts")
        object.__setattr__(self, "bins", bins)

    @property
    def primary_evidence_insufficient(self) -> bool:
        return self.bins[0].status == ("primary_evidence_insufficient_zero_training_events")

    @property
    def rate_multipliers(self) -> FloatArray:
        return readonly_float_vector(
            "rate_multipliers",
            [item.rate_multiplier for item in self.bins],
        )

    def by_id(self, magnitude_bin_id: str) -> MagnitudeBinRateHead:
        for item in self.bins:
            if item.magnitude_bin_id == magnitude_bin_id:
                return item
        raise KeyError(magnitude_bin_id)

    def as_mapping(self) -> dict[str, object]:
        return {
            "bins": [item.as_mapping() for item in self.bins],
            "fit_family": self.fit_family,
            "frozen_before_variant_fits": True,
            "protocol_version": self.protocol_version,
            "shared_across_all_variants_and_placebos": True,
            "validation_refit_forbidden": True,
        }

    @property
    def sha256(self) -> str:
        return canonical_mapping_sha256(self.as_mapping())


__all__ = [
    "STAGE4_MAGNITUDE_BIN_IDS",
    "STAGE4_MATHEMATICS_PROTOCOL_VERSION",
    "BoolArray",
    "DesignMatrix",
    "FeatureColumnContract",
    "FloatArray",
    "FrozenTargetRateHead",
    "MagnitudeBinRateHead",
    "RateHeadStatus",
    "ScaleBranch",
    "TransformName",
    "canonical_mapping_sha256",
    "readonly_bool_vector",
    "readonly_float_matrix",
    "readonly_float_vector",
]
