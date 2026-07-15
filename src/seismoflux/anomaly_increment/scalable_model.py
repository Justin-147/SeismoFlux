"""Memory-bounded stage-4 point-process objective.

The preregistered fit uses many issue-by-cell rows and a half-day temporal
quadrature.  Materialising one copy of the design matrix for every midpoint
would require several gigabytes.  This module keeps one issue-by-cell matrix
and evaluates the identical midpoint sum in float64 without changing the
objective, gradient, offsets, rate heads, or optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from seismoflux.anomaly_increment.contracts import (
    STAGE4_MAGNITUDE_BIN_IDS,
    DesignMatrix,
    FloatArray,
    FrozenTargetRateHead,
    readonly_float_matrix,
    readonly_float_vector,
)
from seismoflux.anomaly_increment.model import (
    RIDGE_LAMBDA,
    ObjectiveEvaluation,
)


@dataclass(frozen=True, slots=True)
class GroupedMidpointSharedPoissonObjective:
    """Exact dense-objective equivalent with time midpoints kept grouped."""

    issue_design: DesignMatrix
    background_spatial_mass_by_row_and_bin: FloatArray
    midpoint_widths_days: FloatArray
    midpoint_decays: FloatArray
    event_design: DesignMatrix
    event_background_intensity: FloatArray
    event_decay: FloatArray
    event_magnitude_bin_ids: tuple[str, ...]
    rate_head: FrozenTargetRateHead
    ridge_lambda: float = RIDGE_LAMBDA

    def __post_init__(self) -> None:
        mass = readonly_float_matrix(
            "background_spatial_mass_by_row_and_bin",
            self.background_spatial_mass_by_row_and_bin,
            allow_empty_rows=False,
            allow_empty_columns=False,
        )
        widths = readonly_float_vector("midpoint_widths_days", self.midpoint_widths_days)
        decays = readonly_float_vector("midpoint_decays", self.midpoint_decays)
        event_background = readonly_float_vector(
            "event_background_intensity",
            self.event_background_intensity,
            allow_empty=True,
        )
        event_decay = readonly_float_vector("event_decay", self.event_decay, allow_empty=True)
        event_bins = tuple(self.event_magnitude_bin_ids)
        ridge = float(self.ridge_lambda)
        if mass.shape != (self.issue_design.row_count, len(STAGE4_MAGNITUDE_BIN_IDS)):
            raise ValueError("grouped background mass must align with issue rows and two bins")
        if np.any(mass < 0.0) or not np.any(mass > 0.0):
            raise ValueError("grouped background mass must be non-negative and non-empty")
        if widths.shape != decays.shape or np.any(widths <= 0.0):
            raise ValueError("midpoint widths and decays must be aligned and positive")
        if np.any(decays <= 0.0) or np.any(decays > 1.0):
            raise ValueError("midpoint decays must lie in (0, 1]")
        if not math.isclose(
            math.fsum(float(value) for value in widths),
            7.0,
            rel_tol=0.0,
            abs_tol=2.0e-13,
        ):
            raise ValueError("the shared stage-4 fit objective must use the frozen 7-day horizon")
        if self.issue_design.column_names != self.event_design.column_names:
            raise ValueError("grouped issue and event designs must share one coefficient vector")
        if not np.array_equal(
            self.issue_design.penalty_factors,
            self.event_design.penalty_factors,
        ) or not np.array_equal(
            self.issue_design.active_coefficients,
            self.event_design.active_coefficients,
        ):
            raise ValueError("grouped issue and event design metadata differ")
        if event_background.size != self.event_design.row_count:
            raise ValueError("event background intensity must align with event rows")
        if event_decay.size != self.event_design.row_count or len(event_bins) != (
            self.event_design.row_count
        ):
            raise ValueError("event decay and magnitude bins must align with event rows")
        if np.any(event_background <= 0.0):
            raise ValueError("event background intensities must be positive")
        if np.any(event_decay <= 0.0) or np.any(event_decay > 1.0):
            raise ValueError("event decay must lie in (0, 1]")
        for bin_id in event_bins:
            if bin_id not in STAGE4_MAGNITUDE_BIN_IDS:
                raise ValueError("event magnitude-bin identity is not frozen")
            if not self.rate_head.by_id(bin_id).active:
                raise ValueError("inactive exact-zero magnitude bin cannot contain fit events")
        if not math.isfinite(ridge) or ridge != RIDGE_LAMBDA:
            raise ValueError("stage-4 ridge_lambda is frozen at 1.0")
        object.__setattr__(self, "background_spatial_mass_by_row_and_bin", mass)
        object.__setattr__(self, "midpoint_widths_days", widths)
        object.__setattr__(self, "midpoint_decays", decays)
        object.__setattr__(self, "event_background_intensity", event_background)
        object.__setattr__(self, "event_decay", event_decay)
        object.__setattr__(self, "event_magnitude_bin_ids", event_bins)
        object.__setattr__(self, "ridge_lambda", ridge)

    @property
    def quadrature_design(self) -> DesignMatrix:
        """Expose the optimizer's active-coefficient metadata without expansion."""

        return self.issue_design

    @property
    def coefficient_count(self) -> int:
        return self.issue_design.column_count

    @property
    def compensator_base_mass(self) -> FloatArray:
        result = np.asarray(
            self.background_spatial_mass_by_row_and_bin @ self.rate_head.rate_multipliers,
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @property
    def event_log_offsets(self) -> FloatArray:
        offsets = np.empty(self.event_design.row_count, dtype=np.float64)
        for index, (background, bin_id) in enumerate(
            zip(
                self.event_background_intensity,
                self.event_magnitude_bin_ids,
                strict=True,
            )
        ):
            log_rate = self.rate_head.by_id(bin_id).log_rate_multiplier
            if log_rate is None:  # pragma: no cover - rejected in __post_init__
                raise AssertionError("active event bin omitted its finite log rate")
            offsets[index] = math.log(float(background)) + log_rate
        offsets.setflags(write=False)
        return offsets

    def _beta(self, beta: object) -> FloatArray:
        vector = readonly_float_vector("beta", beta)
        if vector.size != self.coefficient_count:
            raise ValueError("beta must have one value per grouped design column")
        return vector

    def evaluate(self, beta: object) -> ObjectiveEvaluation:
        value, _ = self.value_and_gradient(beta)
        beta_vector = self._beta(beta)
        linear = np.asarray(self.issue_design.values @ beta_vector, dtype=np.float64)
        base = self.compensator_base_mass
        compensator = 0.0
        for width, decay in zip(
            self.midpoint_widths_days,
            self.midpoint_decays,
            strict=True,
        ):
            with np.errstate(over="raise", invalid="raise"):
                multiplier = np.exp(float(decay) * linear)
            compensator += float(width) * float(np.dot(base, multiplier))
        event_eta = self.event_decay * np.asarray(
            self.event_design.values @ beta_vector,
            dtype=np.float64,
        )
        event_log_sum = math.fsum(float(item) for item in self.event_log_offsets) + math.fsum(
            float(item) for item in event_eta
        )
        penalty = (
            0.5
            * self.ridge_lambda
            * float(
                np.dot(
                    self.issue_design.penalty_factors,
                    beta_vector * beta_vector,
                )
            )
        )
        # Reuse the value computed by the single-pass path as a consistency guard.
        if not math.isclose(
            value,
            compensator - event_log_sum + penalty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("grouped objective evaluation paths diverged")
        return ObjectiveEvaluation(
            objective=value,
            compensator=compensator,
            event_log_intensity_sum=event_log_sum,
            ridge_penalty=penalty,
        )

    def value_and_gradient(self, beta: object) -> tuple[float, FloatArray]:
        beta_vector = self._beta(beta)
        linear = np.asarray(self.issue_design.values @ beta_vector, dtype=np.float64)
        base = self.compensator_base_mass
        compensator = 0.0
        row_gradient_weight = np.zeros(self.issue_design.row_count, dtype=np.float64)
        for width, decay in zip(
            self.midpoint_widths_days,
            self.midpoint_decays,
            strict=True,
        ):
            width_float = float(width)
            decay_float = float(decay)
            with np.errstate(over="raise", invalid="raise"):
                multiplier = np.exp(decay_float * linear)
            weighted = base * multiplier
            compensator += width_float * float(np.dot(base, multiplier))
            row_gradient_weight += width_float * decay_float * weighted
        event_eta = self.event_decay * np.asarray(
            self.event_design.values @ beta_vector,
            dtype=np.float64,
        )
        event_log_sum = math.fsum(float(item) for item in self.event_log_offsets) + math.fsum(
            float(item) for item in event_eta
        )
        penalty = (
            0.5
            * self.ridge_lambda
            * float(
                np.dot(
                    self.issue_design.penalty_factors,
                    beta_vector * beta_vector,
                )
            )
        )
        objective = compensator - event_log_sum + penalty
        gradient = np.asarray(
            self.issue_design.values.T @ row_gradient_weight,
            dtype=np.float64,
        )
        if self.event_design.row_count:
            gradient -= np.asarray(
                self.event_design.values.T @ self.event_decay,
                dtype=np.float64,
            )
        gradient += self.ridge_lambda * self.issue_design.penalty_factors * beta_vector
        if not math.isfinite(objective) or not np.isfinite(gradient).all():
            raise FloatingPointError("grouped stage-4 objective or gradient is non-finite")
        owned = np.array(gradient, dtype=np.float64, copy=True, order="C")
        owned.setflags(write=False)
        return objective, owned


__all__ = ["GroupedMidpointSharedPoissonObjective"]
