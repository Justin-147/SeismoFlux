"""CPU-float64 shared-coefficient ridge Poisson mathematics for stage 4."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from seismoflux.anomaly_increment.contracts import (
    STAGE4_MAGNITUDE_BIN_IDS,
    DesignMatrix,
    FloatArray,
    FrozenTargetRateHead,
    MagnitudeBinRateHead,
    canonical_mapping_sha256,
    readonly_float_matrix,
    readonly_float_vector,
)
from seismoflux.anomaly_increment.integration import lead_decay

RIDGE_LAMBDA = 1.0
OPTIMIZER_MAXITER = 2_000
OPTIMIZER_MAXFUN = 100_000
OPTIMIZER_FTOL = 1.0e-12
OPTIMIZER_GTOL = 1.0e-7


class PrimaryRateHeadEvidenceInsufficient(RuntimeError):
    """The primary magnitude bin has no fit events, so optimization is forbidden."""


class SharedObjectiveProtocol(Protocol):
    """Small optimizer contract shared by dense and grouped objectives."""

    @property
    def quadrature_design(self) -> DesignMatrix: ...

    @property
    def rate_head(self) -> FrozenTargetRateHead: ...

    @property
    def coefficient_count(self) -> int: ...

    def value_and_gradient(self, beta: object) -> tuple[float, FloatArray]: ...


def _event_count(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def fit_frozen_target_rate_head(
    *,
    training_event_counts: Mapping[str, int],
    background_exposures: Mapping[str, float],
) -> FrozenTargetRateHead:
    """Fit both background-only Poisson heads once, without a pseudocount."""

    expected = set(STAGE4_MAGNITUDE_BIN_IDS)
    if set(training_event_counts) != expected or set(background_exposures) != expected:
        raise ValueError("target-rate inputs must contain exactly M5_6 and M6_plus")
    bins: list[MagnitudeBinRateHead] = []
    for bin_id in STAGE4_MAGNITUDE_BIN_IDS:
        count = _event_count(f"training_event_counts[{bin_id}]", training_event_counts[bin_id])
        exposure = float(background_exposures[bin_id])
        if not math.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("background exposures must be finite and positive")
        if count > 0:
            rate = count / exposure
            bins.append(
                MagnitudeBinRateHead(
                    magnitude_bin_id=bin_id,
                    training_event_count=count,
                    background_exposure=exposure,
                    rate_multiplier=rate,
                    log_rate_multiplier=math.log(rate),
                    status="active",
                )
            )
        elif bin_id == "M5_6":
            bins.append(
                MagnitudeBinRateHead(
                    magnitude_bin_id=bin_id,
                    training_event_count=0,
                    background_exposure=exposure,
                    rate_multiplier=0.0,
                    log_rate_multiplier=None,
                    status="primary_evidence_insufficient_zero_training_events",
                )
            )
        else:
            bins.append(
                MagnitudeBinRateHead(
                    magnitude_bin_id=bin_id,
                    training_event_count=0,
                    background_exposure=exposure,
                    rate_multiplier=0.0,
                    log_rate_multiplier=None,
                    status="inactive_zero_training_events",
                )
            )
    return FrozenTargetRateHead(tuple(bins))


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    """Unnormalized penalized point-process objective components."""

    objective: float
    compensator: float
    event_log_intensity_sum: float
    ridge_penalty: float

    def __post_init__(self) -> None:
        values = (
            self.objective,
            self.compensator,
            self.event_log_intensity_sum,
            self.ridge_penalty,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("objective components must be finite")
        if self.compensator < 0.0 or self.ridge_penalty < 0.0:
            raise ValueError("compensator and ridge penalty must be non-negative")


@dataclass(frozen=True, slots=True)
class SharedPoissonObjective:
    """One shared anomaly beta across the two frozen magnitude-rate heads.

    ``quadrature_background_exposure_by_bin`` contains only background density,
    exact clipped cell area, and midpoint time width.  The rate-head multipliers
    are attached here, once and unchanged, so variants and placebos cannot refit
    them accidentally.
    """

    quadrature_design: DesignMatrix
    quadrature_background_exposure_by_bin: FloatArray
    quadrature_decay: FloatArray
    event_design: DesignMatrix
    event_background_intensity: FloatArray
    event_decay: FloatArray
    event_magnitude_bin_ids: tuple[str, ...]
    rate_head: FrozenTargetRateHead
    ridge_lambda: float = RIDGE_LAMBDA

    def __post_init__(self) -> None:
        exposure = readonly_float_matrix(
            "quadrature_background_exposure_by_bin",
            self.quadrature_background_exposure_by_bin,
            allow_empty_rows=False,
            allow_empty_columns=False,
        )
        quadrature_decay = readonly_float_vector("quadrature_decay", self.quadrature_decay)
        event_background = readonly_float_vector(
            "event_background_intensity",
            self.event_background_intensity,
            allow_empty=True,
        )
        event_decay = readonly_float_vector("event_decay", self.event_decay, allow_empty=True)
        event_bins = tuple(self.event_magnitude_bin_ids)
        ridge = float(self.ridge_lambda)
        if not math.isfinite(ridge) or ridge != RIDGE_LAMBDA:
            raise ValueError("stage-4 ridge_lambda is frozen at 1.0")
        if exposure.shape != (
            self.quadrature_design.row_count,
            len(STAGE4_MAGNITUDE_BIN_IDS),
        ):
            raise ValueError("quadrature exposure must have one row per design row and two bins")
        if np.any(exposure < 0.0) or not np.any(exposure > 0.0):
            raise ValueError("quadrature background exposure must be non-negative and non-empty")
        if quadrature_decay.size != self.quadrature_design.row_count:
            raise ValueError("quadrature decay must align with quadrature rows")
        if np.any(quadrature_decay <= 0.0) or np.any(quadrature_decay > 1.0):
            raise ValueError("quadrature decay must lie in (0, 1]")
        if event_background.size != self.event_design.row_count:
            raise ValueError("event background intensity must align with event design rows")
        if event_decay.size != self.event_design.row_count or len(event_bins) != (
            self.event_design.row_count
        ):
            raise ValueError("event decay and bin identity must align with event rows")
        if np.any(event_background <= 0.0):
            raise ValueError("event background intensities must be positive")
        if np.any(event_decay <= 0.0) or np.any(event_decay > 1.0):
            raise ValueError("event decay must lie in (0, 1]")
        if self.quadrature_design.column_names != self.event_design.column_names:
            raise ValueError("quadrature and event rows must share one coefficient vector")
        if not np.array_equal(
            self.quadrature_design.penalty_factors,
            self.event_design.penalty_factors,
        ) or not np.array_equal(
            self.quadrature_design.active_coefficients,
            self.event_design.active_coefficients,
        ):
            raise ValueError("quadrature and event design metadata must be identical")
        for bin_id in event_bins:
            if bin_id not in STAGE4_MAGNITUDE_BIN_IDS:
                raise ValueError("event magnitude-bin identity is not frozen")
            if not self.rate_head.by_id(bin_id).active:
                raise ValueError("an inactive exact-zero magnitude bin cannot contain fit events")
        object.__setattr__(self, "quadrature_background_exposure_by_bin", exposure)
        object.__setattr__(self, "quadrature_decay", quadrature_decay)
        object.__setattr__(self, "event_background_intensity", event_background)
        object.__setattr__(self, "event_decay", event_decay)
        object.__setattr__(self, "event_magnitude_bin_ids", event_bins)
        object.__setattr__(self, "ridge_lambda", ridge)

    @property
    def coefficient_count(self) -> int:
        return self.quadrature_design.column_count

    @property
    def compensator_base_weights(self) -> FloatArray:
        result = np.asarray(
            self.quadrature_background_exposure_by_bin @ self.rate_head.rate_multipliers,
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
            if log_rate is None:  # pragma: no cover - inactive event bins rejected above
                raise AssertionError("active event bin omitted its finite log rate")
            offsets[index] = math.log(float(background)) + log_rate
        offsets.setflags(write=False)
        return offsets

    def _beta(self, beta: object) -> FloatArray:
        vector = readonly_float_vector("beta", beta)
        if vector.size != self.coefficient_count:
            raise ValueError("beta must have one value per design column")
        return vector

    def evaluate(self, beta: object) -> ObjectiveEvaluation:
        beta_vector = self._beta(beta)
        quadrature_linear = np.asarray(
            self.quadrature_design.values @ beta_vector,
            dtype=np.float64,
        )
        quadrature_eta = self.quadrature_decay * quadrature_linear
        with np.errstate(over="raise", invalid="raise"):
            anomaly_multiplier = np.exp(quadrature_eta)
        compensator = float(np.dot(self.compensator_base_weights, anomaly_multiplier))
        event_eta = self.event_decay * np.asarray(
            self.event_design.values @ beta_vector,
            dtype=np.float64,
        )
        event_log_sum = math.fsum(float(value) for value in self.event_log_offsets) + math.fsum(
            float(value) for value in event_eta
        )
        penalty = (
            0.5
            * self.ridge_lambda
            * float(
                np.dot(
                    self.quadrature_design.penalty_factors,
                    beta_vector * beta_vector,
                )
            )
        )
        return ObjectiveEvaluation(
            objective=compensator - event_log_sum + penalty,
            compensator=compensator,
            event_log_intensity_sum=event_log_sum,
            ridge_penalty=penalty,
        )

    def value_and_gradient(self, beta: object) -> tuple[float, FloatArray]:
        """Return the exact CPU-float64 objective and analytic gradient."""

        beta_vector = self._beta(beta)
        evaluation = self.evaluate(beta_vector)
        quadrature_linear = np.asarray(
            self.quadrature_design.values @ beta_vector,
            dtype=np.float64,
        )
        with np.errstate(over="raise", invalid="raise"):
            weighted = (
                self.compensator_base_weights
                * np.exp(self.quadrature_decay * quadrature_linear)
                * self.quadrature_decay
            )
        gradient = np.asarray(
            self.quadrature_design.values.T @ weighted,
            dtype=np.float64,
        )
        if self.event_design.row_count:
            gradient -= np.asarray(
                self.event_design.values.T @ self.event_decay,
                dtype=np.float64,
            )
        gradient += self.ridge_lambda * self.quadrature_design.penalty_factors * beta_vector
        if not np.isfinite(gradient).all():
            raise FloatingPointError("ridge Poisson gradient is non-finite")
        gradient = np.array(gradient, dtype=np.float64, copy=True, order="C")
        gradient.setflags(write=False)
        return evaluation.objective, gradient


FitStatus = Literal["converged", "no_active_coefficients", "optimizer_failed"]


@dataclass(frozen=True, slots=True)
class RidgePoissonFitResult:
    """Deterministic record of the one frozen L-BFGS-B fit."""

    beta: FloatArray
    objective: float
    maximum_absolute_active_gradient: float
    iteration_count: int
    function_evaluation_count: int
    status: FitStatus
    optimizer_message: str
    rate_head_sha256: str

    def __post_init__(self) -> None:
        beta = readonly_float_vector("beta", self.beta)
        if not math.isfinite(self.objective):
            raise ValueError("fit objective must be finite")
        if (
            not math.isfinite(self.maximum_absolute_active_gradient)
            or self.maximum_absolute_active_gradient < 0.0
        ):
            raise ValueError("maximum active gradient must be finite and non-negative")
        if self.iteration_count < 0 or self.function_evaluation_count < 0:
            raise ValueError("optimizer counts must be non-negative")
        if len(self.rate_head_sha256) != 64:
            raise ValueError("fit result must bind a SHA-256 rate-head identity")
        object.__setattr__(self, "beta", beta)

    @property
    def converged(self) -> bool:
        return self.status in ("converged", "no_active_coefficients")

    def as_mapping(self) -> dict[str, object]:
        return {
            "backend": "cpu_numpy_scipy_float64",
            "beta": [float(value) for value in self.beta],
            "deterministic_zero_start": True,
            "function_evaluation_count": self.function_evaluation_count,
            "iteration_count": self.iteration_count,
            "maximum_absolute_active_gradient": self.maximum_absolute_active_gradient,
            "objective": self.objective,
            "optimizer": "scipy_L-BFGS-B",
            "optimizer_message": self.optimizer_message,
            "rate_head_sha256": self.rate_head_sha256,
            "status": self.status,
        }

    @property
    def sha256(self) -> str:
        return canonical_mapping_sha256(self.as_mapping())


def fit_shared_ridge_poisson(objective: SharedObjectiveProtocol) -> RidgePoissonFitResult:
    """Fit only active coefficients from an all-zero deterministic start."""

    if objective.rate_head.primary_evidence_insufficient:
        raise PrimaryRateHeadEvidenceInsufficient(
            "M5_6 has zero training events; the optimizer must not run"
        )
    active = objective.quadrature_design.active_coefficients
    active_indices = np.flatnonzero(active)
    beta = np.zeros(objective.coefficient_count, dtype=np.float64)
    if active_indices.size == 0:
        value, gradient = objective.value_and_gradient(beta)
        return RidgePoissonFitResult(
            beta=beta,
            objective=value,
            maximum_absolute_active_gradient=0.0,
            iteration_count=0,
            function_evaluation_count=1,
            status="no_active_coefficients",
            optimizer_message="all coefficients fixed zero by the fit-scope preprocessor",
            rate_head_sha256=objective.rate_head.sha256,
        )

    def reduced_value_and_gradient(theta: FloatArray) -> tuple[float, FloatArray]:
        full_beta = np.zeros(objective.coefficient_count, dtype=np.float64)
        full_beta[active_indices] = theta
        value, full_gradient = objective.value_and_gradient(full_beta)
        return value, np.asarray(full_gradient[active_indices], dtype=np.float64)

    result = minimize(
        reduced_value_and_gradient,
        np.zeros(active_indices.size, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": OPTIMIZER_MAXITER,
            "maxfun": OPTIMIZER_MAXFUN,
            "ftol": OPTIMIZER_FTOL,
            "gtol": OPTIMIZER_GTOL,
        },
    )
    fitted_active = np.asarray(result.x, dtype=np.float64)
    beta[active_indices] = fitted_active
    value, gradient = objective.value_and_gradient(beta)
    max_gradient = float(np.max(np.abs(gradient[active_indices])))
    status: FitStatus = "converged" if bool(result.success) else "optimizer_failed"
    return RidgePoissonFitResult(
        beta=beta,
        objective=value,
        maximum_absolute_active_gradient=max_gradient,
        iteration_count=int(result.nit),
        function_evaluation_count=int(result.nfev),
        status=status,
        optimizer_message=str(result.message),
        rate_head_sha256=objective.rate_head.sha256,
    )


def conditional_intensity(
    *,
    background_intensity: object,
    design_values: object,
    beta: object,
    lead_days: float | Sequence[float] | FloatArray,
    magnitude_bin_id: str,
    rate_head: FrozenTargetRateHead,
    increment_enabled: bool = True,
) -> FloatArray:
    """Evaluate conditional relative intensity; this is not an absolute probability."""

    background = readonly_float_vector("background_intensity", background_intensity)
    if np.any(background < 0.0):
        raise ValueError("background intensity must be non-negative")
    design = readonly_float_matrix(
        "design_values",
        design_values,
        allow_empty_rows=False,
        allow_empty_columns=False,
    )
    beta_vector = readonly_float_vector("beta", beta)
    if design.shape != (background.size, beta_vector.size):
        raise ValueError("design values must align with background rows and beta columns")
    rate = rate_head.by_id(magnitude_bin_id).rate_multiplier
    base = np.asarray(background * rate, dtype=np.float64)
    if rate == 0.0 or not increment_enabled:
        result = np.array(base, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result
    lead_values = np.asarray(lead_days, dtype=np.float64)
    if lead_values.ndim == 0:
        lead_values = np.full(background.size, float(lead_values), dtype=np.float64)
    elif lead_values.ndim != 1 or lead_values.size != background.size:
        raise ValueError("lead_days must be a scalar or one value per intensity row")
    decay = lead_decay(lead_values)
    if not isinstance(decay, np.ndarray):  # pragma: no cover - lead_values is an array
        raise AssertionError("array lead decay unexpectedly returned a scalar")
    eta = decay * np.asarray(design @ beta_vector, dtype=np.float64)
    if not np.any(eta):
        result = np.array(base, dtype=np.float64, copy=True, order="C")
    else:
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(base * np.exp(eta), dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("conditional intensity is non-finite")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


__all__ = [
    "OPTIMIZER_FTOL",
    "OPTIMIZER_GTOL",
    "OPTIMIZER_MAXFUN",
    "OPTIMIZER_MAXITER",
    "RIDGE_LAMBDA",
    "ObjectiveEvaluation",
    "PrimaryRateHeadEvidenceInsufficient",
    "RidgePoissonFitResult",
    "SharedObjectiveProtocol",
    "SharedPoissonObjective",
    "conditional_intensity",
    "fit_frozen_target_rate_head",
    "fit_shared_ridge_poisson",
]
