"""Conditional spatial ridge model for the D1 retrospective replay.

The total earthquake rate is profiled out issue by issue.  The fitted object
therefore changes only the spatial distribution relative to a causal base mass;
it does not add an intercept and it does not estimate the number of earthquakes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ScalarType = TypeVar("ScalarType", bound=np.generic)


def _readonly(array: NDArray[ScalarType]) -> NDArray[ScalarType]:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return cast(NDArray[ScalarType], result)


class D1ModelFitError(RuntimeError):
    """Raised when a D1 optimization cannot produce one valid preregistered fit."""


@dataclass(frozen=True, slots=True)
class ConditionalSpatialTrainingIssue:
    """One issue's causal base, feature design, and future 30-day M4+ counts."""

    issue_id: str
    log_base_mass: FloatArray
    design: FloatArray
    future_counts: FloatArray

    def __post_init__(self) -> None:
        base = np.asarray(self.log_base_mass, dtype=np.float64)
        design = np.asarray(self.design, dtype=np.float64)
        counts = np.asarray(self.future_counts, dtype=np.float64)
        if not self.issue_id:
            raise ValueError("conditional spatial issue_id must be non-empty")
        if base.ndim != 1 or design.ndim != 2 or counts.ndim != 1:
            raise ValueError("conditional spatial issue arrays have invalid dimensions")
        if design.shape[0] != base.size or counts.shape != base.shape or base.size == 0:
            raise ValueError("conditional spatial issue arrays do not align by cell")
        if not np.isfinite(base).all() or not np.isfinite(design).all():
            raise ValueError("base log mass and D1 design must be finite")
        if (
            not np.isfinite(counts).all()
            or np.any(counts < 0.0)
            or not np.array_equal(counts, np.floor(counts))
        ):
            raise ValueError("future M4+ cell counts must be finite non-negative integers")
        object.__setattr__(self, "log_base_mass", _readonly(base))
        object.__setattr__(self, "design", _readonly(design))
        object.__setattr__(self, "future_counts", _readonly(counts))

    @property
    def event_count(self) -> int:
        return int(np.sum(self.future_counts, dtype=np.float64))

    @property
    def coefficient_count(self) -> int:
        return int(self.design.shape[1])


@dataclass(frozen=True, slots=True)
class ConditionalObjectiveEvaluation:
    objective: float
    equal_event_negative_log_likelihood: float
    ridge_penalty: float
    event_count: int


@dataclass(frozen=True, slots=True)
class ConditionalSpatialRidgeObjective:
    """Pooled equal-event conditional multinomial objective with analytic gradient."""

    issues: tuple[ConditionalSpatialTrainingIssue, ...]
    ridge_lambda: float
    active_coefficients: BoolArray

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        ridge = float(self.ridge_lambda)
        if not issues:
            raise ValueError("conditional spatial objective requires training issues")
        coefficient_count = issues[0].coefficient_count
        if any(item.coefficient_count != coefficient_count for item in issues):
            raise ValueError("conditional spatial issue designs differ in column count")
        active = np.asarray(self.active_coefficients, dtype=np.bool_)
        if active.shape != (coefficient_count,):
            raise ValueError("active coefficient mask does not align with D1 design")
        if not math.isfinite(ridge) or ridge < 0.0:
            raise ValueError("D1 ridge lambda must be finite and non-negative")
        if sum(item.event_count for item in issues) == 0:
            raise ValueError("conditional spatial fit has no future M4+ training events")
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "ridge_lambda", ridge)
        object.__setattr__(self, "active_coefficients", _readonly(active))

    @property
    def coefficient_count(self) -> int:
        return self.active_coefficients.size

    @property
    def event_count(self) -> int:
        return sum(item.event_count for item in self.issues)

    def _beta(self, beta: object) -> FloatArray:
        vector = np.asarray(beta, dtype=np.float64)
        if vector.shape != (self.coefficient_count,) or not np.isfinite(vector).all():
            raise ValueError("beta must be one finite float64 value per D1 design column")
        return vector

    def evaluate(self, beta: object) -> ConditionalObjectiveEvaluation:
        vector = self._beta(beta)
        effective = np.where(self.active_coefficients, vector, 0.0)
        negative_log_likelihood_sum = 0.0
        for issue in self.issues:
            event_count = issue.event_count
            if event_count == 0:
                continue
            eta = issue.log_base_mass + issue.design @ effective
            normalizer = float(logsumexp(eta))
            if not math.isfinite(normalizer):
                raise FloatingPointError("D1 conditional log normalizer is non-finite")
            negative_log_likelihood_sum += event_count * normalizer - float(
                np.dot(issue.future_counts, eta)
            )
        mean_nll = negative_log_likelihood_sum / self.event_count
        penalty = 0.5 * self.ridge_lambda * float(np.dot(effective, effective))
        objective = mean_nll + penalty
        if not math.isfinite(objective):
            raise FloatingPointError("D1 conditional objective is non-finite")
        return ConditionalObjectiveEvaluation(
            objective=objective,
            equal_event_negative_log_likelihood=mean_nll,
            ridge_penalty=penalty,
            event_count=self.event_count,
        )

    def value_and_gradient(self, beta: object) -> tuple[float, FloatArray]:
        vector = self._beta(beta)
        effective = np.where(self.active_coefficients, vector, 0.0)
        negative_log_likelihood_sum = 0.0
        gradient_sum = np.zeros(self.coefficient_count, dtype=np.float64)
        for issue in self.issues:
            event_count = issue.event_count
            if event_count == 0:
                continue
            eta = issue.log_base_mass + issue.design @ effective
            normalizer = float(logsumexp(eta))
            probabilities = np.asarray(np.exp(eta - normalizer), dtype=np.float64)
            if (
                not math.isfinite(normalizer)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0.0)
            ):
                raise FloatingPointError(
                    "D1 conditional normalization is non-positive or non-finite"
                )
            negative_log_likelihood_sum += event_count * normalizer - float(
                np.dot(issue.future_counts, eta)
            )
            gradient_sum += issue.design.T @ (event_count * probabilities - issue.future_counts)
        gradient = gradient_sum / self.event_count + self.ridge_lambda * effective
        gradient[~self.active_coefficients] = 0.0
        objective = (
            negative_log_likelihood_sum / self.event_count
            + 0.5 * self.ridge_lambda * float(np.dot(effective, effective))
        )
        if not math.isfinite(objective) or not np.isfinite(gradient).all():
            raise FloatingPointError("D1 conditional objective or gradient is non-finite")
        return objective, _readonly(gradient)


def predict_conditional_cell_mass(
    log_base_mass: object,
    design: object,
    coefficients: object,
    *,
    active_coefficients: object | None = None,
) -> FloatArray:
    """Return exactly normalized cell probability mass without clipping or epsilon."""

    base = np.asarray(log_base_mass, dtype=np.float64)
    matrix = np.asarray(design, dtype=np.float64)
    beta = np.asarray(coefficients, dtype=np.float64)
    if base.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != base.size:
        raise ValueError("D1 prediction base/design arrays do not align")
    if beta.shape != (matrix.shape[1],):
        raise ValueError("D1 prediction coefficients do not align with design columns")
    if not np.isfinite(base).all() or not np.isfinite(matrix).all() or not np.isfinite(beta).all():
        raise ValueError("D1 prediction inputs must be finite")
    if active_coefficients is None:
        active = np.ones(beta.shape, dtype=np.bool_)
    else:
        active = np.asarray(active_coefficients, dtype=np.bool_)
        if active.shape != beta.shape:
            raise ValueError("D1 prediction active mask does not align with coefficients")
    effective = np.where(active, beta, 0.0)
    eta = base + matrix @ effective
    normalizer = float(logsumexp(eta))
    probabilities = np.asarray(np.exp(eta - normalizer), dtype=np.float64)
    total = float(np.sum(probabilities, dtype=np.float64))
    if (
        not math.isfinite(normalizer)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0.0)
        or total <= 0.0
    ):
        raise FloatingPointError("D1 prediction normalization failed")
    probabilities /= total
    if not np.isfinite(probabilities).all() or not math.isclose(
        float(np.sum(probabilities, dtype=np.float64)),
        1.0,
        rel_tol=0.0,
        abs_tol=5.0e-15,
    ):
        raise FloatingPointError("D1 predicted cell mass is not normalized")
    return _readonly(probabilities)


@dataclass(frozen=True, slots=True)
class ConditionalSpatialRidgeFit:
    coefficients: FloatArray
    active_coefficients: BoolArray
    ridge_lambda: float
    objective: float
    iteration_count: int
    training_event_count: int

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        active = np.asarray(self.active_coefficients, dtype=np.bool_)
        if coefficients.ndim != 1 or active.shape != coefficients.shape:
            raise ValueError("D1 fitted coefficients/active mask do not align")
        if not np.isfinite(coefficients).all() or np.any(coefficients[~active] != 0.0):
            raise ValueError("D1 fitted coefficients are invalid")
        if (
            not math.isfinite(float(self.objective))
            or self.iteration_count < 0
            or self.training_event_count <= 0
        ):
            raise ValueError("D1 fit diagnostics are invalid")
        object.__setattr__(self, "coefficients", _readonly(coefficients))
        object.__setattr__(self, "active_coefficients", _readonly(active))

    def predict(self, log_base_mass: object, design: object) -> FloatArray:
        return predict_conditional_cell_mass(
            log_base_mass,
            design,
            self.coefficients,
            active_coefficients=self.active_coefficients,
        )


def fit_conditional_spatial_ridge(
    issues: tuple[ConditionalSpatialTrainingIssue, ...],
    *,
    ridge_lambda: float,
    active_coefficients: object,
    max_iterations: int = 2_000,
    ftol: float = 1.0e-10,
    gtol: float = 1.0e-6,
) -> ConditionalSpatialRidgeFit:
    """Fit one float64 L-BFGS-B model, failing closed on any invalid result."""

    objective = ConditionalSpatialRidgeObjective(
        issues=tuple(issues),
        ridge_lambda=ridge_lambda,
        active_coefficients=np.asarray(active_coefficients, dtype=np.bool_),
    )
    if max_iterations <= 0 or not math.isfinite(ftol) or ftol <= 0.0:
        raise ValueError("D1 optimizer max_iterations/ftol are invalid")
    if not math.isfinite(gtol) or gtol <= 0.0:
        raise ValueError("D1 optimizer gtol is invalid")
    initial = np.zeros(objective.coefficient_count, dtype=np.float64)
    if objective.coefficient_count == 0:
        value, _ = objective.value_and_gradient(initial)
        return ConditionalSpatialRidgeFit(
            coefficients=initial,
            active_coefficients=objective.active_coefficients,
            ridge_lambda=float(ridge_lambda),
            objective=value,
            iteration_count=0,
            training_event_count=objective.event_count,
        )
    bounds = [(None, None) if active else (0.0, 0.0) for active in objective.active_coefficients]
    try:
        result = minimize(
            objective.value_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(max_iterations),
                "ftol": float(ftol),
                "gtol": float(gtol),
                "maxls": 50,
            },
        )
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise D1ModelFitError("D1 conditional optimization produced an invalid value") from exc
    coefficients = np.asarray(result.x, dtype=np.float64)
    gradient = np.asarray(result.jac, dtype=np.float64)
    if (
        not bool(result.success)
        or coefficients.shape != initial.shape
        or not np.isfinite(coefficients).all()
        or not math.isfinite(float(result.fun))
        or gradient.shape != initial.shape
        or not np.isfinite(gradient).all()
    ):
        raise D1ModelFitError(f"D1 conditional optimization did not converge: {result.message!s}")
    coefficients[~objective.active_coefficients] = 0.0
    checked_value, checked_gradient = objective.value_and_gradient(coefficients)
    if not math.isfinite(checked_value) or not np.isfinite(checked_gradient).all():
        raise D1ModelFitError("D1 conditional optimum failed the final finite check")
    return ConditionalSpatialRidgeFit(
        coefficients=coefficients,
        active_coefficients=objective.active_coefficients,
        ridge_lambda=float(ridge_lambda),
        objective=checked_value,
        iteration_count=int(result.nit),
        training_event_count=objective.event_count,
    )


__all__ = [
    "ConditionalObjectiveEvaluation",
    "ConditionalSpatialRidgeFit",
    "ConditionalSpatialRidgeObjective",
    "ConditionalSpatialTrainingIssue",
    "D1ModelFitError",
    "fit_conditional_spatial_ridge",
    "predict_conditional_cell_mass",
]
