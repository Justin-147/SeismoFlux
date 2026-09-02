"""Pure count-increment mathematics, with no dates, target loading, or model selection.

The offset is an expected count, not a spatial score.  The Poisson count mean is
``offset * exp(intercept + standardized_features @ coefficients)``.  These models
do not establish that model-based occurrence probabilities are calibrated.
"""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import gammaln  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ArrayScalar = TypeVar("ArrayScalar", bound=np.generic)


def _readonly(values: NDArray[ArrayScalar]) -> NDArray[ArrayScalar]:
    result = np.array(values, copy=True, order="C")
    result.setflags(write=False)
    return cast(NDArray[ArrayScalar], result)


def _offsets(values: object, size: int) -> FloatArray:
    offsets = np.asarray(values, dtype=np.float64)
    if offsets.shape != (size,) or not np.isfinite(offsets).all() or np.any(offsets <= 0):
        raise ValueError("one finite positive expected-count offset is required per issue")
    return offsets


def _design(values: object) -> FloatArray:
    features = np.asarray(values, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("features must have finite shape (issues, features)")
    return features


def _counts(values: object, size: int) -> FloatArray:
    counts = np.asarray(values, dtype=np.float64)
    if (
        counts.shape != (size,)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.array_equal(counts, np.floor(counts))
        or not math.isfinite(float(np.sum(counts / max(size, 1), dtype=np.float64)))
    ):
        raise ValueError("counts must be aligned finite non-negative integers")
    return counts


def _ridge(value: float) -> float:
    ridge = float(value)
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    return ridge


def offset_poisson_value_and_gradient(
    parameters: FloatArray,
    standardized_features: FloatArray,
    offsets: FloatArray,
    counts: FloatArray,
    *,
    ridge_lambda: float,
) -> tuple[float, FloatArray]:
    """Equal-issue mean Poisson NLL plus ridge; the intercept is not penalized.

    ``parameters[0]`` is the intercept.  Zero-count issues remain in both the
    likelihood and its denominator.  This function fits no preprocessing state.
    Non-finite arithmetic raises instead of clipping means or parameters.
    """
    features = _design(standardized_features)
    size, feature_count = features.shape
    if size == 0:
        raise ValueError("the objective needs at least one training issue")
    offset = _offsets(offsets, size)
    target = _counts(counts, size)
    params = np.asarray(parameters, dtype=np.float64)
    if params.shape != (feature_count + 1,) or not np.isfinite(params).all():
        raise ValueError("parameters must be finite and align with intercept plus features")
    ridge = _ridge(ridge_lambda)
    beta = params[1:]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        log_mean = np.log(offset) + params[0] + features @ beta
        mean = np.exp(log_mean)
        losses = mean - target * log_mean + gammaln(target + 1.0)
        value = float(np.sum(losses / size)) + 0.5 * ridge * float(beta @ beta)
        residual = (mean - target) / size
        gradient = np.concatenate(
            (np.array([np.sum(residual)]), features.T @ residual + ridge * beta)
        )
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("offset Poisson objective or gradient is non-finite")
    return value, gradient


def _training_scaler(features: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Equal-issue population moments from the supplied training rows only."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        center = np.sum(features / features.shape[0], axis=0)
        lower, upper = np.min(features, axis=0), np.max(features, axis=0)
        active = upper > lower
        center[~active] = lower[~active]
        deviation = np.maximum(np.abs(lower - center), np.abs(upper - center))
        divisor = np.where(active, deviation, 1.0)
        variance = np.mean(np.square((features - center) / divisor), axis=0)
        scale = np.where(active, divisor * np.sqrt(variance), 1.0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise FloatingPointError("training scaler is not finite and positive")
    return center, scale, active


@dataclass(frozen=True, slots=True)
class OffsetPoissonFit:
    coefficients: FloatArray
    intercept: float
    center: FloatArray
    scale: FloatArray
    active_coefficients: BoolArray
    ridge_lambda: float
    training_issue_count: int
    event_count: int
    objective: float | None
    iteration_count: int
    status: str
    message: str

    def __post_init__(self) -> None:
        beta = np.asarray(self.coefficients, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        active = np.asarray(self.active_coefficients, dtype=np.bool_)
        if beta.ndim != 1 or any(item.shape != beta.shape for item in (center, scale, active)):
            raise ValueError("fit coefficient and scaler arrays must align")
        if (
            not all(np.isfinite(item).all() for item in (beta, center, scale))
            or np.any(scale <= 0)
            or np.any(beta[~active] != 0)
            or not math.isfinite(self.intercept)
            or (self.objective is not None and not math.isfinite(self.objective))
        ):
            raise ValueError("fitted parameters, scaler, and objective must be finite")
        if self.status.startswith("baseline_") and (self.intercept != 0 or np.any(beta != 0)):
            raise ValueError("a baseline fallback must have exactly zero correction")
        for name, value in (
            ("coefficients", beta),
            ("center", center),
            ("scale", scale),
            ("active_coefficients", active),
        ):
            object.__setattr__(self, name, _readonly(value))

    def predict_log_mean(self, features: FloatArray, offsets: FloatArray) -> FloatArray:
        """Return finite log expected counts for scoring, without labels or refitting.

        Use this output directly in the Poisson log-likelihood.  Taking the log of
        ``predict`` instead can lose finite log means when their exponent underflows.
        """
        design = _design(features)
        if design.shape[1] != self.coefficients.size:
            raise ValueError("prediction feature columns must align with the fitted model")
        offset = _offsets(offsets, design.shape[0])
        if self.status.startswith("baseline_"):
            return _readonly(np.log(offset))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaled = np.zeros_like(design)
            active = self.active_coefficients
            scaled[:, active] = (design[:, active] - self.center[active]) / self.scale[active]
            log_mean = np.log(offset) + self.intercept + scaled @ self.coefficients
        if not np.isfinite(log_mean).all():
            raise FloatingPointError("predicted log expected counts are non-finite")
        return _readonly(log_mean)

    def predict(self, features: FloatArray, offsets: FloatArray) -> FloatArray:
        """Exponentiate the frozen log mean; a fallback preserves the exact offset."""
        log_mean = self.predict_log_mean(features, offsets)
        if self.status.startswith("baseline_"):
            return _readonly(np.asarray(offsets, dtype=np.float64))
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            prediction = np.exp(log_mean)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("predicted expected counts are non-finite")
        return _readonly(prediction)

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "active_coefficients": self.active_coefficients.tolist(),
            "ridge_lambda": self.ridge_lambda,
            "training_issue_count": self.training_issue_count,
            "event_count": self.event_count,
            "objective": self.objective,
            "iteration_count": self.iteration_count,
            "status": self.status,
            "message": self.message,
        }


def _fallback(
    features: FloatArray,
    offsets: FloatArray,
    counts: FloatArray,
    ridge_lambda: float,
    *,
    status: str,
    message: str,
) -> OffsetPoissonFit:
    feature_count = features.shape[1]
    objective = None
    if features.shape[0]:
        with suppress(FloatingPointError, OverflowError, ValueError):
            objective, _ = offset_poisson_value_and_gradient(
                np.zeros(feature_count + 1),
                np.zeros_like(features),
                offsets,
                counts,
                ridge_lambda=ridge_lambda,
            )
    return OffsetPoissonFit(
        coefficients=np.zeros(feature_count),
        intercept=0.0,
        center=np.zeros(feature_count),
        scale=np.ones(feature_count),
        active_coefficients=np.zeros(feature_count, dtype=np.bool_),
        ridge_lambda=ridge_lambda,
        training_issue_count=features.shape[0],
        event_count=sum(int(value) for value in counts),
        objective=objective,
        iteration_count=0,
        status=status,
        message=message,
    )


def fit_offset_poisson_ridge(
    features: FloatArray,
    offsets: FloatArray,
    counts: FloatArray,
    *,
    ridge_lambda: float,
) -> OffsetPoissonFit:
    """Single zero-start L-BFGS-B fit, or an explicit unmodified-offset fallback.

    An all-zero target set has no finite unpenalized-intercept MLE; it is retained
    and reported as a fallback, not repaired with pseudo-counts.  Constant training
    columns remain fixed at coefficient zero even if they vary at prediction time.
    """
    design = _design(features)
    offset = _offsets(offsets, design.shape[0])
    target = _counts(counts, design.shape[0])
    ridge = _ridge(ridge_lambda)
    if design.shape[0] == 0 or not np.any(target > 0):
        return _fallback(
            design,
            offset,
            target,
            ridge,
            status="baseline_no_training" if design.shape[0] == 0 else "baseline_all_zero_targets",
            message="No training issues." if design.shape[0] == 0 else "No finite intercept MLE.",
        )
    try:
        center, scale, active = _training_scaler(design)
        standardized = (design - center) / scale
        standardized[:, ~active] = 0.0

        def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
            return offset_poisson_value_and_gradient(
                parameters, standardized, offset, target, ridge_lambda=ridge
            )

        result = minimize(
            objective,
            np.zeros(design.shape[1] + 1),
            method="L-BFGS-B",
            jac=True,
            bounds=[(None, None)] + [(None, None) if enabled else (0.0, 0.0) for enabled in active],
            options={"maxiter": 2000, "ftol": 1.0e-10, "gtol": 1.0e-6},
        )
        parameters = np.asarray(result.x, dtype=np.float64).copy()
        if (
            not bool(result.success)
            or parameters.shape != (design.shape[1] + 1,)
            or not np.isfinite(parameters).all()
            or not math.isfinite(float(result.fun))
            or np.asarray(result.jac).shape != parameters.shape
            or not np.isfinite(result.jac).all()
        ):
            raise FloatingPointError(f"The frozen optimizer did not converge: {result.message!s}")
        parameters[1:][~active] = 0.0
        value, _ = objective(parameters)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return _fallback(
            design, offset, target, ridge, status="baseline_fit_not_evaluable", message=str(exc)
        )
    return OffsetPoissonFit(
        coefficients=parameters[1:],
        intercept=float(parameters[0]),
        center=center,
        scale=scale,
        active_coefficients=active,
        ridge_lambda=ridge,
        training_issue_count=design.shape[0],
        event_count=sum(int(value) for value in target),
        objective=value,
        iteration_count=int(result.nit),
        status="fitted",
        message=str(result.message),
    )


def fit_offset_poisson_intercept(offsets: FloatArray, counts: FloatArray) -> OffsetPoissonFit:
    """Analytic same-family calibration: a = log(sum(counts) / sum(offsets))."""
    offset_array = np.asarray(offsets, dtype=np.float64)
    if offset_array.ndim != 1:
        raise ValueError("offsets must be a vector")
    offset = _offsets(offset_array, offset_array.size)
    target = _counts(counts, offset.size)
    design = np.empty((offset.size, 0), dtype=np.float64)
    if offset.size == 0 or not np.any(target > 0):
        return _fallback(
            design,
            offset,
            target,
            0.0,
            status="baseline_no_training" if offset.size == 0 else "baseline_all_zero_targets",
            message="No training issues." if offset.size == 0 else "No finite intercept MLE.",
        )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            intercept = float(
                np.log(np.sum(target / target.size)) - np.log(np.sum(offset / offset.size))
            )
        value, _ = offset_poisson_value_and_gradient(
            np.array([intercept]), design, offset, target, ridge_lambda=0.0
        )
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return _fallback(
            design, offset, target, 0.0, status="baseline_fit_not_evaluable", message=str(exc)
        )
    return OffsetPoissonFit(
        coefficients=np.empty(0),
        intercept=intercept,
        center=np.empty(0),
        scale=np.empty(0),
        active_coefficients=np.empty(0, dtype=np.bool_),
        ridge_lambda=0.0,
        training_issue_count=offset.size,
        event_count=sum(int(value) for value in target),
        objective=value,
        iteration_count=0,
        status="fitted_intercept_only",
        message="Analytic unpenalized-intercept Poisson calibration.",
    )


def nonnegative_log1p(values: FloatArray) -> FloatArray:
    """Compress non-negative reported counts/ages without replacing missing values."""
    array = np.asarray(values, dtype=np.float64)
    if np.isinf(array).any() or np.any(array < 0):
        raise ValueError("log1p inputs must be non-negative finite values or NaN")
    return _readonly(np.log1p(array))


def signed_asinh(values: FloatArray) -> FloatArray:
    """Compress signed changes while preserving their sign and any missing values."""
    array = np.asarray(values, dtype=np.float64)
    if np.isinf(array).any():
        raise ValueError("asinh inputs must be finite values or NaN")
    return _readonly(np.arcsinh(array))


@dataclass(frozen=True, slots=True)
class TrainingAreaImputer:
    """Finite-value means learned with equal-issue and actual-cell-area weights."""

    fill_values: FloatArray
    active_columns: BoolArray
    training_issue_count: int

    def __post_init__(self) -> None:
        fill = np.asarray(self.fill_values, dtype=np.float64)
        active = np.asarray(self.active_columns, dtype=np.bool_)
        if fill.ndim != 1 or active.shape != fill.shape or not np.isfinite(fill).all():
            raise ValueError(
                "imputer fill values must be a finite vector aligned with active flags"
            )
        if np.any(fill[~active] != 0):
            raise ValueError("entirely missing training columns must have zero fill values")
        object.__setattr__(self, "fill_values", _readonly(fill))
        object.__setattr__(self, "active_columns", _readonly(active))

    def transform(self, features: FloatArray) -> FloatArray:
        """Fill NaNs only; columns absent throughout training stay inactive and zero."""
        values = np.asarray(features, dtype=np.float64)
        if values.ndim not in (2, 3) or values.shape[-1] != self.fill_values.size:
            raise ValueError("features must align with the imputer's last-axis columns")
        if np.isinf(values).any():
            raise ValueError("features may contain NaN but not infinity")
        result = np.where(np.isnan(values), self.fill_values, values)
        result[..., ~self.active_columns] = 0.0
        return _readonly(result)


def fit_training_area_imputer(
    training_features: FloatArray, area_km2: FloatArray
) -> TrainingAreaImputer:
    """Fit on (issues, cells, features); missing observations contribute no weight.

    Each issue receives equal total area weight before missing observations are
    removed.  This is a pooled finite-observation mean, not a mean of per-issue
    complete-case means.  No target counts enter imputation.
    """
    values = np.asarray(training_features, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] == 0 or np.isinf(values).any():
        raise ValueError("training features need shape (issues, cells, features), with no infinity")
    area = np.asarray(area_km2, dtype=np.float64)
    if area.shape != (values.shape[1],) or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("actual cell areas must be finite, positive, and aligned")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        weights = area / np.max(area)
        weights /= np.sum(weights)
        weights /= max(values.shape[0], 1)
        finite = np.isfinite(values)
        denominator = np.sum(finite * weights[None, :, None], axis=(0, 1))
        numerator = np.sum(np.where(finite, values, 0.0) * weights[None, :, None], axis=(0, 1))
        active = denominator > 0
        fill = np.zeros(values.shape[2], dtype=np.float64)
        fill[active] = numerator[active] / denominator[active]
    if not np.isfinite(fill).all():
        raise FloatingPointError("training finite-area means are non-finite")
    return TrainingAreaImputer(fill, active, values.shape[0])


__all__ = [
    "OffsetPoissonFit",
    "TrainingAreaImputer",
    "fit_offset_poisson_intercept",
    "fit_offset_poisson_ridge",
    "fit_training_area_imputer",
    "nonnegative_log1p",
    "offset_poisson_value_and_gradient",
    "signed_asinh",
]
