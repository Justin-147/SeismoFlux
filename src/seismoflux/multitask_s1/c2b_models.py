"""Finite C2B location mathematics; no catalog, calendar, or selection logic.

All outputs are normalized log cell masses, not absolute occurrence probabilities.
Underflow of derived masses is harmless: observed-cell likelihoods stay in log space.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ArrayScalar = TypeVar("ArrayScalar", bound=np.generic)


def _readonly(values: NDArray[ArrayScalar]) -> NDArray[ArrayScalar]:
    result = np.array(values, copy=True, order="C")
    result.setflags(write=False)
    return cast(NDArray[ArrayScalar], result)


def _normalize_log_mass(values: FloatArray) -> FloatArray:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("log cell masses must be a nonempty finite vector")
    normalized = np.asarray(values - logsumexp(values), dtype=np.float64)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("log cell mass normalization is non-finite")
    return normalized


def _areas(values: object, cell_count: int) -> FloatArray:
    area = np.asarray(values, dtype=np.float64)
    if area.shape != (cell_count,) or not np.isfinite(area).all() or np.any(area <= 0.0):
        raise ValueError("actual cell areas must be finite, positive, and aligned")
    return area


def gaussian_log_masses(
    training_xy_km: FloatArray,
    query_xy_km: FloatArray,
    area_km2: FloatArray,
    *,
    bandwidths_km: tuple[float, ...] = (25.0, 75.0, 150.0),
    log_event_weights: FloatArray | None = None,
    chunk_size: int = 256,
) -> dict[float, FloatArray]:
    """Compute weighted isotropic KDEs, each normalized over the full given grid.

    Finite log weights may be negative (e.g. exponential age weights). Gaussian
    and total-event-weight constants cancel in each national normalization. Empty
    training panels return area-uniform mass; empty *recent* fallback is the caller's
    responsibility because it needs that panel's long-history KDE75.
    """
    training = np.asarray(training_xy_km, dtype=np.float64)
    query = np.asarray(query_xy_km, dtype=np.float64)
    if training.ndim != 2 or training.shape[1] != 2:
        raise ValueError("training coordinates must have shape (events, 2)")
    if query.ndim != 2 or query.shape[1] != 2 or query.shape[0] == 0:
        raise ValueError("query coordinates must have nonempty shape (cells, 2)")
    if not np.isfinite(training).all() or not np.isfinite(query).all():
        raise ValueError("projected coordinates must be finite")
    area = _areas(area_km2, query.shape[0])
    bandwidths = tuple(float(value) for value in bandwidths_km)
    if (
        not bandwidths
        or len(set(bandwidths)) != len(bandwidths)
        or any(not math.isfinite(value) or value <= 0.0 for value in bandwidths)
    ):
        raise ValueError("kernel bandwidths must be distinct finite positive values")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    weights = (
        np.zeros(training.shape[0], dtype=np.float64)
        if log_event_weights is None
        else np.asarray(log_event_weights, dtype=np.float64)
    )
    if weights.shape != (training.shape[0],) or not np.isfinite(weights).all():
        raise ValueError("one finite log event weight is required per training event")
    log_area = np.log(area)
    if training.shape[0] == 0:
        uniform = _normalize_log_mass(log_area)
        return {bandwidth: _readonly(uniform) for bandwidth in bandwidths}
    weights = weights - np.max(weights)
    surfaces = {bandwidth: np.empty(query.shape[0], dtype=np.float64) for bandwidth in bandwidths}
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        squared_distance = cdist(query[start:stop], training, metric="sqeuclidean")
        if not np.isfinite(squared_distance).all():
            raise FloatingPointError("projected squared distances are non-finite")
        for bandwidth, surface in surfaces.items():
            kernel = -0.5 * (squared_distance / bandwidth / bandwidth) + weights[None, :]
            surface[start:stop] = logsumexp(kernel, axis=1) + log_area[start:stop]
    return {
        bandwidth: _readonly(_normalize_log_mass(value)) for bandwidth, value in surfaces.items()
    }


def mix_log_masses(components: Sequence[FloatArray], weights: Sequence[float]) -> FloatArray:
    """Mix individually normalized components; exactly zero weights are omitted."""
    arrays = tuple(np.asarray(item, dtype=np.float64) for item in components)
    coefficients = np.asarray(weights, dtype=np.float64)
    if not arrays or coefficients.shape != (len(arrays),):
        raise ValueError("mixture weights must align with at least one component")
    if not np.isfinite(coefficients).all() or np.any(coefficients < 0.0):
        raise ValueError("mixture weights must be finite and non-negative")
    if not np.any(coefficients > 0.0):
        raise ValueError("at least one mixture weight must be positive")
    if any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("mixture components must align by grid cell")
    normalized = tuple(_normalize_log_mass(item) for item in arrays)
    positive = np.flatnonzero(coefficients > 0.0)
    log_weights = np.log(coefficients[positive])
    log_weights -= logsumexp(log_weights)
    terms = np.stack(
        [normalized[index] + weight for index, weight in zip(positive, log_weights, strict=True)]
    )
    return _readonly(_normalize_log_mass(np.asarray(logsumexp(terms, axis=0), dtype=np.float64)))


class C2BFitError(RuntimeError):
    """The one frozen numerical fit was not evaluable; callers keep the baseline."""


@dataclass(frozen=True, slots=True)
class C2BTrainingIssue:
    issue_id: str
    log_base_mass: FloatArray
    features: FloatArray
    future_counts: FloatArray

    def __post_init__(self) -> None:
        base = np.asarray(self.log_base_mass, dtype=np.float64)
        features = np.asarray(self.features, dtype=np.float64)
        counts = np.asarray(self.future_counts, dtype=np.float64)
        if not isinstance(self.issue_id, str) or not self.issue_id:
            raise ValueError("issue_id must be a nonempty string")
        if base.ndim != 1 or base.size == 0 or features.ndim != 2:
            raise ValueError("base/features must have shapes (cells,) and (cells, features)")
        if features.shape[0] != base.size or counts.shape != base.shape:
            raise ValueError("issue arrays must align by cell")
        if not np.isfinite(base).all() or not np.isfinite(features).all():
            raise ValueError("base log mass and features must be finite")
        if (
            not np.isfinite(counts).all()
            or np.any(counts < 0.0)
            or not np.array_equal(counts, np.floor(counts))
            or not math.isfinite(float(np.sum(counts, dtype=np.float64)))
        ):
            raise ValueError("future cell counts must be finite non-negative integers")
        object.__setattr__(self, "log_base_mass", _readonly(base))
        object.__setattr__(self, "features", _readonly(features))
        object.__setattr__(self, "future_counts", _readonly(counts))

    @property
    def event_count(self) -> int:
        return int(np.sum(self.future_counts, dtype=np.float64))


def spatial_ridge_value_and_gradient(
    coefficients: FloatArray,
    issues: Sequence[C2BTrainingIssue],
    *,
    ridge_lambda: float,
) -> tuple[float, FloatArray]:
    """Pooled equal-event conditional NLL and analytic gradient for supplied designs.

    The fitter supplies training-standardized features; this pure objective does
    not estimate a scaler or inspect any validation data. Empty periods contribute
    no event likelihood. Derived zero masses from underflow are valid gradients.
    """
    blocks = tuple(issues)
    beta = np.asarray(coefficients, dtype=np.float64)
    ridge = float(ridge_lambda)
    if not blocks:
        raise ValueError("at least one training issue is required")
    if beta.shape != (blocks[0].features.shape[1],) or not np.isfinite(beta).all():
        raise ValueError("coefficients must be finite and aligned with features")
    if any(item.features.shape[1] != beta.size for item in blocks):
        raise ValueError("all issue feature columns must align")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    event_count = sum(item.event_count for item in blocks)
    nll = 0.0
    gradient = np.zeros_like(beta)
    if event_count:
        for issue in blocks:
            if issue.event_count == 0:
                continue
            log_mass = _normalize_log_mass(issue.log_base_mass + issue.features @ beta)
            nll -= float(np.dot(issue.future_counts, log_mass)) / event_count
            residual = issue.event_count * np.exp(log_mass) - issue.future_counts
            gradient += (issue.features.T @ residual) / event_count
    value = nll + 0.5 * ridge * float(np.dot(beta, beta))
    gradient += ridge * beta
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("conditional spatial objective or gradient is non-finite")
    return value, gradient


def _training_scaler(
    issues: tuple[C2BTrainingIssue, ...], area: FloatArray
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Equal-issue, actual-area weighted population moments from training only."""
    cell_weights = np.exp(_normalize_log_mass(np.log(area)))
    feature_count = issues[0].features.shape[1]
    center = np.zeros(feature_count, dtype=np.float64)
    lower = np.full(feature_count, np.inf, dtype=np.float64)
    upper = np.full(feature_count, -np.inf, dtype=np.float64)
    for issue in issues:
        center += (cell_weights @ issue.features) / len(issues)
        lower = np.minimum(lower, np.min(issue.features, axis=0))
        upper = np.maximum(upper, np.max(issue.features, axis=0))
    active = upper > lower
    center[~active] = lower[~active]
    # Rescale deviations before squaring to avoid under/overflow; no data clipping.
    maximum_deviation = np.maximum(np.abs(upper - center), np.abs(lower - center))
    divisor = np.where(active, maximum_deviation, 1.0)
    variance = np.zeros(feature_count, dtype=np.float64)
    for issue in issues:
        standardized = (issue.features - center) / divisor
        variance += (cell_weights @ np.square(standardized)) / len(issues)
    scale = np.where(active, divisor * np.sqrt(variance), 1.0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise C2BFitError("training weighted scaler is not finite and positive")
    return center, scale, active


@dataclass(frozen=True, slots=True)
class C2BRidgeFit:
    coefficients: FloatArray
    center: FloatArray
    scale: FloatArray
    active_coefficients: BoolArray
    ridge_lambda: float
    event_count: int
    training_issue_count: int
    objective: float
    iteration_count: int
    optimizer_status: str
    optimizer_message: str

    def __post_init__(self) -> None:
        beta = np.asarray(self.coefficients, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        active = np.asarray(self.active_coefficients, dtype=np.bool_)
        if beta.ndim != 1 or any(value.shape != beta.shape for value in (center, scale, active)):
            raise ValueError("fit coefficient and scaler arrays do not align")
        if (
            not all(np.isfinite(value).all() for value in (beta, center, scale))
            or np.any(scale <= 0.0)
            or np.any(beta[~active] != 0.0)
        ):
            raise ValueError("fitted coefficients and scaler must be valid")
        for name, value in (
            ("coefficients", beta),
            ("center", center),
            ("scale", scale),
            ("active_coefficients", active),
        ):
            object.__setattr__(self, name, _readonly(value))

    def predict_log_mass(self, base: FloatArray, rawfeatures: FloatArray) -> FloatArray:
        features = np.asarray(rawfeatures, dtype=np.float64)
        base_array = np.asarray(base, dtype=np.float64)
        if features.ndim != 2 or features.shape != (base_array.size, self.coefficients.size):
            raise ValueError("prediction features do not align with base/scaler")
        if not np.isfinite(features).all():
            raise ValueError("prediction features must be finite")
        scaled = (features - self.center) / self.scale
        scaled[:, ~self.active_coefficients] = 0.0
        return _readonly(_normalize_log_mass(base_array + scaled @ self.coefficients))

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": self.coefficients.tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "active_coefficients": self.active_coefficients.tolist(),
            "ridge_lambda": self.ridge_lambda,
            "event_count": self.event_count,
            "training_issue_count": self.training_issue_count,
            "objective": self.objective,
            "iteration_count": self.iteration_count,
            "optimizer_status": self.optimizer_status,
            "optimizer_message": self.optimizer_message,
        }


def fit_spatial_ridge(
    issues: Sequence[C2BTrainingIssue],
    area_km2: FloatArray,
    *,
    ridge_lambda: float,
) -> C2BRidgeFit:
    """Fit the frozen single-start L-BFGS-B model, or raise C2BFitError."""
    blocks = tuple(issues)
    if not blocks:
        raise ValueError("cannot fit without any legal training issues")
    shape = blocks[0].features.shape
    if any(item.features.shape != shape for item in blocks):
        raise ValueError("training issue grids and feature counts must align")
    area = _areas(area_km2, shape[0])
    ridge = float(ridge_lambda)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    try:
        center, scale, active = _training_scaler(blocks, area)
        standardized = []
        for issue in blocks:
            features = (issue.features - center) / scale
            features[:, ~active] = 0.0
            standardized.append(
                C2BTrainingIssue(issue.issue_id, issue.log_base_mass, features, issue.future_counts)
            )
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise C2BFitError("training feature standardization was not numerically evaluable") from exc
    designs = tuple(standardized)
    event_count = sum(item.event_count for item in blocks)
    beta = np.zeros(shape[1], dtype=np.float64)

    def objective(values: FloatArray) -> tuple[float, FloatArray]:
        return spatial_ridge_value_and_gradient(values, designs, ridge_lambda=ridge)

    status, message, iterations = "converged", "", 0
    try:
        if event_count == 0:
            status = "no_training_events"
            message = "Beta zero; all legal training periods retained."
        elif not np.any(active):
            status, message = "no_active_features", "Beta zero; every training feature is constant."
        else:
            result = minimize(
                objective,
                beta,
                method="L-BFGS-B",
                jac=True,
                bounds=[(None, None) if enabled else (0.0, 0.0) for enabled in active],
                options={"maxiter": 2000, "ftol": 1.0e-10, "gtol": 1.0e-6},
            )
            beta = np.asarray(result.x, dtype=np.float64).copy()
            if (
                not bool(result.success)
                or beta.shape != (shape[1],)
                or not np.isfinite(beta).all()
                or not math.isfinite(float(result.fun))
                or np.asarray(result.jac).shape != beta.shape
                or not np.isfinite(result.jac).all()
            ):
                raise C2BFitError(f"frozen spatial ridge fit did not converge: {result.message!s}")
            beta[~active] = 0.0
            message, iterations = str(result.message), int(result.nit)
        checked_objective, checked_gradient = objective(beta)
        if not np.isfinite(checked_gradient).all():
            raise C2BFitError("final spatial ridge gradient is non-finite")
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise C2BFitError("frozen spatial ridge numerical fit was not evaluable") from exc
    return C2BRidgeFit(
        coefficients=beta,
        center=center,
        scale=scale,
        active_coefficients=active,
        ridge_lambda=ridge,
        event_count=event_count,
        training_issue_count=len(blocks),
        objective=checked_objective,
        iteration_count=iterations,
        optimizer_status=status,
        optimizer_message=message,
    )


__all__ = [
    "C2BFitError",
    "C2BRidgeFit",
    "C2BTrainingIssue",
    "fit_spatial_ridge",
    "gaussian_log_masses",
    "mix_log_masses",
    "spatial_ridge_value_and_gradient",
]
