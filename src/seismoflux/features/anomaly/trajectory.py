"""Causal trajectory features evaluated on actual, irregular issue timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

TRAJECTORY_WINDOWS_WEEKS: Final[tuple[int, ...]] = (4, 8, 13, 26, 52)
_NANOSECONDS_PER_WEEK: Final[int] = 7 * 24 * 60 * 60 * 1_000_000_000
_ACCELERATION_CENTRE_SEPARATION_WEEKS: Final[float] = (13.0 - 4.0) / 2.0
_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    [f"slope_{window}w_per_week" for window in TRAJECTORY_WINDOWS_WEEKS]
    + [
        "acceleration_4v13_per_week2",
        "surge_z_13w",
        "peak_drop_52w",
        "peak_ratio_52w",
    ]
)


@dataclass(frozen=True, slots=True)
class TrajectoryFeatureResult:
    """Trajectory arrays shaped ``(n_issue, n_series)``.

    Invalid estimates remain NaN and have a false entry in ``valid_masks``.
    ``sample_counts`` records actual observed values; absent report dates are never
    inserted and therefore can never be mistaken for zero observations.
    """

    issue_times_utc: NDArray[np.datetime64]
    features: dict[str, FloatArray]
    valid_masks: dict[str, BoolArray]
    sample_counts: dict[str, IntArray]


@dataclass(frozen=True, slots=True)
class LatestTrajectoryFeatureResult:
    """Only the latest issue's arrays, each shaped ``(n_series,)``."""

    issue_time_utc: np.datetime64
    features: dict[str, FloatArray]
    valid_masks: dict[str, BoolArray]
    sample_counts: dict[str, IntArray]


@dataclass(frozen=True, slots=True)
class _PreparedTrajectory:
    issue_times_utc: NDArray[np.datetime64]
    time_ns: IntArray
    values: FloatArray


def _ols_slopes(
    elapsed_weeks: FloatArray,
    values: FloatArray,
    *,
    minimum_observations: int,
) -> tuple[FloatArray, BoolArray, IntArray]:
    valid = np.isfinite(values)
    counts = np.asarray(np.sum(valid, axis=0), dtype=np.int64)
    x = elapsed_weeks[:, None]
    safe_values = np.where(valid, values, 0.0)
    x_sum = np.sum(np.where(valid, x, 0.0), axis=0)
    y_sum = np.sum(safe_values, axis=0)
    divisor = np.maximum(counts, 1)
    x_mean = x_sum / divisor
    y_mean = y_sum / divisor
    dx = np.where(valid, x - x_mean, 0.0)
    dy = np.where(valid, values - y_mean, 0.0)
    denominator = np.sum(dx * dx, axis=0)
    numerator = np.sum(dx * dy, axis=0)
    estimable = (counts >= minimum_observations) & (denominator > np.finfo(np.float64).eps)
    slopes = np.full(values.shape[1], np.nan, dtype=np.float64)
    slopes[estimable] = numerator[estimable] / denominator[estimable]
    return slopes, np.asarray(estimable, dtype=np.bool_), counts


def _initial_outputs(
    n_issue: int,
    n_series: int,
) -> tuple[dict[str, FloatArray], dict[str, BoolArray], dict[str, IntArray]]:
    shape = (n_issue, n_series)
    features = {name: np.full(shape, np.nan, dtype=np.float64) for name in _FEATURE_NAMES}
    valid_masks = {name: np.zeros(shape, dtype=np.bool_) for name in _FEATURE_NAMES}
    sample_counts = {name: np.zeros(shape, dtype=np.int64) for name in _FEATURE_NAMES}
    return features, valid_masks, sample_counts


def _initial_issue_outputs(
    n_series: int,
) -> tuple[dict[str, FloatArray], dict[str, BoolArray], dict[str, IntArray]]:
    features = {name: np.full(n_series, np.nan, dtype=np.float64) for name in _FEATURE_NAMES}
    valid_masks = {name: np.zeros(n_series, dtype=np.bool_) for name in _FEATURE_NAMES}
    sample_counts = {name: np.zeros(n_series, dtype=np.int64) for name in _FEATURE_NAMES}
    return features, valid_masks, sample_counts


def _prepare_trajectory_inputs(
    issue_times_utc: ArrayLike,
    values: ArrayLike,
    *,
    minimum_slope_observations: int,
    minimum_surge_baseline_observations: int,
    minimum_peak_observations: int,
) -> _PreparedTrajectory:
    raw_times = np.asarray(issue_times_utc)
    if raw_times.ndim != 1 or not np.issubdtype(raw_times.dtype, np.datetime64):
        raise TypeError("issue_times_utc must be a one-dimensional datetime64 array")
    times = np.asarray(raw_times, dtype="datetime64[ns]")
    time_ns = np.asarray(times.astype(np.int64), dtype=np.int64)
    if np.isnat(times).any():
        raise ValueError("issue_times_utc cannot contain NaT")
    if time_ns.size > 1 and np.any(np.diff(time_ns) <= 0):
        raise ValueError("issue_times_utc must be strictly increasing")

    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim == 1:
        value_array = value_array[:, None]
    if value_array.ndim != 2 or value_array.shape[0] != times.size:
        raise ValueError("values must have shape (n_issue,) or (n_issue, n_series)")
    if np.isinf(value_array).any() or np.any(value_array[np.isfinite(value_array)] < 0.0):
        raise ValueError("values must contain only non-negative finite values or NaN")
    if minimum_slope_observations < 2:
        raise ValueError("minimum_slope_observations must be at least two")
    if minimum_surge_baseline_observations < 2:
        raise ValueError("minimum_surge_baseline_observations must be at least two")
    if minimum_peak_observations < 1:
        raise ValueError("minimum_peak_observations must be positive")

    return _PreparedTrajectory(
        issue_times_utc=times,
        time_ns=time_ns,
        values=value_array,
    )


def _compute_one_issue(
    prepared: _PreparedTrajectory,
    issue_index: int,
    *,
    minimum_slope_observations: int,
    minimum_surge_baseline_observations: int,
    minimum_peak_observations: int,
) -> tuple[dict[str, FloatArray], dict[str, BoolArray], dict[str, IntArray]]:
    time_ns = prepared.time_ns
    value_array = prepared.values
    n_series = value_array.shape[1]
    features, valid_masks, sample_counts = _initial_issue_outputs(n_series)
    current_ns = int(time_ns[issue_index])
    causal_time_ns = time_ns[: issue_index + 1]

    for window_weeks in TRAJECTORY_WINDOWS_WEEKS:
        lower_ns = current_ns - window_weeks * _NANOSECONDS_PER_WEEK
        in_window = np.flatnonzero(causal_time_ns > lower_ns)
        elapsed_weeks = np.asarray(
            (time_ns[in_window] - current_ns) / _NANOSECONDS_PER_WEEK,
            dtype=np.float64,
        )
        slope, estimable, counts = _ols_slopes(
            elapsed_weeks,
            value_array[in_window],
            minimum_observations=minimum_slope_observations,
        )
        name = f"slope_{window_weeks}w_per_week"
        features[name] = slope
        valid_masks[name] = estimable
        sample_counts[name] = counts

    slope_4 = features["slope_4w_per_week"]
    slope_13 = features["slope_13w_per_week"]
    acceleration_valid = valid_masks["slope_4w_per_week"] & valid_masks["slope_13w_per_week"]
    acceleration = np.full(n_series, np.nan, dtype=np.float64)
    acceleration[acceleration_valid] = (
        slope_4[acceleration_valid] - slope_13[acceleration_valid]
    ) / _ACCELERATION_CENTRE_SEPARATION_WEEKS
    acceleration_name = "acceleration_4v13_per_week2"
    features[acceleration_name] = acceleration
    valid_masks[acceleration_name] = acceleration_valid
    sample_counts[acceleration_name] = np.minimum(
        sample_counts["slope_4w_per_week"],
        sample_counts["slope_13w_per_week"],
    )

    surge_lower_ns = current_ns - 13 * _NANOSECONDS_PER_WEEK
    prior_indices = np.flatnonzero(time_ns[:issue_index] > surge_lower_ns)
    current_values = value_array[issue_index]
    surge_name = "surge_z_13w"
    if prior_indices.size:
        prior = value_array[prior_indices]
        prior_valid = np.isfinite(prior)
        prior_counts = np.asarray(np.sum(prior_valid, axis=0), dtype=np.int64)
        safe_prior = np.where(prior_valid, prior, 0.0)
        divisor = np.maximum(prior_counts, 1)
        prior_mean = np.sum(safe_prior, axis=0) / divisor
        centered = np.where(prior_valid, prior - prior_mean, 0.0)
        variance_divisor = np.maximum(prior_counts - 1, 1)
        prior_std = np.sqrt(np.sum(centered * centered, axis=0) / variance_divisor)
        surge_valid = (
            (prior_counts >= minimum_surge_baseline_observations)
            & np.isfinite(current_values)
            & (prior_std > np.sqrt(np.finfo(np.float64).eps))
        )
        surge = np.full(n_series, np.nan, dtype=np.float64)
        surge[surge_valid] = (current_values[surge_valid] - prior_mean[surge_valid]) / prior_std[
            surge_valid
        ]
        features[surge_name] = surge
        valid_masks[surge_name] = surge_valid
        sample_counts[surge_name] = prior_counts

    peak_lower_ns = current_ns - 52 * _NANOSECONDS_PER_WEEK
    peak_indices = np.flatnonzero(causal_time_ns > peak_lower_ns)
    peak_values = value_array[peak_indices]
    peak_valid_values = np.isfinite(peak_values)
    peak_counts = np.asarray(np.sum(peak_valid_values, axis=0), dtype=np.int64)
    filled_peak_values = np.where(peak_valid_values, peak_values, -np.inf)
    peaks = np.max(filled_peak_values, axis=0)
    current_finite = np.isfinite(current_values)
    drop_valid = (peak_counts >= minimum_peak_observations) & current_finite
    drop = np.full(n_series, np.nan, dtype=np.float64)
    drop[drop_valid] = peaks[drop_valid] - current_values[drop_valid]
    drop_name = "peak_drop_52w"
    features[drop_name] = drop
    valid_masks[drop_name] = drop_valid
    sample_counts[drop_name] = peak_counts

    ratio_valid = drop_valid & (peaks > 0.0)
    ratio = np.full(n_series, np.nan, dtype=np.float64)
    ratio[ratio_valid] = current_values[ratio_valid] / peaks[ratio_valid]
    ratio_name = "peak_ratio_52w"
    features[ratio_name] = ratio
    valid_masks[ratio_name] = ratio_valid
    sample_counts[ratio_name] = peak_counts
    return features, valid_masks, sample_counts


def compute_trajectory_features(
    issue_times_utc: ArrayLike,
    values: ArrayLike,
    *,
    minimum_slope_observations: int = 3,
    minimum_surge_baseline_observations: int = 3,
    minimum_peak_observations: int = 2,
) -> TrajectoryFeatureResult:
    """Compute causal rolling features without regularising the issue calendar.

    ``issue_times_utc`` must be a strictly increasing NumPy datetime64 sequence.
    ``values`` may be one-dimensional or ``(n_issue, n_series)`` and must contain
    non-negative finite values or NaN.  Rolling windows are left-open and right-closed
    for slopes and peaks.  The 13-week surge z-score compares the current value with
    *prior* observed reports in ``(T-13 weeks, T)`` using sample standard deviation.

    The 4-vs-13 acceleration is the difference between the two OLS slopes divided
    by the 4.5-week separation between their window centres.  Peak ratio is undefined
    when the causal 52-week peak is zero.
    """

    prepared = _prepare_trajectory_inputs(
        issue_times_utc,
        values,
        minimum_slope_observations=minimum_slope_observations,
        minimum_surge_baseline_observations=minimum_surge_baseline_observations,
        minimum_peak_observations=minimum_peak_observations,
    )
    n_issue, n_series = prepared.values.shape
    features, valid_masks, sample_counts = _initial_outputs(n_issue, n_series)

    for issue_index in range(n_issue):
        issue_features, issue_valid_masks, issue_sample_counts = _compute_one_issue(
            prepared,
            issue_index,
            minimum_slope_observations=minimum_slope_observations,
            minimum_surge_baseline_observations=minimum_surge_baseline_observations,
            minimum_peak_observations=minimum_peak_observations,
        )
        for name in _FEATURE_NAMES:
            features[name][issue_index] = issue_features[name]
            valid_masks[name][issue_index] = issue_valid_masks[name]
            sample_counts[name][issue_index] = issue_sample_counts[name]

    return TrajectoryFeatureResult(
        issue_times_utc=prepared.issue_times_utc,
        features=features,
        valid_masks=valid_masks,
        sample_counts=sample_counts,
    )


def compute_latest_trajectory_features(
    issue_times_utc: ArrayLike,
    values: ArrayLike,
    *,
    minimum_slope_observations: int = 3,
    minimum_surge_baseline_observations: int = 3,
    minimum_peak_observations: int = 2,
) -> LatestTrajectoryFeatureResult:
    """Compute the frozen trajectory formulas only for the latest issue time.

    This streaming-oriented API validates the complete causal history but allocates
    feature outputs only for the last issue, with every array shaped ``(n_series,)``.
    Its results are identical to the last row returned by
    :func:`compute_trajectory_features` for the same inputs and thresholds.
    """

    prepared = _prepare_trajectory_inputs(
        issue_times_utc,
        values,
        minimum_slope_observations=minimum_slope_observations,
        minimum_surge_baseline_observations=minimum_surge_baseline_observations,
        minimum_peak_observations=minimum_peak_observations,
    )
    if prepared.issue_times_utc.size == 0:
        raise ValueError("latest trajectory features require at least one issue time")

    features, valid_masks, sample_counts = _compute_one_issue(
        prepared,
        prepared.issue_times_utc.size - 1,
        minimum_slope_observations=minimum_slope_observations,
        minimum_surge_baseline_observations=minimum_surge_baseline_observations,
        minimum_peak_observations=minimum_peak_observations,
    )
    return LatestTrajectoryFeatureResult(
        issue_time_utc=np.datetime64(prepared.issue_times_utc[-1], "ns"),
        features=features,
        valid_masks=valid_masks,
        sample_counts=sample_counts,
    )


__all__ = [
    "TRAJECTORY_WINDOWS_WEEKS",
    "LatestTrajectoryFeatureResult",
    "TrajectoryFeatureResult",
    "compute_latest_trajectory_features",
    "compute_trajectory_features",
]
