"""Deterministic row-level reason codes for nullable stage-3 features."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from seismoflux.features.anomaly.trajectory import LatestTrajectoryFeatureResult


class NullReasonCode(IntEnum):
    """Compact reason codes stored next to every nullable scientific value."""

    VALID = 0
    NO_ELIGIBLE_ENTITY = 1
    NO_KNOWN_AGE = 2
    NO_KNOWN_DURATION = 3
    ZERO_DENOMINATOR = 4
    DIRECTION_UNDEFINED = 5
    INSUFFICIENT_ACTUAL_SNAPSHOTS = 6
    ZERO_OR_UNDEFINED_BASELINE_VARIANCE = 7
    ZERO_CAUSAL_PEAK = 8
    CURRENT_VALUE_UNAVAILABLE = 9
    ZERO_ELAPSED_TIME_VARIANCE = 10


NULL_REASON_DEFINITIONS: dict[int, str] = {int(code): code.name.lower() for code in NullReasonCode}


def spatial_null_reason_codes(
    feature_name: str,
    features: dict[str, NDArray[np.float64]],
) -> NDArray[np.int8]:
    """Explain each NaN in one spatial feature array from its companion counts."""

    values = features[feature_name]
    reasons = np.full(values.shape, int(NullReasonCode.VALID), dtype=np.int8)
    missing = ~np.isfinite(values)
    if not np.any(missing):
        return reasons

    if feature_name == "age_mean_days":
        reason = NullReasonCode.NO_KNOWN_AGE
    elif feature_name == "known_duration_mean_days":
        reason = NullReasonCode.NO_KNOWN_DURATION
    elif feature_name in {
        "first_seen_rate",
        "first_seen_weighted_rate",
        "right_censored_fraction",
        "right_censored_weighted_fraction",
        "multidisciplinary_entity_fraction",
        "multidisciplinary_entity_weighted_fraction",
    }:
        reason = NullReasonCode.ZERO_DENOMINATOR
    elif feature_name == "principal_direction_deg":
        listed = features["listed_count"]
        reasons[missing & (listed <= 0.0)] = int(NullReasonCode.NO_ELIGIBLE_ENTITY)
        reasons[missing & (listed > 0.0)] = int(NullReasonCode.DIRECTION_UNDEFINED)
        return reasons
    else:
        reason = NullReasonCode.NO_ELIGIBLE_ENTITY
    reasons[missing] = int(reason)
    return reasons


def trajectory_null_reason_codes(
    result: LatestTrajectoryFeatureResult,
    feature_name: str,
) -> NDArray[np.int8]:
    """Explain invalid latest-trajectory values using frozen formula diagnostics."""

    valid = result.valid_masks[feature_name]
    counts = result.sample_counts[feature_name]
    reasons = np.full(valid.shape, int(NullReasonCode.VALID), dtype=np.int8)
    invalid = ~valid
    if not np.any(invalid):
        return reasons

    if feature_name.startswith("slope_"):
        reasons[invalid & (counts < 3)] = int(NullReasonCode.INSUFFICIENT_ACTUAL_SNAPSHOTS)
        reasons[invalid & (counts >= 3)] = int(NullReasonCode.ZERO_ELAPSED_TIME_VARIANCE)
    elif feature_name == "acceleration_4v13_per_week2":
        reasons[invalid] = int(NullReasonCode.INSUFFICIENT_ACTUAL_SNAPSHOTS)
    elif feature_name == "surge_z_13w":
        reasons[invalid & (counts < 3)] = int(NullReasonCode.INSUFFICIENT_ACTUAL_SNAPSHOTS)
        enough_history = invalid & (counts >= 3)
        current_unavailable = (
            enough_history
            & ~result.valid_masks["peak_drop_52w"]
            & (result.sample_counts["peak_drop_52w"] >= 2)
        )
        reasons[current_unavailable] = int(NullReasonCode.CURRENT_VALUE_UNAVAILABLE)
        reasons[enough_history & ~current_unavailable] = int(
            NullReasonCode.ZERO_OR_UNDEFINED_BASELINE_VARIANCE
        )
    elif feature_name == "peak_drop_52w":
        reasons[invalid & (counts < 2)] = int(NullReasonCode.INSUFFICIENT_ACTUAL_SNAPSHOTS)
        reasons[invalid & (counts >= 2)] = int(NullReasonCode.CURRENT_VALUE_UNAVAILABLE)
    elif feature_name == "peak_ratio_52w":
        drop_valid = result.valid_masks["peak_drop_52w"]
        drop_counts = result.sample_counts["peak_drop_52w"]
        reasons[invalid & ~drop_valid & (drop_counts < 2)] = int(
            NullReasonCode.INSUFFICIENT_ACTUAL_SNAPSHOTS
        )
        reasons[invalid & ~drop_valid & (drop_counts >= 2)] = int(
            NullReasonCode.CURRENT_VALUE_UNAVAILABLE
        )
        reasons[invalid & drop_valid] = int(NullReasonCode.ZERO_CAUSAL_PEAK)
    else:
        raise KeyError(f"unsupported trajectory feature for null audit: {feature_name}")
    if np.any(reasons[invalid] == int(NullReasonCode.VALID)):
        raise AssertionError("every invalid trajectory value must have a reason")
    return reasons


__all__ = [
    "NULL_REASON_DEFINITIONS",
    "NullReasonCode",
    "spatial_null_reason_codes",
    "trajectory_null_reason_codes",
]
