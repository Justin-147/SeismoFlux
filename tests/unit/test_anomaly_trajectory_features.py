from __future__ import annotations

import numpy as np
import pytest

from seismoflux.features.anomaly.trajectory import (
    compute_latest_trajectory_features,
    compute_trajectory_features,
)


def _times(day_offsets: list[int]) -> np.ndarray[tuple[int], np.dtype[np.datetime64]]:
    base = np.datetime64("2022-01-01T00:00:00", "ns")
    return base + np.asarray(day_offsets, dtype="timedelta64[D]")


def test_ols_uses_actual_irregular_issue_times() -> None:
    issue_times = _times([0, 3, 17, 28])
    elapsed_weeks = np.asarray([0.0, 3.0 / 7.0, 17.0 / 7.0, 4.0])
    values = 1.0 + 2.0 * elapsed_weeks

    result = compute_trajectory_features(issue_times, values)

    assert result.features["slope_4w_per_week"][-1, 0] == pytest.approx(2.0)
    assert result.valid_masks["slope_4w_per_week"][-1, 0]
    assert result.sample_counts["slope_4w_per_week"][-1, 0] == 3


def test_missing_report_periods_are_not_inserted_as_zero() -> None:
    day_offsets = np.asarray([0.0, 7.0, 27.0])
    issue_times = _times(day_offsets.astype(int).tolist())
    values = 1.0 + day_offsets / 7.0

    result = compute_trajectory_features(issue_times, values)

    assert result.sample_counts["slope_4w_per_week"][-1, 0] == 3
    assert result.features["slope_4w_per_week"][-1, 0] == pytest.approx(1.0)


def test_insufficient_samples_have_nan_and_false_validity_mask() -> None:
    issue_times = _times([0, 7, 14])
    values = np.asarray([1.0, 2.0, 3.0])

    result = compute_trajectory_features(issue_times, values)
    slope = result.features["slope_4w_per_week"][:, 0]
    valid = result.valid_masks["slope_4w_per_week"][:, 0]

    assert np.isnan(slope[:2]).all()
    assert not valid[0]
    assert not valid[1]
    assert valid[2]


def test_acceleration_uses_recent_four_vs_thirteen_week_slopes() -> None:
    week = np.arange(14, dtype=np.float64)
    issue_times = _times((np.arange(14) * 7).tolist())
    values = week * week

    result = compute_trajectory_features(issue_times, values)

    acceleration = result.features["acceleration_4v13_per_week2"][-1, 0]
    assert result.valid_masks["acceleration_4v13_per_week2"][-1, 0]
    assert acceleration > 0.0


def test_surge_uses_prior_thirteen_week_baseline_only() -> None:
    issue_times = _times((np.arange(15) * 7).tolist())
    values = np.asarray([1.0, 2.0] * 7 + [5.0])

    result = compute_trajectory_features(issue_times, values)

    assert result.sample_counts["surge_z_13w"][-1, 0] == 12
    assert result.valid_masks["surge_z_13w"][-1, 0]
    assert result.features["surge_z_13w"][-1, 0] > 5.0


def test_peak_drop_and_ratio_use_only_causal_fifty_two_week_history() -> None:
    issue_times = _times((np.arange(56) * 7).tolist())
    values = np.ones(56, dtype=np.float64)
    values[50] = 10.0
    values[55] = 4.0

    result = compute_trajectory_features(issue_times, values)

    assert result.features["peak_drop_52w"][-1, 0] == pytest.approx(6.0)
    assert result.features["peak_ratio_52w"][-1, 0] == pytest.approx(0.4)
    assert result.valid_masks["peak_drop_52w"][-1, 0]
    assert result.valid_masks["peak_ratio_52w"][-1, 0]


def test_future_values_cannot_change_earlier_trajectory_rows() -> None:
    issue_times = _times((np.arange(10) * 7).tolist())
    values = np.arange(1.0, 11.0)

    prefix = compute_trajectory_features(issue_times[:6], values[:6])
    full = compute_trajectory_features(issue_times, values)

    for feature_name, expected in prefix.features.items():
        np.testing.assert_allclose(full.features[feature_name][:6], expected, equal_nan=True)
        np.testing.assert_array_equal(
            full.valid_masks[feature_name][:6],
            prefix.valid_masks[feature_name],
        )
        np.testing.assert_array_equal(
            full.sample_counts[feature_name][:6],
            prefix.sample_counts[feature_name],
        )


def test_latest_api_matches_full_result_last_row_for_every_output() -> None:
    day_offsets = [
        0,
        3,
        11,
        18,
        29,
        37,
        51,
        58,
        70,
        83,
        91,
        106,
        119,
        133,
        148,
        166,
        181,
        199,
        218,
        241,
        267,
        294,
        326,
        365,
        401,
    ]
    issue_times = _times(day_offsets)
    elapsed_weeks = np.asarray(day_offsets, dtype=np.float64) / 7.0
    mixed_missing = 2.0 + np.sqrt(elapsed_weeks + 1.0)
    mixed_missing[[1, 5, 12, 19, 24]] = np.nan
    values = np.column_stack(
        (
            1.0 + 0.2 * elapsed_weeks + 0.01 * elapsed_weeks**2,
            np.zeros_like(elapsed_weeks),
            mixed_missing,
        )
    )
    thresholds = {
        "minimum_slope_observations": 4,
        "minimum_surge_baseline_observations": 4,
        "minimum_peak_observations": 3,
    }

    full = compute_trajectory_features(issue_times, values, **thresholds)
    latest = compute_latest_trajectory_features(issue_times, values, **thresholds)

    assert latest.issue_time_utc == full.issue_times_utc[-1]
    assert latest.features.keys() == full.features.keys()
    assert latest.valid_masks.keys() == full.valid_masks.keys()
    assert latest.sample_counts.keys() == full.sample_counts.keys()
    for feature_name in full.features:
        assert latest.features[feature_name].shape == (values.shape[1],)
        assert latest.valid_masks[feature_name].shape == (values.shape[1],)
        assert latest.sample_counts[feature_name].shape == (values.shape[1],)
        np.testing.assert_allclose(
            latest.features[feature_name],
            full.features[feature_name][-1],
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            latest.valid_masks[feature_name],
            full.valid_masks[feature_name][-1],
        )
        np.testing.assert_array_equal(
            latest.sample_counts[feature_name],
            full.sample_counts[feature_name][-1],
        )
