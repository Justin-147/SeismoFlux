"""Synthetic checks of S3 null reconstruction, not a real-data randomization run."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from seismoflux.multitask_s3.calendar import time_null_partitions
from seismoflux.multitask_s3.null_features import (
    DYNAMIC_INDICES,
    FIXED_COVERAGE_INDICES,
    SNAPSHOT_AND_MASK_INDICES,
    permute_time_features,
    rebuild_dynamic_values,
)

START = datetime(2022, 7, 20, 16, tzinfo=UTC)
TRUTH = datetime(2025, 6, 30, 16, tzinfo=UTC)


def _inputs(n=30):
    times = tuple(START + timedelta(days=7 * i) for i in range(n))
    bases = np.arange(n * 3 * 2, dtype=float).reshape(n, 3, 2)
    features = np.arange(n * 3 * 20, dtype=float).reshape(n, 3, 20) / 10
    features[:, :, DYNAMIC_INDICES] = rebuild_dynamic_values(times, bases, cell_chunk_size=2)
    features[:, :, 16] = 0
    features[:, :, 17] = np.isnan(features[:, :, DYNAMIC_INDICES]).mean(axis=2)
    features[:, :, 18] = 0
    features[:, :, 19] = 0
    return times, features, bases


def _permute(times, features, bases, **overrides):
    kwargs = dict(
        issue_times_utc=times,
        features=features,
        radius_bases=bases,
        fold_id="A_DEV_2023_2024",
        horizon_days=30,
        truth_cutoff=TRUTH,
        rng=np.random.default_rng(147),
        cell_chunk_size=2,
    )
    kwargs.update(overrides)
    return permute_time_features(**kwargs)


def test_time_null_joint_donors_masks_coverage_and_inputs_are_preserved():
    times, features, bases = _inputs()
    features[2, 1, 7] = np.nan
    features[2, 1, 16] = 1
    features[3, 0, 11] = np.nan
    features[3, 0, 18] = 1
    features[5, 0, 14] = np.nan
    features[5, 0, 19] = 1
    before_features, before_bases = features.copy(), bases.copy()
    result = _permute(times, features, bases)
    for column in SNAPSHOT_AND_MASK_INDICES:
        np.testing.assert_array_equal(
            result.features[:, :, column], features[result.donor_indices, :, column]
        )
    np.testing.assert_array_equal(
        result.features[:, :, FIXED_COVERAGE_INDICES], features[:, :, FIXED_COVERAGE_INDICES]
    )
    np.testing.assert_array_equal(result.radius_bases, bases[result.donor_indices])
    expected_dynamic = rebuild_dynamic_values(times, result.radius_bases, cell_chunk_size=1)
    np.testing.assert_array_equal(result.features[:, :, DYNAMIC_INDICES], expected_dynamic)
    np.testing.assert_array_equal(
        result.features[:, :, 17], np.isnan(expected_dynamic).mean(axis=2)
    )
    np.testing.assert_array_equal(features, before_features)
    np.testing.assert_array_equal(bases, before_bases)
    assert not result.features.flags.writeable
    assert not result.radius_bases.flags.writeable
    assert not result.donor_indices.flags.writeable
    assert result.diagnostics["role"] == "offline_time_counterfactual_not_causal_prediction"


def test_each_registered_pool_remains_a_bijection_with_no_boundary_crossing():
    times, features, bases = _inputs(100)
    result = _permute(times, features, bases)
    lookup = {time: i for i, time in enumerate(times)}
    pools = time_null_partitions(
        times, fold_id="A_DEV_2023_2024", horizon_days=30, truth_cutoff=TRUTH
    )
    for pool in pools:
        indices = [lookup[time] for time in pool]
        assert set(result.donor_indices[indices]) == set(indices)
    assert result.diagnostics["effective_permutation_fraction"] == np.mean(
        result.donor_indices != np.arange(len(times))
    )
    assert result.diagnostics["later_donor_count"] > 0


def test_singleton_has_identity_donor_and_explicit_zero_effective_fraction():
    times, features, bases = _inputs(1)
    result = _permute(times, features, bases)
    np.testing.assert_array_equal(result.donor_indices, [0])
    np.testing.assert_array_equal(result.features, features)
    assert result.diagnostics["singleton_report_count"] == 1
    assert result.diagnostics["effective_permutation_fraction"] == 0


def test_identity_mapping_restores_original_and_missing_dynamic_controls():
    class IdentityRng:
        def permutation(self, values):
            return values.copy()

    times, features, bases = _inputs()
    result = _permute(times, features, bases, rng=IdentityRng())
    np.testing.assert_array_equal(result.features, features)
    assert np.isnan(result.features[:2, :, DYNAMIC_INDICES]).all()
    assert np.all(result.features[:2, :, 17] == 1)


def test_dynamic_uses_elapsed_weeks_and_never_future_pseudo_values():
    times = tuple(START + timedelta(days=days) for days in (0, 7, 21, 24, 28, 35, 42))
    weeks = np.array([0, 1, 3, 24 / 7, 4, 5, 6])
    bases = np.stack([2 + 3 * weeks, 8 + 2 * weeks], axis=-1)[:, None, :]
    dynamic = rebuild_dynamic_values(times, bases)
    np.testing.assert_allclose(dynamic[2:, 0, 0], np.arcsinh(3), atol=1e-13)
    np.testing.assert_allclose(dynamic[2:, 0, 2], np.arcsinh(2), atol=1e-13)
    changed = bases.copy()
    changed[-1] += 900
    np.testing.assert_array_equal(rebuild_dynamic_values(times, changed)[:-1], dynamic[:-1])
    np.testing.assert_array_equal(rebuild_dynamic_values(times[:4], bases[:4]), dynamic[:4])


@pytest.mark.parametrize(
    "kind", ["future_fold", "naive", "duplicate", "negative", "infinite", "shape"]
)
def test_invalid_or_out_of_fold_inputs_fail_explicitly(kind):
    times, features, bases = _inputs()
    if kind == "future_fold":
        times = (*times[:-1], datetime(2024, 7, 1, tzinfo=UTC))
    elif kind == "naive":
        times = tuple(time.replace(tzinfo=None) for time in times)
    elif kind == "duplicate":
        times = (times[0], *times[:-1])
    elif kind == "negative":
        bases[0, 0, 0] = -1
    elif kind == "infinite":
        bases[0, 0, 0] = np.inf
    else:
        bases = bases[:, :, :1]
    with pytest.raises(ValueError):
        _permute(times, features, bases)
