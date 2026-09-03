"""Synthetic-only S3 training assembly checks, not earthquake skill evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from seismoflux.multitask_s1.c2b_models import C2BFitError
from seismoflux.multitask_s3 import training
from seismoflux.multitask_s3.training import (
    S3InnerBlock,
    S3InnerScore,
    S3Performance,
    S3TrainingSample,
    fit_model,
    select_and_fit,
)


def _sample(
    day: int,
    *,
    count: int = 2,
    spatial: tuple[float, float] = (2.0, 0.0),
    coverage: tuple[float, float] = (0.0, 2.0),
    offset: float = 1.0,
) -> S3TrainingSample:
    features = np.zeros((2, 20))
    features[:, 12] = coverage
    return S3TrainingSample(
        datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=day),
        features,
        np.log(np.array([0.4, 0.6])),
        offset,
        np.array(spatial),
        count,
    )


def test_sample_validates_identity_shapes_counts_and_defends_mutable_arrays() -> None:
    sample = _sample(0)
    with pytest.raises(ValueError, match="timezone"):
        replace(sample, issue_time_utc=datetime(2023, 1, 1))
    with pytest.raises(ValueError, match="shape"):
        replace(sample, features=np.zeros((2, 19)))
    with pytest.raises(ValueError, match="spatial counts"):
        replace(sample, spatial_event_counts=np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(sample, count_ms5plus=True)
    with pytest.raises(ValueError, match="positive"):
        replace(sample, offset_ms5plus=0.0)
    mutable = np.ones((2, 20))
    changed = replace(sample, features=mutable)
    mutable[0, 0] = 999.0
    assert changed.features[0, 0] == 1.0
    assert not changed.features.flags.writeable
    shared = replace(changed, issue_time_utc=changed.issue_time_utc + timedelta(days=1))
    assert shared.features is changed.features


def test_area_imputation_is_training_only_and_national_mean_uses_actual_area() -> None:
    first, second = _sample(0), _sample(7, coverage=(4.0, 6.0), count=4)
    values = first.features.copy()
    values[:, 14] = [np.nan, 8.0]
    first = replace(first, features=values)
    values = second.features.copy()
    values[:, 14] = [0.0, np.nan]
    second = replace(second, features=values)
    fit = fit_model([first, second], design="COV", areas_km2=np.array([1.0, 3.0]), ridge_lambda=1)
    assert fit.imputer.fill_values[2] == pytest.approx(6.0)
    # Coverage is already transformed: no second log1p is performed.
    assert fit.spatial is not None
    assert fit.spatial.center[0] == pytest.approx(3.5)
    assert fit.count.center[0] == pytest.approx(3.5)
    assert fit.count.scale[0] == pytest.approx(2.0)
    before = fit.to_dict()
    future = np.full((2, 20), 1000.0)
    assert np.isfinite(fit.predict_log_mass(future, first.background_log_mass)).all()
    assert np.isfinite(fit.predict_log_mean(future, 1.0))
    assert fit.to_dict() == before


@pytest.mark.parametrize("design,columns", [("COV", 5), ("SNAP", 16), ("DYN", 20)])
def test_designs_and_training_losses_are_serializable(design: str, columns: int) -> None:
    samples = [_sample(0), _sample(7, count=0, spatial=(0.0, 0.0)), _sample(14, count=4)]
    fit = fit_model(
        samples,
        design=design,  # type: ignore[arg-type]
        areas_km2=np.array([1.0, 1.0]),
        ridge_lambda=1.0,
        count_ridge_lambda=10.0,
    )
    assert fit.imputer.fill_values.size == columns
    assert fit.count.ridge_lambda == 10.0
    assert fit.spatial_ridge_lambda == 1.0
    assert fit.count.training_issue_count == 3
    assert fit.count.event_count == 6
    performance = fit.training_performance
    assert performance is not None
    assert performance.issue_count == 3
    assert performance.spatial_event_count == 4
    assert performance.spatial_nll_sum is not None
    assert performance.catalog_spatial_nll_sum is not None
    assert performance.spatial_nll_sum < performance.catalog_spatial_nll_sum
    assert fit.count_calibration.intercept == pytest.approx(np.log(2.0))
    predictions = [fit.predict_log_mean(sample.features, 1.0) for sample in samples]
    expected = sum(
        training._poisson_nll(log_mu, s.count_ms5plus)
        for log_mu, s in zip(predictions, samples, strict=True)
    )
    assert performance.count_nll_sum == pytest.approx(expected)
    json.dumps(fit.to_dict(), allow_nan=False)


def test_no_training_and_no_events_preserve_background_with_explicit_status() -> None:
    area, sample = np.array([1.0, 3.0]), _sample(0, count=0, spatial=(0.0, 0.0))
    empty = fit_model([], design="DYN", areas_km2=area, ridge_lambda=10)
    assert empty.spatial_status == "baseline_no_training"
    assert empty.count.status == "baseline_no_training"
    assert empty.training_performance is not None
    assert empty.training_performance.to_dict()["count_nll_mean"] is None
    no_events = fit_model([sample], design="DYN", areas_km2=area, ridge_lambda=10)
    assert no_events.spatial_status == "baseline_no_training_events"
    assert no_events.count.status == "baseline_all_zero_targets"
    np.testing.assert_allclose(
        no_events.predict_log_mass(sample.features, sample.background_log_mass),
        sample.background_log_mass,
    )
    assert no_events.predict_log_mean(sample.features, 7.0) == pytest.approx(np.log(7.0))


def test_all_missing_columns_and_constant_columns_never_activate_in_future() -> None:
    samples = [_sample(0), _sample(7, count=3)]
    modified = []
    for sample in samples:
        features = sample.features.copy()
        features[:, 14] = np.nan
        features[:, 15] = 5.0
        modified.append(replace(sample, features=features))
    fit = fit_model(modified, design="COV", areas_km2=np.ones(2), ridge_lambda=1.0)
    assert not fit.imputer.active_columns[2]
    assert fit.imputer.fill_values[2] == 0.0
    assert fit.spatial is not None
    assert fit.spatial.coefficients[2] == 0.0
    assert fit.spatial.coefficients[3] == 0.0
    before = fit.predict_log_mass(modified[0].features, modified[0].background_log_mass)
    future = modified[0].features.copy()
    future[:, 14:16] = [[1.0, 2.0], [100.0, 200.0]]
    np.testing.assert_allclose(
        fit.predict_log_mass(future, modified[0].background_log_mass), before
    )


def test_spatial_fit_failure_is_not_silently_reported_as_fitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise C2BFitError("synthetic frozen optimizer failure")

    monkeypatch.setattr(training, "fit_spatial_ridge", fail)
    sample = _sample(0)
    fit = fit_model([sample], design="COV", areas_km2=np.ones(2), ridge_lambda=1)
    assert fit.spatial is None
    assert fit.spatial_status == "baseline_fit_not_evaluable"
    assert "synthetic" in fit.spatial_message
    np.testing.assert_allclose(
        fit.predict_log_mass(sample.features, sample.background_log_mass),
        sample.background_log_mass,
    )


def test_inner_preprocessing_never_uses_validation_or_outer_training_values() -> None:
    early = _sample(0, coverage=(1.0, 1.0))
    later = _sample(7, coverage=(1000.0, 1000.0))
    outer = _sample(14, coverage=(3000.0, 3000.0))
    fit = select_and_fit(
        [early, outer],
        inner_blocks=[S3InnerBlock("I1", (early,), (later,))],
        design="COV",
        areas_km2=np.ones(2),
    )
    assert fit.selection is not None
    assert fit.selection.spatial_ridge_lambda == 10.0
    assert fit.selection.count_ridge_lambda == 10.0
    assert len(fit.selection.inner_scores) == 1
    assert fit.selection.inner_scores[0].imputation_fill_values[0] == 1.0
    assert fit.imputer.fill_values[0] == pytest.approx(1500.5)


def test_two_inner_blocks_select_and_refit_without_outer_validation_input() -> None:
    samples = [_sample(i * 7, count=i % 3, coverage=(float(i), float(i + 2))) for i in range(8)]
    blocks = [
        S3InnerBlock("I1", tuple(samples[:2]), tuple(samples[2:4])),
        S3InnerBlock("I2", tuple(samples[:4]), tuple(samples[4:6])),
    ]
    fit = select_and_fit(samples, inner_blocks=blocks, design="COV", areas_km2=np.ones(2))
    assert fit.selection is not None
    assert len(fit.selection.inner_scores) == 6
    assert fit.selection.spatial_reason == "pooled_inner_event_nll"
    assert fit.selection.count_reason == "pooled_inner_issue_poisson_nll"
    assert fit.imputer.training_issue_count == 8
    assert fit.count.training_issue_count == 8
    json.dumps(fit.to_dict(), allow_nan=False)


def _score(
    block: str, ridge: float, *, events: int, issues: int, spatial: float, count: float
) -> S3InnerScore:
    performance = S3Performance(issues, events, 1, spatial, count, spatial, count, count)
    return S3InnerScore(
        block, ridge, True, True, "converged", "fitted", performance, performance, ()
    )


def test_selection_pools_event_and_issue_losses_separately_not_mean_of_block_means() -> None:
    scores = []
    for ridge, first_space, second_space, first_count, second_count in [
        (0.1, 0.0, 18.0, 10.0, 0.0),
        (1.0, 5.0, 9.0, 5.0, 9.0),
        (10.0, 10.0, 0.0, 0.0, 18.0),
    ]:
        scores.extend(
            [
                _score("I1", ridge, events=1, issues=1, spatial=first_space, count=first_count),
                _score("I2", ridge, events=9, issues=9, spatial=second_space, count=second_count),
            ]
        )
    assert training._select(tuple(scores), spatial=True)[0] == 10.0
    assert training._select(tuple(scores), spatial=False)[0] == 0.1


def test_ties_choose_largest_lambda_and_failed_candidate_cannot_drop_block() -> None:
    scores = tuple(
        _score(block, ridge, events=1, issues=1, spatial=2.0, count=3.0 + 2e-11 * ridge)
        for block in ("I1", "I2")
        for ridge in training.RIDGE_CANDIDATES
    )
    assert training._select(scores, spatial=True)[0] == 10.0
    assert training._select(scores, spatial=False)[0] == 1.0
    failed = tuple(
        replace(score, spatial_status="baseline_fit_not_evaluable")
        if score.ridge_lambda == 10 and score.block_id == "I2"
        else score
        for score in scores
    )
    assert training._select(failed, spatial=True)[0] == 1.0
    all_failed = tuple(
        replace(score, spatial_status="baseline_fit_not_evaluable") for score in scores
    )
    ridge, reason = training._select(all_failed, spatial=True)
    assert ridge == 10.0
    assert reason == "fixed_10_no_candidate_evaluable_on_all_registered_blocks"


def test_count_zero_validation_windows_remain_evaluable_but_empty_training_does_not() -> None:
    train, validation = _sample(0), _sample(7, count=0, spatial=(0.0, 0.0))
    block = S3InnerBlock("I1", (train,), (validation,))
    assert training._eligible(block, spatial=False)
    assert not training._eligible(block, spatial=True)
    assert not training._eligible(S3InnerBlock("I2", (), (validation,)), spatial=False)


def test_log_mean_scoring_survives_expected_count_underflow() -> None:
    sample = _sample(0, count=2)
    fit = fit_model([sample], design="COV", areas_km2=np.ones(2), ridge_lambda=1)
    fit = replace(fit, count=replace(fit.count, intercept=-1000.0))
    assert np.exp(fit.predict_log_mean(sample.features, 1.0)) == 0.0
    result = training._performance(fit, (sample,))
    assert result.count_nll_sum == pytest.approx(2000.0 + np.log(2.0))


def test_invalid_grid_duplicate_issues_unregistered_lambda_and_time_overlap_rejected() -> None:
    sample = _sample(0)
    with pytest.raises(ValueError, match="duplicate issue"):
        fit_model([sample, sample], design="COV", areas_km2=np.ones(2), ridge_lambda=1)
    with pytest.raises(ValueError, match="grid"):
        fit_model([sample], design="COV", areas_km2=np.ones(3), ridge_lambda=1)
    with pytest.raises(ValueError, match="frozen"):
        fit_model([sample], design="COV", areas_km2=np.ones(2), ridge_lambda=0.5)
    with pytest.raises(ValueError, match="precede"):
        S3InnerBlock("I1", (sample,), (sample,))
    with pytest.raises(ValueError, match="duplicate inner"):
        select_and_fit(
            [sample],
            inner_blocks=[S3InnerBlock("I1", (), ())] * 2,
            design="COV",
            areas_km2=np.ones(2),
        )
