"""Synthetic-only checks: these do not establish real earthquake prediction skill."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.special import gammaln  # type: ignore[import-untyped]

from seismoflux.multitask_s3 import models
from seismoflux.multitask_s3.models import (
    fit_offset_poisson_intercept,
    fit_offset_poisson_ridge,
    fit_training_area_imputer,
    nonnegative_log1p,
    offset_poisson_value_and_gradient,
    signed_asinh,
)


def test_objective_gradient_matches_finite_differences() -> None:
    features = np.array([[-1.0, 0.2], [0.0, -0.1], [2.0, 0.7], [0.5, -2.0]])
    offsets, counts = np.array([0.5, 2.0, 0.1, 3.0]), np.array([0.0, 3.0, 0.0, 2.0])
    params, ridge = np.array([0.2, -0.3, 0.1]), 0.7
    value, gradient = offset_poisson_value_and_gradient(
        params, features, offsets, counts, ridge_lambda=ridge
    )
    means = offsets * np.exp(params[0] + features @ params[1:])
    expected = np.mean(means - counts * np.log(means) + gammaln(counts + 1))
    expected += 0.5 * ridge * float(params[1:] @ params[1:])
    assert value == pytest.approx(expected)
    differences = np.empty(params.size)
    for column in range(params.size):
        step = np.zeros_like(params)
        step[column] = 1.0e-6
        upper, _ = offset_poisson_value_and_gradient(
            params + step, features, offsets, counts, ridge_lambda=ridge
        )
        lower, _ = offset_poisson_value_and_gradient(
            params - step, features, offsets, counts, ridge_lambda=ridge
        )
        differences[column] = (upper - lower) / 2.0e-6
    np.testing.assert_allclose(gradient, differences, rtol=1.0e-6, atol=1.0e-8)


def test_equal_issue_objective_is_additive_and_not_event_weighted() -> None:
    features, offsets = np.array([[-1.0], [0.0], [1.0]]), np.array([0.5, 2.0, 4.0])
    counts, params = np.array([0.0, 0.0, 5.0]), np.array([0.2, 0.3])
    value, gradient = offset_poisson_value_and_gradient(
        params, features, offsets, counts, ridge_lambda=0.8
    )
    singles = [
        offset_poisson_value_and_gradient(
            params,
            features[index : index + 1],
            offsets[index : index + 1],
            counts[index : index + 1],
            ridge_lambda=0.8,
        )
        for index in range(3)
    ]
    assert value == pytest.approx(np.mean([item[0] for item in singles]))
    np.testing.assert_allclose(gradient, np.mean([item[1] for item in singles], axis=0))


def test_fit_recovers_synthetic_loglinear_counts() -> None:
    features = np.tile(np.array([[-1.0], [0.0], [1.0]]), (10, 1))
    counts = np.tile(np.array([1.0, 2.0, 4.0]), 10)
    offsets = np.ones(counts.size)
    fitted = fit_offset_poisson_ridge(features, offsets, counts, ridge_lambda=0.0)
    assert fitted.status == "fitted"
    assert fitted.intercept == pytest.approx(np.log(2.0), abs=2.0e-6)
    assert fitted.coefficients[0] / fitted.scale[0] == pytest.approx(np.log(2.0), abs=2.0e-6)
    np.testing.assert_allclose(fitted.predict(features, offsets), counts, rtol=2.0e-6)
    assert fitted.training_issue_count == 30
    assert fitted.event_count == 70


def test_zero_windows_and_unpenalized_intercept_are_retained() -> None:
    offsets, counts = np.array([1.0, 1.0, 2.0]), np.array([0.0, 0.0, 8.0])
    features = np.empty((3, 0))
    analytic = fit_offset_poisson_intercept(offsets, counts)
    fitted = fit_offset_poisson_ridge(features, offsets, counts, ridge_lambda=1.0e6)
    assert analytic.intercept == pytest.approx(np.log(2.0))
    assert fitted.status == "fitted"
    assert fitted.intercept == pytest.approx(analytic.intercept, abs=1.0e-7)
    assert fitted.training_issue_count == 3
    np.testing.assert_allclose(analytic.predict(features, offsets), [2.0, 2.0, 4.0])


def test_constant_columns_cannot_become_active_using_future_values() -> None:
    features = np.array([[4.0, -1.0], [4.0, 0.0], [4.0, 1.0]])
    fitted = fit_offset_poisson_ridge(
        features, np.ones(3), np.array([1.0, 2.0, 3.0]), ridge_lambda=0.5
    )
    assert fitted.status == "fitted"
    assert fitted.coefficients[0] == 0.0
    assert not fitted.active_coefficients[0]
    assert fitted.center[0] == 4.0
    assert fitted.scale[0] == 1.0
    future = np.array([[1.0e308, 0.5], [-1.0e308, 0.5]])
    predictions = fitted.predict(future, np.ones(2))
    assert predictions[0] == predictions[1]


def test_scaler_and_parameters_are_training_only_and_immutable() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    fitted = fit_offset_poisson_ridge(
        features, np.ones(4), np.array([0.0, 1.0, 2.0, 3.0]), ridge_lambda=1.0
    )
    frozen = fitted.to_dict()
    np.testing.assert_allclose(fitted.center, [1.5])
    np.testing.assert_allclose(fitted.scale, [np.std(features)])
    fitted.predict(np.array([[-5.0], [7.0]]), np.array([0.2, 1.5]))
    features[:] = -999
    assert fitted.to_dict() == frozen
    with pytest.raises(ValueError, match="read-only"):
        fitted.center[0] = 999


def test_repeated_issues_do_not_change_equal_issue_ridge_fit() -> None:
    features, counts, offsets = (
        np.array([[-2.0], [0.0], [1.0]]),
        np.array([0.0, 2.0, 4.0]),
        np.ones(3),
    )
    original = fit_offset_poisson_ridge(features, offsets, counts, ridge_lambda=0.7)
    repeated = fit_offset_poisson_ridge(
        np.tile(features, (3, 1)), np.tile(offsets, 3), np.tile(counts, 3), ridge_lambda=0.7
    )
    assert original.status == repeated.status == "fitted"
    np.testing.assert_allclose(original.coefficients, repeated.coefficients, atol=1.0e-10)
    assert original.intercept == pytest.approx(repeated.intercept, abs=1.0e-10)


@pytest.mark.parametrize("mode", ["empty", "all_zero"])
def test_no_training_and_all_zero_counts_have_explicit_exact_baseline_fallback(mode: str) -> None:
    size = 0 if mode == "empty" else 3
    features, offsets, counts = np.zeros((size, 2)), np.ones(size), np.zeros(size)
    fitted = fit_offset_poisson_ridge(features, offsets, counts, ridge_lambda=1.0)
    assert fitted.status == ("baseline_no_training" if size == 0 else "baseline_all_zero_targets")
    assert fitted.training_issue_count == size
    assert fitted.event_count == 0
    future_offset = np.array([0.123456789, 1.0e-300])
    np.testing.assert_array_equal(
        fitted.predict(np.array([[50.0, 99.0], [-5.0, 9.0]]), future_offset), future_offset
    )
    analytic = fit_offset_poisson_intercept(offsets, counts)
    assert analytic.status == fitted.status


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_offsets_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="offset"):
        fit_offset_poisson_ridge(
            np.zeros((1, 1)), np.array([bad]), np.array([1.0]), ridge_lambda=1.0
        )


@pytest.mark.parametrize("bad", [-1.0, 0.5, np.inf, np.nan])
def test_counts_must_be_nonnegative_finite_integers(bad: float) -> None:
    with pytest.raises(ValueError, match="counts"):
        fit_offset_poisson_ridge(np.zeros((1, 1)), np.ones(1), np.array([bad]), ridge_lambda=1.0)


def test_overflow_is_not_clipped_and_failed_optimizer_returns_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FloatingPointError):
        offset_poisson_value_and_gradient(
            np.array([1000.0]), np.empty((1, 0)), np.ones(1), np.ones(1), ridge_lambda=0.0
        )

    def failed_optimizer(*args: Any, **kwargs: Any) -> SimpleNamespace:
        np.testing.assert_array_equal(args[1], np.zeros(2))
        assert kwargs["method"] == "L-BFGS-B"
        assert kwargs["options"] == {"maxiter": 2000, "ftol": 1.0e-10, "gtol": 1.0e-6}
        return SimpleNamespace(x=np.zeros(2), success=False, message="synthetic failure")

    monkeypatch.setattr(models, "minimize", failed_optimizer)
    fitted = fit_offset_poisson_ridge(
        np.array([[-1.0], [1.0]]), np.ones(2), np.array([0.0, 2.0]), ridge_lambda=1.0
    )
    assert fitted.status == "baseline_fit_not_evaluable"
    assert "synthetic failure" in fitted.message
    np.testing.assert_array_equal(
        fitted.predict(np.zeros((2, 1)), np.array([0.5, 2.0])), [0.5, 2.0]
    )


def test_optimizer_overflow_and_scaler_overflow_are_explicit_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overflowing_optimizer(*args: Any, **kwargs: Any) -> None:
        args[0](np.array([1000.0, 0.0]))

    monkeypatch.setattr(models, "minimize", overflowing_optimizer)
    fitted = fit_offset_poisson_ridge(
        np.array([[-1.0], [1.0]]), np.ones(2), np.array([0.0, 2.0]), ridge_lambda=1.0
    )
    assert fitted.status == "baseline_fit_not_evaluable"
    extreme = fit_offset_poisson_ridge(
        np.array([[-1.0e308], [1.0e308]]), np.ones(2), np.array([0.0, 2.0]), ridge_lambda=1.0
    )
    assert extreme.status == "baseline_fit_not_evaluable"


def test_count_transforms_preserve_missingness_and_sign_without_clipping() -> None:
    values = np.array([0.0, 1.0, 1.0e100, np.nan])
    np.testing.assert_allclose(nonnegative_log1p(values), np.log1p(values), equal_nan=True)
    signed = np.array([-1.0e100, -1.0, 0.0, 1.0, 1.0e100, np.nan])
    np.testing.assert_allclose(signed_asinh(signed), np.arcsinh(signed), equal_nan=True)
    with pytest.raises(ValueError, match="non-negative"):
        nonnegative_log1p(np.array([-0.1]))
    with pytest.raises(ValueError, match="finite"):
        signed_asinh(np.array([np.inf]))


def test_imputation_uses_training_finite_area_weights_and_not_future_values() -> None:
    training = np.array([[[0.0, np.nan], [2.0, np.nan]], [[4.0, np.nan], [np.nan, np.nan]]])
    fitted = fit_training_area_imputer(training, np.array([1.0, 3.0]))
    np.testing.assert_array_equal(fitted.fill_values, [2.0, 0.0])
    np.testing.assert_array_equal(fitted.active_columns, [True, False])
    assert fitted.training_issue_count == 2
    future = np.array([[np.nan, 1000.0], [-9.0, np.nan]])
    np.testing.assert_array_equal(fitted.transform(future), [[2.0, 0.0], [-9.0, 0.0]])
    training[:] = 999
    np.testing.assert_array_equal(fitted.fill_values, [2.0, 0.0])


def test_empty_training_imputation_is_explicitly_inactive() -> None:
    fitted = fit_training_area_imputer(np.empty((0, 2, 3)), np.array([1.0, 3.0]))
    assert fitted.training_issue_count == 0
    assert not fitted.active_columns.any()
    np.testing.assert_array_equal(fitted.transform(np.full((2, 3), np.nan)), np.zeros((2, 3)))


def test_log_mean_remains_scorable_when_expected_count_underflows() -> None:
    fitted = replace(fit_offset_poisson_intercept(np.ones(1), np.ones(1)), intercept=-1000.0)
    features, offsets = np.empty((1, 0)), np.ones(1)
    log_mean = fitted.predict_log_mean(features, offsets)
    np.testing.assert_array_equal(log_mean, [-1000.0])
    np.testing.assert_array_equal(fitted.predict(features, offsets), [0.0])
    counts = np.ones(1)
    nll = np.exp(log_mean) - counts * log_mean + gammaln(counts + 1.0)
    np.testing.assert_array_equal(nll, [1000.0])
    assert np.isfinite(nll).all()


def test_log_mean_agrees_with_normal_predictions_and_exact_fallback() -> None:
    features = np.array([[-1.0], [0.0], [1.0]])
    offsets = np.array([0.5, 1.0, 2.0])
    fitted = fit_offset_poisson_ridge(
        features, offsets, np.array([0.0, 1.0, 3.0]), ridge_lambda=1.0
    )
    assert fitted.status == "fitted"
    np.testing.assert_allclose(
        np.exp(fitted.predict_log_mean(features, offsets)), fitted.predict(features, offsets)
    )
    fallback = fit_offset_poisson_ridge(features, offsets, np.zeros(3), ridge_lambda=1.0)
    np.testing.assert_array_equal(fallback.predict_log_mean(features, offsets), np.log(offsets))
    np.testing.assert_array_equal(fallback.predict(features, offsets), offsets)
