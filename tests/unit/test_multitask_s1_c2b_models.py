"""Synthetic checks of the frozen C2B location mathematics; no real catalogs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import logsumexp

from seismoflux.multitask_s1 import c2b_models
from seismoflux.multitask_s1.c2b_models import (
    C2BFitError,
    C2BTrainingIssue,
    fit_spatial_ridge,
    gaussian_log_masses,
    mix_log_masses,
    spatial_ridge_value_and_gradient,
)


def _issue(
    issue_id: str, features: np.ndarray, counts: np.ndarray, base: np.ndarray | None = None
) -> C2BTrainingIssue:
    if base is None:
        base = np.full(features.shape[0], -np.log(features.shape[0]))
    return C2BTrainingIssue(issue_id, base, features, counts)


def test_weighted_gaussian_matches_hand_computation_and_area_normalization() -> None:
    training = np.array([[0.0, 0.0], [2.0, 0.0]])
    query = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    area = np.array([1.0, 3.0, 2.0])
    weights = np.array([0.25, 0.75])
    actual = gaussian_log_masses(
        training,
        query,
        area,
        bandwidths_km=(1.0, 2.0),
        log_event_weights=np.log(weights),
        chunk_size=1,
    )
    squared = np.sum((query[:, None, :] - training[None, :, :]) ** 2, axis=2)
    for bandwidth in (1.0, 2.0):
        expected = (np.exp(-squared / (2 * bandwidth**2)) @ weights) * area
        expected /= expected.sum()
        assert_allclose(np.exp(actual[bandwidth]), expected, rtol=1e-14, atol=1e-15)
        assert_allclose(logsumexp(actual[bandwidth]), 0, atol=1e-14)
    # Multiplying every event weight by one constant must not affect spatial mass.
    shifted = gaussian_log_masses(
        training,
        query,
        area,
        bandwidths_km=(1.0,),
        log_event_weights=np.log(weights) - 10000,
    )
    assert_allclose(shifted[1.0], actual[1.0], atol=1e-12)


def test_gaussian_far_field_logs_remain_finite_when_derived_mass_underflows() -> None:
    result = gaussian_log_masses(
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0], [10000.0, 0.0]]),
        np.array([1.0, 1.0]),
        bandwidths_km=(25.0,),
    )[25.0]
    assert_allclose(result, np.array([0.0, -80000.0]), atol=1e-12)
    assert np.isfinite(result).all()
    assert np.exp(result[1]) == 0.0


def test_empty_long_history_is_area_uniform_and_equal_weights_are_default() -> None:
    query = np.array([[0.0, 0.0], [3.0, 1.0]])
    area = np.array([1.0, 4.0])
    empty = gaussian_log_masses(np.empty((0, 2)), query, area)
    for result in empty.values():
        assert_allclose(np.exp(result), np.array([0.2, 0.8]))
    training = np.array([[1.0, 0.0], [4.0, 0.0]])
    unweighted = gaussian_log_masses(training, query, area, bandwidths_km=(1.0,))
    weighted = gaussian_log_masses(
        training,
        query,
        area,
        bandwidths_km=(1.0,),
        log_event_weights=np.zeros(2),
    )
    assert_array_equal(unweighted[1.0], weighted[1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bandwidths_km": (0.0,)},
        {"bandwidths_km": (25.0, 25.0)},
        {"log_event_weights": np.array([np.nan])},
        {"chunk_size": 0},
    ],
)
def test_gaussian_rejects_invalid_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        gaussian_log_masses(np.zeros((1, 2)), np.zeros((2, 2)), np.ones(2), **kwargs)


def test_mixture_normalizes_each_component_and_omits_zero_weights() -> None:
    first = np.log(np.array([0.9, 0.1]))
    second = np.log(np.array([0.2, 0.8])) + 500.0
    actual = mix_log_masses([first, second, np.array([-90000.0, 0.0])], [3.0, 1.0, 0.0])
    assert_allclose(np.exp(actual), 0.75 * np.exp(first) + 0.25 * np.array([0.2, 0.8]))
    assert_allclose(mix_log_masses([first, second], [1.0, 0.0]), first, atol=1e-14)
    with pytest.raises(ValueError):
        mix_log_masses([first, second], [0.0, 0.0])
    with pytest.raises(ValueError):
        mix_log_masses([first, second], [1.0, -1.0])


def test_objective_gradient_matches_finite_difference_and_pools_events() -> None:
    issues = (
        _issue("one", np.array([[-1.0, 0.0], [1.0, 2.0], [0.0, 1.0]]), np.array([1, 5, 0])),
        _issue("two", np.array([[1.0, -1.0], [2.0, 1.0], [-2.0, 0.0]]), np.array([0, 1, 0])),
        _issue("empty", np.full((3, 2), 1000.0), np.zeros(3)),
    )
    beta = np.array([0.3, -0.2])
    value, gradient = spatial_ridge_value_and_gradient(beta, issues, ridge_lambda=0.1)
    difference = np.empty(2)
    for column in range(2):
        direction = np.zeros(2)
        direction[column] = 1e-6
        high = spatial_ridge_value_and_gradient(beta + direction, issues, ridge_lambda=0.1)[0]
        low = spatial_ridge_value_and_gradient(beta - direction, issues, ridge_lambda=0.1)[0]
        difference[column] = (high - low) / 2e-6
    assert_allclose(gradient, difference, rtol=1e-6, atol=1e-9)
    expected = 0.0
    for issue in issues[:2]:
        eta = issue.log_base_mass + issue.features @ beta
        expected -= np.dot(issue.future_counts, eta - logsumexp(eta)) / 7
    expected += 0.05 * np.dot(beta, beta)
    assert_allclose(value, expected, atol=1e-14)


def test_objective_keeps_far_field_observation_likelihood_in_log_space() -> None:
    issue = _issue("far", np.array([[0.0], [1.0]]), np.array([0, 1]), np.array([0.0, -80000.0]))
    value, gradient = spatial_ridge_value_and_gradient(np.zeros(1), [issue], ridge_lambda=0.1)
    assert value == 80000.0
    assert_array_equal(gradient, np.array([-1.0]))


def test_ridge_fit_does_not_treat_derived_zero_mass_as_failure() -> None:
    issue = _issue("far", np.array([[0.0], [1.0]]), np.array([0, 1]), np.array([0.0, -80000.0]))
    fit = fit_spatial_ridge([issue], np.ones(2), ridge_lambda=0.1)
    prediction = fit.predict_log_mass(issue.log_base_mass, issue.features)
    assert fit.optimizer_status == "converged"
    assert np.isfinite(prediction).all()
    assert np.exp(prediction[1]) == 0.0
    assert prediction[1] > issue.log_base_mass[1]


def test_scaler_is_equal_issue_area_weighted_and_does_not_learn_at_prediction() -> None:
    area = np.array([1.0, 3.0])
    issues = [
        _issue("one", np.array([[0.0, 3.14], [4.0, 3.14]]), np.array([0, 2])),
        _issue("empty_two", np.array([[10.0, 3.14], [14.0, 3.14]]), np.zeros(2)),
    ]
    fit = fit_spatial_ridge(issues, area, ridge_lambda=1.0)
    assert_allclose(fit.center, [8.0, 3.14])
    assert_allclose(fit.scale, [np.sqrt(28.0), 1.0])
    assert_array_equal(fit.active_coefficients, [True, False])
    assert fit.coefficients[1] == 0.0
    before = fit.to_dict()
    future = np.array([[100.0, -1000.0], [200.0, 1000.0]])
    prediction = fit.predict_log_mass(np.log(np.array([0.5, 0.5])), future)
    assert np.isfinite(prediction).all()
    assert_allclose(logsumexp(prediction), 0, atol=1e-13)
    assert before == fit.to_dict()
    json.dumps(fit.to_dict(), allow_nan=False)


def test_ridge_learns_a_synthetic_spatial_signal_with_one_frozen_optimizer() -> None:
    features = np.array([[-1.0], [0.0], [1.0]])
    issues = [_issue(str(index), features, np.array([0, 1, 9])) for index in range(3)]
    fit = fit_spatial_ridge(issues, np.ones(3), ridge_lambda=0.1)
    log_mass = fit.predict_log_mass(issues[0].log_base_mass, features)
    assert fit.optimizer_status == "converged"
    assert fit.coefficients[0] > 0.0
    assert fit.event_count == 30
    assert fit.iteration_count > 0
    assert np.exp(log_mass[2]) > 0.65
    assert -np.dot(issues[0].future_counts, log_mass) / 10 < np.log(3)


def test_empty_targets_or_constant_features_return_the_base_without_dropping_periods() -> None:
    base = np.log(np.array([0.2, 0.8]))
    empty = _issue("empty", np.array([[0.0], [1.0]]), np.zeros(2), base)
    fit = fit_spatial_ridge([empty], np.array([1.0, 4.0]), ridge_lambda=10.0)
    assert fit.optimizer_status == "no_training_events"
    assert fit.event_count == 0
    assert fit.training_issue_count == 1
    assert_array_equal(fit.coefficients, np.zeros(1))
    assert_allclose(fit.predict_log_mass(base, empty.features), base, atol=1e-14)
    constant = _issue("constant", np.full((2, 1), 3.14), np.array([1, 2]), base)
    fixed = fit_spatial_ridge([constant], np.ones(2), ridge_lambda=1.0)
    assert fixed.optimizer_status == "no_active_features"
    assert fixed.coefficients[0] == 0.0
    assert fixed.scale[0] == 1.0
    with pytest.raises(ValueError, match="legal training issues"):
        fit_spatial_ridge([], np.ones(2), ridge_lambda=1.0)


def test_numerical_failure_is_explicit_and_does_not_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def failed(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            success=False,
            x=np.array([0.0]),
            fun=1.0,
            jac=np.array([1.0]),
            message="synthetic failure",
            nit=2,
        )

    monkeypatch.setattr(c2b_models, "minimize", failed)
    issue = _issue("fit", np.array([[0.0], [1.0]]), np.array([0, 1]))
    with pytest.raises(C2BFitError, match="did not converge"):
        fit_spatial_ridge([issue], np.ones(2), ridge_lambda=1.0)
    assert len(calls) == 1
    assert calls[0]["method"] == "L-BFGS-B"
    assert calls[0]["options"] == {"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-6}


def test_training_issue_copies_inputs_and_rejects_fractional_counts() -> None:
    features = np.array([[0.0], [1.0]])
    issue = _issue("immutable", features, np.array([0, 1]))
    features[:] = 50.0
    assert_array_equal(issue.features, [[0.0], [1.0]])
    assert not issue.features.flags.writeable
    with pytest.raises(ValueError, match="integers"):
        _issue("fractional", features, np.array([0.5, 1]))
