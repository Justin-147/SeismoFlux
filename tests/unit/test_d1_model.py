from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import seismoflux.d1_replay.model as d1_model
from seismoflux.d1_replay.model import (
    ConditionalSpatialRidgeObjective,
    ConditionalSpatialTrainingIssue,
    D1ModelFitError,
    fit_conditional_spatial_ridge,
    predict_conditional_cell_mass,
)


def _issue(
    issue_id: str,
    design: np.ndarray,
    counts: np.ndarray,
    *,
    base_mass: np.ndarray | None = None,
) -> ConditionalSpatialTrainingIssue:
    if base_mass is None:
        base_mass = np.full(design.shape[0], 1.0 / design.shape[0])
    return ConditionalSpatialTrainingIssue(
        issue_id=issue_id,
        log_base_mass=np.log(np.asarray(base_mass, dtype=np.float64)),
        design=np.asarray(design, dtype=np.float64),
        future_counts=np.asarray(counts, dtype=np.float64),
    )


def test_conditional_objective_analytic_gradient_matches_finite_difference() -> None:
    issues = (
        _issue(
            "i1",
            np.asarray([[-1.0, 0.5], [0.0, -0.25], [1.5, 0.75]]),
            np.asarray([0.0, 1.0, 3.0]),
            base_mass=np.asarray([0.2, 0.3, 0.5]),
        ),
        _issue(
            "i2",
            np.asarray([[0.25, 1.0], [-0.5, 0.0], [0.75, -1.0]]),
            np.asarray([2.0, 0.0, 1.0]),
            base_mass=np.asarray([0.4, 0.35, 0.25]),
        ),
    )
    objective = ConditionalSpatialRidgeObjective(
        issues,
        ridge_lambda=0.4,
        active_coefficients=np.asarray([True, True]),
    )
    beta = np.asarray([0.31, -0.27], dtype=np.float64)
    value, gradient = objective.value_and_gradient(beta)
    step = 1.0e-6
    finite_difference = np.empty(2, dtype=np.float64)
    for index in range(2):
        delta = np.zeros(2, dtype=np.float64)
        delta[index] = step
        left = objective.evaluate(beta - delta).objective
        right = objective.evaluate(beta + delta).objective
        finite_difference[index] = (right - left) / (2.0 * step)

    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-8)


def test_zero_target_issue_contributes_no_spatial_likelihood() -> None:
    design = np.asarray([[-1.0], [0.0], [1.0]])
    positive = _issue("positive", design, np.asarray([0.0, 1.0, 2.0]))
    zero = _issue("zero", design * 1.0e6, np.zeros(3))
    active = np.asarray([True])
    only_positive = ConditionalSpatialRidgeObjective((positive,), 0.1, active)
    with_zero = ConditionalSpatialRidgeObjective((positive, zero), 0.1, active)
    beta = np.asarray([0.2])

    assert with_zero.evaluate(beta) == only_positive.evaluate(beta)
    np.testing.assert_array_equal(
        with_zero.value_and_gradient(beta)[1],
        only_positive.value_and_gradient(beta)[1],
    )


def test_fit_has_no_intercept_fixes_inactive_group_and_normalizes_prediction() -> None:
    design = np.column_stack(
        [
            np.asarray([-1.0, 0.0, 1.0, 2.0]),
            np.asarray([10.0, 10.0, 10.0, 10.0]),
        ]
    )
    issue = _issue("fit", design, np.asarray([0.0, 0.0, 1.0, 5.0]))
    fit = fit_conditional_spatial_ridge(
        (issue,),
        ridge_lambda=0.1,
        active_coefficients=np.asarray([True, False]),
    )
    prediction = fit.predict(issue.log_base_mass, issue.design)

    assert fit.coefficients[0] > 0.0
    assert fit.coefficients[1] == 0.0
    assert fit.training_event_count == 6
    assert prediction[-1] > prediction[0]
    assert np.sum(prediction, dtype=np.float64) == pytest.approx(1.0, abs=5e-15)


def test_prediction_matches_direct_log_base_plus_design_softmax() -> None:
    base_mass = np.asarray([0.2, 0.3, 0.5])
    design = np.asarray([[-1.0], [0.0], [2.0]])
    beta = np.asarray([0.4])
    predicted = predict_conditional_cell_mass(np.log(base_mass), design, beta)
    eta = np.log(base_mass) + design[:, 0] * beta[0]
    expected = np.exp(eta - np.max(eta))
    expected /= expected.sum()
    np.testing.assert_allclose(predicted, expected, rtol=0.0, atol=1e-15)


def test_prediction_rejects_zero_density_from_numerical_underflow() -> None:
    with pytest.raises(FloatingPointError, match="normalization failed"):
        predict_conditional_cell_mass(
            np.log(np.asarray([0.5, 0.5])),
            np.asarray([[0.0], [2_000.0]]),
            np.asarray([1.0]),
        )


def test_nonfinite_input_and_nonconvergence_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _issue(
            "bad",
            np.asarray([[0.0], [np.inf]]),
            np.asarray([1.0, 0.0]),
        )

    issue = _issue(
        "valid",
        np.asarray([[-1.0], [1.0]]),
        np.asarray([0.0, 2.0]),
    )

    def _failed_minimize(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            x=np.asarray([0.0]),
            fun=1.0,
            jac=np.asarray([0.0]),
            message="synthetic nonconvergence",
            nit=1,
        )

    monkeypatch.setattr(d1_model, "minimize", _failed_minimize)
    with pytest.raises(D1ModelFitError, match="did not converge"):
        fit_conditional_spatial_ridge(
            (issue,),
            ridge_lambda=1.0,
            active_coefficients=np.asarray([True]),
        )


def test_all_zero_training_targets_are_evidence_insufficient_not_a_fake_fit() -> None:
    issue = _issue(
        "zero",
        np.asarray([[-1.0], [1.0]]),
        np.zeros(2),
    )
    with pytest.raises(ValueError, match=r"no future M4\+ training events"):
        ConditionalSpatialRidgeObjective(
            (issue,),
            ridge_lambda=1.0,
            active_coefficients=np.asarray([True]),
        )
