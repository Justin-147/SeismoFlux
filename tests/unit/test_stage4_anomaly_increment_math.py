from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

import seismoflux.anomaly_increment.model as model_module
from seismoflux.anomaly_increment.contracts import (
    DesignMatrix,
    FeatureColumnContract,
    FrozenTargetRateHead,
)
from seismoflux.anomaly_increment.integration import (
    compare_integrals,
    composite_midpoint_quadrature,
    expand_midpoint_compensator_terms,
    integrate_conditional_intensity,
    lead_decay,
    spatial_grid_convergence,
    temporal_midpoint_convergence,
)
from seismoflux.anomaly_increment.model import (
    PrimaryRateHeadEvidenceInsufficient,
    SharedPoissonObjective,
    conditional_intensity,
    fit_frozen_target_rate_head,
    fit_shared_ridge_poisson,
)
from seismoflux.anomaly_increment.preprocessing import fit_frozen_preprocessor


def _design(
    values: object,
    *,
    names: tuple[str, ...] = ("x",),
    active: object | None = None,
) -> DesignMatrix:
    matrix = np.asarray(values, dtype=np.float64)
    width = matrix.shape[1]
    return DesignMatrix(
        values=matrix,
        column_names=names,
        penalty_factors=np.ones(width, dtype=np.float64),
        active_coefficients=(
            np.ones(width, dtype=np.bool_) if active is None else np.asarray(active, dtype=np.bool_)
        ),
    )


def _active_rate_head() -> FrozenTargetRateHead:
    return fit_frozen_target_rate_head(
        training_event_counts={"M5_6": 2, "M6_plus": 1},
        background_exposures={"M5_6": 4.0, "M6_plus": 4.0},
    )


def _hand_objective() -> SharedPoissonObjective:
    return SharedPoissonObjective(
        quadrature_design=_design([[1.0], [2.0]]),
        quadrature_background_exposure_by_bin=np.asarray(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float64,
        ),
        quadrature_decay=np.asarray([0.5, 1.0], dtype=np.float64),
        event_design=_design([[1.5]]),
        event_background_intensity=np.asarray([0.2], dtype=np.float64),
        event_decay=np.asarray([0.25], dtype=np.float64),
        event_magnitude_bin_ids=("M5_6",),
        rate_head=_active_rate_head(),
    )


def test_ridge_poisson_objective_and_gradient_match_hand_calculation() -> None:
    objective = _hand_objective()
    beta = np.asarray([0.4], dtype=np.float64)

    value, gradient = objective.value_and_gradient(beta)

    compensator = math.exp(0.2) + 2.5 * math.exp(0.8)
    event_log_sum = math.log(0.2 * 0.5) + 0.25 * 1.5 * 0.4
    penalty = 0.5 * 0.4**2
    expected_gradient = 0.5 * math.exp(0.2) + 5.0 * math.exp(0.8) - 0.375 + 0.4
    assert objective.evaluate(beta).compensator == pytest.approx(compensator, rel=1.0e-15)
    assert value == pytest.approx(compensator - event_log_sum + penalty, rel=1.0e-15)
    assert gradient[0] == pytest.approx(expected_gradient, rel=1.0e-15)


def test_analytic_gradient_matches_central_finite_difference() -> None:
    head = fit_frozen_target_rate_head(
        training_event_counts={"M5_6": 4, "M6_plus": 2},
        background_exposures={"M5_6": 8.0, "M6_plus": 8.0},
    )
    quadrature = _design(
        [[0.2, -0.3], [1.1, 0.7], [-0.4, 0.6]],
        names=("x", "y"),
    )
    events = _design([[0.3, 0.8], [-0.2, 0.5]], names=("x", "y"))
    objective = SharedPoissonObjective(
        quadrature_design=quadrature,
        quadrature_background_exposure_by_bin=np.asarray(
            [[0.4, 0.2], [0.1, 0.3], [0.7, 0.2]],
            dtype=np.float64,
        ),
        quadrature_decay=np.asarray([0.9, 0.7, 0.4], dtype=np.float64),
        event_design=events,
        event_background_intensity=np.asarray([0.2, 0.3], dtype=np.float64),
        event_decay=np.asarray([0.8, 0.5], dtype=np.float64),
        event_magnitude_bin_ids=("M5_6", "M6_plus"),
        rate_head=head,
    )
    beta = np.asarray([0.17, -0.23], dtype=np.float64)
    _, analytic = objective.value_and_gradient(beta)
    step = 1.0e-6
    finite_difference = np.empty_like(beta)
    for index in range(beta.size):
        delta = np.zeros_like(beta)
        delta[index] = step
        finite_difference[index] = (
            objective.evaluate(beta + delta).objective - objective.evaluate(beta - delta).objective
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, finite_difference, rtol=0.0, atol=1.0e-8)


def test_disabled_or_zero_increment_exactly_equals_frozen_background_rate_head() -> None:
    head = _active_rate_head()
    background = np.asarray([0.2, 0.3], dtype=np.float64)
    design = np.asarray([[4.0], [-5.0]], dtype=np.float64)
    expected = background * 0.5

    disabled = conditional_intensity(
        background_intensity=background,
        design_values=design,
        beta=np.asarray([9.0]),
        lead_days=np.asarray([1.0, 30.0]),
        magnitude_bin_id="M5_6",
        rate_head=head,
        increment_enabled=False,
    )
    zero_beta = conditional_intensity(
        background_intensity=background,
        design_values=design,
        beta=np.asarray([0.0]),
        lead_days=np.asarray([1.0, 30.0]),
        magnitude_bin_id="M5_6",
        rate_head=head,
    )

    assert np.array_equal(disabled, expected)
    assert np.array_equal(zero_beta, expected)


def test_zero_event_m6_is_exactly_inactive_without_pseudocount() -> None:
    head = fit_frozen_target_rate_head(
        training_event_counts={"M5_6": 3, "M6_plus": 0},
        background_exposures={"M5_6": 6.0, "M6_plus": 6.0},
    )

    m6 = head.by_id("M6_plus")
    assert m6.rate_multiplier == 0.0
    assert m6.log_rate_multiplier is None
    assert m6.status == "inactive_zero_training_events"
    assert m6.as_mapping()["pseudocount"] == 0
    intensity = conditional_intensity(
        background_intensity=np.asarray([0.2, 0.8]),
        design_values=np.asarray([[100.0], [-100.0]]),
        beta=np.asarray([5.0]),
        lead_days=7.0,
        magnitude_bin_id="M6_plus",
        rate_head=head,
    )
    assert np.array_equal(intensity, np.zeros(2, dtype=np.float64))

    objective = SharedPoissonObjective(
        quadrature_design=_design([[0.0], [0.0]]),
        quadrature_background_exposure_by_bin=np.asarray([[1.0, 1.0], [1.0, 9.0]]),
        quadrature_decay=np.asarray([1.0, 1.0]),
        event_design=_design([[0.0]]),
        event_background_intensity=np.asarray([0.5]),
        event_decay=np.asarray([1.0]),
        event_magnitude_bin_ids=("M5_6",),
        rate_head=head,
    )
    assert np.array_equal(objective.compensator_base_weights, np.asarray([0.5, 0.5]))


def test_zero_event_primary_bin_stops_before_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    head = fit_frozen_target_rate_head(
        training_event_counts={"M5_6": 0, "M6_plus": 1},
        background_exposures={"M5_6": 7.0, "M6_plus": 7.0},
    )
    objective = SharedPoissonObjective(
        quadrature_design=_design([[0.0]]),
        quadrature_background_exposure_by_bin=np.asarray([[1.0, 1.0]]),
        quadrature_decay=np.asarray([1.0]),
        event_design=_design([[0.0]]),
        event_background_intensity=np.asarray([1.0]),
        event_decay=np.asarray([1.0]),
        event_magnitude_bin_ids=("M6_plus",),
        rate_head=head,
    )

    def forbidden_optimizer(*args: object, **kwargs: object) -> None:
        raise AssertionError("optimizer must not run")

    monkeypatch.setattr(model_module, "minimize", forbidden_optimizer)
    with pytest.raises(PrimaryRateHeadEvidenceInsufficient, match="must not run"):
        fit_shared_ridge_poisson(objective)


def test_frozen_preprocessor_uses_original_null_bitmap_and_cannot_be_rescued() -> None:
    contracts = (
        FeatureColumnContract(
            source_column="count",
            logical_feature="count",
            value_output_column="value__count",
            missing_output_column="missing__count",
            transform="log1p_nonnegative",
        ),
        FeatureColumnContract(
            source_column="present",
            logical_feature="present",
            value_output_column="value__present",
            missing_output_column="missing__present",
            transform="identity_binary",
        ),
        FeatureColumnContract(
            source_column="all_null",
            logical_feature="all_null",
            value_output_column="value__all_null",
            missing_output_column="missing__all_null",
            transform="identity_finite",
        ),
    )
    preprocessor = fit_frozen_preprocessor(
        contracts,
        {
            "count": np.asarray([0.0, 3.0, np.nan, 8.0]),
            "present": np.asarray([0.0, 1.0, np.nan, 1.0]),
            "all_null": np.asarray([np.nan, np.nan, np.nan, np.nan]),
        },
    )
    frozen_hash = preprocessor.sha256
    transformed = preprocessor.transform(
        {
            "count": np.asarray([3.0, np.nan]),
            "present": np.asarray([1.0, np.nan]),
            "all_null": np.asarray([123.0, np.nan]),
        }
    )

    assert preprocessor.sha256 == frozen_hash
    assert np.array_equal(transformed.values[:, 1], np.asarray([0.0, 1.0]))
    assert np.array_equal(transformed.values[:, 3], np.asarray([0.0, 1.0]))
    assert np.array_equal(transformed.values[:, 4], np.asarray([0.0, 0.0]))
    assert np.array_equal(transformed.values[:, 5], np.asarray([0.0, 1.0]))
    assert not transformed.active_coefficients[4]
    assert not transformed.active_coefficients[5]
    assert preprocessor.statistics[2].fixed_zero_reason == "no_finite_values_in_fit_scope"


def test_preprocessor_fallback_and_serialization_are_deterministic() -> None:
    contract = FeatureColumnContract(
        source_column="surge",
        logical_feature="surge",
        value_output_column="value__surge",
        missing_output_column="missing__surge",
        transform="asinh_signed",
    )
    values = np.asarray([0.0, 0.0, 0.0, 0.0, 100.0])
    first = fit_frozen_preprocessor((contract,), {"surge": values})
    second = fit_frozen_preprocessor((contract,), {"surge": values.copy()})

    assert first.statistics[0].scale_branch == "training_population_standard_deviation"
    assert first.as_mapping() == second.as_mapping()
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_rate_head_serialization_is_order_independent_and_deterministic() -> None:
    first = fit_frozen_target_rate_head(
        training_event_counts={"M5_6": 3, "M6_plus": 0},
        background_exposures={"M5_6": 6.0, "M6_plus": 4.0},
    )
    second = fit_frozen_target_rate_head(
        training_event_counts={"M6_plus": 0, "M5_6": 3},
        background_exposures={"M6_plus": 4.0, "M5_6": 6.0},
    )

    assert first.as_mapping() == second.as_mapping()
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_midpoint_rule_partial_step_decay_and_expansion_are_frozen() -> None:
    quadrature = composite_midpoint_quadrature(2.25, maximum_step_days=1.0)
    np.testing.assert_allclose(quadrature.widths_days, [1.0, 1.0, 0.25], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        quadrature.lead_midpoints_days,
        [0.5, 1.5, 2.125],
        rtol=0.0,
        atol=0.0,
    )
    assert lead_decay(90.0) == pytest.approx(0.5, rel=0.0, abs=1.0e-15)

    issue = _design([[1.0], [2.0]])
    expanded = expand_midpoint_compensator_terms(
        issue_design=issue,
        background_spatial_mass_by_cell_and_bin=np.asarray([[0.2, 0.1], [0.8, 0.9]]),
        horizon_days=2.25,
    )
    assert expanded.design.values.shape == (6, 1)
    np.testing.assert_array_equal(expanded.design.values[:, 0], [1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    np.testing.assert_allclose(
        np.sum(expanded.background_exposure_by_bin, axis=0),
        [2.25, 2.25],
        rtol=0.0,
        atol=1.0e-15,
    )


def test_integrated_intensity_is_monotone_over_longer_windows_and_zero_path_exact() -> None:
    background = np.asarray([0.6, 0.4])
    predictor = np.asarray([1.0, -0.5])
    integrals = [
        integrate_conditional_intensity(
            background_spatial_mass=background,
            issue_linear_predictor=predictor,
            rate_multiplier=2.0,
            horizon_days=float(horizon),
        )
        for horizon in (7, 30, 90, 180, 365)
    ]
    assert all(right > left for left, right in pairwise(integrals))
    zero = integrate_conditional_intensity(
        background_spatial_mass=background,
        issue_linear_predictor=np.zeros(2),
        rate_multiplier=2.0,
        horizon_days=365.0,
    )
    assert zero == 730.0


def test_time_and_spatial_convergence_helpers_apply_frozen_gate() -> None:
    primary, reference, time_check = temporal_midpoint_convergence(
        background_spatial_mass=np.asarray([0.6, 0.4]),
        issue_linear_predictor=np.asarray([0.2, -0.1]),
        rate_multiplier=1.5,
        horizon_days=30.0,
    )
    assert primary > 0.0 and reference > 0.0
    assert time_check.passed
    assert compare_integrals(100.4, 100.0).passed
    assert not compare_integrals(100.6, 100.0).passed
    assert compare_integrals(9.0e-11, 0.0).passed
    spatial = spatial_grid_convergence(
        intensity_50km=98.0,
        intensity_25km=100.4,
        intensity_12_5km=100.0,
    )
    assert spatial.passed
    assert spatial.as_mapping()["grid_50km_role"] == (
        "reported_coarse_trend_diagnostic_not_gate_reference"
    )


def test_inputs_are_copied_read_only_and_fit_keeps_fixed_coefficients_zero() -> None:
    raw = np.asarray([[0.2, 99.0], [0.8, -99.0]], dtype=np.float64)
    quadrature = _design(raw, names=("active", "fixed"), active=[True, False])
    raw[0, 0] = 999.0
    assert quadrature.values[0, 0] == 0.2
    assert not quadrature.values.flags.writeable
    with pytest.raises(ValueError):
        quadrature.values[0, 0] = 1.0

    events = _design([[0.5, 1000.0]], names=("active", "fixed"), active=[True, False])
    objective = SharedPoissonObjective(
        quadrature_design=quadrature,
        quadrature_background_exposure_by_bin=np.asarray([[0.5, 0.5], [0.5, 0.5]]),
        quadrature_decay=np.asarray([1.0, 1.0]),
        event_design=events,
        event_background_intensity=np.asarray([0.5]),
        event_decay=np.asarray([1.0]),
        event_magnitude_bin_ids=("M5_6",),
        rate_head=_active_rate_head(),
    )
    result = fit_shared_ridge_poisson(objective)
    assert result.converged
    assert result.beta[1] == 0.0
    assert len(result.sha256) == 64
