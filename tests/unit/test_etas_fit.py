from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

import seismoflux.background.etas_fit as etas_fit_module
from seismoflux.background.etas import (
    ETASParent,
    conditional_intensity,
    inverse_power_density,
    omori_cdf,
    omori_density,
    productivity,
)
from seismoflux.background.etas_fit import (
    ETASEvent,
    ETASExpectedCellMasses,
    ETASIssueIntensityFields,
    ETASLikelihoodProblem,
    ETASLikelihoodResult,
    ETASModelSpec,
    ETASParameterBounds,
    ETASParameters,
    ETASStartResult,
    HessianAudit,
    NumericalStencilError,
    ObservedHessianDeltaUncertainty,
    OptimizerOptions,
    PointAreaQuadrature,
    QuadraturePoint,
    StabilityAudit,
    StabilityThresholds,
    audit_stability,
    central_hessian,
    etas_log_likelihood,
    etas_objective,
    evaluate_etas_cell_expected_masses,
    evaluate_etas_issue_intensity_fields,
    evaluate_prepared_likelihood,
    fit_etas,
    observed_hessian_delta_uncertainty,
    optimizer_start,
    prepare_etas_likelihood,
    run_five_start_lbfgsb,
    three_point_gradient,
)
from seismoflux.background.randomness import SeedContext


class _BatchDensity:
    def __init__(self) -> None:
        self.scalar_calls = 0
        self.batch_calls = 0

    @staticmethod
    def _formula(
        x_km: np.ndarray[tuple[int], np.dtype[np.float64]],
        y_km: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.asarray(0.2 + 1.0e-4 * x_km + 2.0e-4 * y_km, dtype=np.float64)

    def __call__(self, x_km: float, y_km: float) -> float:
        self.scalar_calls += 1
        return float(
            self._formula(
                np.asarray([x_km], dtype=np.float64),
                np.asarray([y_km], dtype=np.float64),
            )[0]
        )

    def density_many(
        self,
        x_km: object,
        y_km: object,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        self.batch_calls += 1
        return self._formula(
            np.asarray(x_km, dtype=np.float64),
            np.asarray(y_km, dtype=np.float64),
        )


def _spec() -> ETASModelSpec:
    return ETASModelSpec(mc=3.0, beta=2.0)


def _parameters() -> ETASParameters:
    return ETASParameters(
        background_rate_per_day=0.0001,
        productivity_k=0.2,
        alpha=1.0,
        c_days=0.05,
        p=1.2,
    )


def _event(
    event_id: str,
    time_days: float,
    magnitude: float,
    *,
    available_time_days: float | None = None,
    inside_study: bool = True,
) -> ETASEvent:
    return ETASEvent(
        event_id=event_id,
        time_days=time_days,
        available_time_days=(time_days if available_time_days is None else available_time_days),
        x_km=0.0,
        y_km=0.0,
        magnitude=magnitude,
        inside_study_area=inside_study,
        inside_parent_domain=True,
    )


def _problem(*, with_ancient_parent: bool = False) -> ETASLikelihoodProblem:
    root = _event("root", 0.0, 4.0, inside_study=False)
    target = _event("target", 1.0, 3.5)
    parents = [root, target]
    if with_ancient_parent:
        parents.append(_event("ancient", -4000.0, 9.0, inside_study=False))
    return ETASLikelihoodProblem(
        assessment_start_days=0.5,
        assessment_end_days=2.0,
        target_events=(target,),
        parent_events=tuple(parents),
        background_density=lambda _x, _y: 1.0,
        spatial_integrator=PointAreaQuadrature((QuadraturePoint(0.0, 0.0, 1.0),)),
    )


def _stability_with_hessian(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    *,
    stable: bool = True,
    hessian_success: bool = True,
) -> StabilityAudit:
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    condition = maximum / minimum if minimum > 0.0 else math.inf
    return StabilityAudit(
        stable=stable,
        converged_start_count=5 if stable else 0,
        best_three_relative_objective_range=0.0 if stable else None,
        best_three_transformed_parameter_range=0.0 if stable else None,
        hessian=HessianAudit(
            success=hessian_success,
            minimum_eigenvalue=minimum if hessian_success else None,
            condition_number=condition if hessian_success else None,
            matrix=(
                tuple(tuple(float(value) for value in row) for row in matrix)
                if hessian_success
                else None
            ),
            failure_reason=None if hessian_success else "synthetic Hessian failure",
        ),
        failure_reasons=() if stable else ("synthetic unstable fit",),
    )


def _naive_log_likelihood(
    problem: ETASLikelihoodProblem,
    parameters: ETASParameters,
    spec: ETASModelSpec,
) -> ETASLikelihoodResult:
    """Small-sample scalar reference that deliberately scans every pair and cell."""

    sorted_targets = sorted(
        problem.target_events,
        key=lambda event: (event.time_days, event.event_id),
    )
    sorted_parents = sorted(
        problem.parent_events,
        key=lambda event: (event.time_days, event.event_id),
    )
    intensities: list[float] = []
    for target in sorted_targets:
        parents = [
            ETASParent(
                time_days=parent.time_days,
                x_km=parent.x_km,
                y_km=parent.y_km,
                magnitude=parent.magnitude,
            )
            for parent in sorted_parents
            if 0.0 < target.time_days - parent.time_days <= spec.history_parent_cutoff_days
            and parent.available_time_days <= target.time_days
        ]
        intensities.append(
            conditional_intensity(
                time_days=target.time_days,
                x_km=target.x_km,
                y_km=target.y_km,
                background_density_per_day_km2=(
                    parameters.background_rate_per_day
                    * problem.background_density(target.x_km, target.y_km)
                ),
                parents=parents,
                mc=spec.mc,
                k=parameters.productivity_k,
                alpha=parameters.alpha,
                c_days=parameters.c_days,
                p=parameters.p,
                d_km2=spec.d_km2,
                q=spec.q,
                gamma=spec.gamma,
                spatial_cutoff_km=spec.spatial_cutoff_km,
            )
        )
    background_mass = problem.spatial_integrator.integrate(problem.background_density)
    background_compensator = (
        parameters.background_rate_per_day
        * (problem.assessment_end_days - problem.assessment_start_days)
        * background_mass
    )
    trigger_terms: list[float] = []
    for parent in sorted_parents:
        lower_age = (
            max(problem.assessment_start_days, parent.available_time_days) - parent.time_days
        )
        upper_age = min(
            spec.history_parent_cutoff_days,
            problem.assessment_end_days - parent.time_days,
        )
        if upper_age <= lower_age:
            continue

        def spatial_density(
            x_km: float,
            y_km: float,
            parent_event: ETASEvent = parent,
        ) -> float:
            return inverse_power_density(
                math.hypot(x_km - parent_event.x_km, y_km - parent_event.y_km),
                parent_event.magnitude,
                d_km2=spec.d_km2,
                q=spec.q,
                gamma=spec.gamma,
                mc=spec.mc,
                cutoff_radius_km=spec.spatial_cutoff_km,
            )

        spatial_mass = problem.spatial_integrator.integrate(spatial_density)
        temporal_mass = omori_cdf(
            upper_age,
            c_days=parameters.c_days,
            p=parameters.p,
        ) - omori_cdf(
            lower_age,
            c_days=parameters.c_days,
            p=parameters.p,
        )
        trigger_terms.append(
            productivity(
                parent.magnitude,
                k=parameters.productivity_k,
                alpha=parameters.alpha,
                mc=spec.mc,
            )
            * temporal_mass
            * spatial_mass
        )
    event_log_sum = math.fsum(math.log(value) for value in intensities)
    triggering_compensator = math.fsum(trigger_terms)
    total_compensator = background_compensator + triggering_compensator
    return ETASLikelihoodResult(
        target_event_ids=tuple(event.event_id for event in sorted_targets),
        event_intensities=tuple(intensities),
        event_log_intensity_sum=event_log_sum,
        background_compensator=background_compensator,
        triggering_compensator=triggering_compensator,
        total_compensator=total_compensator,
        log_likelihood=event_log_sum - total_compensator,
    )


def test_seed_context_matches_frozen_reference_and_pcg64_is_reproducible() -> None:
    context = SeedContext(
        root_seed=147,
        protocol_version="0.2.0",
        namespace="simulation_regression",
        model_id="etas_inverse_power_cut300_v1",
        issue_id=None,
        replicate_index=0,
    )

    assert context.digest().hex() == (
        "ee4831fcb2a99d18069fbb1fe427c2ef8fba0055d8bd2efa26cfdd7d32d36dcb"
    )
    assert context.entropy() == 316731122229485399933275391768849269487
    assert np.array_equal(context.generator().random(8), context.generator().random(8))
    with pytest.raises(ValueError, match="NUL"):
        SeedContext(147, "0.2.0", "future\x00simulation", "etas/final", None, 0).digest()


def test_likelihood_contains_event_and_study_area_compensator_terms() -> None:
    spec = _spec()
    parameters = _parameters()
    result = etas_log_likelihood(_problem(), parameters, spec)

    spatial_root = inverse_power_density(
        0.0,
        4.0,
        d_km2=spec.d_km2,
        q=spec.q,
        gamma=spec.gamma,
        mc=spec.mc,
        cutoff_radius_km=spec.spatial_cutoff_km,
    )
    spatial_target = inverse_power_density(
        0.0,
        3.5,
        d_km2=spec.d_km2,
        q=spec.q,
        gamma=spec.gamma,
        mc=spec.mc,
        cutoff_radius_km=spec.spatial_cutoff_km,
    )
    expected_intensity = (
        parameters.background_rate_per_day
        + productivity(
            4.0,
            k=parameters.productivity_k,
            alpha=parameters.alpha,
            mc=spec.mc,
        )
        * 0.10360884626454828
        * spatial_root
    )
    expected_background_compensator = parameters.background_rate_per_day * 1.5
    expected_trigger = (
        productivity(
            4.0,
            k=parameters.productivity_k,
            alpha=parameters.alpha,
            mc=spec.mc,
        )
        * (omori_cdf(2.0, c_days=0.05, p=1.2) - omori_cdf(0.5, c_days=0.05, p=1.2))
        * spatial_root
        + productivity(
            3.5,
            k=parameters.productivity_k,
            alpha=parameters.alpha,
            mc=spec.mc,
        )
        * omori_cdf(1.0, c_days=0.05, p=1.2)
        * spatial_target
    )

    assert result.target_event_ids == ("target",)
    assert result.event_intensities == pytest.approx((expected_intensity,), rel=5.0e-12)
    assert result.background_compensator == pytest.approx(expected_background_compensator)
    assert result.triggering_compensator == pytest.approx(expected_trigger, rel=5.0e-12)
    assert result.total_compensator == pytest.approx(
        expected_background_compensator + expected_trigger
    )
    assert result.log_likelihood == pytest.approx(
        math.log(expected_intensity) - expected_background_compensator - expected_trigger
    )


def test_parent_available_between_targets_updates_event_term_and_compensator() -> None:
    spec = _spec()
    parameters = _parameters()
    root = _event(
        "root",
        0.0,
        4.0,
        available_time_days=1.5,
        inside_study=False,
    )
    target_before = _event("target-before", 1.0, 3.5, available_time_days=3.0)
    target_after = _event("target-after", 2.0, 3.5, available_time_days=3.0)
    problem = ETASLikelihoodProblem(
        assessment_start_days=0.5,
        assessment_end_days=2.5,
        target_events=(target_before, target_after),
        parent_events=(root, target_before, target_after),
        background_density=lambda _x, _y: 1.0,
        spatial_integrator=PointAreaQuadrature((QuadraturePoint(0.0, 0.0, 1.0),)),
    )

    result = etas_log_likelihood(problem, parameters, spec)
    root_spatial_density = inverse_power_density(
        0.0,
        root.magnitude,
        d_km2=spec.d_km2,
        q=spec.q,
        gamma=spec.gamma,
        mc=spec.mc,
        cutoff_radius_km=spec.spatial_cutoff_km,
    )
    expected_after_intensity = parameters.background_rate_per_day + (
        productivity(
            root.magnitude,
            k=parameters.productivity_k,
            alpha=parameters.alpha,
            mc=spec.mc,
        )
        * omori_density(2.0, c_days=parameters.c_days, p=parameters.p)
        * root_spatial_density
    )
    expected_triggering_compensator = (
        productivity(
            root.magnitude,
            k=parameters.productivity_k,
            alpha=parameters.alpha,
            mc=spec.mc,
        )
        * (
            omori_cdf(2.5, c_days=parameters.c_days, p=parameters.p)
            - omori_cdf(1.5, c_days=parameters.c_days, p=parameters.p)
        )
        * root_spatial_density
    )

    assert result.event_intensities == pytest.approx(
        (parameters.background_rate_per_day, expected_after_intensity)
    )
    assert result.triggering_compensator == pytest.approx(expected_triggering_compensator)


def test_parent_available_after_interval_never_contributes_but_target_remains() -> None:
    parent = _event(
        "late-parent",
        0.0,
        4.0,
        available_time_days=3.0,
        inside_study=False,
    )
    target = _event("late-target", 1.0, 3.5, available_time_days=3.0)
    problem = ETASLikelihoodProblem(
        assessment_start_days=0.5,
        assessment_end_days=2.0,
        target_events=(target,),
        parent_events=(parent, target),
        background_density=lambda _x, _y: 1.0,
        spatial_integrator=PointAreaQuadrature((QuadraturePoint(0.0, 0.0, 1.0),)),
    )

    result = etas_log_likelihood(problem, _parameters(), _spec())

    assert result.target_event_ids == ("late-target",)
    assert result.event_intensities == pytest.approx((_parameters().background_rate_per_day,))
    assert result.triggering_compensator == 0.0


def test_parent_available_exactly_at_target_time_is_visible() -> None:
    parent = _event(
        "boundary-parent",
        0.0,
        4.0,
        available_time_days=1.0,
        inside_study=False,
    )
    target = _event("boundary-target", 1.0, 3.5, available_time_days=2.0)
    problem = ETASLikelihoodProblem(
        assessment_start_days=0.5,
        assessment_end_days=1.5,
        target_events=(target,),
        parent_events=(parent, target),
        background_density=lambda _x, _y: 1.0,
        spatial_integrator=PointAreaQuadrature((QuadraturePoint(0.0, 0.0, 1.0),)),
    )

    prepared = prepare_etas_likelihood(problem, _spec())
    result = evaluate_prepared_likelihood(prepared, _parameters())

    assert prepared.event_parent_delta_days.tolist() == [1.0]
    assert result.event_intensities[0] > _parameters().background_rate_per_day


def test_batch_background_density_matches_scalar_and_avoids_point_calls() -> None:
    base = _problem()
    quadrature = PointAreaQuadrature(
        (
            QuadraturePoint(0.0, 0.0, 1.0),
            QuadraturePoint(10.0, 5.0, 2.0),
            QuadraturePoint(20.0, 10.0, 3.0),
        )
    )
    batch_density = _BatchDensity()
    batch_problem = ETASLikelihoodProblem(
        assessment_start_days=base.assessment_start_days,
        assessment_end_days=base.assessment_end_days,
        target_events=base.target_events,
        parent_events=base.parent_events,
        background_density=batch_density,
        spatial_integrator=quadrature,
    )
    scalar_problem = ETASLikelihoodProblem(
        assessment_start_days=base.assessment_start_days,
        assessment_end_days=base.assessment_end_days,
        target_events=base.target_events,
        parent_events=base.parent_events,
        background_density=lambda x_km, y_km: float(
            _BatchDensity._formula(
                np.asarray([x_km], dtype=np.float64),
                np.asarray([y_km], dtype=np.float64),
            )[0]
        ),
        spatial_integrator=quadrature,
    )

    batch_result = etas_log_likelihood(batch_problem, _parameters(), _spec())
    scalar_result = etas_log_likelihood(scalar_problem, _parameters(), _spec())

    assert batch_density.batch_calls == 2
    assert batch_density.scalar_calls == 0
    assert batch_result.event_intensities == pytest.approx(scalar_result.event_intensities)
    assert batch_result.background_compensator == pytest.approx(
        scalar_result.background_compensator
    )
    assert batch_result.triggering_compensator == pytest.approx(
        scalar_result.triggering_compensator
    )
    assert batch_result.log_likelihood == pytest.approx(scalar_result.log_likelihood)


def test_expected_cell_masses_match_likelihood_and_naive_cells() -> None:
    spec = _spec()
    parameters = _parameters()
    parent = _event(
        "delayed-parent",
        0.0,
        4.0,
        available_time_days=1.5,
        inside_study=False,
    )
    quadrature = PointAreaQuadrature(
        (
            QuadraturePoint(0.0, 0.0, 2.0),
            QuadraturePoint(300.0, 0.0, 3.0),
            QuadraturePoint(300.001, 0.0, 5.0),
        )
    )
    batch_density = _BatchDensity()
    problem = ETASLikelihoodProblem(
        assessment_start_days=0.5,
        assessment_end_days=2.5,
        target_events=(),
        parent_events=(parent,),
        background_density=batch_density,
        spatial_integrator=quadrature,
    )

    cell_masses = evaluate_etas_cell_expected_masses(
        problem,
        parameters,
        spec,
        quadrature,
    )
    likelihood = etas_log_likelihood(problem, parameters, spec)
    temporal_mass = omori_cdf(
        2.5,
        c_days=parameters.c_days,
        p=parameters.p,
    ) - omori_cdf(
        1.5,
        c_days=parameters.c_days,
        p=parameters.p,
    )
    parent_weight = (
        productivity(
            parent.magnitude,
            k=parameters.productivity_k,
            alpha=parameters.alpha,
            mc=spec.mc,
        )
        * temporal_mass
    )
    expected_triggering = np.asarray(
        [
            parent_weight
            * inverse_power_density(
                point.x_km,
                parent.magnitude,
                d_km2=spec.d_km2,
                q=spec.q,
                gamma=spec.gamma,
                mc=spec.mc,
                cutoff_radius_km=spec.spatial_cutoff_km,
            )
            * point.area_km2
            for point in quadrature.points
        ],
        dtype=np.float64,
    )
    expected_background = np.asarray(
        [
            parameters.background_rate_per_day
            * 2.0
            * batch_density(point.x_km, point.y_km)
            * point.area_km2
            for point in quadrature.points
        ],
        dtype=np.float64,
    )

    assert isinstance(cell_masses, ETASExpectedCellMasses)
    assert cell_masses.background_mass == pytest.approx(expected_background)
    assert cell_masses.triggering_mass == pytest.approx(expected_triggering, rel=2.0e-14)
    assert cell_masses.triggering_mass[1] > 0.0
    assert cell_masses.triggering_mass[2] == 0.0
    assert cell_masses.background_total == pytest.approx(likelihood.background_compensator)
    assert cell_masses.triggering_total == pytest.approx(
        likelihood.triggering_compensator,
        rel=2.0e-14,
    )
    assert cell_masses.total == pytest.approx(likelihood.total_compensator, rel=2.0e-14)
    assert np.all(np.isfinite(cell_masses.total_mass))
    assert not cell_masses.background_mass.flags.writeable
    assert not cell_masses.triggering_mass.flags.writeable
    assert not cell_masses.total_mass.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cell_masses.total_mass[0] = 0.0


def test_issue_intensity_fields_match_naive_three_grids_and_causal_boundaries() -> None:
    spec = _spec()
    parameters = _parameters()
    visible = ETASEvent("visible", -1.0, 0.0, 0.0, 0.0, 4.0, False, True)
    delayed = ETASEvent("delayed", -1.0, 0.1, 0.0, 0.0, 9.0, False, True)
    simultaneous = ETASEvent("simultaneous", 0.0, 0.0, 0.0, 0.0, 9.0, False, True)
    cutoff = ETASEvent("cutoff", -3650.0, -3650.0, 1000.0, 0.0, 4.0, False, True)
    too_old = ETASEvent("too-old", -3650.001, -3650.001, 0.0, 0.0, 9.0, False, True)
    quadrature = PointAreaQuadrature(
        (
            QuadraturePoint(0.0, 0.0, 1.0),
            QuadraturePoint(300.0, 0.0, 1.0),
            QuadraturePoint(300.001, 0.0, 1.0),
        )
    )
    batch_density = _BatchDensity()

    fields = evaluate_etas_issue_intensity_fields(
        parameters,
        spec,
        (visible, delayed, simultaneous, cutoff, too_old),
        batch_density,
        {50.0: quadrature, 25.0: quadrature, 12.5: quadrature},
        issue_time_days=0.0,
    )

    visible_weight = productivity(
        visible.magnitude,
        k=parameters.productivity_k,
        alpha=parameters.alpha,
        mc=spec.mc,
    ) * omori_density(1.0, c_days=parameters.c_days, p=parameters.p)
    expected_triggering = np.asarray(
        [
            visible_weight
            * inverse_power_density(
                math.hypot(point.x_km - visible.x_km, point.y_km - visible.y_km),
                visible.magnitude,
                d_km2=spec.d_km2,
                q=spec.q,
                gamma=spec.gamma,
                mc=spec.mc,
                cutoff_radius_km=spec.spatial_cutoff_km,
            )
            for point in quadrature.points
        ],
        dtype=np.float64,
    )
    expected_background = parameters.background_rate_per_day * _BatchDensity._formula(
        np.asarray([point.x_km for point in quadrature.points], dtype=np.float64),
        np.asarray([point.y_km for point in quadrature.points], dtype=np.float64),
    )

    assert isinstance(fields, ETASIssueIntensityFields)
    assert fields.eligible_parent_event_ids == ("cutoff", "visible")
    assert batch_density.batch_calls == 3
    assert batch_density.scalar_calls == 0
    for cell_size in (50.0, 25.0, 12.5):
        grid_field = fields.at(cell_size)
        assert grid_field.background_intensity == pytest.approx(expected_background)
        assert grid_field.triggering_intensity == pytest.approx(
            expected_triggering,
            rel=2.0e-14,
        )
        assert grid_field.triggering_intensity[1] > 0.0
        assert grid_field.triggering_intensity[2] == 0.0
        assert grid_field.total_intensity == pytest.approx(
            expected_background + expected_triggering
        )
        assert np.all(np.isfinite(grid_field.total_intensity))
        assert not grid_field.background_intensity.flags.writeable
        assert not grid_field.triggering_intensity.flags.writeable
        assert not grid_field.total_intensity.flags.writeable

    with pytest.raises(ValueError, match="exactly 50, 25, and 12.5"):
        evaluate_etas_issue_intensity_fields(
            parameters,
            spec,
            (visible,),
            batch_density,
            {25.0: quadrature, 12.5: quadrature},
        )


def test_tree_and_vectorized_likelihood_matches_naive_pair_and_cell_scans() -> None:
    spec = _spec()
    parameters = _parameters()
    root = ETASEvent("root", -1.0, -1.0, 0.0, 0.0, 4.0, False, True)
    cutoff_parent = ETASEvent("at-cutoff", 0.0, 0.0, 300.0, 0.0, 4.5, False, True)
    outside_parent = ETASEvent("outside-cutoff", 0.0, 0.0, 300.001, 0.0, 4.5, False, True)
    target_1 = ETASEvent("target-1", 1.0, 1.0, 0.0, 0.0, 3.5, True, True)
    target_2 = ETASEvent("target-2", 2.0, 2.0, 0.0, 0.0, 3.8, True, True)
    quadrature = PointAreaQuadrature(
        (
            QuadraturePoint(0.0, 0.0, 2.0),
            QuadraturePoint(10.0, 0.0, 3.0),
        )
    )
    problem = ETASLikelihoodProblem(
        assessment_start_days=0.0,
        assessment_end_days=3.0,
        target_events=(target_1, target_2),
        parent_events=(root, cutoff_parent, outside_parent, target_1, target_2),
        background_density=lambda _x, _y: 0.2,
        spatial_integrator=quadrature,
    )

    prepared = prepare_etas_likelihood(problem, spec)
    optimized = evaluate_prepared_likelihood(prepared, parameters)
    naive = _naive_log_likelihood(problem, parameters, spec)

    assert prepared.event_parent_delta_days.size == 5
    assert optimized.target_event_ids == naive.target_event_ids
    assert optimized.event_intensities == pytest.approx(naive.event_intensities, rel=2.0e-14)
    assert optimized.event_log_intensity_sum == pytest.approx(
        naive.event_log_intensity_sum, rel=2.0e-14
    )
    assert optimized.background_compensator == pytest.approx(naive.background_compensator)
    assert optimized.triggering_compensator == pytest.approx(
        naive.triggering_compensator, rel=2.0e-14
    )
    assert optimized.total_compensator == pytest.approx(naive.total_compensator, rel=2.0e-14)
    assert optimized.log_likelihood == pytest.approx(naive.log_likelihood, rel=2.0e-14)


def test_medium_prepare_uses_sparse_tree_pairs_and_batched_quadrature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()

    def coordinates(index: int) -> tuple[float, float]:
        return float(index % 50) * 120.0, float(index // 50) * 120.0

    prehistory = tuple(
        ETASEvent(
            f"history-{index:04d}",
            -1000.0 + float(index),
            -1000.0 + float(index),
            *coordinates(index),
            3.5,
            False,
            True,
        )
        for index in range(800)
    )
    targets = tuple(
        ETASEvent(
            f"target-{index:04d}",
            float(index + 1),
            float(index + 1),
            *coordinates((index * 3) % 800),
            3.5,
            True,
            True,
        )
        for index in range(200)
    )
    points = tuple(
        QuadraturePoint(float(column) * 120.0, float(row) * 120.0, 14_400.0)
        for row in range(20)
        for column in range(50)
    )
    background_call_count = 0

    def background_density(_x_km: float, _y_km: float) -> float:
        nonlocal background_call_count
        background_call_count += 1
        return 1.0 / (len(points) * 14_400.0)

    scalar_density = Mock(wraps=inverse_power_density)
    monkeypatch.setattr(etas_fit_module, "inverse_power_density", scalar_density)
    parents = (*prehistory, *targets)
    problem = ETASLikelihoodProblem(
        assessment_start_days=0.0,
        assessment_end_days=201.0,
        target_events=targets,
        parent_events=parents,
        background_density=background_density,
        spatial_integrator=PointAreaQuadrature(points),
    )

    prepared = prepare_etas_likelihood(problem, spec)
    result = evaluate_prepared_likelihood(
        prepared,
        ETASParameters(0.5, 0.2, 1.0, 0.05, 1.2),
    )

    assert scalar_density.call_count == 0
    assert background_call_count == len(targets) + len(points)
    assert prepared.event_parent_delta_days.size < len(targets) * len(parents) // 10
    assert prepared.compensator_parent_magnitude.size == len(parents)
    assert prepared.event_parent_delta_days.flags.c_contiguous
    assert not prepared.event_parent_delta_days.flags.writeable
    assert len(result.event_intensities) == len(targets)
    assert result.total_compensator >= 0.0
    assert math.isfinite(result.log_likelihood)


def test_3650_day_parent_cutoff_omits_old_parent_without_renormalization() -> None:
    baseline = etas_log_likelihood(_problem(), _parameters(), _spec())
    with_old_parent = etas_log_likelihood(
        _problem(with_ancient_parent=True),
        _parameters(),
        _spec(),
    )

    assert with_old_parent == baseline


def test_likelihood_contract_rejects_leakage_domains_and_magnitude_overflow() -> None:
    with pytest.raises(ValueError, match="available_time_days"):
        _event("reported-before-origin", 1.0, 3.5, available_time_days=0.9)

    target = _event("target", 1.0, 3.5)
    with pytest.raises(ValueError, match="parent domain"):
        ETASLikelihoodProblem(
            assessment_start_days=0.0,
            assessment_end_days=2.0,
            target_events=(target,),
            parent_events=(ETASEvent("target", 1.0, 1.0, 0.0, 0.0, 3.5, True, False),),
            background_density=lambda _x, _y: 1.0,
            spatial_integrator=PointAreaQuadrature((QuadraturePoint(0.0, 0.0, 1.0),)),
        )
    with pytest.raises(ValueError, match="after assessment_end"):
        ETASLikelihoodProblem(
            assessment_start_days=0.0,
            assessment_end_days=2.0,
            target_events=(target,),
            parent_events=(target, _event("future", 2.1, 4.0)),
            background_density=lambda _x, _y: 1.0,
            spatial_integrator=PointAreaQuadrature((QuadraturePoint(0.0, 0.0, 1.0),)),
        )
    with pytest.raises(ValueError, match="targets must equal all"):
        ETASLikelihoodProblem(
            assessment_start_days=0.0,
            assessment_end_days=2.0,
            target_events=(target,),
            parent_events=(target, _event("silently-omitted-target", 1.5, 4.0)),
            background_density=lambda _x, _y: 1.0,
            spatial_integrator=PointAreaQuadrature((QuadraturePoint(0.0, 0.0, 1.0),)),
        )

    over_maximum = _event("over", 0.0, 9.6, inside_study=False)
    problem = ETASLikelihoodProblem(
        assessment_start_days=0.5,
        assessment_end_days=2.0,
        target_events=(target,),
        parent_events=(over_maximum, target),
        background_density=lambda _x, _y: 1.0,
        spatial_integrator=PointAreaQuadrature((QuadraturePoint(0.0, 0.0, 1.0),)),
    )
    with pytest.raises(ValueError, match="exceeds maximum"):
        etas_log_likelihood(problem, _parameters(), _spec())

    with pytest.raises(ValueError, match="frozen at 300"):
        ETASModelSpec(mc=3.0, beta=2.0, spatial_cutoff_km=301.0)


def test_invalid_branching_point_maps_to_positive_infinity_objective() -> None:
    bounds = ETASParameterBounds()
    invalid = ETASParameters(1.0, 0.5, 2.0, 0.05, 1.2)
    objective = etas_objective(_problem(), _spec(), bounds)

    assert math.isinf(objective(bounds.to_transformed(invalid)))


def test_three_point_gradient_and_central_hessian_match_convex_reference() -> None:
    target = np.asarray([0.3, -0.2, 0.7, -0.4, 0.1], dtype=np.float64)
    bounds = tuple((-5.0, 5.0) for _ in range(5))

    def objective(point: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        return float(np.sum((point - target) ** 2))

    point = np.asarray([1.0, 0.0, -0.5, 0.8, -0.9], dtype=np.float64)
    gradient = three_point_gradient(objective, point, bounds)
    hessian = central_hessian(objective, point, bounds)

    assert gradient == pytest.approx(2.0 * (point - target), rel=1.0e-9, abs=1.0e-9)
    assert hessian == pytest.approx(2.0 * np.eye(5), rel=1.0e-7, abs=1.0e-7)

    at_lower_bound = point.copy()
    at_lower_bound[0] = -5.0
    one_sided = three_point_gradient(objective, at_lower_bound, bounds)
    assert one_sided[0] == pytest.approx(2.0 * (-5.0 - target[0]), rel=1.0e-9)
    with pytest.raises(NumericalStencilError, match="crosses"):
        central_hessian(objective, at_lower_bound, bounds)


def test_five_start_lbfgsb_and_stability_audit_are_deterministic() -> None:
    target = np.asarray([0.3, -0.2, 0.7, -0.4, 0.1], dtype=np.float64)
    bounds = tuple((-5.0, 5.0) for _ in range(5))

    def objective(point: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        return float(np.sum((point - target) ** 2))

    first = run_five_start_lbfgsb(
        objective,
        bounds,
        root_seed=147,
        protocol_version="0.2.0",
        model_id="etas/fold_1",
    )
    second = run_five_start_lbfgsb(
        objective,
        bounds,
        root_seed=147,
        protocol_version="0.2.0",
        model_id="etas/fold_1",
    )
    audit = audit_stability(objective, first, bounds)

    assert first == second
    assert len(first) == 5
    assert all(result.scipy_converged for result in first)
    assert audit.stable is True
    assert audit.converged_start_count == 5
    assert audit.hessian.success is True


def test_observed_hessian_delta_uncertainty_matches_diagonal_reference() -> None:
    parameters = ETASParameters(2.0, 0.1, 0.5, 2.0, 1.2)
    spec = _spec()
    hessian = np.diag(np.asarray([4.0, 9.0, 16.0, 25.0, 36.0], dtype=np.float64))
    stability = _stability_with_hessian(hessian)

    uncertainty = observed_hessian_delta_uncertainty(parameters, stability, spec)

    assert isinstance(uncertainty, ObservedHessianDeltaUncertainty)
    assert uncertainty.confidence_level == 0.95
    transformed_covariance = np.asarray(uncertainty.transformed_covariance)
    physical_covariance = np.asarray(uncertainty.physical_covariance)
    assert transformed_covariance == pytest.approx(
        np.diag(1.0 / np.diag(hessian)),
        rel=1.0e-14,
        abs=1.0e-14,
    )
    expected_standard_errors = np.asarray([2.0 / 2.0, 0.1 / 3.0, 0.5 / 4.0, 2.0 / 5.0, 0.2 / 6.0])
    assert np.sqrt(np.diag(physical_covariance)) == pytest.approx(
        expected_standard_errors,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert tuple(item.standard_error for item in uncertainty.parameter_estimates) == pytest.approx(
        expected_standard_errors
    )

    alpha_step = 1.0e-6
    ratio_plus = spec.branching_ratio(
        ETASParameters(
            parameters.background_rate_per_day,
            parameters.productivity_k,
            parameters.alpha + alpha_step,
            parameters.c_days,
            parameters.p,
        )
    )
    ratio_minus = spec.branching_ratio(
        ETASParameters(
            parameters.background_rate_per_day,
            parameters.productivity_k,
            parameters.alpha - alpha_step,
            parameters.c_days,
            parameters.p,
        )
    )
    branching_gradient = np.asarray(
        [
            0.0,
            uncertainty.branching_ratio.estimate / parameters.productivity_k,
            (ratio_plus - ratio_minus) / (2.0 * alpha_step),
            0.0,
            0.0,
        ]
    )
    expected_branching_se = math.sqrt(
        float(branching_gradient @ physical_covariance @ branching_gradient)
    )
    assert uncertainty.branching_ratio.standard_error == pytest.approx(
        expected_branching_se,
        rel=1.0e-9,
    )
    assert uncertainty.branching_ratio.confidence_interval_lower == pytest.approx(
        uncertainty.branching_ratio.estimate - 1.959963984540054 * expected_branching_se
    )
    assert uncertainty.branching_ratio.confidence_interval_upper == pytest.approx(
        uncertainty.branching_ratio.estimate + 1.959963984540054 * expected_branching_se
    )
    assert isinstance(uncertainty.physical_covariance, tuple)
    assert all(isinstance(row, tuple) for row in uncertainty.physical_covariance)


def test_observed_hessian_delta_uncertainty_fails_closed() -> None:
    parameters = ETASParameters(2.0, 0.1, 0.5, 2.0, 1.2)
    spec = _spec()
    valid_hessian = np.eye(5, dtype=np.float64)
    stable = _stability_with_hessian(valid_hessian)

    with pytest.raises(ValueError, match="requires fitted parameters"):
        observed_hessian_delta_uncertainty(None, stable, spec)
    with pytest.raises(ValueError, match="requires a stable"):
        observed_hessian_delta_uncertainty(
            parameters,
            _stability_with_hessian(valid_hessian, stable=False),
            spec,
        )
    with pytest.raises(ValueError, match="successful Hessian"):
        observed_hessian_delta_uncertainty(
            parameters,
            _stability_with_hessian(valid_hessian, hessian_success=False),
            spec,
        )

    non_positive_definite = np.diag(np.asarray([1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="positive definite"):
        observed_hessian_delta_uncertainty(
            parameters,
            _stability_with_hessian(non_positive_definite),
            spec,
        )

    ill_conditioned = np.diag(np.asarray([1.0e-8, 1.0, 1.0, 1.0, 101.0], dtype=np.float64))
    with pytest.raises(ValueError, match="condition-number"):
        observed_hessian_delta_uncertainty(
            parameters,
            _stability_with_hessian(ill_conditioned),
            spec,
        )


def test_fit_etas_attaches_uncertainty_only_to_stable_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = ETASParameters(2.0, 0.1, 0.5, 2.0, 1.2)
    bounds = ETASParameterBounds()
    transformed = tuple(float(value) for value in bounds.to_transformed(parameters))
    start = ETASStartResult(
        start_index=0,
        initial_transformed=transformed,
        final_transformed=transformed,
        objective=1.0,
        scipy_converged=True,
        gradient_infinity_norm=0.0,
        iterations=1,
        function_evaluations=1,
        message="synthetic success",
    )
    stability = _stability_with_hessian(np.eye(5, dtype=np.float64))
    monkeypatch.setattr(
        etas_fit_module,
        "run_five_start_lbfgsb",
        lambda *_args, **_kwargs: (start,),
    )
    monkeypatch.setattr(
        etas_fit_module,
        "audit_stability",
        lambda *_args, **_kwargs: stability,
    )

    result = fit_etas(
        _problem(),
        _spec(),
        root_seed=147,
        protocol_version="0.2.0",
        model_id="etas/fold_1",
        bounds=bounds,
    )

    assert result.best_parameters is not None
    assert result.best_parameters.background_rate_per_day == pytest.approx(
        parameters.background_rate_per_day
    )
    assert result.best_parameters.productivity_k == pytest.approx(parameters.productivity_k)
    assert result.best_parameters.alpha == pytest.approx(parameters.alpha)
    assert result.best_parameters.c_days == pytest.approx(parameters.c_days)
    assert result.best_parameters.p == pytest.approx(parameters.p)
    assert result.stability.stable is True
    assert isinstance(result.uncertainty, ObservedHessianDeltaUncertainty)


def test_optimizer_defaults_and_bounds_match_frozen_protocol() -> None:
    protocol = yaml.safe_load(Path("configs/background.yaml").read_text(encoding="utf-8"))
    etas = protocol["etas"]
    bounds = ETASParameterBounds()
    options = OptimizerOptions()
    stability = StabilityThresholds()

    assert bounds.background_rate_per_day == tuple(
        etas["parameter_bounds"]["background_rate_per_day"]
    )
    assert bounds.productivity_k == tuple(etas["parameter_bounds"]["productivity_k"])
    assert bounds.alpha == tuple(etas["parameter_bounds"]["alpha"])
    assert bounds.c_days == tuple(etas["parameter_bounds"]["c_days"])
    assert bounds.p == tuple(etas["parameter_bounds"]["p"])
    assert options.ftol == etas["optimizer_options"]["ftol"]
    assert options.gtol == etas["optimizer_options"]["gtol"]
    assert options.gradient_relative_step == etas["optimizer_options"]["gradient_relative_step"]
    assert options.maxiter == etas["optimizer_options"]["maxiter"]
    assert (
        stability.minimum_converged_starts
        == etas["numerical_stability"]["minimum_converged_starts"]
    )

    transformed_bounds = bounds.transformed()
    first = optimizer_start(
        transformed_bounds,
        root_seed=147,
        protocol_version="0.2.0",
        model_id="etas/final_validation",
        start_index=0,
    )
    again = optimizer_start(
        transformed_bounds,
        root_seed=147,
        protocol_version="0.2.0",
        model_id="etas/final_validation",
        start_index=0,
    )
    assert np.array_equal(first, again)
    distinct = optimizer_start(
        transformed_bounds,
        root_seed=147,
        protocol_version="0.2.0",
        model_id="etas/final_validation",
        start_index=1,
    )
    assert not np.array_equal(first, distinct)
    assert all(
        lower <= value <= upper
        for value, (lower, upper) in zip(first, transformed_bounds, strict=True)
    )
