"""Strict adapters from the frozen protocol to numerical ETAS primitives."""

from __future__ import annotations

import math

from seismoflux.background.config import BackgroundConfig
from seismoflux.background.etas_fit import (
    ETASModelSpec,
    ETASParameterBounds,
    OptimizerOptions,
    PointAreaQuadrature,
    QuadraturePoint,
    StabilityThresholds,
)
from seismoflux.background.grid import EqualAreaGrid
from seismoflux.background.regression import AnalyticSimulationExpectations


def build_etas_model_spec(
    config: BackgroundConfig,
    *,
    selected_mc: float,
    aki_b_value: float,
    d_km2: float | None = None,
) -> ETASModelSpec:
    """Build one snapshot's ETAS specification without permitting protocol drift."""

    magnitude = config.etas.magnitude_model
    spatial = config.etas.spatial_kernel
    temporal = config.etas.temporal_kernel
    branching = config.etas.branching_ratio
    selected_d = spatial.d_km2 if d_km2 is None else float(d_km2)
    if selected_d not in spatial.sensitivity_d_km2:
        raise ValueError("ETAS D must use a frozen main or sensitivity value")
    beta = float(aki_b_value) * math.log(10.0)
    return ETASModelSpec(
        mc=float(selected_mc),
        beta=beta,
        maximum_magnitude=magnitude.upper_magnitude,
        d_km2=selected_d,
        q=spatial.q,
        gamma=spatial.gamma,
        spatial_cutoff_km=spatial.support_radius_km,
        history_parent_cutoff_days=temporal.history_parent_cutoff_days,
        branching_ratio_maximum=branching.maximum,
        alpha_beta_equality_tolerance=branching.equality_tolerance,
    )


def build_etas_parameter_bounds(config: BackgroundConfig) -> ETASParameterBounds:
    values = config.etas.parameter_bounds
    return ETASParameterBounds(
        background_rate_per_day=values.background_rate_per_day,
        productivity_k=values.productivity_k,
        alpha=values.alpha,
        c_days=values.c_days,
        p=values.p,
    )


def build_optimizer_options(config: BackgroundConfig) -> OptimizerOptions:
    values = config.etas.optimizer_options
    if config.etas.maximum_iterations != values.maxiter:
        raise ValueError("ETAS optimizer iteration limits disagree")
    return OptimizerOptions(
        ftol=values.ftol,
        gtol=values.gtol,
        maxiter=values.maxiter,
        maxfun=values.maxfun,
        maxls=values.maxls,
        gradient_relative_step=values.gradient_relative_step,
    )


def build_stability_thresholds(config: BackgroundConfig) -> StabilityThresholds:
    values = config.etas.numerical_stability
    return StabilityThresholds(
        minimum_converged_starts=values.minimum_converged_starts,
        gradient_infinity_norm_maximum=values.gradient_infinity_norm_maximum,
        best_three_relative_objective_range_maximum=(
            values.best_three_relative_objective_range_maximum
        ),
        transformed_parameter_maximum_range=values.transformed_parameter_maximum_range,
        hessian_minimum_eigenvalue=values.hessian_minimum_eigenvalue,
        hessian_condition_number_maximum=values.hessian_condition_number_maximum,
        hessian_relative_step=values.hessian_relative_step,
    )


def build_analytic_simulation_expectations(
    config: BackgroundConfig,
) -> AnalyticSimulationExpectations:
    values = config.numerical_regression.analytic_simulation
    return AnalyticSimulationExpectations(
        expected_generic_branching_ratio=values.expected_generic_branching_ratio,
        expected_root_direct_offspring_mean=values.expected_root_direct_offspring_mean,
        expected_root_total_descendants_mean=values.expected_root_total_descendants_mean,
        expected_root_total_descendants_variance=values.expected_root_total_descendants_variance,
        expected_root_zero_descendant_probability=(
            values.expected_root_zero_descendant_probability
        ),
        descendant_mean_absolute_tolerance=values.descendant_mean_absolute_tolerance,
        descendant_variance_relative_tolerance=values.descendant_variance_relative_tolerance,
        zero_descendant_probability_absolute_tolerance=(
            values.zero_descendant_probability_absolute_tolerance
        ),
        direct_offspring_pit_ks_maximum=values.direct_offspring_pit_ks_maximum,
        minimum_pooled_direct_offspring=values.minimum_pooled_direct_offspring,
    )


def point_area_quadrature_from_grid(grid: EqualAreaGrid) -> PointAreaQuadrature:
    """Adapt the exact clipped representative-point grid from metres to kilometres."""

    return PointAreaQuadrature.from_points(
        QuadraturePoint(
            x_km=float(cell.representative_point.x) / 1_000.0,
            y_km=float(cell.representative_point.y) / 1_000.0,
            area_km2=cell.clipped_area_m2 / 1_000_000.0,
        )
        for cell in grid.cells
    )


def etas_variant_id(config: BackgroundConfig, *, selected_bandwidth_km: float) -> str:
    spatial = config.etas.spatial_kernel
    bandwidth = float(selected_bandwidth_km)
    if bandwidth not in config.spatial_poisson.bandwidth_candidates_km:
        raise ValueError("ETAS background bandwidth must be a frozen candidate")
    return (
        f"etas/d{spatial.d_km2:g}_q{spatial.q:g}_gamma{spatial.gamma:g}"
        f"_cut{spatial.support_radius_km:g}_kde_bw{bandwidth:g}"
    )


__all__ = [
    "build_analytic_simulation_expectations",
    "build_etas_model_spec",
    "build_etas_parameter_bounds",
    "build_optimizer_options",
    "build_stability_thresholds",
    "etas_variant_id",
    "point_area_quadrature_from_grid",
]
