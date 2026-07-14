from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import box

from seismoflux.background.adapters import (
    build_analytic_simulation_expectations,
    build_etas_model_spec,
    build_etas_parameter_bounds,
    build_optimizer_options,
    build_stability_thresholds,
    etas_variant_id,
    point_area_quadrature_from_grid,
)
from seismoflux.background.config import load_background_protocol
from seismoflux.background.grid import build_clipped_grid

CONFIG = Path("configs/background.yaml")


def test_protocol_adapters_match_every_frozen_etas_numerical_setting() -> None:
    config = load_background_protocol(CONFIG)
    spec = build_etas_model_spec(config, selected_mc=3.5, aki_b_value=0.9)
    bounds = build_etas_parameter_bounds(config)
    options = build_optimizer_options(config)
    thresholds = build_stability_thresholds(config)

    assert spec.mc == 3.5
    assert spec.beta == pytest.approx(0.9 * 2.302585092994046)
    assert spec.d_km2 == 25.0
    assert spec.spatial_cutoff_km == 300.0
    assert spec.history_parent_cutoff_days == 3650.0
    assert spec.branching_ratio_maximum == 0.95
    assert bounds.background_rate_per_day == (0.01, 10.0)
    assert bounds.p == (1.01, 2.5)
    assert options.maxiter == 500
    assert options.maxfun == 100_000
    assert options.gradient_relative_step == 1.0e-6
    assert thresholds.minimum_converged_starts == 4
    assert thresholds.hessian_condition_number_maximum == 1.0e10


def test_analytic_expectations_and_variant_identity_are_protocol_derived() -> None:
    config = load_background_protocol(CONFIG)
    expected = build_analytic_simulation_expectations(config)

    assert expected.expected_generic_branching_ratio == 0.3928055160151634
    assert expected.minimum_pooled_direct_offspring == 3500
    assert etas_variant_id(config, selected_bandwidth_km=300.0) == (
        "etas/d25_q1.5_gamma1_cut300_kde_bw300"
    )
    with pytest.raises(ValueError, match="frozen candidate"):
        etas_variant_id(config, selected_bandwidth_km=125.0)


def test_point_area_quadrature_preserves_exact_clipped_area_and_units() -> None:
    grid = build_clipped_grid(box(0.0, 0.0, 30_000.0, 20_000.0), cell_size_km=25.0)
    quadrature = point_area_quadrature_from_grid(grid)

    assert len(quadrature.points) == len(grid.cells)
    assert sum(point.area_km2 for point in quadrature.points) == pytest.approx(600.0)
    assert quadrature.points[0].x_km == pytest.approx(
        grid.cells[0].representative_point.x / 1_000.0
    )


def test_model_spec_rejects_unregistered_spatial_sensitivity() -> None:
    config = load_background_protocol(CONFIG)
    with pytest.raises(ValueError, match="frozen main or sensitivity"):
        build_etas_model_spec(
            config,
            selected_mc=3.5,
            aki_b_value=1.0,
            d_km2=50.0,
        )
