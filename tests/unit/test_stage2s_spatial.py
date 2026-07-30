"""Pure-synthetic tests for the frozen Stage 2S spatial numerics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from seismoflux.background.grid import GridConvergenceError
from seismoflux.background.poisson import GaussianMixtureFamily
from seismoflux.stage2s.contracts import (
    PRIMARY_ALARM_BUDGET_KM2,
    EvidenceInsufficientError,
    FitEventOrder,
    GridConvergence,
    NormalizedSpatialDensity,
    SpatialGrid,
    SpatialQuadratureFamily,
)
from seismoflux.stage2s.spatial import (
    aggregate_operational_mass_to_25km,
    build_normalized_kde,
    build_recent_component,
    build_stage2s_models,
    estimate_shared_m5_6_rate,
    event_cell_index_25km,
    fit_alpha,
    fit_alpha_log_density,
    mix_density,
    recompute_pair_compensator,
    select_alarm_prefix,
    stable_mixture_log_density,
)


def _square_grid(
    cell_size_km: float,
    *,
    extent_km: float,
    areas_by_parent_25km: tuple[float, float, float, float] | None = None,
) -> SpatialGrid:
    cells_per_side = max(1, int(extent_km / cell_size_km))
    rows: list[int] = []
    columns: list[int] = []
    xy: list[tuple[float, float]] = []
    areas: list[float] = []
    cell_ids: list[str] = []
    for row in range(cells_per_side):
        for column in range(cells_per_side):
            rows.append(row)
            columns.append(column)
            clipped_side = min(cell_size_km, extent_km)
            xy.append(
                (
                    column * cell_size_km + clipped_side / 2.0,
                    row * cell_size_km + clipped_side / 2.0,
                )
            )
            cell_ids.append(f"g{cell_size_km:g}-r{row}-c{column}")
            if areas_by_parent_25km is None:
                areas.append(clipped_side * clipped_side)
            elif cell_size_km == 50.0:
                areas.append(math.fsum(areas_by_parent_25km))
            elif cell_size_km == 25.0:
                areas.append(areas_by_parent_25km[row * 2 + column])
            else:
                parent_area = areas_by_parent_25km[(row // 2) * 2 + column // 2]
                areas.append(parent_area / 4.0)
    return SpatialGrid(
        grid_id=f"synthetic-{cell_size_km:g}",
        cell_size_km=cell_size_km,
        cell_ids=tuple(cell_ids),
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        query_xy_km=np.asarray(xy, dtype=np.float64),
        clipped_area_km2=np.asarray(areas, dtype=np.float64),
    )


def _family(
    *,
    extent_km: float = 25.0,
    areas_by_parent_25km: tuple[float, float, float, float] | None = None,
) -> SpatialQuadratureFamily:
    if areas_by_parent_25km is not None:
        extent_km = 50.0
    return SpatialQuadratureFamily(
        grids=(
            _square_grid(
                50.0,
                extent_km=extent_km,
                areas_by_parent_25km=areas_by_parent_25km,
            ),
            _square_grid(
                25.0,
                extent_km=extent_km,
                areas_by_parent_25km=areas_by_parent_25km,
            ),
            _square_grid(
                12.5,
                extent_km=extent_km,
                areas_by_parent_25km=areas_by_parent_25km,
            ),
        )
    )


def _s0(
    family: SpatialQuadratureFamily | None = None,
) -> NormalizedSpatialDensity:
    selected_family = _family() if family is None else family
    return build_normalized_kde(
        np.asarray([[12.5, 12.5], [10.0, 15.0]], dtype=np.float64),
        selected_family,
        model_id="S0",
    )


def _manual_s0(
    *,
    areas_by_parent_25km: tuple[float, float, float, float],
    mass_25km: tuple[float, float, float, float],
) -> NormalizedSpatialDensity:
    family = _family(areas_by_parent_25km=areas_by_parent_25km)
    fine_mass: list[float] = []
    for row in range(4):
        for column in range(4):
            fine_mass.append(mass_25km[(row // 2) * 2 + column // 2] / 4.0)
    kernel = GaussianMixtureFamily(
        np.asarray([25.0], dtype=np.float64),
        np.asarray([25.0], dtype=np.float64),
    )
    return NormalizedSpatialDensity(
        model_id="S0",
        grid_family=family,
        mass_12_5km=np.asarray(fine_mass, dtype=np.float64),
        mass_25km=np.asarray(mass_25km, dtype=np.float64),
        direct_mass_25km=np.asarray(mass_25km, dtype=np.float64),
        direct_mass_50km=np.asarray([1.0], dtype=np.float64),
        convergence=GridConvergence(
            primary_relative_count_difference=0.0,
            primary_density_l1=0.0,
            diagnostic_relative_count_difference=0.0,
            diagnostic_density_l1=0.0,
            passed=True,
        ),
        normalization_mass=1.0,
        _kernel=kernel,
    )


def test_fixed_mean_kde_uses_fsum_normalization_and_continuous_density() -> None:
    family = _family()
    source_xy = np.asarray([[12.5, 12.5]], dtype=np.float64)
    model = build_normalized_kde(source_xy, family, model_id="S0")
    fine = family.at(12.5)
    raw_masses = [
        math.exp(-((float(x) - 12.5) ** 2 + (float(y) - 12.5) ** 2) / (2.0 * 75.0**2))
        / (2.0 * math.pi * 75.0**2)
        * float(area)
        for x, y, area in zip(
            fine.quadrature.x_km,
            fine.quadrature.y_km,
            fine.quadrature.area_km2,
            strict=True,
        )
    ]
    expected_normalization = math.fsum(raw_masses)
    assert model.normalization_mass == pytest.approx(expected_normalization, rel=0.0, abs=1e-16)
    expected_density = 1.0 / (2.0 * math.pi * 75.0**2) / expected_normalization
    density = model.density(np.asarray([12.5]), np.asarray([12.5]))
    log_density = model.log_density(np.asarray([12.5]), np.asarray([12.5]))
    assert density[0] == pytest.approx(expected_density, rel=1e-14)
    assert log_density[0] == pytest.approx(math.log(expected_density), rel=1e-14)
    assert math.fsum(float(value) for value in model.mass_12_5km) == pytest.approx(
        1.0,
        abs=1e-15,
    )
    assert math.fsum(float(value) for value in model.mass_25km) == pytest.approx(
        1.0,
        abs=1e-15,
    )
    assert model.convergence.passed is True
    assert not model.mass_12_5km.flags.writeable
    assert not model.mass_25km.flags.writeable
    assert not density.flags.writeable
    assert not log_density.flags.writeable
    with pytest.raises(ValueError, match="frozen"):
        build_normalized_kde(source_xy, family, model_id="S0", bandwidth_km=60.0)


def test_empty_recent_windows_are_exact_s0_and_mixtures_remain_exact() -> None:
    family = _family()
    s0 = _s0(family)
    empty = np.empty((0, 2), dtype=np.float64)
    recent = build_recent_component(
        empty,
        family,
        component_id="R",
        empty_fallback_s0=s0,
    )
    preceding = build_recent_component(
        empty,
        family,
        component_id="RP",
        empty_fallback_s0=s0,
    )
    assert recent is s0
    assert preceding is s0
    models = build_stage2s_models(
        s0,
        recent,
        preceding,
        alpha_r=0.73,
        alpha_p=0.31,
    )
    query_x = np.asarray([3.0, 12.5, 24.0], dtype=np.float64)
    query_y = np.asarray([4.0, 12.5, 23.0], dtype=np.float64)
    assert np.array_equal(models.s1.mass_12_5km, s0.mass_12_5km)
    assert np.array_equal(models.sp.mass_25km, s0.mass_25km)
    assert np.array_equal(models.s1.density(query_x, query_y), s0.density(query_x, query_y))
    assert np.array_equal(
        models.sp.log_density(query_x, query_y),
        s0.log_density(query_x, query_y),
    )


def test_nonempty_mixture_uses_stable_continuous_logaddexp() -> None:
    family = _family()
    s0 = _s0(family)
    recent = build_recent_component(
        np.asarray([[5.0, 5.0]], dtype=np.float64),
        family,
        component_id="R",
        empty_fallback_s0=s0,
    )
    mixed = mix_density(s0, recent, 0.4, model_id="S1")
    x = np.asarray([1.0, 12.5, 24.0], dtype=np.float64)
    y = np.asarray([2.0, 12.5, 23.0], dtype=np.float64)
    expected = np.logaddexp(
        math.log(0.6) + s0.log_density(x, y),
        math.log(0.4) + recent.log_density(x, y),
    )
    assert np.array_equal(mixed.log_density(x, y), expected)
    assert np.allclose(
        mixed.density(x, y),
        0.6 * s0.density(x, y) + 0.4 * recent.density(x, y),
        rtol=0.0,
        atol=0.0,
    )


def test_operational_mass_aggregation_uses_aligned_floor_parent_rule() -> None:
    family = _family(extent_km=50.0)
    values = np.arange(1.0, 17.0, dtype=np.float64)
    values /= math.fsum(float(value) for value in values)
    aggregated = aggregate_operational_mass_to_25km(values, family)
    expected = np.asarray([14.0, 22.0, 46.0, 54.0], dtype=np.float64) / 136.0
    assert np.allclose(aggregated, expected, rtol=0.0, atol=1.0e-16)
    assert not aggregated.flags.writeable


def test_alpha_solver_precedence_boundaries_and_fixed_bisection() -> None:
    order = FitEventOrder(
        origin_time_ns=np.asarray([2, 1], dtype=np.int64),
        event_ids=("z", "a"),
    )
    flat = fit_alpha(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]), order)
    assert (flat.alpha, flat.solver_case, flat.iterations) == (0.0, "flat", 0)
    at_zero = fit_alpha(np.asarray([1.0, 1.0]), np.asarray([0.5, 0.5]), order)
    assert (at_zero.alpha, at_zero.solver_case) == (0.0, "derivative_at_zero")
    at_one = fit_alpha(np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0]), order)
    assert (at_one.alpha, at_one.solver_case) == (1.0, "derivative_at_one")
    interior = fit_alpha(np.asarray([1.0, 1.0]), np.asarray([2.0, 0.5]), order)
    assert interior.solver_case == "bisection"
    assert interior.iterations == 64
    assert interior.alpha == pytest.approx(0.5, abs=2.0e-16)
    assert interior.derivative_at_zero.comparison == "greater_than_positive_tolerance"
    assert interior.derivative_at_one.comparison == "less_than_negative_tolerance"


def test_log_alpha_solver_handles_extreme_ratios_and_endpoint_precedence() -> None:
    order = FitEventOrder(
        origin_time_ns=np.asarray([1, 2], dtype=np.int64),
        event_ids=("positive", "negative"),
    )
    mixed = fit_alpha_log_density(
        np.asarray([0.0, 0.0]),
        np.asarray([1_000.0, -1_000.0]),
        order,
    )
    assert mixed.solver_case == "bisection"
    assert mixed.alpha == pytest.approx(0.5, abs=2.0e-16)
    assert mixed.derivative_at_zero.sign == 1
    assert mixed.derivative_at_zero.finite_float_value is None
    assert mixed.derivative_at_one.sign == -1
    assert mixed.derivative_at_one.finite_float_value is None

    zero_first = fit_alpha_log_density(
        np.asarray([0.0, 0.0]),
        np.asarray([math.log(2.0), -1_000.0]),
        order,
    )
    assert (zero_first.alpha, zero_first.solver_case) == (
        0.0,
        "derivative_at_zero",
    )
    assert zero_first.derivative_at_zero.comparison == "within_tolerance"

    safe_q0 = np.asarray([1.0, 1.0])
    safe_qx = np.asarray([2.0, 0.5])
    density_fit = fit_alpha(safe_q0, safe_qx, order)
    log_fit = fit_alpha_log_density(np.log(safe_q0), np.log(safe_qx), order)
    assert (density_fit.alpha, density_fit.solver_case) == (
        log_fit.alpha,
        log_fit.solver_case,
    )


def test_log_alpha_formal_path_survives_thousands_of_km_density_underflow() -> None:
    family = _family()
    s0 = _s0(family)
    recent = build_recent_component(
        np.asarray([[24.0, 24.0]], dtype=np.float64),
        family,
        component_id="R",
        empty_fallback_s0=s0,
    )
    target_x = np.asarray([5_000.0, -5_000.0], dtype=np.float64)
    target_y = np.asarray([5_000.0, -5_000.0], dtype=np.float64)
    assert np.array_equal(s0.density(target_x, target_y), np.zeros(2))
    assert np.array_equal(recent.density(target_x, target_y), np.zeros(2))
    log_q0 = s0.log_density(target_x, target_y)
    log_qr = recent.log_density(target_x, target_y)
    assert np.isfinite(log_q0).all()
    assert np.isfinite(log_qr).all()
    order = FitEventOrder(
        origin_time_ns=np.asarray([1, 2], dtype=np.int64),
        event_ids=("far-positive", "far-negative"),
    )
    fitted = fit_alpha_log_density(log_q0, log_qr, order)
    assert fitted.solver_case == "bisection"
    assert 0.0 < fitted.alpha < 1.0


def test_alpha_solver_fails_closed_before_any_boundary_case() -> None:
    empty_order = FitEventOrder(origin_time_ns=np.asarray([], dtype=np.int64), event_ids=())
    with pytest.raises(EvidenceInsufficientError):
        fit_alpha(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), empty_order)
    order = FitEventOrder(origin_time_ns=np.asarray([0], dtype=np.int64), event_ids=("e",))
    with pytest.raises(ValueError, match="positive"):
        fit_alpha(np.asarray([1.0]), np.asarray([0.0]), order)
    with pytest.raises(ValueError, match="finite"):
        fit_alpha(np.asarray([1.0]), np.asarray([np.nan]), order)


def test_stable_mixture_log_density_uses_exact_endpoints() -> None:
    log_q0 = np.asarray([-1_000.0, -2.0], dtype=np.float64)
    log_qx = np.asarray([-999.0, -3.0], dtype=np.float64)
    assert np.array_equal(stable_mixture_log_density(log_q0, log_qx, 0.0), log_q0)
    assert np.array_equal(stable_mixture_log_density(log_q0, log_qx, 1.0), log_qx)
    expected = np.logaddexp(math.log(0.75) + log_q0, math.log(0.25) + log_qx)
    assert np.array_equal(stable_mixture_log_density(log_q0, log_qx, 0.25), expected)


def test_shared_rate_and_compensators_are_spatial_model_independent() -> None:
    rate = estimate_shared_m5_6_rate(
        ("m5-a", "m5-b", "m5-c"),
        total_exposure_days=21.0,
    )
    assert rate.rate_per_day == 1.0 / 7.0
    assert rate.event_count == 3
    comparison = recompute_pair_compensator(
        rate.rate_per_day,
        7.0,
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([1.0 + 5.0e-13, 1.0 + 5.0e-13], dtype=np.float64),
    )
    assert comparison.compensator_a == 2.0
    assert comparison.compensator_b != comparison.compensator_a
    assert comparison.difference == comparison.compensator_a - comparison.compensator_b
    assert abs(comparison.difference) <= 1.0e-10
    with pytest.raises(ValueError, match="exactly once"):
        estimate_shared_m5_6_rate(("duplicate", "duplicate"), total_exposure_days=14.0)
    with pytest.raises(EvidenceInsufficientError):
        estimate_shared_m5_6_rate((), total_exposure_days=14.0)


def test_alarm_ranks_mass_over_exact_area_then_fixed_ties() -> None:
    areas = (400_000.0, 200_000.0, 400_000.0, 300_000.0)
    density_ranked = _manual_s0(
        areas_by_parent_25km=areas,
        mass_25km=(0.40, 0.25, 0.25, 0.10),
    )
    alarm = select_alarm_prefix(density_ranked)
    grid = density_ranked.grid_family.at(25.0)
    assert alarm.selected_cell_ids == (grid.cell_ids[1], grid.cell_ids[0])
    assert alarm.actual_area_km2 == PRIMARY_ALARM_BUDGET_KM2
    assert not alarm.selected_indices.flags.writeable
    assert len(alarm.ranking_sha256) == 64

    total_area = math.fsum(areas)
    tied = _manual_s0(
        areas_by_parent_25km=areas,
        mass_25km=(
            areas[0] / total_area,
            areas[1] / total_area,
            areas[2] / total_area,
            areas[3] / total_area,
        ),
    )
    tie_alarm = select_alarm_prefix(tied)
    tie_grid = tied.grid_family.at(25.0)
    assert tie_alarm.selected_cell_ids == (tie_grid.cell_ids[0], tie_grid.cell_ids[1])


def test_alarm_is_complete_prefix_and_never_skips_an_oversized_next_cell() -> None:
    areas = (400_000.0, 300_000.0, 200_000.0, 400_000.0)
    total_area = math.fsum(areas)
    tied = _manual_s0(
        areas_by_parent_25km=areas,
        mass_25km=(
            areas[0] / total_area,
            areas[1] / total_area,
            areas[2] / total_area,
            areas[3] / total_area,
        ),
    )
    alarm = select_alarm_prefix(tied)
    grid = tied.grid_family.at(25.0)
    assert alarm.selected_cell_ids == (grid.cell_ids[0],)
    assert alarm.actual_area_km2 == 400_000.0
    assert grid.cell_ids[2] not in alarm.selected_cell_ids


def test_event_cell_index_uses_half_open_floor_without_epsilon() -> None:
    assert event_cell_index_25km(0.0, 0.0) == (0, 0)
    assert event_cell_index_25km(24_999.999999, 24_999.999999) == (0, 0)
    assert event_cell_index_25km(25_000.0, 25_000.0) == (1, 1)
    assert event_cell_index_25km(-0.000001, -0.000001) == (-1, -1)


def test_synthetic_convergence_failure_stops_the_model() -> None:
    distorted = _family(
        areas_by_parent_25km=(1.0, 1.0, 1.0, 10_000.0),
    )
    with pytest.raises(GridConvergenceError):
        build_normalized_kde(
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            distorted,
            model_id="S0",
        )


def test_stage2s_core_has_no_forbidden_import_or_real_data_access() -> None:
    root = Path(__file__).resolve().parents[2]
    source = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "src/seismoflux/stage2s/contracts.py",
            "src/seismoflux/stage2s/spatial.py",
        )
    )
    assert "seismoflux.anomaly_increment" not in source
    assert "data/processed" not in source
