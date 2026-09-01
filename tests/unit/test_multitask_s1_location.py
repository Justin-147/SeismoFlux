from __future__ import annotations

import math

import numpy as np
import pytest

from seismoflux.multitask_s1.location import (
    FROZEN_KDE_BANDWIDTHS_KM,
    CausalRecent30History,
    CausalSpatialHistory,
    EarlierInnerBoundary,
    FrozenSpatialGrid,
    fixed_origin_region_indices,
    l0_uniform_relative_mass,
    l1_regional_constant_relative_mass,
    l2_gaussian_kde_relative_mass,
    l3_b0_r30_relative_mass,
    select_kde_bandwidth_one_se,
    select_recent_alpha,
    select_regional_tau,
)

_DAY_US = 86_400_000_000


def _grid() -> FrozenSpatialGrid:
    return FrozenSpatialGrid(
        x_km=np.asarray([100.0, 200.0, 600.0, 700.0]),
        y_km=np.asarray([100.0, 200.0, 100.0, 200.0]),
        area_km2=np.asarray([1.0, 3.0, 2.0, 4.0]),
    )


def _boundary() -> EarlierInnerBoundary:
    return EarlierInnerBoundary(
        latest_inner_target_end_us=20 * _DAY_US,
        outer_evaluation_start_us=50 * _DAY_US,
    )


def _empty_recent(*, issue_time_us: int = 100 * _DAY_US) -> CausalRecent30History:
    return CausalRecent30History(
        x_km=np.asarray([], dtype=np.float64),
        y_km=np.asarray([], dtype=np.float64),
        origin_time_us=np.asarray([], dtype=np.int64),
        available_at_us=np.asarray([], dtype=np.int64),
        issue_time_us=issue_time_us,
        data_cutoff_us=issue_time_us - _DAY_US,
    )


def test_l0_uses_exact_clipped_area_and_returns_read_only_relative_mass() -> None:
    grid = _grid()

    surface = l0_uniform_relative_mass(grid)

    np.testing.assert_allclose(surface.cell_relative_mass, [0.1, 0.3, 0.2, 0.4])
    assert math.fsum(float(value) for value in surface.cell_relative_mass) == pytest.approx(1.0)
    assert surface.cell_relative_mass.flags.writeable is False
    assert surface.source_event_count == 0


def test_fixed_origin_500km_regions_use_high_side_floor_at_boundaries() -> None:
    actual = fixed_origin_region_indices(
        np.asarray([-0.001, 0.0, 499.999, 500.0]),
        np.asarray([0.0, 0.0, 500.0, -0.001]),
    )

    np.testing.assert_array_equal(
        actual,
        np.asarray([[0, -1], [0, 0], [1, 0], [-1, 1]], dtype=np.int64),
    )


def test_l1_matches_frozen_gamma_poisson_formula_and_conserves_mass() -> None:
    grid = _grid()
    history = CausalSpatialHistory(
        x_km=np.asarray([100.0, 150.0, 250.0, 650.0]),
        y_km=np.asarray([100.0, 150.0, 250.0, 150.0]),
    )

    surface = l1_regional_constant_relative_mass(
        history,
        grid,
        exposure_years=2.0,
        tau_years=1.0,
    )

    national_rate = 4.0 / (10.0 * 2.0)
    region0_rate = (3.0 + 1.0 * 4.0 * national_rate) / ((2.0 + 1.0) * 4.0)
    region1_rate = (1.0 + 1.0 * 6.0 * national_rate) / ((2.0 + 1.0) * 6.0)
    expected_raw = np.asarray(
        [region0_rate * 1.0, region0_rate * 3.0, region1_rate * 2.0, region1_rate * 4.0]
    )
    expected = expected_raw / np.sum(expected_raw)
    np.testing.assert_allclose(surface.cell_relative_mass, expected, rtol=1.0e-15)
    assert math.fsum(float(value) for value in surface.cell_relative_mass) == pytest.approx(1.0)
    repeated = l1_regional_constant_relative_mass(
        history,
        grid,
        exposure_years=2.0,
        tau_years=1.0,
    )
    np.testing.assert_array_equal(surface.cell_relative_mass, repeated.cell_relative_mass)


def test_l1_fails_closed_when_causal_history_is_outside_frozen_grid_regions() -> None:
    history = CausalSpatialHistory(np.asarray([1_100.0]), np.asarray([100.0]))

    with pytest.raises(ValueError, match="outside grid regions"):
        l1_regional_constant_relative_mass(
            history,
            _grid(),
            exposure_years=10.0,
            tau_years=5.0,
        )


def test_l2_kde_is_deterministic_area_integrated_and_near_source_is_stronger() -> None:
    grid = FrozenSpatialGrid(
        x_km=np.asarray([0.0, 100.0, 200.0]),
        y_km=np.asarray([0.0, 0.0, 0.0]),
        area_km2=np.asarray([2.0, 2.0, 2.0]),
    )
    history = CausalSpatialHistory(np.asarray([0.0]), np.asarray([0.0]))

    first = l2_gaussian_kde_relative_mass(history, grid, bandwidth_km=75.0)
    second = l2_gaussian_kde_relative_mass(history, grid, bandwidth_km=75.0)

    assert first.cell_relative_mass[0] > first.cell_relative_mass[1]
    assert first.cell_relative_mass[1] > first.cell_relative_mass[2]
    assert math.fsum(float(value) for value in first.cell_relative_mass) == pytest.approx(1.0)
    np.testing.assert_array_equal(first.cell_relative_mass, second.cell_relative_mass)
    with pytest.raises(ValueError, match="one of 75"):
        l2_gaussian_kde_relative_mass(history, grid, bandwidth_km=125.0)


def test_recent_window_separates_issue_and_cutoff_with_open_lower_closed_upper() -> None:
    issue = 100 * _DAY_US
    cutoff = issue - _DAY_US
    lower = issue - 30 * _DAY_US

    recent = CausalRecent30History(
        x_km=np.asarray([0.0, 1.0]),
        y_km=np.asarray([0.0, 1.0]),
        origin_time_us=np.asarray([lower + 1, cutoff], dtype=np.int64),
        available_at_us=np.asarray([lower + 1, cutoff], dtype=np.int64),
        issue_time_us=issue,
        data_cutoff_us=cutoff,
    )

    assert recent.event_count == 2
    with pytest.raises(ValueError, match=r"\(T-30d, T-24h\]"):
        CausalRecent30History(
            x_km=np.asarray([0.0]),
            y_km=np.asarray([0.0]),
            origin_time_us=np.asarray([lower], dtype=np.int64),
            available_at_us=np.asarray([lower], dtype=np.int64),
            issue_time_us=issue,
            data_cutoff_us=cutoff,
        )
    with pytest.raises(ValueError, match=r"\(T-30d, T-24h\]"):
        CausalRecent30History(
            x_km=np.asarray([0.0]),
            y_km=np.asarray([0.0]),
            origin_time_us=np.asarray([cutoff + 1], dtype=np.int64),
            available_at_us=np.asarray([cutoff], dtype=np.int64),
            issue_time_us=issue,
            data_cutoff_us=cutoff,
        )
    with pytest.raises(ValueError, match="available by"):
        CausalRecent30History(
            x_km=np.asarray([0.0]),
            y_km=np.asarray([0.0]),
            origin_time_us=np.asarray([cutoff], dtype=np.int64),
            available_at_us=np.asarray([cutoff + 1], dtype=np.int64),
            issue_time_us=issue,
            data_cutoff_us=cutoff,
        )
    with pytest.raises(ValueError, match="minus 24 hours"):
        CausalRecent30History(
            x_km=np.asarray([], dtype=np.float64),
            y_km=np.asarray([], dtype=np.float64),
            origin_time_us=np.asarray([], dtype=np.int64),
            available_at_us=np.asarray([], dtype=np.int64),
            issue_time_us=issue,
            data_cutoff_us=issue,
        )


def test_l3_empty_recent_exactly_falls_back_and_nonempty_uses_frozen_convex_mix() -> None:
    grid = FrozenSpatialGrid(
        x_km=np.asarray([0.0, 100.0, 200.0]),
        y_km=np.asarray([0.0, 0.0, 0.0]),
        area_km2=np.ones(3),
    )
    long_history = CausalSpatialHistory(np.asarray([0.0]), np.asarray([0.0]))
    long_surface = l2_gaussian_kde_relative_mass(long_history, grid, bandwidth_km=75.0)

    fallback = l3_b0_r30_relative_mass(
        long_history,
        _empty_recent(),
        grid,
        bandwidth_km=75.0,
        alpha=0.75,
    )
    np.testing.assert_array_equal(fallback.cell_relative_mass, long_surface.cell_relative_mass)
    assert fallback.recent_fallback_to_long is True

    issue = 100 * _DAY_US
    recent_history = CausalRecent30History(
        x_km=np.asarray([200.0]),
        y_km=np.asarray([0.0]),
        origin_time_us=np.asarray([issue - 2 * _DAY_US], dtype=np.int64),
        available_at_us=np.asarray([issue - 2 * _DAY_US], dtype=np.int64),
        issue_time_us=issue,
        data_cutoff_us=issue - _DAY_US,
    )
    recent_surface = l2_gaussian_kde_relative_mass(
        recent_history.as_spatial_history(),
        grid,
        bandwidth_km=75.0,
    )
    mixed = l3_b0_r30_relative_mass(
        long_history,
        recent_history,
        grid,
        bandwidth_km=75.0,
        alpha=0.25,
    )
    np.testing.assert_allclose(
        mixed.cell_relative_mass,
        0.75 * long_surface.cell_relative_mass + 0.25 * recent_surface.cell_relative_mass,
        rtol=1.0e-15,
    )
    assert mixed.recent_fallback_to_long is False
    assert math.fsum(float(value) for value in mixed.cell_relative_mass) == pytest.approx(1.0)


def test_selection_boundary_enforces_frozen_30_day_label_embargo() -> None:
    with pytest.raises(ValueError, match="at least 30 days"):
        EarlierInnerBoundary(
            latest_inner_target_end_us=0,
            outer_evaluation_start_us=29 * _DAY_US,
        )
    with pytest.raises(ValueError, match="at least 30 days"):
        EarlierInnerBoundary(
            latest_inner_target_end_us=0,
            outer_evaluation_start_us=30 * _DAY_US - 1,
        )
    exact = EarlierInnerBoundary(
        latest_inner_target_end_us=0,
        outer_evaluation_start_us=30 * _DAY_US,
    )
    beyond = EarlierInnerBoundary(
        latest_inner_target_end_us=0,
        outer_evaluation_start_us=30 * _DAY_US + 1,
    )
    assert exact.outer_evaluation_start_us - exact.latest_inner_target_end_us == 30 * _DAY_US
    assert beyond.outer_evaluation_start_us - beyond.latest_inner_target_end_us > 30 * _DAY_US


def test_l1_and_l3_selection_use_frozen_tie_and_sparse_target_rules() -> None:
    boundary = _boundary()
    assert (
        select_regional_tau(
            {1.0: 1.0, 5.0: 1.0 - 5.0e-13, 10.0: 1.0 - 9.0e-13},
            boundary=boundary,
        )
        == 10.0
    )
    scores = {0.0: 0.0, 0.25: 2.0, 0.5: 2.0 - 5.0e-13, 0.75: 1.0}
    assert select_recent_alpha(scores, inner_target_count=10, boundary=boundary) == 0.25
    assert select_recent_alpha(scores, inner_target_count=9, boundary=boundary) == 0.0
    with pytest.raises(ValueError, match="exactly the frozen"):
        select_regional_tau({1.0: 1.0, 5.0: 1.0}, boundary=boundary)


def test_kde_selection_reuses_paired_one_se_and_takes_largest_eligible_bandwidth() -> None:
    scores = {
        75.0: (1.0, 1.0, 1.0, 1.0),
        100.0: (0.8, 1.1, 0.8, 1.1),
        150.0: (0.4, 0.4, 0.4, 0.4),
        200.0: (0.2, 0.2, 0.2, 0.2),
        300.0: (0.7, 1.1, 0.7, 1.1),
    }

    selection = select_kde_bandwidth_one_se(scores, boundary=_boundary())

    assert selection.best_mean_bandwidth_km == 75.0
    assert selection.selected_bandwidth_km == 300.0
    audits = {item.bandwidth_km: item for item in selection.candidates}
    expected_se = np.std(np.asarray([-0.3, 0.1, -0.3, 0.1]), ddof=1) / math.sqrt(4.0)
    assert audits[300.0].paired_standard_error == pytest.approx(expected_se)
    assert audits[300.0].eligible is True
    assert audits[200.0].eligible is False


def test_kde_selection_fails_closed_without_two_complete_earlier_inner_folds() -> None:
    one_fold = {bandwidth: (1.0,) for bandwidth in FROZEN_KDE_BANDWIDTHS_KM}
    with pytest.raises(ValueError, match="at least two"):
        select_kde_bandwidth_one_se(one_fold, boundary=_boundary())

    incomplete = {
        bandwidth: (1.0, 1.0) for bandwidth in FROZEN_KDE_BANDWIDTHS_KM if bandwidth != 300.0
    }
    with pytest.raises(ValueError, match="exactly the five"):
        select_kde_bandwidth_one_se(incomplete, boundary=_boundary())
