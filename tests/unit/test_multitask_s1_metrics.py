from __future__ import annotations

import math

import numpy as np
import pytest

from seismoflux.multitask_s1.metrics import (
    FROZEN_D1_AREA_BUDGETS_KM2,
    JOINT_M5_FACTORIZATION,
    compare_m0_m3_on_common_m5_support,
    score_location_events,
    score_m0_conditional_m5_events,
    score_m0_unique_events,
    score_m3_conditional_m5_events,
    score_minimal_joint_m5,
    score_nb2_exposure,
    score_poisson_exposure,
)
from seismoflux.multitask_s1.time_magnitude import (
    NB2DispersionQualification,
    fit_m0_gr_global,
    fit_m3_gr_long_m5,
    fit_nb2_dispersion,
    nb2_at_least_one_probability,
    nb2_logpmf,
    poisson_logpmf,
)
from seismoflux.stage2s.contracts import SpatialGrid


def _grid() -> SpatialGrid:
    return SpatialGrid(
        grid_id="synthetic-s1-metrics-grid",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3"),
        rows=np.asarray([0, 0, 0, 0], dtype=np.int64),
        columns=np.asarray([0, 1, 2, 3], dtype=np.int64),
        query_xy_km=np.asarray(
            [[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0]],
            dtype=np.float64,
        ),
        clipped_area_km2=np.asarray(
            [200_000.0, 250_000.0, 300_000.0, 300_000.0],
            dtype=np.float64,
        ),
    )


def _mass() -> np.ndarray:
    return np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)


def test_location_uses_exact_cell_log_density_and_d1_complete_cell_prefixes() -> None:
    result = score_location_events(
        _mass(),
        _grid(),
        event_ids=("e0", "e1", "e2", "e3"),
        event_cell_indices=(0, 1, 2, 3),
        episode_ids=("a", "a", "b", "b"),
        episode_member_counts=(2, 2, 2, 2),
        is_episode_anchor=(True, False, True, False),
    )

    assert [item.log_density_per_km2 for item in result.event_log_densities] == pytest.approx(
        [
            math.log(0.4 / 200_000.0),
            math.log(0.3 / 250_000.0),
            math.log(0.2 / 300_000.0),
            math.log(0.1 / 300_000.0),
        ]
    )
    assert result.mean_log_density_per_event == pytest.approx(
        math.fsum(item.log_density_per_km2 for item in result.event_log_densities) / 4.0
    )
    all_scores = [item for item in result.alarm_recall if item.basis == "all"]
    assert tuple(item.area_budget_km2 for item in all_scores) == FROZEN_D1_AREA_BUDGETS_KM2
    assert tuple(item.actual_area_km2 for item in all_scores) == (
        200_000.0,
        450_000.0,
        450_000.0,
        750_000.0,
        750_000.0,
    )
    assert tuple(item.recall for item in all_scores) == pytest.approx((0.25, 0.5, 0.5, 0.75, 0.75))


def test_location_reports_all_four_weighting_bases_without_conflating_them() -> None:
    result = score_location_events(
        _mass(),
        _grid(),
        event_ids=("e0", "e1", "e2", "e3"),
        event_cell_indices=(0, 1, 2, 3),
        episode_ids=("a", "a", "b", "b"),
        episode_member_counts=(2, 2, 2, 2),
        is_episode_anchor=(True, False, True, False),
    )
    first_budget = {
        item.basis: item for item in result.alarm_recall if item.area_budget_km2 == 300_000.0
    }

    assert (first_budget["all"].hit_weight, first_budget["all"].total_weight) == (1.0, 4.0)
    assert (first_budget["anchor"].hit_weight, first_budget["anchor"].total_weight) == (
        1.0,
        2.0,
    )
    assert (
        first_budget["episode_balanced"].hit_weight,
        first_budget["episode_balanced"].total_weight,
    ) == (0.5, 2.0)
    assert (
        first_budget["subsequent"].hit_weight,
        first_budget["subsequent"].total_weight,
        first_budget["subsequent"].recall,
    ) == (0.0, 2.0, 0.0)


def test_location_empty_population_is_na_and_zero_mass_event_is_negative_infinity() -> None:
    empty = score_location_events(
        _mass(),
        _grid(),
        event_ids=(),
        event_cell_indices=(),
        episode_ids=(),
        episode_member_counts=(),
        is_episode_anchor=(),
    )
    zero_mass = score_location_events(
        np.asarray([0.5, 0.5, 0.0, 0.0]),
        _grid(),
        event_ids=("miss",),
        event_cell_indices=(2,),
        episode_ids=("episode",),
        episode_member_counts=(3,),
        is_episode_anchor=(False,),
    )

    assert empty.event_log_densities == ()
    assert empty.mean_log_density_per_event is None
    assert all(item.total_weight == 0.0 and item.recall is None for item in empty.alarm_recall)
    assert zero_mass.event_log_densities[0].log_density_per_km2 == -math.inf
    partial_episode = next(
        item
        for item in zero_mass.alarm_recall
        if item.basis == "episode_balanced" and item.area_budget_km2 == 300_000.0
    )
    assert partial_episode.total_weight == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"event_ids": ("e", "e")}, "duplicate"),
        ({"episode_ids": ("a",)}, "common length"),
        ({"event_cell_indices": (0, 4)}, "outside"),
        ({"episode_member_counts": (2, 3)}, "inconsistent"),
        ({"episode_member_counts": (1, 1)}, "smaller"),
    ],
)
def test_location_rejects_duplicate_mismatched_or_outside_targets(
    kwargs: dict[str, tuple[object, ...]],
    message: str,
) -> None:
    arguments: dict[str, tuple[object, ...]] = {
        "event_ids": ("e0", "e1"),
        "event_cell_indices": (0, 1),
        "episode_ids": ("a", "a"),
        "episode_member_counts": (2, 2),
        "is_episode_anchor": (True, False),
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        score_location_events(_mass(), _grid(), **arguments)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sum to one"):
        score_location_events(
            np.asarray([0.4, 0.3, 0.2, 0.2]),
            _grid(),
            event_ids=(),
            event_cell_indices=(),
            episode_ids=(),
            episode_member_counts=(),
            is_episode_anchor=(),
        )


def test_poisson_exposure_scores_log_brier_bias_and_keeps_zero_window() -> None:
    positive = score_poisson_exposure(observed_count=2, expected_count=1.5)
    zero = score_poisson_exposure(observed_count=0, expected_count=0.0)
    probability = 1.0 - math.exp(-1.5)

    assert positive.log_score == pytest.approx(poisson_logpmf(2, 1.5))
    assert positive.at_least_one_probability == pytest.approx(probability)
    assert positive.at_least_one_brier == pytest.approx((probability - 1.0) ** 2)
    assert positive.count_bias == -0.5
    assert zero.observed_count == 0
    assert zero.log_score == 0.0
    assert zero.at_least_one_brier == 0.0


def test_nb2_exposure_uses_dispersion_or_explicit_na_never_fake_zero() -> None:
    qualification = NB2DispersionQualification(
        status="evaluable",
        reason="synthetic_frozen_dispersion",
        historical_block_count=8,
        sample_mean_count=1.0,
        sample_variance_count=2.0,
        dispersion_k=2.5,
        observed_information_k=4.0,
        standard_error_k=0.5,
    )
    scored = score_nb2_exposure(
        observed_count=0,
        expected_count=1.2,
        qualification=qualification,
    )
    unavailable = score_nb2_exposure(
        observed_count=0,
        expected_count=1.2,
        qualification=fit_nb2_dispersion([], []),
    )
    poisson_limit = score_nb2_exposure(
        observed_count=1,
        expected_count=1.2,
        qualification=fit_nb2_dispersion([1, 1, 1], [1.0, 1.0, 1.0]),
    )
    probability = nb2_at_least_one_probability(1.2, 2.5)

    assert scored.log_score == pytest.approx(nb2_logpmf(0, 1.2, 2.5))
    assert scored.at_least_one_brier == pytest.approx(probability**2)
    assert unavailable.observed_count == 0
    assert unavailable.log_score is None
    assert unavailable.at_least_one_probability is None
    assert unavailable.at_least_one_brier is None
    assert unavailable.count_bias == 1.2
    assert poisson_limit.status == "evaluable"
    assert poisson_limit.log_score == pytest.approx(poisson_logpmf(1, 1.2))


def test_m0_scores_unique_events_and_m6_brier_on_one_common_support() -> None:
    model = fit_m0_gr_global([4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 6.4])
    result = score_m0_unique_events(
        model,
        event_ids=("m0", "m1"),
        magnitudes=(5.4, 6.2),
    )
    probability = model.m6_plus_probability_mass
    expected_logs = (model.log_probability(5.4), model.log_probability(6.2))

    assert tuple(item.log_probability for item in result.event_scores) == pytest.approx(
        expected_logs
    )
    assert result.log_probability_sum == pytest.approx(math.fsum(expected_logs))
    assert result.mean_m6_plus_brier == pytest.approx(
        (probability**2 + (probability - 1.0) ** 2) / 2.0
    )
    assert result.direct_cross_support_comparison_allowed is False


def test_m3_is_conditional_m5_and_empty_aggregates_remain_na() -> None:
    model = fit_m3_gr_long_m5([5.0, 5.1, 5.4, 5.8, 6.1, 6.5])
    m0 = fit_m0_gr_global([4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 6.4])
    scored = score_m3_conditional_m5_events(
        model,
        event_ids=("tail-0", "tail-1"),
        magnitudes=(5.7, 6.2),
    )
    m0_common_support = score_m0_conditional_m5_events(
        m0,
        event_ids=("tail-0", "tail-1"),
        magnitudes=(5.7, 6.2),
    )
    paired = compare_m0_m3_on_common_m5_support(
        m0,
        model,
        event_ids=("tail-0", "tail-1"),
        magnitudes=(5.7, 6.2),
    )
    empty = score_m3_conditional_m5_events(model, event_ids=(), magnitudes=())

    assert scored.event_scores[0].log_probability == pytest.approx(model.log_probability(5.7))
    assert scored.m6_plus_probability == pytest.approx(model.m6_plus_probability_mass)
    assert scored.mean_m6_plus_brier == pytest.approx(
        (model.m6_plus_probability_mass**2 + (model.m6_plus_probability_mass - 1.0) ** 2) / 2.0
    )
    m0_tail_mass = m0.m5_6_probability_mass + m0.m6_plus_probability_mass
    assert tuple(item.log_probability for item in m0_common_support.event_scores) == pytest.approx(
        (
            m0.log_probability(5.7) - math.log(m0_tail_mass),
            m0.log_probability(6.2) - math.log(m0_tail_mass),
        )
    )
    assert m0_common_support.m6_plus_probability == pytest.approx(
        m0.m6_plus_probability_mass / m0_tail_mass
    )
    assert scored.comparability_note == m0_common_support.comparability_note
    assert "exact_same_unique_M>=5_event_ids" in scored.comparability_note
    assert paired.event_ids == ("tail-0", "tail-1")
    assert paired.magnitudes == (5.7, 6.2)
    assert paired.m0_conditional_m5.event_scores == m0_common_support.event_scores
    assert paired.m3_conditional_m5.event_scores == scored.event_scores
    assert paired.mean_log_score_difference_m3_minus_m0 == pytest.approx(
        scored.mean_log_probability - m0_common_support.mean_log_probability  # type: ignore[operator]
    )
    assert paired.mean_m6_brier_difference_m3_minus_m0 == pytest.approx(
        scored.mean_m6_plus_brier - m0_common_support.mean_m6_plus_brier  # type: ignore[operator]
    )
    assert empty.log_probability_sum is None
    assert empty.mean_log_probability is None
    with pytest.raises(ValueError, match="duplicate"):
        score_m3_conditional_m5_events(
            model,
            event_ids=("same", "same"),
            magnitudes=(5.2, 5.3),
        )
    with pytest.raises(ValueError, match="common length"):
        score_m3_conditional_m5_events(model, event_ids=("one",), magnitudes=())
    with pytest.raises(ValueError, match="at least M5"):
        score_m0_conditional_m5_events(m0, event_ids=("low",), magnitudes=(4.9,))


def test_minimal_joint_m5_factorization_counts_magnitude_population_once() -> None:
    model = fit_m0_gr_global([4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 6.4])
    result = score_minimal_joint_m5(
        _mass(),
        _grid(),
        model,
        event_ids=("j0", "j1"),
        event_cell_indices=(0, 2),
        event_magnitudes=(5.2, 6.1),
        expected_count=2.5,
    )
    expected_count_log = poisson_logpmf(2, 2.5)
    expected_location = math.log(0.4 / 200_000.0) + math.log(0.2 / 300_000.0)
    m5_plus_probability = model.m5_6_probability_mass + model.m6_plus_probability_mass
    expected_magnitude = (
        model.log_probability(5.2)
        + model.log_probability(6.1)
        - 2.0 * math.log(m5_plus_probability)
    )

    assert result.event_count == 2
    assert result.count_log_score == pytest.approx(expected_count_log)
    assert result.conditional_location_log_density_sum == pytest.approx(expected_location)
    assert result.conditional_magnitude_log_probability_sum == pytest.approx(expected_magnitude)
    assert result.joint_log_score == pytest.approx(
        expected_count_log + expected_location + expected_magnitude
    )
    assert result.factorization == JOINT_M5_FACTORIZATION


def test_minimal_joint_keeps_zero_count_and_propagates_unevaluable_nb2_as_na() -> None:
    model = fit_m0_gr_global([4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 6.4])
    poisson_zero = score_minimal_joint_m5(
        _mass(),
        _grid(),
        model,
        event_ids=(),
        event_cell_indices=(),
        event_magnitudes=(),
        expected_count=0.7,
    )
    nb2_na = score_minimal_joint_m5(
        _mass(),
        _grid(),
        model,
        event_ids=("j0",),
        event_cell_indices=(0,),
        event_magnitudes=(5.2,),
        expected_count=0.7,
        count_distribution="nb2",
        nb2_qualification=fit_nb2_dispersion([], []),
    )

    assert poisson_zero.event_count == 0
    assert poisson_zero.count_log_score == pytest.approx(-0.7)
    assert poisson_zero.conditional_location_log_density_sum == 0.0
    assert poisson_zero.conditional_magnitude_log_probability_sum == 0.0
    assert poisson_zero.joint_log_score == pytest.approx(-0.7)
    assert nb2_na.count_log_score is None
    assert nb2_na.joint_log_score is None


def test_minimal_joint_rejects_wrong_support_duplicates_mismatch_and_bad_mass() -> None:
    m0 = fit_m0_gr_global([4.0, 4.2, 4.5, 5.0, 5.5, 6.0])
    m3 = fit_m3_gr_long_m5([5.0, 5.1, 5.4, 5.8, 6.1, 6.5])
    with pytest.raises(ValueError, match="M0_GR_GLOBAL"):
        score_minimal_joint_m5(
            _mass(),
            _grid(),
            m3,
            event_ids=(),
            event_cell_indices=(),
            event_magnitudes=(),
            expected_count=1.0,
        )
    with pytest.raises(ValueError, match="at least M5"):
        score_minimal_joint_m5(
            _mass(),
            _grid(),
            m0,
            event_ids=("low",),
            event_cell_indices=(0,),
            event_magnitudes=(4.9,),
            expected_count=1.0,
        )
    with pytest.raises(ValueError, match="duplicate"):
        score_minimal_joint_m5(
            _mass(),
            _grid(),
            m0,
            event_ids=("same", "same"),
            event_cell_indices=(0, 1),
            event_magnitudes=(5.1, 5.2),
            expected_count=1.0,
        )
    with pytest.raises(ValueError, match="common length"):
        score_minimal_joint_m5(
            _mass(),
            _grid(),
            m0,
            event_ids=("one",),
            event_cell_indices=(),
            event_magnitudes=(5.1,),
            expected_count=1.0,
        )
    with pytest.raises(ValueError, match="sum to one"):
        score_minimal_joint_m5(
            np.asarray([0.4, 0.3, 0.2, 0.2]),
            _grid(),
            m0,
            event_ids=(),
            event_cell_indices=(),
            event_magnitudes=(),
            expected_count=1.0,
        )
