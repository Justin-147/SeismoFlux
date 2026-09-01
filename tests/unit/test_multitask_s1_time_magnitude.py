from __future__ import annotations

import math

import pytest

import seismoflux.multitask_s1.time_magnitude as time_magnitude
from seismoflux.multitask_s1.time_magnitude import (
    fit_expanding_poisson,
    fit_m0_gr_global,
    fit_m3_gr_long_m5,
    fit_nb2_dispersion,
    nb2_at_least_one_probability,
    nb2_logpmf,
    poisson_at_least_one_probability,
    poisson_logpmf,
    qualified_nb2_logpmf,
)


def test_expanding_poisson_uses_only_supplied_count_and_exposure() -> None:
    model = fit_expanding_poisson(history_event_count=12, history_exposure_days=120.0)

    assert model.rate_per_day == pytest.approx(0.1)
    assert model.expected_count(30.0) == pytest.approx(3.0)
    assert model.at_least_one_probability(30.0) == pytest.approx(1.0 - math.exp(-3.0))
    assert model.logpmf(2, 30.0) == pytest.approx(poisson_logpmf(2, 3.0))


def test_poisson_zero_history_and_zero_target_are_exactly_safe() -> None:
    model = fit_expanding_poisson(history_event_count=0, history_exposure_days=365.0)

    assert model.rate_per_day == 0.0
    assert model.expected_count(7.0) == 0.0
    assert model.at_least_one_probability(7.0) == 0.0
    assert model.logpmf(0, 7.0) == 0.0
    assert model.logpmf(1, 7.0) == -math.inf
    assert poisson_at_least_one_probability(1.0e-12) == pytest.approx(1.0e-12)


def test_poisson_rejects_invalid_history_or_horizon() -> None:
    with pytest.raises(TypeError, match="integer"):
        fit_expanding_poisson(history_event_count=True, history_exposure_days=1.0)
    with pytest.raises(ValueError, match="positive"):
        fit_expanding_poisson(history_event_count=1, history_exposure_days=0.0)
    with pytest.raises(ValueError, match="positive"):
        fit_expanding_poisson(history_event_count=1, history_exposure_days=1.0).expected_count(0.0)


def test_nb2_poisson_limit_is_explicit_and_scores_as_poisson() -> None:
    qualification = fit_nb2_dispersion([1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0])

    assert qualification.status == "poisson_limit"
    assert qualification.dispersion_k is None
    assert qualified_nb2_logpmf(0, 1.0, qualification) == pytest.approx(-1.0)


def test_nb2_overdispersed_blocks_produce_finite_deterministic_mle() -> None:
    counts = [0, 0, 1, 9, 0, 10]
    means = [20.0 / 6.0] * 6

    first = fit_nb2_dispersion(counts, means)
    second = fit_nb2_dispersion(counts, means)

    assert first.status == "evaluable"
    assert first == second
    assert first.dispersion_k is not None
    assert first.dispersion_k == pytest.approx(0.2905545190414773)
    assert first.observed_information_k is not None
    assert first.observed_information_k > 0.0
    probability_mass = math.fsum(
        math.exp(nb2_logpmf(count, 20.0 / 6.0, first.dispersion_k)) for count in range(500)
    )
    assert probability_mass == pytest.approx(1.0, abs=1.0e-12)


def test_nb2_zero_count_probability_and_at_least_one_are_stable() -> None:
    mean = 1.0e-12
    k = 1.0e8

    log_zero = nb2_logpmf(0, mean, k)
    at_least_one = nb2_at_least_one_probability(mean, k)

    assert log_zero == pytest.approx(-mean, abs=1.0e-20)
    assert at_least_one == pytest.approx(mean, abs=1.0e-20)
    assert nb2_logpmf(0, 0.0, k) == 0.0
    assert nb2_logpmf(1, 0.0, k) == -math.inf


def test_nb2_sparse_or_incompatible_history_is_not_evaluable() -> None:
    empty = fit_nb2_dispersion([], [])
    one_block = fit_nb2_dispersion([3], [3.0])
    zero_mean_with_event = fit_nb2_dispersion([0, 3, 0], [0.0, 0.0, 0.0])

    assert empty.status == "not_evaluable"
    assert empty.reason == "no_non_overlapping_history_blocks"
    assert one_block.status == "not_evaluable"
    assert one_block.reason == "fewer_than_two_non_overlapping_history_blocks"
    assert zero_mean_with_event.status == "not_evaluable"
    assert zero_mean_with_event.reason == "positive_count_with_zero_poisson_expectation"
    with pytest.raises(ValueError, match="common length"):
        fit_nb2_dispersion([0, 1], [0.5])


def test_nb2_optimizer_failure_is_not_evaluable_not_a_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_optimizer(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic non-convergence")

    monkeypatch.setattr(time_magnitude, "brentq", fail_optimizer)
    qualification = fit_nb2_dispersion(
        [0, 0, 1, 9, 0, 10],
        [20.0 / 6.0] * 6,
    )

    assert qualification.status == "not_evaluable"
    assert qualification.reason == "nb2_maximum_likelihood_did_not_converge"
    with pytest.raises(ValueError, match="must not be scored"):
        qualified_nb2_logpmf(1, 1.0, qualification)


def test_m0_gr_uses_frozen_aki_formula_bins_and_truncation() -> None:
    magnitudes = [4.0, 4.1, 4.2, 4.4, 4.8, 5.1, 5.8, 6.2]

    model = fit_m0_gr_global(magnitudes)
    expected_b = math.log10(math.e) / (math.fsum(magnitudes) / len(magnitudes) - 3.95)

    assert model.model_id == "M0_GR_GLOBAL"
    assert model.mc == 4.0
    assert model.maximum_magnitude == 9.5
    assert model.bin_width == 0.1
    assert model.b_value == pytest.approx(expected_b)
    assert len(model.bin_probability_masses) == 55
    assert math.fsum(model.bin_probability_masses) == pytest.approx(1.0, abs=1.0e-12)
    assert model.m5_6_probability_mass == pytest.approx(
        math.fsum(model.bin_probability_masses[10:20]), abs=1.0e-12
    )
    assert model.m6_plus_probability_mass == pytest.approx(
        math.fsum(model.bin_probability_masses[20:]), abs=1.0e-12
    )


def test_gr_scoring_respects_magnitude_bin_boundaries_and_closed_upper_edge() -> None:
    model = fit_m0_gr_global([4.0, 4.2, 4.5, 5.0, 5.5, 6.0])

    assert model.log_probability(4.0) == pytest.approx(math.log(model.bin_probability_masses[0]))
    assert model.log_probability(4.1) == pytest.approx(math.log(model.bin_probability_masses[1]))
    assert model.log_probability(5.999) == pytest.approx(math.log(model.bin_probability_masses[19]))
    assert model.log_probability(6.0) == pytest.approx(math.log(model.bin_probability_masses[20]))
    assert model.log_probability(9.5) == pytest.approx(math.log(model.bin_probability_masses[-1]))
    with pytest.raises(ValueError, match="outside"):
        model.log_probability(3.9)
    with pytest.raises(ValueError, match="outside"):
        model.log_probability(9.6)


@pytest.mark.parametrize(
    ("boundary", "expected_index"),
    [(6.3, 23), (6.8, 28), (7.3, 33)],
)
def test_gr_decimal_tenth_boundaries_use_frozen_half_open_bins(
    boundary: float,
    expected_index: int,
) -> None:
    model = fit_m0_gr_global([4.0, 4.2, 4.5, 5.0, 5.5, 6.0])
    epsilon = 1.0e-12

    assert model.log_probability(boundary - epsilon) == pytest.approx(
        math.log(model.bin_probability_masses[expected_index - 1])
    )
    assert model.log_probability(boundary) == pytest.approx(
        math.log(model.bin_probability_masses[expected_index])
    )
    assert model.log_probability(boundary + epsilon) == pytest.approx(
        math.log(model.bin_probability_masses[expected_index])
    )
    arithmetic_boundary = 4.0 + expected_index * 0.1
    assert model.log_probability(arithmetic_boundary) == pytest.approx(
        math.log(model.bin_probability_masses[expected_index])
    )


def test_gr_upper_boundary_remains_in_final_closed_bin() -> None:
    model = fit_m0_gr_global([4.0, 4.2, 4.5, 5.0, 5.5, 6.0])

    assert model.log_probability(9.5 - 1.0e-12) == pytest.approx(
        math.log(model.bin_probability_masses[-1])
    )
    assert model.log_probability(9.5) == pytest.approx(math.log(model.bin_probability_masses[-1]))


def test_m3_long_m5_is_a_separate_normalized_tail_model() -> None:
    magnitudes = [5.0, 5.1, 5.2, 5.5, 5.9, 6.0, 6.4]

    first = fit_m3_gr_long_m5(magnitudes)
    second = fit_m3_gr_long_m5(magnitudes)

    assert first == second
    assert first.model_id == "M3_GR_LONG_M5"
    assert first.mc == 5.0
    assert len(first.bin_probability_masses) == 45
    assert math.fsum(first.bin_probability_masses) == pytest.approx(1.0, abs=1.0e-12)
    assert first.m5_6_probability_mass + first.m6_plus_probability_mass == pytest.approx(
        1.0, abs=1.0e-12
    )


def test_gr_fit_fails_closed_outside_frozen_support() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_m0_gr_global([])
    with pytest.raises(ValueError, match="inside"):
        fit_m0_gr_global([3.9, 4.1])
    with pytest.raises(ValueError, match="inside"):
        fit_m3_gr_long_m5([5.0, 9.6])
