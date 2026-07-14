from __future__ import annotations

import math
from decimal import Decimal, localcontext
from typing import cast

import numpy as np
import pytest
from shapely.geometry import box

import seismoflux.background.poisson as poisson_module
from seismoflux.background.grid import build_clipped_grid
from seismoflux.background.poisson import (
    FROZEN_BANDWIDTHS_KM,
    BandwidthPreScoreGateEvidence,
    BandwidthPreScoreGateItem,
    GaussianMixtureFamily,
    SpatialQuadrature,
    evaluate_spatial_poisson_family,
    evaluate_spatial_poisson_family_cell_masses,
    evaluate_spatial_poisson_family_log_densities,
    fit_spatial_poisson_family,
    fit_uniform_poisson,
    score_log_densities,
    score_spatial_poisson,
    select_kde_bandwidth,
)

_DECIMAL_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
)


def _pre_score_gate_evidence(
    passed_bandwidths_km: tuple[float, ...] = FROZEN_BANDWIDTHS_KM,
) -> BandwidthPreScoreGateEvidence:
    passed = set(passed_bandwidths_km)
    return BandwidthPreScoreGateEvidence(
        candidates=tuple(
            BandwidthPreScoreGateItem(
                bandwidth_km=bandwidth,
                passed=bandwidth in passed,
                numerical_evidence_id=f"grid-convergence:{bandwidth:g}",
                failure_reason=(
                    None
                    if bandwidth in passed
                    else "normalization or three-grid convergence gate failed"
                ),
            )
            for bandwidth in FROZEN_BANDWIDTHS_KM
        )
    )


def _naive_raw_densities(
    training_x_km: poisson_module.FloatArray,
    training_y_km: poisson_module.FloatArray,
    x_km: poisson_module.FloatArray,
    y_km: poisson_module.FloatArray,
    bandwidths_km: tuple[float, ...],
) -> dict[float, poisson_module.FloatArray]:
    squared_distance = (x_km[:, None] - training_x_km[None, :]) ** 2 + (
        y_km[:, None] - training_y_km[None, :]
    ) ** 2
    count = float(training_x_km.size)
    return {
        bandwidth: cast(
            poisson_module.FloatArray,
            np.sum(
                np.exp(-squared_distance / (2.0 * bandwidth * bandwidth)),
                axis=1,
                dtype=np.float64,
            ),
        )
        / (count * 2.0 * math.pi * bandwidth * bandwidth)
        for bandwidth in bandwidths_km
    }


def _high_precision_raw_log_densities(
    training_x_km: poisson_module.FloatArray,
    training_y_km: poisson_module.FloatArray,
    x_km: poisson_module.FloatArray,
    y_km: poisson_module.FloatArray,
    bandwidths_km: tuple[float, ...],
) -> dict[float, poisson_module.FloatArray]:
    outputs: dict[float, poisson_module.FloatArray] = {}
    with localcontext() as context:
        context.prec = 80
        event_count = Decimal(training_x_km.size)
        for bandwidth in bandwidths_km:
            bandwidth_decimal = Decimal(str(bandwidth))
            log_normalizer = (
                event_count.ln()
                + (Decimal(2) * _DECIMAL_PI).ln()
                + Decimal(2) * bandwidth_decimal.ln()
            )
            query_logs: list[float] = []
            for query_x, query_y in zip(x_km, y_km, strict=True):
                kernel_sum = Decimal(0)
                for training_x, training_y in zip(
                    training_x_km,
                    training_y_km,
                    strict=True,
                ):
                    delta_x = Decimal(str(float(query_x))) - Decimal(str(float(training_x)))
                    delta_y = Decimal(str(float(query_y))) - Decimal(str(float(training_y)))
                    exponent = -(delta_x * delta_x + delta_y * delta_y) / (
                        Decimal(2) * bandwidth_decimal * bandwidth_decimal
                    )
                    kernel_sum += exponent.exp()
                query_logs.append(float(kernel_sum.ln() - log_normalizer))
            outputs[bandwidth] = np.asarray(query_logs, dtype=np.float64)
    return outputs


def test_uniform_poisson_has_unit_spatial_mass_and_complete_likelihood() -> None:
    model = fit_uniform_poisson(
        training_event_count=20,
        training_duration_days=10.0,
        study_area_km2=100.0,
    )

    result = model.score(event_count=2, duration_days=3.0)

    assert model.rate_per_day == 2.0
    assert model.spatial_density_per_km2 == 0.01
    assert result.event_log_intensity_sum == pytest.approx(2.0 * math.log(0.02))
    assert result.compensator == 6.0
    assert result.log_likelihood == pytest.approx(2.0 * math.log(0.02) - 6.0)


def test_zero_event_poisson_exposure_retains_the_compensator() -> None:
    result = score_spatial_poisson(
        rate_per_day=0.5,
        event_densities=np.asarray([], dtype=np.float64),
        duration_days=7.0,
        spatial_mass=1.0,
    )

    assert result.event_count == 0
    assert result.event_log_intensity_sum == 0.0
    assert result.compensator == 3.5
    assert result.log_likelihood == -3.5


def test_kde_is_normalized_once_over_the_clipped_study_area() -> None:
    study_area = box(0.0, 0.0, 100_000.0, 100_000.0)
    grid = build_clipped_grid(study_area, cell_size_km=12.5)
    quadrature = SpatialQuadrature.from_grid(grid)
    models = fit_spatial_poisson_family(
        np.asarray([20.0, 80.0]),
        np.asarray([20.0, 80.0]),
        training_duration_days=10.0,
        normalization_quadrature=quadrature,
        bandwidths_km=(75.0, 100.0),
        chunk_size=7,
    )

    expected_raw = _naive_raw_densities(
        np.asarray([20.0, 80.0]),
        np.asarray([20.0, 80.0]),
        quadrature.x_km,
        quadrature.y_km,
        (75.0, 100.0),
    )
    for bandwidth, model in models.items():
        masses = model.grid_masses(quadrature)
        assert math.fsum(masses.values()) == pytest.approx(1.0, abs=1.0e-13)
        assert model.rate_per_day == 0.2
        assert model.normalization_mass < 1.0
        assert model.density_scalar(50.0, 50.0) > 0.0
        expected_mass = float(
            np.sum(expected_raw[bandwidth] * quadrature.area_km2, dtype=np.float64)
        )
        assert model.normalization_mass == pytest.approx(expected_mass, rel=2.0e-14)
        cached = model.normalization_cell_masses
        assert cached is not None
        np.testing.assert_allclose(
            cached,
            expected_raw[bandwidth] * quadrature.area_km2 / expected_mass,
            rtol=2.0e-14,
            atol=0.0,
        )
        assert cached.flags.writeable is False


def test_blocked_gaussian_family_matches_naive_infinite_support_formula() -> None:
    training_x = np.asarray([-12.0, 0.0, 21.5, 80.0], dtype=np.float64)
    training_y = np.asarray([7.0, -3.0, 14.0, 42.0], dtype=np.float64)
    query_x = np.asarray([-12.0, 3.0, 101.0], dtype=np.float64)
    query_y = np.asarray([7.0, 5.5, -22.0], dtype=np.float64)
    bandwidths = FROZEN_BANDWIDTHS_KM
    family = GaussianMixtureFamily(training_x, training_y, chunk_size=2)

    actual = family.raw_densities(query_x, query_y, bandwidths_km=bandwidths)
    expected = _naive_raw_densities(
        training_x,
        training_y,
        query_x,
        query_y,
        bandwidths,
    )

    for bandwidth in bandwidths:
        np.testing.assert_allclose(
            actual[bandwidth],
            expected[bandwidth],
            rtol=2.0e-14,
            atol=0.0,
        )


def test_joint_logsumexp_matches_high_precision_gaussian_mixture_logs() -> None:
    training_x = np.asarray([-12.25, 0.5, 21.75, 80.125], dtype=np.float64)
    training_y = np.asarray([7.5, -3.25, 14.125, 42.75], dtype=np.float64)
    query_x = np.asarray([-12.25, 3.125, 101.5], dtype=np.float64)
    query_y = np.asarray([7.5, 5.75, -22.25], dtype=np.float64)
    family = GaussianMixtureFamily(training_x, training_y, chunk_size=2)

    actual = family.raw_log_densities(
        query_x,
        query_y,
        bandwidths_km=FROZEN_BANDWIDTHS_KM,
    )
    expected = _high_precision_raw_log_densities(
        training_x,
        training_y,
        query_x,
        query_y,
        FROZEN_BANDWIDTHS_KM,
    )

    for bandwidth in FROZEN_BANDWIDTHS_KM:
        np.testing.assert_allclose(
            actual[bandwidth],
            expected[bandwidth],
            rtol=0.0,
            atol=2.0e-14,
        )


def test_fitted_grid_masses_reuse_read_only_normalization_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quadrature = SpatialQuadrature(
        cell_ids=("a", "b", "c"),
        x_km=np.asarray([0.0, 12.5, 25.0]),
        y_km=np.asarray([0.0, 0.0, 0.0]),
        area_km2=np.asarray([78.125, 156.25, 78.125]),
    )
    model = fit_spatial_poisson_family(
        np.asarray([0.0, 25.0]),
        np.asarray([0.0, 0.0]),
        training_duration_days=2.0,
        normalization_quadrature=quadrature,
        bandwidths_km=(75.0,),
    )[75.0]
    cached = model.normalization_cell_masses
    assert cached is not None

    def fail_raw_densities(
        self: GaussianMixtureFamily,
        x_km: object,
        y_km: object,
        *,
        bandwidths_km: tuple[float, ...],
    ) -> dict[float, poisson_module.FloatArray]:
        del self, x_km, y_km, bandwidths_km
        raise AssertionError("normalization grid must not be evaluated twice")

    monkeypatch.setattr(GaussianMixtureFamily, "raw_densities", fail_raw_densities)

    masses = model.grid_masses(quadrature)
    family_masses = evaluate_spatial_poisson_family_cell_masses(
        {75.0: model},
        quadrature,
    )

    np.testing.assert_array_equal(np.asarray(tuple(masses.values())), cached)
    assert family_masses[75.0] is cached


def test_medium_evaluation_bounds_distance_blocks_and_shares_all_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260713)
    training_x = rng.uniform(-500.0, 500.0, size=401)
    training_y = rng.uniform(-500.0, 500.0, size=401)
    query_x = rng.uniform(-500.0, 500.0, size=257)
    query_y = rng.uniform(-500.0, 500.0, size=257)
    family = GaussianMixtureFamily(training_x, training_y, chunk_size=32)
    original = poisson_module._squared_distance_block
    block_shapes: list[tuple[int, int]] = []
    memory_cap_elements = 8_000
    monkeypatch.setattr(poisson_module, "_MAX_DISTANCE_BLOCK_ELEMENTS", memory_cap_elements)

    def record_block(
        query_points_km: poisson_module.FloatArray,
        training_points_km: poisson_module.FloatArray,
        training_squared_norms_km2: poisson_module.FloatArray,
    ) -> poisson_module.FloatArray:
        result = original(
            query_points_km,
            training_points_km,
            training_squared_norms_km2,
        )
        block_shapes.append(result.shape)
        return result

    monkeypatch.setattr(poisson_module, "_squared_distance_block", record_block)

    outputs = family.raw_densities(
        query_x,
        query_y,
        bandwidths_km=FROZEN_BANDWIDTHS_KM,
    )

    effective_chunk_size = min(32, memory_cap_elements // training_x.size)
    assert len(block_shapes) == math.ceil(query_x.size / effective_chunk_size)
    assert all(
        rows <= effective_chunk_size
        and columns == training_x.size
        and rows * columns <= memory_cap_elements
        for rows, columns in block_shapes
    )
    assert set(outputs) == set(FROZEN_BANDWIDTHS_KM)
    assert all(values.shape == query_x.shape for values in outputs.values())
    assert all(np.isfinite(values).all() and np.all(values > 0.0) for values in outputs.values())

    block_shapes.clear()
    log_outputs = family.raw_log_densities(
        query_x,
        query_y,
        bandwidths_km=FROZEN_BANDWIDTHS_KM,
    )
    assert len(block_shapes) == math.ceil(query_x.size / effective_chunk_size)
    for bandwidth in FROZEN_BANDWIDTHS_KM:
        np.testing.assert_allclose(
            np.exp(log_outputs[bandwidth]),
            outputs[bandwidth],
            rtol=3.0e-14,
            atol=0.0,
        )


def test_family_target_density_evaluation_shares_one_distance_pass() -> None:
    quadrature = SpatialQuadrature(
        cell_ids=("a", "b", "c", "d"),
        x_km=np.asarray([0.0, 12.5, 0.0, 12.5]),
        y_km=np.asarray([0.0, 0.0, 12.5, 12.5]),
        area_km2=np.full(4, 156.25),
    )
    training_x = np.asarray([0.0, 12.5, 6.25])
    training_y = np.asarray([0.0, 12.5, 6.25])
    models = fit_spatial_poisson_family(
        training_x,
        training_y,
        training_duration_days=3.0,
        normalization_quadrature=quadrature,
    )
    target_x = np.asarray([2.0, 10.0])
    target_y = np.asarray([3.0, 11.0])

    actual = evaluate_spatial_poisson_family(models, target_x, target_y)
    actual_log = evaluate_spatial_poisson_family_log_densities(
        models,
        target_x,
        target_y,
    )
    expected_raw = _naive_raw_densities(
        training_x,
        training_y,
        target_x,
        target_y,
        FROZEN_BANDWIDTHS_KM,
    )

    for bandwidth, model in models.items():
        expected_log = np.log(expected_raw[bandwidth]) - math.log(model.normalization_mass)
        np.testing.assert_allclose(
            actual[bandwidth],
            expected_raw[bandwidth] / model.normalization_mass,
            rtol=2.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            actual_log[bandwidth],
            expected_log,
            rtol=0.0,
            atol=2.0e-14,
        )


def test_spatial_poisson_score_uses_same_rate_and_normalized_density() -> None:
    grid = build_clipped_grid(box(0.0, 0.0, 50_000.0, 50_000.0), cell_size_km=12.5)
    quadrature = SpatialQuadrature.from_grid(grid)
    model = fit_spatial_poisson_family(
        np.asarray([10.0, 40.0]),
        np.asarray([10.0, 40.0]),
        training_duration_days=20.0,
        normalization_quadrature=quadrature,
        bandwidths_km=(75.0,),
    )[75.0]

    score = model.score(
        np.asarray([20.0]),
        np.asarray([20.0]),
        duration_days=5.0,
    )

    expected_intensity = model.rate_per_day * model.density_scalar(20.0, 20.0)
    assert score.event_log_intensity_sum == pytest.approx(math.log(expected_intensity))
    assert score.compensator == pytest.approx(0.5)


def test_far_infinite_support_event_scores_finitely_when_density_exp_underflows() -> None:
    quadrature = SpatialQuadrature(
        cell_ids=("origin",),
        x_km=np.asarray([0.0]),
        y_km=np.asarray([0.0]),
        area_km2=np.asarray([156.25]),
    )
    model = fit_spatial_poisson_family(
        np.asarray([0.0]),
        np.asarray([0.0]),
        training_duration_days=10.0,
        normalization_quadrature=quadrature,
        bandwidths_km=(75.0,),
    )[75.0]
    far_x_km = 1_000_000.0

    assert model.density_scalar(far_x_km, 0.0) == 0.0
    log_density = model.log_density_scalar(far_x_km, 0.0)
    assert math.isfinite(log_density)
    score = model.score(
        np.asarray([far_x_km]),
        np.asarray([0.0]),
        duration_days=3.0,
    )
    direct_log_score = score_log_densities(
        rate_per_day=model.rate_per_day,
        event_log_densities=np.asarray([log_density]),
        duration_days=3.0,
        spatial_mass=1.0,
    )

    assert score == direct_log_score
    assert math.isfinite(score.event_log_intensity_sum)
    assert math.isfinite(score.log_likelihood)
    assert score.event_log_intensity_sum == pytest.approx(
        math.log(model.rate_per_day) + log_density
    )
    assert score.compensator == pytest.approx(model.rate_per_day * 3.0)


def test_bandwidth_selection_uses_paired_four_fold_standard_error_and_largest_eligible() -> None:
    scores = {
        75.0: (1.0, 1.0, 1.0, 1.0),
        100.0: (0.8, 1.1, 0.8, 1.1),
        150.0: (0.4, 0.4, 0.4, 0.4),
        200.0: (0.2, 0.2, 0.2, 0.2),
        300.0: (0.7, 1.1, 0.7, 1.1),
    }

    gate_evidence = _pre_score_gate_evidence()
    selection = select_kde_bandwidth(
        scores,
        pre_score_gate_evidence=gate_evidence,
    )

    assert selection.best_mean_bandwidth_km == 75.0
    assert selection.selected_bandwidth_km == 300.0
    audits = {item.bandwidth_km: item for item in selection.candidates}
    expected_se = np.std(np.asarray([-0.3, 0.1, -0.3, 0.1]), ddof=1) / math.sqrt(4.0)
    assert audits[300.0].paired_standard_error == pytest.approx(expected_se)
    assert audits[300.0].eligible is True
    assert audits[200.0].eligible is False
    assert selection.pre_score_gate_evidence is gate_evidence


def test_exact_best_mean_tie_uses_largest_bandwidth() -> None:
    scores = {bandwidth: (1.0, 1.0, 1.0, 1.0) for bandwidth in FROZEN_BANDWIDTHS_KM}

    selection = select_kde_bandwidth(
        scores,
        pre_score_gate_evidence=_pre_score_gate_evidence(),
    )

    assert selection.best_mean_bandwidth_km == 300.0
    assert selection.selected_bandwidth_km == 300.0
    assert selection.gate_excluded_bandwidths_km == ()


def test_bandwidth_selection_accepts_only_gate_passing_frozen_subset() -> None:
    gate_evidence = _pre_score_gate_evidence((75.0, 150.0, 300.0))
    selection = select_kde_bandwidth(
        {
            75.0: (1.0, 1.0, 1.0, 1.0),
            150.0: (0.8, 0.8, 0.8, 0.8),
            300.0: (1.0, 1.0, 1.0, 1.0),
        },
        pre_score_gate_evidence=gate_evidence,
    )

    assert selection.best_mean_bandwidth_km == 300.0
    assert selection.selected_bandwidth_km == 300.0
    assert tuple(item.bandwidth_km for item in selection.candidates) == (75.0, 150.0, 300.0)
    assert selection.gate_excluded_bandwidths_km == (100.0, 200.0)
    assert selection.gate_exclusions == gate_evidence.exclusions
    assert tuple(item.failure_reason for item in selection.gate_exclusions) == (
        "normalization or three-grid convergence gate failed",
        "normalization or three-grid convergence gate failed",
    )


def test_bandwidth_selection_hard_fails_when_all_candidates_fail_gates() -> None:
    with pytest.raises(ValueError, match="at least one gate-passing"):
        select_kde_bandwidth(
            {},
            pre_score_gate_evidence=_pre_score_gate_evidence(()),
        )


def test_bandwidth_selection_rejects_unknown_or_non_four_fold_candidate() -> None:
    with pytest.raises(ValueError, match="unknown frozen"):
        select_kde_bandwidth(
            {125.0: (1.0, 1.0, 1.0, 1.0)},
            pre_score_gate_evidence=_pre_score_gate_evidence(),
        )
    with pytest.raises(ValueError, match="exactly four"):
        select_kde_bandwidth(
            {75.0: (1.0, 1.0, 1.0)},  # type: ignore[dict-item]
            pre_score_gate_evidence=_pre_score_gate_evidence((75.0,)),
        )


def test_pre_score_gate_requires_complete_fixed_order_and_honest_reasons() -> None:
    valid_items = _pre_score_gate_evidence().candidates
    with pytest.raises(ValueError, match="exactly five.*fixed order"):
        BandwidthPreScoreGateEvidence(candidates=valid_items[:-1])
    with pytest.raises(ValueError, match="exactly five.*fixed order"):
        BandwidthPreScoreGateEvidence(candidates=tuple(reversed(valid_items)))
    with pytest.raises(ValueError, match="must not contain a failure reason"):
        BandwidthPreScoreGateItem(
            bandwidth_km=75.0,
            passed=True,
            numerical_evidence_id="gate:75",
            failure_reason="fabricated failure",
        )
    with pytest.raises(ValueError, match="non-empty failure reason"):
        BandwidthPreScoreGateItem(
            bandwidth_km=75.0,
            passed=False,
            numerical_evidence_id="gate:75",
            failure_reason="  ",
        )


def test_bandwidth_selection_rejects_selectively_deleted_passing_score() -> None:
    scores = {bandwidth: (1.0, 1.0, 1.0, 1.0) for bandwidth in FROZEN_BANDWIDTHS_KM}
    scores.pop(150.0)

    with pytest.raises(ValueError, match="exactly match.*gate-passing"):
        select_kde_bandwidth(
            scores,
            pre_score_gate_evidence=_pre_score_gate_evidence(),
        )
