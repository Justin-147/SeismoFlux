"""End-to-end known-answer checks for the purely synthetic Stage 2P MVP."""

from __future__ import annotations

import math

import numpy as np

from seismoflux.stage2p.synthetic_experiment import (
    QUERY_CUTOFF,
    SYNTHETIC_REGION_COUNT,
    TARGET_CLUSTER_COUNT,
    run_all_synthetic_scenarios,
)


def test_three_known_answer_scenarios_have_the_expected_scientific_direction() -> None:
    results = run_all_synthetic_scenarios(bootstrap_replicates=300)
    by_id = {result.scenario.scenario_id: result for result in results}

    assert tuple(by_id) == (
        "recent_activity_predictive",
        "no_recent_signal",
        "recent_activity_misleading",
    )
    assert all(result.expected_behavior_passed for result in results)
    assert all(result.synthetic_known_answer_status == "passed" for result in results)
    assert all(result.evaluation.unique_event_count == 36 for result in results)
    assert all(
        result.evaluation.independent_cluster_count == TARGET_CLUSTER_COUNT for result in results
    )
    assert SYNTHETIC_REGION_COUNT < TARGET_CLUSTER_COUNT
    assert all(
        result.evaluation.independent_region_count == SYNTHETIC_REGION_COUNT for result in results
    )
    for result in results:
        clusters_by_region: dict[str, set[str]] = {}
        for target in result.targets:
            clusters_by_region.setdefault(target.region_id, set()).add(target.cluster_id)
        assert set(map(len, clusters_by_region.values())) == {3}
    assert all(
        tuple(
            sum(observation.horizon_days == horizon for observation in result.observations)
            for horizon in (7, 30, 90)
        )
        == (12, 24, 36)
        for result in results
    )

    predictive = by_id["recent_activity_predictive"]
    assert predictive.counterfactual_confirmatory_gate.status == "passed"
    assert predictive.evaluation.macro_model_recall["P0"].strict_event_recall == 0.0
    assert predictive.evaluation.macro_model_recall["P1"].strict_event_recall == 1.0
    assert predictive.evaluation.macro_model_recall["PP"].strict_event_recall == 0.0
    for comparison in predictive.evaluation.comparisons.values():
        assert comparison.macro_recall_gain_percentage_points == 100.0
        assert comparison.macro_information_gain_nats_per_event > 1.0
        assert comparison.recall_interval.lower > 0.0
        assert comparison.information_gain_interval.lower > 0.0
        assert all(diagnostic.remains_positive for diagnostic in comparison.removal_diagnostics)
    predictive_diagnostics = tuple(
        diagnostic
        for comparison in predictive.evaluation.comparisons.values()
        for diagnostic in comparison.removal_diagnostics
    )
    diagnostic_endpoints = {item.endpoint for item in predictive_diagnostics}
    assert any(
        not math.isclose(
            next(
                item.residual_with_original_denominator
                for item in predictive_diagnostics
                if item.endpoint == endpoint and item.group_kind == "region"
            ),
            next(
                item.residual_with_original_denominator
                for item in predictive_diagnostics
                if item.endpoint == endpoint and item.group_kind == "cluster"
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for endpoint in diagnostic_endpoints
    )

    no_signal = by_id["no_recent_signal"]
    assert no_signal.counterfactual_confirmatory_gate.status == "failed"
    assert no_signal.forecast.p1.spatial_density is no_signal.forecast.p0.spatial_density
    assert no_signal.forecast.pp.spatial_density is no_signal.forecast.p0.spatial_density
    assert no_signal.forecast.p1.alarm is no_signal.forecast.p0.alarm
    assert no_signal.forecast.pp.alarm is no_signal.forecast.p0.alarm
    for comparison in no_signal.evaluation.comparisons.values():
        assert comparison.macro_recall_gain_percentage_points == 0.0
        assert comparison.macro_information_gain_nats_per_event == 0.0
        assert comparison.recall_interval.lower == 0.0
        assert comparison.information_gain_interval.lower == 0.0

    misleading = by_id["recent_activity_misleading"]
    assert misleading.counterfactual_confirmatory_gate.status == "failed"
    assert misleading.evaluation.macro_model_recall["P0"].strict_event_recall == 1.0
    assert misleading.evaluation.macro_model_recall["P1"].strict_event_recall == 0.0
    assert misleading.evaluation.macro_model_recall["PP"].strict_event_recall == 1.0
    for comparison in misleading.evaluation.comparisons.values():
        assert comparison.macro_recall_gain_percentage_points == -100.0
        assert math.isclose(
            comparison.macro_information_gain_nats_per_event,
            -math.log(2.0),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )


def test_forecasts_are_causal_equal_area_and_target_free() -> None:
    results = run_all_synthetic_scenarios(bootstrap_replicates=50)

    for result in results:
        forecast_ids = {event.id for event in result.scenario.forecast_events}
        target_ids = {target.event_id for target in result.targets}
        assert forecast_ids.isdisjoint(target_ids)
        assert all(
            event.origin_time <= QUERY_CUTOFF
            and event.first_seen < result.forecast.windows.issue_time
            for event in result.scenario.forecast_events
        )
        assert all(
            target.origin_time > result.forecast.windows.issue_time for target in result.targets
        )
        areas = tuple(model.alarm.actual_area_km2 for model in result.forecast.models)
        assert areas == (600_000.0, 600_000.0, 600_000.0)
        assert max(areas) - min(areas) == 0.0
        assert all(model.alarm.selected_indices.size == 960 for model in result.forecast.models)
        for model in result.forecast.models:
            assert np.isclose(
                math.fsum(float(value) for value in model.spatial_density.mass_25km),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
