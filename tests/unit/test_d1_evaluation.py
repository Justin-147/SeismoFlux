from __future__ import annotations

import pytest

from seismoflux.d1_replay.evaluation import (
    D1_AREA_BUDGETS_KM2,
    D1_FOLD_IDS,
    D1_MODEL_ORDER,
    D1BootstrapEffect,
    D1ClusterModelOutcome,
    D1IssueAlarmOutcome,
    classify_primary_raw_effect,
    paired_cluster_bootstrap,
    summarize_alarm_exposure,
    summarize_metrics,
    validate_complete_outcomes,
)


def _hits(first_hit_index: int | None) -> tuple[bool, ...]:
    return tuple(
        first_hit_index is not None and index >= first_hit_index
        for index in range(len(D1_AREA_BUDGETS_KM2))
    )


def _complete_fixture() -> tuple[D1ClusterModelOutcome, ...]:
    outcomes: list[D1ClusterModelOutcome] = []
    for horizon, cluster_count in ((30, 6), (90, 3)):
        for cluster_index in range(cluster_count):
            fold = D1_FOLD_IDS[cluster_index % len(D1_FOLD_IDS)]
            for model in D1_MODEL_ORDER:
                baseline_hit = cluster_index < 2
                candidate_gain = horizon == 30 and cluster_index in {2, 3}
                first_hit = 2 if baseline_hit or (model != "B0" and candidate_gain) else None
                outcomes.append(
                    D1ClusterModelOutcome(
                        cluster_id=f"h{horizon}-c{cluster_index}",
                        fold_id=fold,
                        issue_id=f"h{horizon}-i{cluster_index}",
                        horizon_days=horizon,
                        model_id=model,
                        log_density=-8.0 + 0.01 * cluster_index,
                        outside_support=False,
                        hit_by_area=_hits(first_hit),
                    )
                )
    return tuple(outcomes)


def test_metrics_and_registered_bootstrap_use_cluster_blocks() -> None:
    outcomes = _complete_fixture()
    metrics = summarize_metrics(outcomes)
    baseline = next(
        item
        for item in metrics
        if item.model_id == "B0"
        and item.horizon_days == 30
        and item.fold_id is None
        and item.area_budget_km2 == 600_000.0
    )
    candidate = next(
        item
        for item in metrics
        if item.model_id == "B0_R30_C_A_dynamic"
        and item.horizon_days == 30
        and item.fold_id is None
        and item.area_budget_km2 == 600_000.0
    )
    assert (baseline.cluster_count, baseline.hit_count, baseline.recall) == (6, 2, 2 / 6)
    assert (candidate.cluster_count, candidate.hit_count, candidate.recall) == (6, 4, 4 / 6)

    first = paired_cluster_bootstrap(outcomes)
    second = paired_cluster_bootstrap(outcomes)
    assert first == second
    primary = next(
        item
        for item in first
        if item.model_id == "B0_R30_C_A_dynamic"
        and item.horizon_days == 30
        and item.area_budget_km2 == 600_000.0
    )
    assert primary.observed_hit_gain == 2
    assert primary.probability_gain_positive >= 0.90

    decision = classify_primary_raw_effect(
        metrics,
        first,
        model_id="B0_R30_C_A_dynamic",
    )
    assert decision.raw_effect_level == "strong"
    assert decision.pooled_hit_gain == 2
    assert decision.nonworse_fold_count == 3


def test_same_hits_at_smaller_area_is_promising() -> None:
    outcomes = list(_complete_fixture())
    for index, item in enumerate(outcomes):
        if item.horizon_days == 30 and item.model_id == "B0_C":
            outcomes[index] = D1ClusterModelOutcome(
                cluster_id=item.cluster_id,
                fold_id=item.fold_id,
                issue_id=item.issue_id,
                horizon_days=item.horizon_days,
                model_id=item.model_id,
                log_density=item.log_density,
                outside_support=item.outside_support,
                hit_by_area=_hits(1 if int(item.cluster_id.rsplit("c", 1)[1]) < 2 else None),
            )
    metrics = summarize_metrics(outcomes)
    fake_bootstrap = (
        D1BootstrapEffect(
            model_id="B0_C",
            horizon_days=30,
            area_budget_km2=600_000.0,
            observed_hit_gain=0,
            observed_recall_gain=0.0,
            lower_95=0.0,
            upper_95=0.0,
            probability_gain_positive=0.0,
            replication_count=2_000,
        ),
    )
    decision = classify_primary_raw_effect(
        metrics,
        fake_bootstrap,
        model_id="B0_C",
    )
    assert decision.raw_effect_level == "promising"
    assert decision.area_steps_saved == 1


def test_alarm_exposure_uses_every_issue_instead_of_target_weighting() -> None:
    expected: dict[int, set[tuple[str, str]]] = {30: set(), 90: set()}
    alarms: list[D1IssueAlarmOutcome] = []
    for horizon, per_fold in ((30, 8), (90, 3)):
        for fold in D1_FOLD_IDS:
            for issue_index in range(per_fold):
                issue_id = f"{fold}-{horizon}-{issue_index}"
                expected[horizon].add((fold, issue_id))
                for model in D1_MODEL_ORDER:
                    alarms.append(
                        D1IssueAlarmOutcome(
                            fold_id=fold,
                            issue_id=issue_id,
                            horizon_days=horizon,
                            model_id=model,
                            actual_area_km2=(
                                299_000.0,
                                449_000.0,
                                599_000.0,
                                749_000.0,
                                959_000.0,
                            ),
                        )
                    )
    metrics = summarize_alarm_exposure(
        alarms,
        expected_issues_by_horizon=expected,
        study_area_km2=9_600_000.0,
    )
    pooled = next(
        item
        for item in metrics
        if item.model_id == "B0"
        and item.horizon_days == 30
        and item.fold_id is None
        and item.area_budget_km2 == 600_000.0
    )
    assert pooled.issue_count == 24
    assert pooled.mean_actual_area_km2 == 599_000.0
    assert pooled.mean_alarm_fraction == pytest.approx(599_000.0 / 9_600_000.0)

    with pytest.raises(ValueError, match="frozen exposures"):
        summarize_alarm_exposure(
            alarms[:-1],
            expected_issues_by_horizon=expected,
            study_area_km2=9_600_000.0,
        )


def test_incomplete_model_support_and_nonfinite_density_fail_closed() -> None:
    outcomes = _complete_fixture()
    with pytest.raises(ValueError, match="does not share B0 cluster support"):
        validate_complete_outcomes(outcomes[:-1])
    expected = {
        30: {(f"h30-c{index}", D1_FOLD_IDS[index % 3], f"h30-i{index}") for index in range(6)},
        90: {(f"h90-c{index}", D1_FOLD_IDS[index % 3], f"h90-i{index}") for index in range(3)}
        | {("missing-frozen-cluster", "fold_1", "missing-issue")},
    }
    with pytest.raises(ValueError, match="frozen target support"):
        validate_complete_outcomes(outcomes, expected_support_by_horizon=expected)
    with pytest.raises(ValueError, match="log density"):
        D1ClusterModelOutcome(
            cluster_id="cluster",
            fold_id="fold_1",
            issue_id="issue",
            horizon_days=30,
            model_id="B0",
            log_density=float("nan"),
            outside_support=False,
            hit_by_area=_hits(None),
        )
    outside = D1ClusterModelOutcome(
        cluster_id="outside",
        fold_id="fold_1",
        issue_id="issue",
        horizon_days=30,
        model_id="B0",
        log_density=None,
        outside_support=True,
        hit_by_area=_hits(None),
    )
    assert outside.outside_support
