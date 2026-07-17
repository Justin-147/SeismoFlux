from __future__ import annotations

import math

import pytest

from seismoflux.anomaly_increment.evaluation import (
    AlarmAreaPoint,
    ConfidenceInterval,
    EventHorizonMembership,
    G2Evidence,
    G3FoldEvidence,
    GateOutcome,
    apply_preregistered_adoption_matrix,
    evaluate_g2,
    evaluate_g3,
    frozen_same_recall_budget_grid,
    information_gain_per_physical_event,
    percentile_interval,
    same_area_recall_gain_percentage_points,
    same_recall_union_area_relative_reduction,
    stratified_five_horizon_bootstrap_indices,
    strict_recall,
)
from seismoflux.anomaly_increment.placebo import (
    InfrastructureInterruption,
    PermutationReplication,
    build_space_bijection,
    build_time_bijection,
    reduce_permutation_test,
)
from seismoflux.anomaly_increment.preregistration import (
    RandomPartitionRole,
    RandomPurpose,
    Stage4SeedContext,
)


def _context(
    purpose: RandomPurpose,
    *,
    replicate_index: int = 0,
    partition_role: RandomPartitionRole = "joint",
) -> Stage4SeedContext:
    if purpose == "space_permutation":
        return Stage4SeedContext(
            purpose=purpose,
            evaluation_id="formal-validation",
            partition_role=partition_role,
            replicate_index=replicate_index,
            frozen_input_seal_sha256="a" * 64,
            issue_id="issue-01",
            construction_stratum_id="stratum-a",
        )
    return Stage4SeedContext(
        purpose=purpose,
        evaluation_id="formal-validation",
        partition_role=partition_role,
        replicate_index=replicate_index,
        frozen_input_seal_sha256="a" * 64,
    )


def test_typed_time_and_space_mappings_are_complete_deterministic_bijections() -> None:
    issue_ids = ("issue-03", "issue-01", "issue-04", "issue-02")
    time_context = _context("time_permutation", partition_role="assessment")
    first_time = build_time_bijection(issue_ids, context=time_context)
    second_time = build_time_bijection(issue_ids, context=time_context)
    assert first_time == second_time
    assert set(first_time.donor_issue_ids) == set(issue_ids)
    assert len(first_time.mapping_sha256) == 64

    states = ("state-c", "state-a", "state-b")
    coordinates = ((30.0, 3.0), (10.0, 1.0), (20.0, 2.0))
    space_context = _context("space_permutation", partition_role="fit")
    first_space = build_space_bijection(states, coordinates, context=space_context)
    second_space = build_space_bijection(
        tuple(reversed(states)), tuple(reversed(coordinates)), context=space_context
    )
    assert first_space == second_space
    assert set(first_space.donor_state_ids) == set(states)
    assert first_space.coordinate_multiset_verified
    assert len(first_space.mapping_sha256) == 64

    identical = build_space_bijection(("a", "b"), ((1.0, 2.0), (1.0, 2.0)), context=space_context)
    assert identical.no_effect
    assert identical.moved_entity_row_count == 0
    assert identical.donor_state_ids == ("a", "b")


def test_permutation_failure_rule_counts_infinity_and_never_drops_mapping() -> None:
    replications = tuple(
        PermutationReplication(
            index,
            None if 1 <= index <= 10 else (0.2 if index == 0 else -0.1),
            converged=not 1 <= index <= 10,
        )
        for index in range(1_000)
    )
    result = reduce_permutation_test(0.1, replications)
    assert result.null_statistics[1:11] == (math.inf,) * 10
    assert result.null_greater_or_equal_count == 11
    assert result.scientific_failure_count == 10
    assert result.monte_carlo_p_value == pytest.approx(12 / 1_001)
    assert result.denominator == 1_001
    assert result.status == "passed"

    above_limit = list(replications)
    above_limit[11] = PermutationReplication(11, float("nan"))
    assert reduce_permutation_test(0.1, above_limit).status == "evidence_insufficient"

    with pytest.raises(InfrastructureInterruption, match="resume"):
        reduce_permutation_test(0.1, replications[:-1])


def test_point_process_information_gain_recall_and_percentile_interval() -> None:
    gain = information_gain_per_physical_event(
        event_ids=("e1", "e2"),
        candidate_event_log_intensities=(2.0, 3.0),
        comparator_event_log_intensities=(1.0, 1.0),
        candidate_integrated_intensity=4.0,
        comparator_integrated_intensity=5.0,
    )
    assert gain == pytest.approx(2.0)
    assert (
        information_gain_per_physical_event(
            event_ids=(),
            candidate_event_log_intensities=(),
            comparator_event_log_intensities=(),
            candidate_integrated_intensity=1.0,
            comparator_integrated_intensity=1.0,
        )
        is None
    )
    candidate = strict_recall(("e1", "e2", "unsupported"), ("e1", "e2"))
    background = strict_recall(("e1", "e2", "unsupported"), ("e1",))
    assert candidate.value == pytest.approx(2 / 3)
    assert same_area_recall_gain_percentage_points(candidate, background) == pytest.approx(100 / 3)
    interval = percentile_interval((0.0, 1.0, 2.0, 3.0))
    assert interval.lower == pytest.approx(0.075)
    assert interval.upper == pytest.approx(2.925)


def test_five_window_membership_bootstrap_preserves_every_horizon_marginal() -> None:
    events = (
        EventHorizonMembership("e1", "M5_6", (True, True, False, False, False)),
        EventHorizonMembership("e2", "M5_6", (True, True, False, False, False)),
        EventHorizonMembership("e3", "M5_6", (False, False, True, True, True)),
        EventHorizonMembership("e4", "M6_plus", (True, True, True, True, True)),
        EventHorizonMembership("e5", "M6_plus", (True, True, True, True, True)),
    )
    context = _context("bootstrap")
    first = stratified_five_horizon_bootstrap_indices(events, context=context)
    second = stratified_five_horizon_bootstrap_indices(events, context=context)
    assert first == second
    assert len(first.sampled_indices) == len(events)
    assert sum(first.multiplicities) == len(events)
    assert first.marginal_counts_by_horizon == ((7, 4), (30, 4), (90, 3), (180, 3), (365, 3))
    assert ("M5_6:00111", 1) in first.stratum_sample_sizes


def _candidate_profile(*, maximum_hits: int = 4) -> tuple[AlarmAreaPoint, ...]:
    points: list[AlarmAreaPoint] = []
    for budget in frozen_same_recall_budget_grid():
        if budget < 100_000:
            hits = 0
        elif budget < 400_000:
            hits = min(2, maximum_hits)
        else:
            hits = maximum_hits
        points.append(AlarmAreaPoint(budget, float(budget), hits))
    return tuple(points)


def test_same_recall_area_has_explicit_evaluable_zero_and_unreachable_branches() -> None:
    evaluable = same_recall_union_area_relative_reduction(
        background_primary=AlarmAreaPoint(600_000, 600_000.0, 4),
        candidate_budget_profile=_candidate_profile(),
    )
    assert evaluable.status == "evaluable"
    assert evaluable.candidate_budget_km2 == 400_000
    assert evaluable.relative_reduction == pytest.approx(1 / 3)
    assert evaluable.pass_eligible

    zero = same_recall_union_area_relative_reduction(
        background_primary=AlarmAreaPoint(600_000, 600_000.0, 0),
        candidate_budget_profile=_candidate_profile(),
    )
    assert zero.status == "zero_reference_hits"
    assert zero.bootstrap_numeric_value == 0.0
    assert not zero.pass_eligible

    unreachable = same_recall_union_area_relative_reduction(
        background_primary=AlarmAreaPoint(600_000, 600_000.0, 4),
        candidate_budget_profile=_candidate_profile(maximum_hits=3),
    )
    assert unreachable.status == "unreachable_at_maximum_budget"
    assert unreachable.bootstrap_numeric_value == 0.0
    assert not unreachable.pass_eligible


def _g2_evidence(**overrides: object) -> G2Evidence:
    values: dict[str, object] = {
        "unique_union_event_count": 30,
        "event_count_by_horizon": {7: 12, 30: 10, 90: 8},
        "information_gain_by_horizon": {7: 0.3, 30: 0.2, 90: 0.1},
        "macro_information_gain_interval": ConfidenceInterval(0.05, 0.35),
        "time_permutation_p_value": 0.01,
        "dynamic_minus_coverage_interval": ConfidenceInterval(0.01, 0.15),
        "same_area_recall_gain_percentage_points": 6.0,
        "same_area_recall_gain_interval": ConfidenceInterval(1.0, 11.0),
        "same_recall_area_relative_reduction": None,
        "same_recall_area_interval": None,
        "same_recall_branch_evaluable": False,
        "permutation_scientific_failure_fraction": 0.0,
    }
    values.update(overrides)
    return G2Evidence(**values)  # type: ignore[arg-type]


def test_g2_positive_negative_and_insufficient_outcomes_are_distinct() -> None:
    passed = evaluate_g2(_g2_evidence())
    assert passed.status == "passed"

    failed = evaluate_g2(
        _g2_evidence(
            information_gain_by_horizon={7: 0.3, 30: -0.1, 90: 0.1},
            same_area_recall_gain_percentage_points=1.0,
        )
    )
    assert failed.status == "failed"
    assert "all_horizon_information_gain_positive" in failed.reasons
    assert "same_area_practical_branch" in failed.reasons

    insufficient = evaluate_g2(_g2_evidence(unique_union_event_count=19))
    assert insufficient.status == "evidence_insufficient"
    assert insufficient.reasons == ("fewer_than_20_unique_union_events",)

    permutation_insufficient = evaluate_g2(
        _g2_evidence(permutation_scientific_failure_fraction=0.011)
    )
    assert permutation_insufficient.status == "evidence_insufficient"
    assert permutation_insufficient.reasons == (
        "time_permutation_scientific_failure_fraction_above_0_01",
    )

    space_permutation_insufficient = evaluate_g2(
        _g2_evidence(space_permutation_scientific_failure_fraction=0.011)
    )
    assert space_permutation_insufficient.status == "evidence_insufficient"
    assert space_permutation_insufficient.reasons == (
        "space_permutation_scientific_failure_fraction_above_0_01",
    )


def _fold(fold_id: str, value: float, *, zero_event: bool = False) -> G3FoldEvidence:
    return G3FoldEvidence(
        fold_id,
        {7: 0 if zero_event else 2, 30: 2, 90: 1},
        {7: value, 30: value, 90: value},
    )


def test_g3_and_adoption_matrix_do_not_conflate_failure_with_insufficiency() -> None:
    g3_pass = evaluate_g3((_fold("f1", 0.1), _fold("f2", -0.1), _fold("f3", 0.2)))
    g3_fail = evaluate_g3((_fold("f1", 0.1), _fold("f2", -0.2), _fold("f3", -0.1)))
    g3_insufficient = evaluate_g3(
        (_fold("f1", 0.1, zero_event=True), _fold("f2", 0.1), _fold("f3", 0.1))
    )
    assert g3_pass.status == "passed"
    assert g3_fail.status == "failed"
    assert g3_insufficient.status == "evidence_insufficient"

    g2_pass = evaluate_g2(_g2_evidence())
    g2_fail = evaluate_g2(_g2_evidence(information_gain_by_horizon={7: -0.1, 30: -0.1, 90: -0.1}))
    assert (
        apply_preregistered_adoption_matrix(
            dynamic_g2=g2_pass,
            dynamic_g3=g3_pass,
            snapshot_equivalent_g2=g2_pass,
        ).choice
        == "dynamic"
    )
    assert (
        apply_preregistered_adoption_matrix(
            dynamic_g2=g2_pass,
            dynamic_g3=g3_fail,
            snapshot_equivalent_g2=g2_pass,
        ).choice
        == "snapshot"
    )
    negative = apply_preregistered_adoption_matrix(
        dynamic_g2=g2_fail,
        dynamic_g3=g3_pass,
        snapshot_equivalent_g2=g2_pass,
    )
    assert negative.choice == "background_only"
    assert negative.status == "credible_negative"
    insufficient_gate = GateOutcome("G2", "evidence_insufficient", (), ("synthetic",))
    insufficient = apply_preregistered_adoption_matrix(
        dynamic_g2=insufficient_gate,
        dynamic_g3=g3_pass,
        snapshot_equivalent_g2=None,
    )
    assert insufficient.choice == "background_only"
    assert insufficient.status == "evidence_insufficient"
