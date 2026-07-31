from __future__ import annotations

import math

import pytest

from seismoflux.stage2p.evaluation import (
    BONFERRONI_LOWER_QUANTILE,
    BONFERRONI_UPPER_QUANTILE,
    BootstrapReplicateDenominatorError,
    EvidenceInsufficientError,
    ModelId,
    TargetObservation,
    assess_confirmatory_gate,
    evaluate_science_targets,
    evaluate_science_targets_outcome,
    evaluation_to_dict,
)


def _observation(
    *,
    event_id: str,
    horizon: int,
    cluster: str,
    region: str,
    p0_density: float = 1.0,
    p1_density: float = 2.0,
    pp_density: float = 1.0,
    p0_hit: bool = False,
    p1_hit: bool = True,
    pp_hit: bool = False,
    in_support: bool = True,
) -> TargetObservation:
    densities: dict[ModelId, float | None] = (
        {"P0": p0_density, "P1": p1_density, "PP": pp_density}
        if in_support
        else {"P0": None, "P1": None, "PP": None}
    )
    return TargetObservation(
        event_id=event_id,
        horizon_days=horizon,
        cluster_id=cluster,
        region_id=region,
        in_support=in_support,
        model_densities=densities,
        alarm_hits={
            "P0": p0_hit if in_support else False,
            "P1": p1_hit if in_support else False,
            "PP": pp_hit if in_support else False,
        },
    )


def _clear_positive_sample() -> tuple[TargetObservation, ...]:
    return tuple(
        _observation(
            event_id=f"e-{horizon}-{cluster_index:02d}",
            horizon=horizon,
            cluster=f"c-{cluster_index:02d}",
            region=f"r-{cluster_index:02d}",
        )
        for horizon in (7, 30, 90)
        for cluster_index in range(10)
    )


def test_equal_horizon_macro_recall_information_gain_and_gate() -> None:
    result = evaluate_science_targets(_clear_positive_sample())

    for horizon in (7, 30, 90):
        metrics = result.horizons[horizon]
        assert metrics.model_recall["P0"].strict_event_recall == 0.0
        assert metrics.model_recall["P1"].strict_event_recall == 1.0
        assert metrics.model_recall["P1"].independent_cluster_recall == 1.0
        assert metrics.model_recall["P1"].independent_region_recall == 1.0
        assert metrics.comparisons["P1_minus_P0"].recall_gain_percentage_points == 100.0
        assert metrics.comparisons["P1_minus_P0"].information_gain_nats_per_event == pytest.approx(
            math.log(2.0)
        )

    p0 = result.comparisons["P1_minus_P0"]
    pp = result.comparisons["P1_minus_PP"]
    assert p0.macro_recall_gain_percentage_points == 100.0
    assert pp.macro_recall_gain_percentage_points == 100.0
    assert p0.macro_information_gain_nats_per_event == pytest.approx(math.log(2.0))
    assert pp.macro_information_gain_nats_per_event == pytest.approx(math.log(2.0))
    assert p0.recall_interval.lower == pytest.approx(100.0)
    assert p0.information_gain_interval.lower == pytest.approx(math.log(2.0))
    assert assess_confirmatory_gate(result).status == "passed"


def test_macro_is_equal_horizon_not_event_weighted() -> None:
    rows = [
        _observation(
            event_id="h7",
            horizon=7,
            cluster="shared",
            region="r1",
            p1_density=math.e,
        ),
        _observation(
            event_id="h30",
            horizon=30,
            cluster="shared",
            region="r1",
            p1_density=math.exp(2.0),
        ),
    ]
    rows.extend(
        _observation(
            event_id=f"h90-{index}",
            horizon=90,
            cluster="shared",
            region="r1",
            p1_density=math.exp(3.0),
        )
        for index in range(5)
    )
    result = evaluate_science_targets(rows, bootstrap_replicates=8)

    assert result.comparisons["P1_minus_P0"].macro_information_gain_nats_per_event == pytest.approx(
        2.0
    )


def test_outside_support_is_miss_for_recall_and_excluded_from_information_gain() -> None:
    rows = list(_clear_positive_sample())
    rows[0] = _observation(
        event_id=rows[0].event_id,
        horizon=7,
        cluster=rows[0].cluster_id,
        region=rows[0].region_id,
        in_support=False,
    )
    result = evaluate_science_targets(rows, bootstrap_replicates=32)

    horizon = result.horizons[7]
    assert horizon.model_recall["P1"].event_count == 10
    assert horizon.model_recall["P1"].event_hit_count == 9
    assert horizon.model_recall["P1"].strict_event_recall == pytest.approx(0.9)
    assert horizon.comparisons["P1_minus_P0"].supported_event_count == 9
    assert horizon.comparisons["P1_minus_P0"].information_gain_nats_per_event == pytest.approx(
        math.log(2.0)
    )


def test_bootstrap_is_deterministic_paired_and_uses_frozen_quantiles() -> None:
    rows = _clear_positive_sample()
    first = evaluate_science_targets(rows)
    second = evaluate_science_targets(rows)

    assert first.bootstrap_seed == 147
    assert first.bootstrap_replicates == 2_000
    assert first.bootstrap_endpoint_order == (
        "P1_minus_P0_information_gain",
        "P1_minus_P0_strict_recall",
        "P1_minus_PP_information_gain",
        "P1_minus_PP_strict_recall",
    )
    assert first.bootstrap_samples == second.bootstrap_samples
    assert (
        first.comparisons["P1_minus_P0"].recall_interval.lower_quantile == BONFERRONI_LOWER_QUANTILE
    )
    assert (
        first.comparisons["P1_minus_P0"].recall_interval.upper_quantile == BONFERRONI_UPPER_QUANTILE
    )


def test_largest_positive_removal_uses_original_denominator_and_id_tie_break() -> None:
    result = evaluate_science_targets(_clear_positive_sample(), bootstrap_replicates=32)
    diagnostics = result.comparisons["P1_minus_P0"].removal_diagnostics

    recall_region = next(
        item
        for item in diagnostics
        if item.endpoint.endswith("_recall") and item.group_kind == "region"
    )
    information_cluster = next(
        item
        for item in diagnostics
        if item.endpoint.endswith("_information_gain") and item.group_kind == "cluster"
    )
    assert recall_region.removed_id == "r-00"
    assert recall_region.removed_contribution == pytest.approx(10.0)
    assert recall_region.residual_with_original_denominator == pytest.approx(90.0)
    assert information_cluster.removed_id == "c-00"
    assert information_cluster.removed_contribution == pytest.approx(math.log(2.0) / 10.0)
    assert information_cluster.residual_with_original_denominator == pytest.approx(
        math.log(2.0) * 0.9
    )


def test_gate_reports_sample_insufficiency_before_effect_failure() -> None:
    rows = tuple(
        _observation(
            event_id=f"e-{horizon}",
            horizon=horizon,
            cluster="one-cluster",
            region="one-region",
        )
        for horizon in (7, 30, 90)
    )
    result = evaluate_science_targets(rows, bootstrap_replicates=8)
    assessment = assess_confirmatory_gate(result)

    assert assessment.status == "evidence_insufficient"
    assert assessment.reasons == ("fewer_than_20_unique_events",)


def test_heterogeneous_horizon_bootstrap_zero_denominator_is_structured_no_redraw() -> None:
    rows = (
        _observation(
            event_id="h7-only",
            horizon=7,
            cluster="c0",
            region="r0",
        ),
        _observation(
            event_id="h30-only",
            horizon=30,
            cluster="c1",
            region="r1",
        ),
        _observation(
            event_id="h90-only",
            horizon=90,
            cluster="c2",
            region="r2",
        ),
    )

    with pytest.raises(BootstrapReplicateDenominatorError) as raised:
        evaluate_science_targets(
            rows,
            bootstrap_replicates=2,
            bootstrap_seed=147,
        )
    assert raised.value.replicate_index == 1
    assert raised.value.horizon_days == 7
    assert raised.value.all_event_denominator == 0
    assert raised.value.supported_event_denominator == 0
    assert "no redraw is allowed" in str(raised.value)

    first = evaluate_science_targets_outcome(
        rows,
        bootstrap_replicates=2,
        bootstrap_seed=147,
    )
    second = evaluate_science_targets_outcome(
        rows,
        bootstrap_replicates=2,
        bootstrap_seed=147,
    )
    assert first == second
    assert first.status == "evidence_insufficient"
    assert first.evaluation is None
    assert first.reason_code == "bootstrap_replicate_zero_horizon_denominator"
    assert first.failed_bootstrap_replicate_index == 1
    assert first.failed_horizon_days == 7
    assert first.bootstrap_redraw_performed is False


def test_structured_outcome_returns_completed_evaluation_when_all_replicates_work() -> None:
    outcome = evaluate_science_targets_outcome(
        _clear_positive_sample(),
        bootstrap_replicates=8,
        bootstrap_seed=147,
    )

    assert outcome.status == "evaluated"
    assert outcome.evaluation is not None
    assert outcome.reason_code is None
    assert outcome.failed_bootstrap_replicate_index is None
    assert outcome.bootstrap_redraw_performed is False


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="outside support"):
        TargetObservation(
            event_id="e",
            horizon_days=7,
            cluster_id="c",
            region_id="r",
            in_support=False,
            model_densities={"P0": None, "P1": None, "PP": None},
            alarm_hits={"P0": False, "P1": True, "PP": False},
        )

    duplicate = _observation(event_id="duplicate", horizon=7, cluster="c", region="r")
    with pytest.raises(ValueError, match="unique within each horizon"):
        evaluate_science_targets(
            (
                duplicate,
                duplicate,
                _observation(event_id="h30", horizon=30, cluster="c", region="r"),
                _observation(event_id="h90", horizon=90, cluster="c", region="r"),
            ),
            bootstrap_replicates=8,
        )

    with pytest.raises(EvidenceInsufficientError, match="all 7, 30, and 90"):
        evaluate_science_targets((duplicate,), bootstrap_replicates=8)


def test_report_is_json_ready() -> None:
    result = evaluate_science_targets(_clear_positive_sample(), bootstrap_replicates=8)
    payload = evaluation_to_dict(result)

    assert payload["bootstrap_seed"] == 147
    assert payload["comparisons"]["P1_minus_P0"]["macro_recall_gain_percentage_points"] == 100.0
