from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from seismoflux.stage2s.evaluation import (
    BOOTSTRAP_ENTROPY,
    BOOTSTRAP_REPLICATIONS,
    CONTRASTS,
    HORIZONS,
    METRICS,
    MODEL_IDS,
    BootstrapFamilies,
    CellScore,
    Contrast,
    EventBlock,
    LatencyMetrics,
    MetricKey,
    Model,
    RegionContribution,
    RegionRobustness,
    SequenceClosureEvidence,
    SequenceDiagnostic,
    SequenceEvent,
    Stage2SEvaluationInvalid,
    Stage2SEvidenceInsufficient,
    bootstrap_draw_indices,
    bootstrap_families,
    build_sequence_closure_evidence,
    compute_region_robustness,
    compute_sequence_diagnostic,
    decide_stage2s,
    descriptive_sp_minus_s0_point_estimates,
    evaluate_stage2s_gate,
    score_fold_horizon,
)

FOLDS = (1, 2, 3)


def _metric_values(value: float) -> dict[MetricKey, float]:
    return {(contrast, metric): value for contrast in CONTRASTS for metric in METRICS}


def _zero_compensators() -> dict[tuple[Contrast, int, int], float]:
    return {
        (contrast, fold_index, horizon): 0.0
        for contrast in CONTRASTS
        for fold_index in FOLDS
        for horizon in HORIZONS
    }


def _bootstrap_events() -> tuple[EventBlock, ...]:
    events: list[EventBlock] = []
    base_time = datetime(2022, 1, 1, tzinfo=UTC)
    for fold_index in FOLDS:
        for event_index in range(10):
            ig_terms: dict[tuple[Contrast, int], float] = {}
            recall_terms: dict[tuple[Contrast, int], float] = {}
            for horizon_index, horizon in enumerate(HORIZONS):
                positive_count = (fold_index - 1) * len(HORIZONS) + horizon_index + 1
                contribution = float(event_index < positive_count)
                for contrast in CONTRASTS:
                    ig_terms[(contrast, horizon)] = contribution
                    recall_terms[(contrast, horizon)] = contribution
            events.append(
                EventBlock(
                    event_id=f"supported-f{fold_index}-{event_index:02d}",
                    origin_time_utc=base_time
                    + timedelta(days=100 * fold_index, seconds=event_index),
                    fold_index=fold_index,
                    horizons=HORIZONS,
                    supported_ig=True,
                    ig_by_contrast_horizon=ig_terms,
                    recall_by_contrast_horizon=recall_terms,
                )
            )
        for event_index in range(2):
            recall_terms = {
                (contrast, horizon): 0.0 for contrast in CONTRASTS for horizon in HORIZONS
            }
            events.append(
                EventBlock(
                    event_id=f"unsupported-f{fold_index}-{event_index:02d}",
                    origin_time_utc=base_time
                    + timedelta(days=100 * fold_index, hours=1, seconds=event_index),
                    fold_index=fold_index,
                    horizons=HORIZONS,
                    supported_ig=False,
                    ig_by_contrast_horizon={},
                    recall_by_contrast_horizon=recall_terms,
                )
            )
    return tuple(events)


def test_bootstrap_entropy_and_first_five_draws_are_frozen() -> None:
    assert BOOTSTRAP_ENTROPY == 145027427185234188584049408348082519483

    draws = bootstrap_draw_indices(stratum_sizes=[3], replications=5)

    expected = (
        [2, 1, 1],
        [2, 2, 2],
        [0, 0, 1],
        [1, 2, 1],
        [1, 2, 1],
    )
    assert len(draws) == len(expected)
    for replication, expected_draw in zip(draws, expected, strict=True):
        assert len(replication) == 1
        np.testing.assert_array_equal(replication[0], expected_draw)
        assert not replication[0].flags.writeable


def test_bootstrap_uses_2000_paired_event_blocks_and_nine_cell_macros() -> None:
    result = bootstrap_families(
        _bootstrap_events(),
        compensators=_zero_compensators(),
    )

    assert result.entropy_uint128 == BOOTSTRAP_ENTROPY
    assert BOOTSTRAP_REPLICATIONS == 2000
    for fold_index in FOLDS:
        for horizon in HORIZONS:
            assert result.cell_denominators[(fold_index, horizon, "IG")] == 10
            assert result.cell_denominators[(fold_index, horizon, "recall")] == 12

    for contrast in CONTRASTS:
        ig = result.intervals[(contrast, "IG")]
        recall = result.intervals[(contrast, "recall")]
        assert ig.point == pytest.approx(0.5)
        assert recall.point == pytest.approx(5.0 / 12.0)
        assert len(ig.replicates) == BOOTSTRAP_REPLICATIONS
        assert len(recall.replicates) == BOOTSTRAP_REPLICATIONS
        np.testing.assert_allclose(
            np.asarray(recall.replicates),
            np.asarray(ig.replicates) * (5.0 / 6.0),
            rtol=0.0,
            atol=1.0e-15,
        )

    assert result.intervals[("S1_minus_S0", "IG")].replicates == pytest.approx(
        result.intervals[("S1_minus_SP", "IG")].replicates
    )
    assert result.intervals[("S1_minus_S0", "recall")].replicates == pytest.approx(
        result.intervals[("S1_minus_SP", "recall")].replicates
    )

    with pytest.raises(Stage2SEvaluationInvalid, match="exactly 2000"):
        bootstrap_families(
            _bootstrap_events(),
            compensators=_zero_compensators(),
            replications=1999,
        )


def test_score_fold_horizon_recomputes_compensator_and_uses_full_recall_denominator() -> None:
    mass_delta = 5.0e-13
    masses = {
        "S0": np.array([[0.5, 0.5], [0.4, 0.6]]),
        "S1": np.array([[0.5 + mass_delta, 0.5], [0.4 + mass_delta, 0.6]]),
        "SP": np.array([[0.5, 0.5], [0.4, 0.6]]),
    }
    score = score_fold_horizon(
        fold_index=1,
        horizon_days=90,
        event_ids=("e0", "e1", "outside-support-0", "outside-support-1"),
        supported_ig=np.array([True, True, False, False]),
        log_density_by_model={
            "S0": np.array([0.0, 2.0, 0.0, 0.0]),
            "S1": np.array([1.0, 3.0, 0.0, 0.0]),
            "SP": np.array([0.5, 2.0, 0.0, 0.0]),
        },
        hit_by_model={
            "S0": np.array([False, True, False, False]),
            "S1": np.array([True, True, False, False]),
            "SP": np.array([False, False, False, False]),
        },
        operational_mass_by_model=masses,
        shared_rate_per_day=1.0,
    )

    expected_compensator = math.fsum(
        90.0 * math.fsum(float(value) for value in issue_mass) for issue_mass in masses["S1"]
    ) - math.fsum(
        90.0 * math.fsum(float(value) for value in issue_mass) for issue_mass in masses["S0"]
    )
    assert expected_compensator > 0.0
    assert score.issue_count == 2
    assert score.compensator_differences["S1_minus_S0"] == pytest.approx(
        expected_compensator,
        abs=1.0e-18,
    )
    assert score.compensator_differences["S1_minus_SP"] == pytest.approx(
        expected_compensator,
        abs=1.0e-18,
    )
    assert score.information_gain["S1_minus_S0"] == pytest.approx(1.0 - expected_compensator / 2.0)
    assert score.information_gain["S1_minus_SP"] == pytest.approx(0.75 - expected_compensator / 2.0)
    assert score.recall_gain["S1_minus_S0"] == pytest.approx(0.25)
    assert score.recall_gain["S1_minus_SP"] == pytest.approx(0.5)
    np.testing.assert_array_equal(
        score.recall_hit_differences["S1_minus_S0"],
        [1.0, 0.0, 0.0, 0.0],
    )


def test_score_fold_horizon_rejects_compensator_above_tolerance() -> None:
    mass_delta = 5.0e-13
    with pytest.raises(Stage2SEvaluationInvalid, match="compensator differs"):
        score_fold_horizon(
            fold_index=1,
            horizon_days=90,
            event_ids=("e0",),
            supported_ig=np.array([True]),
            log_density_by_model={
                "S0": np.array([0.0]),
                "S1": np.array([1.0]),
                "SP": np.array([0.0]),
            },
            hit_by_model={
                "S0": np.array([False]),
                "S1": np.array([True]),
                "SP": np.array([False]),
            },
            operational_mass_by_model={
                "S0": np.array([[0.5, 0.5]]),
                "S1": np.array([[0.5 + mass_delta, 0.5]]),
                "SP": np.array([[0.5, 0.5]]),
            },
            shared_rate_per_day=3.0,
        )


def test_score_fold_horizon_forces_unsupported_targets_to_miss() -> None:
    with pytest.raises(Stage2SEvaluationInvalid, match="force every unsupported"):
        score_fold_horizon(
            fold_index=1,
            horizon_days=7,
            event_ids=("supported", "unsupported"),
            supported_ig=np.array([True, False]),
            log_density_by_model={
                "S0": np.array([0.0, 0.0]),
                "S1": np.array([1.0, 0.0]),
                "SP": np.array([0.5, 0.0]),
            },
            hit_by_model={
                "S0": np.array([False, False]),
                "S1": np.array([True, True]),
                "SP": np.array([False, False]),
            },
            operational_mass_by_model={
                "S0": np.array([[0.5, 0.5]]),
                "S1": np.array([[0.5, 0.5]]),
                "SP": np.array([[0.5, 0.5]]),
            },
            shared_rate_per_day=0.1,
        )


def _passing_regions() -> tuple[RegionContribution, ...]:
    identities = (
        ("é", 1, 0.35),
        ("z", 1, 0.35),
        ("other", 1, 0.20),
        ("zero-event-with-compensator", 0, 0.10),
        *((f"zero-{index:02d}", 0, 0.0) for index in range(35)),
    )
    return tuple(
        RegionContribution(
            zone_id=zone_id,
            ig_event_count=event_count,
            recall_event_count=event_count,
            contributions=_metric_values(contribution),
        )
        for zone_id, event_count, contribution in identities
    )


def test_region_robustness_requires_39_zone_closure_and_fixed_loro() -> None:
    result = compute_region_robustness(
        _passing_regions(),
        primary_metrics=_metric_values(1.0),
    )

    assert result.passed
    for key in _metric_values(0.0):
        metric_result = result.results[key]
        assert metric_result.event_bearing_zone_count == 3
        assert metric_result.positive_event_bearing_zone_count == 3
        assert metric_result.strongest_positive_zone_id == "z"
        assert metric_result.strongest_positive_contribution == pytest.approx(0.35)
        assert metric_result.leave_strongest_out_residual == pytest.approx(0.65)
        assert metric_result.passed


def test_region_robustness_rejects_additive_closure_failure() -> None:
    regions = list(_passing_regions())
    regions[-1] = RegionContribution(
        zone_id=regions[-1].zone_id,
        ig_event_count=0,
        recall_event_count=0,
        contributions=_metric_values(1.0e-6),
    )

    with pytest.raises(Stage2SEvaluationInvalid, match="regional closure failed"):
        compute_region_robustness(
            regions,
            primary_metrics=_metric_values(1.0),
        )


def test_region_robustness_marks_fewer_than_two_event_bearing_zones_insufficient() -> None:
    regions = tuple(
        RegionContribution(
            zone_id=f"zone-{index:02d}",
            ig_event_count=int(index == 0),
            recall_event_count=int(index == 0),
            contributions=_metric_values(float(index == 0)),
        )
        for index in range(39)
    )

    with pytest.raises(Stage2SEvidenceInsufficient, match="fewer than two"):
        compute_region_robustness(
            regions,
            primary_metrics=_metric_values(1.0),
        )


def test_region_robustness_returns_failed_when_breadth_or_loro_does_not_pass() -> None:
    regions = tuple(
        RegionContribution(
            zone_id=f"zone-{index:02d}",
            ig_event_count=int(index < 2),
            recall_event_count=int(index < 2),
            contributions=_metric_values(float(index == 0)),
        )
        for index in range(39)
    )

    result = compute_region_robustness(
        regions,
        primary_metrics=_metric_values(1.0),
    )

    assert not result.passed
    assert len(result.failures) == 4
    for metric_result in result.results.values():
        assert metric_result.event_bearing_zone_count == 2
        assert metric_result.positive_event_bearing_zone_count == 1
        assert metric_result.leave_strongest_out_residual == pytest.approx(0.0)
        assert not metric_result.passed


def _sequence_event(
    *,
    event_id: str,
    origin_time_utc: datetime,
    longitude: float,
    contribution: float,
) -> SequenceEvent:
    return SequenceEvent(
        event_id=event_id,
        origin_time_utc=origin_time_utc,
        longitude=longitude,
        latitude=0.0,
        contributions=_metric_values(contribution),
        model_hit_contributions={"S0": 0.2, "S1": 0.2, "SP": 0.2},
    )


def _transitive_sequence_events() -> tuple[SequenceEvent, ...]:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    return (
        _sequence_event(
            event_id="b",
            origin_time_utc=start,
            longitude=0.0,
            contribution=0.15,
        ),
        _sequence_event(
            event_id="a",
            origin_time_utc=start + timedelta(days=14),
            longitude=0.6,
            contribution=0.15,
        ),
        _sequence_event(
            event_id="c",
            origin_time_utc=start + timedelta(days=28),
            longitude=1.2,
            contribution=0.10,
        ),
        _sequence_event(
            event_id="d",
            origin_time_utc=start + timedelta(days=14),
            longitude=10.0,
            contribution=0.40,
        ),
        _sequence_event(
            event_id="e",
            origin_time_utc=start + timedelta(days=90),
            longitude=1.2,
            contribution=0.10,
        ),
    )


def _sequence_closure(
    events: tuple[SequenceEvent, ...],
    *,
    residual: float,
) -> SequenceClosureEvidence:
    return SequenceClosureEvidence(
        expected_event_ids=tuple(
            sorted((event.event_id for event in events), key=lambda value: value.encode())
        ),
        global_residual={
            (contrast, metric): (residual if metric == "IG" else 0.0)
            for contrast in CONTRASTS
            for metric in METRICS
        },
        primary_model_recall={
            model: math.fsum(event.model_hit_contributions[model] for event in events)
            for model in MODEL_IDS
        },
    )


def test_sequence_diagnostic_uses_30_day_75km_transitive_components_and_closure() -> None:
    result = compute_sequence_diagnostic(
        (events := _transitive_sequence_events()),
        primary_metrics={
            (contrast, metric): (1.0 if metric == "IG" else 0.9)
            for contrast in CONTRASTS
            for metric in METRICS
        },
        closure=_sequence_closure(events, residual=0.10),
    )

    assert tuple(component.component_id for component in result.components) == ("a", "d", "e")
    assert result.components[0].event_ids == ("a", "b", "c")
    assert result.largest_count_component_id == "a"
    assert {event_id for component in result.components for event_id in component.event_ids} == {
        "a",
        "b",
        "c",
        "d",
        "e",
    }
    for key in _metric_values(0.0):
        assert result.largest_gain_component_id[key] == "a"
        expected_leave = 0.60 if key[1] == "IG" else 0.50
        assert result.leave_largest_gain_out[key] == pytest.approx(expected_leave)
        assert result.leave_largest_count_out[key] == pytest.approx(expected_leave)
    largest = result.components[0]
    assert largest.event_fraction == pytest.approx(3.0 / 5.0)
    assert largest.model_hit_contributions["S1"] == pytest.approx(0.6)
    assert largest.model_hit_fractions["S1"] == pytest.approx(0.6)
    assert largest.ig_fractions["S1_minus_S0"] == pytest.approx(0.4)
    assert not result.claim_limited


def test_sequence_diagnostic_rejects_additive_closure_failure() -> None:
    with pytest.raises(Stage2SEvaluationInvalid, match="sequence additive closure failed"):
        compute_sequence_diagnostic(
            (events := _transitive_sequence_events()),
            primary_metrics={
                (contrast, metric): (1.01 if metric == "IG" else 0.9)
                for contrast in CONTRASTS
                for metric in METRICS
            },
            closure=_sequence_closure(events, residual=0.10),
        )


def test_sequence_diagnostic_rejects_missing_or_duplicate_event_contributions() -> None:
    events = _transitive_sequence_events()
    closure = _sequence_closure(events, residual=0.10)
    primary = {
        (contrast, metric): (1.0 if metric == "IG" else 0.9)
        for contrast in CONTRASTS
        for metric in METRICS
    }

    with pytest.raises(Stage2SEvaluationInvalid, match="event union"):
        compute_sequence_diagnostic(
            events[:-1],
            primary_metrics=primary,
            closure=closure,
        )
    with pytest.raises(Stage2SEvaluationInvalid, match="non-empty and unique"):
        compute_sequence_diagnostic(
            (*events, events[0]),
            primary_metrics=primary,
            closure=closure,
        )


def test_sequence_reports_distinct_largest_count_and_largest_gain_leave_outs() -> None:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    clustered = tuple(
        _sequence_event(
            event_id=event_id,
            origin_time_utc=start + timedelta(days=offset),
            longitude=0.1 * offset,
            contribution=0.1,
        )
        for event_id, offset in (("a", 0), ("b", 1), ("c", 2))
    )
    strongest = _sequence_event(
        event_id="z",
        origin_time_utc=start,
        longitude=10.0,
        contribution=0.6,
    )
    events = (*clustered, strongest)

    result = compute_sequence_diagnostic(
        events,
        primary_metrics={
            (contrast, metric): (1.0 if metric == "IG" else 0.9)
            for contrast in CONTRASTS
            for metric in METRICS
        },
        closure=_sequence_closure(events, residual=0.1),
    )

    assert result.largest_count_component_id == "a"
    assert set(result.largest_gain_component_id.values()) == {"z"}
    for contrast in CONTRASTS:
        assert result.leave_largest_count_out[(contrast, "IG")] == pytest.approx(0.7)
        assert result.leave_largest_gain_out[(contrast, "IG")] == pytest.approx(0.4)
    largest_count = next(
        component for component in result.components if component.component_id == "a"
    )
    assert largest_count.event_fraction == pytest.approx(0.75)
    assert largest_count.origin_time_span_days == pytest.approx(2.0)
    assert largest_count.max_pairwise_geodesic_distance_km > 0.0


def test_sequence_diagnostic_limits_claim_without_changing_primary_result() -> None:
    event = _sequence_event(
        event_id="solo",
        origin_time_utc=datetime(2023, 1, 1, tzinfo=UTC),
        longitude=105.0,
        contribution=1.0,
    )

    result = compute_sequence_diagnostic(
        (event,),
        primary_metrics=_metric_values(1.0),
        closure=_sequence_closure((event,), residual=0.0),
    )

    assert result.components[0].event_ids == ("solo",)
    assert result.claim_limited
    assert all(value == pytest.approx(0.0) for value in result.leave_largest_gain_out.values())
    assert (
        result.interpretation_limit
        == "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
    )


def test_decision_priority_is_invalid_then_insufficient_then_failed_then_passed() -> None:
    invalid = decide_stage2s(
        invalid_reasons=("identity",),
        evidence_insufficient_reasons=("sample",),
        failed_reasons=("gate",),
    )
    insufficient = decide_stage2s(
        evidence_insufficient_reasons=("sample",),
        failed_reasons=("gate",),
    )
    failed = decide_stage2s(failed_reasons=("gate",))
    passed = decide_stage2s()

    assert invalid.status == "invalid"
    assert invalid.reasons == ("identity",)
    assert insufficient.status == "evidence_insufficient"
    assert insufficient.reasons == ("sample",)
    assert failed.status == "failed"
    assert failed.reasons == ("gate",)
    assert passed.status == "passed_development_signal"
    assert passed.reasons == ()


def _passing_cell_scores() -> dict[tuple[int, int], CellScore]:
    scores: dict[tuple[int, int], CellScore] = {}
    for fold_index in FOLDS:
        event_ids = tuple(
            [
                *(f"supported-f{fold_index}-{index:02d}" for index in range(10)),
                *(f"unsupported-f{fold_index}-{index:02d}" for index in range(2)),
            ]
        )
        supported = np.asarray([True] * 10 + [False] * 2)
        for horizon_index, horizon in enumerate(HORIZONS):
            positive_count = (fold_index - 1) * len(HORIZONS) + horizon_index + 1
            ig = np.asarray([float(index < positive_count) for index in range(10)])
            recall = np.asarray([float(index < positive_count) for index in range(10)] + [0.0, 0.0])
            candidate_hits = np.asarray(
                [index < positive_count for index in range(10)] + [False, False],
                dtype=np.bool_,
            )
            baseline_hits = np.zeros(12, dtype=np.bool_)
            scores[(fold_index, horizon)] = CellScore(
                fold_index=fold_index,
                horizon_days=horizon,
                issue_count=1,
                event_ids=event_ids,
                supported_ig=supported,
                hit_by_model={
                    "S0": baseline_hits,
                    "S1": candidate_hits,
                    "SP": baseline_hits,
                },
                ig_event_log_ratios={contrast: ig for contrast in CONTRASTS},
                recall_hit_differences={contrast: recall for contrast in CONTRASTS},
                compensator_differences={contrast: 0.0 for contrast in CONTRASTS},
                information_gain={contrast: positive_count / 10.0 for contrast in CONTRASTS},
                recall_gain={contrast: positive_count / 12.0 for contrast in CONTRASTS},
            )
    return scores


def _sequence_for_cells(
    scores: dict[tuple[int, int], CellScore],
) -> SequenceDiagnostic:
    contributions: dict[str, dict[MetricKey, float]] = {}
    model_hits: dict[str, dict[Model, float]] = {}
    for score in scores.values():
        supported_count = int(np.count_nonzero(score.supported_ig))
        recall_count = len(score.event_ids)
        for position, (event_id, supported) in enumerate(
            zip(score.event_ids, score.supported_ig, strict=True)
        ):
            values = contributions.setdefault(event_id, _metric_values(0.0))
            hits = model_hits.setdefault(
                event_id,
                {"S0": 0.0, "S1": 0.0, "SP": 0.0},
            )
            supported_position = int(np.count_nonzero(score.supported_ig[:position]))
            for contrast in CONTRASTS:
                if bool(supported):
                    values[(contrast, "IG")] += (
                        float(score.ig_event_log_ratios[contrast][supported_position])
                        / supported_count
                        / 9.0
                    )
                values[(contrast, "recall")] += (
                    float(score.recall_hit_differences[contrast][position]) / recall_count / 9.0
                )
            for model in ("S0", "S1", "SP"):
                hits[model] += float(score.hit_by_model[model][position]) / recall_count / 9.0
    ordered_ids = sorted(contributions, key=lambda value: value.encode())
    events = tuple(
        SequenceEvent(
            event_id=event_id,
            origin_time_utc=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=40 * index),
            longitude=float(index * 2),
            latitude=0.0,
            contributions=contributions[event_id],
            model_hit_contributions=model_hits[event_id],
        )
        for index, event_id in enumerate(ordered_ids)
    )
    primary = {
        (contrast, metric): math.fsum(
            (score.information_gain[contrast] if metric == "IG" else score.recall_gain[contrast])
            for score in scores.values()
        )
        / 9.0
        for contrast in CONTRASTS
        for metric in METRICS
    }
    return compute_sequence_diagnostic(
        events,
        primary_metrics=primary,
        closure=build_sequence_closure_evidence(scores),
    )


def test_sequence_closure_recomputes_compensator_residual_independently() -> None:
    scores = _passing_cell_scores()
    key = (1, 7)
    original = scores[key]
    scores[key] = replace(
        original,
        compensator_differences={
            "S1_minus_S0": 0.9,
            "S1_minus_SP": -0.45,
        },
    )

    closure = build_sequence_closure_evidence(scores)

    assert closure.global_residual[("S1_minus_S0", "IG")] == pytest.approx(-0.01)
    assert closure.global_residual[("S1_minus_SP", "IG")] == pytest.approx(0.005)
    assert closure.global_residual[("S1_minus_S0", "recall")] == 0.0
    assert closure.global_residual[("S1_minus_SP", "recall")] == 0.0


def test_sp_minus_s0_is_descriptive_linear_point_only() -> None:
    values: dict[MetricKey, float] = {
        ("S1_minus_S0", "IG"): 0.30,
        ("S1_minus_S0", "recall"): 0.08,
        ("S1_minus_SP", "IG"): 0.12,
        ("S1_minus_SP", "recall"): 0.03,
    }

    descriptive = descriptive_sp_minus_s0_point_estimates(values)

    assert descriptive["IG"] == pytest.approx(0.18)
    assert descriptive["recall"] == pytest.approx(0.05)


def _gate_regions() -> tuple[RegionContribution, ...]:
    values: list[RegionContribution] = []
    for index in range(39):
        contributions: dict[MetricKey, float] = {}
        for contrast in CONTRASTS:
            contributions[(contrast, "IG")] = 0.2 if index < 2 else 0.1 if index == 2 else 0.0
            contributions[(contrast, "recall")] = (
                1.0 / 6.0 if index < 2 else 1.0 / 12.0 if index == 2 else 0.0
            )
        values.append(
            RegionContribution(
                zone_id=f"zone-{index:02d}",
                ig_event_count=int(index < 3),
                recall_event_count=int(index < 3),
                contributions=contributions,
            )
        )
    return tuple(values)


def _passing_gate_evidence() -> tuple[
    dict[tuple[int, int], CellScore],
    BootstrapFamilies,
    RegionRobustness,
    tuple[LatencyMetrics, LatencyMetrics],
]:
    cells = _passing_cell_scores()
    bootstrap = bootstrap_families(
        _bootstrap_events(),
        compensators=_zero_compensators(),
    )
    regional = compute_region_robustness(
        _gate_regions(),
        primary_metrics={(contrast, "IG"): 0.5 for contrast in CONTRASTS}
        | {(contrast, "recall"): 5.0 / 12.0 for contrast in CONTRASTS},
    )
    latency = (
        LatencyMetrics(
            delay_days=1,
            values=_metric_values(0.1),
        ),
        LatencyMetrics(
            delay_days=7,
            values=_metric_values(0.1),
        ),
    )
    return cells, bootstrap, regional, latency


def test_internal_gate_derives_a_complete_passing_decision() -> None:
    cells, bootstrap, regional, latency = _passing_gate_evidence()

    assessment = evaluate_stage2s_gate(
        cells,
        bootstrap=bootstrap,
        regional=regional,
        latency=latency,
        sequence=_sequence_for_cells(cells),
    )

    assert assessment.decision.status == "passed_development_signal"
    assert assessment.supported_event_union_count == 30
    assert assessment.recall_event_union_count == 36
    assert not assessment.claim_limited
    assert assessment.interpretation_limit == "no_sequence_interpretation_limit"
    for contrast in CONTRASTS:
        assert assessment.overall_macros[(contrast, "IG")] == pytest.approx(0.5)
        assert assessment.overall_macros[(contrast, "recall")] == pytest.approx(5.0 / 12.0)


def test_internal_gate_marks_nonpositive_latency_as_evidence_insufficient() -> None:
    cells, bootstrap, regional, latency = _passing_gate_evidence()
    changed = (
        latency[0],
        LatencyMetrics(
            delay_days=7,
            values=_metric_values(0.1) | {("S1_minus_SP", "recall"): 0.0},
        ),
    )

    assessment = evaluate_stage2s_gate(
        cells,
        bootstrap=bootstrap,
        regional=regional,
        latency=changed,
        sequence=_sequence_for_cells(cells),
    )

    assert assessment.decision.status == "evidence_insufficient"
    assert assessment.decision.reasons == ("latency_7d:S1_minus_SP:recall:lte_1e_12",)
