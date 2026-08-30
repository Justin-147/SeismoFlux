from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from seismoflux.p1_b0_r30.core import (
    GRID_CELL_AREA_KM2,
    MAXIMUM_ALARM_AREA_KM2,
    PRIMARY_HORIZON_DAYS,
    AlarmPrefix,
    ClusterScore,
    IssueCandidate,
    RelativeIntensitySurface,
    ScoreSummary,
    SyntheticEvent,
    build_dual_model_forecast,
    build_pending_sequential_reviews,
    cluster_guarded_exposures,
    cluster_target_events,
    make_equal_area_grid,
    ordered_cluster_registry_sha256,
    score_clusters,
    select_guarded_issues,
)
from seismoflux.p1_b0_r30.records import (
    RECORD_TYPES,
    build_record,
    canonical_record_sha256,
    seal_record,
    validate_record_against_schema,
    validate_record_chain,
)
from seismoflux.p1_b0_r30.synthetic import (
    SYNTHETIC_ISSUE_TIME_UTC,
    SYNTHETIC_QUERY_CUTOFF_UTC,
    build_all_synthetic_scenarios,
    build_synthetic_scenario,
    make_synthetic_model_events,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "data" / "contracts" / "p1_prospective_records_v1.json"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _schema() -> dict[str, Any]:
    value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _record_fields(record: dict[str, Any]) -> dict[str, Any]:
    header = {
        "schema_version",
        "record_type",
        "recorded_at_utc",
        "chain_sequence",
        "previous_record_type",
        "previous_record_sha256",
        "content_sha256",
    }
    return {key: value for key, value in record.items() if key not in header}


def _event(
    event_id: str,
    *,
    origin: datetime,
    x_km: float,
    y_km: float,
    magnitude: float = 5.3,
) -> SyntheticEvent:
    return SyntheticEvent(
        event_id=event_id,
        origin_time_utc=origin,
        available_at_utc=origin + timedelta(minutes=1),
        x_km=x_km,
        y_km=y_km,
        magnitude=magnitude,
        source_id="synthetic_ComCat",
    )


def _append_authorized_missed_records(
    *,
    previous_record: dict[str, Any],
    authorization_sha256: object,
    first_scheduled_utc: datetime,
    before_utc: datetime,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous = previous_record
    scheduled = first_scheduled_utc
    while scheduled < before_utc:
        text = scheduled.isoformat().replace("+00:00", "Z")
        missed = build_record(
            "MissedIssueRecord",
            recorded_at_utc=(scheduled + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            previous_record=previous,
            fields={
                "issue_id": f"p1-{scheduled.strftime('%Y%m%dT%H%M%SZ')}",
                "status": "missed_issue",
                "scheduled_issue_time_utc": text,
                "authorization_state": "authorized",
                "authorization_record_sha256": authorization_sha256,
                "reason": "source_snapshot_unavailable_before_T",
                "prediction_generated": False,
                "backfill_forbidden": True,
                "valid_from_remains_fixed": True,
            },
        )
        records.append(missed)
        previous = missed
        scheduled += timedelta(days=7)
    return records


def _record_scores(
    count: int,
    *,
    issue_id: str = "p1-20260916T160000Z",
    scheduled_issue_utc: datetime = datetime(2026, 9, 16, 16, tzinfo=UTC),
) -> tuple[ClusterScore, ...]:
    return tuple(
        ClusterScore(
            issue_id=issue_id,
            cluster_id=f"{issue_id}-cluster-{index:02d}",
            representative_origin_time_utc=scheduled_issue_utc + timedelta(days=1 + index),
            representative_event_id=f"{issue_id}-event-{index:02d}",
            B0_hit=index < 3,
            B0_R30_hit=index < 4,
        )
        for index in range(count)
    )


def _score_registry_map(
    *registries: tuple[ClusterScore, ...],
) -> dict[str, list[dict[str, object]]]:
    return {
        ordered_cluster_registry_sha256(registry): [score.as_mapping() for score in registry]
        for registry in registries
    }


def _base_score_registry_map() -> dict[str, list[dict[str, object]]]:
    return _score_registry_map(_record_scores(10))


def _six_record_chain() -> list[dict[str, Any]]:
    protocol = build_record(
        "ProtocolDefinition",
        recorded_at_utc="2026-08-30T00:00:00Z",
        previous_record=None,
        fields={
            "protocol_id": "p1-b0-r30-prospective-v1",
            "protocol_tag": "v0.2.7-p1-b0-r30-protocol",
            "code_tag": "v0.2.7-p1-b0-r30-code",
            "valid_from_utc": "2026-09-09T16:00:00Z",
            "historical_catalog_cutoff_utc": "2026-07-09T04:25:56Z",
            "source_boundary_manifest_sha256": _sha("source-boundary"),
            "model_manifest_sha256": _sha("model-manifest"),
            "protocol_commit": _git_sha("protocol-commit"),
            "real_issue_authorized": False,
        },
    )
    missed = build_record(
        "MissedIssueRecord",
        recorded_at_utc="2026-09-09T16:05:00Z",
        previous_record=protocol,
        fields={
            "issue_id": "p1-20260909T160000Z",
            "status": "missed_issue",
            "scheduled_issue_time_utc": "2026-09-09T16:00:00Z",
            "authorization_state": "not_authorized",
            "authorization_record_sha256": None,
            "reason": "real_issue_not_authorized_before_T",
            "prediction_generated": False,
            "backfill_forbidden": True,
            "valid_from_remains_fixed": True,
        },
    )
    authorization = build_record(
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-09-10T08:00:00Z",
        previous_record=missed,
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_commit": _git_sha("authorization-commit"),
            "code_commit": _git_sha("code-commit"),
            "remote_verified_at_utc": "2026-09-10T07:59:00Z",
            "authorized_from_scheduled_issue_utc": "2026-09-16T16:00:00Z",
            "real_issue_authorized": True,
        },
    )
    forecasts = [
        {
            "model_id": model_id,
            "relative_intensity_grid_sha256": _sha(f"{model_id}-grid"),
            "alarm_mask_sha256": _sha(f"{model_id}-mask"),
            "alarm_ranking_sha256": _sha(f"{model_id}-ranking"),
            "actual_alarm_area_km2": area,
        }
        for model_id, area in (("B0", 599_900), ("B0_R30", 599_500))
    ]
    forecast = build_record(
        "ForecastIssueRecord",
        recorded_at_utc="2026-09-16T15:59:30Z",
        previous_record=authorization,
        fields={
            "issue_id": "p1-20260916T160000Z",
            "status": "on_time",
            "scheduled_issue_time_utc": "2026-09-16T16:00:00Z",
            "query_cutoff_utc": "2026-09-16T15:45:00Z",
            "forecast_created_at_utc": "2026-09-16T15:55:00Z",
            "publication_completed_at_utc": "2026-09-16T15:59:00Z",
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "model_manifest_sha256": _sha("model-manifest"),
            "source_boundary_manifest_sha256": _sha("source-boundary"),
            "source_snapshot_sha256": _sha("source-snapshot"),
            "code_commit": _git_sha("code-commit"),
            "forecasts": forecasts,
            "static_svg_sha256": _sha("static-svg"),
            "offline_interactive_html_sha256": _sha("interactive-html"),
            "B0_reference_area_km2": 599_900,
            "B0_R30_next_complete_cell_area_km2": 500,
            "actual_area_difference_km2": 400,
            "area_fairness_status": "passed",
            "original_artifacts_immutable": True,
        },
    )
    continuity_missed = _append_authorized_missed_records(
        previous_record=forecast,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 9, 23, 16, tzinfo=UTC),
        before_utc=datetime(2026, 11, 16, 16, tzinfo=UTC),
    )
    base_scores = _record_scores(10)
    base_registry_sha = ordered_cluster_registry_sha256(base_scores)
    truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2026-11-16T16:00:00Z",
        previous_record=continuity_missed[-1],
        fields={
            "issue_id": "p1-20260916T160000Z",
            "horizon_days": 30,
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "source_snapshot_sha256": _sha("truth-source"),
            "status": "mature_truth",
            "mature_after_utc": "2026-11-15T16:00:00Z",
            "truth_fetched_at_utc": "2026-11-16T16:00:00Z",
            "target_event_count": 12,
            "independent_cluster_count": 10,
            "cluster_assignment_sha256": _sha("cluster-assignment"),
            "exposure_cluster_registry_sha256": base_registry_sha,
            "magnitude_minimum": 5.0,
            "magnitude_maximum_exclusive": 6.0,
        },
    )
    review_value = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=base_scores), elapsed_months=2.2
    )[0]
    review = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2026-11-17T00:00:00Z",
        previous_record=truth,
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **review_value.as_mapping(),
        },
    )
    return [
        protocol,
        missed,
        authorization,
        forecast,
        *continuity_missed,
        truth,
        review,
    ]


def _chain_landmarks(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    truth_index = next(
        index
        for index, record in enumerate(records)
        if record["record_type"] == "TruthSnapshotRecord"
    )
    return (
        records[0],
        records[1],
        records[2],
        records[3],
        records[4:truth_index],
        records[truth_index],
        records[truth_index + 1],
    )


def _build_first_issue_zero_secondary_truth(
    *, previous_record: dict[str, Any], truth_template: dict[str, Any]
) -> dict[str, Any]:
    empty_scores: tuple[ClusterScore, ...] = ()
    fields = _record_fields(truth_template)
    fields.update(
        {
            "horizon_days": 90,
            "source_snapshot_sha256": _sha("truth-source-90"),
            "mature_after_utc": "2027-01-14T16:00:00Z",
            "truth_fetched_at_utc": "2027-01-15T16:00:00Z",
            "target_event_count": 0,
            "independent_cluster_count": 0,
            "cluster_assignment_sha256": _sha("cluster-assignment-90"),
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(empty_scores),
        }
    )
    return build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2027-01-15T16:00:00Z",
        previous_record=previous_record,
        fields=fields,
    )


def test_three_in_memory_visual_scenarios_recover_positive_zero_and_negative() -> None:
    scenarios = build_all_synthetic_scenarios()
    assert [scenario.scenario_id for scenario in scenarios] == [
        "positive",
        "zero",
        "negative",
    ]
    assert [scenario.observed_direction for scenario in scenarios] == [
        "positive",
        "zero",
        "negative",
    ]
    assert [(item.score.B0_hit_clusters, item.score.B0_R30_hit_clusters) for item in scenarios] == [
        (6, 10),
        (5, 5),
        (10, 6),
    ]
    for scenario in scenarios:
        assert scenario.score.cluster_count == 10
        assert scenario.forecast.B0_reference_area_km2 == MAXIMUM_ALARM_AREA_KM2
        assert scenario.forecast.actual_area_difference_km2 == 0.0
        assert len(scenario.reviews) == 1
        assert scenario.reviews[0].review_trigger == "cluster_10"
        mapping = scenario.as_mapping()
        assert set(mapping) >= {
            "scenario_id",
            "label",
            "expected_direction",
            "interpretation",
            "query_cutoff_utc",
            "grid",
            "targets",
            "models",
            "comparison",
        }
        assert mapping["value_semantics"] == "relative_intensity_not_absolute_probability"


def test_empty_recent_is_a_fourth_exact_fallback_rehearsal() -> None:
    scenario = build_synthetic_scenario("zero", empty_recent=True)
    assert scenario.scenario_id == "empty_recent_fallback"
    assert scenario.observed_direction == "zero"
    assert (scenario.score.B0_hit_clusters, scenario.score.B0_R30_hit_clusters) == (5, 5)
    fallback = scenario.forecast
    assert fallback.recent_fallback_to_B0 is True
    assert fallback.R30.active_event_count == 0
    assert np.array_equal(fallback.B0.relative_intensity, fallback.B0_R30.relative_intensity)
    assert fallback.B0_alarm.selected_cell_ids == fallback.B0_R30_alarm.selected_cell_ids


def test_query_cutoff_is_causal_and_recent_window_is_left_open_right_closed() -> None:
    grid = make_equal_area_grid()
    q = SYNTHETIC_QUERY_CUTOFF_UTC
    events = (
        SyntheticEvent(
            "old",
            q - timedelta(days=100),
            q - timedelta(days=100) + timedelta(minutes=1),
            100,
            100,
            4.2,
            "synthetic_history",
        ),
        SyntheticEvent(
            "left-boundary",
            q - timedelta(days=30),
            q - timedelta(days=30) + timedelta(minutes=1),
            200,
            200,
            4.2,
            "synthetic_ComCat",
        ),
        SyntheticEvent("at-q", q, q, 700, 700, 4.2, "synthetic_ComCat"),
        SyntheticEvent(
            "late-available",
            q - timedelta(days=1),
            q + timedelta(seconds=1),
            650,
            650,
            4.2,
            "synthetic_ComCat",
        ),
        SyntheticEvent(
            "future-origin",
            q + timedelta(seconds=1),
            q + timedelta(seconds=1),
            600,
            600,
            4.2,
            "synthetic_ComCat",
        ),
    )
    forecast = build_dual_model_forecast(
        events,
        grid,
        issue_id="synthetic-cutoff",
        scheduled_issue_time_utc=q + timedelta(minutes=15),
    )
    assert forecast.B0.active_event_count == 3
    assert forecast.R30.active_event_count == 1
    expected = 0.75 * forecast.B0.relative_intensity + 0.25 * forecast.R30.relative_intensity
    assert np.allclose(forecast.B0_R30.relative_intensity, expected, atol=0.0, rtol=1e-15)


def test_B0_excludes_pre_1970_and_outside_support_events() -> None:
    q = SYNTHETIC_QUERY_CUTOFF_UTC
    grid = make_equal_area_grid()
    events = (
        SyntheticEvent(
            "valid",
            datetime(2010, 1, 1, tzinfo=UTC),
            datetime(2010, 1, 2, tzinfo=UTC),
            100,
            100,
            4.2,
            "synthetic_history",
        ),
        SyntheticEvent(
            "pre-1970",
            datetime(1960, 1, 1, tzinfo=UTC),
            datetime(1960, 1, 2, tzinfo=UTC),
            200,
            200,
            4.2,
            "synthetic_history",
        ),
        SyntheticEvent(
            "outside-support",
            datetime(2011, 1, 1, tzinfo=UTC),
            datetime(2011, 1, 2, tzinfo=UTC),
            -500,
            -500,
            4.2,
            "synthetic_history",
        ),
    )
    forecast = build_dual_model_forecast(
        events,
        grid,
        issue_id="synthetic-support",
        scheduled_issue_time_utc=q + timedelta(minutes=15),
    )
    assert forecast.B0.active_event_count == 1

    invalid_source_boundary = (
        *events,
        SyntheticEvent(
            "late-local",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 1, 0, 1, tzinfo=UTC),
            300,
            300,
            4.2,
            "synthetic_history",
        ),
    )
    with pytest.raises(ValueError, match="frozen local cutoff"):
        build_dual_model_forecast(
            invalid_source_boundary,
            grid,
            issue_id="synthetic-support",
            scheduled_issue_time_utc=q + timedelta(minutes=15),
        )


def test_source_cutover_dedup_is_deterministic_one_to_one_with_local_anchor() -> None:
    cutoff = datetime(2026, 7, 9, 4, 25, 56, tzinfo=UTC)
    grid = make_equal_area_grid()
    local = SyntheticEvent(
        "local-anchor",
        cutoff - timedelta(seconds=1),
        cutoff - timedelta(seconds=1),
        300,
        300,
        4.2,
        "synthetic_history",
        longitude=100.0,
        latitude=30.0,
    )
    duplicate = SyntheticEvent(
        "comcat-duplicate",
        cutoff + timedelta(seconds=1),
        cutoff + timedelta(seconds=2),
        300,
        300,
        4.3,
        "synthetic_ComCat",
        longitude=100.0,
        latitude=30.0,
    )
    forecast = build_dual_model_forecast(
        (duplicate, local),
        grid,
        issue_id="synthetic-dedup",
        scheduled_issue_time_utc=SYNTHETIC_ISSUE_TIME_UTC,
    )
    assert forecast.B0.active_event_count == 1

    outside_time_tolerance = replace(
        duplicate,
        event_id="comcat-not-duplicate",
        origin_time_utc=cutoff + timedelta(seconds=301),
        available_at_utc=cutoff + timedelta(seconds=302),
    )
    unmatched = build_dual_model_forecast(
        (local, outside_time_tolerance),
        grid,
        issue_id="synthetic-dedup",
        scheduled_issue_time_utc=SYNTHETIC_ISSUE_TIME_UTC,
    )
    assert unmatched.B0.active_event_count == 2


def test_source_cutover_dedup_ignores_ineligible_candidate_before_matching() -> None:
    cutoff = datetime(2026, 7, 9, 4, 25, 56, tzinfo=UTC)
    grid = make_equal_area_grid()
    local = SyntheticEvent(
        "local-anchor",
        cutoff - timedelta(seconds=1),
        cutoff - timedelta(seconds=1),
        300,
        300,
        4.2,
        "synthetic_history",
        longitude=100.0,
        latitude=30.0,
    )
    unavailable_but_closer = SyntheticEvent(
        "comcat-unavailable",
        cutoff + timedelta(seconds=1),
        SYNTHETIC_QUERY_CUTOFF_UTC + timedelta(seconds=1),
        300,
        300,
        4.2,
        "synthetic_ComCat",
        longitude=100.0,
        latitude=30.0,
    )
    available_duplicate = SyntheticEvent(
        "comcat-available",
        cutoff + timedelta(seconds=2),
        cutoff + timedelta(seconds=3),
        300,
        300,
        4.2,
        "synthetic_ComCat",
        longitude=100.0,
        latitude=30.0,
    )
    forecast = build_dual_model_forecast(
        (local, unavailable_but_closer, available_duplicate),
        grid,
        issue_id="synthetic-dedup-causal",
        scheduled_issue_time_utc=SYNTHETIC_ISSUE_TIME_UTC,
    )
    assert forecast.B0.active_event_count == 1


def test_alarm_is_complete_prefix_and_challenger_cannot_use_more_area() -> None:
    scenario = build_synthetic_scenario("positive")
    forecast = scenario.forecast
    assert len(forecast.B0_alarm.selected_cell_ids) == 960
    assert len(forecast.B0_R30_alarm.selected_cell_ids) == 960
    assert forecast.B0_alarm.selected_cell_ids == forecast.B0_alarm.ranked_cell_ids[:960]
    assert forecast.B0_R30_alarm.selected_cell_ids == forecast.B0_R30_alarm.ranked_cell_ids[:960]
    assert forecast.B0_R30_alarm.next_complete_cell_area_km2 == GRID_CELL_AREA_KM2

    with pytest.raises(ValueError, match="unbroken prefix"):
        AlarmPrefix(
            model_id="B0",
            ranked_cell_ids=("a", "b", "c"),
            selected_cell_ids=("a", "c"),
            actual_area_km2=2.0,
            area_cap_km2=2.0,
            next_complete_cell_area_km2=1.0,
        )


def test_forecast_rejects_wrong_mixture_and_surface_inconsistent_ranking() -> None:
    forecast = build_synthetic_scenario("positive").forecast
    false_mixture = RelativeIntensitySurface(
        "B0_R30",
        np.array(forecast.B0.relative_intensity, copy=True),
        forecast.B0.active_event_count,
    )
    false_mixture_alarm = AlarmPrefix(
        model_id="B0_R30",
        ranked_cell_ids=forecast.B0_alarm.ranked_cell_ids,
        selected_cell_ids=forecast.B0_alarm.selected_cell_ids,
        actual_area_km2=forecast.B0_alarm.actual_area_km2,
        area_cap_km2=forecast.B0_alarm.actual_area_km2,
        next_complete_cell_area_km2=forecast.B0_alarm.next_complete_cell_area_km2,
    )
    with pytest.raises(ValueError, match="frozen 0.75"):
        replace(
            forecast,
            B0_R30=false_mixture,
            B0_R30_alarm=false_mixture_alarm,
        )

    reversed_ids = tuple(reversed(forecast.B0_alarm.ranked_cell_ids))
    reversed_alarm = AlarmPrefix(
        model_id="B0",
        ranked_cell_ids=reversed_ids,
        selected_cell_ids=reversed_ids[: len(forecast.B0_alarm.selected_cell_ids)],
        actual_area_km2=forecast.B0_alarm.actual_area_km2,
        area_cap_km2=MAXIMUM_ALARM_AREA_KM2,
        next_complete_cell_area_km2=forecast.B0_alarm.next_complete_cell_area_km2,
    )
    with pytest.raises(ValueError, match="derived from its relative-intensity surface"):
        replace(forecast, B0_alarm=reversed_alarm)

    one_cell_B0 = AlarmPrefix(
        model_id="B0",
        ranked_cell_ids=forecast.B0_alarm.ranked_cell_ids,
        selected_cell_ids=forecast.B0_alarm.ranked_cell_ids[:1],
        actual_area_km2=GRID_CELL_AREA_KM2,
        area_cap_km2=MAXIMUM_ALARM_AREA_KM2,
        next_complete_cell_area_km2=GRID_CELL_AREA_KM2,
    )
    one_cell_challenger = AlarmPrefix(
        model_id="B0_R30",
        ranked_cell_ids=forecast.B0_R30_alarm.ranked_cell_ids,
        selected_cell_ids=forecast.B0_R30_alarm.ranked_cell_ids[:1],
        actual_area_km2=GRID_CELL_AREA_KM2,
        area_cap_km2=GRID_CELL_AREA_KM2,
        next_complete_cell_area_km2=GRID_CELL_AREA_KM2,
    )
    with pytest.raises(ValueError, match="largest complete-cell prefix"):
        replace(
            forecast,
            B0_alarm=one_cell_B0,
            B0_R30_alarm=one_cell_challenger,
        )

    extra_cell = forecast.B0_R30_alarm.ranked_cell_ids[960]
    over_alarm = AlarmPrefix(
        model_id="B0_R30",
        ranked_cell_ids=forecast.B0_R30_alarm.ranked_cell_ids,
        selected_cell_ids=(*forecast.B0_R30_alarm.selected_cell_ids, extra_cell),
        actual_area_km2=MAXIMUM_ALARM_AREA_KM2 + GRID_CELL_AREA_KM2,
        area_cap_km2=MAXIMUM_ALARM_AREA_KM2 + GRID_CELL_AREA_KM2,
        next_complete_cell_area_km2=GRID_CELL_AREA_KM2,
    )
    with pytest.raises(ValueError, match="cap must equal|never use more area"):
        replace(forecast, B0_R30_alarm=over_alarm)


def test_guard_gap_is_applied_separately_to_30_and_90_day_on_time_issues() -> None:
    origin = datetime(2026, 9, 9, 16, tzinfo=UTC)
    issues = tuple(
        IssueCandidate(
            issue_id=f"issue-{days:03d}",
            scheduled_issue_time_utc=origin + timedelta(days=days),
            status="missed_issue" if days == 59 else "on_time",
        )
        for days in (0, 30, 59, 60, 61, 120, 180, 240)
    )
    selected_30 = select_guarded_issues(issues, horizon_days=30)
    selected_90 = select_guarded_issues(issues, horizon_days=90)
    assert [item.scheduled_issue_time_utc for item in selected_30] == [
        origin,
        origin + timedelta(days=60),
        origin + timedelta(days=120),
        origin + timedelta(days=180),
        origin + timedelta(days=240),
    ]
    assert [item.scheduled_issue_time_utc for item in selected_90] == [
        origin,
        origin + timedelta(days=120),
        origin + timedelta(days=240),
    ]


def test_guarded_exposure_clustering_binds_exact_selected_windows_and_rejects_duplicates() -> None:
    origin = SYNTHETIC_ISSUE_TIME_UTC
    issues = tuple(
        IssueCandidate(
            issue_id=f"guard-{days:03d}",
            scheduled_issue_time_utc=origin + timedelta(days=days),
            status="on_time",
        )
        for days in (0, 7, 60, 120)
    )
    selected_ids = ("guard-000", "guard-060", "guard-120")
    events = {
        issue_id: (
            _event(
                f"target-{index}",
                origin=origin + timedelta(days=(0, 60, 120)[index] + 1),
                x_km=100 + 100 * index,
                y_km=100,
            ),
        )
        for index, issue_id in enumerate(selected_ids)
    }
    fetched = {
        issue_id: origin + timedelta(days=(0, 60, 120)[index] + 61)
        for index, issue_id in enumerate(selected_ids)
    }
    clusters = cluster_guarded_exposures(
        issues,
        events,
        fetched,
        horizon_days=30,
        grid=make_equal_area_grid(),
    )
    assert len(clusters) == 3
    with pytest.raises(ValueError, match="exactly the guard-selected"):
        cluster_guarded_exposures(
            issues,
            {**events, "guard-007": ()},
            fetched,
            horizon_days=30,
            grid=make_equal_area_grid(),
        )
    duplicate_events = dict(events)
    duplicate_events["guard-060"] = (
        _event(
            "target-0",
            origin=origin + timedelta(days=61),
            x_km=200,
            y_km=100,
        ),
    )
    with pytest.raises(ValueError, match="across selected exposures"):
        cluster_guarded_exposures(
            issues,
            duplicate_events,
            fetched,
            horizon_days=30,
            grid=make_equal_area_grid(),
        )


def test_window_clustering_is_transitive_and_uses_stable_earliest_representative() -> None:
    issue = SYNTHETIC_ISSUE_TIME_UTC
    events = (
        _event("z-event", origin=issue + timedelta(days=1), x_km=100, y_km=100),
        _event("a-event", origin=issue + timedelta(days=1), x_km=170, y_km=100),
        _event("c-event", origin=issue + timedelta(days=2), x_km=240, y_km=100),
        _event("m6-excluded", origin=issue + timedelta(days=3), x_km=500, y_km=500, magnitude=6.0),
        _event("late-excluded", origin=issue + timedelta(days=31), x_km=600, y_km=600),
    )
    clusters = cluster_target_events(
        events,
        issue_id="synthetic-transitive",
        issue_time_utc=issue,
        horizon_days=30,
        truth_fetched_at_utc=issue + timedelta(days=61),
        grid=make_equal_area_grid(),
    )
    assert len(clusters) == 1
    assert clusters[0].member_event_ids == ("a-event", "c-event", "z-event")
    assert clusters[0].representative.event_id == "a-event"


def test_truth_requires_maturity_filters_late_availability_and_uses_WGS84_distance() -> None:
    issue = SYNTHETIC_ISSUE_TIME_UTC
    grid = make_equal_area_grid()
    on_time = _event("on-time", origin=issue + timedelta(days=1), x_km=100, y_km=100)
    late = SyntheticEvent(
        "late-available",
        issue + timedelta(days=2),
        issue + timedelta(days=62),
        200,
        200,
        5.3,
        "synthetic_ComCat",
    )
    with pytest.raises(ValueError, match="cannot be read before"):
        cluster_target_events(
            (on_time,),
            issue_id="mature-issue",
            issue_time_utc=issue,
            horizon_days=30,
            truth_fetched_at_utc=issue + timedelta(days=59),
            grid=grid,
        )
    clusters = cluster_target_events(
        (on_time, late),
        issue_id="mature-issue",
        issue_time_utc=issue,
        horizon_days=30,
        truth_fetched_at_utc=issue + timedelta(days=61),
        grid=grid,
    )
    assert len(clusters) == 1
    assert clusters[0].member_event_ids == ("on-time",)

    geographically_far = (
        SyntheticEvent(
            "geo-a",
            issue + timedelta(days=1),
            issue + timedelta(days=1, minutes=1),
            100,
            100,
            5.3,
            "synthetic_ComCat",
            longitude=100.0,
            latitude=30.0,
        ),
        SyntheticEvent(
            "geo-b",
            issue + timedelta(days=1),
            issue + timedelta(days=1, minutes=1),
            170,
            100,
            5.3,
            "synthetic_ComCat",
            longitude=102.0,
            latitude=30.0,
        ),
    )
    separate = cluster_target_events(
        geographically_far,
        issue_id="geodesic-issue",
        issue_time_utc=issue,
        horizon_days=30,
        truth_fetched_at_utc=issue + timedelta(days=61),
        grid=grid,
    )
    assert len(separate) == 2


def test_truth_rejects_non_ComCat_source_fail_closed() -> None:
    issue = SYNTHETIC_ISSUE_TIME_UTC
    invalid_truth = replace(
        _event("local-truth", origin=issue + timedelta(days=1), x_km=100, y_km=100),
        source_id="synthetic_history",
    )
    with pytest.raises(ValueError, match="only from synthetic_ComCat"):
        cluster_target_events(
            (invalid_truth,),
            issue_id="synthetic-truth-source",
            issue_time_utc=issue,
            horizon_days=30,
            truth_fetched_at_utc=issue + timedelta(days=61),
            grid=make_equal_area_grid(),
        )


def test_score_rejects_cross_issue_forecast_target_mismatch() -> None:
    scenario = build_synthetic_scenario("positive")
    wrong_issue = replace(scenario.target_clusters[0], issue_id="other-issue")
    with pytest.raises(ValueError, match="exact forecast issue_id"):
        score_clusters(scenario.forecast, (wrong_issue,), horizon_days=30)


def _scores(count: int) -> tuple[ClusterScore, ...]:
    issue = datetime(2026, 9, 9, 16, tzinfo=UTC)
    values: list[ClusterScore] = []
    for index in range(count):
        if index < 10 or index >= 20:
            B0_hit, challenger_hit = False, True
        else:
            B0_hit, challenger_hit = True, False
        values.append(
            ClusterScore(
                issue_id=f"issue-{index:02d}",
                cluster_id=f"cluster-{index:02d}",
                representative_origin_time_utc=issue + timedelta(days=61 * index),
                representative_event_id=f"event-{index:02d}",
                B0_hit=B0_hit,
                B0_R30_hit=challenger_hit,
            )
        )
    return tuple(values)


def test_sequential_looks_use_exact_stable_first_10_20_30_cluster_prefixes() -> None:
    score_30 = ScoreSummary(horizon_days=30, scores=_scores(30))
    reviews = build_pending_sequential_reviews(score_30, elapsed_months=18.0)
    assert [review.review_trigger for review in reviews] == [
        "cluster_10",
        "cluster_20",
        "cluster_30",
    ]
    assert [review.cumulative_cluster_count for review in reviews] == [10, 20, 30]
    assert [review.look_sequence for review in reviews] == [1, 2, 3]
    assert [(review.B0_hit_clusters, review.B0_R30_hit_clusters) for review in reviews] == [
        (0, 10),
        (10, 10),
        (10, 20),
    ]
    first_ten_only = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=_scores(10)), elapsed_months=6.0
    )[0]
    assert (
        first_ten_only.selected_cluster_prefix_sha256 == reviews[0].selected_cluster_prefix_sha256
    )
    assert (
        first_ten_only.ordered_cluster_registry_sha256 != reviews[0].ordered_cluster_registry_sha256
    )
    assert reviews[0].decision == "continue_accumulation"
    assert reviews[1].decision == "continue_accumulation"
    assert reviews[2].decision != "continue_accumulation"
    assert (
        build_pending_sequential_reviews(
            ScoreSummary(horizon_days=30, scores=_scores(35)),
            elapsed_months=20.0,
            completed_reviews=reviews,
        )
        == ()
    )

    valid_continuation = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=_scores(20)),
        elapsed_months=12.0,
        completed_reviews=(first_ten_only,),
    )
    assert [item.review_trigger for item in valid_continuation] == ["cluster_20"]

    earlier = tuple(
        ClusterScore(
            issue_id=f"earlier-issue-{index:02d}",
            cluster_id=f"earlier-cluster-{index:02d}",
            representative_origin_time_utc=datetime(2000, 1, 1, tzinfo=UTC) + timedelta(days=index),
            representative_event_id=f"earlier-event-{index:02d}",
            B0_hit=False,
            B0_R30_hit=True,
        )
        for index in range(10)
    )
    reordered = ScoreSummary(horizon_days=30, scores=tuple((*earlier, *_scores(10))))
    with pytest.raises(ValueError, match="frozen cluster prefix or result changed"):
        build_pending_sequential_reviews(
            reordered,
            elapsed_months=12.0,
            completed_reviews=(first_ten_only,),
        )

    flipped_first_ten = tuple(
        replace(score, B0_hit=not score.B0_hit, B0_R30_hit=not score.B0_R30_hit)
        for score in _scores(10)
    )
    rescored = ScoreSummary(
        horizon_days=30,
        scores=(*flipped_first_ten, *_scores(20)[10:]),
    )
    with pytest.raises(ValueError, match="frozen cluster prefix or result changed"):
        build_pending_sequential_reviews(
            rescored,
            elapsed_months=12.0,
            completed_reviews=(first_ten_only,),
        )


def test_terminal_batch_catches_up_pending_looks_in_order_and_scores_are_canonical() -> None:
    at_25 = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=_scores(25)), elapsed_months=36.0
    )
    assert [review.review_trigger for review in at_25] == [
        "cluster_10",
        "cluster_20",
        "time_36_months",
    ]
    assert [review.look_sequence for review in at_25] == [1, 2, 3]
    at_30 = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=_scores(30)), elapsed_months=36.0
    )
    assert [review.review_trigger for review in at_30] == [
        "cluster_10",
        "cluster_20",
        "cluster_30",
    ]
    with pytest.raises(ValueError, match="stable order"):
        ScoreSummary(horizon_days=30, scores=tuple(reversed(_scores(2))))
    with pytest.raises(ValueError, match="globally unique"):
        ScoreSummary(horizon_days=30, scores=(_scores(1)[0], _scores(1)[0]))


def test_zero_cluster_36_month_review_uses_null_effect_and_interval() -> None:
    terminal = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=PRIMARY_HORIZON_DAYS, scores=()),
        elapsed_months=36.0,
    )
    assert len(terminal) == 1
    review = terminal[0]
    assert review.review_trigger == "time_36_months"
    assert review.cumulative_cluster_count == 0
    assert review.B0_hit_clusters == review.B0_R30_hit_clusters == 0
    assert review.recall_gain_percentage_points is None
    assert review.sequentially_adjusted_interval_lower is None
    assert review.sequentially_adjusted_interval_upper is None
    assert review.decision == "report_evidence_insufficient_at_final_review"
    with pytest.raises(ValueError, match="30-day primary"):
        build_pending_sequential_reviews(
            ScoreSummary(horizon_days=90, scores=()),
            elapsed_months=36.0,
        )


def test_six_record_types_form_one_schema_valid_canonical_chain() -> None:
    records = _six_record_chain()
    record_types = tuple(record["record_type"] for record in records)
    assert record_types[:4] == (
        "ProtocolDefinition",
        "MissedIssueRecord",
        "RealIssueAuthorizationRecord",
        "ForecastIssueRecord",
    )
    assert record_types[-2:] == ("TruthSnapshotRecord", "SequentialReviewRecord")
    assert set(RECORD_TYPES) == set(record_types)
    validate_record_chain(
        records,
        _schema(),
        score_registries_by_sha256=_base_score_registry_map(),
    )
    for record in records:
        assert record["content_sha256"] == canonical_record_sha256(record)

    tampered = [dict(record) for record in records]
    truth_index = record_types.index("TruthSnapshotRecord")
    tampered[truth_index]["previous_record_sha256"] = "f" * 64
    tampered[truth_index] = seal_record(tampered[truth_index])
    with pytest.raises(ValueError, match="previous_record_sha256"):
        validate_record_chain(
            tampered,
            _schema(),
            score_registries_by_sha256=_base_score_registry_map(),
        )

    _, _, _, _, _, truth, review = _chain_landmarks(records)
    forged_fields = _record_fields(review)
    forged_fields.update(
        {
            "B0_hit_clusters": 10,
            "B0_R30_hit_clusters": 0,
            "recall_gain_percentage_points": -100.0,
            "sequentially_adjusted_interval_lower": -100.0,
            "sequentially_adjusted_interval_upper": -100.0,
            "decision": "continue_accumulation",
        }
    )
    forged_review = build_record(
        "SequentialReviewRecord",
        recorded_at_utc=cast(str, review["recorded_at_utc"]),
        previous_record=truth,
        fields=forged_fields,
    )
    with pytest.raises(ValueError, match="hit counts differ"):
        validate_record_chain(
            [*records[:-1], forged_review],
            _schema(),
            score_registries_by_sha256=_base_score_registry_map(),
        )


def test_schema_and_chain_reject_forecast_before_authorization_and_hash_tampering() -> None:
    records = _six_record_chain()
    protocol = records[0]
    forecast = records[3]
    forbidden = build_record(
        "ForecastIssueRecord",
        recorded_at_utc="2026-09-09T15:59:30Z",
        previous_record=protocol,
        fields={
            key: value
            for key, value in forecast.items()
            if key
            not in {
                "schema_version",
                "record_type",
                "recorded_at_utc",
                "chain_sequence",
                "previous_record_type",
                "previous_record_sha256",
                "content_sha256",
            }
        },
    )
    with pytest.raises(ValueError, match="schema validation failed|cannot precede"):
        validate_record_chain([protocol, forbidden], _schema())

    malformed = dict(records[-1])
    malformed["B0_R30_hit_clusters"] = 999
    with pytest.raises(ValueError, match="content_sha256"):
        validate_record_against_schema(malformed, _schema())


def test_authorization_remote_verification_cannot_precede_recorded_protocol() -> None:
    records = _six_record_chain()
    protocol, missed, authorization = records[:3]
    late_protocol = build_record(
        "ProtocolDefinition",
        recorded_at_utc="2026-09-09T16:00:00Z",
        previous_record=None,
        fields=_record_fields(protocol),
    )
    with pytest.raises(ValueError, match="recorded before valid_from"):
        validate_record_chain([late_protocol], _schema())

    authorization_fields = _record_fields(authorization)
    authorization_fields["remote_verified_at_utc"] = "2026-08-29T23:59:00Z"
    impossible_authorization = build_record(
        "RealIssueAuthorizationRecord",
        recorded_at_utc=cast(str, authorization["recorded_at_utc"]),
        previous_record=missed,
        fields=authorization_fields,
    )
    with pytest.raises(ValueError, match="must follow the recorded protocol"):
        validate_record_chain([protocol, missed, impossible_authorization], _schema())


def test_missed_issue_reason_must_match_authorization_state() -> None:
    records = _six_record_chain()
    protocol, missed, authorization = records[:3]
    unauthorized_fields = _record_fields(missed)
    unauthorized_fields["reason"] = "source_snapshot_unavailable_before_T"
    unauthorized_with_source_reason = build_record(
        "MissedIssueRecord",
        recorded_at_utc=cast(str, missed["recorded_at_utc"]),
        previous_record=protocol,
        fields=unauthorized_fields,
    )
    with pytest.raises(ValueError, match="reason is inconsistent"):
        validate_record_chain([protocol, unauthorized_with_source_reason], _schema())

    authorized_with_auth_reason = build_record(
        "MissedIssueRecord",
        recorded_at_utc="2026-09-16T16:05:00Z",
        previous_record=authorization,
        fields={
            "issue_id": "p1-20260916T160000Z",
            "status": "missed_issue",
            "scheduled_issue_time_utc": "2026-09-16T16:00:00Z",
            "authorization_state": "authorized",
            "authorization_record_sha256": authorization["content_sha256"],
            "reason": "real_issue_not_authorized_before_T",
            "prediction_generated": False,
            "backfill_forbidden": True,
            "valid_from_remains_fixed": True,
        },
    )
    with pytest.raises(ValueError, match="reason is inconsistent"):
        validate_record_chain(
            [protocol, missed, authorization, authorized_with_auth_reason], _schema()
        )


def test_record_chain_rejects_early_missed_and_calendar_or_issue_id_mismatch() -> None:
    records = _six_record_chain()
    protocol = records[0]
    early_missed = build_record(
        "MissedIssueRecord",
        recorded_at_utc="2026-09-09T15:59:00Z",
        previous_record=protocol,
        fields=_record_fields(records[1]),
    )
    with pytest.raises(ValueError, match="cannot be written before"):
        validate_record_chain([protocol, early_missed], _schema())

    authorization = records[2]
    mismatched_fields = _record_fields(records[3])
    mismatched_fields["issue_id"] = "p1-20260917T160000Z"
    mismatch = build_record(
        "ForecastIssueRecord",
        recorded_at_utc=cast(str, records[3]["recorded_at_utc"]),
        previous_record=authorization,
        fields=mismatched_fields,
    )
    with pytest.raises(ValueError, match="exactly encode"):
        validate_record_chain([protocol, records[1], authorization, mismatch], _schema())


def test_record_chain_recomputes_area_maturity_truth_support_and_review_order() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, review = _chain_landmarks(records)

    unfair_fields = _record_fields(forecast)
    unfair_forecasts = [
        dict(item) for item in cast(list[dict[str, Any]], unfair_fields["forecasts"])
    ]
    unfair_forecasts[1]["actual_alarm_area_km2"] = 600_000
    unfair_fields["forecasts"] = unfair_forecasts
    unfair = build_record(
        "ForecastIssueRecord",
        recorded_at_utc=cast(str, forecast["recorded_at_utc"]),
        previous_record=authorization,
        fields=unfair_fields,
    )
    with pytest.raises(ValueError, match="may not use more"):
        validate_record_chain([protocol, missed, authorization, unfair], _schema())

    early_truth_fields = _record_fields(truth)
    early_truth_fields["mature_after_utc"] = "2026-10-17T16:00:00Z"
    early_truth_fields["truth_fetched_at_utc"] = "2026-10-18T16:00:00Z"
    early_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2026-10-18T16:00:00Z",
        previous_record=forecast,
        fields=early_truth_fields,
    )
    with pytest.raises(ValueError, match="T plus horizon plus 30"):
        validate_record_chain([protocol, missed, authorization, forecast, early_truth], _schema())

    premature_review = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2026-11-17T00:00:00Z",
        previous_record=forecast,
        fields=_record_fields(review),
    )
    with pytest.raises(ValueError, match="due guard-selected truth"):
        validate_record_chain(
            [protocol, missed, authorization, forecast, premature_review], _schema()
        )

    duplicate_review = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2026-11-18T00:00:00Z",
        previous_record=review,
        fields=_record_fields(review),
    )
    with pytest.raises(ValueError, match="ordered 10, 20, 30"):
        validate_record_chain(
            [*records, duplicate_review],
            _schema(),
            score_registries_by_sha256=_base_score_registry_map(),
        )


def test_record_chain_guard_gap_prevents_overlapping_truths_from_triggering_a_look() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, _, truth, _ = _chain_landmarks(records)
    second_forecast_fields = _record_fields(forecast)
    second_forecast_fields.update(
        {
            "issue_id": "p1-20260923T160000Z",
            "scheduled_issue_time_utc": "2026-09-23T16:00:00Z",
            "query_cutoff_utc": "2026-09-23T15:45:00Z",
            "forecast_created_at_utc": "2026-09-23T15:55:00Z",
            "publication_completed_at_utc": "2026-09-23T15:59:00Z",
        }
    )
    second_forecast = build_record(
        "ForecastIssueRecord",
        recorded_at_utc="2026-09-23T15:59:30Z",
        previous_record=forecast,
        fields=second_forecast_fields,
    )
    continuity_before_first_truth = _append_authorized_missed_records(
        previous_record=second_forecast,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 9, 30, 16, tzinfo=UTC),
        before_utc=datetime(2026, 11, 16, 16, tzinfo=UTC),
    )
    first_scores = _record_scores(5)
    first_truth_fields = _record_fields(truth)
    first_truth_fields.update(
        {
            "target_event_count": 5,
            "independent_cluster_count": 5,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(first_scores),
        }
    )
    first_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2026-11-16T16:00:00Z",
        previous_record=continuity_before_first_truth[-1],
        fields=first_truth_fields,
    )
    second_scores = _record_scores(
        5,
        issue_id="p1-20260923T160000Z",
        scheduled_issue_utc=datetime(2026, 9, 23, 16, tzinfo=UTC),
    )
    second_truth_fields = _record_fields(truth)
    second_truth_fields.update(
        {
            "issue_id": "p1-20260923T160000Z",
            "mature_after_utc": "2026-11-22T16:00:00Z",
            "truth_fetched_at_utc": "2026-11-23T16:00:00Z",
            "target_event_count": 5,
            "independent_cluster_count": 5,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(second_scores),
        }
    )
    continuity_before_second_truth = _append_authorized_missed_records(
        previous_record=first_truth,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 11, 18, 16, tzinfo=UTC),
        before_utc=datetime(2026, 11, 23, 16, tzinfo=UTC),
    )
    second_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2026-11-23T16:00:00Z",
        previous_record=continuity_before_second_truth[-1],
        fields=second_truth_fields,
    )
    with pytest.raises(ValueError, match="not guard-selected"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                second_forecast,
                *continuity_before_first_truth,
                first_truth,
                *continuity_before_second_truth,
                second_truth,
            ],
            _schema(),
            score_registries_by_sha256=_score_registry_map(first_scores, second_scores),
        )


def test_record_chain_rejects_duplicate_truth_event_across_selected_exposures() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, review = _chain_landmarks(records)
    second_issue_time = datetime(2026, 11, 18, 16, tzinfo=UTC)
    second_forecast_fields = _record_fields(forecast)
    second_forecast_fields.update(
        {
            "issue_id": "p1-20261118T160000Z",
            "scheduled_issue_time_utc": "2026-11-18T16:00:00Z",
            "query_cutoff_utc": "2026-11-18T15:45:00Z",
            "forecast_created_at_utc": "2026-11-18T15:55:00Z",
            "publication_completed_at_utc": "2026-11-18T15:59:00Z",
        }
    )
    second_forecast = build_record(
        "ForecastIssueRecord",
        recorded_at_utc="2026-11-18T15:59:30Z",
        previous_record=review,
        fields=second_forecast_fields,
    )
    second_continuity = _append_authorized_missed_records(
        previous_record=second_forecast,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 11, 25, 16, tzinfo=UTC),
        before_utc=datetime(2027, 1, 18, 16, tzinfo=UTC),
    )
    base_scores = _record_scores(10)
    duplicate_scores = tuple(
        replace(
            score,
            issue_id="p1-20261118T160000Z",
            cluster_id=f"p1-20261118T160000Z-cluster-{index:02d}",
            representative_origin_time_utc=second_issue_time + timedelta(days=1 + index),
        )
        for index, score in enumerate(base_scores)
    )
    duplicate_truth_fields = _record_fields(truth)
    duplicate_truth_fields.update(
        {
            "issue_id": "p1-20261118T160000Z",
            "mature_after_utc": "2027-01-17T16:00:00Z",
            "truth_fetched_at_utc": "2027-01-18T16:00:00Z",
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(duplicate_scores),
        }
    )
    first_secondary_truth = _build_first_issue_zero_secondary_truth(
        previous_record=second_continuity[-1], truth_template=truth
    )
    duplicate_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2027-01-18T16:00:00Z",
        previous_record=first_secondary_truth,
        fields=duplicate_truth_fields,
    )
    with pytest.raises(ValueError, match="may not be counted in multiple selected exposures"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                *continuity,
                truth,
                review,
                second_forecast,
                *second_continuity,
                first_secondary_truth,
                duplicate_truth,
            ],
            _schema(),
            score_registries_by_sha256=_score_registry_map(base_scores, duplicate_scores, ()),
        )


@pytest.mark.parametrize("cluster_count", [10, 30])
def test_crossed_cluster_review_cannot_be_omitted_before_next_issue(
    cluster_count: int,
) -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, _ = _chain_landmarks(records)
    scores = _record_scores(cluster_count)
    truth_fields = _record_fields(truth)
    truth_fields.update(
        {
            "target_event_count": cluster_count,
            "independent_cluster_count": cluster_count,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(scores),
        }
    )
    threshold_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=cast(str, truth["recorded_at_utc"]),
        previous_record=continuity[-1],
        fields=truth_fields,
    )
    next_forecast_fields = _record_fields(forecast)
    next_forecast_fields.update(
        {
            "issue_id": "p1-20261118T160000Z",
            "scheduled_issue_time_utc": "2026-11-18T16:00:00Z",
            "query_cutoff_utc": "2026-11-18T15:45:00Z",
            "forecast_created_at_utc": "2026-11-18T15:55:00Z",
            "publication_completed_at_utc": "2026-11-18T15:59:00Z",
        }
    )
    next_forecast = build_record(
        "ForecastIssueRecord",
        recorded_at_utc="2026-11-18T15:59:30Z",
        previous_record=threshold_truth,
        fields=next_forecast_fields,
    )
    with pytest.raises(ValueError, match="review thresholds must be recorded"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                *continuity,
                threshold_truth,
                next_forecast,
            ],
            _schema(),
            score_registries_by_sha256=_score_registry_map(scores),
        )


def test_due_primary_and_secondary_truth_cannot_be_silently_omitted() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, review = _chain_landmarks(records)
    omitted_primary_next_issue = build_record(
        "MissedIssueRecord",
        recorded_at_utc="2026-11-18T16:05:00Z",
        previous_record=continuity[-1],
        fields={
            "issue_id": "p1-20261118T160000Z",
            "status": "missed_issue",
            "scheduled_issue_time_utc": "2026-11-18T16:00:00Z",
            "authorization_state": "authorized",
            "authorization_record_sha256": authorization["content_sha256"],
            "reason": "source_snapshot_unavailable_before_T",
            "prediction_generated": False,
            "backfill_forbidden": True,
            "valid_from_remains_fixed": True,
        },
    )
    with pytest.raises(ValueError, match="cannot skip a due guard-selected truth"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                *continuity,
                omitted_primary_next_issue,
            ],
            _schema(),
        )

    secondary_continuity = _append_authorized_missed_records(
        previous_record=review,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 11, 18, 16, tzinfo=UTC),
        before_utc=datetime(2027, 1, 21, 16, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="cannot skip a due guard-selected truth"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                *continuity,
                truth,
                review,
                *secondary_continuity,
            ],
            _schema(),
            score_registries_by_sha256=_base_score_registry_map(),
        )


def test_record_chain_rejects_forged_early_36_month_terminal() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, review = _chain_landmarks(records)
    terminal_fields = _record_fields(review)
    terminal_fields.update(
        {
            "review_trigger": "time_36_months",
            "cumulative_cluster_count": 5,
            "elapsed_months": 36.0,
            "B0_hit_clusters": 2,
            "B0_R30_hit_clusters": 3,
            "recall_gain_percentage_points": 20.0,
            "sequentially_adjusted_interval_lower": -10.0,
            "sequentially_adjusted_interval_upper": 50.0,
            "decision": "report_uncertain_at_final_review",
        }
    )
    early_terminal = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2026-11-17T00:00:00Z",
        previous_record=truth,
        fields=terminal_fields,
    )
    with pytest.raises(ValueError, match="cannot precede the frozen terminal"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                *continuity,
                truth,
                early_terminal,
            ],
            _schema(),
            score_registries_by_sha256=_base_score_registry_map(),
        )


def test_record_chain_requires_every_look_crossed_in_one_mature_batch() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, _ = _chain_landmarks(records)
    scores_25 = _record_scores(25)
    truth_fields = _record_fields(truth)
    truth_fields.update(
        {
            "target_event_count": 25,
            "independent_cluster_count": 25,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(scores_25),
        }
    )
    truth_25 = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=cast(str, truth["recorded_at_utc"]),
        previous_record=continuity[-1],
        fields=truth_fields,
    )
    first_only = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=scores_25),
        elapsed_months=2.2,
    )[0]
    incomplete_batch = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2026-11-17T00:00:00Z",
        previous_record=truth_25,
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **first_only.as_mapping(),
        },
    )
    with pytest.raises(ValueError, match="every crossed cluster look"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                *continuity,
                truth_25,
                incomplete_batch,
            ],
            _schema(),
            score_registries_by_sha256=_score_registry_map(scores_25),
        )


def test_record_chain_terminal_cannot_drop_mature_primary_clusters() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, review = _chain_landmarks(records)
    scores_5 = _record_scores(5)
    truth_fields = _record_fields(truth)
    truth_fields.update(
        {
            "target_event_count": 5,
            "independent_cluster_count": 5,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(scores_5),
        }
    )
    truth_5 = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=cast(str, truth["recorded_at_utc"]),
        previous_record=continuity[-1],
        fields=truth_fields,
    )
    terminal_fields = _record_fields(review)
    terminal_fields.update(
        {
            "review_trigger": "time_36_months",
            "cumulative_cluster_count": 0,
            "elapsed_months": 36.0,
            "B0_hit_clusters": 0,
            "B0_R30_hit_clusters": 0,
            "recall_gain_percentage_points": None,
            "sequentially_adjusted_interval_lower": None,
            "sequentially_adjusted_interval_upper": None,
            "decision": "report_evidence_insufficient_at_final_review",
        }
    )
    pre_secondary_missed = _append_authorized_missed_records(
        previous_record=truth_5,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 11, 18, 16, tzinfo=UTC),
        before_utc=datetime(2027, 1, 15, 16, tzinfo=UTC),
    )
    secondary_truth = _build_first_issue_zero_secondary_truth(
        previous_record=pre_secondary_missed[-1], truth_template=truth
    )
    terminal_missed = _append_authorized_missed_records(
        previous_record=secondary_truth,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2027, 1, 20, 16, tzinfo=UTC),
        before_utc=datetime(2029, 9, 9, 16, tzinfo=UTC),
    )
    incomplete_terminal = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2029-09-09T16:00:00Z",
        previous_record=terminal_missed[-1],
        fields=terminal_fields,
    )
    with pytest.raises(ValueError, match="include every primary cluster"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                *continuity,
                truth_5,
                *pre_secondary_missed,
                secondary_truth,
                *terminal_missed,
                incomplete_terminal,
            ],
            _schema(),
            score_registries_by_sha256=_score_registry_map(scores_5, ()),
        )


def test_terminal_zero_clusters_distinguishes_valid_empty_truth_from_unavailable_truth() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, _ = _chain_landmarks(records)
    empty_scores: tuple[ClusterScore, ...] = ()
    empty_registry_sha = ordered_cluster_registry_sha256(empty_scores)
    zero_truth_fields = _record_fields(truth)
    zero_truth_fields.update(
        {
            "target_event_count": 0,
            "independent_cluster_count": 0,
            "exposure_cluster_registry_sha256": empty_registry_sha,
        }
    )
    zero_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=cast(str, truth["recorded_at_utc"]),
        previous_record=continuity[-1],
        fields=zero_truth_fields,
    )
    pre_secondary_missed = _append_authorized_missed_records(
        previous_record=zero_truth,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 11, 18, 16, tzinfo=UTC),
        before_utc=datetime(2027, 1, 15, 16, tzinfo=UTC),
    )
    zero_secondary_truth = _build_first_issue_zero_secondary_truth(
        previous_record=pre_secondary_missed[-1], truth_template=truth
    )
    terminal_missed = _append_authorized_missed_records(
        previous_record=zero_secondary_truth,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2027, 1, 20, 16, tzinfo=UTC),
        before_utc=datetime(2029, 9, 9, 16, tzinfo=UTC),
    )
    empty_terminal = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=()), elapsed_months=36.0
    )[0]
    evidence_insufficient = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2029-09-09T16:00:00Z",
        previous_record=terminal_missed[-1],
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **empty_terminal.as_mapping(),
        },
    )
    valid_zero_chain = [
        protocol,
        missed,
        authorization,
        forecast,
        *continuity,
        zero_truth,
        *pre_secondary_missed,
        zero_secondary_truth,
        *terminal_missed,
        evidence_insufficient,
    ]
    validate_record_chain(
        valid_zero_chain,
        _schema(),
        score_registries_by_sha256={empty_registry_sha: []},
    )

    unavailable_fields = _record_fields(truth)
    unavailable_fields.update(
        {
            "source_snapshot_sha256": None,
            "status": "truth_snapshot_unavailable",
            "target_event_count": None,
            "independent_cluster_count": None,
            "cluster_assignment_sha256": None,
            "exposure_cluster_registry_sha256": None,
        }
    )
    unavailable_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=cast(str, truth["recorded_at_utc"]),
        previous_record=continuity[-1],
        fields=unavailable_fields,
    )
    unavailable_pre_secondary_missed = _append_authorized_missed_records(
        previous_record=unavailable_truth,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 11, 18, 16, tzinfo=UTC),
        before_utc=datetime(2027, 1, 15, 16, tzinfo=UTC),
    )
    unavailable_secondary_truth = _build_first_issue_zero_secondary_truth(
        previous_record=unavailable_pre_secondary_missed[-1], truth_template=truth
    )
    unavailable_terminal_missed = _append_authorized_missed_records(
        previous_record=unavailable_secondary_truth,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2027, 1, 20, 16, tzinfo=UTC),
        before_utc=datetime(2029, 9, 9, 16, tzinfo=UTC),
    )
    pause_fields = empty_terminal.as_mapping()
    pause_fields["decision"] = "pause_scientific_integrity_failure"
    integrity_pause = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2029-09-09T16:00:00Z",
        previous_record=unavailable_terminal_missed[-1],
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **pause_fields,
        },
    )
    unavailable_chain = [
        protocol,
        missed,
        authorization,
        forecast,
        *continuity,
        unavailable_truth,
        *unavailable_pre_secondary_missed,
        unavailable_secondary_truth,
        *unavailable_terminal_missed,
        integrity_pause,
    ]
    validate_record_chain(
        unavailable_chain,
        _schema(),
        score_registries_by_sha256={empty_registry_sha: []},
    )
    unavailable_as_false_zero = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2029-09-09T16:00:00Z",
        previous_record=unavailable_terminal_missed[-1],
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **empty_terminal.as_mapping(),
        },
    )
    with pytest.raises(ValueError, match="integrity pause must correspond exactly"):
        validate_record_chain(
            [*unavailable_chain[:-1], unavailable_as_false_zero],
            _schema(),
            score_registries_by_sha256={empty_registry_sha: []},
        )


def test_terminal_without_any_selected_exposure_requires_integrity_pause() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, _, _, _, _ = _chain_landmarks(records)
    terminal_missed = _append_authorized_missed_records(
        previous_record=authorization,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 9, 16, 16, tzinfo=UTC),
        before_utc=datetime(2029, 9, 9, 16, tzinfo=UTC),
    )
    empty_terminal = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=()), elapsed_months=36.0
    )[0]
    pause_fields = empty_terminal.as_mapping()
    pause_fields["decision"] = "pause_scientific_integrity_failure"
    integrity_pause = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2029-09-09T16:00:00Z",
        previous_record=terminal_missed[-1],
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **pause_fields,
        },
    )
    validate_record_chain(
        [protocol, missed, authorization, *terminal_missed, integrity_pause], _schema()
    )
    skipped_calendar_pause = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2029-09-09T16:00:00Z",
        previous_record=authorization,
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **pause_fields,
        },
    )
    with pytest.raises(ValueError, match="omitted an already elapsed weekly issue"):
        validate_record_chain([protocol, missed, authorization, skipped_calendar_pause], _schema())


def test_record_chain_allows_ordered_terminal_catch_up_batch() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, _ = _chain_landmarks(records)
    scores_25 = _record_scores(25)
    truth_fields = _record_fields(truth)
    truth_fields.update(
        {
            "target_event_count": 25,
            "independent_cluster_count": 25,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(scores_25),
        }
    )
    truth_25 = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=cast(str, truth["recorded_at_utc"]),
        previous_record=continuity[-1],
        fields=truth_fields,
    )
    immediate_values = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=scores_25),
        elapsed_months=2.2,
    )
    immediate_reviews: list[dict[str, Any]] = []
    previous: dict[str, Any] = truth_25
    for item in immediate_values:
        current = build_record(
            "SequentialReviewRecord",
            recorded_at_utc="2026-11-17T00:00:00Z",
            previous_record=previous,
            fields={
                "protocol_definition_sha256": protocol["content_sha256"],
                "authorization_record_sha256": authorization["content_sha256"],
                **item.as_mapping(),
            },
        )
        immediate_reviews.append(current)
        previous = current
    assert [item.review_trigger for item in immediate_values] == ["cluster_10", "cluster_20"]

    pre_secondary_missed = _append_authorized_missed_records(
        previous_record=immediate_reviews[-1],
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 11, 18, 16, tzinfo=UTC),
        before_utc=datetime(2027, 1, 15, 16, tzinfo=UTC),
    )
    secondary_truth = _build_first_issue_zero_secondary_truth(
        previous_record=pre_secondary_missed[-1], truth_template=truth
    )
    terminal_missed = _append_authorized_missed_records(
        previous_record=secondary_truth,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2027, 1, 20, 16, tzinfo=UTC),
        before_utc=datetime(2029, 9, 9, 16, tzinfo=UTC),
    )
    terminal_values = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=scores_25),
        elapsed_months=36.0,
        completed_reviews=immediate_values,
    )
    assert [item.review_trigger for item in terminal_values] == ["time_36_months"]
    terminal_review = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2029-09-09T16:00:00Z",
        previous_record=terminal_missed[-1],
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **terminal_values[0].as_mapping(),
        },
    )
    validate_record_chain(
        [
            protocol,
            missed,
            authorization,
            forecast,
            *continuity,
            truth_25,
            *immediate_reviews,
            *pre_secondary_missed,
            secondary_truth,
            *terminal_missed,
            terminal_review,
        ],
        _schema(),
        score_registries_by_sha256=_score_registry_map(scores_25, ()),
    )
    post_final_missed = build_record(
        "MissedIssueRecord",
        recorded_at_utc="2029-09-09T16:05:00Z",
        previous_record=terminal_review,
        fields={
            "issue_id": "p1-20290909T160000Z",
            "status": "missed_issue",
            "scheduled_issue_time_utc": "2029-09-09T16:00:00Z",
            "authorization_state": "authorized",
            "authorization_record_sha256": authorization["content_sha256"],
            "reason": "source_snapshot_unavailable_before_T",
            "prediction_generated": False,
            "backfill_forbidden": True,
            "valid_from_remains_fixed": True,
        },
    )
    with pytest.raises(ValueError, match="only pending truth follow-up"):
        validate_record_chain(
            [
                protocol,
                missed,
                authorization,
                forecast,
                *continuity,
                truth_25,
                *immediate_reviews,
                *pre_secondary_missed,
                secondary_truth,
                *terminal_missed,
                terminal_review,
                post_final_missed,
            ],
            _schema(),
            score_registries_by_sha256=_score_registry_map(scores_25, ()),
        )


def test_cluster_30_final_allows_only_pending_truth_follow_up() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, _ = _chain_landmarks(records)
    scores_30 = _record_scores(30)
    truth_fields = _record_fields(truth)
    truth_fields.update(
        {
            "target_event_count": 30,
            "independent_cluster_count": 30,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(scores_30),
        }
    )
    truth_30 = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=cast(str, truth["recorded_at_utc"]),
        previous_record=continuity[-1],
        fields=truth_fields,
    )
    pending_reviews = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=scores_30), elapsed_months=2.2
    )
    built_reviews: list[dict[str, Any]] = []
    previous: dict[str, Any] = truth_30
    for item in pending_reviews:
        current = build_record(
            "SequentialReviewRecord",
            recorded_at_utc="2026-11-17T00:00:00Z",
            previous_record=previous,
            fields={
                "protocol_definition_sha256": protocol["content_sha256"],
                "authorization_record_sha256": authorization["content_sha256"],
                **item.as_mapping(),
            },
        )
        built_reviews.append(current)
        previous = current

    empty_scores: tuple[ClusterScore, ...] = ()
    secondary_truth_fields = _record_fields(truth)
    secondary_truth_fields.update(
        {
            "horizon_days": 90,
            "mature_after_utc": "2027-01-14T16:00:00Z",
            "truth_fetched_at_utc": "2027-01-15T16:00:00Z",
            "target_event_count": 0,
            "independent_cluster_count": 0,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(empty_scores),
        }
    )
    pending_secondary_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2027-01-15T16:00:00Z",
        previous_record=built_reviews[-1],
        fields=secondary_truth_fields,
    )
    validate_record_chain(
        [
            protocol,
            missed,
            authorization,
            forecast,
            *continuity,
            truth_30,
            *built_reviews,
            pending_secondary_truth,
        ],
        _schema(),
        score_registries_by_sha256=_score_registry_map(scores_30, empty_scores),
    )


def test_integrity_pause_allows_committed_truth_follow_up_without_reopening_reviews() -> None:
    records = _six_record_chain()
    protocol, missed, authorization, forecast, continuity, truth, _ = _chain_landmarks(records)

    unavailable_fields = _record_fields(truth)
    unavailable_fields.update(
        {
            "source_snapshot_sha256": None,
            "status": "truth_snapshot_unavailable",
            "target_event_count": None,
            "independent_cluster_count": None,
            "cluster_assignment_sha256": None,
            "exposure_cluster_registry_sha256": None,
        }
    )
    unavailable_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=cast(str, truth["recorded_at_utc"]),
        previous_record=continuity[-1],
        fields=unavailable_fields,
    )

    pre_second_continuity = _append_authorized_missed_records(
        previous_record=unavailable_truth,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2026, 11, 18, 16, tzinfo=UTC),
        before_utc=datetime(2027, 1, 15, 16, tzinfo=UTC),
    )
    first_secondary_truth = _build_first_issue_zero_secondary_truth(
        previous_record=pre_second_continuity[-1], truth_template=truth
    )

    second_issue_time = datetime(2027, 1, 20, 16, tzinfo=UTC)
    second_forecast_fields = _record_fields(forecast)
    second_forecast_fields.update(
        {
            "issue_id": "p1-20270120T160000Z",
            "scheduled_issue_time_utc": "2027-01-20T16:00:00Z",
            "query_cutoff_utc": "2027-01-20T15:45:00Z",
            "forecast_created_at_utc": "2027-01-20T15:55:00Z",
            "publication_completed_at_utc": "2027-01-20T15:59:00Z",
        }
    )
    second_forecast = build_record(
        "ForecastIssueRecord",
        recorded_at_utc="2027-01-20T15:59:30Z",
        previous_record=first_secondary_truth,
        fields=second_forecast_fields,
    )
    second_continuity = _append_authorized_missed_records(
        previous_record=second_forecast,
        authorization_sha256=authorization["content_sha256"],
        first_scheduled_utc=datetime(2027, 1, 27, 16, tzinfo=UTC),
        before_utc=datetime(2027, 3, 22, 16, tzinfo=UTC),
    )

    second_scores = _record_scores(
        25,
        issue_id="p1-20270120T160000Z",
        scheduled_issue_utc=second_issue_time,
    )
    second_primary_fields = _record_fields(truth)
    second_primary_fields.update(
        {
            "issue_id": "p1-20270120T160000Z",
            "source_snapshot_sha256": _sha("second-primary-truth-source"),
            "mature_after_utc": "2027-03-21T16:00:00Z",
            "truth_fetched_at_utc": "2027-03-22T16:00:00Z",
            "target_event_count": 25,
            "independent_cluster_count": 25,
            "cluster_assignment_sha256": _sha("second-primary-cluster-assignment"),
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(second_scores),
        }
    )
    second_primary_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2027-03-22T16:00:00Z",
        previous_record=second_continuity[-1],
        fields=second_primary_fields,
    )
    pause_value = build_pending_sequential_reviews(
        ScoreSummary(horizon_days=30, scores=second_scores), elapsed_months=6.4
    )[0].as_mapping()
    pause_value["decision"] = "pause_scientific_integrity_failure"
    integrity_pause = build_record(
        "SequentialReviewRecord",
        recorded_at_utc="2027-03-22T16:00:00Z",
        previous_record=second_primary_truth,
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            **pause_value,
        },
    )

    empty_scores: tuple[ClusterScore, ...] = ()
    second_secondary_fields = _record_fields(truth)
    second_secondary_fields.update(
        {
            "issue_id": "p1-20270120T160000Z",
            "horizon_days": 90,
            "source_snapshot_sha256": _sha("second-secondary-truth-source"),
            "mature_after_utc": "2027-05-20T16:00:00Z",
            "truth_fetched_at_utc": "2027-05-21T16:00:00Z",
            "target_event_count": 0,
            "independent_cluster_count": 0,
            "cluster_assignment_sha256": _sha("second-secondary-cluster-assignment"),
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(empty_scores),
        }
    )
    post_pause_secondary_truth = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc="2027-05-21T16:00:00Z",
        previous_record=integrity_pause,
        fields=second_secondary_fields,
    )

    validate_record_chain(
        [
            protocol,
            missed,
            authorization,
            forecast,
            *continuity,
            unavailable_truth,
            *pre_second_continuity,
            first_secondary_truth,
            second_forecast,
            *second_continuity,
            second_primary_truth,
            integrity_pause,
            post_pause_secondary_truth,
        ],
        _schema(),
        score_registries_by_sha256=_score_registry_map(second_scores, empty_scores),
    )


def test_non_synthetic_source_is_rejected_at_the_input_boundary() -> None:
    with pytest.raises(ValueError, match="explicitly synthetic"):
        SyntheticEvent(
            event_id="not-synthetic",
            origin_time_utc=SYNTHETIC_QUERY_CUTOFF_UTC,
            available_at_utc=SYNTHETIC_QUERY_CUTOFF_UTC,
            x_km=0,
            y_km=0,
            magnitude=4.0,
            source_id=cast(Any, "usgs_comcat"),
        )


def test_truth_clustering_excludes_a_target_outside_the_frozen_grid() -> None:
    forecast = build_dual_model_forecast(
        make_synthetic_model_events(),
        make_equal_area_grid(),
        issue_id="outside-issue",
        scheduled_issue_time_utc=SYNTHETIC_ISSUE_TIME_UTC,
    )
    outside = _event(
        "outside",
        origin=SYNTHETIC_ISSUE_TIME_UTC + timedelta(days=1),
        x_km=-10,
        y_km=-10,
    )
    clusters = cluster_target_events(
        (outside,),
        issue_id="outside-issue",
        issue_time_utc=SYNTHETIC_ISSUE_TIME_UTC,
        horizon_days=30,
        truth_fetched_at_utc=SYNTHETIC_ISSUE_TIME_UTC + timedelta(days=61),
        grid=make_equal_area_grid(),
    )
    assert clusters == ()
    assert score_clusters(forecast, clusters, horizon_days=30).cluster_count == 0
