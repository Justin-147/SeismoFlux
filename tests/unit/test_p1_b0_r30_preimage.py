from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30.core import (
    ClusterScore,
    DualModelForecast,
    ScoreSummary,
    SyntheticEvent,
    TargetCluster,
    build_pending_sequential_reviews,
    cluster_target_events,
    ordered_cluster_registry_sha256,
    score_clusters,
)
from seismoflux.p1_b0_r30.preimage import (
    ForecastScientificPreimage,
    ScientificPreimageStore,
    TruthScientificPreimage,
    build_catalog_snapshot_bytes,
    source_snapshot_sha256,
    validate_scientific_record_chain,
)
from seismoflux.p1_b0_r30.records import (
    RecordType,
    build_record,
    validate_record_chain,
)
from seismoflux.p1_b0_r30.synthetic import (
    SYNTHETIC_ISSUE_TIME_UTC,
    build_synthetic_scenario,
    make_synthetic_model_events,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "data" / "contracts" / "p1_prospective_records_v1.json"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


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


def _score_map(scores: tuple[ClusterScore, ...]) -> dict[str, list[dict[str, object]]]:
    return {ordered_cluster_registry_sha256(scores): [score.as_mapping() for score in scores]}


@dataclass(frozen=True, slots=True)
class _Fixture:
    records: list[dict[str, Any]]
    schema: dict[str, Any]
    preimages: ScientificPreimageStore
    forecast: DualModelForecast
    forecast_source_bytes: bytes
    truth_source_bytes: bytes
    clusters: tuple[TargetCluster, ...]
    scores: tuple[ClusterScore, ...]


def _build_fixture(*, zero_clusters: bool = False) -> _Fixture:
    scenario = build_synthetic_scenario("positive")
    forecast = scenario.forecast
    issue = SYNTHETIC_ISSUE_TIME_UTC
    issue_id = forecast.issue_id
    grid = forecast.grid
    forecast_source_bytes = build_catalog_snapshot_bytes(
        role="forecast",
        issue_id=issue_id,
        scheduled_issue_time_utc=issue,
        grid=grid,
        events=make_synthetic_model_events(),
    )
    target_events = () if zero_clusters else scenario.target_events
    truth_fetched = issue + timedelta(days=61)
    truth_source_bytes = build_catalog_snapshot_bytes(
        role="truth",
        issue_id=issue_id,
        scheduled_issue_time_utc=issue,
        grid=grid,
        events=target_events,
        horizon_days=30,
        truth_fetched_at_utc=truth_fetched,
    )
    clusters = cluster_target_events(
        target_events,
        issue_id=issue_id,
        issue_time_utc=issue,
        horizon_days=30,
        truth_fetched_at_utc=truth_fetched,
        grid=grid,
    )
    score_summary = score_clusters(forecast, clusters, horizon_days=30)
    forecast_preimage = ForecastScientificPreimage.from_forecast(
        raw_source_bytes=forecast_source_bytes,
        forecast=forecast,
    )
    truth_preimage = TruthScientificPreimage.from_clusters(
        issue_id=issue_id,
        horizon_days=30,
        raw_source_bytes=truth_source_bytes,
        clusters=clusters,
    )

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
    authorization = build_record(
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-09-01T08:00:00Z",
        previous_record=protocol,
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_commit": _git_sha("authorization-commit"),
            "code_commit": _git_sha("code-commit"),
            "remote_verified_at_utc": "2026-09-01T07:59:00Z",
            "authorized_from_scheduled_issue_utc": "2026-09-09T16:00:00Z",
            "real_issue_authorized": True,
        },
    )
    forecast_rows = [
        {
            "model_id": model_id,
            "relative_intensity_grid_sha256": surface.sha256,
            "alarm_mask_sha256": alarm.mask_sha256,
            "alarm_ranking_sha256": alarm.ranking_sha256,
            "actual_alarm_area_km2": alarm.actual_area_km2,
        }
        for model_id, surface, alarm in (
            ("B0", forecast.B0, forecast.B0_alarm),
            ("B0_R30", forecast.B0_R30, forecast.B0_R30_alarm),
        )
    ]
    next_area = forecast.B0_R30_alarm.next_complete_cell_area_km2
    assert next_area is not None
    forecast_record = build_record(
        "ForecastIssueRecord",
        recorded_at_utc="2026-09-09T15:59:30Z",
        previous_record=authorization,
        fields={
            "issue_id": issue_id,
            "status": "on_time",
            "scheduled_issue_time_utc": _utc_text(issue),
            "query_cutoff_utc": _utc_text(issue - timedelta(minutes=15)),
            "forecast_created_at_utc": "2026-09-09T15:55:00Z",
            "publication_completed_at_utc": "2026-09-09T15:59:00Z",
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "model_manifest_sha256": _sha("model-manifest"),
            "source_boundary_manifest_sha256": _sha("source-boundary"),
            "source_snapshot_sha256": source_snapshot_sha256(forecast_source_bytes),
            "code_commit": _git_sha("code-commit"),
            "forecasts": forecast_rows,
            "static_svg_sha256": _sha("static-svg"),
            "offline_interactive_html_sha256": _sha("interactive-html"),
            "B0_reference_area_km2": forecast.B0_reference_area_km2,
            "B0_R30_next_complete_cell_area_km2": next_area,
            "actual_area_difference_km2": forecast.actual_area_difference_km2,
            "area_fairness_status": "passed",
            "original_artifacts_immutable": True,
        },
    )
    records = [protocol, authorization, forecast_record]
    scheduled = issue + timedelta(days=7)
    while scheduled < truth_fetched:
        missed = build_record(
            "MissedIssueRecord",
            recorded_at_utc=_utc_text(scheduled + timedelta(minutes=5)),
            previous_record=records[-1],
            fields={
                "issue_id": f"p1-{scheduled.strftime('%Y%m%dT%H%M%SZ')}",
                "status": "missed_issue",
                "scheduled_issue_time_utc": _utc_text(scheduled),
                "authorization_state": "authorized",
                "authorization_record_sha256": authorization["content_sha256"],
                "reason": "source_snapshot_unavailable_before_T",
                "prediction_generated": False,
                "backfill_forbidden": True,
                "valid_from_remains_fixed": True,
            },
        )
        records.append(missed)
        scheduled += timedelta(days=7)
    truth_record = build_record(
        "TruthSnapshotRecord",
        recorded_at_utc=_utc_text(truth_fetched),
        previous_record=records[-1],
        fields={
            "issue_id": issue_id,
            "horizon_days": 30,
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "source_snapshot_sha256": source_snapshot_sha256(truth_source_bytes),
            "status": "mature_truth",
            "mature_after_utc": _utc_text(issue + timedelta(days=60)),
            "truth_fetched_at_utc": _utc_text(truth_fetched),
            "target_event_count": sum(len(cluster.member_event_ids) for cluster in clusters),
            "independent_cluster_count": len(clusters),
            "cluster_assignment_sha256": truth_preimage.cluster_assignment_sha256,
            "exposure_cluster_registry_sha256": ordered_cluster_registry_sha256(
                score_summary.scores
            ),
            "magnitude_minimum": 5.0,
            "magnitude_maximum_exclusive": 6.0,
        },
    )
    records.append(truth_record)
    reviews = build_pending_sequential_reviews(score_summary, elapsed_months=2.1)
    for offset, review in enumerate(reviews, start=1):
        review_record = build_record(
            "SequentialReviewRecord",
            recorded_at_utc=_utc_text(truth_fetched + timedelta(minutes=offset)),
            previous_record=records[-1],
            fields={
                "protocol_definition_sha256": protocol["content_sha256"],
                "authorization_record_sha256": authorization["content_sha256"],
                **review.as_mapping(),
            },
        )
        records.append(review_record)
    store = ScientificPreimageStore(
        forecasts_by_issue_id={issue_id: forecast_preimage},
        truths_by_issue_horizon={(issue_id, 30): truth_preimage},
    )
    return _Fixture(
        records=records,
        schema=_schema(),
        preimages=store,
        forecast=forecast,
        forecast_source_bytes=forecast_source_bytes,
        truth_source_bytes=truth_source_bytes,
        clusters=clusters,
        scores=score_summary.scores,
    )


def _record_index(records: list[dict[str, Any]], record_type: str) -> int:
    return next(
        index for index, record in enumerate(records) if record["record_type"] == record_type
    )


def _rechain(
    records: list[dict[str, Any]],
    updates: dict[int, dict[str, object]],
) -> list[dict[str, Any]]:
    rebuilt: list[dict[str, Any]] = []
    for index, original in enumerate(records):
        fields = _record_fields(original)
        fields.update(updates.get(index, {}))
        rebuilt.append(
            build_record(
                cast(RecordType, original["record_type"]),
                recorded_at_utc=cast(str, original["recorded_at_utc"]),
                previous_record=rebuilt[-1] if rebuilt else None,
                fields=fields,
            )
        )
    return rebuilt


def _alarm_mask_sha256(model_id: str, selected_cell_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"model_id": model_id, "selected_cell_ids": list(selected_cell_ids)})
    ).hexdigest()


def test_typed_preimages_recompute_the_complete_scientific_chain() -> None:
    fixture = _build_fixture()
    validate_scientific_record_chain(
        fixture.records,
        fixture.schema,
        preimages=fixture.preimages,
    )


def test_synchronized_score_registry_and_review_replacement_is_rejected() -> None:
    fixture = _build_fixture()
    forged_scores = tuple(replace(score, B0_hit=True, B0_R30_hit=False) for score in fixture.scores)
    forged_summary = ScoreSummary(horizon_days=30, scores=forged_scores)
    forged_review = build_pending_sequential_reviews(forged_summary, elapsed_months=2.1)[0]
    forged_sha = ordered_cluster_registry_sha256(forged_scores)
    truth_index = _record_index(fixture.records, "TruthSnapshotRecord")
    review_index = _record_index(fixture.records, "SequentialReviewRecord")
    forged_chain = _rechain(
        fixture.records,
        {
            truth_index: {"exposure_cluster_registry_sha256": forged_sha},
            review_index: forged_review.as_mapping(),
        },
    )
    validate_record_chain(
        forged_chain,
        fixture.schema,
        score_registries_by_sha256=_score_map(forged_scores),
    )
    with pytest.raises(ValueError, match="score registry differs"):
        validate_scientific_record_chain(
            forged_chain,
            fixture.schema,
            preimages=fixture.preimages,
        )


def test_cluster_member_and_representative_replacement_is_rejected() -> None:
    fixture = _build_fixture()
    original_truth = fixture.preimages.truths_by_issue_horizon[(fixture.forecast.issue_id, 30)]
    first = original_truth.clusters[0]
    invented = replace(first.representative, event_id="invented-representative")
    forged_first = replace(
        first,
        member_event_ids=(*first.member_event_ids, invented.event_id),
        representative=invented,
    )
    forged_truth = replace(
        original_truth,
        clusters=(forged_first, *original_truth.clusters[1:]),
    )
    forged_store = ScientificPreimageStore(
        forecasts_by_issue_id=fixture.preimages.forecasts_by_issue_id,
        truths_by_issue_horizon={(fixture.forecast.issue_id, 30): forged_truth},
    )
    truth_index = _record_index(fixture.records, "TruthSnapshotRecord")
    forged_chain = _rechain(
        fixture.records,
        {truth_index: {"cluster_assignment_sha256": forged_truth.cluster_assignment_sha256}},
    )
    validate_record_chain(
        forged_chain,
        fixture.schema,
        score_registries_by_sha256=_score_map(fixture.scores),
    )
    with pytest.raises(ValueError, match="cluster members or representative"):
        validate_scientific_record_chain(
            forged_chain,
            fixture.schema,
            preimages=forged_store,
        )


def test_same_area_alarm_mask_replacement_is_rejected() -> None:
    fixture = _build_fixture()
    issue_id = fixture.forecast.issue_id
    original = fixture.preimages.forecasts_by_issue_id[issue_id]
    B0 = original.models[0]
    assert len(B0.ranked_cell_ids) > len(B0.selected_cell_ids)
    forged_mask = (
        *B0.selected_cell_ids[:-1],
        B0.ranked_cell_ids[len(B0.selected_cell_ids)],
    )
    forged_B0 = replace(B0, selected_cell_ids=forged_mask)
    forged_forecast = replace(original, models=(forged_B0, original.models[1]))
    forged_store = ScientificPreimageStore(
        forecasts_by_issue_id={issue_id: forged_forecast},
        truths_by_issue_horizon=fixture.preimages.truths_by_issue_horizon,
    )
    forecast_index = _record_index(fixture.records, "ForecastIssueRecord")
    forecast_fields = _record_fields(fixture.records[forecast_index])
    model_rows = [dict(row) for row in cast(list[dict[str, object]], forecast_fields["forecasts"])]
    model_rows[0]["alarm_mask_sha256"] = _alarm_mask_sha256("B0", forged_mask)
    forged_chain = _rechain(
        fixture.records,
        {forecast_index: {"forecasts": model_rows}},
    )
    validate_record_chain(
        forged_chain,
        fixture.schema,
        score_registries_by_sha256=_score_map(fixture.scores),
    )
    with pytest.raises(ValueError, match="alarm mask differs"):
        validate_scientific_record_chain(
            forged_chain,
            fixture.schema,
            preimages=forged_store,
        )


def test_raw_source_byte_replacement_is_rejected() -> None:
    fixture = _build_fixture()
    issue_id = fixture.forecast.issue_id
    original = fixture.preimages.forecasts_by_issue_id[issue_id]
    forged_store = ScientificPreimageStore(
        forecasts_by_issue_id={
            issue_id: replace(original, raw_source_bytes=fixture.forecast_source_bytes + b"\n")
        },
        truths_by_issue_horizon=fixture.preimages.truths_by_issue_horizon,
    )
    with pytest.raises(ValueError, match="raw source bytes differ"):
        validate_scientific_record_chain(
            fixture.records,
            fixture.schema,
            preimages=forged_store,
        )


@pytest.mark.parametrize(
    ("issue_id", "horizon"),
    [("p1-20260910T160000Z", 30), ("p1-20260909T160000Z", 90)],
)
def test_missing_or_cross_context_preimages_fail_closed(issue_id: str, horizon: int) -> None:
    fixture = _build_fixture()
    original_truth = fixture.preimages.truths_by_issue_horizon[(fixture.forecast.issue_id, 30)]
    cross_truth = replace(
        original_truth,
        issue_id=issue_id,
        horizon_days=cast(Any, horizon),
    )
    cross_store = ScientificPreimageStore(
        forecasts_by_issue_id=fixture.preimages.forecasts_by_issue_id,
        truths_by_issue_horizon={(issue_id, horizon): cross_truth},
    )
    with pytest.raises(ValueError, match="truth scientific preimages"):
        validate_scientific_record_chain(
            fixture.records,
            fixture.schema,
            preimages=cross_store,
        )
    missing_store = ScientificPreimageStore(
        forecasts_by_issue_id={},
        truths_by_issue_horizon={},
    )
    with pytest.raises(ValueError, match="forecast scientific preimages"):
        validate_scientific_record_chain(
            fixture.records,
            fixture.schema,
            preimages=missing_store,
        )


def test_post_Q_event_fails_even_when_source_hash_and_chain_are_resealed() -> None:
    fixture = _build_fixture()
    issue = fixture.forecast.scheduled_issue_time_utc
    query_cutoff = issue - timedelta(minutes=15)
    post_Q = SyntheticEvent(
        event_id="post-Q-event",
        origin_time_utc=query_cutoff + timedelta(seconds=1),
        available_at_utc=query_cutoff + timedelta(seconds=2),
        x_km=100.0,
        y_km=100.0,
        magnitude=4.4,
        source_id="synthetic_ComCat",
    )
    post_Q_bytes = build_catalog_snapshot_bytes(
        role="forecast",
        issue_id=fixture.forecast.issue_id,
        scheduled_issue_time_utc=issue,
        grid=fixture.forecast.grid,
        events=(*make_synthetic_model_events(), post_Q),
    )
    original_forecast = fixture.preimages.forecasts_by_issue_id[fixture.forecast.issue_id]
    forged_store = ScientificPreimageStore(
        forecasts_by_issue_id={
            fixture.forecast.issue_id: replace(
                original_forecast,
                raw_source_bytes=post_Q_bytes,
            )
        },
        truths_by_issue_horizon=fixture.preimages.truths_by_issue_horizon,
    )
    forecast_index = _record_index(fixture.records, "ForecastIssueRecord")
    forged_chain = _rechain(
        fixture.records,
        {forecast_index: {"source_snapshot_sha256": source_snapshot_sha256(post_Q_bytes)}},
    )
    validate_record_chain(
        forged_chain,
        fixture.schema,
        score_registries_by_sha256=_score_map(fixture.scores),
    )
    with pytest.raises(ValueError, match="post-Q"):
        validate_scientific_record_chain(
            forged_chain,
            fixture.schema,
            preimages=forged_store,
        )


def test_truth_revision_after_fetch_fails_even_when_raw_bytes_and_chain_are_resealed() -> None:
    fixture = _build_fixture()
    issue = fixture.forecast.scheduled_issue_time_utc
    truth_fetched = issue + timedelta(days=61)
    unavailable_revision = SyntheticEvent(
        event_id="truth-revision-after-fetch",
        origin_time_utc=issue + timedelta(days=20),
        available_at_utc=truth_fetched + timedelta(seconds=1),
        x_km=100.0,
        y_km=100.0,
        magnitude=5.4,
        source_id="synthetic_ComCat",
    )
    forged_truth_bytes = build_catalog_snapshot_bytes(
        role="truth",
        issue_id=fixture.forecast.issue_id,
        scheduled_issue_time_utc=issue,
        grid=fixture.forecast.grid,
        events=(*build_synthetic_scenario("positive").target_events, unavailable_revision),
        horizon_days=30,
        truth_fetched_at_utc=truth_fetched,
    )
    original_truth = fixture.preimages.truths_by_issue_horizon[(fixture.forecast.issue_id, 30)]
    forged_store = ScientificPreimageStore(
        forecasts_by_issue_id=fixture.preimages.forecasts_by_issue_id,
        truths_by_issue_horizon={
            (fixture.forecast.issue_id, 30): replace(
                original_truth,
                raw_source_bytes=forged_truth_bytes,
            )
        },
    )
    truth_index = _record_index(fixture.records, "TruthSnapshotRecord")
    forged_chain = _rechain(
        fixture.records,
        {truth_index: {"source_snapshot_sha256": source_snapshot_sha256(forged_truth_bytes)}},
    )
    validate_record_chain(
        forged_chain,
        fixture.schema,
        score_registries_by_sha256=_score_map(fixture.scores),
    )
    with pytest.raises(ValueError, match="unavailable at truth fetch"):
        validate_scientific_record_chain(
            forged_chain,
            fixture.schema,
            preimages=forged_store,
        )


def test_complete_truth_response_keeps_visible_non_target_rows_without_scoring_them() -> None:
    fixture = _build_fixture()
    issue = fixture.forecast.scheduled_issue_time_utc
    truth_fetched = issue + timedelta(days=61)

    def response_event(
        event_id: str,
        *,
        origin: datetime,
        magnitude: float,
        x_km: float = 100.0,
        y_km: float = 100.0,
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

    visible_non_targets = (
        response_event("truth-at-open-boundary", origin=issue, magnitude=5.4),
        response_event(
            "truth-after-window",
            origin=issue + timedelta(days=31),
            magnitude=5.4,
        ),
        response_event(
            "truth-below-target-bin",
            origin=issue + timedelta(days=10),
            magnitude=4.9,
        ),
        response_event(
            "truth-at-upper-magnitude-boundary",
            origin=issue + timedelta(days=11),
            magnitude=6.0,
        ),
        response_event(
            "truth-outside-study-grid",
            origin=issue + timedelta(days=12),
            magnitude=5.4,
            x_km=-100.0,
            y_km=-100.0,
        ),
    )
    complete_events = (
        *build_synthetic_scenario("positive").target_events,
        *visible_non_targets,
    )
    assert all(event.available_at_utc <= truth_fetched for event in complete_events)
    recomputed_clusters = cluster_target_events(
        complete_events,
        issue_id=fixture.forecast.issue_id,
        issue_time_utc=issue,
        horizon_days=30,
        truth_fetched_at_utc=truth_fetched,
        grid=fixture.forecast.grid,
    )
    assert recomputed_clusters == fixture.clusters

    complete_truth_bytes = build_catalog_snapshot_bytes(
        role="truth",
        issue_id=fixture.forecast.issue_id,
        scheduled_issue_time_utc=issue,
        grid=fixture.forecast.grid,
        events=complete_events,
        horizon_days=30,
        truth_fetched_at_utc=truth_fetched,
    )
    original_truth = fixture.preimages.truths_by_issue_horizon[(fixture.forecast.issue_id, 30)]
    complete_store = ScientificPreimageStore(
        forecasts_by_issue_id=fixture.preimages.forecasts_by_issue_id,
        truths_by_issue_horizon={
            (fixture.forecast.issue_id, 30): replace(
                original_truth,
                raw_source_bytes=complete_truth_bytes,
            )
        },
    )
    truth_index = _record_index(fixture.records, "TruthSnapshotRecord")
    complete_chain = _rechain(
        fixture.records,
        {truth_index: {"source_snapshot_sha256": source_snapshot_sha256(complete_truth_bytes)}},
    )
    validate_scientific_record_chain(
        complete_chain,
        fixture.schema,
        preimages=complete_store,
    )


def test_available_empty_truth_is_zero_but_missing_zero_preimage_is_not() -> None:
    fixture = _build_fixture(zero_clusters=True)
    assert fixture.clusters == ()
    assert fixture.scores == ()
    validate_scientific_record_chain(
        fixture.records,
        fixture.schema,
        preimages=fixture.preimages,
    )
    missing_truth = ScientificPreimageStore(
        forecasts_by_issue_id=fixture.preimages.forecasts_by_issue_id,
        truths_by_issue_horizon={},
    )
    with pytest.raises(ValueError, match="truth scientific preimages"):
        validate_scientific_record_chain(
            fixture.records,
            fixture.schema,
            preimages=missing_truth,
        )
