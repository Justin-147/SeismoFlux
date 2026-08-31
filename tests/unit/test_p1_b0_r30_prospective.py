from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from seismoflux.d1_replay.spatial import build_causal_background_components
from seismoflux.p1_b0_r30.production import (
    ComCatEvent,
    ComCatHttpExchange,
    ComCatIssueInputAcquisition,
    acquire_issue_input,
    build_issue_count_url,
    build_issue_query_url,
    issue_schedule,
)
from seismoflux.p1_b0_r30.production_rendering import parse_production_forecast_view
from seismoflux.p1_b0_r30.prospective import (
    build_production_forecast,
    deduplicate_local_comcat_boundary,
    validate_washout,
)
from seismoflux.stage2s.catalog import parse_frozen_catalog_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "stage1"
    / "debc98054172a4a1"
    / "earthquake_event.parquet"
)
STUDY_AREA_PATH = REPOSITORY_ROOT / "data" / "processed" / "china_mainland.geojson"
SUPPORT_PATH = REPOSITORY_ROOT / "data" / "manifests" / "background_local_support_manifest.json"

FIRST_T = datetime(2026, 9, 9, 16, 0, tzinfo=UTC)
FIRST_Q = FIRST_T - timedelta(minutes=15)
CODE_COMMIT = "a" * 40


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _exchange(
    request_url: str,
    payload: bytes,
    *,
    started: datetime,
    completed: datetime,
) -> ComCatHttpExchange:
    return ComCatHttpExchange(
        request_url=request_url,
        fetch_started_at_utc=started,
        fetch_completed_at_utc=completed,
        http_status=200,
        response_headers={
            "Date": "Wed, 09 Sep 2026 15:46:00 GMT",
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(payload)),
        },
        raw_response_bytes=payload,
    )


def _empty_acquisition() -> ComCatIssueInputAcquisition:
    schedule = issue_schedule(FIRST_T)
    count_payload = _json_bytes({"count": 0})
    query_payload = _json_bytes(
        {"type": "FeatureCollection", "metadata": {"count": 0}, "features": []}
    )
    count_url = build_issue_count_url(schedule)
    query_url = build_issue_query_url(schedule)

    def transport(request_url: str) -> ComCatHttpExchange:
        if request_url == count_url:
            return _exchange(
                request_url,
                count_payload,
                started=FIRST_Q + timedelta(seconds=1),
                completed=FIRST_Q + timedelta(seconds=2),
            )
        if request_url == query_url:
            return _exchange(
                request_url,
                query_payload,
                started=FIRST_Q + timedelta(seconds=3),
                completed=FIRST_Q + timedelta(seconds=4),
            )
        raise AssertionError("unexpected request URL")

    return acquire_issue_input(schedule=schedule, transport=transport)


def _single_recent_acquisition() -> ComCatIssueInputAcquisition:
    schedule = issue_schedule(FIRST_T)
    origin = FIRST_Q - timedelta(days=5)
    updated = FIRST_Q - timedelta(hours=1)
    query_payload = _json_bytes(
        {
            "type": "FeatureCollection",
            "metadata": {"count": 1},
            "features": [
                {
                    "type": "Feature",
                    "id": "recent-inside-m4",
                    "properties": {
                        "ids": ",recent-inside-m4,",
                        "mag": 4.5,
                        "time": int(origin.timestamp() * 1_000),
                        "updated": int(updated.timestamp() * 1_000),
                        "type": "earthquake",
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [105.0, 35.0, 10.0],
                    },
                }
            ],
        }
    )
    count_payload = _json_bytes({"count": 1})
    count_url = build_issue_count_url(schedule)
    query_url = build_issue_query_url(schedule)

    def transport(request_url: str) -> ComCatHttpExchange:
        if request_url == count_url:
            return _exchange(
                request_url,
                count_payload,
                started=FIRST_Q + timedelta(seconds=1),
                completed=FIRST_Q + timedelta(seconds=2),
            )
        if request_url == query_url:
            return _exchange(
                request_url,
                query_payload,
                started=FIRST_Q + timedelta(seconds=3),
                completed=FIRST_Q + timedelta(seconds=4),
            )
        raise AssertionError("unexpected request URL")

    return acquire_issue_input(schedule=schedule, transport=transport)


def test_empty_comcat_production_path_reproduces_frozen_history_and_fair_area() -> None:
    schedule = issue_schedule(FIRST_T)
    validate_washout(schedule)
    acquisition = _empty_acquisition()
    bundle = build_production_forecast(
        schedule=schedule,
        acquisition=acquisition,
        local_catalog_bytes=CATALOG_PATH.read_bytes(),
        study_area_bytes=STUDY_AREA_PATH.read_bytes(),
        support_manifest_bytes=SUPPORT_PATH.read_bytes(),
    )

    assert bundle.catalog_artifact.catalog.row_count == 40_898
    assert bundle.b0_source_count == 5_991
    assert bundle.recent_source_count == 0
    assert bundle.b0_mass.tobytes() == bundle.challenger_mass.tobytes()
    assert bundle.b0_alarm.selected_cell_ids == bundle.challenger_alarm.selected_cell_ids
    assert bundle.b0_alarm.actual_area_km2 == 599_494.3733448011
    assert bundle.challenger_alarm.actual_area_km2 == 599_494.3733448011
    view = parse_production_forecast_view(bundle.forecast_mapping(code_commit=CODE_COMMIT))
    assert view.issue_id == schedule.issue_id
    assert view.B0_source_count == bundle.b0_source_count == 5_991
    assert view.R30_source_count == bundle.recent_source_count == 0
    assert view.models[0].alarm_cell_ids == view.models[1].alarm_cell_ids
    assert bundle.audit_mapping()["future_outcomes_absent"] is True
    assert [row["model_id"] for row in bundle.record_forecasts()] == ["B0", "B0_R30"]


def test_local_anchor_wins_one_to_one_cutover_match() -> None:
    local = parse_frozen_catalog_bytes(CATALOG_PATH.read_bytes())
    index = local.row_count - 1
    origin = datetime.fromtimestamp(int(local.origin_time_us[index]) / 1_000_000, tz=UTC)
    event = ComCatEvent(
        event_id="boundary-duplicate",
        associated_ids=("boundary-duplicate",),
        origin_time_utc=origin + timedelta(seconds=1),
        provider_updated_at_utc=origin + timedelta(seconds=2),
        first_seen_at_utc=FIRST_Q + timedelta(seconds=1),
        observed_at_utc=FIRST_Q + timedelta(seconds=1),
        longitude=float(local.longitude[index]),
        latitude=float(local.latitude[index]),
        depth_km=10.0,
        magnitude=float(local.magnitude[index]),
        feature_canonical_sha256="b" * 64,
    )

    retained, matches = deduplicate_local_comcat_boundary(local, (event,))

    assert retained == ()
    assert len(matches) == 1
    assert matches[0].local_event_id == local.event_ids[index]
    assert matches[0].comcat_event_id == event.event_id


def test_recent_real_comcat_event_changes_challenger_without_buying_more_area() -> None:
    schedule = issue_schedule(FIRST_T)
    bundle = build_production_forecast(
        schedule=schedule,
        acquisition=_single_recent_acquisition(),
        local_catalog_bytes=CATALOG_PATH.read_bytes(),
        study_area_bytes=STUDY_AREA_PATH.read_bytes(),
        support_manifest_bytes=SUPPORT_PATH.read_bytes(),
    )
    background = build_causal_background_components(
        bundle.catalog_artifact.catalog,
        schedule.query_cutoff_utc,
        bundle.domain,
    )

    assert bundle.recent_source_count == background.audit.recent_30d_source_count == 1
    assert not np.array_equal(bundle.b0_mass, bundle.challenger_mass)
    assert np.array_equal(bundle.challenger_mass, background.mass_for_alpha(0.25))
    assert bundle.challenger_alarm.actual_area_km2 <= bundle.b0_alarm.actual_area_km2
    area_difference = bundle.b0_alarm.actual_area_km2 - bundle.challenger_alarm.actual_area_km2
    assert 0.0 <= area_difference < 625.0
    next_area = bundle.challenger_alarm.next_complete_cell_area_km2
    assert next_area is not None and area_difference < next_area
    view = parse_production_forecast_view(bundle.forecast_mapping(code_commit=CODE_COMMIT))
    assert view.B0_source_count == bundle.b0_source_count
    assert view.R30_source_count == bundle.recent_source_count == 1
    assert view.models[0].ranked_cell_ids != view.models[1].ranked_cell_ids
    assert (
        bundle.record_forecasts()[0]["relative_intensity_grid_sha256"]
        != (bundle.record_forecasts()[1]["relative_intensity_grid_sha256"])
    )
