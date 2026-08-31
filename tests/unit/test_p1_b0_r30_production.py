from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from seismoflux.p1_b0_r30.core import LOCAL_CATALOG_CUTOFF_UTC
from seismoflux.p1_b0_r30.production import (
    COMCAT_RESPONSE_LIMIT,
    COMCAT_SOURCE_ID,
    ComCatEvent,
    ComCatHttpExchange,
    P1IssueSchedule,
    acquire_issue_input,
    build_comcat_count_snapshot,
    build_comcat_snapshot,
    build_issue_count_url,
    build_issue_query_url,
    deduplicate_comcat_revisions,
    issue_schedule,
    next_issue_schedule,
    parse_comcat_count_geojson,
    parse_comcat_geojson,
    validate_issue_count_url,
    validate_issue_query_url,
)

FIRST_T = datetime(2026, 9, 9, 16, 0, tzinfo=UTC)
FIRST_Q = datetime(2026, 9, 9, 15, 45, tzinfo=UTC)


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _feature(
    event_id: str,
    *,
    origin: datetime,
    updated: datetime,
    magnitude: float = 4.2,
    identifiers: str | None = None,
    longitude: float = 105.0,
    latitude: float = 35.0,
    depth_km: float = 10.0,
) -> dict[str, object]:
    properties: dict[str, object] = {
        "mag": magnitude,
        "time": _epoch_ms(origin),
        "updated": _epoch_ms(updated),
        "ids": identifiers if identifiers is not None else f",{event_id},",
        "type": "earthquake",
    }
    return {
        "type": "Feature",
        "id": event_id,
        "properties": properties,
        "geometry": {
            "type": "Point",
            "coordinates": [longitude, latitude, depth_km],
        },
    }


def _geojson_bytes(features: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {
            "type": "FeatureCollection",
            "metadata": {"count": len(features)},
            "features": features,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _headers(
    payload: bytes, *, content_type: str = "application/json; charset=utf-8"
) -> dict[str, str]:
    return {
        "Date": "Wed, 09 Sep 2026 15:47:00 GMT",
        "ETag": '"fixture"',
        "Last-Modified": "Wed, 09 Sep 2026 15:46:00 GMT",
        "Content-Type": content_type,
        "Content-Length": str(len(payload)),
    }


def _exchange(
    request_url: str,
    payload: bytes,
    *,
    started: datetime = FIRST_Q + timedelta(seconds=5),
    completed: datetime = FIRST_Q + timedelta(seconds=10),
    status: int = 200,
) -> ComCatHttpExchange:
    return ComCatHttpExchange(
        request_url=request_url,
        fetch_started_at_utc=started,
        fetch_completed_at_utc=completed,
        http_status=status,
        response_headers=_headers(payload),
        raw_response_bytes=payload,
    )


def test_dynamic_weekly_schedule_is_strictly_future_and_never_backfills() -> None:
    schedule = next_issue_schedule(datetime(2026, 8, 31, tzinfo=UTC))
    assert schedule.issue_id == "p1-20260909T160000Z"
    assert schedule.scheduled_issue_time_utc == FIRST_T
    assert schedule.query_cutoff_utc == FIRST_Q
    assert next_issue_schedule(FIRST_T).scheduled_issue_time_utc == FIRST_T + timedelta(days=7)
    assert issue_schedule(FIRST_T).as_mapping() == {
        "issue_id": "p1-20260909T160000Z",
        "scheduled_issue_time_utc": "2026-09-09T16:00:00Z",
        "query_cutoff_utc": "2026-09-09T15:45:00Z",
    }

    with pytest.raises(ValueError, match="Thursday"):
        issue_schedule(FIRST_T + timedelta(minutes=1))
    with pytest.raises(ValueError, match="timezone-aware"):
        next_issue_schedule(datetime(2026, 8, 31))
    with pytest.raises(ValueError, match="do not match"):
        P1IssueSchedule("wrong", FIRST_T, FIRST_Q)


def test_canonical_request_urls_bind_full_frozen_selection_and_q() -> None:
    schedule = issue_schedule(FIRST_T)
    query_url = build_issue_query_url(schedule)
    count_url = build_issue_count_url(schedule)

    assert query_url.startswith("https://earthquake.usgs.gov/fdsnws/event/1/query?")
    assert "starttime=2026-07-09T04%3A25%3A56Z" in query_url
    assert "endtime=2026-09-09T15%3A45%3A00Z" in query_url
    assert "minmagnitude=3.9" in query_url
    assert "orderby=time-asc&limit=20000" in query_url
    assert "includeallorigins=false&includeallmagnitudes=false" in query_url
    assert count_url.startswith("https://earthquake.usgs.gov/fdsnws/event/1/count?")
    assert "format=geojson" in count_url
    assert "orderby" not in count_url and "limit" not in count_url and "offset" not in count_url
    validate_issue_query_url(query_url, schedule)
    validate_issue_count_url(count_url, schedule)

    with pytest.raises(ValueError, match="complete canonical"):
        validate_issue_query_url(query_url + "&unknown=1", schedule)
    with pytest.raises(ValueError, match="complete canonical"):
        validate_issue_count_url(
            count_url.replace("minmagnitude=3.9", "minmagnitude=4.0"), schedule
        )


def test_geojson_parser_keeps_real_identity_and_strictly_clips_origin_at_q() -> None:
    observed = FIRST_Q + timedelta(minutes=2)
    features = [
        _feature(
            "valid-1",
            origin=datetime(2026, 8, 20, tzinfo=UTC),
            updated=FIRST_Q - timedelta(hours=1),
            identifiers=",valid-1,alias-a,",
        ),
        _feature(
            "valid-2",
            origin=FIRST_Q,
            updated=FIRST_Q,
            magnitude=3.9,
        ),
        _feature(
            "cutover-row",
            origin=LOCAL_CATALOG_CUTOFF_UTC,
            updated=LOCAL_CATALOG_CUTOFF_UTC + timedelta(minutes=1),
        ),
        _feature(
            "future-row",
            origin=FIRST_Q + timedelta(seconds=1),
            updated=FIRST_Q + timedelta(seconds=2),
        ),
        _feature(
            "revised-after-q",
            origin=datetime(2026, 8, 21, tzinfo=UTC),
            updated=FIRST_Q + timedelta(minutes=1),
        ),
    ]
    payload = _geojson_bytes(features)
    parsed = parse_comcat_geojson(
        payload,
        observed_at_utc=observed,
        origin_end_inclusive_utc=FIRST_Q,
    )

    assert parsed.feature_count == 5
    assert parsed.before_start_excluded_count == 1
    assert parsed.after_end_excluded_count == 1
    assert parsed.unavailable_at_Q_excluded_count == 1
    assert [event.event_id for event in parsed.events] == ["valid-1", "valid-2"]
    assert all(event.provider_updated_at_utc <= FIRST_Q for event in parsed.events)
    assert parsed.events[0].source_id == COMCAT_SOURCE_ID
    assert parsed.events[0].first_seen_at_utc == observed
    assert parsed.events[0].associated_ids == ("alias-a", "valid-1")
    assert parsed.events[1].origin_time_utc == FIRST_Q
    assert parsed.events[1].magnitude == 3.9  # model-side M>=4 filtering is separate


def test_geojson_parser_rejects_unverifiable_shape_and_count() -> None:
    payload = _geojson_bytes(
        [
            _feature(
                "event",
                origin=datetime(2026, 8, 20, tzinfo=UTC),
                updated=FIRST_Q - timedelta(hours=1),
            )
        ]
    )
    malformed = json.loads(payload)
    assert isinstance(malformed, dict)
    metadata = malformed["metadata"]
    assert isinstance(metadata, dict)
    metadata["count"] = 2
    with pytest.raises(ValueError, match="metadata.count"):
        parse_comcat_geojson(
            json.dumps(malformed).encode(),
            observed_at_utc=FIRST_Q + timedelta(minutes=1),
            origin_end_inclusive_utc=FIRST_Q,
        )
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        parse_comcat_geojson(
            b"not-json",
            observed_at_utc=FIRST_Q + timedelta(minutes=1),
            origin_end_inclusive_utc=FIRST_Q,
        )

    non_earthquake = json.loads(payload)
    assert isinstance(non_earthquake, dict)
    features = non_earthquake["features"]
    assert isinstance(features, list)
    feature = features[0]
    assert isinstance(feature, dict)
    properties = feature["properties"]
    assert isinstance(properties, dict)
    properties["type"] = "quarry blast"
    with pytest.raises(ValueError, match="properties.type=earthquake"):
        parse_comcat_geojson(
            json.dumps(non_earthquake).encode(),
            observed_at_utc=FIRST_Q + timedelta(minutes=1),
            origin_end_inclusive_utc=FIRST_Q,
        )


def test_cross_query_dedup_uses_identifier_graph_latest_revision_and_first_seen() -> None:
    origin = datetime(2026, 8, 20, tzinfo=UTC)
    first_observed = FIRST_Q - timedelta(days=7)
    later_observed = FIRST_Q - timedelta(days=1)

    old = ComCatEvent(
        event_id="us-old",
        associated_ids=("us-old", "shared"),
        origin_time_utc=origin,
        provider_updated_at_utc=origin + timedelta(hours=1),
        first_seen_at_utc=first_observed,
        observed_at_utc=first_observed,
        longitude=105,
        latitude=35,
        depth_km=10,
        magnitude=4.1,
        feature_canonical_sha256="f" * 64,
    )
    bridge = replace(
        old,
        event_id="bridge",
        associated_ids=("shared", "third-id"),
        provider_updated_at_utc=origin + timedelta(hours=2),
        first_seen_at_utc=later_observed,
        observed_at_utc=later_observed,
        feature_canonical_sha256="e" * 64,
    )
    newest = replace(
        old,
        event_id="us-new",
        associated_ids=("third-id", "us-new"),
        provider_updated_at_utc=origin + timedelta(hours=3),
        first_seen_at_utc=later_observed,
        observed_at_utc=later_observed,
        magnitude=4.3,
        feature_canonical_sha256="a" * 64,
    )
    result = deduplicate_comcat_revisions([newest, old, bridge])

    assert len(result) == 1
    assert result[0].event_id == "us-new"
    assert result[0].magnitude == 4.3
    assert result[0].first_seen_at_utc == first_observed
    assert result[0].associated_ids == (
        "bridge",
        "shared",
        "third-id",
        "us-new",
        "us-old",
    )


def test_query_snapshot_binds_exact_raw_headers_events_and_causal_timeline() -> None:
    schedule = issue_schedule(FIRST_T)
    payload = _geojson_bytes(
        [
            _feature(
                "event",
                origin=datetime(2026, 8, 20, tzinfo=UTC),
                updated=FIRST_Q - timedelta(hours=1),
            )
        ]
    )
    exchange = _exchange(build_issue_query_url(schedule), payload)
    snapshot = build_comcat_snapshot(exchange, schedule=schedule)

    assert (
        snapshot.request_url_utf8_sha256
        == hashlib.sha256(exchange.request_url.encode("utf-8")).hexdigest()
    )
    assert snapshot.raw_response_sha256 == hashlib.sha256(payload).hexdigest()
    assert snapshot.response_body_byte_count == len(payload)
    assert snapshot.feature_count == snapshot.deduplicated_event_count == 1
    assert snapshot.unavailable_at_Q_excluded_count == 0
    assert snapshot.query_end_inclusive_utc == FIRST_Q
    assert snapshot.events[0].observed_at_utc == exchange.fetch_completed_at_utc
    assert snapshot.as_mapping()["snapshot_sha256"] == snapshot.snapshot_sha256

    with pytest.raises(ValueError, match="exact body"):
        replace(snapshot, raw_response_bytes=b"tampered")
    with pytest.raises(ValueError, match="must not start before Q"):
        build_comcat_snapshot(
            _exchange(
                build_issue_query_url(schedule),
                payload,
                started=FIRST_Q - timedelta(microseconds=1),
            ),
            schedule=schedule,
        )
    with pytest.raises(ValueError, match="strictly before T"):
        build_comcat_snapshot(
            _exchange(
                build_issue_query_url(schedule),
                payload,
                started=FIRST_T - timedelta(seconds=1),
                completed=FIRST_T,
            ),
            schedule=schedule,
        )


def test_count_snapshot_requires_json_integer_and_exact_provenance() -> None:
    schedule = issue_schedule(FIRST_T)
    payload = b'{"count":7,"maxAllowed":20000}'
    snapshot = build_comcat_count_snapshot(
        _exchange(build_issue_count_url(schedule), payload), schedule=schedule
    )

    assert parse_comcat_count_geojson(payload) == 7
    assert snapshot.parsed_count == 7
    assert snapshot.raw_response_sha256 == hashlib.sha256(payload).hexdigest()
    assert snapshot.as_mapping()["snapshot_sha256"] == snapshot.snapshot_sha256

    for invalid in (b'{"count":true}', b'{"count":7.0}', b'{"count":-1}', b"7"):
        with pytest.raises(ValueError):
            parse_comcat_count_geojson(invalid)


def test_count_first_acquisition_uses_injected_transport_and_matches_query_count() -> None:
    schedule = issue_schedule(FIRST_T)
    query_payload = _geojson_bytes(
        [
            _feature(
                "event",
                origin=datetime(2026, 8, 20, tzinfo=UTC),
                updated=FIRST_Q - timedelta(hours=1),
            )
        ]
    )
    seen: list[str] = []

    def transport(request_url: str) -> ComCatHttpExchange:
        seen.append(request_url)
        if request_url == build_issue_count_url(schedule):
            return _exchange(request_url, b'{"count":1}', completed=FIRST_Q + timedelta(seconds=10))
        return _exchange(
            request_url,
            query_payload,
            started=FIRST_Q + timedelta(seconds=11),
            completed=FIRST_Q + timedelta(seconds=20),
        )

    acquisition = acquire_issue_input(schedule=schedule, transport=transport)
    assert acquisition.status == "available"
    assert acquisition.query_snapshot is not None
    assert acquisition.query_snapshot.feature_count == acquisition.count_snapshot.parsed_count == 1
    assert seen == [build_issue_count_url(schedule), build_issue_query_url(schedule)]


def test_count_at_limit_fails_closed_without_calling_query_transport() -> None:
    schedule = issue_schedule(FIRST_T)
    seen: list[str] = []

    def transport(request_url: str) -> ComCatHttpExchange:
        seen.append(request_url)
        return _exchange(request_url, f'{{"count":{COMCAT_RESPONSE_LIMIT}}}'.encode())

    acquisition = acquire_issue_input(schedule=schedule, transport=transport)
    assert acquisition.status == "unavailable_count_limit"
    assert acquisition.query_snapshot is None
    assert acquisition.unavailable_reason == "count_gte_20000_query_forbidden"
    assert seen == [build_issue_count_url(schedule)]


def test_http_204_is_a_proven_zero_response_not_a_parse_failure() -> None:
    schedule = issue_schedule(FIRST_T)
    empty_headers: Mapping[str, str] = {"Content-Length": "0"}
    count_exchange = ComCatHttpExchange(
        request_url=build_issue_count_url(schedule),
        fetch_started_at_utc=FIRST_Q,
        fetch_completed_at_utc=FIRST_Q + timedelta(seconds=1),
        http_status=204,
        response_headers=empty_headers,
        raw_response_bytes=b"",
    )
    query_exchange = replace(count_exchange, request_url=build_issue_query_url(schedule))
    assert build_comcat_count_snapshot(count_exchange, schedule=schedule).parsed_count == 0
    assert build_comcat_snapshot(query_exchange, schedule=schedule).events == ()
