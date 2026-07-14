from __future__ import annotations

from datetime import UTC, datetime, timedelta

from shapely.geometry import box

from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.local_support import (
    build_local_support_manifest,
    build_local_support_snapshot,
)

FIT_END = datetime(1975, 1, 1, tzinfo=UTC)


def _historical_events() -> list[CompletenessEvent]:
    start = datetime(1970, 1, 2, tzinfo=UTC)
    return [
        CompletenessEvent(
            event_id=f"history-{index:04d}",
            origin_time_utc=start + timedelta(minutes=index),
            available_at=start + timedelta(minutes=index),
            magnitude=3.0 if index < 150 else 3.2,
            inside_study_area=True,
            x_m=1_000.0,
            y_m=1_000.0,
        )
        for index in range(200)
    ]


def test_post_fit_and_not_yet_available_events_cannot_change_support_manifest() -> None:
    historical = _historical_events()
    future = CompletenessEvent(
        event_id="future-high-mc",
        origin_time_utc=datetime(1975, 1, 2, tzinfo=UTC),
        available_at=datetime(1975, 1, 3, tzinfo=UTC),
        magnitude=9.0,
        inside_study_area=True,
        x_m=501_000.0,
        y_m=1_000.0,
    )
    delayed = CompletenessEvent(
        event_id="historical-but-delayed",
        origin_time_utc=datetime(1974, 12, 31, tzinfo=UTC),
        available_at=datetime(1975, 1, 2, tzinfo=UTC),
        magnitude=9.0,
        inside_study_area=True,
        x_m=501_000.0,
        y_m=1_000.0,
    )
    geometry = box(0.0, 0.0, 1_000_000.0, 500_000.0)

    baseline = build_local_support_snapshot(
        historical,
        fit_end_utc=FIT_END,
        study_area_equal_area=geometry,
    )
    contaminated_input = build_local_support_snapshot(
        [*historical, future, delayed],
        fit_end_utc=FIT_END,
        study_area_equal_area=geometry,
    )

    assert contaminated_input.audit.excluded_future_origin_count == 1
    assert contaminated_input.audit.excluded_unavailable_count == 1
    assert contaminated_input.support_id == baseline.support_id
    assert build_local_support_manifest(contaminated_input) == build_local_support_manifest(
        baseline
    )


def test_event_available_exactly_at_fit_end_is_causally_included() -> None:
    historical = _historical_events()
    at_cutoff = CompletenessEvent(
        event_id="available-at-cutoff",
        origin_time_utc=FIT_END - timedelta(minutes=1),
        available_at=FIT_END,
        magnitude=3.2,
        inside_study_area=True,
        x_m=1_000.0,
        y_m=1_000.0,
    )

    result = build_local_support_snapshot(
        [*historical, at_cutoff],
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 500_000.0, 500_000.0),
    )

    assert result.audit.included_historical_inside_count == 201
    assert result.audit.excluded_unavailable_count == 0
