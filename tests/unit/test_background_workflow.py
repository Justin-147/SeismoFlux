from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from numpy.typing import NDArray

from seismoflux.background.catalog import EarthquakeCatalog, utc_timestamp_to_day
from seismoflux.background.config import load_background_protocol
from seismoflux.background.workflow import (
    SnapshotDefinition,
    analyze_snapshot_completeness,
    build_snapshot_definitions,
    catalog_completeness_events,
    catalog_etas_events,
    event_mask,
    physical_target_mask,
)


def _catalog(
    origin_days: NDArray[np.float64],
    available_days: NDArray[np.float64],
    magnitudes: NDArray[np.float64],
    inside: NDArray[np.bool_],
) -> EarthquakeCatalog:
    count = len(origin_days)
    return EarthquakeCatalog(
        event_id=np.asarray([f"e{index:04d}" for index in range(count)]),
        origin_day=origin_days,
        available_day=available_days,
        longitude=np.full(count, 100.0),
        latitude=np.full(count, 30.0),
        x_km=np.full(count, 10.0),
        y_km=np.full(count, 10.0),
        magnitude=magnitudes,
        inside_study_area=inside,
        inside_external_buffer=np.ones(count, dtype=np.bool_),
    )


def test_snapshot_definitions_preserve_four_purged_folds_and_local_validation() -> None:
    config = load_background_protocol("configs/background.yaml")

    snapshots = build_snapshot_definitions(config)

    assert tuple(item.snapshot_id for item in snapshots) == (
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "final_validation",
    )
    assert all(item.assessment_start_day - item.fit_end_day >= 365.0 for item in snapshots)
    assert snapshots[-1].assessment_start_utc == "2024-06-30T16:00:00Z"
    assert snapshots[-1].assessment_end_utc == "2025-06-30T16:00:00Z"
    assert snapshots[-1].optimizer_model_id == "etas/final_validation"


def test_completeness_adapter_supports_pre_unix_epoch_days() -> None:
    timestamp = "1899-12-31T16:00:00Z"
    day = utc_timestamp_to_day(timestamp)
    catalog = _catalog(
        np.asarray([day]),
        np.asarray([day]),
        np.asarray([5.0]),
        np.asarray([True]),
    )

    event = catalog_completeness_events(catalog)[0]

    assert event.origin_time_utc == datetime(1899, 12, 31, 16, tzinfo=UTC)
    assert event.available_at == event.origin_time_utc


def test_event_mask_enforces_origin_availability_magnitude_and_domain() -> None:
    origin = np.asarray([1.0, 2.0, 3.0, 4.0])
    available = np.asarray([1.0, 5.0, 3.0, 4.0])
    magnitude = np.asarray([3.5, 4.0, 2.9, 5.0])
    inside = np.asarray([True, True, True, False])
    catalog = _catalog(origin, available, magnitude, inside)

    mask = event_mask(
        catalog,
        minimum_magnitude=3.0,
        origin_after_day=0.0,
        origin_through_day=4.0,
        available_through_day=4.0,
        spatial_domain="inside",
    )
    parent_mask = event_mask(
        catalog,
        minimum_magnitude=3.0,
        origin_after_day=0.0,
        origin_through_day=4.0,
        available_through_day=4.0,
        spatial_domain="parent_buffer",
    )

    assert mask.tolist() == [True, False, False, False]
    assert parent_mask.tolist() == [True, False, False, True]


def test_physical_targets_ignore_publication_for_labels_but_etas_keeps_it_for_history() -> None:
    origin = np.asarray([1.0, 2.0, 3.0, 4.0])
    available = np.asarray([1.0, 5.0, 3.0, 4.0])
    magnitude = np.asarray([3.5, 4.0, 2.9, 5.0])
    inside = np.asarray([True, True, True, False])
    catalog = _catalog(origin, available, magnitude, inside)
    target_mask = physical_target_mask(
        catalog,
        minimum_magnitude=3.0,
        origin_after_day=1.5,
        origin_through_day=3.5,
    )

    assert target_mask.tolist() == [False, True, False, False]
    etas_events = catalog_etas_events(
        catalog,
        target_mask,
        time_origin_day=1.0,
        publication_delay_days=7.0,
    )
    assert len(etas_events) == 1
    assert etas_events[0].time_days == 1.0
    assert etas_events[0].available_time_days == 11.0


def test_etas_adapter_rejects_wrong_mask_or_negative_delay() -> None:
    catalog = _catalog(
        np.asarray([1.0]),
        np.asarray([1.0]),
        np.asarray([3.5]),
        np.asarray([True]),
    )
    with pytest.raises(ValueError, match="wrong shape"):
        catalog_etas_events(catalog, np.asarray([True, False]))
    with pytest.raises(ValueError, match="non-negative"):
        catalog_etas_events(
            catalog,
            np.ones(len(catalog), dtype=np.bool_),
            publication_delay_days=-1.0,
        )


def test_snapshot_completeness_is_fit_cutoff_specific_and_inside_only() -> None:
    anchor = datetime(1970, 1, 1, tzinfo=UTC)
    count = 300
    origin = np.asarray(
        [(anchor + timedelta(days=index * 5)).timestamp() / 86_400.0 for index in range(count)]
    )
    magnitudes = np.asarray([3.0] * 240 + [3.2] * 60)
    inside = np.ones(count, dtype=np.bool_)
    catalog = _catalog(origin, origin.copy(), magnitudes, inside)
    fit_end = datetime(1975, 1, 1, tzinfo=UTC)
    assessment_start = datetime(1976, 1, 1, tzinfo=UTC)
    assessment_end = datetime(1977, 1, 1, tzinfo=UTC)
    snapshot = SnapshotDefinition(
        snapshot_id="synthetic",
        fit_end_utc=fit_end.isoformat(),
        fit_end_day=fit_end.timestamp() / 86_400.0,
        assessment_start_utc=assessment_start.isoformat(),
        assessment_start_day=assessment_start.timestamp() / 86_400.0,
        assessment_end_utc=assessment_end.isoformat(),
        assessment_end_day=assessment_end.timestamp() / 86_400.0,
        optimizer_model_id="etas/synthetic",
        is_validation=False,
    )

    result = analyze_snapshot_completeness(catalog, (snapshot,))[0]

    assert result.analysis.selected_mc == 3.2
    assert result.analysis.audit.included_historical_inside_count == count
    assert result.analysis.selected_event_count == 60
    assert result.analysis.cutoff_utc == fit_end


def test_snapshot_rejects_a_shorter_than_365_day_purge() -> None:
    fit_end = utc_timestamp_to_day("2020-01-01T00:00:00Z")

    with pytest.raises(ValueError, match="365-day purge"):
        SnapshotDefinition(
            snapshot_id="bad",
            fit_end_utc="2020-01-01T00:00:00Z",
            fit_end_day=fit_end,
            assessment_start_utc="2020-06-01T00:00:00Z",
            assessment_start_day=utc_timestamp_to_day("2020-06-01T00:00:00Z"),
            assessment_end_utc="2021-01-01T00:00:00Z",
            assessment_end_day=utc_timestamp_to_day("2021-01-01T00:00:00Z"),
            optimizer_model_id="etas/bad",
            is_validation=False,
        )
