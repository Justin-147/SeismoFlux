from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from numpy.typing import NDArray

from seismoflux.background.catalog import EarthquakeCatalog, utc_timestamp_to_day
from seismoflux.background.config import load_background_protocol
from seismoflux.background.workflow import (
    ETASParentRoleMasks,
    SnapshotDefinition,
    analyze_snapshot_completeness,
    build_local_support_etas_parent_roles,
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
    inside_buffer: NDArray[np.bool_] | None = None,
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
        inside_external_buffer=(
            np.ones(count, dtype=np.bool_) if inside_buffer is None else inside_buffer
        ),
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


def test_local_support_parent_roles_apply_local_mc_without_external_buffer_leakage() -> None:
    catalog = _catalog(
        np.arange(1.0, 6.0),
        np.arange(1.0, 6.0),
        np.asarray([4.0, 4.4, 4.5, 4.0, 5.0]),
        np.asarray([True, True, True, False, False]),
        np.asarray([True, True, True, True, False]),
    )
    supported = np.asarray([True, False, False, False, False])
    unsupported = np.asarray([False, True, True, False, False])
    local_mc = np.asarray([np.nan, 4.5, 4.5, np.nan, np.nan])

    roles = build_local_support_etas_parent_roles(
        catalog,
        supported_domain_mask=supported,
        unsupported_domain_mask=unsupported,
        common_mc=4.0,
        unsupported_local_mc=local_mc,
    )

    assert isinstance(roles, ETASParentRoleMasks)
    assert roles.supported_parent.tolist() == [True, False, False, False, False]
    assert roles.true_external_buffer_parent.tolist() == [False, False, False, True, False]
    assert roles.unsupported_conditional_parent.tolist() == [False, False, True, False, False]
    assert roles.parent_mask.tolist() == [True, False, True, True, False]

    events = catalog_etas_events(
        catalog,
        roles.parent_mask,
        inside_target_domain_mask=supported,
        inside_parent_domain_mask=roles.parent_mask,
    )
    assert tuple(event.event_id for event in events) == ("e0000", "e0002", "e0003")
    assert tuple(event.inside_study_area for event in events) == (True, False, False)
    assert all(event.inside_parent_domain for event in events)


def test_local_support_parent_roles_accept_prevalidated_mask_and_exclude_sensitivity() -> None:
    catalog = _catalog(
        np.arange(1.0, 5.0),
        np.arange(1.0, 5.0),
        np.asarray([4.0, 4.4, 4.5, 4.0]),
        np.asarray([True, True, True, False]),
    )
    supported = np.asarray([True, False, False, False])
    unsupported = np.asarray([False, True, True, False])
    prevalidated = np.asarray([False, False, True, False])

    primary = build_local_support_etas_parent_roles(
        catalog,
        supported_domain_mask=supported,
        unsupported_domain_mask=unsupported,
        common_mc=4.0,
        prevalidated_unsupported_parent_mask=prevalidated,
    )
    sensitivity = primary.excluding_unsupported_parents()

    assert primary.parent_mask.tolist() == [True, False, True, True]
    assert sensitivity.supported_parent.tolist() == [True, False, False, False]
    assert sensitivity.true_external_buffer_parent.tolist() == [False, False, False, True]
    assert sensitivity.unsupported_conditional_parent.tolist() == [False] * 4
    assert sensitivity.parent_mask.tolist() == [True, False, False, True]
    assert not sensitivity.parent_mask.flags.writeable


def test_local_support_parent_roles_reject_unpartitioned_or_leaking_masks() -> None:
    catalog = _catalog(
        np.arange(1.0, 4.0),
        np.arange(1.0, 4.0),
        np.asarray([4.0, 4.5, 4.0]),
        np.asarray([True, True, False]),
    )
    supported = np.asarray([True, False, False])
    unsupported = np.asarray([False, True, False])

    with pytest.raises(ValueError, match="contained in unsupported_domain_mask"):
        build_local_support_etas_parent_roles(
            catalog,
            supported_domain_mask=supported,
            unsupported_domain_mask=unsupported,
            common_mc=4.0,
            prevalidated_unsupported_parent_mask=np.asarray([True, False, False]),
        )
    with pytest.raises(ValueError, match="exactly partition"):
        build_local_support_etas_parent_roles(
            catalog,
            supported_domain_mask=np.asarray([True, False, False]),
            unsupported_domain_mask=np.asarray([False, False, False]),
            common_mc=4.0,
            prevalidated_unsupported_parent_mask=np.zeros(3, dtype=np.bool_),
        )


def test_etas_adapter_custom_domains_are_validated_without_changing_defaults() -> None:
    catalog = _catalog(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 2.0]),
        np.asarray([4.0, 4.5]),
        np.asarray([True, False]),
    )
    selected = np.asarray([True, True])

    default = catalog_etas_events(catalog, selected)
    assert tuple(event.inside_study_area for event in default) == (True, False)
    assert tuple(event.inside_parent_domain for event in default) == (True, True)
    explicit_legacy_domains = catalog_etas_events(
        catalog,
        selected,
        inside_target_domain_mask=catalog.inside_study_area,
        inside_parent_domain_mask=catalog.inside_external_buffer,
    )
    assert explicit_legacy_domains == default

    with pytest.raises(ValueError, match="selected custom ETAS targets"):
        catalog_etas_events(
            catalog,
            selected,
            inside_target_domain_mask=np.asarray([True, False]),
            inside_parent_domain_mask=np.asarray([False, True]),
        )
    with pytest.raises(ValueError, match="event mask must be contained"):
        catalog_etas_events(
            catalog,
            selected,
            inside_target_domain_mask=np.asarray([False, False]),
            inside_parent_domain_mask=np.asarray([True, False]),
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
