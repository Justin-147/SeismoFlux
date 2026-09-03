"""Synthetic target identities/boundaries only; no real data or model scores."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from seismoflux.multitask_s0 import CATALOG_COLUMNS, build_episodes
from seismoflux.multitask_s3.calendar import HORIZONS, REPORT_END, REPORT_START
from seismoflux.multitask_s3.input_waterlevel import summarize_windows
from seismoflux.multitask_s3.targets import (
    S3WindowTargets,
    UnevaluableTargetWindowError,
    build_window_targets,
    prepare_anchor_ids,
)

ISSUE = datetime(2023, 7, 20, 16, tzinfo=UTC)


def _row(
    event_id: str,
    days: float,
    magnitude: float,
    *,
    available: datetime | None = None,
    inside: bool = True,
    longitude: float = 105.0,
) -> dict[str, object]:
    origin = ISSUE + timedelta(days=days)
    return {
        "event_id": event_id,
        "origin_time_utc": origin,
        "available_at": origin if available is None else available,
        "longitude": longitude,
        "latitude": 35.0,
        "magnitude": magnitude,
        "inside_study_area": inside,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))


def _targets(frame: pd.DataFrame, **overrides: Any) -> S3WindowTargets:
    arguments: dict[str, Any] = {
        "issue_time": ISSUE,
        "horizon_days": 7,
        "available_by": ISSUE + timedelta(days=10),
        "cell_indices": np.zeros(len(frame), dtype=np.int64),
        "cell_count": 3,
        "anchor_ids_by_band": {"Ms5_6": set(), "Ms6_plus": set()},
    }
    arguments.update(overrides)
    return build_window_targets(frame, **arguments)


def test_strict_left_closed_right_and_inclusive_availability_boundaries() -> None:
    cutoff = ISSUE + timedelta(days=10)
    frame = _frame(
        [
            _row("at_issue", 0, 5.0),
            _row("within", 1, 5.2),
            _row("at_end", 7, 6.0),
            _row("after_end", 7, 5.8),
            _row("at_cutoff", 2, 5.5, available=cutoff),
            _row("late", 3, 5.3, available=cutoff + timedelta(microseconds=1)),
        ]
    )
    for name in ("origin_time_utc", "available_at"):
        frame.loc[3, name] = ISSUE + timedelta(days=7, microseconds=1)
    targets = _targets(
        frame,
        cell_indices=np.array([-1, 2, 0, -1, 2, -1]),
        anchor_ids_by_band={"Ms5_6": {"within"}, "Ms6_plus": {"at_end"}},
    )
    assert targets.count_ms4plus == targets.count_ms5plus == 3
    np.testing.assert_array_equal(targets.spatial_counts_ms4, [1, 0, 2])
    assert targets.bands["Ms5_6"].event_ids == ("within", "at_cutoff")
    np.testing.assert_array_equal(targets.bands["Ms5_6"].anchor_mask, [True, False])
    assert targets.bands["Ms5_6"].anchor_count == 1
    assert targets.bands["Ms6_plus"].event_ids == ("at_end",)


def test_ms4_spatial_targets_and_disjoint_formal_bands_keep_raw_magnitudes() -> None:
    frame = _frame(
        [
            _row("below4", 1, 3.999),
            _row("at4", 1, 4.0),
            _row("below5", 1, 4.999),
            _row("at5", 1, 5.0),
            _row("below6", 1, 5.999),
            _row("at6", 1, 6.0),
            _row("above6", 1, 7.0),
            _row("outside", 1, 7.5, inside=False),
        ]
    )
    frame["magnitude_type"] = None
    targets = _targets(frame, cell_indices=np.array([-1, 0, 0, 1, 1, 2, 2, -1]))
    assert targets.count_ms4plus == 6
    assert targets.count_ms5plus == 4
    np.testing.assert_array_equal(targets.spatial_counts_ms4, [2, 2, 2])
    assert targets.bands["Ms5_6"].event_ids == ("at5", "below6")
    assert targets.bands["Ms6_plus"].event_ids == ("at6", "above6")
    assert sum(band.event_count for band in targets.bands.values()) == targets.count_ms5plus


def test_positional_mapping_preserves_unsorted_nonunique_dataframe_index() -> None:
    frame = _frame([_row("later", 6, 5.2), _row("earlier", 1, 6.1), _row("middle", 3, 5.5)])
    frame.index = [9, 2, 9]
    before = frame.copy(deep=True)
    positions = np.array([2, 0, 1])
    targets = _targets(frame, cell_indices=positions)
    assert targets.bands["Ms5_6"].event_ids == ("later", "middle")
    np.testing.assert_array_equal(targets.bands["Ms5_6"].cell_indices, [2, 1])
    np.testing.assert_array_equal(targets.bands["Ms6_plus"].cell_indices, [0])
    pd.testing.assert_frame_equal(frame, before)
    positions[:] = 0
    np.testing.assert_array_equal(targets.bands["Ms5_6"].cell_indices, [2, 1])
    for values in (
        targets.spatial_counts_ms4,
        targets.bands["Ms5_6"].cell_indices,
        targets.bands["Ms5_6"].anchor_mask,
    ):
        assert not values.flags.writeable


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_timestamp_units_preserve_exact_target_boundaries(unit: str) -> None:
    frame = _frame([_row("at_end", 7, 5.0), _row("past_end", 7, 6.0)])
    for name in ("origin_time_utc", "available_at"):
        frame.loc[1, name] += timedelta(microseconds=1)
        frame[name] = frame[name].dt.as_unit(unit)
    targets = _targets(frame)
    assert targets.count_ms5plus == 1
    assert targets.bands["Ms5_6"].event_ids == ("at_end",)


def test_timezone_conversion_does_not_shift_target_window() -> None:
    frame = _frame([_row("e", 7, 5.0)])
    local = timezone(timedelta(hours=8))
    targets = _targets(
        frame,
        issue_time=ISSUE.astimezone(local),
        available_by=(ISSUE + timedelta(days=7)).astimezone(local),
    )
    assert targets.issue_time_utc == ISSUE
    assert targets.target_end_utc == ISSUE + timedelta(days=7)
    assert targets.count_ms5plus == 1


@pytest.mark.parametrize("horizon", HORIZONS)
def test_all_five_complete_horizons_preserve_real_zero_windows(horizon: int) -> None:
    frame = _frame([_row("history", -3, 6.0)])
    targets = _targets(frame, horizon_days=horizon, available_by=ISSUE + timedelta(days=horizon))
    assert targets.horizon_days == horizon
    assert targets.count_ms4plus == targets.count_ms5plus == 0
    np.testing.assert_array_equal(targets.spatial_counts_ms4, np.zeros(3, dtype=np.int64))
    assert all(band.event_count == band.anchor_count == 0 for band in targets.bands.values())


def test_empty_catalog_can_describe_a_legal_zero_window() -> None:
    frame = _frame([])
    targets = _targets(frame)
    assert targets.count_ms4plus == targets.count_ms5plus == 0
    assert prepare_anchor_ids(frame) == {"Ms5_6": set(), "Ms6_plus": set()}


@pytest.mark.parametrize("horizon", [0, 8, 30.0, True])
def test_unregistered_or_noninteger_horizons_are_rejected(horizon: Any) -> None:
    with pytest.raises(ValueError, match="registered integer horizons"):
        _targets(_frame([]), horizon_days=horizon)


def test_immature_window_is_explicitly_unevaluable_not_zero() -> None:
    with pytest.raises(UnevaluableTargetWindowError, match="not mature"):
        _targets(_frame([]), available_by=ISSUE + timedelta(days=7, microseconds=-1))


@pytest.mark.parametrize(
    "issue", [REPORT_START - timedelta(microseconds=1), REPORT_END - timedelta(days=7), REPORT_END]
)
def test_unauthorized_issue_or_closed_end_is_not_a_zero_window(issue: datetime) -> None:
    with pytest.raises(UnevaluableTargetWindowError, match="authorization"):
        _targets(_frame([]), issue_time=issue, available_by=REPORT_END + timedelta(days=30))


@pytest.mark.parametrize("field", ["issue_time", "available_by"])
def test_naive_window_dates_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _targets(_frame([]), **{field: datetime(2023, 7, 20)})


@pytest.mark.parametrize("field", ["origin_time_utc", "available_at"])
def test_naive_or_missing_catalog_dates_are_rejected(field: str) -> None:
    frame = _frame([_row("e", 1, 5.0)])
    frame[field] = frame[field].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        _targets(frame)
    frame[field] = pd.NaT
    with pytest.raises(ValueError, match="missing timestamps"):
        _targets(frame)


@pytest.mark.parametrize("magnitude", [4.0, 5.0, 6.0])
def test_unmapped_eligible_target_is_never_silently_dropped(magnitude: float) -> None:
    frame = _frame([_row("unmapped", 2, magnitude)])
    with pytest.raises(UnevaluableTargetWindowError, match="1 eligible Ms4\\+"):
        _targets(frame, cell_indices=np.array([-1]))


@pytest.mark.parametrize(
    "positions",
    [np.array([], dtype=int), np.array([[0]]), np.array([0.0]), np.array([-2]), np.array([3])],
)
def test_mapping_shape_type_and_range_errors_are_rejected(positions: np.ndarray[Any, Any]) -> None:
    with pytest.raises(ValueError, match="cell_indices"):
        _targets(_frame([_row("e", 1, 5.0)]), cell_indices=positions)


def test_duplicate_identifiers_are_rejected_before_counting_or_episode_assignment() -> None:
    frame = _frame([_row("same", 1, 5.0), _row("same", 2, 6.0)])
    with pytest.raises(ValueError, match="unique"):
        _targets(frame)
    with pytest.raises(ValueError, match="unique"):
        prepare_anchor_ids(frame)


def test_availability_before_origin_and_nonfinite_magnitude_are_rejected() -> None:
    frame = _frame([_row("e", 1, 5.0, available=ISSUE)])
    with pytest.raises(ValueError, match="cannot precede"):
        _targets(frame)
    frame = _frame([_row("e", 1, float("nan"))])
    with pytest.raises(ValueError, match="finite raw"):
        _targets(frame)


def test_anchor_sets_require_both_disjoint_formal_bands() -> None:
    frame = _frame([])
    with pytest.raises(ValueError, match="exactly"):
        _targets(frame, anchor_ids_by_band={"Ms5_6": set()})
    with pytest.raises(ValueError, match="both disjoint"):
        _targets(frame, anchor_ids_by_band={"Ms5_6": {"e"}, "Ms6_plus": {"e"}})


def test_anchor_preparation_reuses_s0_not_window_or_chained_episodes() -> None:
    frame = _frame(
        [
            _row("follower", 18, 5.4, longitude=105.1),
            _row("anchor_before_issue", -2, 5.2),
            _row("new_anchor_after_30d", 38, 5.6, longitude=105.2),
            _row("six_separate_band", 19, 6.0, longitude=105.1),
            _row("four_not_a_formal_anchor", -3, 4.8),
            _row("outside", -4, 6.5, inside=False),
        ]
    )
    anchors = prepare_anchor_ids(frame)
    assert anchors == {
        "Ms5_6": {"anchor_before_issue", "new_anchor_after_30d"},
        "Ms6_plus": {"six_separate_band"},
    }
    expected = {
        str(episode["anchor_event_id"])
        for episode in build_episodes(frame[(frame.magnitude >= 5) & (frame.magnitude < 6)])
    }
    assert anchors["Ms5_6"] == expected
    target = _targets(
        frame,
        horizon_days=90,
        available_by=ISSUE + timedelta(days=90),
        anchor_ids_by_band=anchors,
    )
    assert target.bands["Ms5_6"].event_ids == ("follower", "new_anchor_after_30d")
    np.testing.assert_array_equal(target.bands["Ms5_6"].anchor_mask, [False, True])


def test_full_history_anchors_and_counts_match_existing_waterlevel() -> None:
    frame = _frame(
        [
            _row("old_anchor", -5, 5.0),
            _row("subsequent", 1, 5.1),
            _row("four", 2, 4.1),
            _row("six", 3, 6.0),
            _row("late", 4, 5.5, available=ISSUE + timedelta(days=20)),
        ]
    )
    anchors = prepare_anchor_ids(frame)
    cutoff = ISSUE + timedelta(days=10)
    water = summarize_windows(frame, (ISSUE,), 7, cutoff, anchors)
    target = _targets(frame, anchor_ids_by_band=anchors, available_by=cutoff)
    assert water["unique_events"] == {
        "Ms4_plus": target.count_ms4plus,
        **{name: band.event_count for name, band in target.bands.items()},
    }
    assert water["fixed_first_anchor_events"] == {
        name: band.anchor_count for name, band in target.bands.items()
    }
