from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from seismoflux.background.completeness import (
    CompletenessError,
    CompletenessEvent,
    analyze_catalog_completeness,
    maximum_curvature_estimate,
    select_candidate_magnitude,
)


def _magnitudes(peak: float, upper: float, *, count: int = 200) -> list[float]:
    peak_count = count * 3 // 4
    return [peak] * peak_count + [upper] * (count - peak_count)


def _events(
    prefix: str,
    start: datetime,
    magnitudes: Sequence[float],
    *,
    x_m: float = 1_000.0,
    y_m: float = 1_000.0,
) -> list[CompletenessEvent]:
    return [
        CompletenessEvent(
            event_id=f"{prefix}-{index:04d}",
            origin_time_utc=start + timedelta(hours=index),
            available_at=start + timedelta(hours=index),
            magnitude=magnitude,
            inside_study_area=True,
            x_m=x_m,
            y_m=y_m,
        )
        for index, magnitude in enumerate(magnitudes)
    ]


def test_maximum_curvature_uses_fixed_bins_correction_and_conservative_tie() -> None:
    estimate = maximum_curvature_estimate([3.0] * 10 + [3.1] * 10 + [3.4])

    assert estimate.event_count == 21
    assert estimate.peak_bin_magnitude == 3.1
    assert estimate.corrected_magnitude == 3.3
    assert [(item.magnitude, item.count) for item in estimate.histogram] == [
        (3.0, 10),
        (3.1, 10),
        (3.4, 1),
    ]
    assert select_candidate_magnitude(3.0) == 3.0
    assert select_candidate_magnitude(3.01) == 3.2
    assert select_candidate_magnitude(3.3) == 3.5
    assert select_candidate_magnitude(3.5000000000001) == 3.5
    with pytest.raises(CompletenessError, match="exceeds frozen maximum"):
        select_candidate_magnitude(4.01)


def test_analysis_uses_only_inside_historical_events_known_at_cutoff() -> None:
    historical = _events(
        "history",
        datetime(1970, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2),
    )
    excluded = [
        CompletenessEvent(
            "outside",
            datetime(1971, 1, 1, tzinfo=UTC),
            datetime(1971, 1, 1, tzinfo=UTC),
            4.9,
            False,
            1_000.0,
            1_000.0,
        ),
        CompletenessEvent(
            "pre-1970",
            datetime(1969, 12, 31, tzinfo=UTC),
            datetime(1969, 12, 31, tzinfo=UTC),
            4.9,
            True,
            1_000.0,
            1_000.0,
        ),
        CompletenessEvent(
            "future-origin",
            datetime(1975, 1, 2, tzinfo=UTC),
            datetime(1975, 1, 2, tzinfo=UTC),
            4.9,
            True,
            1_000.0,
            1_000.0,
        ),
        CompletenessEvent(
            "future-availability",
            datetime(1974, 1, 1, tzinfo=UTC),
            datetime(1975, 1, 2, tzinfo=UTC),
            4.9,
            True,
            1_000.0,
            1_000.0,
        ),
    ]

    result = analyze_catalog_completeness(
        [*historical, *excluded],
        cutoff_utc=datetime(1975, 1, 1, tzinfo=UTC),
    )

    assert result.audit.input_event_count == 204
    assert result.audit.included_historical_inside_count == 200
    assert result.audit.excluded_outside_count == 1
    assert result.audit.excluded_pre_1970_count == 1
    assert result.audit.excluded_future_origin_count == 1
    assert result.audit.excluded_unavailable_count == 1
    assert result.selected_mc == 3.2
    assert result.selected_event_count == 50
    assert result.selected_aki_b_value == pytest.approx(math.log10(math.e) / 0.05)
    sensitivity = {item.magnitude_threshold: item for item in result.sensitivities}
    assert sensitivity[3.0].event_count == 200
    assert sensitivity[3.2].event_count == 50
    assert sensitivity[4.0].aki_b_value is None


def test_final_partial_temporal_block_requires_three_calendar_years() -> None:
    full = _events(
        "full",
        datetime(1970, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2),
    )
    partial = _events(
        "partial",
        datetime(1975, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2),
    )

    eligible = analyze_catalog_completeness(
        [*full, *partial],
        cutoff_utc=datetime(1978, 1, 1, tzinfo=UTC),
    )
    assert [block.block_id for block in eligible.temporal_blocks] == [
        "1970-1974",
        "1975-partial-1978-01-01",
    ]
    assert eligible.temporal_blocks[-1].is_partial is True
    assert eligible.temporal_blocks[-1].eligible is True

    too_short = analyze_catalog_completeness(
        [*full, *partial],
        cutoff_utc=datetime(1977, 12, 31, 23, 59, 59, tzinfo=UTC),
    )
    assert too_short.temporal_blocks[-1].is_partial is True
    assert too_short.temporal_blocks[-1].event_count == 200
    assert too_short.temporal_blocks[-1].eligible is False
    assert too_short.temporal_blocks[-1].estimate is None


def test_sparse_500km_cells_merge_into_one_fixed_negative_1000km_parent() -> None:
    first = _events(
        "west-a",
        datetime(1970, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2, count=100),
        x_m=-1_000.0,
    )
    second = _events(
        "west-b",
        datetime(1971, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2, count=100),
        x_m=-600_000.0,
    )

    result = analyze_catalog_completeness(
        [*first, *second],
        cutoff_utc=datetime(1975, 1, 1, tzinfo=UTC),
    )

    assert len(result.spatial_strata) == 1
    parent = result.spatial_strata[0]
    assert parent.source == "sparse_parent_1000km"
    assert (parent.row, parent.column) == (0, -1)
    assert parent.event_count == 200
    assert parent.eligible is True
    assert parent.indeterminate is False
    assert parent.applied_mc == 3.2
    assert {(item.base_column, item.parent_column) for item in result.sparse_cell_resolutions} == {
        (-2, -1),
        (-1, -1),
    }
    assert all(item.parent_eligible for item in result.sparse_cell_resolutions)


def test_parent_still_sparse_is_indeterminate_and_uses_global_mc() -> None:
    dense = _events(
        "dense",
        datetime(1970, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2),
    )
    isolated = _events(
        "isolated",
        datetime(1971, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2, count=50),
        x_m=2_100_000.0,
    )

    result = analyze_catalog_completeness(
        [*dense, *isolated],
        cutoff_utc=datetime(1975, 1, 1, tzinfo=UTC),
    )

    indeterminate = [item for item in result.spatial_strata if item.indeterminate]
    assert len(indeterminate) == 1
    assert indeterminate[0].event_count == 50
    assert indeterminate[0].estimate is None
    assert indeterminate[0].applied_mc == result.selected_mc == 3.2
    resolution = result.sparse_cell_resolutions[0]
    assert resolution.indeterminate is True
    assert resolution.parent_eligible is False
    assert resolution.applied_mc == result.selected_mc


def test_no_eligible_temporal_or_spatial_stratum_is_a_hard_failure() -> None:
    too_few = _events(
        "few",
        datetime(1970, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2, count=199),
    )
    with pytest.raises(CompletenessError, match="no eligible temporal"):
        analyze_catalog_completeness(
            too_few,
            cutoff_utc=datetime(1975, 1, 1, tzinfo=UTC),
        )

    scattered = [
        CompletenessEvent(
            event_id=f"scattered-{index:04d}",
            origin_time_utc=datetime(1970, 1, 2, tzinfo=UTC) + timedelta(hours=index),
            available_at=datetime(1970, 1, 2, tzinfo=UTC) + timedelta(hours=index),
            magnitude=3.2 if index % 4 == 0 else 3.0,
            inside_study_area=True,
            x_m=index * 1_000_000.0 + 1_000.0,
            y_m=1_000.0,
        )
        for index in range(200)
    ]
    with pytest.raises(CompletenessError, match="no eligible spatial"):
        analyze_catalog_completeness(
            scattered,
            cutoff_utc=datetime(1975, 1, 1, tzinfo=UTC),
        )


def test_eligible_estimate_above_four_is_a_hard_failure() -> None:
    events = _events(
        "high-mc",
        datetime(1970, 1, 2, tzinfo=UTC),
        _magnitudes(4.0, 4.2),
    )

    with pytest.raises(CompletenessError, match=r"temporal/1970-1974:.*exceeds"):
        analyze_catalog_completeness(
            events,
            cutoff_utc=datetime(1975, 1, 1, tzinfo=UTC),
        )


def test_regime_flags_use_only_frozen_mc_and_annual_count_thresholds() -> None:
    first = _events(
        "regime-a",
        datetime(1970, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2),
    )
    second = _events(
        "regime-b",
        datetime(1975, 1, 2, tzinfo=UTC),
        _magnitudes(3.3, 3.5),
    )
    third = _events(
        "regime-c",
        datetime(1980, 1, 2, tzinfo=UTC),
        _magnitudes(3.3, 3.5, count=410),
    )

    result = analyze_catalog_completeness(
        [*first, *second, *third],
        cutoff_utc=datetime(1985, 1, 1, tzinfo=UTC),
    )

    assert result.selected_mc == 3.5
    assert len(result.regime_changes) == 2
    magnitude_change, count_change = result.regime_changes
    assert magnitude_change.mc_difference == pytest.approx(0.3)
    assert magnitude_change.mc_threshold_reached is True
    assert magnitude_change.flagged is True
    assert count_change.mc_difference == 0.0
    assert count_change.annual_count_ratio > 2.0
    assert count_change.count_ratio_threshold_reached is True
    assert count_change.flagged is True


def test_analysis_is_invariant_to_input_order() -> None:
    events = [
        *_events(
            "stable-a",
            datetime(1970, 1, 2, tzinfo=UTC),
            _magnitudes(3.0, 3.2),
        ),
        *_events(
            "stable-b",
            datetime(1975, 1, 2, tzinfo=UTC),
            _magnitudes(3.3, 3.5),
        ),
    ]
    cutoff = datetime(1980, 1, 1, tzinfo=UTC)

    assert analyze_catalog_completeness(events, cutoff_utc=cutoff) == analyze_catalog_completeness(
        reversed(events), cutoff_utc=cutoff
    )


def test_event_contract_rejects_non_utc_and_duplicate_physical_ids() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        CompletenessEvent(
            "naive",
            datetime(1970, 1, 1),
            datetime(1970, 1, 1),
            3.0,
            True,
            0.0,
            0.0,
        )

    events = _events(
        "duplicate",
        datetime(1970, 1, 2, tzinfo=UTC),
        _magnitudes(3.0, 3.2),
    )
    duplicate = CompletenessEvent(
        event_id=events[0].event_id,
        origin_time_utc=datetime(1971, 1, 1, tzinfo=UTC),
        available_at=datetime(1971, 1, 1, tzinfo=UTC),
        magnitude=3.2,
        inside_study_area=True,
        x_m=1_000.0,
        y_m=1_000.0,
    )
    with pytest.raises(ValueError, match="duplicate physical event_id"):
        analyze_catalog_completeness(
            [*events, duplicate],
            cutoff_utc=datetime(1975, 1, 1, tzinfo=UTC),
        )
