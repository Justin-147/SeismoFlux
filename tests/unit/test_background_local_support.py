from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from shapely.geometry import MultiPolygon, box

from seismoflux.background.completeness import (
    CompletenessEvent,
    CompletenessScientificInability,
)
from seismoflux.background.local_support import (
    MINIMUM_RETAINED_AREA_FRACTION,
    LocalSupportRetentionError,
    build_local_support_manifest,
    build_local_support_snapshot,
)
from seismoflux.background.scientific import scientific_json

FIT_END = datetime(1975, 1, 1, tzinfo=UTC)
START = datetime(1970, 1, 2, tzinfo=UTC)


def _magnitudes(peak: float, upper: float, *, count: int) -> list[float]:
    peak_count = count * 3 // 4
    return [peak] * peak_count + [upper] * (count - peak_count)


def _events(
    prefix: str,
    magnitudes: Sequence[float],
    *,
    x_m: float,
    y_m: float = 1_000.0,
    start: datetime = START,
) -> list[CompletenessEvent]:
    return [
        CompletenessEvent(
            event_id=f"{prefix}-{index:04d}",
            origin_time_utc=start + timedelta(minutes=index),
            available_at=start + timedelta(minutes=index),
            magnitude=magnitude,
            inside_study_area=True,
            x_m=x_m,
            y_m=y_m,
        )
        for index, magnitude in enumerate(magnitudes)
    ]


def _parent_supported_events() -> list[CompletenessEvent]:
    return [
        *_events("west", _magnitudes(3.0, 3.2, count=100), x_m=1_000.0),
        *_events("east", _magnitudes(3.0, 3.2, count=100), x_m=501_000.0),
    ]


def test_local_high_mc_excludes_only_small_exactly_clipped_base_cell() -> None:
    study_area = box(0.0, 0.0, 520_000.0, 500_000.0)
    low = _events("low", _magnitudes(3.0, 3.2, count=600), x_m=1_000.0)
    high = _events("high", _magnitudes(4.2, 4.4, count=200), x_m=510_000.0)

    result = build_local_support_snapshot(
        [*low, *high],
        fit_end_utc=FIT_END,
        study_area_equal_area=study_area,
    )

    assert result.common_mc == 3.2
    assert len(result.cells) == 2
    retained, unsupported = result.cells
    assert retained.status == "supported"
    assert retained.raw_mc == 3.2
    assert retained.applied_mc == result.common_mc
    assert retained.clipped_area_m2 == 500_000.0 * 500_000.0
    assert unsupported.status == "unsupported"
    assert unsupported.raw_mc == 4.4
    assert unsupported.candidate_mc is None
    assert unsupported.applied_mc is None
    assert unsupported.clipped_area_m2 == 20_000.0 * 500_000.0
    assert result.total_area_m2 == study_area.area
    assert result.retained_geometry.area == retained.clipped_area_m2
    assert result.retained_area_fraction == pytest.approx(250.0 / 260.0)
    assert result.retained_area_fraction >= MINIMUM_RETAINED_AREA_FRACTION


def test_sparse_base_cells_inherit_one_fixed_1000km_parent() -> None:
    result = build_local_support_snapshot(
        _parent_supported_events(),
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 1_000_000.0, 500_000.0),
    )

    assert result.common_mc == 3.2
    assert [cell.base_event_count for cell in result.cells] == [100, 100]
    assert all(cell.status == "supported" for cell in result.cells)
    assert all(cell.source == "fixed_1000km_parent" for cell in result.cells)
    assert len({cell.parent_cell_id for cell in result.cells}) == 1
    assert all(cell.source_event_count == 200 for cell in result.cells)
    assert all(cell.applied_mc == result.common_mc for cell in result.cells)


def test_event_on_internal_500km_line_uses_half_open_higher_index_cell() -> None:
    boundary_events = _events(
        "grid-line",
        _magnitudes(3.0, 3.2, count=200),
        x_m=500_000.0,
    )

    result = build_local_support_snapshot(
        boundary_events,
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 1_000_000.0, 500_000.0),
    )

    assert [cell.base_event_count for cell in result.cells] == [0, 200]
    assert result.cells[1].column == 1


@pytest.mark.parametrize(
    ("x_m", "y_m"),
    [
        (0.0, 250_000.0),
        (500_000.0, 250_000.0),
        (250_000.0, 0.0),
        (250_000.0, 500_000.0),
        (0.0, 0.0),
        (0.0, 500_000.0),
        (500_000.0, 0.0),
        (500_000.0, 500_000.0),
    ],
)
def test_study_area_edges_and_corners_use_the_covered_positive_area_cell(
    x_m: float,
    y_m: float,
) -> None:
    result = build_local_support_snapshot(
        _events("outer-boundary", _magnitudes(3.0, 3.2, count=200), x_m=x_m, y_m=y_m),
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 500_000.0, 500_000.0),
    )

    assert len(result.cells) == 1
    assert (result.cells[0].row, result.cells[0].column) == (0, 0)
    assert result.cells[0].base_event_count == 200


@pytest.mark.parametrize(
    ("x_m", "y_m", "expected_key"),
    [
        (500_000.0, 1_000.0, (0, 1)),
        (1_000.0, 500_000.0, (1, 0)),
        (500_000.0, 500_000.0, (1, 1)),
    ],
)
def test_internal_grid_lines_and_corner_keep_high_side_priority(
    x_m: float,
    y_m: float,
    expected_key: tuple[int, int],
) -> None:
    result = build_local_support_snapshot(
        _events("internal-boundary", _magnitudes(3.0, 3.2, count=200), x_m=x_m, y_m=y_m),
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 1_000_000.0, 1_000_000.0),
    )

    counts = {(cell.row, cell.column): cell.base_event_count for cell in result.cells}
    assert counts[expected_key] == 200
    assert sum(counts.values()) == 200


def test_still_sparse_parent_is_indeterminate_and_uses_common_mc() -> None:
    dense = _events("dense", _magnitudes(3.0, 3.2, count=200), x_m=1_000.0)
    result = build_local_support_snapshot(
        dense,
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 1_500_000.0, 500_000.0),
    )

    isolated_parent_cell = result.cells[2]
    assert isolated_parent_cell.status == "indeterminate"
    assert isolated_parent_cell.source == "fixed_1000km_parent"
    assert isolated_parent_cell.base_event_count == 0
    assert isolated_parent_cell.source_event_count == 0
    assert isolated_parent_cell.raw_mc is None
    assert isolated_parent_cell.candidate_mc is None
    assert isolated_parent_cell.applied_mc == result.common_mc == 3.2
    assert result.retained_area_fraction == 1.0


def test_core_cell_rejects_count_source_status_and_candidate_inconsistency() -> None:
    dense = build_local_support_snapshot(
        _events("dense-validation", _magnitudes(3.0, 3.2, count=200), x_m=1_000.0),
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 500_000.0, 500_000.0),
    ).cells[0]
    with pytest.raises(ValueError, match="base-cell support"):
        dataclasses.replace(dense, base_event_count=199, source_event_count=199)
    with pytest.raises(ValueError, match="frozen upward mapping"):
        dataclasses.replace(dense, candidate_mc=4.0)
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(dense, raw_mc=math.nan)

    sparse_result = build_local_support_snapshot(
        _events("sparse-validation", _magnitudes(3.0, 3.2, count=200), x_m=1_000.0),
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 1_500_000.0, 500_000.0),
    )
    indeterminate = sparse_result.cells[2]
    with pytest.raises(ValueError, match="inconsistent with its fixed parent"):
        dataclasses.replace(indeterminate, source_event_count=200)


def test_core_snapshot_rejects_temporal_calendar_and_candidate_drift() -> None:
    result = build_local_support_snapshot(
        _events("temporal-validation", _magnitudes(3.0, 3.2, count=200), x_m=1_000.0),
        fit_end_utc=FIT_END,
        study_area_equal_area=box(0.0, 0.0, 500_000.0, 500_000.0),
    )
    block = result.temporal_blocks[0]
    with pytest.raises(ValueError, match="frozen calendar"):
        dataclasses.replace(
            result,
            temporal_blocks=(dataclasses.replace(block, block_id="1970-wrong"),),
        )

    assert block.estimate is not None
    wrong_estimate = dataclasses.replace(block.estimate, candidate_mc=4.0)
    with pytest.raises(ValueError, match="frozen upward mapping"):
        dataclasses.replace(
            result,
            temporal_blocks=(dataclasses.replace(block, estimate=wrong_estimate),),
        )


def test_spatial_exclusions_below_95_percent_retained_area_are_a_hard_gate() -> None:
    low = _events("low", _magnitudes(3.0, 3.2, count=600), x_m=1_000.0)
    high = _events("high", _magnitudes(4.2, 4.4, count=200), x_m=501_000.0)

    with pytest.raises(LocalSupportRetentionError) as raised:
        build_local_support_snapshot(
            [*low, *high],
            fit_end_utc=FIT_END,
            study_area_equal_area=box(0.0, 0.0, 1_000_000.0, 500_000.0),
        )

    assert raised.value.retained_area_fraction == 0.5
    assert len(raised.value.unsupported_cell_ids) == 1


def test_exactly_95_percent_retained_area_passes_the_inclusive_boundary() -> None:
    low = _events(
        "low-boundary",
        _magnitudes(3.0, 3.2, count=600),
        x_m=1_000.0,
        y_m=1_000.0,
    )
    high = _events(
        "high-boundary",
        _magnitudes(4.2, 4.4, count=200),
        x_m=510_000.0,
        y_m=1_000.0,
    )
    study_area = MultiPolygon(
        [
            box(0.0, 0.0, 475_000.0, 2_000.0),
            box(500_000.0, 0.0, 525_000.0, 2_000.0),
        ]
    )

    result = build_local_support_snapshot(
        [*low, *high],
        fit_end_utc=FIT_END,
        study_area_equal_area=study_area,
    )

    assert result.retained_area_fraction == MINIMUM_RETAINED_AREA_FRACTION


def test_temporal_raw_mc_above_four_remains_a_hard_failure() -> None:
    high = _events("high", _magnitudes(4.2, 4.4, count=200), x_m=510_000.0)

    with pytest.raises(
        CompletenessScientificInability,
        match=r"temporal/1970-1974:.*exceeds frozen maximum",
    ):
        build_local_support_snapshot(
            high,
            fit_end_utc=FIT_END,
            study_area_equal_area=box(0.0, 0.0, 520_000.0, 500_000.0),
        )


def test_no_eligible_temporal_stratum_remains_a_hard_failure() -> None:
    too_sparse = _events(
        "temporally-sparse",
        _magnitudes(3.0, 3.2, count=199),
        x_m=1_000.0,
    )

    with pytest.raises(CompletenessScientificInability) as raised:
        build_local_support_snapshot(
            too_sparse,
            fit_end_utc=FIT_END,
            study_area_equal_area=box(0.0, 0.0, 500_000.0, 500_000.0),
        )

    assert raised.value.reason_code == "no_eligible_temporal_stratum"


def test_support_id_and_manifest_are_deterministic_and_target_free() -> None:
    events = _parent_supported_events()
    geometry = box(0.0, 0.0, 1_000_000.0, 500_000.0)

    forward = build_local_support_snapshot(
        events,
        fit_end_utc=FIT_END,
        study_area_equal_area=geometry,
    )
    reverse = build_local_support_snapshot(
        reversed(events),
        fit_end_utc=FIT_END,
        study_area_equal_area=geometry,
    )
    forward_manifest = build_local_support_manifest(forward)
    reverse_manifest = build_local_support_manifest(reverse)

    assert forward.support_id == reverse.support_id
    assert forward_manifest == reverse_manifest
    assert forward_manifest.support_id.startswith("local-support-")
    assert forward_manifest.historical_event_sha256 == reverse_manifest.historical_event_sha256
    assert all(cell.clipped_area_m2 > 0.0 for cell in forward_manifest.cells)

    manifest_value = scientific_json(forward_manifest)
    assert isinstance(manifest_value, dict)

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    forbidden = ("assessment", "target", "score")
    assert not any(token in key.casefold() for key in keys(manifest_value) for token in forbidden)
    assert "clipped_geometry_wkb_hex" not in keys(manifest_value)
    assert dataclasses.is_dataclass(forward_manifest)
    assert math.isfinite(forward_manifest.retained_selected_aki_b_value)
