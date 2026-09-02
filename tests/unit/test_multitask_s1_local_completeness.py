from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from shapely.geometry import box

from seismoflux.background.local_support import build_local_support_base_partition
from seismoflux.multitask_s1.local_completeness import (
    C1_HISTORY_START_UTC,
    CompletenessSnapshotAnchor,
    LocalCompletenessEvent,
    build_completeness_snapshot_anchors,
    build_local_completeness_snapshot,
    locate_completeness_events,
    snapshot_cell_records,
)

ROOT = Path(__file__).resolve().parents[2]


def _anchor(*, cutoff: datetime = datetime(2000, 1, 1, tzinfo=UTC)) -> CompletenessSnapshotAnchor:
    return CompletenessSnapshotAnchor(
        snapshot_id="C_DEV_SYNTHETIC__I1",
        fold_id="C_DEV_SYNTHETIC",
        role="inner_block_start",
        block_id="I1",
        anchor_utc=cutoff + timedelta(hours=24),
        cutoff_utc=cutoff,
    )


def _event(
    event_id: str,
    magnitude: float,
    *,
    x_m: float,
    origin: datetime = datetime(1990, 1, 1, tzinfo=UTC),
    available: datetime | None = None,
) -> LocalCompletenessEvent:
    return LocalCompletenessEvent(
        event_id=event_id,
        origin_time_utc=origin,
        available_at_utc=available or origin,
        magnitude=magnitude,
        x_m=x_m,
        y_m=250_000.0,
    )


def _snapshot(events: list[LocalCompletenessEvent], *, cell_count: int = 3):
    partition = build_local_support_base_partition(box(0.0, 0.0, cell_count * 500_000.0, 500_000.0))
    located = locate_completeness_events(events, partition)
    return build_local_completeness_snapshot(located, anchor=_anchor(), partition=partition)


def test_contract_builds_exact_inner_and_outer_anchor_minus_24h_axis() -> None:
    protocol = cast(
        dict[str, Any],
        yaml.safe_load(
            (ROOT / "configs/multitask_s1_development_contract.yaml").read_text("utf-8")
        ),
    )
    anchors = build_completeness_snapshot_anchors(protocol)
    assert len(anchors) == 16
    assert sum(item.role == "inner_block_start" for item in anchors) == 12
    assert sum(item.role == "outer_fold_start" for item in anchors) == 4
    assert all(item.cutoff_utc == item.anchor_utc - timedelta(hours=24) for item in anchors)
    assert anchors[0].snapshot_id == "C_DEV_2000_2004__I1"
    assert anchors[0].anchor_utc == datetime(1984, 12, 31, 16, tzinfo=UTC)
    assert anchors[-1].snapshot_id == "C_DEV_2015_2019__OUTER"


def test_maxc_tie_uses_higher_bin_plus_point_two() -> None:
    events = [_event(f"low-{index:03d}", 3.0, x_m=250_000.0) for index in range(100)] + [
        _event(f"high-{index:03d}", 3.1, x_m=250_000.0) for index in range(100)
    ]
    result = _snapshot(events, cell_count=1)
    cell = result.cells[0]
    assert cell.base_event_count == 200
    assert cell.raw_mc == pytest.approx(3.3)
    assert cell.status == "supported"


def test_high_mc_excludes_only_its_own_base_cell_and_supported_area_is_strict() -> None:
    events = [_event(f"high-{index:03d}", 4.0, x_m=250_000.0) for index in range(200)] + [
        _event(f"good-{index:03d}", 3.7, x_m=750_000.0) for index in range(200)
    ]
    result = _snapshot(events, cell_count=3)
    high, good, unknown = result.cells
    assert high.raw_mc == pytest.approx(4.2)
    assert high.status == "unsupported"
    assert not high.main_common_mc4_training_allowed
    assert good.raw_mc == pytest.approx(3.9)
    assert good.status == "supported"
    assert good.main_common_mc4_training_allowed
    assert good.exclude_indeterminate_training_allowed
    assert unknown.status == "indeterminate"
    assert unknown.main_common_mc4_training_allowed
    assert not unknown.exclude_indeterminate_training_allowed
    assert result.supported_area_m2 == good.clipped_area_m2
    assert result.supported_area_fraction == pytest.approx(1.0 / 3.0)
    assert not result.support_gate_passed


def test_sparse_children_inherit_fixed_parent_and_indeterminate_masks_differ() -> None:
    events = [_event(f"left-{index:03d}", 3.6, x_m=250_000.0) for index in range(100)] + [
        _event(f"right-{index:03d}", 3.6, x_m=750_000.0) for index in range(100)
    ]
    result = _snapshot(events, cell_count=4)
    inherited = result.cells[:2]
    indeterminate = result.cells[2:]
    assert all(cell.estimate_source == "parent_1000km" for cell in inherited)
    assert all(
        cell.parent_event_count == 200 and cell.raw_mc == pytest.approx(3.8) for cell in inherited
    )
    assert all(cell.status == "supported" for cell in inherited)
    assert all(cell.status == "indeterminate" for cell in indeterminate)
    assert all(cell.main_common_mc4_training_allowed for cell in indeterminate)
    assert all(not cell.exclude_indeterminate_training_allowed for cell in indeterminate)
    assert result.supported_area_fraction == pytest.approx(0.5)


def test_origin_and_availability_cutoffs_are_both_causal_and_time_diagnostic_never_hard_fails() -> (
    None
):
    cutoff = datetime(2000, 1, 1, tzinfo=UTC)
    events = [
        _event("at-history-start", 3.5, x_m=250_000.0, origin=C1_HISTORY_START_UTC),
        _event("at-cutoff", 3.5, x_m=250_000.0, origin=cutoff),
        _event(
            "late-availability",
            3.5,
            x_m=250_000.0,
            origin=cutoff - timedelta(days=1),
            available=cutoff + timedelta(microseconds=1),
        ),
        _event(
            "future-origin",
            3.5,
            x_m=250_000.0,
            origin=cutoff + timedelta(microseconds=1),
        ),
    ]
    partition = build_local_support_base_partition(box(0.0, 0.0, 500_000.0, 500_000.0))
    result = build_local_completeness_snapshot(
        locate_completeness_events(events, partition),
        anchor=_anchor(cutoff=cutoff),
        partition=partition,
    )
    assert result.visible_event_count == 2
    assert result.temporal_coverage.visible_event_count == 2
    assert result.cells[0].status == "indeterminate"
    assert not result.support_gate_passed
    temporal = cast(dict[str, object], snapshot_cell_records(result)[0])
    assert temporal["main_common_mc4_training_allowed"] is True


def test_empty_time_history_returns_evidence_insufficient_diagnostic_not_exception() -> None:
    result = _snapshot([], cell_count=2)
    assert result.visible_event_count == 0
    assert result.temporal_coverage.first_origin_utc is None
    assert all(cell.status == "indeterminate" for cell in result.cells)
    assert result.supported_area_m2 == 0.0
    assert not result.support_gate_passed
