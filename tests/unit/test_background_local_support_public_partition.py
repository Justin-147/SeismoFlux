from __future__ import annotations

import dataclasses
import math
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest
from shapely.geometry import box

from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.local_support import (
    LocalSupportBaseCell,
    LocalSupportBasePartition,
    build_local_support_base_partition,
    build_local_support_snapshot,
    build_local_support_study_area_identity,
    resolve_local_support_cell,
)


def _events() -> tuple[CompletenessEvent, ...]:
    start = datetime(1970, 1, 2, tzinfo=UTC)
    coordinates = (
        (-999_999.0, -499_999.0),
        (-500_000.0, -1.0),
        (0.0, -1.0),
        (-999_999.0, 0.0),
        (-500_000.0, 0.0),
        (500_000.0, 500_000.0),
    )
    return tuple(
        CompletenessEvent(
            event_id=f"partition-{index:04d}",
            origin_time_utc=start + timedelta(minutes=index),
            available_at=start + timedelta(minutes=index),
            magnitude=3.0 if index < 150 else 3.2,
            inside_study_area=True,
            x_m=coordinates[index % len(coordinates)][0],
            y_m=coordinates[index % len(coordinates)][1],
        )
        for index in range(200)
    )


def _key(cell: LocalSupportBaseCell | None) -> tuple[int, int] | None:
    return None if cell is None else (cell.row, cell.column)


def test_public_partition_preserves_exact_clipping_order_and_existing_digest() -> None:
    study_area = box(-750_000.0, -250_000.0, 750_000.0, 750_000.0)

    partition = build_local_support_base_partition(study_area)
    identity = build_local_support_study_area_identity(study_area)

    assert isinstance(partition, LocalSupportBasePartition)
    assert partition.study_area_sha256 == identity.study_area_sha256
    assert (
        partition.total_area_m2
        == study_area.area
        == math.fsum(cell.clipped_area_m2 for cell in partition.cells)
    )
    assert tuple((cell.row, cell.column) for cell in partition.cells) == tuple(
        sorted((cell.row, cell.column) for cell in partition.cells)
    )
    assert tuple(
        (cell.cell_id, cell.row, cell.column, cell.clipped_area_m2) for cell in partition.cells
    ) == tuple(
        (cell.cell_id, cell.row, cell.column, cell.clipped_area_m2) for cell in identity.fixed_cells
    )

    areas = {(cell.row, cell.column): cell.clipped_area_m2 for cell in partition.cells}
    assert areas[(-1, -2)] == 250_000.0 * 250_000.0
    assert areas[(0, -1)] == 500_000.0 * 500_000.0
    assert areas[(1, 1)] == 250_000.0 * 250_000.0

    with pytest.raises(dataclasses.FrozenInstanceError):
        partition.total_area_m2 = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        partition.cells[0].clipped_area_m2 = 0.0  # type: ignore[misc]


def test_public_locator_preserves_negative_floor_and_grid_boundary_rules() -> None:
    partition = build_local_support_base_partition(
        box(-1_000_000.0, -500_000.0, 500_000.0, 500_000.0)
    )

    assert _key(partition.resolve(x_m=-500_001.0, y_m=-1.0)) == (-1, -2)
    assert _key(partition.resolve(x_m=-500_000.0, y_m=-1.0)) == (-1, -1)
    assert _key(partition.resolve(x_m=0.0, y_m=-1.0)) == (-1, 0)
    assert _key(partition.resolve(x_m=-1.0, y_m=0.0)) == (0, -1)
    assert _key(partition.resolve(x_m=0.0, y_m=0.0)) == (0, 0)

    # Exact high-side outer boundaries have no positive-area high cell and
    # therefore fall back to the covered adjacent cell.
    assert _key(partition.resolve(x_m=500_000.0, y_m=1.0)) == (0, 0)
    assert _key(partition.resolve(x_m=1.0, y_m=500_000.0)) == (0, 0)
    assert _key(partition.resolve(x_m=500_000.0, y_m=500_000.0)) == (0, 0)

    assert partition.resolve(x_m=500_001.0, y_m=0.0) is None
    assert partition.resolve(x_m=-1_000_001.0, y_m=0.0) is None
    with pytest.raises(ValueError, match="finite"):
        partition.resolve(x_m=math.nan, y_m=0.0)


def test_public_partition_and_locator_regress_exactly_to_snapshot_behavior() -> None:
    study_area = box(-1_000_000.0, -500_000.0, 500_000.0, 500_000.0)
    events = _events()
    partition = build_local_support_base_partition(study_area)
    snapshot = build_local_support_snapshot(
        events,
        fit_end_utc=datetime(1975, 1, 1, tzinfo=UTC),
        study_area_equal_area=study_area,
    )

    assert partition.study_area_sha256 == snapshot.study_area_sha256
    assert partition.total_area_m2 == snapshot.total_area_m2
    assert tuple(
        (
            cell.cell_id,
            cell.row,
            cell.column,
            cell.clipped_area_m2,
            cell.clipped_geometry.wkb,
        )
        for cell in partition.cells
    ) == tuple(
        (
            cell.cell_id,
            cell.row,
            cell.column,
            cell.clipped_area_m2,
            cell.clipped_geometry.wkb,
        )
        for cell in snapshot.cells
    )

    public_counts = Counter(
        _key(partition.resolve(x_m=event.x_m, y_m=event.y_m)) for event in events
    )
    assert None not in public_counts
    assert {(cell.row, cell.column): cell.base_event_count for cell in snapshot.cells} == {
        (cell.row, cell.column): public_counts[(cell.row, cell.column)] for cell in snapshot.cells
    }

    for event in events:
        public_cell = partition.resolve(x_m=event.x_m, y_m=event.y_m)
        legacy_cell = resolve_local_support_cell(
            snapshot,
            x_m=event.x_m,
            y_m=event.y_m,
        )
        assert _key(public_cell) == (
            None if legacy_cell is None else (legacy_cell.row, legacy_cell.column)
        )
