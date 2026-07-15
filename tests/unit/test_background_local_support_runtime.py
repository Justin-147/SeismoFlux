from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from shapely.geometry import box

from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.local_support_manifest import (
    BackgroundLocalSupportManifest,
    LocalSupportSourceFile,
    LocalSupportSources,
    build_background_local_support_manifest,
)
from seismoflux.background.local_support_runtime import (
    LocalSupportRuntimeError,
    LocalSupportRuntimeSnapshot,
    build_local_support_runtime,
)

STUDY_AREA = box(0.0, 0.0, 520_000.0, 500_000.0)


def _magnitudes(peak: float, upper: float, *, count: int) -> list[float]:
    peak_count = count * 3 // 4
    return [peak] * peak_count + [upper] * (count - peak_count)


def _events(
    prefix: str,
    magnitudes: Sequence[float],
    *,
    start: datetime,
    x_m: float,
    inside: bool = True,
) -> list[CompletenessEvent]:
    return [
        CompletenessEvent(
            event_id=f"{prefix}-{index:04d}",
            origin_time_utc=start + timedelta(minutes=index),
            available_at=start + timedelta(minutes=index),
            magnitude=magnitude,
            inside_study_area=inside,
            x_m=x_m,
            y_m=1_000.0,
        )
        for index, magnitude in enumerate(magnitudes)
    ]


def _catalog_events() -> tuple[CompletenessEvent, ...]:
    initial = _events(
        "initial-main",
        _magnitudes(3.0, 3.2, count=400),
        start=datetime(1971, 1, 1, tzinfo=UTC),
        x_m=1_000.0,
    )
    later_main = _events(
        "later-main",
        _magnitudes(3.0, 3.2, count=400),
        start=datetime(2006, 1, 1, tzinfo=UTC),
        x_m=1_000.0,
    )
    later_high = _events(
        "later-high",
        _magnitudes(4.2, 4.4, count=200),
        start=datetime(2006, 6, 1, tzinfo=UTC),
        x_m=510_000.0,
    )
    outside = _events(
        "outside",
        [5.0],
        start=datetime(1972, 1, 1, tzinfo=UTC),
        x_m=2_000_000.0,
        inside=False,
    )
    return tuple([*initial, *later_main, *later_high, *outside])


def _manifest(events: tuple[CompletenessEvent, ...]) -> BackgroundLocalSupportManifest:
    return build_background_local_support_manifest(
        events,
        study_area_equal_area=STUDY_AREA,
        sources=LocalSupportSources(
            earthquake_dataset=LocalSupportSourceFile(
                path="data/processed/earthquake.parquet",
                sha256="a" * 64,
            ),
            study_area=LocalSupportSourceFile(
                path="data/processed/study.geojson",
                sha256="b" * 64,
            ),
        ),
    )


def _index(events: tuple[CompletenessEvent, ...], event_id: str) -> int:
    return next(index for index, event in enumerate(events) if event.event_id == event_id)


def test_runtime_reconstructs_frozen_snapshots_caches_domains_and_exposes_no_targets() -> None:
    events = _catalog_events()
    manifest = _manifest(events)

    runtime = build_local_support_runtime(
        manifest,
        events,
        study_area_equal_area=STUDY_AREA,
    )

    assert runtime.manifest_id == manifest.manifest_id
    assert tuple(snapshot.snapshot_id for snapshot in runtime.snapshots) == (
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "final_validation",
    )
    fold_1 = runtime.snapshot("fold_1")
    fold_2 = runtime.snapshot("fold_2")
    assert fold_1.support.retained_area_fraction == 1.0
    assert fold_2.support.retained_area_fraction == pytest.approx(250.0 / 260.0)
    assert fold_1.grid_family is not fold_2.grid_family
    assert all(snapshot.grid_family is fold_2.grid_family for snapshot in runtime.snapshots[1:])
    assert all(
        snapshot.compensator_domain_id == fold_2.compensator_domain_id
        for snapshot in runtime.snapshots[1:]
    )
    assert fold_1.compensator_domain_id != fold_2.compensator_domain_id
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", snapshot.compensator_domain_id)
        for snapshot in runtime.snapshots
    )
    for snapshot in runtime.snapshots:
        assert not snapshot.supported_mask.flags.writeable
        assert not snapshot.etas_primary_parent_role_mask.flags.writeable
        assert not snapshot.etas_sensitivity_parent_role_mask.flags.writeable
        assert all(
            math.isclose(
                math.fsum(cell.clipped_area_m2 for cell in grid.cells),
                snapshot.support.retained_area_m2,
                rel_tol=1.0e-12,
                abs_tol=1.0e-6,
            )
            for grid in snapshot.grid_family.grids
        )

    forbidden = ("target", "assessment", "score", "likelihood", "information_gain")
    runtime_fields = {
        field.name.casefold() for field in dataclasses.fields(LocalSupportRuntimeSnapshot)
    }
    assert not any(token in field for token in forbidden for field in runtime_fields)


def test_runtime_masks_apply_supported_and_two_frozen_etas_parent_roles() -> None:
    events = _catalog_events()
    runtime = build_local_support_runtime(
        _manifest(events),
        events,
        study_area_equal_area=STUDY_AREA,
    )
    fold_1 = runtime.snapshot("fold_1")
    fold_2 = runtime.snapshot("fold_2")
    main_low = _index(events, "initial-main-0000")
    main_complete = _index(events, "initial-main-0300")
    high_below_local_mc = _index(events, "later-high-0000")
    high_at_local_mc = _index(events, "later-high-0150")
    outside = _index(events, "outside-0000")

    assert fold_1.supported_mask[high_below_local_mc]
    assert fold_1.etas_sensitivity_parent_role_mask[high_below_local_mc]

    assert fold_2.supported_mask[main_low]
    assert not fold_2.etas_primary_parent_role_mask[main_low]
    assert fold_2.etas_primary_parent_role_mask[main_complete]
    assert fold_2.etas_sensitivity_parent_role_mask[main_complete]

    assert not fold_2.supported_mask[high_below_local_mc]
    assert not fold_2.etas_primary_parent_role_mask[high_below_local_mc]
    assert not fold_2.etas_sensitivity_parent_role_mask[high_below_local_mc]
    assert not fold_2.supported_mask[high_at_local_mc]
    assert fold_2.etas_primary_parent_role_mask[high_at_local_mc]
    assert not fold_2.etas_sensitivity_parent_role_mask[high_at_local_mc]

    assert fold_2.event_cell_ids[outside] is None
    assert not fold_2.supported_mask[outside]
    assert not fold_2.etas_primary_parent_role_mask[outside]


def test_runtime_rejects_catalog_that_differs_from_frozen_snapshot_fields() -> None:
    events = _catalog_events()
    manifest = _manifest(events)
    changed = (
        *events,
        CompletenessEvent(
            event_id="extra-causal-event",
            origin_time_utc=datetime(1972, 6, 1, tzinfo=UTC),
            available_at=datetime(1972, 6, 1, tzinfo=UTC),
            magnitude=3.2,
            inside_study_area=True,
            x_m=1_000.0,
            y_m=1_000.0,
        ),
    )

    with pytest.raises(LocalSupportRuntimeError, match="frozen support manifest"):
        build_local_support_runtime(
            manifest,
            changed,
            study_area_equal_area=STUDY_AREA,
        )
