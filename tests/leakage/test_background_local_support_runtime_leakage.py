from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from shapely.geometry import box

from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.local_support_manifest import (
    LocalSupportSourceFile,
    LocalSupportSources,
    build_background_local_support_manifest,
)
from seismoflux.background.local_support_runtime import build_local_support_runtime

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
) -> list[CompletenessEvent]:
    return [
        CompletenessEvent(
            event_id=f"{prefix}-{index:04d}",
            origin_time_utc=start + timedelta(minutes=index),
            available_at=start + timedelta(minutes=index),
            magnitude=magnitude,
            inside_study_area=True,
            x_m=x_m,
            y_m=1_000.0,
        )
        for index, magnitude in enumerate(magnitudes)
    ]


def _catalog_events() -> tuple[CompletenessEvent, ...]:
    return tuple(
        [
            *_events(
                "initial-main",
                _magnitudes(3.0, 3.2, count=400),
                start=datetime(1971, 1, 1, tzinfo=UTC),
                x_m=1_000.0,
            ),
            *_events(
                "later-main",
                _magnitudes(3.0, 3.2, count=400),
                start=datetime(2006, 1, 1, tzinfo=UTC),
                x_m=1_000.0,
            ),
            *_events(
                "later-high",
                _magnitudes(4.2, 4.4, count=200),
                start=datetime(2006, 6, 1, tzinfo=UTC),
                x_m=510_000.0,
            ),
        ]
    )


def test_post_fit_event_cannot_change_any_reconstructed_support_or_grid_domain() -> None:
    events = _catalog_events()
    manifest = build_background_local_support_manifest(
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
    baseline = build_local_support_runtime(
        manifest,
        events,
        study_area_equal_area=STUDY_AREA,
    )
    future = CompletenessEvent(
        event_id="post-final-fit",
        origin_time_utc=datetime(2024, 1, 1, tzinfo=UTC),
        available_at=datetime(2024, 1, 1, tzinfo=UTC),
        magnitude=9.0,
        inside_study_area=True,
        x_m=510_000.0,
        y_m=1_000.0,
    )
    contaminated = build_local_support_runtime(
        manifest,
        (*events, future),
        study_area_equal_area=STUDY_AREA,
    )

    assert tuple(snapshot.support.support_id for snapshot in contaminated.snapshots) == tuple(
        snapshot.support.support_id for snapshot in baseline.snapshots
    )
    assert tuple(snapshot.compensator_domain_id for snapshot in contaminated.snapshots) == tuple(
        snapshot.compensator_domain_id for snapshot in baseline.snapshots
    )
    for observed, expected in zip(contaminated.snapshots, baseline.snapshots, strict=True):
        assert observed.supported_mask[:-1].tolist() == expected.supported_mask.tolist()
        assert observed.etas_primary_parent_role_mask[:-1].tolist() == (
            expected.etas_primary_parent_role_mask.tolist()
        )
        assert observed.etas_sensitivity_parent_role_mask[:-1].tolist() == (
            expected.etas_sensitivity_parent_role_mask.tolist()
        )
