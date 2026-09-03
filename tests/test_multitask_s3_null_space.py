"""Tiny synthetic coordinate counterfactuals; no real snapshots or target data."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from pyproj import Transformer

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot
from seismoflux.features.anomaly.spatial import compute_selected_spatial_features
from seismoflux.features.anomaly.state import AnomalyState
from seismoflux.multitask_s3 import null_space
from seismoflux.multitask_s3.null_features import FIXED_COVERAGE_INDICES, rebuild_dynamic_values

T0 = datetime(2023, 1, 1, tzinfo=UTC)


def _snapshot(index: int) -> Stage3IssueSnapshot:
    time = T0 + timedelta(weeks=index)
    states = []
    for pos, (longitude, discipline) in enumerate(
        zip((100.0, 110.0, 90.0, 120.0), ("形变", "流体", "电磁", "跨断层"), strict=True)
    ):
        arguments: dict[str, Any] = {field.name: None for field in fields(AnomalyState)}
        arguments.update(
            state_id=f"{index}_{pos}",
            issue_time_utc=time,
            state_row_kind="entity_state",
            lineage_max_available_at_utc=time,
            longitude=longitude,
            latitude=30.0,
            spatial_eligible=True,
            identity_complete=True,
            current_report_listed=pos % 2 == 0,
            current_reporting_disciplines=(discipline,),
            known_disciplines=(discipline,),
            discipline=discipline,
            station_id=f"station_{pos}",
            measurement=f"measurement_{pos}",
            reliability_grade="high",
            source_new=pos == 0,
            system_first_seen=pos in (0, 3),
            age_days=float(10 + index) if pos % 2 == 0 else None,
        )
        states.append(AnomalyState(**arguments))
    summary = replace(
        states[0], state_id=f"summary_{index}", state_row_kind="report_period_summary"
    )
    return Stage3IssueSnapshot(index, time, summary, tuple(states), "snapshot", "lineage")


def _strata(snapshots: list[Stage3IssueSnapshot]) -> dict[str, str]:
    return {
        state.state_id: "zone:inside" if index < 2 else "zone:outside"
        for snapshot in snapshots
        for index, state in enumerate(snapshot.entities)
    }


def _query() -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    x, y = transformer.transform([100.0, 110.0, 160.0], [30.0, 30.0, 30.0])
    return np.column_stack((x, y))


def test_coordinate_pairs_are_bijective_inside_each_stratum_and_other_attributes_unchanged() -> (
    None
):
    snapshot = _snapshot(0)
    strata = _strata([snapshot])
    changed, audit = null_space.permute_snapshot_coordinates(
        snapshot=snapshot,
        strata_by_state_id=strata,
        all_zone_ids=["zone"],
        rng=np.random.default_rng(147),
    )
    originals = {state.state_id: state for state in snapshot.entities}
    replacements = {state.state_id: state for state in changed.entities}
    restored = {}
    for group in audit:
        assert sorted(group.recipient_state_ids) == sorted(group.donor_state_ids)
        for recipient, donor in zip(group.recipient_state_ids, group.donor_state_ids, strict=True):
            assert strata[recipient] == strata[donor]
            state = replacements[recipient]
            assert (state.longitude, state.latitude) == (
                originals[donor].longitude,
                originals[donor].latitude,
            )
            restored[donor] = (state.longitude, state.latitude)
    assert restored == {key: (state.longitude, state.latitude) for key, state in originals.items()}
    for state in changed.entities:
        original = originals[state.state_id]
        for field in fields(AnomalyState):
            if field.name not in ("longitude", "latitude"):
                assert getattr(state, field.name) == getattr(original, field.name)
    assert changed.summary is snapshot.summary


def test_rebuild_uses_only_200km_and_preserves_recipient_coverage_including_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(0)
    features = np.arange(60, dtype=float).reshape(3, 20)
    features[0, 14] = np.nan
    before = features.copy()
    real = compute_selected_spatial_features
    calls = []

    def spy(query: Any, entities: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(query, entities, **kwargs)

    monkeypatch.setattr(null_space, "compute_selected_spatial_features", spy)
    result = null_space.permute_space_issue(
        snapshot=snapshot,
        strata_by_state_id=_strata([snapshot]),
        all_zone_ids=["zone"],
        query_xy_m=_query(),
        features=features,
        rng=np.random.default_rng(147),
    )
    assert calls == [{"scales_km": (200.0,), "query_chunk_size": 256}]
    np.testing.assert_equal(
        result.features[:, FIXED_COVERAGE_INDICES], before[:, FIXED_COVERAGE_INDICES]
    )
    np.testing.assert_equal(features, before)
    assert result.radius_bases.shape == (3, 2)
    assert result.features[2, 16] == result.features[2, 18] == 1
    assert np.isnan(result.features[2, 7]) and np.isnan(result.features[2, 11])
    assert not result.features.flags.writeable and not result.radius_bases.flags.writeable


def test_whole_axis_dynamic_reconstruction_does_not_use_future_snapshot() -> None:
    snapshots = [_snapshot(index) for index in range(15)]
    times = [snapshot.issue_time_utc for snapshot in snapshots]
    features = np.zeros((len(times), 3, 20))
    features[:, :, 12] = np.arange(len(times))[:, None]
    features[3, 0, 15] = np.nan

    def run(values: list[Stage3IssueSnapshot]) -> Any:
        return null_space.permute_space_features(
            issue_times_utc=times,
            snapshots_by_issue={s.issue_time_utc: s for s in values},
            strata_by_state_id=_strata(values),
            all_zone_ids=["zone"],
            query_xy_m=_query(),
            features=features,
            rng=np.random.default_rng(10),
        )

    result = run(snapshots)
    expected = rebuild_dynamic_values(times, result.radius_bases)
    np.testing.assert_equal(result.features[:, :, 8:11], expected)
    np.testing.assert_equal(result.features[:, :, 17], np.isnan(expected).mean(axis=2))
    np.testing.assert_equal(
        result.features[:, :, FIXED_COVERAGE_INDICES], features[:, :, FIXED_COVERAGE_INDICES]
    )
    modified = snapshots.copy()
    modified[-1] = replace(
        modified[-1],
        entities=tuple(
            replace(state, longitude=160.0, age_days=999.0) for state in modified[-1].entities
        ),
    )
    changed = run(modified)
    np.testing.assert_equal(changed.features[:-1], result.features[:-1])
    np.testing.assert_equal(changed.radius_bases[:-1], result.radius_bases[:-1])
    assert len(result.diagnostics["issues"]) == 15
    assert result.diagnostics["models_refitted"] is False


def test_missing_stratum_future_lineage_and_misaligned_snapshot_fail() -> None:
    snapshot = _snapshot(0)
    for strata in ({}, {state.state_id: "zone" for state in snapshot.entities}):
        with pytest.raises(ValueError, match="stratum"):
            null_space.permute_snapshot_coordinates(
                snapshot=snapshot,
                strata_by_state_id=strata,
                all_zone_ids=["zone"],
                rng=np.random.default_rng(0),
            )
    bad = replace(
        snapshot,
        entities=(
            replace(snapshot.entities[0], lineage_max_available_at_utc=T0 + timedelta(days=1)),
        ),
    )
    with pytest.raises(ValueError, match="lineage"):
        null_space.permute_snapshot_coordinates(
            snapshot=bad,
            strata_by_state_id=_strata([snapshot]),
            all_zone_ids=["zone"],
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match="mapping key"):
        null_space.permute_space_features(
            issue_times_utc=[T0],
            snapshots_by_issue={T0: _snapshot(1)},
            strata_by_state_id=_strata([snapshot]),
            all_zone_ids=["zone"],
            query_xy_m=_query(),
            features=np.zeros((1, 3, 20)),
            rng=np.random.default_rng(0),
        )
