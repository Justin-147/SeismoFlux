"""Pure frozen S3 spatial counterfactuals; no loaders, targets, fits, or scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from seismoflux.d1_replay.placebos import (
    D1CoordinateEntity,
    D1CoordinatePermutation,
    permute_d1_coordinates_within_zones,
)
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot, spatial_entity_arrays
from seismoflux.features.anomaly.spatial import compute_selected_spatial_features
from seismoflux.multitask_s3.features import RAW_FEATURE_COLUMNS, REPORT_END_UTC, REPORT_START_UTC
from seismoflux.multitask_s3.models import nonnegative_log1p
from seismoflux.multitask_s3.null_features import SNAPSHOT_INDICES, rebuild_dynamic_values

FloatArray = NDArray[np.float64]


def _time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("snapshot issue must be timezone-aware")
    result = value.astimezone(UTC)
    if not REPORT_START_UTC <= result < REPORT_END_UTC:
        raise ValueError("snapshot issue lies outside authorized S3 development")
    return result


def permute_snapshot_coordinates(
    *,
    snapshot: Stage3IssueSnapshot,
    strata_by_state_id: Mapping[str, str],
    all_zone_ids: Sequence[str],
    rng: np.random.Generator,
) -> tuple[Stage3IssueSnapshot, tuple[D1CoordinatePermutation, ...]]:
    """Change only eligible entity coordinate pairs within authenticated strata."""
    issue = _time(snapshot.issue_time_utc)
    if _time(snapshot.summary.issue_time_utc) != issue:
        raise ValueError("snapshot summary time differs from its issue")
    rows = []
    for state in snapshot.entities:
        if _time(state.issue_time_utc) != issue or state.lineage_max_available_at_utc > issue:
            raise ValueError("entity issue/lineage is not the supplied causal snapshot")
        if state.spatial_eligible:
            stratum = strata_by_state_id.get(state.state_id)
            if stratum is None or stratum.rsplit(":", 1)[-1] not in {"inside", "outside"}:
                raise ValueError("eligible entity needs an authenticated inside/outside stratum")
            if state.longitude is None or state.latitude is None:
                raise ValueError("eligible entity is missing a coordinate pair")
            rows.append(
                D1CoordinateEntity(state.state_id, stratum, state.longitude, state.latitude)
            )
    coordinates, audit = permute_d1_coordinates_within_zones(rows, all_zone_ids, rng=rng)
    changed = tuple(
        replace(
            state, longitude=coordinates[state.state_id][0], latitude=coordinates[state.state_id][1]
        )
        if state.spatial_eligible
        else state
        for state in snapshot.entities
    )
    # Original identifiers/lineage are retained solely as counterfactual provenance.
    return replace(snapshot, entities=changed), audit


@dataclass(frozen=True, slots=True)
class S3SpaceIssueFeatures:
    features: FloatArray
    radius_bases: FloatArray
    diagnostics: dict[str, Any]
    coordinate_permutations: tuple[D1CoordinatePermutation, ...]


def permute_space_issue(
    *,
    snapshot: Stage3IssueSnapshot,
    strata_by_state_id: Mapping[str, str],
    all_zone_ids: Sequence[str],
    query_xy_m: FloatArray,
    features: FloatArray,
    rng: np.random.Generator,
) -> S3SpaceIssueFeatures:
    """Rebuild nine snapshots/two raw bases; leave dynamic columns for the full axis."""
    query = np.asarray(query_xy_m, dtype=np.float64)
    original = np.asarray(features, dtype=np.float64)
    if (
        query.ndim != 2
        or query.shape[1] != 2
        or not len(query)
        or not np.isfinite(query).all()
        or original.shape != (len(query), 20)
        or np.isinf(original).any()
    ):
        raise ValueError("query and 20 feature columns must align on the independent grid")
    changed, audit = permute_snapshot_coordinates(
        snapshot=snapshot,
        strata_by_state_id=strata_by_state_id,
        all_zone_ids=all_zone_ids,
        rng=rng,
    )
    spatial = compute_selected_spatial_features(
        query, spatial_entity_arrays(changed), scales_km=(200.0,), query_chunk_size=256
    )
    result = original.copy()
    for column in SNAPSHOT_INDICES:
        name = RAW_FEATURE_COLUMNS[column].removeprefix("gaussian_200km__")
        values = spatial.gaussian_features[name][:, 0]
        result[:, column] = nonnegative_log1p(values) if column < 8 else values
    result[:, 16] = np.isnan(result[:, 7])
    result[:, 18] = np.isnan(result[:, 11])
    bases = np.column_stack(
        [spatial.radius_features[name][:, 0] for name in ("listed_count", "first_seen_count")]
    )
    eligible = sum(len(item.recipient_state_ids) for item in audit)
    moved = sum(item.moved_coordinate_count for item in audit)
    diagnostics = {
        "issue_time_utc": snapshot.issue_time_utc.isoformat(),
        "eligible_entity_count": eligible,
        "fixed_identity_count": sum(item.fixed_point_count for item in audit),
        "moved_coordinate_count": moved,
        "effective_permutation_fraction": moved / eligible if eligible else 0.0,
        "coordinate_multiset_verified": all(item.coordinate_multiset_verified for item in audit),
        "mapping_sha256s": [item.mapping_sha256 for item in audit],
        "coverage_kept_at_recipient": True,
        "dynamic_rebuilt": False,
    }
    result.setflags(write=False)
    bases.setflags(write=False)
    return S3SpaceIssueFeatures(result, bases, diagnostics, audit)


@dataclass(frozen=True, slots=True)
class S3SpaceNullFeatures:
    issue_times_utc: tuple[datetime, ...]
    features: FloatArray
    radius_bases: FloatArray
    diagnostics: dict[str, Any]


def permute_space_features(
    *,
    issue_times_utc: Sequence[datetime],
    snapshots_by_issue: Mapping[datetime, Stage3IssueSnapshot],
    strata_by_state_id: Mapping[str, str],
    all_zone_ids: Sequence[str],
    query_xy_m: FloatArray,
    features: FloatArray,
    rng: np.random.Generator,
) -> S3SpaceNullFeatures:
    """Rebuild along the caller's complete authorized fold report axis, causally in time."""
    times = tuple(_time(value) for value in issue_times_utc)
    if not times or any(right <= left for left, right in pairwise(times)):
        raise ValueError("spatial-null issues must be nonempty, unique, and chronological")
    original = np.asarray(features, dtype=np.float64)
    if original.ndim != 3 or original.shape[0] != len(times) or original.shape[2] != 20:
        raise ValueError("space-null features need shape (issues, cells, 20)")
    if any(time not in snapshots_by_issue for time in times):
        raise ValueError("a requested actual-issue snapshot is missing")
    results = []
    for index, time in enumerate(times):
        snapshot = snapshots_by_issue[time]
        if _time(snapshot.issue_time_utc) != time:
            raise ValueError("snapshot mapping key and actual issue differ")
        results.append(
            permute_space_issue(
                snapshot=snapshot,
                strata_by_state_id=strata_by_state_id,
                all_zone_ids=all_zone_ids,
                query_xy_m=query_xy_m,
                features=original[index],
                rng=rng,
            )
        )
    values = np.stack([item.features for item in results])
    bases = np.stack([item.radius_bases for item in results])
    dynamic = rebuild_dynamic_values(times, bases)
    values[:, :, 8:11] = dynamic
    values[:, :, 17] = np.isnan(dynamic).mean(axis=2)
    eligible = sum(item.diagnostics["eligible_entity_count"] for item in results)
    moved = sum(item.diagnostics["moved_coordinate_count"] for item in results)
    diagnostics = {
        "role": "offline_space_counterfactual_not_a_forecast",
        "report_count": len(times),
        "eligible_entity_exposures": eligible,
        "moved_coordinate_exposures": moved,
        "effective_permutation_fraction": moved / eligible if eligible else 0.0,
        "issues": [{**item.diagnostics, "dynamic_rebuilt": True} for item in results],
        "coverage_kept_at_recipient": True,
        "dynamic_rebuilt_from_pseudo_prefix": True,
        "models_refitted": False,
    }
    values.setflags(write=False)
    bases.setflags(write=False)
    return S3SpaceNullFeatures(times, values, bases, diagnostics)
