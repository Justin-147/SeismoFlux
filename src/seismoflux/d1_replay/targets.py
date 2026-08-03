"""Score-blind D1 target windows and rule-based physical clusters.

This module reads no model output.  It derives fit and assessment memberships
from the already verified Stage 2S earthquake catalog, then checks every count
and canonical identity against the D1-0 water-level manifest.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np
from pyproj import Geod

from seismoflux.d1_replay.protocol import D1Protocol
from seismoflux.data.common import canonical_json_bytes
from seismoflux.stage2s.catalog import Stage2SEarthquakeCatalog

MICROSECONDS_PER_DAY = 86_400_000_000
ASSESSMENT_HORIZONS_DAYS = (30, 90)
FOLD_IDS = ("fold_1", "fold_2", "fold_3")
CLUSTER_MAX_TIME_SECONDS = 2_592_000
CLUSTER_MAX_DISTANCE_METRES = 75_000.0

EVENT_IDENTITY_DOMAIN = "seismoflux.d1.target-event-identity.v1"
EVENT_SET_IDENTITY_DOMAIN = "seismoflux.d1.target-event-set.v1"
CLUSTER_IDENTITY_DOMAIN = "seismoflux.d1.global-cluster-identity.v1"
ACTIVE_CLUSTER_SET_DOMAIN = "seismoflux.d1.active-cluster-set.v1"


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _ordered_ids(event_ids: Sequence[str]) -> tuple[str, ...]:
    values = tuple(event_ids)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("event IDs must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("event IDs must be unique before canonical hashing")
    return tuple(sorted(values, key=lambda value: value.encode("utf-8")))


def event_identity_sha256(event_id: str) -> str:
    """Return the frozen private-event identity without publishing the ID."""

    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")
    return _sha256({"domain": EVENT_IDENTITY_DOMAIN, "event_id": event_id})


def event_set_identity_sha256(event_ids: Sequence[str]) -> str:
    """Hash a set of raw event IDs under its preregistered domain."""

    return _sha256(
        {
            "domain": EVENT_SET_IDENTITY_DOMAIN,
            "ordered_event_ids": list(_ordered_ids(event_ids)),
        }
    )


def cluster_identity_sha256(event_ids: Sequence[str]) -> str:
    """Hash one connected component from its canonically ordered members."""

    return _sha256(
        {
            "domain": CLUSTER_IDENTITY_DOMAIN,
            "ordered_member_event_ids": list(_ordered_ids(event_ids)),
        }
    )


def active_cluster_set_identity_sha256(
    cluster_identities: Sequence[str], *, fold_id: str, horizon_days: int
) -> str:
    """Bind an active cluster set to its fold and horizon."""

    if fold_id not in FOLD_IDS:
        raise ValueError("fold_id must be fold_1, fold_2, or fold_3")
    if horizon_days not in ASSESSMENT_HORIZONS_DAYS:
        raise ValueError("assessment horizon must be 30 or 90 days")
    identities = tuple(cluster_identities)
    if len(identities) != len(set(identities)):
        raise ValueError("active cluster identities must be unique")
    if any(len(value) != 64 for value in identities):
        raise ValueError("active cluster identities must be SHA-256 hex strings")
    ordered = sorted(identities)
    return _sha256(
        {
            "domain": ACTIVE_CLUSTER_SET_DOMAIN,
            "fold_id": fold_id,
            "horizon_days": horizon_days,
            "ordered_cluster_identity_sha256": ordered,
        }
    )


def _datetime_us(value: datetime, *, label: str) -> int:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    utc = value.astimezone(UTC)
    delta = utc - datetime(1970, 1, 1, tzinfo=UTC)
    return delta.days * MICROSECONDS_PER_DAY + delta.seconds * 1_000_000 + delta.microseconds


def _parse_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include an offset")
    return parsed


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(Sequence[object], value)


def _exact_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _issue_times_us(values: object, *, horizon_days: int, label: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for index, value in enumerate(_sequence(values, label=label)):
        local = _parse_datetime(value, label=f"{label}[{index}]")
        is_local_midnight = local.timetz().replace(tzinfo=None) == datetime.min.time()
        if local.utcoffset() != timedelta(hours=8) or not is_local_midnight:
            raise ValueError(f"{label}[{index}] must be local UTC+08 midnight")
        parsed.append(_datetime_us(local, label=f"{label}[{index}]"))
    if not parsed or parsed != sorted(set(parsed)):
        raise ValueError(f"{label} must be non-empty, unique, and chronological")
    gap = horizon_days * MICROSECONDS_PER_DAY
    if any(right - left < gap for left, right in pairwise(parsed)):
        raise ValueError(f"{label} contains overlapping target windows")
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class TargetSet:
    """Unique catalog events assigned to one fold/role/horizon issue axis."""

    role: Literal["fit", "assessment"]
    fold_id: str
    horizon_days: int
    issue_times_us: tuple[int, ...]
    event_indices: tuple[int, ...]
    assigned_issue_times_us: tuple[int, ...]
    event_identity_set_sha256: str
    late_available_target_count: int

    def __post_init__(self) -> None:
        if self.role not in {"fit", "assessment"}:
            raise ValueError("target role must be fit or assessment")
        if self.fold_id not in FOLD_IDS:
            raise ValueError("invalid target fold_id")
        if self.horizon_days <= 0:
            raise ValueError("target horizon must be positive")
        if len(self.event_indices) != len(self.assigned_issue_times_us):
            raise ValueError("event indices and assigned issue times must align")
        if len(self.event_indices) != len(set(self.event_indices)):
            raise ValueError("target events must be unique")
        if any(issue not in self.issue_times_us for issue in self.assigned_issue_times_us):
            raise ValueError("assigned target issue is outside the frozen issue axis")
        if self.late_available_target_count < 0:
            raise ValueError("late target count may not be negative")

    @property
    def event_count(self) -> int:
        return len(self.event_indices)

    def event_ids(self, catalog: Stage2SEarthquakeCatalog) -> tuple[str, ...]:
        return tuple(catalog.event_ids[index] for index in self.event_indices)


@dataclass(frozen=True, slots=True)
class ClusterRepresentative:
    """Earliest eligible member for one horizon, with its generating issue."""

    horizon_days: int
    event_index: int
    event_identity_sha256: str
    fold_id: str
    assigned_issue_time_us: int


@dataclass(frozen=True, slots=True)
class GlobalCluster:
    """One WGS84/time connected component over the 30d/90d target union."""

    identity_sha256: str
    member_event_indices: tuple[int, ...]
    member_event_identity_set_sha256: str
    representatives: tuple[ClusterRepresentative, ...]

    def representative(self, horizon_days: int) -> ClusterRepresentative | None:
        return next(
            (item for item in self.representatives if item.horizon_days == horizon_days),
            None,
        )


@dataclass(frozen=True, slots=True)
class D1TargetLayer:
    """Fully water-level-verified D1 fit targets and assessment clusters."""

    catalog: Stage2SEarthquakeCatalog
    catalog_freeze_us: int
    fit_targets: tuple[TargetSet, ...]
    assessment_targets: tuple[TargetSet, ...]
    clusters: tuple[GlobalCluster, ...]
    target_union_identity_set_sha256: str

    def fit_for(self, fold_id: str) -> TargetSet:
        return next(item for item in self.fit_targets if item.fold_id == fold_id)

    def assessment_for(self, fold_id: str, horizon_days: int) -> TargetSet:
        return next(
            item
            for item in self.assessment_targets
            if item.fold_id == fold_id and item.horizon_days == horizon_days
        )

    def active_clusters(self, fold_id: str, horizon_days: int) -> tuple[GlobalCluster, ...]:
        return tuple(
            cluster
            for cluster in self.clusters
            if (representative := cluster.representative(horizon_days)) is not None
            and representative.fold_id == fold_id
        )


def assign_target_events(
    catalog: Stage2SEarthquakeCatalog,
    *,
    role: Literal["fit", "assessment"],
    fold_id: str,
    horizon_days: int,
    issue_times_us: Sequence[int],
    catalog_freeze_us: int,
    magnitude_minimum_inclusive: float,
    magnitude_maximum_exclusive: float | None,
) -> TargetSet:
    """Assign eligible ``(T,T+h]`` events once, using no post-freeze record."""

    if fold_id not in FOLD_IDS:
        raise ValueError("invalid fold_id")
    issues = tuple(int(value) for value in issue_times_us)
    if not issues or issues != tuple(sorted(set(issues))):
        raise ValueError("issue times must be non-empty, unique, and sorted")
    horizon_us = horizon_days * MICROSECONDS_PER_DAY
    if any(right - left < horizon_us for left, right in pairwise(issues)):
        raise ValueError("target windows may not overlap")

    magnitude_mask = catalog.magnitude >= float(magnitude_minimum_inclusive)
    if magnitude_maximum_exclusive is not None:
        magnitude_mask &= catalog.magnitude < float(magnitude_maximum_exclusive)
    scientific_mask = catalog.inside_study_area & magnitude_mask
    frozen_mask = scientific_mask & (catalog.available_at_us <= int(catalog_freeze_us))
    late_mask = scientific_mask & (catalog.available_at_us > int(catalog_freeze_us))

    unassigned = np.iinfo(np.int64).min
    assigned_issue = np.full(catalog.row_count, unassigned, dtype=np.int64)
    assigned = np.zeros(catalog.row_count, dtype=np.bool_)
    late = np.zeros(catalog.row_count, dtype=np.bool_)
    for issue in issues:
        in_window = (catalog.origin_time_us > issue) & (
            catalog.origin_time_us <= issue + horizon_us
        )
        membership = frozen_mask & in_window
        if np.any(assigned & membership):
            raise ValueError("one target event matched more than one issue window")
        assigned[membership] = True
        assigned_issue[membership] = issue
        late |= late_mask & in_window

    indices = tuple(int(value) for value in np.flatnonzero(assigned))
    assigned_values = tuple(int(assigned_issue[index]) for index in indices)
    event_ids = tuple(catalog.event_ids[index] for index in indices)
    return TargetSet(
        role=role,
        fold_id=fold_id,
        horizon_days=horizon_days,
        issue_times_us=issues,
        event_indices=indices,
        assigned_issue_times_us=assigned_values,
        event_identity_set_sha256=event_set_identity_sha256(event_ids),
        late_available_target_count=int(np.count_nonzero(late)),
    )


class _DisjointSet:
    def __init__(self, values: Sequence[int]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self._parent[value]
        if parent != value:
            parent = self.find(parent)
            self._parent[value] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def build_global_clusters(
    catalog: Stage2SEarthquakeCatalog,
    assessment_targets: Sequence[TargetSet],
    *,
    max_time_seconds: int = CLUSTER_MAX_TIME_SECONDS,
    max_distance_metres: float = CLUSTER_MAX_DISTANCE_METRES,
) -> tuple[GlobalCluster, ...]:
    """Build transitive WGS84 connected components on the two-horizon union."""

    if max_time_seconds < 0 or max_distance_metres < 0.0:
        raise ValueError("cluster edge limits must be non-negative")
    by_horizon: dict[int, dict[int, tuple[str, int]]] = {
        horizon: {} for horizon in ASSESSMENT_HORIZONS_DAYS
    }
    for targets in assessment_targets:
        if targets.role != "assessment" or targets.horizon_days not in by_horizon:
            raise ValueError("global clusters require only 30d/90d assessment targets")
        memberships = by_horizon[targets.horizon_days]
        for event_index, issue_time in zip(
            targets.event_indices, targets.assigned_issue_times_us, strict=True
        ):
            if event_index in memberships:
                raise ValueError("assessment event repeats across folds within one horizon")
            memberships[event_index] = (targets.fold_id, issue_time)

    union_indices = tuple(sorted(set(by_horizon[30]) | set(by_horizon[90])))
    if not union_indices:
        return ()
    disjoint = _DisjointSet(union_indices)
    geod = Geod(ellps="WGS84")
    max_time_us = max_time_seconds * 1_000_000
    for position, left in enumerate(union_indices):
        for right in union_indices[position + 1 :]:
            time_difference = abs(
                int(catalog.origin_time_us[left]) - int(catalog.origin_time_us[right])
            )
            if time_difference > max_time_us:
                continue
            _, _, distance = geod.inv(
                float(catalog.longitude[left]),
                float(catalog.latitude[left]),
                float(catalog.longitude[right]),
                float(catalog.latitude[right]),
            )
            if abs(float(distance)) <= max_distance_metres:
                disjoint.union(left, right)

    members_by_root: dict[int, list[int]] = {}
    for event_index in union_indices:
        members_by_root.setdefault(disjoint.find(event_index), []).append(event_index)

    clusters: list[GlobalCluster] = []
    for raw_members in members_by_root.values():
        members = tuple(
            sorted(raw_members, key=lambda index: catalog.event_ids[index].encode("utf-8"))
        )
        event_ids = tuple(catalog.event_ids[index] for index in members)
        representatives: list[ClusterRepresentative] = []
        for horizon in ASSESSMENT_HORIZONS_DAYS:
            eligible = tuple(index for index in members if index in by_horizon[horizon])
            if not eligible:
                continue
            representative_index = min(
                eligible,
                key=lambda index: (
                    int(catalog.origin_time_us[index]),
                    catalog.event_ids[index].encode("utf-8"),
                ),
            )
            fold_id, issue_time = by_horizon[horizon][representative_index]
            representatives.append(
                ClusterRepresentative(
                    horizon_days=horizon,
                    event_index=representative_index,
                    event_identity_sha256=event_identity_sha256(
                        catalog.event_ids[representative_index]
                    ),
                    fold_id=fold_id,
                    assigned_issue_time_us=issue_time,
                )
            )
        clusters.append(
            GlobalCluster(
                identity_sha256=cluster_identity_sha256(event_ids),
                member_event_indices=members,
                member_event_identity_set_sha256=event_set_identity_sha256(event_ids),
                representatives=tuple(representatives),
            )
        )
    return tuple(sorted(clusters, key=lambda item: item.identity_sha256))


def _verify_equal(observed: object, expected: object, *, label: str) -> None:
    if observed != expected:
        raise ValueError(f"D1 water-level mismatch for {label}: {observed!r} != {expected!r}")


def _verify_manifest(
    layer: D1TargetLayer,
    manifest: Mapping[str, Any],
) -> None:
    _verify_equal(manifest.get("manifest_type"), "d1_fold_water_level_manifest", label="type")
    _verify_equal(manifest.get("schema_version"), 1, label="schema version")
    _verify_equal(manifest.get("target_window"), "(T,T+h]", label="target window")
    _verify_equal(manifest.get("model_effect_fields_read"), [], label="score blindness")
    _verify_equal(
        manifest.get("model_scoring_authorized_at_manifest_creation"),
        False,
        label="scoring authorization",
    )

    input_binding = _mapping(
        _mapping(manifest.get("input_bindings"), label="input_bindings").get("earthquake_event"),
        label="earthquake input binding",
    )
    _verify_equal(
        layer.catalog.identity.file_sha256,
        input_binding.get("file_sha256"),
        label="earthquake file SHA-256",
    )
    _verify_equal(layer.catalog.row_count, input_binding.get("row_count"), label="catalog rows")

    fold_records = _sequence(manifest.get("folds"), label="folds")
    _verify_equal(len(fold_records), len(FOLD_IDS), label="fold count")
    for raw_fold, fold_id in zip(fold_records, FOLD_IDS, strict=True):
        fold = _mapping(raw_fold, label=fold_id)
        _verify_equal(fold.get("fold_id"), fold_id, label=f"{fold_id} identity")
        fit_expected = _mapping(fold.get("fit"), label=f"{fold_id}.fit")
        fit = layer.fit_for(fold_id)
        _verify_equal(
            fit.event_count,
            fit_expected.get("m4_plus_target_count"),
            label=f"{fold_id} fit count",
        )
        _verify_equal(
            fit.event_identity_set_sha256,
            fit_expected.get("m4_plus_target_identity_set_sha256"),
            label=f"{fold_id} fit identity",
        )
        assessment_expected = _mapping(fold.get("assessment"), label=f"{fold_id}.assessment")
        for horizon in ASSESSMENT_HORIZONS_DAYS:
            expected = _mapping(
                assessment_expected.get(str(horizon)),
                label=f"{fold_id}.{horizon}d",
            )
            targets = layer.assessment_for(fold_id, horizon)
            _verify_equal(
                targets.event_count,
                expected.get("m5_6_target_count"),
                label=f"{fold_id} {horizon}d target count",
            )
            _verify_equal(
                targets.event_identity_set_sha256,
                expected.get("m5_6_target_identity_set_sha256"),
                label=f"{fold_id} {horizon}d target identity",
            )
            _verify_equal(
                targets.late_available_target_count,
                expected.get("late_available_target_count"),
                label=f"{fold_id} {horizon}d late targets",
            )
            active = layer.active_clusters(fold_id, horizon)
            _verify_equal(
                len(active),
                expected.get("active_cluster_count"),
                label=f"{fold_id} {horizon}d active cluster count",
            )
            active_identity = active_cluster_set_identity_sha256(
                [item.identity_sha256 for item in active],
                fold_id=fold_id,
                horizon_days=horizon,
            )
            _verify_equal(
                active_identity,
                expected.get("active_cluster_identity_set_sha256"),
                label=f"{fold_id} {horizon}d active cluster identity",
            )

    global_expected = _mapping(
        manifest.get("global_cluster_catalog"), label="global_cluster_catalog"
    )
    construction = _mapping(
        global_expected.get("construction"), label="global cluster construction"
    )
    _verify_equal(
        construction.get("graph_rule"),
        "undirected_connected_components",
        label="cluster graph rule",
    )
    _verify_equal(
        construction.get("distance_method"),
        "pyproj.Geod(ellps=WGS84).inv",
        label="cluster distance method",
    )
    _verify_equal(
        construction.get("edge_time_difference_max_seconds_inclusive"),
        CLUSTER_MAX_TIME_SECONDS,
        label="cluster time edge",
    )
    _verify_equal(
        construction.get("edge_distance_max_m_inclusive"),
        int(CLUSTER_MAX_DISTANCE_METRES),
        label="cluster distance edge",
    )
    union_indices = sorted(
        {index for targets in layer.assessment_targets for index in targets.event_indices}
    )
    _verify_equal(
        len(union_indices),
        global_expected.get("target_union_event_count"),
        label="target union count",
    )
    _verify_equal(
        layer.target_union_identity_set_sha256,
        global_expected.get("target_union_event_identity_set_sha256"),
        label="target union identity",
    )
    _verify_equal(len(layer.clusters), global_expected.get("cluster_count"), label="cluster count")
    size_distribution = {
        str(size): count
        for size, count in sorted(
            Counter(len(cluster.member_event_indices) for cluster in layer.clusters).items()
        )
    }
    _verify_equal(
        size_distribution,
        global_expected.get("cluster_size_distribution"),
        label="cluster size distribution",
    )

    member_folds: dict[int, set[str]] = {}
    for targets in layer.assessment_targets:
        for event_index in targets.event_indices:
            member_folds.setdefault(event_index, set()).add(targets.fold_id)

    observed_cluster_records: list[dict[str, object]] = []
    cross_fold_count = 0
    for cluster in layer.clusters:
        cluster_folds = {
            fold_id
            for event_index in cluster.member_event_indices
            for fold_id in member_folds[event_index]
        }
        if len(cluster_folds) > 1:
            cross_fold_count += 1
        representatives = {
            str(item.horizon_days): {
                "fold_id": item.fold_id,
                "representative_event_identity_sha256": item.event_identity_sha256,
            }
            for item in cluster.representatives
        }
        observed_cluster_records.append(
            {
                "cluster_identity_sha256": cluster.identity_sha256,
                "member_event_count": len(cluster.member_event_indices),
                "member_event_identity_set_sha256": cluster.member_event_identity_set_sha256,
                "representative_by_horizon": representatives,
            }
        )
    _verify_equal(
        observed_cluster_records,
        global_expected.get("cluster_records"),
        label="cluster records",
    )
    _verify_equal(
        cross_fold_count,
        global_expected.get("cross_fold_cluster_count"),
        label="cross-fold clusters",
    )

    active_by_horizon = {
        str(horizon): sum(cluster.representative(horizon) is not None for cluster in layer.clusters)
        for horizon in ASSESSMENT_HORIZONS_DAYS
    }
    _verify_equal(
        active_by_horizon,
        global_expected.get("active_cluster_count_by_horizon"),
        label="active clusters by horizon",
    )
    active_by_fold = {
        str(horizon): {
            fold_id: len(layer.active_clusters(fold_id, horizon)) for fold_id in FOLD_IDS
        }
        for horizon in ASSESSMENT_HORIZONS_DAYS
    }
    _verify_equal(
        active_by_fold,
        global_expected.get("active_cluster_count_by_horizon_and_representative_fold"),
        label="active clusters by horizon and fold",
    )


def build_score_blind_target_layer(
    protocol: D1Protocol,
    catalog: Stage2SEarthquakeCatalog,
) -> D1TargetLayer:
    """Build and verify all D1 target memberships before any score is read."""

    if not isinstance(protocol, D1Protocol):
        raise TypeError("protocol must be a validated D1Protocol")
    if not isinstance(catalog, Stage2SEarthquakeCatalog):
        raise TypeError("catalog must be a verified Stage2SEarthquakeCatalog")
    manifest = protocol.water_level
    truth = _mapping(manifest.get("truth_coverage"), label="truth_coverage")
    freeze = _parse_datetime(truth.get("catalog_available_at_max_utc"), label="catalog freeze")
    freeze_us = _datetime_us(freeze, label="catalog freeze")

    fit_targets: list[TargetSet] = []
    assessment_targets: list[TargetSet] = []
    folds = _sequence(manifest.get("folds"), label="folds")
    for raw_fold, expected_fold_id in zip(folds, FOLD_IDS, strict=True):
        fold = _mapping(raw_fold, label=expected_fold_id)
        fold_id = fold.get("fold_id")
        _verify_equal(fold_id, expected_fold_id, label=f"{expected_fold_id} order")
        fit = _mapping(fold.get("fit"), label=f"{expected_fold_id}.fit")
        fit_horizon = _exact_int(
            fit.get("target_horizon_days"), label=f"{expected_fold_id}.fit horizon"
        )
        fit_issues = _issue_times_us(
            fit.get("selected_issue_times_local"),
            horizon_days=fit_horizon,
            label=f"{expected_fold_id}.fit issues",
        )
        _verify_equal(
            len(fit_issues),
            fit.get("selected_issue_count"),
            label=f"{expected_fold_id} fit issue count",
        )
        fit_maturity_us = _datetime_us(
            _parse_datetime(
                fit.get("target_maturity_end_local"),
                label=f"{expected_fold_id}.fit maturity",
            ),
            label=f"{expected_fold_id}.fit maturity",
        )
        _verify_equal(
            fit_maturity_us,
            fit_issues[-1] + fit_horizon * MICROSECONDS_PER_DAY,
            label=f"{expected_fold_id} fit maturity",
        )
        fit_targets.append(
            assign_target_events(
                catalog,
                role="fit",
                fold_id=expected_fold_id,
                horizon_days=fit_horizon,
                issue_times_us=fit_issues,
                catalog_freeze_us=freeze_us,
                magnitude_minimum_inclusive=4.0,
                magnitude_maximum_exclusive=None,
            )
        )

        assessment = _mapping(fold.get("assessment"), label=f"{expected_fold_id}.assessment")
        for horizon in ASSESSMENT_HORIZONS_DAYS:
            expected = _mapping(
                assessment.get(str(horizon)),
                label=f"{expected_fold_id}.{horizon}d",
            )
            issues = _issue_times_us(
                expected.get("selected_issue_times_local"),
                horizon_days=horizon,
                label=f"{expected_fold_id}.{horizon}d issues",
            )
            _verify_equal(
                len(issues),
                expected.get("selected_issue_count"),
                label=f"{expected_fold_id} {horizon}d issue count",
            )
            maturity_us = _datetime_us(
                _parse_datetime(
                    expected.get("target_maturity_end_local"),
                    label=f"{expected_fold_id}.{horizon}d maturity",
                ),
                label=f"{expected_fold_id}.{horizon}d maturity",
            )
            _verify_equal(
                maturity_us,
                issues[-1] + horizon * MICROSECONDS_PER_DAY,
                label=f"{expected_fold_id} {horizon}d maturity",
            )
            assessment_targets.append(
                assign_target_events(
                    catalog,
                    role="assessment",
                    fold_id=expected_fold_id,
                    horizon_days=horizon,
                    issue_times_us=issues,
                    catalog_freeze_us=freeze_us,
                    magnitude_minimum_inclusive=5.0,
                    magnitude_maximum_exclusive=6.0,
                )
            )

    clusters = build_global_clusters(catalog, assessment_targets)
    union_event_ids = tuple(
        catalog.event_ids[index]
        for index in sorted(
            {index for targets in assessment_targets for index in targets.event_indices}
        )
    )
    layer = D1TargetLayer(
        catalog=catalog,
        catalog_freeze_us=freeze_us,
        fit_targets=tuple(fit_targets),
        assessment_targets=tuple(assessment_targets),
        clusters=clusters,
        target_union_identity_set_sha256=event_set_identity_sha256(union_event_ids),
    )
    _verify_manifest(layer, manifest)
    return layer


__all__ = [
    "ACTIVE_CLUSTER_SET_DOMAIN",
    "ASSESSMENT_HORIZONS_DAYS",
    "CLUSTER_IDENTITY_DOMAIN",
    "CLUSTER_MAX_DISTANCE_METRES",
    "CLUSTER_MAX_TIME_SECONDS",
    "EVENT_IDENTITY_DOMAIN",
    "EVENT_SET_IDENTITY_DOMAIN",
    "FOLD_IDS",
    "MICROSECONDS_PER_DAY",
    "D1TargetLayer",
    "GlobalCluster",
    "TargetSet",
    "active_cluster_set_identity_sha256",
    "assign_target_events",
    "build_global_clusters",
    "build_score_blind_target_layer",
    "cluster_identity_sha256",
    "event_identity_sha256",
    "event_set_identity_sha256",
]
