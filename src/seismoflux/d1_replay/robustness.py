"""Descriptive regional and leave-one-cluster-out diagnostics for D1.

This module never fits or scores a model.  It consumes the completed observed
retrospective replay and asks a deliberately small question at the frozen
30-day, 600,000 km2 endpoint: is each anomaly-component hit gain spread over
more than one target-independent construction zone and more than one physical
cluster?  Target locations are represented only by the post-hoc evaluation
cell already recorded in the observed replay; they never generate an alarm,
candidate, grid, zone, or model feature here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, cast

import pyarrow.parquet as pq

from seismoflux.d1_replay.protocol import D1Protocol, load_d1_protocol, sha256_file
from seismoflux.data.common import canonical_json_bytes, write_json_atomic

D1_ROBUSTNESS_CONTRASTS: Final = (
    ("B0_C_A_snapshot_minus_B0_C", "B0_C_A_snapshot", "B0_C"),
    ("B0_C_A_dynamic_minus_B0_C", "B0_C_A_dynamic", "B0_C"),
    (
        "B0_C_A_dynamic_minus_B0_C_A_snapshot",
        "B0_C_A_dynamic",
        "B0_C_A_snapshot",
    ),
)

_PROTOCOL_VERSION: Final = "d1.0.0"
_FOLD_IDS: Final = ("fold_1", "fold_2", "fold_3")
_MODEL_IDS: Final = (
    "B0",
    "B0_R30",
    "B0_C",
    "B0_C_A_snapshot",
    "B0_C_A_dynamic",
    "B0_R30_C_A_dynamic",
)
_PRIMARY_HORIZON_DAYS: Final = 30
_PRIMARY_AREA_KM2: Final = 600_000
_PRIMARY_AREA_INDEX: Final = 2
_AREA_COUNT: Final = 5
_CLUSTER_COUNT: Final = 21
_ZONE_COUNT: Final = 39
_CELL_COUNT: Final = 15_697
_FOLD_CLUSTER_COUNTS: Final = {"fold_1": 8, "fold_2": 6, "fold_3": 7}
_SHA256_HEX_LENGTH: Final = 64
_GIT_HEX_LENGTH: Final = 40
_CELL_MAPPING_COLUMNS: Final = (
    "grid_id",
    "cell_id",
    "cell_row",
    "cell_column",
    "query_x_m",
    "query_y_m",
    "construction_zone_id",
)
_ZONE_GEOMETRY_COLUMNS: Final = (
    "construction_zone_id",
    "geometry_wkb_equal_area_m",
)
_SPATIAL_ARTIFACT_NAMES: Final = (
    "cell_mapping",
    "entity_mapping",
    "zone_geometry",
    "connectors",
)


class D1RobustnessError(ValueError):
    """Raised when a D1 robustness input fails a frozen identity or axis check."""


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise D1RobustnessError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise D1RobustnessError(f"{label} must be a list")
    return cast(Sequence[object], value)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise D1RobustnessError(f"{label} must be a non-empty string")
    return value


def _exact_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise D1RobustnessError(f"{label} must be an integer")
    return value


def _hex_identity(value: object, *, length: int, label: str) -> str:
    identity = _string(value, label=label)
    if len(identity) != length or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise D1RobustnessError(f"{label} must be lowercase hexadecimal")
    return identity


def _sha256_identity(value: object, *, label: str) -> str:
    return _hex_identity(value, length=_SHA256_HEX_LENGTH, label=label)


def _bool_axis(value: object, *, label: str) -> tuple[bool, ...]:
    sequence = _sequence(value, label=label)
    if len(sequence) != _AREA_COUNT or any(type(item) is not bool for item in sequence):
        raise D1RobustnessError(f"{label} must contain five booleans")
    hits = tuple(cast(bool, item) for item in sequence)
    if any(left and not right for left, right in pairwise(hits)):
        raise D1RobustnessError(f"{label} must be monotone across alarm areas")
    return hits


@dataclass(frozen=True, slots=True)
class _PrimaryOutcome:
    cluster_id: str
    fold_id: str
    issue_id: str
    model_id: str
    representative_cell_index: int | None
    outside_support: bool
    hit: bool


@dataclass(frozen=True, slots=True)
class _ObservedPrimary:
    identities: Mapping[str, Any]
    cluster_order: tuple[str, ...]
    fold_by_cluster: Mapping[str, str]
    issue_by_cluster: Mapping[str, str]
    outcomes: Mapping[tuple[str, str], _PrimaryOutcome]


def _validate_observed_header(
    observed: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
    expected_manifest_content_sha256: str,
) -> Mapping[str, Any]:
    if (
        observed.get("schema_version") != 1
        or observed.get("protocol_version") != _PROTOCOL_VERSION
        or observed.get("result_kind") != "observed_replay"
        or observed.get("status") != "completed"
    ):
        raise D1RobustnessError("robustness requires the completed D1 observed replay")
    if observed.get("retrospective_only") is not True:
        raise D1RobustnessError("observed replay must remain retrospective-only")
    if observed.get("relative_strength_not_absolute_probability") is not True:
        raise D1RobustnessError("observed replay changed its relative-strength semantics")

    identities = _mapping(observed.get("identities"), label="observed identities")
    contract = _sha256_identity(identities.get("contract_sha256"), label="observed contract_sha256")
    manifest = _sha256_identity(
        identities.get("manifest_content_sha256"),
        label="observed manifest_content_sha256",
    )
    _sha256_identity(identities.get("input_sha256"), label="observed input_sha256")
    _hex_identity(
        identities.get("git_commit"),
        length=_GIT_HEX_LENGTH,
        label="observed git_commit",
    )
    if contract != expected_contract_sha256:
        raise D1RobustnessError("observed contract identity differs from the frozen config")
    if manifest != expected_manifest_content_sha256:
        raise D1RobustnessError("observed manifest identity differs from the frozen water level")
    return identities


def _primary_support(
    observed: Mapping[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    support_by_horizon = _mapping(
        observed.get("expected_support_by_horizon"),
        label="expected_support_by_horizon",
    )
    raw_support = _sequence(
        support_by_horizon.get(str(_PRIMARY_HORIZON_DAYS)),
        label="30d expected support",
    )
    support: list[tuple[str, str, str]] = []
    for index, raw in enumerate(raw_support):
        item = _mapping(raw, label=f"30d support[{index}]")
        cluster_id = _sha256_identity(
            item.get("cluster_id"), label=f"30d support[{index}].cluster_id"
        )
        fold_id = _string(item.get("fold_id"), label=f"30d support[{index}].fold_id")
        issue_id = _string(item.get("issue_id"), label=f"30d support[{index}].issue_id")
        if fold_id not in _FOLD_IDS:
            raise D1RobustnessError("30d support contains an unknown fold")
        support.append((cluster_id, fold_id, issue_id))
    ordered = tuple(sorted(support))
    if len(ordered) != _CLUSTER_COUNT or len(set(ordered)) != _CLUSTER_COUNT:
        raise D1RobustnessError("30d support must contain exactly 21 unique clusters")
    if tuple(support) != ordered:
        raise D1RobustnessError("30d support changed its canonical cluster order")
    if Counter(item[1] for item in ordered) != Counter(_FOLD_CLUSTER_COUNTS):
        raise D1RobustnessError("30d cluster support changed its frozen 8/6/7 fold allocation")
    if len({item[0] for item in ordered}) != _CLUSTER_COUNT:
        raise D1RobustnessError("one 30d physical cluster appears in multiple support rows")
    return ordered


def _parse_primary_outcome(raw: Mapping[str, Any], *, index: int) -> _PrimaryOutcome:
    cluster_id = _sha256_identity(raw.get("cluster_id"), label=f"outcomes[{index}].cluster_id")
    fold_id = _string(raw.get("fold_id"), label=f"outcomes[{index}].fold_id")
    issue_id = _string(raw.get("issue_id"), label=f"outcomes[{index}].issue_id")
    model_id = _string(raw.get("model_id"), label=f"outcomes[{index}].model_id")
    if fold_id not in _FOLD_IDS or model_id not in _MODEL_IDS:
        raise D1RobustnessError("30d outcome has an unregistered fold or model")
    outside = raw.get("outside_support")
    if type(outside) is not bool:
        raise D1RobustnessError("30d outcome outside_support must be boolean")
    representative_raw = raw.get("representative_cell_index")
    representative = (
        None
        if representative_raw is None
        else _exact_int(representative_raw, label="representative_cell_index")
    )
    hits = _bool_axis(raw.get("hit_by_area"), label=f"outcomes[{index}].hit_by_area")
    if outside:
        if representative is not None or any(hits) or raw.get("log_density") is not None:
            raise D1RobustnessError("outside-support outcome carries an inside-support result")
    else:
        if representative is None or representative < 0:
            raise D1RobustnessError("inside-support outcome omitted its representative cell")
        log_density = raw.get("log_density")
        if isinstance(log_density, bool) or not isinstance(log_density, int | float):
            raise D1RobustnessError("inside-support outcome omitted finite log density")
        if not math.isfinite(float(log_density)):
            raise D1RobustnessError("inside-support outcome has nonfinite log density")
    return _PrimaryOutcome(
        cluster_id=cluster_id,
        fold_id=fold_id,
        issue_id=issue_id,
        model_id=model_id,
        representative_cell_index=representative,
        outside_support=outside,
        hit=hits[_PRIMARY_AREA_INDEX],
    )


def _validate_observed_primary(
    observed: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
    expected_manifest_content_sha256: str,
) -> _ObservedPrimary:
    identities = _validate_observed_header(
        observed,
        expected_contract_sha256=expected_contract_sha256,
        expected_manifest_content_sha256=expected_manifest_content_sha256,
    )
    support = _primary_support(observed)
    support_by_cluster = {cluster: (fold, issue) for cluster, fold, issue in support}
    outcomes: dict[tuple[str, str], _PrimaryOutcome] = {}
    for index, raw in enumerate(_sequence(observed.get("outcomes"), label="outcomes")):
        item = _mapping(raw, label=f"outcomes[{index}]")
        if _exact_int(item.get("horizon_days"), label=f"outcomes[{index}].horizon_days") != 30:
            continue
        parsed = _parse_primary_outcome(item, index=index)
        key = (parsed.cluster_id, parsed.model_id)
        if key in outcomes:
            raise D1RobustnessError("30d observed outcomes contain a duplicate cluster/model")
        outcomes[key] = parsed

    expected_keys = {
        (cluster_id, model_id) for cluster_id in support_by_cluster for model_id in _MODEL_IDS
    }
    if set(outcomes) != expected_keys:
        raise D1RobustnessError("30d observed outcomes do not contain 21 clusters x six models")
    for cluster_id, (fold_id, issue_id) in support_by_cluster.items():
        rows = tuple(outcomes[(cluster_id, model_id)] for model_id in _MODEL_IDS)
        if any(row.fold_id != fold_id or row.issue_id != issue_id for row in rows):
            raise D1RobustnessError("30d outcome changed its frozen support fold or issue")
        coordinates = {(row.representative_cell_index, row.outside_support) for row in rows}
        if len(coordinates) != 1:
            raise D1RobustnessError("models disagree on the post-hoc target cell mapping")
    return _ObservedPrimary(
        identities=dict(identities),
        cluster_order=tuple(item[0] for item in support),
        fold_by_cluster={item[0]: item[1] for item in support},
        issue_by_cluster={item[0]: item[2] for item in support},
        outcomes=outcomes,
    )


def _sign_counts(values: Sequence[int]) -> dict[str, int]:
    return {
        "positive_count": sum(value > 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
    }


def _regional_diagnostic(
    primary: _ObservedPrimary,
    *,
    candidate_model_id: str,
    reference_model_id: str,
    zone_ids: tuple[str, ...],
    zone_by_cluster: Mapping[str, str | None],
) -> dict[str, object]:
    gains: dict[str, int] = {}
    candidate_hits: dict[str, int] = {}
    reference_hits: dict[str, int] = {}
    for cluster_id in primary.cluster_order:
        candidate = int(primary.outcomes[(cluster_id, candidate_model_id)].hit)
        reference = int(primary.outcomes[(cluster_id, reference_model_id)].hit)
        candidate_hits[cluster_id] = candidate
        reference_hits[cluster_id] = reference
        gains[cluster_id] = candidate - reference

    zone_rows: list[dict[str, object]] = []
    for zone_id in zone_ids:
        clusters = tuple(
            cluster for cluster in primary.cluster_order if zone_by_cluster[cluster] == zone_id
        )
        zone_gains = tuple(gains[cluster] for cluster in clusters)
        gain_sum = sum(zone_gains)
        zone_rows.append(
            {
                "zone_id": zone_id,
                "target_cluster_count": len(clusters),
                "candidate_hit_count": sum(candidate_hits[cluster] for cluster in clusters),
                "reference_hit_count": sum(reference_hits[cluster] for cluster in clusters),
                "hit_gain_sum": gain_sum,
                "additive_recall_gain_on_full_21_cluster_denominator": gain_sum / _CLUSTER_COUNT,
                "within_zone_recall_gain": (None if not clusters else gain_sum / len(clusters)),
                "cluster_gain_sign_counts": _sign_counts(zone_gains),
                "cluster_ids": list(clusters),
            }
        )

    target_rows = tuple(row for row in zone_rows if cast(int, row["target_cluster_count"]) > 0)
    zone_gain_values = tuple(cast(int, row["hit_gain_sum"]) for row in target_rows)
    positive_rows = tuple(row for row in target_rows if cast(int, row["hit_gain_sum"]) > 0)
    largest_positive = (
        None
        if not positive_rows
        else min(
            positive_rows,
            key=lambda row: (-cast(int, row["hit_gain_sum"]), cast(str, row["zone_id"])),
        )
    )
    total_gain = sum(gains.values())
    positive_gain_total = sum(max(value, 0) for value in zone_gain_values)
    if largest_positive is None:
        largest_summary: dict[str, object] | None = None
        survives: bool | None = None
        single_zone_dominant = False
    else:
        removed_gain = cast(int, largest_positive["hit_gain_sum"])
        removed_clusters = cast(int, largest_positive["target_cluster_count"])
        remaining_count = _CLUSTER_COUNT - removed_clusters
        remaining_gain = total_gain - removed_gain
        survives = remaining_count > 0 and remaining_gain > 0
        single_zone_dominant = total_gain > 0 and not survives
        largest_summary = {
            "zone_id": largest_positive["zone_id"],
            "target_cluster_count": removed_clusters,
            "hit_gain_sum": removed_gain,
            "fraction_of_all_positive_zone_gain": removed_gain / positive_gain_total,
            "remaining_cluster_count": remaining_count,
            "remaining_hit_gain_sum": remaining_gain,
            "remaining_recall_gain": (
                None if remaining_count == 0 else remaining_gain / remaining_count
            ),
        }

    return {
        "zone_count": _ZONE_COUNT,
        "target_bearing_zone_count": len(target_rows),
        "outside_support_cluster_count": sum(
            zone_by_cluster[cluster] is None for cluster in primary.cluster_order
        ),
        "target_bearing_zone_gain_sign_counts": _sign_counts(zone_gain_values),
        "positive_zone_gain_sum": positive_gain_total,
        "largest_positive_zone": largest_summary,
        "direction_survives_largest_positive_zone_removal": survives,
        "single_zone_direction_dominant": single_zone_dominant,
        "additive_recall_gain_closure": sum(
            cast(float, row["additive_recall_gain_on_full_21_cluster_denominator"])
            for row in zone_rows
        ),
        "zone_rows": zone_rows,
    }


def _leave_one_cluster_out(
    primary: _ObservedPrimary,
    *,
    candidate_model_id: str,
    reference_model_id: str,
    zone_by_cluster: Mapping[str, str | None],
) -> dict[str, object]:
    candidate_hits = {
        cluster: int(primary.outcomes[(cluster, candidate_model_id)].hit)
        for cluster in primary.cluster_order
    }
    reference_hits = {
        cluster: int(primary.outcomes[(cluster, reference_model_id)].hit)
        for cluster in primary.cluster_order
    }
    candidate_total = sum(candidate_hits.values())
    reference_total = sum(reference_hits.values())
    rows: list[dict[str, object]] = []
    remaining_gain_sums: list[int] = []
    for cluster_id in primary.cluster_order:
        candidate_remaining = candidate_total - candidate_hits[cluster_id]
        reference_remaining = reference_total - reference_hits[cluster_id]
        gain_sum = candidate_remaining - reference_remaining
        remaining_gain_sums.append(gain_sum)
        rows.append(
            {
                "omitted_cluster_id": cluster_id,
                "fold_id": primary.fold_by_cluster[cluster_id],
                "issue_id": primary.issue_by_cluster[cluster_id],
                "posthoc_zone_id": zone_by_cluster[cluster_id],
                "omitted_cluster_hit_gain": candidate_hits[cluster_id] - reference_hits[cluster_id],
                "remaining_cluster_count": _CLUSTER_COUNT - 1,
                "remaining_candidate_hit_count": candidate_remaining,
                "remaining_reference_hit_count": reference_remaining,
                "remaining_hit_gain_sum": gain_sum,
                "remaining_recall_gain": gain_sum / (_CLUSTER_COUNT - 1),
            }
        )
    recall_gains = tuple(value / (_CLUSTER_COUNT - 1) for value in remaining_gain_sums)
    return {
        "replication_count": _CLUSTER_COUNT,
        "remaining_cluster_count_per_replication": _CLUSTER_COUNT - 1,
        "recall_gain_minimum": min(recall_gains),
        "recall_gain_maximum": max(recall_gains),
        "recall_gain_sign_counts": _sign_counts(remaining_gain_sums),
        "direction_survives_every_cluster_removal": all(value > 0 for value in remaining_gain_sums),
        "single_cluster_direction_dominant": (
            candidate_total - reference_total > 0 and min(remaining_gain_sums) <= 0
        ),
        "rows": rows,
    }


def build_d1_robustness_result(
    observed: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
    expected_manifest_content_sha256: str,
    zone_ids: Sequence[str],
    zone_id_by_cell_index: Sequence[str],
    spatial_strata_identity: Mapping[str, Any],
) -> dict[str, object]:
    """Build the canonical descriptive robustness result from verified inputs.

    ``zone_id_by_cell_index`` must retain the frozen operational-grid row order.
    This function uses the target representative cell only after the alarms and
    hit flags already exist in ``observed``.
    """

    expected_contract = _sha256_identity(
        expected_contract_sha256, label="expected contract SHA-256"
    )
    expected_manifest = _sha256_identity(
        expected_manifest_content_sha256,
        label="expected manifest content SHA-256",
    )
    zones = tuple(sorted(set(zone_ids)))
    if len(zones) != _ZONE_COUNT or any(
        _sha256_identity(zone, label="construction zone ID") != zone for zone in zones
    ):
        raise D1RobustnessError("robustness requires exactly 39 SHA-256 construction zones")
    cell_zones = tuple(zone_id_by_cell_index)
    if not cell_zones or any(zone not in zones for zone in cell_zones):
        raise D1RobustnessError("cell-zone axis contains an unknown construction zone")
    if set(cell_zones) != set(zones):
        raise D1RobustnessError("cell-zone axis does not cover all 39 nonempty zones")

    primary = _validate_observed_primary(
        observed,
        expected_contract_sha256=expected_contract,
        expected_manifest_content_sha256=expected_manifest,
    )
    zone_by_cluster: dict[str, str | None] = {}
    for cluster_id in primary.cluster_order:
        outcome = primary.outcomes[(cluster_id, "B0")]
        cell_index = outcome.representative_cell_index
        if outcome.outside_support:
            zone_by_cluster[cluster_id] = None
        elif cell_index is None or cell_index >= len(cell_zones):
            raise D1RobustnessError("post-hoc target cell index is outside the frozen grid")
        else:
            zone_by_cluster[cluster_id] = cell_zones[cell_index]

    contrasts: list[dict[str, object]] = []
    for contrast_id, candidate_model_id, reference_model_id in D1_ROBUSTNESS_CONTRASTS:
        candidate_count = sum(
            primary.outcomes[(cluster, candidate_model_id)].hit for cluster in primary.cluster_order
        )
        reference_count = sum(
            primary.outcomes[(cluster, reference_model_id)].hit for cluster in primary.cluster_order
        )
        gain_sum = candidate_count - reference_count
        regional = _regional_diagnostic(
            primary,
            candidate_model_id=candidate_model_id,
            reference_model_id=reference_model_id,
            zone_ids=zones,
            zone_by_cluster=zone_by_cluster,
        )
        if not math.isclose(
            cast(float, regional["additive_recall_gain_closure"]),
            gain_sum / _CLUSTER_COUNT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise D1RobustnessError("regional additive contributions do not close")
        contrasts.append(
            {
                "contrast_id": contrast_id,
                "candidate_model_id": candidate_model_id,
                "reference_model_id": reference_model_id,
                "observed_candidate_hit_count": candidate_count,
                "observed_reference_hit_count": reference_count,
                "observed_hit_gain_sum": gain_sum,
                "observed_recall_gain": gain_sum / _CLUSTER_COUNT,
                "regional": regional,
                "leave_one_cluster_out": _leave_one_cluster_out(
                    primary,
                    candidate_model_id=candidate_model_id,
                    reference_model_id=reference_model_id,
                    zone_by_cluster=zone_by_cluster,
                ),
            }
        )

    return {
        "schema_version": 1,
        "protocol_version": _PROTOCOL_VERSION,
        "result_kind": "d1_regional_and_leave_one_cluster_robustness",
        "status": "completed",
        "retrospective_only": True,
        "identities": dict(primary.identities),
        "regional_diagnostic_completed": True,
        "leave_one_cluster_out_completed": True,
        "model_refit_performed": False,
        "locked_test_read": False,
        "target_mapping_role": (
            "posthoc_observed_cluster_to_target_independent_zone_only_never_alarm_generation"
        ),
        "primary_endpoint": {
            "horizon_days": _PRIMARY_HORIZON_DAYS,
            "alarm_area_km2": _PRIMARY_AREA_KM2,
            "cluster_count": _CLUSTER_COUNT,
            "fold_cluster_counts": dict(_FOLD_CLUSTER_COUNTS),
            "construction_zone_count": _ZONE_COUNT,
        },
        "spatial_strata_identity": dict(spatial_strata_identity),
        "zone_axis": list(zones),
        "contrasts": contrasts,
    }


def _resolve_data_path(protocol: D1Protocol, raw_path: object, *, label: str) -> Path:
    relative = Path(_string(raw_path, label=label))
    if relative.is_absolute() or ".." in relative.parts:
        raise D1RobustnessError(f"{label} must remain repository-relative")
    local = (protocol.repository_root / relative).resolve()
    if local.is_file():
        return local
    data = _mapping(protocol.config.get("data"), label="D1 data")
    configured_root = Path(
        _string(data.get("local_data_root_current_machine"), label="local data root")
    ).resolve()
    if not relative.parts or relative.parts[0].lower() != "data":
        raise FileNotFoundError(local)
    shared = (configured_root / Path(*relative.parts[1:])).resolve()
    try:
        shared.relative_to(configured_root)
    except ValueError as exc:
        raise D1RobustnessError(f"{label} escapes the configured data root") from exc
    if not shared.is_file():
        raise FileNotFoundError(shared)
    return shared


def _verify_public_spatial_manifest(
    path: Path,
    *,
    expected_content_sha256: str,
) -> Mapping[str, Any]:
    try:
        payload = _mapping(json.loads(path.read_text(encoding="utf-8")), label="spatial manifest")
    except (OSError, json.JSONDecodeError) as exc:
        raise D1RobustnessError("spatial manifest is not readable JSON") from exc
    stated = _sha256_identity(payload.get("content_sha256"), label="spatial content SHA-256")
    body = dict(payload)
    body.pop("content_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if stated != actual or actual != expected_content_sha256:
        raise D1RobustnessError("spatial public-manifest content SHA-256 changed")
    aggregate = _mapping(payload.get("aggregate"), label="spatial manifest aggregate")
    if (
        payload.get("nonempty_stratum_count") != _ZONE_COUNT
        or aggregate.get("assigned_nonempty_zone_count") != _ZONE_COUNT
        or aggregate.get("assigned_query_cell_count") != _CELL_COUNT
        or aggregate.get("zone_count") != 65
    ):
        raise D1RobustnessError("spatial manifest changed its frozen 39/65/15697 water level")
    return payload


def _verify_spatial_inputs(
    protocol: D1Protocol,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, object]]:
    data = _mapping(protocol.config.get("data"), label="D1 data")
    spatial = _mapping(data.get("spatial_strata"), label="D1 spatial_strata")
    if (
        spatial.get("target_independent_nonempty_zone_count") != _ZONE_COUNT
        or spatial.get("may_not_enter_model_features") is not True
    ):
        raise D1RobustnessError("D1 spatial-strata scientific role changed")
    local_paths = _mapping(
        spatial.get("local_coordinate_artifacts_not_committed"),
        label="spatial local artifact paths",
    )
    expected_hashes = _mapping(
        spatial.get("local_artifact_sha256"),
        label="spatial local artifact SHA-256",
    )
    if set(local_paths) != set(_SPATIAL_ARTIFACT_NAMES) or set(expected_hashes) != set(
        _SPATIAL_ARTIFACT_NAMES
    ):
        raise D1RobustnessError("D1 spatial artifact axis changed")
    paths = {
        name: _resolve_data_path(protocol, local_paths[name], label=f"spatial {name}")
        for name in _SPATIAL_ARTIFACT_NAMES
    }
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    frozen_hashes = {
        name: _sha256_identity(expected_hashes[name], label=f"spatial {name} SHA-256")
        for name in _SPATIAL_ARTIFACT_NAMES
    }
    if observed_hashes != frozen_hashes:
        raise D1RobustnessError("one or more spatial artifact SHA-256 identities changed")

    public_path = _resolve_data_path(
        protocol,
        spatial.get("public_manifest"),
        label="spatial public manifest",
    )
    public_content_sha = _sha256_identity(
        spatial.get("public_manifest_content_sha256"),
        label="spatial public-manifest content SHA-256",
    )
    _verify_public_spatial_manifest(
        public_path,
        expected_content_sha256=public_content_sha,
    )

    cell_file = pq.ParquetFile(paths["cell_mapping"])
    if tuple(cell_file.schema_arrow.names) != _CELL_MAPPING_COLUMNS:
        raise D1RobustnessError("cell-zone mapping schema changed")
    if cell_file.metadata.num_rows != _CELL_COUNT:
        raise D1RobustnessError("cell-zone mapping no longer has 15,697 rows")
    cell_table = pq.read_table(paths["cell_mapping"], use_threads=False)
    grid_ids = cell_table["grid_id"].combine_chunks().to_pylist()
    cell_ids = cell_table["cell_id"].combine_chunks().to_pylist()
    cell_zones_raw = cell_table["construction_zone_id"].combine_chunks().to_pylist()
    if len(set(grid_ids)) != 1 or len(set(cell_ids)) != _CELL_COUNT:
        raise D1RobustnessError("cell-zone mapping changed its grid or cell identities")
    if any(not isinstance(value, str) or not value for value in cell_zones_raw):
        raise D1RobustnessError("cell-zone mapping contains an invalid zone identity")
    cell_zones = tuple(cast(list[str], cell_zones_raw))
    zones = tuple(sorted(set(cell_zones)))
    if len(zones) != _ZONE_COUNT:
        raise D1RobustnessError("cell-zone mapping no longer contains 39 nonempty zones")

    geometry_file = pq.ParquetFile(paths["zone_geometry"])
    if tuple(geometry_file.schema_arrow.names) != _ZONE_GEOMETRY_COLUMNS:
        raise D1RobustnessError("construction-zone geometry schema changed")
    if geometry_file.metadata.num_rows != 65:
        raise D1RobustnessError("construction-zone geometry no longer contains 65 polygons")
    geometry_table = pq.read_table(paths["zone_geometry"], use_threads=False)
    geometry_zones_raw = geometry_table["construction_zone_id"].combine_chunks().to_pylist()
    geometry_wkb = geometry_table["geometry_wkb_equal_area_m"].combine_chunks().to_pylist()
    if (
        any(not isinstance(value, str) or not value for value in geometry_zones_raw)
        or len(set(geometry_zones_raw)) != 65
        or any(not isinstance(value, bytes) or not value for value in geometry_wkb)
        or not set(zones).issubset(set(geometry_zones_raw))
    ):
        raise D1RobustnessError("construction-zone geometry has invalid or missing zone rows")

    return (
        zones,
        cell_zones,
        {
            "public_manifest_content_sha256": public_content_sha,
            "artifact_sha256": observed_hashes,
            "operational_grid_id": cast(str, grid_ids[0]),
            "operational_cell_count": _CELL_COUNT,
            "nonempty_zone_count": _ZONE_COUNT,
            "geometry_zone_count": 65,
            "zero_cell_geometry_zone_count": 26,
        },
    )


def _validate_observed_input_binding(
    observed: Mapping[str, Any],
    protocol: D1Protocol,
) -> None:
    identities = _mapping(observed.get("identities"), label="observed identities")
    observed_files = _mapping(identities.get("input_files"), label="observed input_files")
    data = _mapping(protocol.config.get("data"), label="D1 data")
    expected = {
        "earthquake_event": _sha256_identity(
            _mapping(data.get("earthquake_event"), label="earthquake_event").get("file_sha256"),
            label="earthquake event SHA-256",
        ),
        "study_area": _sha256_identity(
            _mapping(data.get("study_area"), label="study_area").get("file_sha256"),
            label="study area SHA-256",
        ),
        "anomaly_features": _sha256_identity(
            _mapping(data.get("anomaly_features"), label="anomaly_features").get(
                "feature_store_file_sha256"
            ),
            label="anomaly feature-store SHA-256",
        ),
    }
    if dict(observed_files) != expected:
        raise D1RobustnessError("observed replay input-file identities differ from the config")
    expected_input_identity = hashlib.sha256(canonical_json_bytes(expected)).hexdigest()
    if identities.get("input_sha256") != expected_input_identity:
        raise D1RobustnessError("observed replay aggregate input identity is inconsistent")


def run_d1_robustness_diagnostics(
    project_root: Path | str,
    config_path: Path | str,
    observed_result_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Validate inputs, compute diagnostics without refitting, and write canonical JSON."""

    root = Path(project_root).resolve()
    protocol = load_d1_protocol(root)
    supplied_config = Path(config_path)
    supplied_config = (
        supplied_config if supplied_config.is_absolute() else root / supplied_config
    ).resolve()
    if supplied_config != protocol.config_path:
        raise D1RobustnessError("robustness accepts only the frozen D1 config")
    observed_path = Path(observed_result_path).resolve()
    try:
        observed = _mapping(
            json.loads(observed_path.read_text(encoding="utf-8")),
            label="observed result",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise D1RobustnessError("observed result is not readable JSON") from exc
    _validate_observed_input_binding(observed, protocol)
    zones, cell_zones, spatial_identity = _verify_spatial_inputs(protocol)
    result = build_d1_robustness_result(
        observed,
        expected_contract_sha256=protocol.config_sha256,
        expected_manifest_content_sha256=protocol.water_level_content_sha256,
        zone_ids=zones,
        zone_id_by_cell_index=cell_zones,
        spatial_strata_identity=spatial_identity,
    )

    destination = Path(output_path).resolve()
    if destination in {observed_path, protocol.config_path, protocol.water_level_path}:
        raise D1RobustnessError("robustness output may not overwrite an input")
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise D1RobustnessError("existing robustness output is unreadable") from exc
        if existing != result:
            raise D1RobustnessError("refusing to overwrite a different robustness result")
        return result
    write_json_atomic(destination, result)
    return result


__all__ = [
    "D1_ROBUSTNESS_CONTRASTS",
    "D1RobustnessError",
    "build_d1_robustness_result",
    "run_d1_robustness_diagnostics",
]
