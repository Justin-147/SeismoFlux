"""Score-blind S1 development calendar and causal-boundary validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import yaml

HORIZONS_DAYS = (7, 30, 90, 180, 365)
DEVELOPMENT_FOLD_IDS = (
    "C_DEV_2000_2004",
    "C_DEV_2005_2009",
    "C_DEV_2010_2014",
    "C_DEV_2015_2019",
)
SHANGHAI_OFFSET = timezone(timedelta(hours=8))
EXPECTED_SOURCE_IDENTITIES = {
    "multitask_s0_config": {
        "path": "configs/multitask_s0.yaml",
        "sha256": "fac63c49e0c6f05beada3b268ce255483bbfc3ced37562b6236c0bd6ec71dd02",
    },
    "issue_maturity_ledger": {
        "path": "outputs/multitask_s0/s0_score_blind_20260901/issue_maturity_ledger.csv",
        "sha256": "23f581783e64ebf29b508fc95e96dd3805d5de952de6093af5f86860247096b0",
    },
    "catalog_sample_ledger": {
        "path": "outputs/multitask_s0/s0_score_blind_20260901/catalog_sample_ledger.json",
        "sha256": "bef4a85379ca2c9aaf9917f501e8a84fe59311f9ddd029fac85bde95847c223e",
    },
}
EXPECTED_SELECTION_FALLBACKS = {
    "minimum_mature_inner_blocks": 2,
    "minimum_positive_anchor_blocks_for_event_metric": 2,
    "insufficient_mature_blocks": (
        "independent_selection_unavailable_use_frozen_no_tuning_or_shared_model"
    ),
    "insufficient_positive_anchor_blocks": (
        "do_not_merge_or_move_blocks_use_preregistered_proper_score_or_shared_parameter"
    ),
    "zero_anchor_block": "retain_as_zero_for_proper_scores_event_dependent_metric_is_na",
    "na_is_zero": False,
    "posthoc_block_merge": False,
    "m6_plus_7d_independent_hyperparameter_selection": False,
}
EXPECTED_FOLD_CALENDAR = {
    "C_DEV_2000_2004": (
        "2000-01-01T00:00:00+08:00",
        "2005-01-01T00:00:00+08:00",
        (
            ("I1", "1985-01-01T00:00:00+08:00", "1990-01-01T00:00:00+08:00"),
            ("I2", "1990-01-01T00:00:00+08:00", "1995-01-01T00:00:00+08:00"),
            ("I3", "1995-01-01T00:00:00+08:00", "1999-12-02T00:00:00+08:00"),
        ),
    ),
    "C_DEV_2005_2009": (
        "2005-01-01T00:00:00+08:00",
        "2010-01-01T00:00:00+08:00",
        (
            ("I1", "1990-01-01T00:00:00+08:00", "1995-01-01T00:00:00+08:00"),
            ("I2", "1995-01-01T00:00:00+08:00", "2000-01-01T00:00:00+08:00"),
            ("I3", "2000-01-01T00:00:00+08:00", "2004-12-02T00:00:00+08:00"),
        ),
    ),
    "C_DEV_2010_2014": (
        "2010-01-01T00:00:00+08:00",
        "2015-01-01T00:00:00+08:00",
        (
            ("I1", "1995-01-01T00:00:00+08:00", "2000-01-01T00:00:00+08:00"),
            ("I2", "2000-01-01T00:00:00+08:00", "2005-01-01T00:00:00+08:00"),
            ("I3", "2005-01-01T00:00:00+08:00", "2009-12-02T00:00:00+08:00"),
        ),
    ),
    "C_DEV_2015_2019": (
        "2015-01-01T00:00:00+08:00",
        "2020-01-01T00:00:00+08:00",
        (
            ("I1", "2000-01-01T00:00:00+08:00", "2005-01-01T00:00:00+08:00"),
            ("I2", "2005-01-01T00:00:00+08:00", "2010-01-01T00:00:00+08:00"),
            ("I3", "2010-01-01T00:00:00+08:00", "2014-12-02T00:00:00+08:00"),
        ),
    ),
}
EXPECTED_WATERLEVEL_SHA256 = "4daa5ce2fb8a0c534552774a2c6576af161843e45f1eb3833eda9036b605ef7d"


class DevelopmentContractError(ValueError):
    """Raised when the frozen score-blind execution contract is inconsistent."""


@dataclass(frozen=True)
class DevelopmentContractSummary:
    fold_count: int
    inner_block_count: int
    waterlevel_row_count: int
    earliest_inner_start: datetime
    latest_inner_end: datetime
    m6_plus_7d_zero_anchor_block_count: int


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentContractError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise DevelopmentContractError(f"{label} must be a sequence")
    return value


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise DevelopmentContractError(f"{label} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(hours=8):
        raise DevelopmentContractError(f"{label} must use the frozen +08:00 offset")
    if parsed.time().replace(tzinfo=None) != datetime.min.time():
        raise DevelopmentContractError(f"{label} must be local midnight")
    return parsed


def _integer_vector(value: object, label: str) -> tuple[int, ...]:
    items = _sequence(value, label)
    if len(items) != len(HORIZONS_DAYS):
        raise DevelopmentContractError(f"{label} must align with all frozen horizons")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in items):
        raise DevelopmentContractError(f"{label} must contain non-negative integers")
    return tuple(cast(int, item) for item in items)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_identities(
    contract: Mapping[str, Any], project_root: str | Path | None = None
) -> None:
    sources = _mapping(contract.get("source_identities"), "source_identities")
    if set(sources) != set(EXPECTED_SOURCE_IDENTITIES):
        raise DevelopmentContractError("source identity set changed")
    root = Path(project_root).resolve() if project_root is not None else None
    for source_id, expected_record in EXPECTED_SOURCE_IDENTITIES.items():
        raw = sources.get(source_id)
        record = _mapping(raw, f"source_identities.{source_id}")
        if dict(record) != expected_record:
            raise DevelopmentContractError(f"source identity changed for {source_id}")
        if root is None:
            continue
        source_path = (root / expected_record["path"]).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise DevelopmentContractError(
                f"source path escaped project root for {source_id}"
            ) from exc
        actual = sha256_file(source_path)
        if actual != expected_record["sha256"]:
            raise DevelopmentContractError(f"source hash mismatch for {source_id}")


def validate_development_contract(
    contract: Mapping[str, Any], *, project_root: str | Path | None = None
) -> DevelopmentContractSummary:
    if contract.get("status") != "score_blind_development_execution_contract_frozen":
        raise DevelopmentContractError("contract status is not frozen")
    if contract.get("score_blind") is not True or contract.get("model_scores_read") is not False:
        raise DevelopmentContractError("contract must remain score-blind")
    if contract.get("locked_test_run") is not False:
        raise DevelopmentContractError("locked test must remain unopened")

    scope = _mapping(contract.get("execution_scope"), "execution_scope")
    enabled = tuple(_sequence(scope.get("enabled_outer_folds"), "enabled_outer_folds"))
    if enabled != DEVELOPMENT_FOLD_IDS or scope.get("enabled_roles") != ["development"]:
        raise DevelopmentContractError("only the four frozen development folds may run")
    for key in ("holdout_enabled", "audit_enabled", "locked_test_enabled"):
        if scope.get(key) is not False:
            raise DevelopmentContractError(f"{key} must remain false")
    if scope.get("construction_blocks_used_for_tuning") is not False:
        raise DevelopmentContractError("construction blocks may not tune S1 models")
    if scope.get("prediction_must_be_sealed_before_target_scoring") is not True:
        raise DevelopmentContractError("predictions must be sealed before target scoring")

    causal = _mapping(contract.get("causal_boundaries"), "causal_boundaries")
    required_causal = {
        "timezone": "Asia/Shanghai",
        "issue_weekday": "Thursday",
        "issue_local_time": "00:00:00",
        "issue_frequency_days": 7,
        "horizons_days": list(HORIZONS_DAYS),
        "target_interval": "(issue,issue+horizon]",
        "main_catalog_delay_hours": 24,
        "feature_visibility_rule": "available_at<=issue-24h",
        "recent_30d_window": "(issue-30d,issue-24h]",
        "parameter_label_embargo_days": 30,
        "fit_label_rule": "fit_issue+horizon<=inner_start-30d",
        "inner_target_rule": "issue+horizon<=inner_end",
        "primary_exposure_rule": "greedy_issue_ascending_next>=previous+horizon+30d",
    }
    for key, expected in required_causal.items():
        if causal.get(key) != expected:
            raise DevelopmentContractError(f"causal boundary changed: {key}")

    fallbacks = _mapping(contract.get("selection_fallbacks"), "selection_fallbacks")
    if dict(fallbacks) != EXPECTED_SELECTION_FALLBACKS:
        raise DevelopmentContractError("selection fallback semantics changed")

    folds = _sequence(contract.get("outer_folds"), "outer_folds")
    if len(folds) != len(DEVELOPMENT_FOLD_IDS):
        raise DevelopmentContractError("outer fold count changed")
    expected_pairs: set[tuple[str, str]] = set()
    starts: list[datetime] = []
    ends: list[datetime] = []
    for raw_fold, expected_id in zip(folds, DEVELOPMENT_FOLD_IDS, strict=True):
        fold = _mapping(raw_fold, f"outer_folds.{expected_id}")
        if fold.get("id") != expected_id:
            raise DevelopmentContractError("outer fold order or identity changed")
        outer_start = _timestamp(fold.get("outer_start"), f"{expected_id}.outer_start")
        outer_end = _timestamp(fold.get("outer_end"), f"{expected_id}.outer_end")
        expected_start_raw, expected_end_raw, expected_blocks = EXPECTED_FOLD_CALENDAR[expected_id]
        expected_start = _timestamp(expected_start_raw, f"expected.{expected_id}.outer_start")
        expected_end = _timestamp(expected_end_raw, f"expected.{expected_id}.outer_end")
        if (outer_start, outer_end) != (expected_start, expected_end):
            raise DevelopmentContractError(f"outer calendar changed for {expected_id}")
        blocks = _sequence(fold.get("inner_blocks"), f"{expected_id}.inner_blocks")
        if len(blocks) != 3:
            raise DevelopmentContractError(f"{expected_id} must have three inner blocks")
        for raw_block, expected_block in zip(blocks, expected_blocks, strict=True):
            block_id, expected_block_start_raw, expected_block_end_raw = expected_block
            block = _mapping(raw_block, f"{expected_id}.{block_id}")
            start = _timestamp(block.get("start"), f"{expected_id}.{block_id}.start")
            end = _timestamp(block.get("end"), f"{expected_id}.{block_id}.end")
            expected_block_start = _timestamp(
                expected_block_start_raw, f"expected.{expected_id}.{block_id}.start"
            )
            expected_block_end = _timestamp(
                expected_block_end_raw, f"expected.{expected_id}.{block_id}.end"
            )
            if block.get("id") != block_id or (start, end) != (
                expected_block_start,
                expected_block_end,
            ):
                raise DevelopmentContractError(
                    f"inner calendar changed for {expected_id}.{block_id}"
                )
            expected_pairs.add((expected_id, block_id))
            starts.append(start)
            ends.append(end)
    if min(starts) < datetime(1970, 1, 1, tzinfo=SHANGHAI_OFFSET):
        raise DevelopmentContractError("inner evaluation escaped the 1970+ supported catalog")

    if tuple(_sequence(contract.get("waterlevel_columns"), "waterlevel_columns")) != HORIZONS_DAYS:
        raise DevelopmentContractError("water-level horizon order changed")
    water_rows = _sequence(contract.get("inner_block_waterlevels"), "inner_block_waterlevels")
    if len(water_rows) != len(expected_pairs):
        raise DevelopmentContractError("water-level row count changed")
    canonical_waterlevels = json.dumps(
        water_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if hashlib.sha256(canonical_waterlevels).hexdigest() != EXPECTED_WATERLEVEL_SHA256:
        raise DevelopmentContractError("frozen water-level values changed")
    seen: set[tuple[str, str]] = set()
    zero_m6_7d = 0
    mature_by_fold_horizon: dict[tuple[str, int], int] = {}
    for index, raw_row in enumerate(water_rows):
        row = _mapping(raw_row, f"inner_block_waterlevels[{index}]")
        pair = (str(row.get("fold")), str(row.get("block")))
        if pair not in expected_pairs or pair in seen:
            raise DevelopmentContractError(
                "water-level fold/block identity is incomplete or duplicated"
            )
        seen.add(pair)
        exposures = _integer_vector(row.get("exposures"), f"{pair}.exposures")
        m56_events = _integer_vector(row.get("m5_6_events"), f"{pair}.m5_6_events")
        m56_anchors = _integer_vector(row.get("m5_6_anchors"), f"{pair}.m5_6_anchors")
        m6_events = _integer_vector(row.get("m6_plus_events"), f"{pair}.m6_plus_events")
        m6_anchors = _integer_vector(row.get("m6_plus_anchors"), f"{pair}.m6_plus_anchors")
        for position, horizon in enumerate(HORIZONS_DAYS):
            if exposures[position] <= 0:
                raise DevelopmentContractError(
                    "every frozen inner block must retain mature exposures"
                )
            mature_by_fold_horizon[(pair[0], horizon)] = (
                mature_by_fold_horizon.get((pair[0], horizon), 0) + 1
            )
            if (
                m56_anchors[position] > m56_events[position]
                or m6_anchors[position] > m6_events[position]
            ):
                raise DevelopmentContractError("episode anchors cannot exceed unique events")
        if m6_anchors[0] == 0:
            if pair != ("C_DEV_2015_2019", "I3") or m6_events[0] != 0:
                raise DevelopmentContractError("frozen M6+ 7d zero block identity changed")
            zero_m6_7d += 1
    if seen != expected_pairs:
        raise DevelopmentContractError("water-level coverage is incomplete")
    if min(mature_by_fold_horizon.values()) < 2:
        raise DevelopmentContractError(
            "an outer task/horizon has fewer than two mature inner blocks"
        )
    if zero_m6_7d != 1:
        raise DevelopmentContractError("the frozen M6+ 7d zero-anchor disclosure changed")

    validate_source_identities(contract, project_root)
    return DevelopmentContractSummary(
        fold_count=len(folds),
        inner_block_count=len(expected_pairs),
        waterlevel_row_count=len(water_rows),
        earliest_inner_start=min(starts),
        latest_inner_end=max(ends),
        m6_plus_7d_zero_anchor_block_count=zero_m6_7d,
    )


def load_development_contract(
    path: str | Path, *, project_root: str | Path | None = None
) -> tuple[Mapping[str, Any], DevelopmentContractSummary]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    contract = _mapping(raw, "development contract")
    return contract, validate_development_contract(contract, project_root=project_root)


__all__ = [
    "DEVELOPMENT_FOLD_IDS",
    "HORIZONS_DAYS",
    "DevelopmentContractError",
    "DevelopmentContractSummary",
    "load_development_contract",
    "sha256_file",
    "validate_development_contract",
    "validate_source_identities",
]
