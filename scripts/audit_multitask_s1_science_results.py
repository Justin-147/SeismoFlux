"""Post-score S1-C0 scientific diagnostics fixed before inspecting their results.

This command is deliberately diagnostic-only.  It reads the already sealed
development scores and the authoritative earthquake catalogue, but it cannot
train, select, or score a forecasting model and it never opens a holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import yaml

from seismoflux.data.common import canonical_json_bytes
from seismoflux.multitask_s0 import build_episodes

CONFIG_SHA256: Final = "2b47b432d63522edf43c4062f740ea7d4a1ec3a9990dc38be14ce96fba74cdf5"
L0: Final = "L0_UNIFORM"
L1: Final = "L1_REGIONAL_CONSTANT"
L2: Final = "L2_KDE_CAUSAL"
L3: Final = "L3_B0_R30_CAUSAL"
M0: Final = "M0_GR_GLOBAL"
M3: Final = "M3_GR_LONG_M5"
M0_SUPPORT: Final = "M>=5 unique physical events, M0 re-normalized tail"
M3_SUPPORT: Final = "M>=5 unique physical events, conditional tail"


class ScienceDiagnosticError(RuntimeError):
    """Raised when a frozen input or diagnostic invariant is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ScienceDiagnosticError(f"{label} must be a string-keyed object")
    return cast(dict[str, Any], value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScienceDiagnosticError(f"{label} must be an array")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScienceDiagnosticError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScienceDiagnosticError(f"{label} must be finite")
    return result


def _resolve_under(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ScienceDiagnosticError(f"{label} path must be a non-empty string")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ScienceDiagnosticError(f"{label} path escapes its declared root") from exc
    if not candidate.is_file():
        raise ScienceDiagnosticError(f"{label} does not exist: {candidate}")
    return candidate


def _load_frozen_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScienceDiagnosticError(f"diagnostic config does not exist: {path}")
    observed = _sha256(path)
    if observed != CONFIG_SHA256:
        raise ScienceDiagnosticError(
            f"diagnostic config SHA-256 changed: expected {CONFIG_SHA256}, observed {observed}"
        )
    try:
        config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label="config")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ScienceDiagnosticError("diagnostic config is not valid UTF-8 YAML") from exc
    if config.get("role") != "interpretation_repair_only_no_model_selection_no_retraining":
        raise ScienceDiagnosticError("diagnostic-only role changed")
    if config.get("stage") != "S1-C0-post-score-scientific-diagnostic":
        raise ScienceDiagnosticError("diagnostic stage changed")
    return config


def _payloads(frame: pd.DataFrame, family: str) -> list[dict[str, Any]]:
    required = {
        "score_family",
        "fold_id",
        "issue_time_utc",
        "horizon_days",
        "model_id",
        "status",
        "payload_json",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ScienceDiagnosticError(f"raw score table is missing columns: {missing}")
    selected = frame.loc[frame["score_family"] == family]
    output: list[dict[str, Any]] = []
    for raw in selected["payload_json"]:
        try:
            output.append(_mapping(json.loads(str(raw)), label=f"{family} payload"))
        except json.JSONDecodeError as exc:
            raise ScienceDiagnosticError(f"{family} payload is not valid JSON") from exc
    return output


def _main_location(payload: Mapping[str, Any], model_id: str) -> bool:
    return bool(
        payload.get("model_id") == model_id
        and payload.get("metric") == "strict_recall"
        and payload.get("basis") == "anchor"
        and payload.get("horizon_days") == 30
        and payload.get("magnitude_bin") == "M5_6"
        and payload.get("area_budget_km2") == 600_000.0
        and payload.get("catalog_delay_hours") == 24
        and payload.get("hit_tolerance_km") == 0.0
        and payload.get("is_main_scientific_anchor") is True
    )


def _uniform_diagnostic(
    location: Sequence[Mapping[str, Any]], *, study_area_km2: float
) -> dict[str, Any]:
    density_logs: list[float] = []
    for row in location:
        if row.get("model_id") == L0 and row.get("metric") == "spatial_log_density":
            density_logs.extend(
                _finite(value, label="L0 event log density")
                for value in _sequence(
                    row.get("event_log_densities_per_km2"), label="L0 event log densities"
                )
            )
    if not density_logs:
        raise ScienceDiagnosticError("no L0 event log densities were found")
    density_hex = {value.hex() for value in density_logs}
    if len(density_hex) != 1:
        raise ScienceDiagnosticError("L0 event log densities are not exactly identical")

    main = [row for row in location if _main_location(row, L0)]
    if not main:
        raise ScienceDiagnosticError("no L0 main-anchor rows were found")
    areas = [_finite(row.get("actual_area_km2"), label="L0 actual alarm area") for row in main]
    area_hex = {value.hex() for value in areas}
    if len(area_hex) != 1:
        raise ScienceDiagnosticError("L0 main-anchor actual alarm area is not fixed")
    actual_area = areas[0]
    if not 0.0 < actual_area <= 600_000.0:
        raise ScienceDiagnosticError("L0 actual alarm area violates the frozen budget")

    seen_events: set[str] = set()
    episode_ids: set[str] = set()
    fixed_hits = 0.0
    for row in main:
        event_ids = _sequence(row.get("event_ids"), label="L0 main event IDs")
        episodes = _sequence(row.get("episode_ids"), label="L0 main episode IDs")
        weights = _sequence(row.get("event_weights"), label="L0 main event weights")
        hits = _sequence(row.get("hit_flags"), label="L0 main hit flags")
        if not (len(event_ids) == len(episodes) == len(weights) == len(hits)):
            raise ScienceDiagnosticError("L0 main target vectors have different lengths")
        for event_id, episode_id, raw_weight, hit in zip(
            event_ids, episodes, weights, hits, strict=True
        ):
            if not isinstance(event_id, str) or not isinstance(episode_id, str):
                raise ScienceDiagnosticError("L0 main event and episode IDs must be strings")
            if event_id in seen_events:
                raise ScienceDiagnosticError("L0 main physical event is duplicated")
            seen_events.add(event_id)
            weight = _finite(raw_weight, label="L0 main event weight")
            if weight == 0.0:
                continue
            if weight != 1.0 or not isinstance(hit, bool):
                raise ScienceDiagnosticError("L0 anchor weights/hits changed")
            if episode_id in episode_ids:
                raise ScienceDiagnosticError("L0 main anchor episode is duplicated")
            episode_ids.add(episode_id)
            fixed_hits += 1.0 if hit else 0.0
    episode_count = len(episode_ids)
    if episode_count == 0:
        raise ScienceDiagnosticError("L0 main population contains no independent episode")
    expected_recall = actual_area / study_area_km2
    common_density = density_logs[0]
    return {
        "model_id": L0,
        "event_log_density_count": len(density_logs),
        "all_event_log_densities_exactly_identical": True,
        "common_event_log_density_per_km2": common_density,
        "analytic_uniform_log_density_per_km2": -math.log(study_area_km2),
        "analytic_density_absolute_difference": abs(common_density + math.log(study_area_km2)),
        "main_anchor_exposure_count": len(main),
        "actual_alarm_area_is_exactly_fixed": True,
        "actual_alarm_area_km2": actual_area,
        "study_area_km2": study_area_km2,
        "random_area_expected_recall": expected_recall,
        "pooled_episode_count": episode_count,
        "random_area_expected_hit_count": expected_recall * episode_count,
        "sealed_fixed_prefix_hit_count": fixed_hits,
        "sealed_fixed_prefix_recall": fixed_hits / episode_count,
        "sealed_fixed_prefix_role": "audit_record_only_not_uniform_random_baseline",
        "attempt2_L0_bootstrap_random_tie_inference_allowed": False,
    }


def _location_units(
    location: Sequence[Mapping[str, Any]], candidate_model: str
) -> list[dict[str, Any]]:
    def index(model_id: str) -> dict[tuple[str, str], Mapping[str, Any]]:
        result: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in location:
            if not _main_location(row, model_id):
                continue
            key = (str(row.get("fold_id")), str(row.get("issue_time_utc")))
            if key in result:
                raise ScienceDiagnosticError(f"duplicate {model_id} main exposure")
            result[key] = row
        return result

    baseline = index(L1)
    candidate = index(candidate_model)
    if not baseline or set(baseline) != set(candidate):
        raise ScienceDiagnosticError(f"{candidate_model} and L1 main exposures differ")
    units: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    seen_episodes: set[str] = set()
    for key in sorted(baseline):
        left, right = baseline[key], candidate[key]
        vector_names = ("event_ids", "episode_ids", "event_weights", "hit_flags")
        vectors = {
            f"left_{name}": _sequence(left.get(name), label=f"L1 {name}") for name in vector_names
        }
        vectors.update(
            {
                f"right_{name}": _sequence(right.get(name), label=f"{candidate_model} {name}")
                for name in vector_names
            }
        )
        if (
            vectors["left_event_ids"] != vectors["right_event_ids"]
            or vectors["left_episode_ids"] != vectors["right_episode_ids"]
            or vectors["left_event_weights"] != vectors["right_event_weights"]
        ):
            raise ScienceDiagnosticError(f"{candidate_model} and L1 target payloads differ")
        lengths = {len(value) for value in vectors.values()}
        if len(lengths) != 1:
            raise ScienceDiagnosticError("paired location vectors have different lengths")
        for event_id, episode_id, weight, l1_hit, candidate_hit in zip(
            vectors["left_event_ids"],
            vectors["left_episode_ids"],
            vectors["left_event_weights"],
            vectors["left_hit_flags"],
            vectors["right_hit_flags"],
            strict=True,
        ):
            if not isinstance(event_id, str) or not isinstance(episode_id, str):
                raise ScienceDiagnosticError("paired location IDs must be strings")
            if event_id in seen_events:
                raise ScienceDiagnosticError("main-anchor physical event is duplicated")
            seen_events.add(event_id)
            parsed_weight = _finite(weight, label="paired anchor weight")
            if parsed_weight == 0.0:
                continue
            if (
                parsed_weight != 1.0
                or not isinstance(l1_hit, bool)
                or not isinstance(candidate_hit, bool)
            ):
                raise ScienceDiagnosticError("paired main-anchor unit changed")
            if episode_id in seen_episodes:
                raise ScienceDiagnosticError("paired main-anchor episode is duplicated")
            seen_episodes.add(episode_id)
            units.append(
                {
                    "fold_id": key[0],
                    "episode_id": episode_id,
                    "l1_hit": l1_hit,
                    "candidate_hit": candidate_hit,
                }
            )
    return units


def _point_difference(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not units:
        raise ScienceDiagnosticError("location comparison contains no episodes")
    count = len(units)
    baseline_hits = sum(bool(unit["l1_hit"]) for unit in units)
    candidate_hits = sum(bool(unit["candidate_hit"]) for unit in units)
    return {
        "independent_episode_count": count,
        "L1_hit_count": baseline_hits,
        "candidate_hit_count": candidate_hits,
        "L1_recall": baseline_hits / count,
        "candidate_recall": candidate_hits / count,
        "point_difference_candidate_minus_L1": (candidate_hits - baseline_hits) / count,
    }


def _location_point_comparison(
    location: Sequence[Mapping[str, Any]], candidate_model: str
) -> dict[str, Any]:
    units = _location_units(location, candidate_model)
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_fold[str(unit["fold_id"])].append(unit)
    return {
        "candidate_model_id": candidate_model,
        "baseline_model_id": L1,
        "inference": "posthoc_point_difference_only_no_pairwise_CI",
        "pooled": _point_difference(units),
        "by_fold": [
            {"fold_id": fold_id, **_point_difference(by_fold[fold_id])}
            for fold_id in sorted(by_fold)
        ],
    }


def _magnitude_effects(magnitude: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def rows(model_id: str, support: str) -> dict[tuple[str, str], Mapping[str, Any]]:
        result: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in magnitude:
            if row.get("model_id") != model_id or row.get("conditional_support") != support:
                continue
            key = (str(row.get("fold_id")), str(row.get("forecast_issue_time_utc")))
            if key in result:
                raise ScienceDiagnosticError(f"duplicate {model_id} M5-tail issue")
            result[key] = row
        return result

    m0 = rows(M0, M0_SUPPORT)
    m3 = rows(M3, M3_SUPPORT)
    if not m0 or set(m0) != set(m3):
        raise ScienceDiagnosticError("M0-conditioned and M3 issue populations differ")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in sorted(m0):
        m0_ids = _sequence(m0[key].get("event_ids"), label="M0-conditioned event IDs")
        m3_ids = _sequence(m3[key].get("event_ids"), label="M3 event IDs")
        m0_logs = _sequence(
            m0[key].get("event_log_probabilities"), label="M0-conditioned event scores"
        )
        m3_logs = _sequence(m3[key].get("event_log_probabilities"), label="M3 event scores")
        if m0_ids != m3_ids or not (len(m0_ids) == len(m0_logs) == len(m3_logs)):
            raise ScienceDiagnosticError("paired magnitude event vectors differ")
        for event_id, m0_log, m3_log in zip(m0_ids, m0_logs, m3_logs, strict=True):
            if not isinstance(event_id, str) or not event_id or event_id in seen:
                raise ScienceDiagnosticError("M5-tail event IDs are invalid or duplicated")
            seen.add(event_id)
            output.append(
                {
                    "event_id": event_id,
                    "fold_id": key[0],
                    "effect": _finite(m3_log, label="M3 event log probability")
                    - _finite(m0_log, label="M0 event log probability"),
                }
            )
    return output


def _cluster_magnitude_effects(
    effects: Sequence[Mapping[str, Any]],
    catalog: pd.DataFrame,
    *,
    max_time_days: int,
    max_distance_km: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    required = {
        "event_id",
        "origin_time_utc",
        "available_at",
        "longitude",
        "latitude",
        "magnitude",
        "inside_study_area",
    }
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise ScienceDiagnosticError(f"authoritative catalog is missing columns: {missing}")
    selected = catalog.loc[
        catalog["inside_study_area"].astype(bool) & (catalog["magnitude"].astype(float) >= 5.0)
    ].copy()
    episodes = build_episodes(
        selected.reset_index(drop=True),
        max_time_days=max_time_days,
        max_distance_km=max_distance_km,
    )
    event_to_episode: dict[str, tuple[str, int]] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        raw_member_count = episode["member_count"]
        if isinstance(raw_member_count, bool) or not isinstance(raw_member_count, int):
            raise ScienceDiagnosticError("catalog episode member count is not an integer")
        member_count = raw_member_count
        for event_id in cast(list[str], episode["member_event_ids"]):
            if event_id in event_to_episode:
                raise ScienceDiagnosticError("catalog episode assignment duplicated an event")
            event_to_episode[event_id] = (episode_id, member_count)

    cluster_effects: dict[str, list[float]] = defaultdict(list)
    cluster_full_counts: dict[str, int] = {}
    fold_effects: dict[str, list[float]] = defaultdict(list)
    for row in effects:
        event_id = str(row["event_id"])
        if event_id not in event_to_episode:
            raise ScienceDiagnosticError(
                f"scored M5-tail event is absent from M>=5 episodes: {event_id}"
            )
        episode_id, full_count = event_to_episode[event_id]
        effect = _finite(row["effect"], label="magnitude event effect")
        cluster_effects[episode_id].append(effect)
        cluster_full_counts[episode_id] = full_count
        fold_effects[str(row["fold_id"])].append(effect)
    if len(effects) == 0 or len(fold_effects) != 4:
        raise ScienceDiagnosticError("magnitude diagnostic requires events in exactly four folds")
    if replicates <= 0:
        raise ScienceDiagnosticError("cluster bootstrap replicates must be positive")

    cluster_ids = sorted(cluster_effects)
    sums = np.asarray([math.fsum(cluster_effects[key]) for key in cluster_ids], dtype=np.float64)
    counts = np.asarray([len(cluster_effects[key]) for key in cluster_ids], dtype=np.int64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    batch_size = 1024
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, len(cluster_ids), size=(stop - start, len(cluster_ids)))
        estimates[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    lower, upper = np.percentile(estimates, [2.5, 97.5], method="linear")

    largest_id = min(
        cluster_ids,
        key=lambda key: (-cluster_full_counts[key], -len(cluster_effects[key]), key),
    )
    retained = [
        value
        for cluster_id in cluster_ids
        if cluster_id != largest_id
        for value in cluster_effects[cluster_id]
    ]
    if not retained:
        raise ScienceDiagnosticError("largest-cluster sensitivity would remove every event")
    point = math.fsum(float(row["effect"]) for row in effects) / len(effects)
    interval_positive = float(lower) > 0.0
    return {
        "event_count": len(effects),
        "combined_M5_plus_episode_count": len(cluster_ids),
        "full_catalog_M5_plus_event_count": len(selected),
        "full_catalog_combined_episode_count": len(episodes),
        "effect": "per_event_log_probability_M3_minus_M0_conditioned_M5_plus",
        "point_mean_effect_nats_per_event": point,
        "cluster_bootstrap": {
            "unit": "combined_M5_plus_fixed_anchor_episode",
            "replicates": replicates,
            "seed": seed,
            "interval": "percentile_2.5_97.5",
            "lower": float(lower),
            "upper": float(upper),
            "strictly_positive": interval_positive,
        },
        "four_outer_fold_mean_effects": [
            {
                "fold_id": fold_id,
                "event_count": len(fold_effects[fold_id]),
                "mean_effect_nats_per_event": math.fsum(fold_effects[fold_id])
                / len(fold_effects[fold_id]),
            }
            for fold_id in sorted(fold_effects)
        ],
        "remove_largest_combined_episode": {
            "selection_rule": "max_full_catalog_member_count_then_scored_count_then_episode_id",
            "episode_id": largest_id,
            "full_catalog_member_count": cluster_full_counts[largest_id],
            "scored_event_count": len(cluster_effects[largest_id]),
            "remaining_event_count": len(retained),
            "remaining_mean_effect_nats_per_event": math.fsum(retained) / len(retained),
        },
        "wording_gate": (
            "cluster_robust_small_development_signal"
            if interval_positive
            else "four_fold_same_direction_small_development_signal_not_cluster_confirmed"
        ),
    }


def compute_science_diagnostic(
    raw_scores: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    study_area_km2: float,
    max_time_days: int,
    max_distance_km: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compute the frozen diagnostics from in-memory inputs (used by synthetic tests)."""

    area = _finite(study_area_km2, label="study area")
    if area <= 0.0:
        raise ScienceDiagnosticError("study area must be positive")
    location = _payloads(raw_scores, "location")
    magnitude = _payloads(raw_scores, "magnitude")
    effects = _magnitude_effects(magnitude)
    return {
        "schema_version": 1,
        "record_type": "s1_c0_post_score_science_diagnostic",
        "scientific_role": "interpretation_repair_only_no_model_selection_no_retraining",
        "holdout_opened": False,
        "audit_opened": False,
        "locked_test_run": False,
        "uniform_tie_diagnostic": _uniform_diagnostic(location, study_area_km2=area),
        "location_point_differences_vs_interpretable_L1": [
            _location_point_comparison(location, L2),
            _location_point_comparison(location, L3),
        ],
        "magnitude_cluster_dependence_diagnostic": _cluster_magnitude_effects(
            effects,
            catalog,
            max_time_days=max_time_days,
            max_distance_km=max_distance_km,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "decision_restrictions": {
            "candidate_selection_allowed": False,
            "retraining_allowed": False,
            "posthoc_location_pairwise_CI_allowed": False,
            "holdout_opening_allowed": False,
        },
    }


def _identity(path: Path, *, root: Path, row_count: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if row_count is not None:
        value["row_count"] = row_count
    return value


def run_diagnostic(
    *, config_path: Path, project_root: Path, data_root: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Validate frozen identities, compute diagnostics, and create two new files."""

    config_path = config_path.resolve()
    project_root = project_root.resolve()
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(project_root)
    except ValueError as exc:
        raise ScienceDiagnosticError("output directory must stay under project root") from exc
    config = _load_frozen_config(config_path)
    frozen = _mapping(config.get("frozen_inputs"), label="frozen_inputs")
    raw_spec = _mapping(frozen.get("raw_scores"), label="frozen raw_scores")
    summary_spec = _mapping(frozen.get("development_summary"), label="frozen summary")
    catalog_spec = _mapping(frozen.get("authoritative_catalog"), label="frozen catalog")
    raw_path = _resolve_under(project_root, raw_spec.get("path"), label="raw scores")
    summary_path = _resolve_under(project_root, summary_spec.get("path"), label="summary")
    catalog_path = _resolve_under(data_root, catalog_spec.get("data_root_path"), label="catalog")
    for path, spec, label in (
        (raw_path, raw_spec, "raw scores"),
        (summary_path, summary_spec, "development summary"),
        (catalog_path, catalog_spec, "authoritative catalog"),
    ):
        expected = spec.get("sha256")
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ScienceDiagnosticError(f"{label} SHA-256 differs from the frozen config")

    raw_scores = pd.read_parquet(raw_path)
    expected_rows = raw_spec.get("row_count")
    if not isinstance(expected_rows, int) or len(raw_scores) != expected_rows:
        raise ScienceDiagnosticError("raw score row count differs from the frozen config")
    catalog = pd.read_parquet(catalog_path)
    try:
        summary = _mapping(json.loads(summary_path.read_text(encoding="utf-8")), label="summary")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScienceDiagnosticError("development summary is not valid UTF-8 JSON") from exc
    if (
        summary.get("champion_selection_allowed") is not False
        or summary.get("holdout_opening_allowed") is not False
    ):
        raise ScienceDiagnosticError("development summary violates the diagnostic-only gate")

    uniform = _mapping(config.get("uniform_tie_diagnostic"), label="uniform diagnostic")
    magnitude_config = _mapping(
        config.get("magnitude_dependence_diagnostic"), label="magnitude diagnostic"
    )
    cluster = _mapping(magnitude_config.get("cluster_definition"), label="cluster definition")
    bootstrap = _mapping(magnitude_config.get("bootstrap"), label="bootstrap")
    if (
        uniform.get("required_model") != L0
        or cluster.get("method") != "fixed_first_event_anchor_non_transitive"
        or cluster.get("combine_M5_6_and_M6_plus") is not True
        or bootstrap.get("unit") != "combined_M5_plus_fixed_anchor_episode"
    ):
        raise ScienceDiagnosticError("frozen scientific diagnostic definition changed")
    max_time_days = cluster.get("max_time_days")
    replicates = bootstrap.get("replicates")
    seed = bootstrap.get("seed")
    if (
        isinstance(max_time_days, bool)
        or not isinstance(max_time_days, int)
        or isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ScienceDiagnosticError("frozen cluster/bootstrap integer parameters changed")
    result = compute_science_diagnostic(
        raw_scores,
        catalog,
        study_area_km2=_finite(frozen.get("study_area_km2"), label="frozen study area"),
        max_time_days=max_time_days,
        max_distance_km=_finite(cluster.get("max_distance_km"), label="cluster distance"),
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )
    input_identities = {
        "config": _identity(config_path, root=project_root),
        "raw_scores": _identity(raw_path, root=project_root, row_count=len(raw_scores)),
        "development_summary": _identity(summary_path, root=project_root),
        "authoritative_catalog": _identity(catalog_path, root=data_root, row_count=len(catalog)),
    }
    result["input_identities"] = input_identities
    result["config_sha256"] = CONFIG_SHA256

    result_path = output_dir / str(_mapping(config.get("outputs"), label="outputs")["result"])
    manifest_path = output_dir / str(_mapping(config.get("outputs"), label="outputs")["manifest"])
    if result_path.exists() or manifest_path.exists():
        raise ScienceDiagnosticError("diagnostic outputs already exist and cannot be overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_bytes = canonical_json_bytes(result) + b"\n"
    with result_path.open("xb") as handle:
        handle.write(result_bytes)
        handle.flush()
    result_sha = _sha256(result_path)
    manifest_body: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "s1_c0_science_diagnostic_manifest",
        "config_sha256": CONFIG_SHA256,
        "inputs": input_identities,
        "outputs": {
            "science_diagnostic": {
                "path": result_path.relative_to(project_root).as_posix(),
                "sha256": result_sha,
                "size_bytes": result_path.stat().st_size,
            }
        },
        "self_hash_scheme": "sha256_of_canonical_manifest_without_manifest_content_sha256",
    }
    manifest_body["manifest_content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest_body)
    ).hexdigest()
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest_body) + b"\n")
        handle.flush()
    return result_path, manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_path, manifest_path = run_diagnostic(
        config_path=args.config,
        project_root=args.project_root,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "science_diagnostic": result_path.as_posix(),
                "manifest": manifest_path.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
