"""Existing paired bootstrap components for fixed primary national summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from seismoflux.multitask_s1.c2b_score import exposure_bootstrap
from seismoflux.multitask_s1.development_summary import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_ROOT_SEED,
    _bootstrap_episode_ratio,
    _EpisodeUnit,
)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite real number")
    return number


def _cluster_values(cluster: Mapping[str, Any]) -> tuple[float, float, float, float]:
    total = _finite_number(cluster["total_weight"], "total_weight")
    candidate = _finite_number(cluster["candidate_weighted_hits"], "candidate_weighted_hits")
    reference = _finite_number(cluster["reference_weighted_hits"], "reference_weighted_hits")
    delta = _finite_number(
        cluster["candidate_minus_reference_weighted_hit_sum"], "paired hit difference"
    )
    if total < 0 or not 0 <= candidate <= total or not 0 <= reference <= total:
        raise ValueError("cluster hit weights must be nonnegative and at most total_weight")
    if not math.isclose(delta, candidate - reference, rel_tol=1e-12, abs_tol=1e-12 * total):
        raise ValueError("cluster paired difference disagrees with candidate/reference hits")
    return total, candidate, reference, delta


def paired_uncertainty(
    summary: Mapping[str, Any],
    *,
    issue_keys: Sequence[str],
    label: str,
    global_member_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Describe uncertainty only for a fixed primary-nonoverlap national group.

    Caller must supply the complete frozen fold|issue calendar, including every
    no-event window. The event-based summary alone cannot reveal an omitted
    empty window. Duplicate calendar entries or event-bearing clusters absent
    from the supplied calendar are errors. No horizon, magnitude band, area,
    comparison or event view is selected or pooled here.

    Episode clusters are full-history episode identities, already pooled across
    folds by ``summarize_paired_group``. Frozen global member counts are metadata;
    the supplied weighted hits and denominator are never recalculated. Both
    bootstrap procedures are reused unchanged. Their different percentile
    conventions are preserved, and their intervals describe this development
    comparison rather than an adoption or statistical-significance threshold.
    """
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a nonempty fixed comparison identity")
    if any(
        not isinstance(key, str)
        or key.count("|") != 1
        or not all(key.split("|"))
        for key in issue_keys
    ):
        raise ValueError("issue_keys must contain nonempty fold_id|issue_time_utc identities")
    if len(set(issue_keys)) != len(issue_keys):
        raise ValueError("duplicate issue in the frozen primary issue_keys")
    episodes = summary["_local"]["episode_clusters"]
    issues = summary["_local"]["issue_clusters"]
    if set(issues) - set(issue_keys):
        raise ValueError("issue_keys is missing an event-bearing issue cluster")
    total_weight = _finite_number(summary["total_weight"], "summary total_weight")
    if total_weight < 0:
        raise ValueError("summary total_weight cannot be negative")

    units: list[_EpisodeUnit] = []
    for episode_id in sorted(episodes):
        if episode_id not in global_member_counts:
            raise ValueError(f"missing frozen global member count for {episode_id}")
        count = global_member_counts[episode_id]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("frozen global member counts must be positive integers")
        total, candidate, reference, _ = _cluster_values(episodes[episode_id])
        units.append(
            _EpisodeUnit(
                unit_id=episode_id,
                fold_id="pooled_authorized_A_development",
                episode_id=episode_id,
                global_member_count=count,
                candidate_hits=candidate,
                baseline_hits=reference,
                total_weight=total,
            )
        )
    issue_totals: list[float] = []
    issue_deltas: list[float] = []
    for key in issue_keys:
        if key in issues:
            total, _, _, delta = _cluster_values(issues[key])
        else:
            total, delta = 0.0, 0.0
        issue_totals.append(total)
        issue_deltas.append(delta)
    for value in (
        math.fsum(unit.total_weight for unit in units),
        math.fsum(issue_totals),
    ):
        if not math.isclose(value, total_weight, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("cluster denominators disagree with the paired summary")

    episode_result = dict(_bootstrap_episode_ratio(units, label=label))
    episode_interval = episode_result.pop("confidence_interval_95")
    episode_result.update(
        {
            "confidence_interval_95_pp": None
            if episode_interval is None
            else [100.0 * float(value) for value in episode_interval],
            "effect_unit": "percentage_points",
            "episode_count": len(units),
            "positive_weight_episode_count": sum(unit.total_weight > 0 for unit in units),
            "cluster_scope": "full_history_episode_id_not_split_by_fold",
            "component": "multitask_s1.development_summary._bootstrap_episode_ratio",
        }
    )
    reason = (
        "zero_total_weight"
        if total_weight == 0
        else "fewer_than_two_primary_issues"
        if len(issue_keys) < 2
        else None
    )
    time_interval = None
    if reason is None:
        time_interval = exposure_bootstrap(
            np.asarray(issue_deltas, dtype=np.float64),
            np.asarray(issue_totals, dtype=np.float64),
        )
        if time_interval is None:
            reason = "no_positive_denominator_resamples"
    return {
        "scope": "primary_nonoverlapping_national_fixed_group_only",
        "point_estimate_delta_recall_pp": summary["delta_recall_pp"],
        "episode": episode_result,
        "time_block": {
            "status": "evaluable" if time_interval is not None else "not_evaluable",
            "reason": reason,
            "confidence_interval_95_pp": time_interval,
            "effect_unit": "percentage_points",
            "resampling_unit": "paired_primary_nonoverlapping_fold_issue",
            "issue_count": len(issue_keys),
            "empty_issue_count": sum(total == 0 for total in issue_totals),
            "root_seed": BOOTSTRAP_ROOT_SEED,
            "replicates_requested": BOOTSTRAP_REPLICATES,
            "replicates_evaluable": None if time_interval is not None else 0,
            "replicate_accounting": "inherited_component_excludes_zero_denominator_draws_"
            "and_does_not_return_valid_draw_count",
            "component": "multitask_s1.c2b_score.exposure_bootstrap",
        },
        "interpretation": "descriptive_development_uncertainty_not_independent_confirmation_"
        "or_adoption_gate; NA_interval_does_not_negate_point_estimate",
        "adoption_threshold": None,
    }
