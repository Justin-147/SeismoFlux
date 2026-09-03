"""Paired descriptive accounting within one caller-defined scientific group."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _cluster_summaries(
    weights: Mapping[str, Sequence[float]],
    differences: Mapping[str, Sequence[float]],
    candidate_weights: Mapping[str, Sequence[float]],
    reference_weights: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float]]:
    return {
        key: {
            "candidate_minus_reference_weighted_hit_sum": math.fsum(differences[key]),
            "total_weight": math.fsum(weights[key]),
            "candidate_weighted_hits": math.fsum(candidate_weights[key]),
            "reference_weighted_hits": math.fsum(reference_weights[key]),
        }
        for key in sorted(weights)
    }


def summarize_paired_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize an already-fixed fold/task/area/view/region paired comparison.

    Every row is an event exposure in one forecast window, not necessarily a
    different earthquake. The caller selects the group, supplies canonical
    string identities, and supplies positive weights. In an episode-balanced
    view these are 1 / the frozen full-history episode member count; they are
    never recomputed or renormalized within this group. Other event views may
    supply unit weights. Weighted hits and denominator are not independent
    sample counts. Recall fields are fractions; ``delta_recall_pp`` is in
    percentage points. Empty groups have NA recalls rather than zero recall.

    Local cluster sums support the existing paired episode/time-block bootstrap
    in the caller; this function implements no resampling, selection or gate.
    """
    weights: list[float] = []
    candidate_weights: list[float] = []
    reference_weights: list[float] = []
    gained: list[float] = []
    lost: list[float] = []
    shared_hit: list[float] = []
    shared_miss: list[float] = []
    differences: list[float] = []
    events: set[str] = set()
    episodes: set[str] = set()
    episode_weights: dict[str, list[float]] = defaultdict(list)
    episode_differences: dict[str, list[float]] = defaultdict(list)
    episode_candidate_weights: dict[str, list[float]] = defaultdict(list)
    episode_reference_weights: dict[str, list[float]] = defaultdict(list)
    issue_weights: dict[str, list[float]] = defaultdict(list)
    issue_differences: dict[str, list[float]] = defaultdict(list)
    issue_candidate_weights: dict[str, list[float]] = defaultdict(list)
    issue_reference_weights: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        event = _identifier(row["event_id"], "event_id")
        episode = _identifier(row["episode_id"], "episode_id")
        fold = _identifier(row["fold_id"], "fold_id")
        issue = _identifier(row["issue_time_utc"], "issue_time_utc")
        if "|" in fold or "|" in issue:
            raise ValueError("fold_id and issue_time_utc cannot contain the cluster separator |")
        raw_weight = row["weight"]
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
            raise ValueError("weight must be a finite positive real number, not a boolean")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive")
        candidate, reference = row["candidate_hit"], row["reference_hit"]
        if not isinstance(candidate, bool) or not isinstance(reference, bool):
            raise ValueError("candidate_hit and reference_hit must be booleans")

        weights.append(weight)
        if candidate:
            candidate_weights.append(weight)
        if reference:
            reference_weights.append(weight)
        if candidate and reference:
            shared_hit.append(weight)
        elif candidate:
            gained.append(weight)
        elif reference:
            lost.append(weight)
        else:
            shared_miss.append(weight)
        difference = weight * (int(candidate) - int(reference))
        differences.append(difference)
        events.add(event)
        episodes.add(episode)
        episode_weights[episode].append(weight)
        episode_differences[episode].append(difference)
        episode_candidate_weights[episode].append(weight if candidate else 0.0)
        episode_reference_weights[episode].append(weight if reference else 0.0)
        issue_key = f"{fold}|{issue}"
        issue_weights[issue_key].append(weight)
        issue_differences[issue_key].append(difference)
        issue_candidate_weights[issue_key].append(weight if candidate else 0.0)
        issue_reference_weights[issue_key].append(weight if reference else 0.0)

    total_weight = math.fsum(weights)
    candidate_hits = math.fsum(candidate_weights)
    reference_hits = math.fsum(reference_weights)
    delta = math.fsum(differences)
    return {
        "status": "available" if rows else "empty_group_NA",
        "event_exposure_count": len(rows),
        "unique_event_count": len(events),
        "unique_episode_count": len(episodes),
        "total_weight": total_weight,
        "candidate_weighted_hits": candidate_hits,
        "reference_weighted_hits": reference_hits,
        "candidate_weighted_recall": candidate_hits / total_weight if rows else None,
        "reference_weighted_recall": reference_hits / total_weight if rows else None,
        "gained_weight": math.fsum(gained),
        "lost_weight": math.fsum(lost),
        "shared_hit_weight": math.fsum(shared_hit),
        "shared_miss_weight": math.fsum(shared_miss),
        "delta_weighted_hits": delta,
        "delta_recall_pp": 100.0 * (delta / total_weight) if rows else None,
        "weight_source": "caller_supplied_no_within_group_renormalization",
        "_local": {
            "episode_clusters": _cluster_summaries(
                episode_weights,
                episode_differences,
                episode_candidate_weights,
                episode_reference_weights,
            ),
            "issue_clusters": _cluster_summaries(
                issue_weights, issue_differences, issue_candidate_weights, issue_reference_weights
            ),
        },
    }
